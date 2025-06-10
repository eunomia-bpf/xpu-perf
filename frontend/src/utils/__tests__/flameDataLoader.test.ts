import { describe, it, expect, beforeEach } from 'vitest';
import { FlameGraphDataLoader } from '../flameDataLoader';

describe('FlameGraphDataLoader', () => {
  let loader: FlameGraphDataLoader;

  beforeEach(() => {
    loader = new FlameGraphDataLoader();
  });

  describe('parseFoldedContent', () => {
    it('should parse valid folded content correctly', () => {
      const content = `
pthread_condattr_setpshared;worker_thread;simulate_cpu_sort_work 1099
pthread_condattr_setpshared;worker_thread;f64xsubf128 424
pthread_condattr_setpshared;worker_thread;fsync 119
      `.trim();

      const result = loader.parseFoldedContent(content, 'test_thread');

      expect(result).toHaveLength(3);
      expect(result[0]).toEqual({
        stack: ['pthread_condattr_setpshared', 'worker_thread', 'simulate_cpu_sort_work'],
        count: 1099
      });
      expect(result[1]).toEqual({
        stack: ['pthread_condattr_setpshared', 'worker_thread', 'f64xsubf128'],
        count: 424
      });
    });

    it('should handle empty lines and invalid formats', () => {
      const content = `
pthread_condattr_setpshared;worker_thread;simulate_cpu_sort_work 1099

invalid_line_without_count
pthread_condattr_setpshared;worker_thread;fsync abc
pthread_condattr_setpshared;worker_thread;valid_function 42
      `.trim();

      const result = loader.parseFoldedContent(content, 'test_thread');

      expect(result).toHaveLength(2);
      expect(result[0]!.count).toBe(1099);
      expect(result[1]!.count).toBe(42);
    });

    it('should filter out empty stacks', () => {
      const content = ` 1099\n;; 424\npthread_condattr_setpshared;worker_thread 42`;

      const result = loader.parseFoldedContent(content, 'test_thread');

      expect(result).toHaveLength(1);
      expect(result[0]!.stack).toEqual(['pthread_condattr_setpshared', 'worker_thread']);
    });
  });

  describe('buildFlameTree', () => {
    it('should build correct hierarchical tree structure', () => {
      const data = [
        { stack: ['a', 'b', 'c'], count: 10 },
        { stack: ['a', 'b', 'd'], count: 5 },
        { stack: ['a', 'e'], count: 8 }
      ];

      const tree = loader.buildFlameTree(data);

      expect(tree.a!.count).toBe(23); // 10 + 5 + 8
      expect(tree.a!.children.b!.count).toBe(15); // 10 + 5
      expect(tree.a!.children.b!.children.c!.count).toBe(10);
      expect(tree.a!.children.b!.children.d!.count).toBe(5);
      expect(tree.a!.children.e!.count).toBe(8);
    });

    it('should handle single-level stacks', () => {
      const data = [
        { stack: ['function1'], count: 10 },
        { stack: ['function2'], count: 5 }
      ];

      const tree = loader.buildFlameTree(data);

      expect(Object.keys(tree)).toHaveLength(2);
      expect(tree.function1!.count).toBe(10);
      expect(tree.function2!.count).toBe(5);
    });
  });

  describe('filterByMinCount', () => {
    it('should filter out entries below minimum count', () => {
      const data = {
        thread1: [
          { stack: ['func1'], count: 100 },
          { stack: ['func2'], count: 5 },
          { stack: ['func3'], count: 50 }
        ]
      };

      const filtered = loader.filterByMinCount(data, 10);

      expect(filtered.thread1).toHaveLength(2);
      expect(filtered.thread1![0]!.count).toBe(100);
      expect(filtered.thread1![1]!.count).toBe(50);
    });

    it('should handle empty data', () => {
      const filtered = loader.filterByMinCount({}, 10);
      expect(filtered).toEqual({});
    });
  });

  describe('getSummaryStats', () => {
    it('should calculate correct statistics', () => {
      const data = {
        thread1: [
          { stack: ['func1', 'func2'], count: 100 },
          { stack: ['func1', 'func3'], count: 50 },
          { stack: ['func4'], count: 25 }
        ]
      };

      const stats = loader.getSummaryStats(data);

      expect(stats.thread1).toEqual({
        totalSamples: 175,
        maxCount: 100,
        avgCount: 58, // Math.round(175/3)
        stackTraces: 3,
        uniqueFunctions: 4, // func1, func2, func3, func4
        maxDepth: 2
      });
    });
  });

  describe('getSampleDataFromFiles', () => {
    it('should return sample data with expected structure', () => {
      const sampleData = loader.getSampleDataFromFiles();

      expect(Object.keys(sampleData)).toContain('req_generator');
      expect(Object.keys(sampleData)).toContain('worker_0');
      expect(Array.isArray(sampleData.req_generator)).toBe(true);
      expect(sampleData.req_generator![0]!).toHaveProperty('stack');
      expect(sampleData.req_generator![0]!).toHaveProperty('count');
    });
  });

  describe('normalizeFunctionNames', () => {
    it('should clean up function names correctly', () => {
      const data = {
        thread1: [
          { 
            stack: ['pthread_condattr_setpshared', 'entry_SYSCALL_64_after_hwframe', 'some_function_name'], 
            count: 100 
          }
        ]
      };

      const normalized = loader.normalizeFunctionNames(data);

      expect(normalized.thread1![0]!.stack).toEqual(['some']);
    });

    it('should filter out empty function names', () => {
      const data = {
        thread1: [
          { 
            stack: ['pthread_condattr_setpshared', '', 'valid_function'], 
            count: 100 
          }
        ]
      };

      const normalized = loader.normalizeFunctionNames(data);

      expect(normalized.thread1![0]!.stack).toEqual(['valid']);
    });
  });
}); 