import { FlameData, ThreadStats } from '@/types/flame.types';

export class FlameGraphDataLoader {
  getSampleDataFromFiles(): FlameData {
    // MVP: Return simple sample data
    return {
      'main': [
        {
          stack: ['main', 'processData', 'parseInput'],
          count: 45
        },
        {
          stack: ['main', 'processData', 'validateData'],
          count: 23
        },
        {
          stack: ['main', 'compute', 'algorithm'],
          count: 67
        },
        {
          stack: ['main', 'cleanup'],
          count: 12
        }
      ],
      'worker_1': [
        {
          stack: ['worker_thread', 'process_chunk', 'hash_data'],
          count: 34
        },
        {
          stack: ['worker_thread', 'process_chunk', 'sort_data'],
          count: 28
        }
      ]
    };
  }

  getSummaryStats(data: FlameData): Record<string, ThreadStats> {
    const stats: Record<string, ThreadStats> = {};

    Object.entries(data).forEach(([threadName, threadData]) => {
      if (Array.isArray(threadData)) {
        const totalSamples = threadData.reduce((sum, entry) => sum + entry.count, 0);
        const maxDepth = Math.max(...threadData.map(entry => entry.stack?.length || 0));
        const uniqueFunctions = new Set(
          threadData.flatMap(entry => entry.stack || [])
        ).size;
        const maxCount = Math.max(...threadData.map(entry => entry.count));
        const avgCount = totalSamples / threadData.length;
        const stackTraces = threadData.length;

        stats[threadName] = {
          totalSamples,
          maxDepth,
          uniqueFunctions,
          maxCount,
          avgCount,
          stackTraces
        };
      }
    });

    return stats;
  }
} 