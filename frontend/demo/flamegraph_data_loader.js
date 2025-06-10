// Flame Graph Data Loader
// Parses .folded files and converts them to the format needed by Three.js

class FlameGraphDataLoader {
    constructor() {
        this.data = {};
    }

    // Parse a folded file content
    parseFoldedContent(content, threadName) {
        const lines = content.split('\n');
        const flameData = [];
        
        for (let line of lines) {
            line = line.trim();
            if (line) {
                const parts = line.split(' ');
                if (parts.length >= 2) {
                    const count = parseInt(parts[parts.length - 1]);
                    const stackTrace = parts.slice(0, -1).join(' ');
                    
                    if (!isNaN(count) && stackTrace) {
                        const stack = stackTrace.split(';').map(f => f.trim()).filter(f => f);
                        if (stack.length > 0) {
                            flameData.push({
                                stack: stack,
                                count: count
                            });
                        }
                    }
                }
            }
        }
        
        this.data[threadName] = flameData;
        return flameData;
    }

    // Load multiple files
    async loadFiles(fileInfos) {
        const promises = fileInfos.map(async (fileInfo) => {
            try {
                const response = await fetch(fileInfo.path);
                if (!response.ok) {
                    throw new Error(`Failed to load ${fileInfo.path}: ${response.statusText}`);
                }
                const content = await response.text();
                return this.parseFoldedContent(content, fileInfo.name);
            } catch (error) {
                console.error(`Error loading ${fileInfo.path}:`, error);
                return [];
            }
        });

        await Promise.all(promises);
        return this.data;
    }

    // Get sample data based on the provided files
    getSampleDataFromFiles() {
        // Sample data extracted from your files
        return {
            'req_generator': [
                {stack: ['pthread_condattr_setpshared', 'request_generator', '__clock_gettime', '[unknown]'], count: 539},
                {stack: ['pthread_condattr_setpshared', 'random'], count: 430},
                {stack: ['pthread_condattr_setpshared', 'request_generator', '__pthread_mutex_trylock'], count: 240},
                {stack: ['pthread_condattr_setpshared', 'request_generator', '__pthread_mutex_lock'], count: 232},
                {stack: ['pthread_condattr_setpshared', 'random_r'], count: 100},
                {stack: ['pthread_condattr_setpshared', 'request_generator'], count: 80},
                {stack: ['pthread_condattr_setpshared', '__lll_lock_wake_private'], count: 38},
                {stack: ['pthread_condattr_setpshared', 'request_generator', '[unknown]'], count: 38},
                {stack: ['pthread_condattr_setpshared', 'request_generator', 'enqueue_request'], count: 33},
                {stack: ['pthread_condattr_setpshared', 'request_generator', 'get_timestamp'], count: 22}
            ],
            'stats_monitor': [
                {stack: ['clock_nanosleep', 'entry_SYSCALL_64_after_hwframe', 'do_syscall_64', 'x64_sys_call', '__x64_sys_clock_nanosleep', 'common_nsleep', 'hrtimer_nanosleep', 'do_nanosleep', 'schedule', '__schedule', '__traceiter_sched_switch'], count: 1485}
            ],
            'worker_0': [
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'simulate_cpu_sort_work'], count: 1099},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'f64xsubf128'], count: 424},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'fsync', 'entry_SYSCALL_64_after_hwframe', 'do_syscall_64'], count: 119},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'simulate_cpu_math_work'], count: 82},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', '__poll', 'entry_SYSCALL_64_after_hwframe'], count: 20},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', '__lll_lock_wake_private'], count: 9},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'random_r'], count: 7},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'random'], count: 6},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'memcpy'], count: 4}
            ],
            'worker_1': [
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'simulate_cpu_sort_work'], count: 1270},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'f64xsubf128'], count: 294},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'fsync', 'entry_SYSCALL_64_after_hwframe', 'do_syscall_64'], count: 115},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'simulate_cpu_math_work'], count: 55},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', '__poll', 'entry_SYSCALL_64_after_hwframe'], count: 18},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', '__lll_lock_wake_private'], count: 11},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'random'], count: 10},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', '__lll_lock_wait_private'], count: 4},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'memcpy'], count: 3}
            ],
            'worker_2': [
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'simulate_cpu_sort_work'], count: 1125},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'f64xsubf128'], count: 332},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'fsync', 'entry_SYSCALL_64_after_hwframe', 'do_syscall_64'], count: 124},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'simulate_cpu_math_work'], count: 68},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', '__poll', 'entry_SYSCALL_64_after_hwframe'], count: 30},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'random'], count: 14},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', '__lll_lock_wake_private'], count: 6},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', '__lll_lock_wait_private'], count: 5},
                {stack: ['pthread_condattr_setpshared', 'worker_thread', 'memcpy'], count: 3}
            ]
        };
    }

    // Build flame tree from flat data
    buildFlameTree(data) {
        const tree = {};
        
        data.forEach(entry => {
            let current = tree;
            entry.stack.forEach(func => {
                if (!current[func]) {
                    current[func] = {
                        count: 0,
                        children: {}
                    };
                }
                current[func].count += entry.count;
                current = current[func].children;
            });
        });

        return tree;
    }

    // Get data for Three.js visualization
    getVisualizationData() {
        return this.data;
    }

    // Filter data by minimum count threshold
    filterByMinCount(data, minCount = 10) {
        const filtered = {};
        
        for (const [threadName, threadData] of Object.entries(data)) {
            filtered[threadName] = threadData.filter(entry => entry.count >= minCount);
        }
        
        return filtered;
    }

    // Normalize function names for better visualization
    normalizeFunctionNames(data) {
        const normalized = {};
        
        for (const [threadName, threadData] of Object.entries(data)) {
            normalized[threadName] = threadData.map(entry => ({
                ...entry,
                stack: entry.stack.map(func => {
                    // Remove common prefixes and clean up function names
                    return func
                        .replace(/^pthread_condattr_setpshared;?/, '')
                        .replace(/^entry_SYSCALL_64_after_hwframe;?/, '')
                        .replace(/_\[c\]$/, '')
                        .replace(/_\[o\]$/, '')
                        .split('_')[0] // Take first part before underscore
                        .trim();
                }).filter(func => func && func !== '')
            })).filter(entry => entry.stack.length > 0);
        }
        
        return normalized;
    }

    // Get summary statistics
    getSummaryStats(data) {
        const stats = {};
        
        for (const [threadName, threadData] of Object.entries(data)) {
            const counts = threadData.map(entry => entry.count);
            const totalSamples = counts.reduce((sum, count) => sum + count, 0);
            const maxCount = Math.max(...counts);
            const avgCount = totalSamples / counts.length;
            const uniqueFunctions = new Set();
            
            threadData.forEach(entry => {
                entry.stack.forEach(func => uniqueFunctions.add(func));
            });
            
            stats[threadName] = {
                totalSamples,
                maxCount,
                avgCount: Math.round(avgCount),
                stackTraces: threadData.length,
                uniqueFunctions: uniqueFunctions.size,
                maxDepth: Math.max(...threadData.map(entry => entry.stack.length))
            };
        }
        
        return stats;
    }
} 