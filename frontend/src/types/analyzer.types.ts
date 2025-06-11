// Dynamic Analyzer System Types

export interface AnalyzerConfig {
  id: string;
  name: string;
  displayName: string;
  description: string;
  icon?: string;
  
  // Configuration schema
  configSchema: ConfigField[];
  defaultConfig: Record<string, any>;
  
  // Supported views
  supportedViews: string[];
  
  // Component paths for dynamic loading
  controlComponent?: string;
  processorComponent?: string;
}

export interface ConfigField {
  key: string;
  label: string;
  type: 'text' | 'number' | 'boolean' | 'select' | 'range' | 'multi-select';
  description?: string;
  defaultValue: any;
  validation?: {
    required?: boolean;
    min?: number;
    max?: number;
    options?: Array<{ value: any; label: string }>;
  };
  group?: string;
}

export interface AnalyzerInstance {
  id: string;
  analyzerId: string;
  name: string;
  config: Record<string, any>;
  status: 'idle' | 'configuring' | 'running' | 'completed' | 'error';
  data?: any;
  error?: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface ViewConfig {
  id: string;
  name: string;
  displayName: string;
  description: string;
  icon?: string;
  
  // Configuration schema for view-specific settings
  configSchema: ConfigField[];
  defaultConfig: Record<string, any>;
  
  // Component path for dynamic loading
  component: string;
  
  // Data requirements
  dataRequirements: {
    format: string;
    requiredFields: string[];
  };
}

export interface AnalyzerRegistry {
  analyzers: Record<string, AnalyzerConfig>;
  views: Record<string, ViewConfig>;
}

// Built-in analyzer configurations
export const BUILT_IN_ANALYZERS: AnalyzerConfig[] = [
  {
    id: 'flamegraph',
    name: 'flamegraph',
    displayName: 'Flame Graph Profiler',
    description: 'CPU profiling with stack trace sampling',
    icon: '🔥',
    configSchema: [
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        description: 'Profiling duration in seconds',
        defaultValue: 30,
        validation: { required: true, min: 1, max: 300 },
        group: 'timing'
      },
      {
        key: 'frequency',
        label: 'Frequency',
        type: 'number',
        description: 'Sampling frequency in Hz',
        defaultValue: 99,
        validation: { required: true, min: 1, max: 999 },
        group: 'timing'
      },
      {
        key: 'target',
        label: 'Target Process',
        type: 'text',
        description: 'Process name or PID to profile',
        defaultValue: '',
        group: 'target'
      }
    ],
    defaultConfig: {
      duration: 30,
      frequency: 99,
      target: ''
    },
    supportedViews: ['3d-flame', 'data-table', 'flame-chart']
  },
  {
    id: 'wallclock',
    name: 'wallclock',
    displayName: 'Wall Clock Analyzer',
    description: 'Combined on-CPU and off-CPU profiling',
    icon: '⏰',
    configSchema: [
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        description: 'Profiling duration in seconds',
        defaultValue: 30,
        validation: { required: true, min: 1, max: 300 },
        group: 'timing'
      },
      {
        key: 'frequency',
        label: 'Frequency',
        type: 'number',
        description: 'Sampling frequency in Hz',
        defaultValue: 99,
        validation: { required: true, min: 1, max: 999 },
        group: 'timing'
      },
      {
        key: 'includeOffCpu',
        label: 'Include Off-CPU',
        type: 'boolean',
        description: 'Include off-CPU time analysis',
        defaultValue: true,
        group: 'features'
      },
      {
        key: 'target',
        label: 'Target Process',
        type: 'text',
        description: 'Process name or PID to profile',
        defaultValue: '',
        group: 'target'
      }
    ],
    defaultConfig: {
      duration: 30,
      frequency: 99,
      includeOffCpu: true,
      target: ''
    },
    supportedViews: ['3d-flame', 'data-table', 'timeline']
  },
  {
    id: 'offcpu',
    name: 'offcpu',
    displayName: 'Off-CPU Time Analyzer', 
    description: 'Analyze time spent off-CPU (blocking)',
    icon: '💤',
    configSchema: [
      {
        key: 'duration',
        label: 'Duration',
        type: 'number',
        description: 'Profiling duration in seconds',
        defaultValue: 30,
        validation: { required: true, min: 1, max: 300 },
        group: 'timing'
      },
      {
        key: 'minBlockTime',
        label: 'Min Block Time',
        type: 'number',
        description: 'Minimum blocking time in microseconds',
        defaultValue: 1000,
        validation: { required: true, min: 1 },
        group: 'filtering'
      },
      {
        key: 'target',
        label: 'Target Process',
        type: 'text', 
        description: 'Process name or PID to profile',
        defaultValue: '',
        group: 'target'
      }
    ],
    defaultConfig: {
      duration: 30,
      minBlockTime: 1000,
      target: ''
    },
    supportedViews: ['3d-flame', 'data-table']
  }
];

// Built-in view configurations
export const BUILT_IN_VIEWS: ViewConfig[] = [
  {
    id: '3d-flame',
    name: '3d-flame',
    displayName: '3D Flame Graph',
    description: 'Interactive 3D visualization of stack traces',
    icon: '🎯',
    configSchema: [
      {
        key: 'zSpacing',
        label: 'Z-Spacing',
        type: 'range',
        description: 'Spacing between stack levels',
        defaultValue: 25,
        validation: { min: 10, max: 50 },
        group: 'visualization'
      },
      {
        key: 'minCount',
        label: 'Min Count',
        type: 'range',
        description: 'Minimum sample count to display',
        defaultValue: 10,
        validation: { min: 1, max: 100 },
        group: 'filtering'
      },
      {
        key: 'maxDepth',
        label: 'Max Depth',
        type: 'range', 
        description: 'Maximum stack depth to display',
        defaultValue: 8,
        validation: { min: 1, max: 20 },
        group: 'filtering'
      },
      {
        key: 'colorScheme',
        label: 'Color Scheme',
        type: 'select',
        description: 'Color scheme for visualization',
        defaultValue: 'hot-cold',
        validation: {
          options: [
            { value: 'hot-cold', label: 'Hot/Cold' },
            { value: 'thread-based', label: 'Thread-based' },
            { value: 'function-based', label: 'Function-based' }
          ]
        },
        group: 'visualization'
      }
    ],
    defaultConfig: {
      zSpacing: 25,
      minCount: 10, 
      maxDepth: 8,
      colorScheme: 'hot-cold'
    },
    component: 'FlameGraph3DView',
    dataRequirements: {
      format: 'flamegraph',
      requiredFields: ['stack', 'value']
    }
  },
  {
    id: 'data-table',
    name: 'data-table',
    displayName: 'Data Table',
    description: 'Tabular view of profiling data',
    icon: '📊',
    configSchema: [
      {
        key: 'maxRows',
        label: 'Max Rows',
        type: 'select',
        description: 'Maximum rows to display',
        defaultValue: 100,
        validation: {
          options: [
            { value: 50, label: 'Top 50' },
            { value: 100, label: 'Top 100' },
            { value: 500, label: 'Top 500' },
            { value: 1000, label: 'All' }
          ]
        },
        group: 'display'
      },
      {
        key: 'sortBy',
        label: 'Sort By',
        type: 'select',
        description: 'Default sorting column',
        defaultValue: 'totalTime',
        validation: {
          options: [
            { value: 'totalTime', label: 'Total Time' },
            { value: 'selfTime', label: 'Self Time' },
            { value: 'callCount', label: 'Call Count' },
            { value: 'name', label: 'Function Name' }
          ]
        },
        group: 'display'
      }
    ],
    defaultConfig: {
      maxRows: 100,
      sortBy: 'totalTime'
    },
    component: 'DataTableView',
    dataRequirements: {
      format: 'table',
      requiredFields: ['name', 'value']
    }
  }
]; 