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
  
  // Data output format - used by views to determine compatibility
  outputFormat: string;
  outputFields: string[];
  
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
    outputFormat: 'flamegraph',
    outputFields: ['stack', 'value', 'count']
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
    outputFormat: 'timeline',
    outputFields: ['timestamp', 'stack', 'value', 'type']
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
    outputFormat: 'flamegraph',
    outputFields: ['stack', 'value', 'duration']
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
    displayName: 'Data View',
    description: 'Simple text view of profiling data',
    icon: '📊',
    configSchema: [],
    defaultConfig: {},
    component: 'DataTableView',
    dataRequirements: {
      format: 'any',
      requiredFields: []
    }
  }
];

// Data Source Management Types
export interface DataSource {
  id: string;
  name: string;
  type: 'analyzer-instance' | 'file' | 'api';
  format: string;
  fields: string[];
  data: any;
  lastUpdated: Date;
  metadata?: Record<string, any>;
}

export interface DataSelection {
  id: string;
  name: string;
  sources: string[]; // Array of data source IDs
  combinationMode: 'merge' | 'append' | 'override';
  filters?: Record<string, any>;
  resultFormat: string;
  resultFields: string[];
}

export interface CurrentDataContext {
  selection: DataSelection | null;
  resolvedData: any;
  format: string;
  fields: string[];
  sources: DataSource[];
} 