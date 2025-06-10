# Zero-Instrument Interactive Profiler Tool - Architecture Design

## 🎯 Project Vision

A comprehensive, zero-instrument, online profiler tool that provides multi-session management, multiple analyzer types, real-time data collection, and multi-modal visualization (2D/3D/4D) for complete system performance analysis.

## 🏗️ Application Architecture

### Core Architecture Principles
- **Multi-Session Management**: Each profiling session runs independently with its own tab
- **Multi-Analyzer Support**: Each session can run multiple analyzer types simultaneously
- **Real-time Data Streaming**: Live data collection from various analyzer sources
- **Modular Visualization**: Pluggable visualization components tied to specific view types
- **Data-View Separation**: Clear separation between data collection, storage, and visualization
- **Analyzer-Specific Controls**: Each analyzer type has dedicated control components

### System Component Hierarchy

```
ProfilerApp/
├── SessionManager/
│   ├── TabManager (multi-session tabs)
│   ├── SessionContainer
│   ├── SessionState
│   └── SessionPersistence
├── AnalyzerEngine/
│   ├── TraceAnalyzer/          # Function probes, events with timestamps
│   │   ├── FunctionProbeCollector
│   │   ├── EventStreamProcessor
│   │   └── TimestampCorrelator
│   ├── MetricsAnalyzer/        # PMU data, system metrics
│   │   ├── PMUDataCollector
│   │   ├── SystemMetricsReader
│   │   └── MetricsAggregator
│   ├── FlameGraphAnalyzer/     # Stack traces, call graphs
│   │   ├── StackTraceCollector
│   │   ├── CallGraphBuilder
│   │   └── FlameDataProcessor
│   └── StaticAnalyzer/         # Program structure, symbols
│       ├── SymbolTableReader
│       ├── BinaryAnalyzer
│       └── SourceCodeMapper
├── ControlCenter/
│   ├── AnalyzerControls/       # Start/stop/config analyzers
│   │   ├── AnalyzerManager
│   │   ├── AnalyzerConfigPanel
│   │   └── AnalyzerStatusMonitor
│   ├── VisualizationControls/  # Configure view types and data sources
│   │   ├── ViewportManager
│   │   ├── DataSourceSelector
│   │   └── ViewConfigPanel
│   └── DataBrowser/           # Browse all collected data
│       ├── DataExplorer
│       ├── DataFilterPanel
│       └── DataExportManager
├── VisualizationEngine/
│   ├── ViewportContainer/      # Manages multiple view instances
│   ├── FlameGraph3D/          # 3D flame graph with specific controls
│   │   ├── FlameGraph3DRenderer
│   │   ├── FlameGraph3DControls (tied to this view)
│   │   └── FlameGraph3DInteractions
│   ├── FlameGraph2D/          # 2D flame graph with specific controls
│   │   ├── FlameGraph2DRenderer
│   │   ├── FlameGraph2DControls (tied to this view)
│   │   └── FlameGraph2DInteractions
│   ├── TimelineChart/         # Timeline views with specific controls
│   │   ├── TimelineRenderer
│   │   ├── TimelineControls (tied to this view)
│   │   └── TimelineInteractions
│   ├── MetricsChart/          # System metrics visualization
│   │   ├── MetricsRenderer
│   │   ├── MetricsControls
│   │   └── MetricsInteractions
│   └── TraceViewer/           # Event trace visualization
│       ├── TraceRenderer
│       ├── TraceControls
│       └── TraceInteractions
├── DataManager/
│   ├── SessionDataStore/      # Per-session data storage
│   ├── AnalyzerDataBuffer/    # Real-time data buffering
│   ├── DataSynchronizer/      # Cross-analyzer data correlation
│   └── BrowserStorage/        # Temporary browser storage
└── LayoutManager/
    ├── AppShell
    ├── TabSystem
    └── ViewportLayout
```

## 📊 Enhanced Data Architecture

### Multi-Session Data Flow
```
Session 1 Tab ─┐
               ├─► SessionDataStore ─► VisualizationEngine
Session 2 Tab ─┤                      │
               └─► (Independent)       ▼
Session N Tab...                   ViewportContainer
                                      │
┌─────────────────────────────────────┼────────────────────────────────────┐
│ Per-Session Analyzer Pipeline       ▼                                    │
│                                                                          │
│ TraceAnalyzer ──┐                                                       │
│ MetricsAnalyzer ├──► DataSynchronizer ──► AnalyzerDataBuffer ──►        │
│ FlameAnalyzer ──┤                                                │       │
│ StaticAnalyzer ─┘                                                ▼       │
│                                                                  │       │
│                                        BrowserStorage ◄─────────┘       │
└──────────────────────────────────────────────────────────────────────────┘
```

### Analyzer-Specific Data Models

#### Trace Analyzer Data
```typescript
interface TraceData {
  events: TraceEvent[]
  timeline: Timeline
  correlationMap: CorrelationMap
}

interface TraceEvent {
  timestamp: number
  eventType: 'function_entry' | 'function_exit' | 'custom_event'
  functionName: string
  threadId: string
  processId: string
  parameters?: Record<string, any>
  stackTrace?: string[]
}
```

#### Metrics Analyzer Data
```typescript
interface MetricsData {
  systemMetrics: SystemMetrics[]
  pmuData: PMUData[]
  resourceUsage: ResourceUsage[]
}

interface PMUData {
  timestamp: number
  cpuCycles: number
  instructions: number
  cacheHits: number
  cacheMisses: number
  branchPredictions: number
}
```

#### FlameGraph Analyzer Data
```typescript
interface FlameGraphData {
  stackTraces: StackTrace[]
  callGraph: CallGraph
  aggregatedData: AggregatedFlameData
}

interface StackTrace {
  timestamp: number
  threadId: string
  frames: StackFrame[]
  sampleCount: number
}
```

#### Static Analyzer Data
```typescript
interface StaticData {
  symbolTable: SymbolTable
  binaryInfo: BinaryInfo
  sourceMapping: SourceMapping
  dependencies: Dependency[]
}
```

## 🎛️ Enhanced Control Architecture

### Session-Level Controls
```typescript
interface SessionControls {
  sessionManager: {
    createSession: () => SessionId
    switchSession: (id: SessionId) => void
    closeSession: (id: SessionId) => void
    duplicateSession: (id: SessionId) => SessionId
  }
  
  analyzerManager: {
    startAnalyzer: (type: AnalyzerType, config: AnalyzerConfig) => void
    stopAnalyzer: (analyzerId: AnalyzerId) => void
    configureAnalyzer: (analyzerId: AnalyzerId, config: AnalyzerConfig) => void
    getAnalyzerStatus: (analyzerId: AnalyzerId) => AnalyzerStatus
  }
}
```

### View-Specific Controls
```typescript
interface ViewControls {
  flameGraph3D: FlameGraph3DControls
  flameGraph2D: FlameGraph2DControls
  timelineChart: TimelineControls
  metricsChart: MetricsControls
  traceViewer: TraceControls
}

interface FlameGraph3DControls {
  camera: CameraControls
  rendering: RenderingControls
  interaction: InteractionControls
  data: DataControls
}
```

### Visualization Controls
```typescript
interface VisualizationControls {
  viewportManager: {
    addViewport: (type: ViewType, config: ViewConfig) => ViewportId
    removeViewport: (id: ViewportId) => void
    configureViewport: (id: ViewportId, config: ViewConfig) => void
    setDataSource: (viewportId: ViewportId, dataSource: DataSource) => void
  }
  
  layoutManager: {
    setLayout: (layout: LayoutType) => void // grid, tabs, split
    arrangeViewports: (arrangement: ViewportArrangement) => void
  }
}
```

## 🔄 Analyzer Types & Capabilities

### 1. Trace Analyzer
**Purpose**: Function-level tracing with events and timestamps
**Data Sources**: 
- Function entry/exit probes
- Custom event markers
- System call traces
- User-defined trace points

**Capabilities**:
- Real-time function call tracking
- Event correlation across threads
- Timeline visualization
- Performance bottleneck identification

### 2. Metrics Analyzer  
**Purpose**: System and hardware performance metrics
**Data Sources**:
- Performance Monitoring Unit (PMU)
- System resource utilization
- Hardware counters
- Custom metric endpoints

**Capabilities**:
- CPU performance analysis
- Memory utilization tracking
- Cache performance monitoring
- Hardware event correlation

### 3. FlameGraph Analyzer
**Purpose**: Call stack profiling and flame graph generation
**Data Sources**:
- Stack trace sampling
- Call graph generation
- Symbol resolution
- Sample aggregation

**Capabilities**:
- Traditional 2D flame graphs
- Interactive 3D flame stacks
- Call hierarchy analysis
- Hot path identification

### 4. Static Analyzer
**Purpose**: Program structure and symbol analysis
**Data Sources**:
- Binary symbol tables
- Source code mapping
- Dependency analysis
- Program metadata

**Capabilities**:
- Symbol resolution
- Source code correlation
- Dependency visualization
- Program structure analysis

## 🎨 Multi-Modal Visualization Integration

### View Type Registration
```typescript
interface ViewTypeRegistry {
  '3d-flame-graph': {
    component: FlameGraph3D
    controls: FlameGraph3DControls
    dataTypes: ['flamegraph', 'stacktrace']
    requirements: ['webgl']
  }
  
  '2d-flame-graph': {
    component: FlameGraph2D
    controls: FlameGraph2DControls
    dataTypes: ['flamegraph', 'stacktrace']
    requirements: ['canvas']
  }
  
  'timeline-chart': {
    component: TimelineChart
    controls: TimelineControls
    dataTypes: ['trace', 'metrics', 'events']
    requirements: ['d3']
  }
  
  'metrics-dashboard': {
    component: MetricsChart
    controls: MetricsControls
    dataTypes: ['metrics', 'pmu']
    requirements: ['charts']
  }
}
```

### Data-View Binding
```typescript
interface DataViewBinding {
  viewportId: ViewportId
  viewType: ViewType
  dataSource: {
    analyzerId: AnalyzerId
    dataType: DataType
    filters: DataFilter[]
    transformations: DataTransformation[]
  }
  config: ViewConfig
}
```

## 🚀 Implementation Phases

### Phase 1: Multi-Session Foundation (Weeks 1-4)
1. **Session Management System**
   - Tab-based session interface
   - Session state management
   - Session persistence in browser storage

2. **Basic Analyzer Framework**
   - Analyzer base classes
   - Simple trace analyzer implementation
   - Basic data collection pipeline

### Phase 2: Analyzer Engine (Weeks 5-8)
1. **Complete Analyzer Implementation**
   - All four analyzer types
   - Real-time data streaming
   - Data synchronization between analyzers

2. **Enhanced Control System**
   - Analyzer control panels
   - Visualization control interface
   - Data browser implementation

### Phase 3: Advanced Visualization (Weeks 9-12)
1. **View-Specific Controls**
   - Dedicated controls for each view type
   - Dynamic control panel generation
   - View configuration persistence

2. **Multi-Viewport Support**
   - Multiple simultaneous views
   - Cross-view data correlation
   - Layout management

### Phase 4: Production Features (Weeks 13-16)
1. **Performance Optimization**
   - Large dataset handling
   - Real-time streaming optimization
   - Memory management

2. **Enterprise Features**
   - Data export/import
   - Session sharing
   - Advanced analytics

## 🔧 Technical Implementation Strategy

### Modular Architecture Benefits
1. **Analyzer Independence**: Each analyzer can be developed and tested separately
2. **View Flexibility**: New visualization types can be added without affecting others
3. **Control Separation**: View-specific controls maintain tight coupling with their views
4. **Data Isolation**: Session data is completely isolated, preventing cross-contamination
5. **Scalability**: System can handle multiple concurrent sessions with different analyzer configurations

### Real-time Data Coordination
```typescript
interface DataCoordinator {
  synchronizeAnalyzers: (analyzers: AnalyzerId[]) => void
  correlateTimestamps: (data: AnalyzerData[]) => CorrelatedData
  distributeToViews: (data: CorrelatedData, views: ViewportId[]) => void
  handleDataConflicts: (conflicts: DataConflict[]) => Resolution[]
}
```

This enhanced architecture provides a solid foundation for building a comprehensive, professional-grade profiler that can handle complex multi-analyzer scenarios while maintaining clean separation of concerns and excellent user experience. 