# Zero-Instrument Interactive Profiler Tool - Architecture Design

||

## 🎯 Project Vision

A zero-instrument, online profiler tool that starts simple but scales to comprehensive multi-analyzer capabilities. Built with modularity and extensibility as core principles, allowing progressive enhancement from MVP to full-featured system.

## 🏗️ **MVP-First Architecture**

### **Core Architecture Principles**
- **Progressive Enhancement**: Start simple, add complexity incrementally
- **Modular Design**: Each component can be developed and deployed independently
- **Extensible Framework**: Easy addition of new analyzer types and view types
- **Data-View Separation**: Clear separation between data collection, processing, and visualization
- **Plugin Architecture**: New analyzers and views can be added without core changes

### **MVP System Architecture (Phase 1)**

```
ProfilerApp/
├── AnalyzerEngine/
│   ├── AnalyzerManager/           # Core analyzer orchestration
│   ├── FlameGraphAnalyzer/        # MVP: Primary analyzer (existing enhanced)
│   └── BaseAnalyzer/              # Extension point for new analyzers
├── ViewportEngine/
│   ├── ViewportManager/           # Single viewport for MVP
│   ├── FlameGraph3D/              # MVP: Enhanced existing 3D view
│   ├── DataTableView/             # MVP: Simple data display view
│   └── BaseView/                  # Extension point for new view types
├── ControlCenter/
│   ├── AnalyzerControls/          # Simple start/stop/config for active analyzer
│   ├── ViewControls/              # View type selector and basic controls
│   └── DataControls/              # Simple data filtering and export
├── DataManager/
│   ├── DataStore/                 # Single session store for MVP
│   ├── DataProcessor/             # Basic data processing pipeline
│   └── DataExporter/              # Simple export functionality
└── LayoutManager/
    ├── AppShell/                  # Basic layout wrapper
    └── SingleViewLayout/          # MVP: Single view, no complex layouts
```

### **Extension Architecture (Post-MVP)**

```
// New analyzer types plug into existing framework
AnalyzerEngine/
├── FlameGraphAnalyzer/            ✅ MVP
├── TraceAnalyzer/                 📋 Phase 2
├── MetricsAnalyzer/               📋 Phase 3
├── StaticAnalyzer/                📋 Phase 4
└── CustomAnalyzer/                📋 Plugin system

// New view types plug into existing framework
ViewportEngine/
├── FlameGraph3D/                  ✅ MVP
├── DataTableView/                 ✅ MVP
├── FlameGraph2D/                  📋 Phase 2
├── TimelineChart/                 📋 Phase 3
├── MetricsChart/                  📋 Phase 4
└── CustomView/                    📋 Plugin system

// Session management added later
SessionManager/                    📋 Phase 2+
├── MultiSessionTabs/
├── SessionPersistence/
└── SessionSharing/
```

## 📊 **MVP Data Architecture**

### **Simplified Data Flow (MVP)**
```
Single Analyzer → Data Processing → Single View
     │                 │               │
FlameGraphAnalyzer → DataStore → [FlameGraph3D | DataTable]
```

### **Extensible Data Flow (Post-MVP)**
```
Multiple Analyzers → Data Correlation → Multiple Views
        │                  │               │
[Flame|Trace|Metrics] → Correlation → [3D|2D|Timeline|Table]
```

### **MVP Data Models**

#### **Simplified Analyzer Interface**
```typescript
interface BaseAnalyzer {
  id: string
  type: AnalyzerType
  status: 'stopped' | 'starting' | 'running' | 'stopping'
  
  // Core methods
  start(): Promise<void>
  stop(): Promise<void>
  configure(config: AnalyzerConfig): void
  getData(): AnalyzerData
  
  // Event system for real-time updates
  onDataUpdate(callback: (data: AnalyzerData) => void): void
  onStatusChange(callback: (status: AnalyzerStatus) => void): void
}
```

#### **Simplified View Interface**
```typescript
interface BaseView {
  id: string
  type: ViewType
  
  // Core methods
  render(data: AnalyzerData): void
  configure(config: ViewConfig): void
  getControls(): React.ComponentType
  
  // Interaction events
  onSelection(callback: (selection: ViewSelection) => void): void
  onFilter(callback: (filter: DataFilter) => void): void
}
```

#### **MVP Data Store**
```typescript
interface DataStore {
  // Single analyzer for MVP
  activeAnalyzer: BaseAnalyzer | null
  analyzerData: AnalyzerData | null
  
  // Single view for MVP
  activeView: BaseView
  viewConfig: ViewConfig
  
  // Simple state management
  setAnalyzer(analyzer: BaseAnalyzer): void
  setView(view: BaseView): void
  updateData(data: AnalyzerData): void
}
```

## 🎯 **Analyzer Plugin System**

### **Analyzer Registration (Extensible)**
```typescript
interface AnalyzerRegistry {
  registerAnalyzer(type: string, factory: AnalyzerFactory): void
  createAnalyzer(type: string, config: AnalyzerConfig): BaseAnalyzer
  getAvailableAnalyzers(): AnalyzerInfo[]
}

// Example: Adding new analyzer type
analyzerRegistry.registerAnalyzer('trace', (config) => new TraceAnalyzer(config))
analyzerRegistry.registerAnalyzer('metrics', (config) => new MetricsAnalyzer(config))
```

### **View Plugin System**
```typescript
interface ViewRegistry {
  registerView(type: string, factory: ViewFactory): void
  createView(type: string, config: ViewConfig): BaseView
  getAvailableViews(): ViewInfo[]
  getCompatibleViews(dataType: DataType): ViewInfo[]
}

// Example: Adding new view type
viewRegistry.registerView('timeline', (config) => new TimelineChart(config))
viewRegistry.registerView('heatmap', (config) => new HeatmapView(config))
```

## 🔧 **MVP Implementation Strategy**

### **Phase 1: MVP Foundation (Weeks 1-4)**
**Goal**: Validate core concept with minimal viable product

**Scope**:
- Single analyzer: Enhanced FlameGraph (build on existing)
- Two view types: 3D FlameGraph + Data Table
- Simple controls: Start/Stop analyzer, switch views
- Basic data export

**Implementation**:
```typescript
// MVP Analyzer Manager
class MVPAnalyzerManager {
  private activeAnalyzer: FlameGraphAnalyzer | null = null
  
  async startFlameAnalyzer(config: FlameConfig): Promise<void> {
    this.activeAnalyzer = new FlameGraphAnalyzer(config)
    await this.activeAnalyzer.start()
  }
  
  stopAnalyzer(): void {
    this.activeAnalyzer?.stop()
    this.activeAnalyzer = null
  }
}

// MVP View Manager
class MVPViewManager {
  private views = {
    '3d-flame': () => new FlameGraph3D(),
    'data-table': () => new DataTableView()
  }
  
  switchView(type: '3d-flame' | 'data-table'): BaseView {
    return this.views[type]()
  }
}
```

### **Phase 2: Multi-Analyzer Support (Weeks 5-8)**
**Goal**: Add analyzer extensibility and basic coordination

**New Features**:
- Analyzer plugin system
- Second analyzer type (Trace or Metrics)
- Basic data correlation
- Analyzer switching in UI

### **Phase 3: Multi-View Support (Weeks 9-12)**
**Goal**: Add view extensibility and multi-viewport

**New Features**:
- View plugin system
- Multi-viewport layouts
- View-specific controls
- Cross-view interactions

### **Phase 4: Session Management (Weeks 13-16)**
**Goal**: Add session persistence and sharing

**New Features**:
- Multi-session tabs
- Session persistence
- Data export/import
- Basic collaboration

## 🎨 **Modular Component Architecture**

### **MVP Component Structure**
```typescript
components/
├── App.tsx                        # Simple app wrapper
├── analyzers/
│   ├── AnalyzerControl.tsx        # Simple start/stop controls
│   └── FlameAnalyzerConfig.tsx    # Configuration for flame analyzer
├── views/
│   ├── ViewSelector.tsx           # Switch between 3D/Table views
│   ├── FlameGraph3D/              # Enhanced existing component
│   └── DataTable/                 # Simple data display
├── controls/
│   ├── SimpleControls.tsx         # Basic controls only
│   └── ExportControls.tsx         # Simple export functionality
└── layout/
    └── SimpleLayout.tsx           # Basic header + main content
```

### **Extension Points**
```typescript
// Adding new analyzer
components/analyzers/TraceAnalyzerConfig.tsx     # Drops in automatically
components/analyzers/MetricsAnalyzerConfig.tsx   # Drops in automatically

// Adding new view
components/views/TimelineChart/                  # Drops in automatically
components/views/HeatmapView/                    # Drops in automatically
```

## 🚀 **Technical Benefits of This Design**

### **1. Progressive Complexity**
- Start with 1 analyzer → Add more incrementally
- Start with single view → Add multi-viewport later
- Start with single session → Add multi-session later

### **2. True Modularity**
- New analyzers implement `BaseAnalyzer` interface
- New views implement `BaseView` interface
- Plugin registration system allows runtime extensibility

### **3. Reduced Cognitive Load**
- MVP UI shows only essential information
- Advanced features added progressively as users need them
- Clear separation between core and extension features

### **4. Easy Testing**
- Each analyzer can be tested independently
- Each view can be tested independently
- Plugin system allows isolated development

### **5. Deployment Flexibility**
- Core system can be deployed independently
- Plugins can be deployed separately
- Feature flags can control which analyzers/views are available

## 📈 **Scalability Path**

### **MVP → Basic Multi-Analyzer**
```typescript
// MVP: Single analyzer
const analyzer = new FlameGraphAnalyzer(config)

// Multi-analyzer: Plugin system
const analyzer1 = analyzerRegistry.create('flamegraph', config1)
const analyzer2 = analyzerRegistry.create('trace', config2)
```

### **Single View → Multi-Viewport**
```typescript
// MVP: Single view
<ViewportContainer>
  <FlameGraph3D data={data} />
</ViewportContainer>

// Multi-viewport: Layout system
<ViewportContainer layout="grid-2x2">
  <FlameGraph3D data={flameData} />
  <TimelineChart data={traceData} />
  <MetricsChart data={metricsData} />
  <DataTable data={allData} />
</ViewportContainer>
```

This design provides a clear, extensible foundation that starts simple but can grow into a comprehensive profiling platform without architectural rewrites. 