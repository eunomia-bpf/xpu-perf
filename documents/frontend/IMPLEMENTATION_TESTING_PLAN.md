# Zero-Instrument Profiler Frontend - Implementation & Testing Plan

## 📋 Project Overview

**Project**: Multi-Session, Multi-Analyzer Profiler Frontend  
**Tech Stack**: React 19 + TypeScript + Vite + Three.js + Zustand + pnpm  
**Current Status**: Basic 3D visualization foundation established ✅  
**Target**: Production-ready comprehensive profiler with real-time multi-analyzer capabilities

## 🏗️ **Current Code Analysis & Architectural Transformation**

### **Existing Implementation Assessment**
```
✅ IMPLEMENTED (Foundation):
├── Basic 3D FlameGraph3D.tsx with React Three Fiber
├── FlameBlocks.tsx and ThreadLabel.tsx components  
├── Core flameGraphStore.ts with Zustand + devtools
├── ControlPanel.tsx with basic interactive controls
├── InfoPanel.tsx for data display
├── Utils: flameDataLoader.ts, colorSchemes.ts
├── Types: flame.types.ts
└── Basic layout components (AppLayout, NavigationHeader, etc.)

🔄 NEEDS TRANSFORMATION:
├── Single-session → Multi-session tab system
├── Single analyzer → Multi-analyzer engine
├── Monolithic controls → View-specific control components
├── Static data → Real-time streaming architecture
├── Single view → Multi-viewport system
└── Basic state → Comprehensive session management
```

### **Target Multi-Analyzer Architecture**

#### **1. Session Management System**
```typescript
// New: Multi-session architecture
src/
├── stores/
│   ├── session/
│   │   ├── sessionManager.ts      📝 (Multi-session orchestration)
│   │   ├── sessionStore.ts        📝 (Individual session state)
│   │   └── sessionPersistence.ts  📝 (Browser storage)
│   ├── analyzers/
│   │   ├── analyzerManager.ts     📝 (Analyzer lifecycle)
│   │   ├── traceAnalyzer.ts       📝 (Trace data collection)
│   │   ├── metricsAnalyzer.ts     📝 (PMU/system metrics)
│   │   ├── flameAnalyzer.ts       📝 (Enhanced existing)
│   │   └── staticAnalyzer.ts      📝 (Binary/symbol analysis)
│   └── ui/
│       ├── viewportStore.ts       📝 (Multi-viewport management)
│       ├── layoutStore.ts         📝 (Grid/tab layouts)
│       └── correlationStore.ts    📝 (Cross-view data correlation)
```

#### **2. Enhanced Component Architecture**
```typescript
// Evolved: View-specific controls with tight coupling
components/
├── Sessions/                      📝 (New: session management)
│   ├── SessionTabs.tsx
│   ├── SessionManager.tsx
│   └── SessionPersistence.tsx
├── Analyzers/                     📝 (New: analyzer controls)
│   ├── AnalyzerManager.tsx
│   ├── TraceAnalyzerControl.tsx
│   ├── MetricsAnalyzerControl.tsx
│   ├── FlameAnalyzerControl.tsx
│   └── StaticAnalyzerControl.tsx
├── Visualizations/                🔄 (Enhanced: view-specific)
│   ├── FlameGraph3D/             ✅ (Enhanced existing)
│   │   ├── FlameGraph3D.tsx      ✅ (Core component)
│   │   ├── FlameGraph3DControls.tsx 📝 (Dedicated controls)
│   │   └── FlameGraph3DInteractions.tsx 📝 (Interaction logic)
│   ├── FlameGraph2D/             📝 (New: 2D alternative)
│   │   ├── FlameGraph2D.tsx
│   │   ├── FlameGraph2DControls.tsx
│   │   └── FlameGraph2DInteractions.tsx
│   ├── TimelineChart/            📝 (New: trace visualization)
│   │   ├── TimelineChart.tsx
│   │   ├── TimelineControls.tsx
│   │   └── TimelineInteractions.tsx
│   ├── MetricsChart/             📝 (New: system metrics)
│   │   ├── MetricsChart.tsx
│   │   ├── MetricsControls.tsx
│   │   └── MetricsInteractions.tsx
│   └── DataTable/                📝 (New: data browser)
│       ├── DataTable.tsx
│       ├── DataTableControls.tsx
│       └── DataTableInteractions.tsx
├── ControlCenter/                📝 (New: centralized controls)
│   ├── AnalyzerControlPanel.tsx
│   ├── VisualizationControlPanel.tsx
│   └── DataBrowserPanel.tsx
└── Layout/                       ✅ (Enhanced existing)
    ├── AppLayout.tsx             ✅ (Multi-session support)
    ├── ViewportContainer.tsx     📝 (Multi-viewport)
    └── GridLayout.tsx            📝 (Dynamic layouts)
```

## 🛠️ **Phase 1: Multi-Session Foundation (Weeks 1-4)**

### **1.1 Session Management Implementation**

#### **Enhanced Package Configuration for Multi-Session Support**
```json
// package.json additions for session management
{
  "dependencies": {
    // Existing dependencies...
    "uuid": "^9.0.0",              // Session ID generation
    "zustand": "^4.4.0",           // Enhanced state management
    "immer": "^10.0.0",            // Immutable state updates
    "broadcast-channel": "^5.1.0", // Cross-tab communication
    "idb": "^8.0.0"                // IndexedDB for persistence
  },
  "scripts": {
    // Enhanced development workflow
    "dev:multi": "concurrently \"pnpm dev\" \"pnpm dev:storybook\"",
    "test:session": "vitest run --config vitest.session.config.ts",
    "test:analyzers": "vitest run --config vitest.analyzers.config.ts"
  }
}
```

#### **Session Store Architecture**
```typescript
// stores/session/sessionManager.ts
interface SessionManager {
  // Session lifecycle
  sessions: Map<SessionId, SessionState>
  activeSessionId: SessionId | null
  
  // Session operations
  createSession: (config?: SessionConfig) => SessionId
  switchSession: (id: SessionId) => void
  closeSession: (id: SessionId) => void
  duplicateSession: (id: SessionId) => SessionId
  
  // Session persistence
  saveSession: (id: SessionId) => Promise<void>
  loadSession: (id: SessionId) => Promise<void>
  exportSession: (id: SessionId, format: ExportFormat) => Promise<Blob>
}

// stores/session/sessionStore.ts  
interface SessionState {
  id: SessionId
  name: string
  createdAt: Date
  updatedAt: Date
  
  // Analyzer state
  analyzers: Map<AnalyzerId, AnalyzerState>
  activeAnalyzers: Set<AnalyzerId>
  
  // Visualization state
  viewports: Map<ViewportId, ViewportState>
  layout: LayoutConfig
  
  // Data state
  data: SessionData
  metadata: SessionMetadata
}
```

#### **Session Tab Component**
```typescript
// components/Sessions/SessionTabs.tsx
const SessionTabs: React.FC = () => {
  const { sessions, activeSessionId, createSession, switchSession, closeSession } = useSessionManager()
  
  return (
    <div className="session-tabs">
      {Array.from(sessions.entries()).map(([id, session]) => (
        <SessionTab
          key={id}
          session={session}
          isActive={id === activeSessionId}
          onActivate={() => switchSession(id)}
          onClose={() => closeSession(id)}
          onDuplicate={() => duplicateSession(id)}
        />
      ))}
      <NewSessionButton onCreate={createSession} />
    </div>
  )
}
```

### **1.2 Analyzer Framework Foundation**

#### **Base Analyzer Architecture**
```typescript
// stores/analyzers/analyzerManager.ts
interface AnalyzerManager {
  // Analyzer lifecycle
  startAnalyzer: (sessionId: SessionId, type: AnalyzerType, config: AnalyzerConfig) => AnalyzerId
  stopAnalyzer: (analyzerId: AnalyzerId) => void
  configureAnalyzer: (analyzerId: AnalyzerId, config: AnalyzerConfig) => void
  
  // Analyzer state
  getAnalyzerStatus: (analyzerId: AnalyzerId) => AnalyzerStatus
  getAnalyzerData: (analyzerId: AnalyzerId) => AnalyzerData
  
  // Data synchronization
  synchronizeAnalyzers: (analyzerIds: AnalyzerId[]) => void
  correlateData: (analyzerIds: AnalyzerId[]) => CorrelatedData
}

// Base analyzer interface
interface BaseAnalyzer {
  id: AnalyzerId
  type: AnalyzerType
  sessionId: SessionId
  config: AnalyzerConfig
  status: AnalyzerStatus
  
  start: () => Promise<void>
  stop: () => Promise<void>
  configure: (config: AnalyzerConfig) => void
  getData: () => AnalyzerData
}
```

#### **FlameGraph Analyzer Enhancement (Build on Existing)**
```typescript
// stores/analyzers/flameAnalyzer.ts (Enhanced from existing flameGraphStore.ts)
class FlameAnalyzer extends BaseAnalyzer {
  constructor(sessionId: SessionId, config: FlameAnalyzerConfig) {
    super(sessionId, 'flamegraph', config)
  }
  
  async start() {
    // Enhanced from existing loadSampleData functionality
    this.status = 'starting'
    await this.initializeStackSampling()
    this.status = 'running'
    this.startDataCollection()
  }
  
  private async initializeStackSampling() {
    // Use existing flameDataLoader logic
    const loader = new FlameDataLoader()
    const initialData = await loader.loadSampleData()
    this.updateData(initialData)
  }
  
  private startDataCollection() {
    // Real-time stack sampling
    this.samplingInterval = setInterval(() => {
      this.collectStackSample()
    }, this.config.sampleInterval || 100)
  }
}
```

### **1.3 Multi-Viewport System**

#### **Viewport Container Implementation**
```typescript
// components/Layout/ViewportContainer.tsx
const ViewportContainer: React.FC<ViewportContainerProps> = ({ sessionId }) => {
  const session = useSession(sessionId)
  const { viewports, layout } = session
  
  return (
    <div className="viewport-container">
      <ViewportLayoutRenderer layout={layout}>
        {Array.from(viewports.entries()).map(([id, viewport]) => (
          <ViewportWrapper key={id} viewport={viewport}>
            <ViewportComponent
              type={viewport.type}
              dataSource={viewport.dataSource}
              config={viewport.config}
              controls={viewport.controls}
            />
          </ViewportWrapper>
        ))}
      </ViewportLayoutRenderer>
    </div>
  )
}
```

## 🎨 **Phase 2: Analyzer Engine Implementation (Weeks 5-8)**

### **2.1 Complete Analyzer Implementation**

#### **Trace Analyzer (Function Probes + Events)**
```typescript
// stores/analyzers/traceAnalyzer.ts
class TraceAnalyzer extends BaseAnalyzer {
  private eventBuffer: TraceEvent[] = []
  private webSocketConnection: WebSocket | null = null
  
  async start() {
    await this.establishConnection()
    await this.setupFunctionProbes()
    this.startEventCollection()
  }
  
  private async setupFunctionProbes() {
    const { includePatterns, excludePatterns } = this.config
    // Setup function entry/exit probes
    await this.instrumentFunctions(includePatterns, excludePatterns)
  }
  
  private startEventCollection() {
    this.webSocketConnection?.addEventListener('message', (event) => {
      const traceEvent: TraceEvent = JSON.parse(event.data)
      this.processTraceEvent(traceEvent)
    })
  }
  
  private processTraceEvent(event: TraceEvent) {
    this.eventBuffer.push(event)
    if (this.eventBuffer.length >= this.config.bufferSize) {
      this.flushBuffer()
    }
  }
}
```

#### **Metrics Analyzer (PMU + System Metrics)**
```typescript
// stores/analyzers/metricsAnalyzer.ts
class MetricsAnalyzer extends BaseAnalyzer {
  private metricsCollector: MetricsCollector
  private pmuReader: PMUReader
  
  async start() {
    await this.initializePMU()
    this.startMetricsCollection()
  }
  
  private async initializePMU() {
    this.pmuReader = new PMUReader({
      events: ['cpu_cycles', 'instructions', 'cache_misses'],
      interval: this.config.interval || 100
    })
  }
  
  private startMetricsCollection() {
    setInterval(() => {
      const systemMetrics = this.collectSystemMetrics()
      const pmuData = this.pmuReader.read()
      
      this.updateData({
        timestamp: Date.now(),
        systemMetrics,
        pmuData
      })
    }, this.config.interval)
  }
}
```

#### **Static Analyzer (Binary + Symbol Analysis)**
```typescript
// stores/analyzers/staticAnalyzer.ts
class StaticAnalyzer extends BaseAnalyzer {
  private symbolTable: SymbolTable = new Map()
  private binaryInfo: BinaryInfo | null = null
  
  async start() {
    await this.loadBinary()
    await this.parseSymbols()
    await this.mapSourceCode()
  }
  
  private async loadBinary() {
    const binaryPath = this.config.binaryPath
    this.binaryInfo = await this.parseBinary(binaryPath)
  }
  
  private async parseSymbols() {
    if (!this.binaryInfo) return
    
    this.symbolTable = await this.extractSymbols(this.binaryInfo)
    this.updateData({ symbols: this.symbolTable })
  }
}
```

### **2.2 View-Specific Control Components**

#### **3D Flame Graph Controls (Tight Coupling)**
```typescript
// components/Visualizations/FlameGraph3D/FlameGraph3DControls.tsx
const FlameGraph3DControls: React.FC<FlameGraph3DControlsProps> = ({ 
  flameGraphRef,
  data,
  onConfigChange 
}) => {
  // Direct integration with 3D scene
  const handleCameraReset = () => {
    flameGraphRef.current?.resetCamera()
  }
  
  const handleZSpacingChange = (spacing: number) => {
    flameGraphRef.current?.updateZSpacing(spacing)
    onConfigChange({ zSpacing: spacing })
  }
  
  return (
    <div className="flamegraph-3d-controls">
      <ControlSection title="Camera">
        <Button onClick={handleCameraReset}>Reset View</Button>
        <Button onClick={() => flameGraphRef.current?.fitToView()}>Fit All</Button>
      </ControlSection>
      
      <ControlSection title="Rendering">
        <Slider
          label="Z-Spacing"
          value={config.zSpacing}
          onChange={handleZSpacingChange}
          min={5}
          max={50}
        />
      </ControlSection>
      
      <ControlSection title="Data">
        <ThreadFilter
          threads={data.threads}
          selected={config.selectedThreads}
          onChange={handleThreadSelection}
        />
      </ControlSection>
    </div>
  )
}
```

#### **Timeline Chart Controls**
```typescript
// components/Visualizations/TimelineChart/TimelineControls.tsx
const TimelineControls: React.FC<TimelineControlsProps> = ({
  timelineRef,
  data,
  onTimeRangeChange
}) => {
  return (
    <div className="timeline-controls">
      <ControlSection title="Navigation">
        <TimeRangeSlider
          min={data.startTime}
          max={data.endTime}
          value={[config.viewStart, config.viewEnd]}
          onChange={onTimeRangeChange}
        />
      </ControlSection>
      
      <ControlSection title="Correlation">
        <Checkbox
          label="Link with 3D Flame Graph"
          checked={config.linkWithFlameGraph}
          onChange={handleLinkChange}
        />
      </ControlSection>
    </div>
  )
}
```

## 🧪 **Phase 3: Enhanced Testing Strategy (Weeks 9-12)**

### **3.1 Multi-Session Testing Framework**
```typescript
// src/__tests__/session/sessionManager.test.ts
describe('SessionManager', () => {
  let sessionManager: SessionManager
  
  beforeEach(() => {
    sessionManager = new SessionManager()
  })
  
  describe('session lifecycle', () => {
    it('should create new session with unique ID', () => {
      const sessionId = sessionManager.createSession({ name: 'Test Session' })
      expect(sessionManager.sessions.has(sessionId)).toBe(true)
    })
    
    it('should switch between sessions without data loss', () => {
      const session1 = sessionManager.createSession({ name: 'Session 1' })
      const session2 = sessionManager.createSession({ name: 'Session 2' })
      
      sessionManager.switchSession(session1)
      expect(sessionManager.activeSessionId).toBe(session1)
      
      sessionManager.switchSession(session2)
      expect(sessionManager.activeSessionId).toBe(session2)
      expect(sessionManager.sessions.size).toBe(2)
    })
  })
})
```

### **3.2 Analyzer Testing Framework**
```typescript
// src/__tests__/analyzers/traceAnalyzer.test.ts
describe('TraceAnalyzer', () => {
  let analyzer: TraceAnalyzer
  let mockWebSocket: MockWebSocket
  
  beforeEach(() => {
    mockWebSocket = new MockWebSocket()
    analyzer = new TraceAnalyzer('session-1', { 
      target: 'test-process',
      bufferSize: 1000
    })
  })
  
  it('should collect and process trace events', async () => {
    await analyzer.start()
    
    const mockEvent: TraceEvent = {
      timestamp: Date.now(),
      eventType: 'function_entry',
      functionName: 'main',
      threadId: 'thread-1'
    }
    
    mockWebSocket.simulateMessage(mockEvent)
    
    const data = analyzer.getData()
    expect(data.events).toContainEqual(mockEvent)
  })
})
```

### **3.3 Visual Testing for Multi-Viewport**
```typescript
// src/__tests__/visual/multiViewport.test.tsx
describe('Multi-Viewport Layouts', () => {
  it('should render grid layout with multiple views', async () => {
    const { container } = render(
      <SessionProvider sessionId="test-session">
        <ViewportContainer layout="grid-2x2">
          <FlameGraph3D />
          <TimelineChart />
          <MetricsChart />
          <DataTable />
        </ViewportContainer>
      </SessionProvider>
    )
    
    expect(container.querySelectorAll('.viewport')).toHaveLength(4)
    expect(container.querySelector('.grid-layout')).toBeInTheDocument()
  })
  
  it('should maintain view state during layout changes', async () => {
    const { rerender } = render(
      <ViewportContainer layout="grid-2x2">
        <FlameGraph3D config={{ zSpacing: 25 }} />
      </ViewportContainer>
    )
    
    rerender(
      <ViewportContainer layout="tab">
        <FlameGraph3D config={{ zSpacing: 25 }} />
      </ViewportContainer>
    )
    
    // Verify 3D view maintains configuration
    expect(screen.getByDisplayValue('25')).toBeInTheDocument()
  })
})
```

## 🚀 **Phase 4: Production Features & Optimization (Weeks 13-16)**

### **4.1 Real-time Performance Optimization**

#### **Data Streaming Optimization**
```typescript
// utils/streaming/dataStreamOptimizer.ts
class DataStreamOptimizer {
  private bufferManager: CircularBuffer
  private compressionHandler: CompressionHandler
  private rateLimiter: RateLimiter
  
  optimizeDataStream(stream: DataStream): OptimizedStream {
    return stream
      .pipe(this.rateLimiter.throttle(60)) // 60 FPS max
      .pipe(this.compressionHandler.compress())
      .pipe(this.bufferManager.buffer(1000))
  }
}
```

#### **Multi-Analyzer Coordination**
```typescript
// stores/coordination/dataCoordinator.ts
class DataCoordinator {
  synchronizeAnalyzers(analyzers: AnalyzerId[]): void {
    const timestamps = this.getLatestTimestamps(analyzers)
    const syncTime = Math.min(...timestamps)
    
    analyzers.forEach(id => {
      this.syncAnalyzerToTime(id, syncTime)
    })
  }
  
  correlateData(analyzerIds: AnalyzerId[]): CorrelatedData {
    const datasets = analyzerIds.map(id => this.getAnalyzerData(id))
    return this.correlationEngine.correlate(datasets)
  }
}
```

### **4.2 Advanced Visualization Features**

#### **Cross-View Data Correlation**
```typescript
// components/Visualizations/CrossViewCorrelation.tsx
const CrossViewCorrelation: React.FC = () => {
  const { correlatedData, activeSelection } = useDataCorrelation()
  
  useEffect(() => {
    if (activeSelection.type === 'function' && activeSelection.source === '3d-flame') {
      // Highlight function in timeline
      highlightInTimeline(activeSelection.functionName)
      // Show metrics for function
      showMetricsForFunction(activeSelection.functionName)
    }
  }, [activeSelection])
  
  return null // Pure coordination component
}
```

#### **Advanced Export Capabilities**
```typescript
// utils/export/sessionExporter.ts
class SessionExporter {
  async exportSession(sessionId: SessionId, format: ExportFormat): Promise<Blob> {
    const session = await this.getSessionData(sessionId)
    
    switch (format) {
      case 'json':
        return this.exportAsJSON(session)
      case 'report':
        return this.generateReport(session)
      case 'flamegraph':
        return this.exportFlameGraphSVG(session)
      case 'csv':
        return this.exportAsCSV(session)
    }
  }
}
```

## 📊 **Migration Timeline & Risk Assessment**

### **Week-by-Week Implementation Plan**

**Week 1-2: Session Foundation**
- Implement SessionManager and SessionStore ✅
- Create SessionTabs component 
- Add session persistence to browser storage
- Migrate existing flameGraphStore to session-aware architecture

**Week 3-4: Basic Multi-Analyzer**
- Implement AnalyzerManager foundation
- Enhance existing FlameAnalyzer (from flameGraphStore)
- Create basic TraceAnalyzer and MetricsAnalyzer
- Add analyzer control components

**Week 5-6: Visualization Enhancement**
- Extract view-specific controls from existing components
- Implement ViewportContainer with layout management
- Add TimelineChart and MetricsChart components
- Integrate controls with view components

**Week 7-8: Data Coordination**
- Implement cross-analyzer data synchronization
- Add real-time streaming capabilities
- Create data correlation engine
- Add cross-view interaction features

**Week 9-10: Testing & Quality**
- Comprehensive test suite for session management
- Analyzer testing framework
- Visual regression testing for multi-viewport
- Performance testing for real-time data

**Week 11-12: Performance Optimization**
- Data streaming optimization
- Memory management for large datasets
- Rendering performance optimization
- Bundle optimization

**Week 13-14: Advanced Features**
- Export/import functionality
- Session sharing capabilities
- Advanced analytics features
- Cross-browser compatibility

**Week 15-16: Production Readiness**
- Final performance optimization
- Documentation and user guides
- Deployment preparation
- Production monitoring setup

### **Risk Mitigation Strategies**

1. **Data Loss During Migration**
   - *Risk*: Existing session data loss during store refactoring
   - *Mitigation*: Implement migration utilities, maintain backward compatibility
   - *Testing*: Comprehensive data migration tests

2. **Performance Degradation**
   - *Risk*: Multi-session overhead affects performance
   - *Mitigation*: Lazy loading, memory management, efficient state updates
   - *Testing*: Continuous performance monitoring

3. **UI Complexity**
   - *Risk*: Multi-analyzer interface becomes overwhelming
   - *Mitigation*: Progressive disclosure, smart defaults, user guidance
   - *Testing*: Usability testing with target users

## 🛠️ **Enhanced Development Workflow**

### **Multi-Environment Development**
```bash
# Development commands for multi-analyzer system
pnpm dev:session          # Session-focused development
pnpm dev:analyzers        # Analyzer development mode
pnpm dev:visualizations   # Visualization development
pnpm dev:full            # Full system development

# Testing commands
pnpm test:session        # Session management tests
pnpm test:analyzers      # Analyzer engine tests
pnpm test:ui             # UI component tests
pnpm test:integration    # Cross-component integration tests
pnpm test:performance    # Performance regression tests
```

### **Enhanced CI/CD Pipeline**
```yaml
# .github/workflows/multi-analyzer.yml
name: Multi-Analyzer Profiler CI/CD
on: [push, pull_request]

jobs:
  test-session:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
      - run: pnpm test:session

  test-analyzers:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
      - run: pnpm test:analyzers

  test-integration:
    runs-on: ubuntu-latest
    needs: [test-session, test-analyzers]
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v2
      - run: pnpm test:integration
      - run: pnpm test:performance
```

This comprehensive implementation plan provides a clear, phased approach to transforming the existing basic 3D flame graph viewer into a professional, multi-session, multi-analyzer profiler while maintaining stability and performance throughout the migration. 