# 🏗️ Zero-Instrument Profiler Frontend Architecture

## 📋 **Overview**

The frontend implements a **completely decoupled, data-focused architecture** where views and analyzers are independent components that communicate through a centralized data management system.

## 🎯 **Core Design Principles**

### ✅ **Complete Decoupling**
- **Views** know nothing about analyzers - they only understand data formats
- **Analyzers** produce data without knowing how it will be visualized  
- **Data format compatibility** drives view selection, not analyzer types

### ✅ **Dynamic Configuration**
- **Analyzers** are dynamically configurable with schema-driven UIs
- **Views** have their own configuration schemas independent of analyzers
- **New analyzers/views** can be added without modifying existing code

### ✅ **Data-First Approach**
- **Data Sources** are the central concept, not analyzers or views
- **Multiple analyzer instances** can be combined into data selections
- **Views** consume the current data context regardless of source

---

## 🏛️ **Architecture Layers**

```
┌─────────────────────────────────────────────────────────────┐
│                    🎨 Presentation Layer                    │
├─────────────────────────────────────────────────────────────┤
│  AnalyzerEngine  │  ControlCenter   │  ViewportEngine      │
│  - DynamicAnalyzer│ - DataSourceSelector│ - FlameGraph3DView │
│                  │ - DynamicViewControls│ - DataTableView   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   📊 Data Management Layer                  │
├─────────────────────────────────────────────────────────────┤
│  DataSourceStore │  AnalyzerStore   │  Legacy Stores       │
│  - Data Sources  │ - Analyzer Configs│ - FlameGraphStore   │
│  - Data Selections│ - Instances      │ - ConfigStore       │
│  - Current Context│                  │ - InteractionStore  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    🔧 Configuration Layer                   │
├─────────────────────────────────────────────────────────────┤
│           analyzer.types.ts │ flame.types.ts              │
│  - AnalyzerConfig schemas   │ - Legacy flame types        │
│  - ViewConfig schemas       │                             │
│  - DataSource types         │                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 **Data Flow Architecture**

### **1. Analyzer Execution → Data Production**
```typescript
Analyzer Instance (with config) → Produces Data → Registers as DataSource
```

### **2. Data Source Management**  
```typescript
DataSourceStore manages:
├── Individual DataSources (from analyzers, files, APIs)
├── DataSelections (combinations of multiple sources)
└── CurrentDataContext (active data for views)
```

### **3. View Rendering**
```typescript
View checks currentDataContext.format → 
  If compatible → Render data
  If incompatible → Show format mismatch
```

---

## 📁 **Module Structure**

### **🔬 AnalyzerEngine**
**Purpose**: Analyzer instance management
- `DynamicAnalyzer.tsx` - UI wrapper for analyzer controls
- `index.ts` - Module exports

### **🎛️ ControlCenter** 
**Purpose**: All control components for the UI
- `DataSourceSelector.tsx` - **NEW** - Select & combine data sources
- `DynamicAnalyzerControls.tsx` - Create/manage analyzer instances  
- `DynamicViewControls.tsx` - Select views based on data compatibility
- `ViewSelector.tsx` - Simple view selection wrapper
- Legacy controls for backward compatibility

### **📊 DataManager**
**Purpose**: All data management and state

#### **DataStore/** - State Management
- `dataSourceStore.ts` - **NEW** - Central data source management
- `analyzerStore.ts` - Analyzer configs and instances
- `dataStore.ts` - Legacy flame graph data
- `configStore.ts` - View configurations  
- `interactionStore.ts` - UI interaction state

#### **DataProcessor/** - Data Processing
- `flameDataLoader.ts` - Sample data loading utilities

### **🖼️ ViewportEngine**
**Purpose**: All visualization components
- `FlameGraph3DView.tsx` - 3D flame graph (simplified)
- `DataTableView.tsx` - Simple text data display
- `ConfigPanel.tsx` - Generic configuration UI component
- `ViewportManager.tsx` - View orchestration

### **🏗️ LayoutManager**
**Purpose**: App layout and shell components
- `AppShell.tsx` - Main application shell
- `ViewContext.tsx` - Legacy view context for compatibility
- `SingleViewLayout.tsx` - Single view layout manager

### **🏷️ types/**
**Purpose**: TypeScript type definitions
- `analyzer.types.ts` - **CORE** - All analyzer, view, and data source types
- `flame.types.ts` - Legacy flame graph specific types

---

## 🔗 **Key Interfaces**

### **DataSource** - Central Data Concept
```typescript
interface DataSource {
  id: string;
  name: string;
  type: 'analyzer-instance' | 'file' | 'api';
  format: string;        // e.g., 'flamegraph', 'timeline', 'table'
  fields: string[];      // Available data fields
  data: any;            // Actual data
  lastUpdated: Date;
  metadata?: Record<string, any>;
}
```

### **DataSelection** - Multi-Source Combinations
```typescript
interface DataSelection {
  id: string;
  name: string;
  sources: string[];    // Array of DataSource IDs
  combinationMode: 'merge' | 'append' | 'override';
  resultFormat: string; // Combined format
  resultFields: string[]; // Combined fields
}
```

### **CurrentDataContext** - Active Data
```typescript
interface CurrentDataContext {
  selection: DataSelection | null;
  resolvedData: any;    // Combined/processed data
  format: string;       // Current data format
  fields: string[];     // Available fields
  sources: DataSource[]; // Source DataSources
}
```

### **AnalyzerConfig** - Dynamic Analyzer Definition
```typescript
interface AnalyzerConfig {
  id: string;
  displayName: string;
  description: string;
  configSchema: ConfigField[];  // Dynamic configuration
  outputFormat: string;         // Data format it produces
  outputFields: string[];       // Data fields it produces
}
```

### **ViewConfig** - Dynamic View Definition  
```typescript
interface ViewConfig {
  id: string;
  displayName: string;
  description: string;
  configSchema: ConfigField[];    // View-specific settings
  dataRequirements: {            // What data it can display
    format: string;              // Required format
    requiredFields: string[];    // Required fields
  };
}
```

---

## 🔀 **Data Flow Examples**

### **Example 1: Single Analyzer Data**
```
1. User creates "Flame Profiler" analyzer instance
2. Instance produces data with format="flamegraph", fields=["stack", "value"]  
3. DataSourceStore registers this as a DataSource
4. User selects this data source → becomes CurrentDataContext
5. Views check compatibility:
   - FlameGraph3DView: ✅ supports "flamegraph" format
   - DataTableView: ✅ supports any format
```

### **Example 2: Multi-Analyzer Combination**
```
1. User has multiple analyzer instances:
   - Instance A: format="flamegraph", fields=["stack", "value"]
   - Instance B: format="timeline", fields=["timestamp", "event"]
2. User creates DataSelection combining A + B with mode="merge"
3. Result: format="mixed", fields=["stack", "value", "timestamp", "event"]
4. Views check compatibility:
   - FlameGraph3DView: ⚠️ "mixed" not optimal, shows warning
   - DataTableView: ✅ accepts any format
```

---

## 🚀 **Adding New Components**

### **Adding a New Analyzer**
1. Define `AnalyzerConfig` in `analyzer.types.ts`
2. Add to `BUILT_IN_ANALYZERS` array
3. Implement data production logic (future: dynamic loading)

### **Adding a New View**
1. Define `ViewConfig` in `analyzer.types.ts` 
2. Add to `BUILT_IN_VIEWS` array
3. Create React component that consumes `useDataSourceStore()`
4. Check `currentDataContext.format` for compatibility

### **Adding New Data Source Types**
1. Extend `DataSource.type` union in `analyzer.types.ts`
2. Implement registration logic in `DataSourceSelector.tsx`
3. Views automatically support new data if format matches

---

## 🏆 **Benefits of This Architecture**

### ✅ **Complete Decoupling**
- Views work with any data source that matches their format requirements
- Analyzers don't need to know about visualization
- New analyzers/views can be added independently

### ✅ **Flexible Data Combination**  
- Multiple analyzer instances can be combined
- Different combination strategies (merge, append, override)
- Views adapt to combined data formats

### ✅ **Type Safety**
- Full TypeScript coverage with strict typing
- Schema-driven configuration prevents runtime errors
- Clear interfaces between all layers

### ✅ **Extensibility**
- Plugin-like architecture for analyzers and views
- Configuration schemas enable dynamic UIs
- Data format compatibility system is flexible

### ✅ **User Experience**
- Users can mix and match any data sources
- Views show clear compatibility status
- Simplified controls focused on data selection

---

## 🔄 **Migration from Legacy Design**

The architecture maintains backward compatibility:

- **Legacy stores** (`dataStore`, `configStore`, etc.) still exist
- **Old components** continue to work via composite `useFlameGraphStore()` 
- **New data-focused components** can coexist with legacy ones
- **Gradual migration** path available for existing code

---

## 🏁 **Current Implementation Status**

✅ **Completed:**
- Data-focused architecture with `DataSourceStore`
- Complete view-analyzer decoupling
- Dynamic data source selection and combination
- Format-based view compatibility checking
- Simplified view components (3D + text display)
- Schema-driven analyzer and view configuration

🔄 **Ready for Extension:**
- Additional analyzer types
- More sophisticated view components  
- File/API data source integration
- Advanced data transformation pipelines

This architecture provides a solid foundation for a extensible, maintainable profiler frontend that truly separates concerns and enables flexible data visualization workflows. 