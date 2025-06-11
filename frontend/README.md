# 🔥 Zero-Instrument Profiler Frontend

A **data-focused, completely decoupled** profiler frontend built with React, TypeScript, and Three.js.

## 🎯 **Key Features**

### ✅ **Complete Decoupling**
- **Views** are independent of **analyzers** - they only understand data formats
- **Data Sources** can be combined from multiple analyzer instances
- **Format-driven compatibility** determines which views can display data

### ✅ **Dynamic Configuration**
- **Schema-driven UIs** for both analyzers and views
- **Multiple analyzer instances** can run simultaneously
- **Flexible data combination** with merge/append/override modes

### ✅ **Modern Architecture**
- **TypeScript** with strict typing throughout
- **Zustand** for state management with proper separation of concerns
- **Modular design** with clear boundaries between layers

## 🚀 **Quick Start**

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

## 🏗️ **Architecture Overview**

```
🎨 Presentation Layer
├── AnalyzerEngine     # Analyzer instance management
├── ControlCenter      # Control components (NEW: DataSourceSelector)
├── ViewportEngine     # View components (simplified & decoupled)
└── LayoutManager      # App shell and layout

📊 Data Management Layer  
├── DataSourceStore    # NEW: Central data source management
├── AnalyzerStore      # Analyzer configs and instances
└── Legacy Stores      # Backward compatibility

🔧 Configuration Layer
├── analyzer.types.ts  # Core type definitions
└── flame.types.ts     # Legacy flame graph types
```

## 📊 **Data Flow**

1. **Analyzer Instances** → Produce data with specific formats
2. **DataSourceStore** → Manages and combines data sources  
3. **Views** → Check format compatibility and render data
4. **User** → Selects data sources and compatible views

## 🔄 **Usage Workflow**

1. **Create analyzer instances** with custom configurations
2. **Run analyzers** to generate profiling data
3. **Select data sources** - combine multiple analyzer outputs
4. **Choose compatible views** based on data format
5. **Visualize data** with format-appropriate rendering

## 📁 **Key Components**

### **NEW: Data-Focused Components**
- `DataSourceSelector` - Select and combine multiple data sources
- `DynamicViewControls` - Format-based view compatibility checking  
- `DataSourceStore` - Central data source management

### **Enhanced Components**
- `DynamicAnalyzer` - Manage multiple analyzer instances
- `FlameGraph3DView` - Simplified 3D visualization  
- `DataTableView` - Universal text-based data display

## 🎛️ **Available Analyzers**

- **🔥 Flame Graph Profiler**: CPU profiling with stack trace sampling
- **⏰ Wall Clock Analyzer**: Combined on-CPU and off-CPU profiling  
- **💤 Off-CPU Time Analyzer**: Analyze time spent off-CPU (blocking)

## 🖼️ **Available Views**

- **🎯 3D Flame Graph**: Interactive 3D visualization (optimized for flamegraph format)
- **📊 Data View**: Universal text display (works with any format)

## 🔧 **Configuration**

All components use **schema-driven configuration**:

```typescript
// Analyzer configuration
{
  duration: 30,        // seconds
  frequency: 99,       // Hz  
  target: "process",   // process name or PID
}

// View configuration  
{
  colorScheme: "hot-cold"  // visualization style
}
```

## 📚 **Documentation**

- **[Architecture Guide](./docs/ARCHITECTURE.md)** - Detailed technical architecture
- **[User Guide](./docs/USER_GUIDE.md)** - How to use the data-focused interface

## 🧪 **Testing**

```bash
# Run tests
npm test

# Run tests with UI
npm run test:ui

# Generate coverage
npm run coverage
```

## 🏆 **Benefits of This Design**

- **🔄 Extensible**: Add new analyzers/views without coupling
- **🧩 Modular**: Clear separation of concerns  
- **🎯 Flexible**: Combine data from multiple sources
- **🛡️ Type-Safe**: Full TypeScript coverage
- **👤 User-Friendly**: Data-focused workflow

## 🔮 **Roadmap**

- [ ] File upload data sources
- [ ] API-based data streaming  
- [ ] Additional view types (timeline, statistics)
- [ ] Advanced data filtering and transformation
- [ ] Export and sharing capabilities

---

**Built with ❤️ using a completely decoupled, data-first architecture!**
