# Zero-Instrument Profiler Frontend

## 🏗️ Architecture Overview

This frontend is built with a **modular, MVP-first architecture** that prioritizes maintainability, extensibility, and clean separation of concerns.

### Core Technologies
- **React 19** + **TypeScript** + **Vite**
- **Three.js** + **@react-three/fiber** for 3D visualization
- **Zustand** for state management
- **Tailwind CSS** for styling
- **Vitest** for testing

## 📁 Project Structure

```
frontend/src/
├── components/
│   ├── analyzers/           # Analyzer control components
│   │   ├── AnalyzerControlPanel.tsx
│   │   └── index.ts
│   ├── views/              # Self-contained view components
│   │   ├── FlameGraph3DView.tsx    # 3D visualization with integrated controls
│   │   ├── FlameGraph2DView.tsx    # 2D flame graph placeholder
│   │   ├── DataTableView.tsx       # Data table with real data
│   │   ├── LineChartView.tsx       # Chart visualization placeholder
│   │   ├── ViewportContainer.tsx   # View switcher
│   │   └── index.ts
│   ├── FlameGraph3D/       # Core 3D rendering components
│   │   ├── FlameGraphContent.tsx
│   │   ├── FlameBlocks.tsx
│   │   ├── LightingSystem.tsx
│   │   ├── ThreadLabel.tsx
│   │   └── index.ts
│   ├── Layout/             # Layout and shell components
│   │   ├── AppLayout.tsx
│   │   ├── NavigationHeader.tsx
│   │   ├── StatusBar.tsx
│   │   ├── Sidebar.tsx
│   │   ├── MainViewport.tsx
│   │   └── index.ts
│   ├── UI/                 # Shared UI components
│   │   ├── shared/
│   │   │   ├── LoadingSpinner.tsx
│   │   │   ├── ErrorDisplay.tsx
│   │   │   └── index.ts
│   │   └── index.ts
│   └── index.ts            # Main component barrel export
├── stores/                 # Modular Zustand stores
│   ├── core/
│   │   ├── dataStore.ts    # Data and loading state
│   │   └── configStore.ts  # Configuration state
│   ├── ui/
│   │   └── interactionStore.ts  # UI interaction state
│   └── index.ts            # Store composition
├── types/                  # TypeScript type definitions
│   └── flame.types.ts
├── utils/                  # Utility functions
│   ├── flameDataLoader.ts
│   ├── colorSchemes.ts
│   └── __tests__/
└── App.tsx                 # Main application component
```

## 🎯 Key Architectural Principles

### 1. **Self-Contained Views**
Each view component includes:
- Main visualization logic
- Integrated, collapsible controls panel
- View-specific state management
- Export functionality

**Example**: `FlameGraph3DView` contains both the 3D canvas AND its control panel.

### 2. **Modular State Management**
```typescript
// Separated concerns in stores
const dataStore = useDataStore();      // Data loading, processing
const configStore = useConfigStore();  // Configuration settings  
const uiStore = useInteractionStore(); // UI interactions

// Backward compatible composite
const store = useFlameGraphStore();    // Combines all stores
```

### 3. **Clean Component Separation**
- **Analyzers**: Control data collection
- **Views**: Display and interact with data
- **Layout**: Application shell and navigation
- **FlameGraph3D**: Core 3D rendering logic
- **UI**: Shared components

### 4. **Progressive Enhancement Ready**
The architecture supports easy addition of:
- New analyzer types (plugin system ready)
- New view types (registry pattern)
- Multi-viewport layouts
- Session management

## 🚀 Development Workflow

### Running the Application
```bash
npm install
npm run dev
```

### Testing
```bash
npm run test
```

### Building
```bash
npm run build
```

## 🎨 UI/UX Features

### Current Features
- **Simplified Header**: Clean "menu" with action buttons
- **Analyzer Control Panel**: Start/stop, status, basic config
- **View Switching**: Radio buttons for 4 view types
- **Self-Contained Views**: Each view manages its own controls
- **Responsive Design**: Works on different screen sizes

### View Types
1. **3D Flame Graph**: Interactive 3D visualization with hover info
2. **2D Flame Graph**: Traditional horizontal flame graph (placeholder)
3. **Data Table**: Searchable, sortable table with real data
4. **Line Chart**: Time-series visualization (placeholder)

## 📊 Data Flow

```
Data Loading → Processing → Store → View → User Interaction
     ↓            ↓          ↓      ↓         ↓
flameDataLoader → dataStore → React → Canvas → Controls
```

## 🔧 Extension Points

### Adding a New View Type
1. Create new view component in `src/components/views/`
2. Add to `ViewType` union in `ViewportContainer.tsx`
3. Add radio button option in `AnalyzerControlPanel.tsx`
4. Export from `src/components/views/index.ts`

### Adding a New Analyzer
1. Create analyzer component in `src/components/analyzers/`
2. Extend store if needed
3. Add to analyzer selection UI

## 🏆 Benefits of This Architecture

### ✅ **Maintainability**
- Clear separation of concerns
- Modular components that can be developed independently
- Consistent patterns throughout codebase

### ✅ **Extensibility**
- Easy to add new view types
- Plugin-ready analyzer system
- Scalable state management

### ✅ **Performance**
- Lazy loading of heavy components
- Optimized re-renders with Zustand
- Efficient 3D rendering with Three.js

### ✅ **Developer Experience**
- TypeScript for type safety
- Comprehensive testing setup
- Hot module replacement with Vite
- Clear import paths with barrel exports

### ✅ **User Experience**
- Self-contained views reduce cognitive load
- Smooth transitions between view types
- Responsive, professional interface
- Progressive disclosure of complex features

This architecture successfully balances **immediate usability** (MVP) with **future extensibility** (enterprise features), making it perfect for both rapid prototyping and long-term development.
