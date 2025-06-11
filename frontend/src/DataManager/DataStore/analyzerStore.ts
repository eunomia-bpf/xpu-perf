import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { 
  AnalyzerConfig, 
  AnalyzerInstance, 
  ViewConfig, 
  AnalyzerRegistry,
  BUILT_IN_ANALYZERS,
  BUILT_IN_VIEWS
} from '@/types/analyzer.types';

interface AnalyzerState {
  // Registry of available analyzers and views
  registry: AnalyzerRegistry;
  
  // Active analyzer instances
  instances: Record<string, AnalyzerInstance>;
  
  // Current selections
  selectedAnalyzerId: string | null;
  selectedInstanceId: string | null;
  selectedViewId: string | null;
}

interface AnalyzerActions {
  // Registry management
  registerAnalyzer: (analyzer: AnalyzerConfig) => void;
  registerView: (view: ViewConfig) => void;
  unregisterAnalyzer: (analyzerId: string) => void;
  unregisterView: (viewId: string) => void;
  
  // Instance management
  createAnalyzerInstance: (analyzerId: string, name: string, config?: Record<string, any>) => string;
  updateAnalyzerInstance: (instanceId: string, updates: Partial<AnalyzerInstance>) => void;
  deleteAnalyzerInstance: (instanceId: string) => void;
  duplicateAnalyzerInstance: (instanceId: string, newName: string) => string;
  
  // Selection management
  selectAnalyzer: (analyzerId: string) => void;
  selectInstance: (instanceId: string) => void;
  selectView: (viewId: string) => void;
  
  // Configuration updates
  updateInstanceConfig: (instanceId: string, config: Record<string, any>) => void;
  
  // Instance control
  startAnalyzer: (instanceId: string) => void;
  stopAnalyzer: (instanceId: string) => void;
  
  // Getters
  getAvailableAnalyzers: () => AnalyzerConfig[];
  getAvailableViews: (analyzerId?: string) => ViewConfig[];
  getInstancesByAnalyzer: (analyzerId: string) => AnalyzerInstance[];
  getCurrentInstance: () => AnalyzerInstance | null;
  getCurrentAnalyzer: () => AnalyzerConfig | null;
  getCurrentView: () => ViewConfig | null;
}

// Build initial registry from built-in configs
const buildInitialRegistry = (): AnalyzerRegistry => {
  const analyzers: Record<string, AnalyzerConfig> = {};
  const views: Record<string, ViewConfig> = {};
  
  BUILT_IN_ANALYZERS.forEach(analyzer => {
    analyzers[analyzer.id] = analyzer;
  });
  
  BUILT_IN_VIEWS.forEach(view => {
    views[view.id] = view;
  });
  
  return { analyzers, views };
};

const initialState: AnalyzerState = {
  registry: buildInitialRegistry(),
  instances: {},
  selectedAnalyzerId: 'flamegraph', // Default to flamegraph
  selectedInstanceId: null,
  selectedViewId: '3d-flame' // Default view
};

// Helper function to generate unique IDs
const generateId = () => `instance_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

export const useAnalyzerStore = create<AnalyzerState & AnalyzerActions>()(
  devtools(
    (set, get) => ({
      ...initialState,

      // Registry management
      registerAnalyzer: (analyzer: AnalyzerConfig) =>
        set((state) => ({
          registry: {
            ...state.registry,
            analyzers: { ...state.registry.analyzers, [analyzer.id]: analyzer }
          }
        })),

      registerView: (view: ViewConfig) =>
        set((state) => ({
          registry: {
            ...state.registry,
            views: { ...state.registry.views, [view.id]: view }
          }
        })),

      unregisterAnalyzer: (analyzerId: string) =>
        set((state) => {
          const { [analyzerId]: removed, ...analyzers } = state.registry.analyzers;
          return {
            registry: { ...state.registry, analyzers }
          };
        }),

      unregisterView: (viewId: string) =>
        set((state) => {
          const { [viewId]: removed, ...views } = state.registry.views;
          return {
            registry: { ...state.registry, views }
          };
        }),

      // Instance management
      createAnalyzerInstance: (analyzerId: string, name: string, config?: Record<string, any>) => {
        const analyzer = get().registry.analyzers[analyzerId];
        if (!analyzer) {
          throw new Error(`Analyzer ${analyzerId} not found`);
        }

        const instanceId = generateId();
        const instance: AnalyzerInstance = {
          id: instanceId,
          analyzerId,
          name,
          config: { ...analyzer.defaultConfig, ...config },
          status: 'idle',
          createdAt: new Date(),
          updatedAt: new Date()
        };

        set((state) => ({
          instances: { ...state.instances, [instanceId]: instance },
          selectedInstanceId: instanceId
        }));

        return instanceId;
      },

      updateAnalyzerInstance: (instanceId: string, updates: Partial<AnalyzerInstance>) =>
        set((state) => {
          const instance = state.instances[instanceId];
          if (!instance) return state;

          return {
            instances: {
              ...state.instances,
              [instanceId]: { ...instance, ...updates, updatedAt: new Date() }
            }
          };
        }),

      deleteAnalyzerInstance: (instanceId: string) =>
        set((state) => {
          const { [instanceId]: removed, ...instances } = state.instances;
          return {
            instances,
            selectedInstanceId: state.selectedInstanceId === instanceId ? null : state.selectedInstanceId
          };
        }),

      duplicateAnalyzerInstance: (instanceId: string, newName: string) => {
        const instance = get().instances[instanceId];
        if (!instance) {
          throw new Error(`Instance ${instanceId} not found`);
        }

        return get().createAnalyzerInstance(instance.analyzerId, newName, instance.config);
      },

      // Selection management
      selectAnalyzer: (analyzerId: string) =>
        set({ selectedAnalyzerId: analyzerId }),

      selectInstance: (instanceId: string) =>
        set({ selectedInstanceId: instanceId }),

      selectView: (viewId: string) =>
        set({ selectedViewId: viewId }),

      // Configuration updates
      updateInstanceConfig: (instanceId: string, config: Record<string, any>) =>
        set((state) => {
          const instance = state.instances[instanceId];
          if (!instance) return state;

          return {
            instances: {
              ...state.instances,
              [instanceId]: {
                ...instance,
                config: { ...instance.config, ...config },
                updatedAt: new Date()
              }
            }
          };
        }),

      // Instance control
      startAnalyzer: (instanceId: string) =>
        set((state) => {
          const instance = state.instances[instanceId];
          if (!instance) return state;

          // TODO: Implement actual analyzer start logic
          console.log(`Starting analyzer instance: ${instanceId}`);

          return {
            instances: {
              ...state.instances,
              [instanceId]: {
                ...instance,
                status: 'running',
                updatedAt: new Date()
              }
            }
          };
        }),

      stopAnalyzer: (instanceId: string) =>
        set((state) => {
          const instance = state.instances[instanceId];
          if (!instance) return state;

          // TODO: Implement actual analyzer stop logic
          console.log(`Stopping analyzer instance: ${instanceId}`);

          return {
            instances: {
              ...state.instances,
              [instanceId]: {
                ...instance,
                status: 'idle',
                updatedAt: new Date()
              }
            }
          };
        }),

      // Getters
      getAvailableAnalyzers: () => Object.values(get().registry.analyzers),

      getAvailableViews: (analyzerId?: string) => {
        const views = Object.values(get().registry.views);
        if (!analyzerId) return views;

        const analyzer = get().registry.analyzers[analyzerId];
        if (!analyzer) return views;

        return views.filter(view => analyzer.supportedViews.includes(view.id));
      },

      getInstancesByAnalyzer: (analyzerId: string) =>
        Object.values(get().instances).filter(instance => instance.analyzerId === analyzerId),

      getCurrentInstance: () => {
        const { selectedInstanceId, instances } = get();
        return selectedInstanceId ? instances[selectedInstanceId] || null : null;
      },

      getCurrentAnalyzer: () => {
        const { selectedAnalyzerId, registry } = get();
        return selectedAnalyzerId ? registry.analyzers[selectedAnalyzerId] || null : null;
      },

      getCurrentView: () => {
        const { selectedViewId, registry } = get();
        return selectedViewId ? registry.views[selectedViewId] || null : null;
      }
    }),
    {
      name: 'analyzer-store'
    }
  )
); 