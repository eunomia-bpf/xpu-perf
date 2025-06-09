import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { FlameGraphState, FlameData, FlameGraphConfig, FlameBlockMetadata, ThreadStats } from '@/types/flame.types';
import { FlameGraphDataLoader } from '@/utils/flameDataLoader';

interface FlameGraphActions {
  setData: (data: FlameData) => void;
  updateConfig: (config: Partial<FlameGraphConfig>) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setHoveredBlock: (block: FlameBlockMetadata | null) => void;
  loadSampleData: () => void;
  resetView: () => void;
  toggleAutoRotate: () => void;
  updateZSpacing: (spacing: number) => void;
  updateMinCount: (count: number) => void;
  updateMaxDepth: (depth: number) => void;
  changeColorScheme: () => void;
  updateStats: (stats: Record<string, ThreadStats>) => void;
}

const initialConfig: FlameGraphConfig = {
  zSpacing: 25,
  minCount: 10,
  maxDepth: 8,
  colorSchemeIndex: 0,
  autoRotate: false
};

const initialState: FlameGraphState = {
  data: {},
  config: initialConfig,
  stats: {},
  isLoading: false,
  error: null,
  hoveredBlock: null
};

export const useFlameGraphStore = create<FlameGraphState & FlameGraphActions>()(
  devtools(
    (set, get) => ({
      ...initialState,

      setData: (data: FlameData) => {
        const loader = new FlameGraphDataLoader();
        const stats = loader.getSummaryStats(data);
        set({ data, stats, error: null });
      },

      updateConfig: (configUpdate: Partial<FlameGraphConfig>) =>
        set((state) => ({
          config: { ...state.config, ...configUpdate }
        })),

      setLoading: (isLoading: boolean) => set({ isLoading }),

      setError: (error: string | null) => set({ error }),

      setHoveredBlock: (hoveredBlock: FlameBlockMetadata | null) => set({ hoveredBlock }),

      loadSampleData: () => {
        set({ isLoading: true });
        try {
          const loader = new FlameGraphDataLoader();
          const sampleData = loader.getSampleDataFromFiles();
          const stats = loader.getSummaryStats(sampleData);
          set({ data: sampleData, stats, isLoading: false, error: null });
        } catch (error) {
          set({ error: error instanceof Error ? error.message : 'Failed to load sample data', isLoading: false });
        }
      },

      resetView: () => set({ hoveredBlock: null }),

      toggleAutoRotate: () =>
        set((state) => ({
          config: { ...state.config, autoRotate: !state.config.autoRotate }
        })),

      updateZSpacing: (zSpacing: number) =>
        set((state) => ({
          config: { ...state.config, zSpacing }
        })),

      updateMinCount: (minCount: number) =>
        set((state) => ({
          config: { ...state.config, minCount }
        })),

      updateMaxDepth: (maxDepth: number) =>
        set((state) => ({
          config: { ...state.config, maxDepth }
        })),

      changeColorScheme: () =>
        set((state) => ({
          config: {
            ...state.config,
            colorSchemeIndex: (state.config.colorSchemeIndex + 1) % 4
          }
        })),

      updateStats: (stats: Record<string, ThreadStats>) => set({ stats })
    }),
    {
      name: 'flame-graph-store'
    }
  )
); 