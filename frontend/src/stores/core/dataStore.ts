import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { FlameData, ThreadStats } from '@/types/flame.types';
import { FlameGraphDataLoader } from '@/utils/flameDataLoader';

interface DataState {
  data: FlameData;
  stats: Record<string, ThreadStats>;
  isLoading: boolean;
  error: string | null;
}

interface DataActions {
  setData: (data: FlameData) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  loadSampleData: () => void;
  updateStats: (stats: Record<string, ThreadStats>) => void;
}

const initialState: DataState = {
  data: {},
  stats: {},
  isLoading: false,
  error: null
};

export const useDataStore = create<DataState & DataActions>()(
  devtools(
    (set, _get) => ({
      ...initialState,

      setData: (data: FlameData) => {
        const loader = new FlameGraphDataLoader();
        const stats = loader.getSummaryStats(data);
        set({ data, stats, error: null });
      },

      setLoading: (isLoading: boolean) => set({ isLoading }),

      setError: (error: string | null) => set({ error }),

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

      updateStats: (stats: Record<string, ThreadStats>) => set({ stats })
    }),
    {
      name: 'data-store'
    }
  )
); 