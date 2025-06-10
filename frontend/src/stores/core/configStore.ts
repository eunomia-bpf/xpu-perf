import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { FlameGraphConfig } from '@/types/flame.types';

interface ConfigState {
  config: FlameGraphConfig;
}

interface ConfigActions {
  updateConfig: (config: Partial<FlameGraphConfig>) => void;
  toggleAutoRotate: () => void;
  updateZSpacing: (spacing: number) => void;
  updateMinCount: (count: number) => void;
  updateMaxDepth: (depth: number) => void;
  changeColorScheme: () => void;
}

const initialConfig: FlameGraphConfig = {
  zSpacing: 25,
  minCount: 10,
  maxDepth: 8,
  colorSchemeIndex: 0,
  autoRotate: false
};

const initialState: ConfigState = {
  config: initialConfig
};

export const useConfigStore = create<ConfigState & ConfigActions>()(
  devtools(
    (set, _get) => ({
      ...initialState,

      updateConfig: (configUpdate: Partial<FlameGraphConfig>) =>
        set((state) => ({
          config: { ...state.config, ...configUpdate }
        })),

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
        }))
    }),
    {
      name: 'config-store'
    }
  )
); 