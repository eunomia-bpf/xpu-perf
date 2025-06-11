import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { DataSource, DataSelection, CurrentDataContext } from '@/types/analyzer.types';

interface DataSourceState {
  // Available data sources
  dataSources: Record<string, DataSource>;
  
  // Data selections (combinations of sources)
  dataSelections: Record<string, DataSelection>;
  
  // Current active data context
  currentDataContext: CurrentDataContext;
}

interface DataSourceActions {
  // Data source management
  registerDataSource: (source: DataSource) => void;
  updateDataSource: (sourceId: string, updates: Partial<DataSource>) => void;
  removeDataSource: (sourceId: string) => void;
  
  // Data selection management
  createDataSelection: (name: string, sourceIds: string[], mode?: 'merge' | 'append' | 'override') => string;
  updateDataSelection: (selectionId: string, updates: Partial<DataSelection>) => void;
  removeDataSelection: (selectionId: string) => void;
  
  // Current data context
  setCurrentDataSelection: (selectionId: string) => void;
  setCurrentDataDirect: (data: any, format: string, fields: string[]) => void;
  
  // Getters
  getAvailableDataSources: () => DataSource[];
  getDataSourcesByFormat: (format: string) => DataSource[];
  getCurrentDataContext: () => CurrentDataContext;
  isFormatAvailable: (format: string) => boolean;
}

const generateId = () => `data_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

// Helper function to combine data from multiple sources
const combineDataSources = (sources: DataSource[], mode: 'merge' | 'append' | 'override'): any => {
  if (sources.length === 0) return {};
  if (sources.length === 1) return sources[0]?.data || {};
  
  switch (mode) {
    case 'merge':
      return sources.reduce((acc, source) => ({ ...acc, ...(source?.data || {}) }), {});
    case 'append':
      return sources.reduce((acc, source) => {
        if (Array.isArray(acc) && Array.isArray(source?.data)) {
          return [...acc, ...(source?.data || [])];
        }
        return { ...acc, ...(source?.data || {}) };
      }, {});
    case 'override':
      return sources[sources.length - 1]?.data || {};
    default:
      return sources[0]?.data || {};
  }
};

const initialState: DataSourceState = {
  dataSources: {},
  dataSelections: {},
  currentDataContext: {
    selection: null,
    resolvedData: {},
    format: 'none',
    fields: [],
    sources: []
  }
};

export const useDataSourceStore = create<DataSourceState & DataSourceActions>()(
  devtools(
    (set, get) => ({
      ...initialState,

      // Data source management
      registerDataSource: (source: DataSource) =>
        set((state) => ({
          dataSources: { ...state.dataSources, [source.id]: source }
        })),

      updateDataSource: (sourceId: string, updates: Partial<DataSource>) =>
        set((state) => {
          const source = state.dataSources[sourceId];
          if (!source) return state;

          const updatedSource = { ...source, ...updates, lastUpdated: new Date() };
          const newDataSources = { ...state.dataSources, [sourceId]: updatedSource };

          // If this source is part of current selection, update context
          const currentSelection = state.currentDataContext.selection;
          if (currentSelection && currentSelection.sources.includes(sourceId)) {
            const selectedSources = currentSelection.sources
              .map(id => newDataSources[id])
              .filter((source): source is DataSource => source !== undefined);
            
            const resolvedData = combineDataSources(selectedSources, currentSelection.combinationMode);
            
            return {
              dataSources: newDataSources,
              currentDataContext: {
                ...state.currentDataContext,
                resolvedData,
                sources: selectedSources
              }
            };
          }

          return { dataSources: newDataSources };
        }),

      removeDataSource: (sourceId: string) =>
        set((state) => {
          const { [sourceId]: removed, ...dataSources } = state.dataSources;
          return { dataSources };
        }),

      // Data selection management
      createDataSelection: (name: string, sourceIds: string[], mode = 'merge' as const) => {
        const selectionId = generateId();
        const sources = sourceIds.map(id => get().dataSources[id]).filter((source): source is DataSource => source !== undefined);
        
        if (sources.length === 0) {
          throw new Error('No valid data sources provided');
        }

        // Determine result format and fields
        const formats = [...new Set(sources.map(s => s.format))];
        const resultFormat = formats.length === 1 ? formats[0] : 'mixed';
        const resultFields = [...new Set(sources.flatMap(s => s.fields))];

        const selection: DataSelection = {
          id: selectionId,
          name,
          sources: sourceIds,
          combinationMode: mode,
          resultFormat: resultFormat || 'unknown',
          resultFields
        };

        set((state) => ({
          dataSelections: { ...state.dataSelections, [selectionId]: selection }
        }));

        return selectionId;
      },

      updateDataSelection: (selectionId: string, updates: Partial<DataSelection>) =>
        set((state) => {
          const selection = state.dataSelections[selectionId];
          if (!selection) return state;

          return {
            dataSelections: {
              ...state.dataSelections,
              [selectionId]: { ...selection, ...updates }
            }
          };
        }),

      removeDataSelection: (selectionId: string) =>
        set((state) => {
          const { [selectionId]: removed, ...dataSelections } = state.dataSelections;
          return { dataSelections };
        }),

      // Current data context
      setCurrentDataSelection: (selectionId: string) =>
        set((state) => {
          const selection = state.dataSelections[selectionId];
          if (!selection) return state;

          const sources = selection.sources
            .map(id => state.dataSources[id])
            .filter((source): source is DataSource => source !== undefined);
          
          const resolvedData = combineDataSources(sources, selection.combinationMode);

          return {
            currentDataContext: {
              selection,
              resolvedData,
              format: selection.resultFormat,
              fields: selection.resultFields,
              sources
            }
          };
        }),

      setCurrentDataDirect: (data: any, format: string, fields: string[]) =>
        set(() => ({
          currentDataContext: {
            selection: null,
            resolvedData: data,
            format,
            fields,
            sources: []
          }
        })),

      // Getters
      getAvailableDataSources: () => Object.values(get().dataSources),

      getDataSourcesByFormat: (format: string) =>
        Object.values(get().dataSources).filter(source => source.format === format),

      getCurrentDataContext: () => get().currentDataContext,

      isFormatAvailable: (format: string) =>
        Object.values(get().dataSources).some(source => source.format === format)
    }),
    {
      name: 'data-source-store'
    }
  )
); 