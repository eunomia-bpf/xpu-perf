import React, { useState } from 'react';
import { useDataSourceStore } from '@/DataManager/DataStore/dataSourceStore';
import { useAnalyzerStore } from '@/DataManager/DataStore/analyzerStore';

export const DataSourceSelector: React.FC = () => {
  const {
    dataSelections,
    currentDataContext,
    createDataSelection,
    setCurrentDataSelection,
    getAvailableDataSources,
    removeDataSelection
  } = useDataSourceStore();

  const { instances } = useAnalyzerStore();

  const [isCreatingSelection, setIsCreatingSelection] = useState(false);
  const [newSelectionName, setNewSelectionName] = useState('');
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [combinationMode, setCombinationMode] = useState<'merge' | 'append' | 'override'>('merge');

  const availableDataSources = getAvailableDataSources();
  const availableSelections = Object.values(dataSelections);

  // Create data sources from analyzer instances
  const analyzerDataSources = Object.values(instances)
    .filter(instance => instance.status === 'completed' || instance.data)
    .map(instance => {
      const analyzer = useAnalyzerStore.getState().registry.analyzers[instance.analyzerId];
      return {
        id: `analyzer-${instance.id}`,
        name: `${instance.name} (${analyzer?.displayName || instance.analyzerId})`,
        type: 'analyzer-instance' as const,
        format: analyzer?.outputFormat || 'unknown',
        fields: analyzer?.outputFields || [],
        data: instance.data || {},
        lastUpdated: instance.updatedAt,
        metadata: { instanceId: instance.id, analyzerId: instance.analyzerId }
      };
    });

  const allDataSources = [...availableDataSources, ...analyzerDataSources];

  const handleCreateSelection = () => {
    if (!newSelectionName.trim() || selectedSources.length === 0) return;
    
    try {
      const selectionId = createDataSelection(newSelectionName.trim(), selectedSources, combinationMode);
      setCurrentDataSelection(selectionId);
      setNewSelectionName('');
      setSelectedSources([]);
      setIsCreatingSelection(false);
    } catch (error) {
      console.error('Failed to create data selection:', error);
    }
  };

  const toggleSourceSelection = (sourceId: string) => {
    setSelectedSources(prev => 
      prev.includes(sourceId) 
        ? prev.filter(id => id !== sourceId)
        : [...prev, sourceId]
    );
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="text-center pb-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Data Sources</h3>
        <p className="text-sm text-gray-600">Select and combine data from multiple sources</p>
      </div>

      {/* Controls */}
      <div className="profiler-panel p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-medium text-gray-800">Manage Data</h4>
          <button
            onClick={() => setIsCreatingSelection(true)}
            disabled={allDataSources.length === 0}
            className="profiler-button text-xs px-2 py-1"
          >
            + Select Data
          </button>
        </div>

        {/* Current Data */}
        <div className="bg-gray-50 rounded p-3 text-sm border border-gray-200">
          <div className="text-gray-600 text-xs mb-1">Current Data:</div>
          {currentDataContext.selection ? (
            <div>
              <div className="font-medium text-gray-800">{currentDataContext.selection.name}</div>
              <div className="text-xs text-gray-600">{currentDataContext.format} • {currentDataContext.sources.length} sources</div>
            </div>
          ) : (
            <div className="text-gray-500 italic">No data selected</div>
          )}
        </div>

        {/* Data Selections */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Available Selections</label>
          <div className="space-y-1">
            {availableSelections.map(selection => (
              <div key={selection.id} className="flex items-center justify-between bg-gray-50 rounded px-3 py-2 border border-gray-200">
                <button
                  onClick={() => setCurrentDataSelection(selection.id)}
                  className={`flex-1 text-left text-sm transition-colors ${
                    currentDataContext.selection?.id === selection.id 
                      ? 'text-blue-600 font-medium' 
                      : 'text-gray-700 hover:text-gray-900'
                  }`}
                >
                  {selection.name} ({selection.sources.length})
                </button>
                <button
                  onClick={() => removeDataSelection(selection.id)}
                  className="text-sm text-red-600 hover:text-red-800 ml-2"
                >
                  ×
                </button>
              </div>
            ))}
            {availableSelections.length === 0 && (
              <div className="text-sm text-gray-500 italic py-2">No selections created</div>
            )}
          </div>
        </div>

        {/* Create Selection Modal */}
        {isCreatingSelection && (
          <div className="bg-white rounded p-4 border border-gray-300 shadow-sm">
            <h4 className="text-sm font-medium text-gray-800 mb-3">Create Data Selection</h4>
            
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Name</label>
                <input
                  type="text"
                  value={newSelectionName}
                  onChange={(e) => setNewSelectionName(e.target.value)}
                  placeholder="Selection name"
                  className="profiler-input w-full text-sm"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">Mode</label>
                <select
                  value={combinationMode}
                  onChange={(e) => setCombinationMode(e.target.value as any)}
                  className="profiler-select w-full text-sm"
                >
                  <option value="merge">Merge</option>
                  <option value="append">Append</option>
                  <option value="override">Override</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-700 mb-2">Sources</label>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {allDataSources.map(source => (
                    <label key={source.id} className="flex items-start space-x-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={selectedSources.includes(source.id)}
                        onChange={() => toggleSourceSelection(source.id)}
                        className="mt-0.5 w-4 h-4 text-blue-600 bg-white border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                      />
                      <div className="flex-1 text-sm">
                        <div className="text-gray-800">{source.name}</div>
                        <div className="text-gray-500 text-xs">{source.format}</div>
                      </div>
                    </label>
                  ))}
                  {allDataSources.length === 0 && (
                    <div className="text-sm text-gray-500 italic">No sources available</div>
                  )}
                </div>
              </div>

              <div className="flex space-x-2 pt-2">
                <button
                  onClick={handleCreateSelection}
                  disabled={!newSelectionName.trim() || selectedSources.length === 0}
                  className="profiler-button text-sm px-3 py-1 bg-green-600 hover:bg-green-700"
                >
                  Create
                </button>
                <button
                  onClick={() => {
                    setIsCreatingSelection(false);
                    setNewSelectionName('');
                    setSelectedSources([]);
                  }}
                  className="text-sm px-3 py-1 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 