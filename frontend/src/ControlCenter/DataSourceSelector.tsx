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
    <div className="bg-gray-700 rounded p-3 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-white">Data Sources</h3>
        <button
          onClick={() => setIsCreatingSelection(true)}
          disabled={allDataSources.length === 0}
          className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-500 text-white rounded transition-colors"
        >
          + Select Data
        </button>
      </div>

      {/* Current Data Context */}
      <div className="bg-gray-800 rounded p-2 text-xs">
        <div className="text-gray-300 mb-1">Current Data:</div>
        {currentDataContext.selection ? (
          <div>
            <div className="text-white font-medium">{currentDataContext.selection.name}</div>
            <div className="text-gray-400">Format: {currentDataContext.format}</div>
            <div className="text-gray-400">{currentDataContext.sources.length} sources</div>
          </div>
        ) : (
          <div className="text-gray-400 italic">No data selected</div>
        )}
      </div>

      {/* Existing Data Selections */}
      <div>
        <label className="block text-xs text-gray-300 mb-2">Available Data Selections</label>
        <div className="space-y-1">
          {availableSelections.map(selection => (
            <div key={selection.id} className="flex items-center justify-between bg-gray-800 rounded px-2 py-1">
              <button
                onClick={() => setCurrentDataSelection(selection.id)}
                className={`flex-1 text-left text-xs ${
                  currentDataContext.selection?.id === selection.id 
                    ? 'text-blue-400 font-medium' 
                    : 'text-white hover:text-gray-300'
                }`}
              >
                {selection.name} ({selection.sources.length} sources)
              </button>
              <button
                onClick={() => removeDataSelection(selection.id)}
                className="text-xs text-red-400 hover:text-red-300 ml-2"
              >
                ×
              </button>
            </div>
          ))}
          {availableSelections.length === 0 && (
            <div className="text-xs text-gray-500 italic">No data selections created</div>
          )}
        </div>
      </div>

      {/* Create New Selection Modal */}
      {isCreatingSelection && (
        <div className="bg-gray-800 rounded p-3 border border-gray-600">
          <h4 className="text-sm font-medium text-white mb-3">Create Data Selection</h4>
          
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-300 mb-1">Selection Name</label>
              <input
                type="text"
                value={newSelectionName}
                onChange={(e) => setNewSelectionName(e.target.value)}
                placeholder="e.g., Combined Flame Data"
                className="w-full bg-gray-600 text-white rounded px-2 py-1 text-xs border border-gray-500 focus:border-blue-400 focus:outline-none"
              />
            </div>

            <div>
              <label className="block text-xs text-gray-300 mb-1">Combination Mode</label>
              <select
                value={combinationMode}
                onChange={(e) => setCombinationMode(e.target.value as any)}
                className="w-full bg-gray-600 text-white rounded px-2 py-1 text-xs border border-gray-500 focus:border-blue-400 focus:outline-none"
              >
                <option value="merge">Merge (combine objects)</option>
                <option value="append">Append (join arrays)</option>
                <option value="override">Override (last wins)</option>
              </select>
            </div>

            <div>
              <label className="block text-xs text-gray-300 mb-2">Select Data Sources</label>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {allDataSources.map(source => (
                  <label key={source.id} className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={selectedSources.includes(source.id)}
                      onChange={() => toggleSourceSelection(source.id)}
                      className="w-3 h-3 text-blue-600 bg-gray-600 border-gray-500 rounded focus:ring-blue-500 focus:ring-1"
                    />
                    <div className="flex-1 text-xs">
                      <div className="text-white">{source.name}</div>
                      <div className="text-gray-400">{source.format} • {source.fields.join(', ')}</div>
                    </div>
                  </label>
                ))}
                {allDataSources.length === 0 && (
                  <div className="text-xs text-gray-500 italic">No data sources available</div>
                )}
              </div>
            </div>

            <div className="flex space-x-2">
              <button
                onClick={handleCreateSelection}
                disabled={!newSelectionName.trim() || selectedSources.length === 0}
                className="text-xs px-3 py-1 bg-green-600 hover:bg-green-500 disabled:bg-gray-500 text-white rounded transition-colors"
              >
                Create
              </button>
              <button
                onClick={() => {
                  setIsCreatingSelection(false);
                  setNewSelectionName('');
                  setSelectedSources([]);
                }}
                className="text-xs px-3 py-1 bg-gray-600 hover:bg-gray-500 text-white rounded transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 