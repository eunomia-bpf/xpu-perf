import React, { useState } from 'react';
import { useAnalyzerStore } from '@/DataManager/DataStore/analyzerStore';

export const DynamicAnalyzerControls: React.FC = () => {
  const {
    getAvailableAnalyzers,
    selectedAnalyzerId,
    selectedInstanceId,
    instances,
    getCurrentInstance,
    getCurrentAnalyzer,
    selectAnalyzer,
    selectInstance,
    createAnalyzerInstance,
    startAnalyzer,
    stopAnalyzer,
    deleteAnalyzerInstance
  } = useAnalyzerStore();

  const [isCreatingInstance, setIsCreatingInstance] = useState(false);
  const [newInstanceName, setNewInstanceName] = useState('');

  const analyzers = getAvailableAnalyzers();
  const currentAnalyzer = getCurrentAnalyzer();
  const currentInstance = getCurrentInstance();
  
  // Get instances for current analyzer
  const analyzerInstances = Object.values(instances).filter(
    instance => instance.analyzerId === selectedAnalyzerId
  );

  const handleCreateInstance = () => {
    if (!selectedAnalyzerId || !newInstanceName.trim()) return;
    
    try {
      createAnalyzerInstance(selectedAnalyzerId, newInstanceName.trim());
      setNewInstanceName('');
      setIsCreatingInstance(false);
    } catch (error) {
      console.error('Failed to create instance:', error);
    }
  };

  const handleStartStop = () => {
    if (!currentInstance) return;
    
    if (currentInstance.status === 'running') {
      stopAnalyzer(currentInstance.id);
    } else {
      startAnalyzer(currentInstance.id);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'idle': return 'bg-gray-500';
      case 'configuring': return 'bg-yellow-500';
      case 'running': return 'bg-green-500';
      case 'completed': return 'bg-blue-500';
      case 'error': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'idle': return 'Ready';
      case 'configuring': return 'Configuring';
      case 'running': return 'Running';
      case 'completed': return 'Completed';
      case 'error': return 'Error';
      default: return 'Unknown';
    }
  };

  return (
    <div className="bg-gray-700 rounded p-3 space-y-4">
      {/* Analyzer Selector */}
      <div>
        <label className="block text-xs text-gray-300 mb-2">Analyzer Type</label>
        <select
          value={selectedAnalyzerId || ''}
          onChange={(e) => selectAnalyzer(e.target.value)}
          className="w-full bg-gray-600 text-white rounded px-2 py-1.5 text-sm border border-gray-500 focus:border-blue-400 focus:outline-none"
        >
          {analyzers.map(analyzer => (
            <option key={analyzer.id} value={analyzer.id}>
              {analyzer.icon} {analyzer.displayName}
            </option>
          ))}
        </select>
        {currentAnalyzer && (
          <p className="text-xs text-gray-400 mt-1">{currentAnalyzer.description}</p>
        )}
      </div>

      {/* Instance Management */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-xs text-gray-300">Analyzer Instance</label>
          <button
            onClick={() => setIsCreatingInstance(true)}
            className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-500 text-white rounded transition-colors"
          >
            + New
          </button>
        </div>

        {isCreatingInstance && (
          <div className="mb-2 p-2 bg-gray-800 rounded border">
            <input
              type="text"
              value={newInstanceName}
              onChange={(e) => setNewInstanceName(e.target.value)}
              placeholder="Instance name..."
              className="w-full bg-gray-600 text-white rounded px-2 py-1 text-xs mb-2 border border-gray-500 focus:border-blue-400 focus:outline-none"
              onKeyPress={(e) => e.key === 'Enter' && handleCreateInstance()}
            />
            <div className="flex space-x-1">
              <button
                onClick={handleCreateInstance}
                disabled={!newInstanceName.trim()}
                className="text-xs px-2 py-1 bg-green-600 hover:bg-green-500 disabled:bg-gray-500 text-white rounded transition-colors"
              >
                Create
              </button>
              <button
                onClick={() => {
                  setIsCreatingInstance(false);
                  setNewInstanceName('');
                }}
                className="text-xs px-2 py-1 bg-gray-600 hover:bg-gray-500 text-white rounded transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        <select
          value={selectedInstanceId || ''}
          onChange={(e) => selectInstance(e.target.value)}
          disabled={analyzerInstances.length === 0}
          className="w-full bg-gray-600 text-white rounded px-2 py-1.5 text-sm border border-gray-500 focus:border-blue-400 focus:outline-none disabled:opacity-50"
        >
          <option value="">Select instance...</option>
          {analyzerInstances.map(instance => (
            <option key={instance.id} value={instance.id}>
              {instance.name} ({getStatusText(instance.status)})
            </option>
          ))}
        </select>
      </div>

      {/* Instance Status & Controls */}
      {currentInstance && (
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${getStatusColor(currentInstance.status)}`}></span>
              <span className="text-xs text-gray-300">{getStatusText(currentInstance.status)}</span>
            </div>
            <button
              onClick={() => deleteAnalyzerInstance(currentInstance.id)}
              disabled={currentInstance.status === 'running'}
              className="text-xs px-2 py-1 bg-red-600 hover:bg-red-500 disabled:bg-gray-500 text-white rounded transition-colors"
            >
              Delete
            </button>
          </div>

          <div className="space-y-2">
            <button
              onClick={handleStartStop}
              className={`w-full px-3 py-2 rounded text-sm border transition-colors ${
                currentInstance.status === 'running'
                  ? 'bg-red-600 hover:bg-red-500 text-white border-red-500'
                  : 'bg-green-600 hover:bg-green-500 text-white border-green-500'
              }`}
            >
              {currentInstance.status === 'running' ? 'Stop' : 'Start'} Analysis
            </button>
          </div>
        </div>
      )}

      {/* Basic Instance Info */}
      {currentInstance && (
        <div className="text-xs text-gray-400 space-y-1 border-t border-gray-600 pt-2">
          <div>Created: {currentInstance.createdAt.toLocaleString()}</div>
          <div>Updated: {currentInstance.updatedAt.toLocaleString()}</div>
          {currentInstance.error && (
            <div className="text-red-400">Error: {currentInstance.error}</div>
          )}
        </div>
      )}
    </div>
  );
}; 