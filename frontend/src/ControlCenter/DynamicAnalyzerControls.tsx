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
      case 'idle': return 'status-idle';
      case 'configuring': return 'bg-yellow-500';
      case 'running': return 'status-running';
      case 'completed': return 'status-completed';
      case 'error': return 'status-error';
      default: return 'status-idle';
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
    <div className="profiler-panel p-4 space-y-4">
      {/* Analyzer Selector */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">Analyzer Type</label>
        <select
          value={selectedAnalyzerId || ''}
          onChange={(e) => selectAnalyzer(e.target.value)}
          className="profiler-select w-full"
        >
          {analyzers.map(analyzer => (
            <option key={analyzer.id} value={analyzer.id}>
              {analyzer.icon} {analyzer.displayName}
            </option>
          ))}
        </select>
        {currentAnalyzer && (
          <p className="text-sm text-gray-500 mt-1">{currentAnalyzer.description}</p>
        )}
      </div>

      {/* Instance Management */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-gray-700">Analyzer Instance</label>
          <button
            onClick={() => setIsCreatingInstance(true)}
            className="profiler-button text-xs px-2 py-1"
          >
            + New
          </button>
        </div>

        {isCreatingInstance && (
          <div className="mb-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
            <input
              type="text"
              value={newInstanceName}
              onChange={(e) => setNewInstanceName(e.target.value)}
              placeholder="Instance name..."
              className="profiler-input w-full mb-2"
              onKeyPress={(e) => e.key === 'Enter' && handleCreateInstance()}
            />
            <div className="flex space-x-2">
              <button
                onClick={handleCreateInstance}
                disabled={!newInstanceName.trim()}
                className="profiler-button text-xs px-2 py-1 bg-green-600 hover:bg-green-700"
              >
                Create
              </button>
              <button
                onClick={() => {
                  setIsCreatingInstance(false);
                  setNewInstanceName('');
                }}
                className="text-xs px-2 py-1 bg-gray-200 hover:bg-gray-300 text-gray-700 rounded-md transition-colors"
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
          className="profiler-select w-full"
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
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-2">
              <span className={`w-2 h-2 rounded-full ${getStatusColor(currentInstance.status)}`}></span>
              <span className="text-sm text-gray-600">{getStatusText(currentInstance.status)}</span>
            </div>
            <button
              onClick={() => deleteAnalyzerInstance(currentInstance.id)}
              disabled={currentInstance.status === 'running'}
              className="text-xs px-2 py-1 bg-red-100 hover:bg-red-200 text-red-700 rounded-md transition-colors disabled:opacity-50"
            >
              Delete
            </button>
          </div>

          <div className="space-y-2">
            <button
              onClick={handleStartStop}
              className={`profiler-button w-full ${
                currentInstance.status === 'running'
                  ? 'bg-red-600 hover:bg-red-700'
                  : 'bg-green-600 hover:bg-green-700'
              }`}
            >
              {currentInstance.status === 'running' ? 'Stop' : 'Start'} Analysis
            </button>
          </div>
        </div>
      )}

      {/* Basic Instance Info */}
      {currentInstance && (
        <div className="text-xs text-gray-500 space-y-1 border-t border-gray-200 pt-3">
          <div>Created: {currentInstance.createdAt.toLocaleString()}</div>
          <div>Updated: {currentInstance.updatedAt.toLocaleString()}</div>
          {currentInstance.error && (
            <div className="text-red-600 font-medium">Error: {currentInstance.error}</div>
          )}
        </div>
      )}
    </div>
  );
}; 