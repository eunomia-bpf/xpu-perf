import React from 'react';
import { useFlameGraphStore } from '@/DataManager';

export const AnalyzerControls: React.FC = () => {
  const { loadSampleData, isLoading } = useFlameGraphStore();

  return (
    <div className="bg-gray-700 rounded p-3 space-y-3">
      {/* Status */}
      <div className="flex items-center space-x-2">
        <span className={`w-2 h-2 rounded-full ${isLoading ? 'bg-orange-500' : 'bg-green-500'}`}></span>
        <span className="text-xs text-gray-300">{isLoading ? 'Running' : 'Ready'}</span>
      </div>

      {/* Control Buttons */}
      <div className="space-y-2">
        <button
          onClick={loadSampleData}
          className={`w-full px-3 py-2 rounded text-sm border transition-colors ${
            isLoading 
              ? 'bg-red-600 hover:bg-red-500 text-white border-red-500' 
              : 'bg-green-600 hover:bg-green-500 text-white border-green-500'
          }`}
        >
          {isLoading ? 'Stop' : 'Start'} Profiling
        </button>
        
        <button className="w-full px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm border border-gray-500 transition-colors">
          Configure
        </button>
      </div>

      {/* Configuration */}
      <div className="text-xs text-gray-400 space-y-1">
        <div>Duration: 30s</div>
        <div>Frequency: 99Hz</div>
      </div>
    </div>
  );
}; 