import React from 'react';
import { useFlameGraphStore } from '@/DataManager';

export const AnalyzerControls: React.FC = () => {
  const { loadSampleData, isLoading } = useFlameGraphStore();

  return (
    <div className="bg-gray-700 rounded-lg p-4 space-y-4">
      {/* Status */}
      <div className="flex items-center space-x-2">
        <span className={`w-3 h-3 rounded-full ${isLoading ? 'bg-orange-500' : 'bg-green-500'}`}></span>
        <span className="text-sm text-gray-300">Status: {isLoading ? 'Running' : 'Ready'}</span>
      </div>

      {/* Control Buttons */}
      <div className="space-y-2">
        <button
          onClick={loadSampleData}
          className={`w-full px-4 py-2 rounded font-medium transition-colors flex items-center justify-center space-x-2 ${
            isLoading 
              ? 'bg-red-600 hover:bg-red-700 text-white' 
              : 'bg-green-600 hover:bg-green-700 text-white'
          }`}
        >
          <span>{isLoading ? '⏹️' : '▶️'}</span>
          <span>{isLoading ? 'Stop Profiling' : 'Start Profiling'}</span>
        </button>
        
        <button className="w-full px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded font-medium transition-colors flex items-center justify-center space-x-2">
          <span>⚙️</span>
          <span>Configure</span>
        </button>
      </div>

      {/* Basic Configuration */}
      <div className="text-sm text-gray-400 space-y-1">
        <div>Duration: 30s <input type="checkbox" className="ml-2" /> Continuous</div>
        <div>Frequency: 99Hz</div>
      </div>
    </div>
  );
}; 