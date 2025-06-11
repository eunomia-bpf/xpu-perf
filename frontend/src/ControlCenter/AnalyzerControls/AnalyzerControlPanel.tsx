import React from 'react';
import { useFlameGraphStore } from '@/DataManager/DataStore';

interface AnalyzerControlPanelProps {
  className?: string;
}

export const AnalyzerControlPanel: React.FC<AnalyzerControlPanelProps> = ({ className }) => {
  const { loadSampleData, isLoading } = useFlameGraphStore();

  return (
    <div className={`p-4 text-white ${className || ''}`}>
      {/* Analyzer Section */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-4">Analyzer</h3>
        <div className="space-y-3">
          <button 
            className={`w-full px-4 py-2 rounded font-medium transition-colors ${
              isLoading 
                ? 'bg-red-600 hover:bg-red-700' 
                : 'bg-green-600 hover:bg-green-700'
            }`}
            onClick={loadSampleData}
            disabled={isLoading}
          >
            {isLoading ? 'Stop' : 'Start'}
          </button>
          
          <div className="text-sm text-gray-400">
            Status: {isLoading ? 'Running' : 'Ready'}
          </div>
          
          <details className="text-sm">
            <summary className="cursor-pointer text-gray-300 hover:text-white">
              config...
            </summary>
            <div className="mt-2 pl-4 space-y-2 text-gray-400">
              <div>Duration: 30s</div>
              <div>Frequency: 99Hz</div>
              <div>Target: Current process</div>
            </div>
          </details>
        </div>
      </div>

      {/* View Selector Section */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-4">View Selector</h3>
        <div className="space-y-2">
          <label className="flex items-center space-x-2 cursor-pointer">
            <input type="radio" name="viewType" value="3d-flame" defaultChecked className="text-blue-600" />
            <span>3D Flame</span>
          </label>
          <label className="flex items-center space-x-2 cursor-pointer">
            <input type="radio" name="viewType" value="2d-flame" className="text-blue-600" />
            <span>2D Flame</span>
          </label>
          <label className="flex items-center space-x-2 cursor-pointer">
            <input type="radio" name="viewType" value="data-table" className="text-blue-600" />
            <span>Data Table</span>
          </label>
          <label className="flex items-center space-x-2 cursor-pointer">
            <input type="radio" name="viewType" value="line-chart" className="text-blue-600" />
            <span>Line Chart</span>
          </label>
        </div>
      </div>
    </div>
  );
}; 