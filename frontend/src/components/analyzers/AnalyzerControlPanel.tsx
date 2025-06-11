import React from 'react';
import { useFlameGraphStore } from '@/DataManager';

export const AnalyzerControlPanel: React.FC = () => {
  const { loadSampleData, isLoading } = useFlameGraphStore();

  return (
    <div className="p-4 space-y-6">
      {/* Analyzer Control - MVP */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-white">Flame Graph Analyzer</h3>
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
      </div>

      {/* View Selector - MVP */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-white">View Type</h3>
        <div className="bg-gray-700 rounded-lg p-4 space-y-3">
          <label className="flex items-center space-x-3 cursor-pointer">
            <input 
              type="radio" 
              name="viewType" 
              value="3d-flame" 
              defaultChecked 
              className="text-blue-500"
            />
            <div>
              <div className="text-white font-medium">● 3D Flame Graph</div>
              <div className="text-sm text-gray-400">Interactive 3D visualization</div>
            </div>
          </label>
          
          <label className="flex items-center space-x-3 cursor-pointer">
            <input 
              type="radio" 
              name="viewType" 
              value="data-table" 
              className="text-blue-500"
            />
            <div>
              <div className="text-gray-300 font-medium">○ Data Table</div>
              <div className="text-sm text-gray-400">Raw data exploration</div>
            </div>
          </label>
        </div>
      </div>

      {/* Dynamic View Controls - MVP */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-white">3D Controls</h3>
        <div className="bg-gray-700 rounded-lg p-4 space-y-4">
          {/* Sliders */}
          <div className="space-y-3">
            <div>
              <label className="block text-sm text-gray-300 mb-1">Z-Spacing: 25</label>
              <input 
                type="range" 
                min="10" 
                max="50" 
                defaultValue="25" 
                className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Min Count: 10</label>
              <input 
                type="range" 
                min="1" 
                max="100" 
                defaultValue="10" 
                className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Max Depth: 8</label>
              <input 
                type="range" 
                min="1" 
                max="20" 
                defaultValue="8" 
                className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>

          {/* Dropdowns */}
          <div className="space-y-2">
            <div>
              <label className="block text-sm text-gray-300 mb-1">Color:</label>
              <select className="w-full bg-gray-600 text-white rounded px-3 py-2">
                <option>Hot/Cold</option>
                <option>Thread-based</option>
                <option>Function-based</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Threads:</label>
              <select className="w-full bg-gray-600 text-white rounded px-3 py-2">
                <option>All</option>
                <option>Main Thread</option>
                <option>Worker Threads</option>
              </select>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-2">
            <button className="flex-1 px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm transition-colors flex items-center justify-center space-x-1">
              <span>🎯</span>
              <span>Reset Camera</span>
            </button>
            <button className="flex-1 px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm transition-colors flex items-center justify-center space-x-1">
              <span>📐</span>
              <span>Fit All</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}; 