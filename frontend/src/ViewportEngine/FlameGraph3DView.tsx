import React, { useState } from 'react';
import { useFlameGraphStore } from '@/DataManager';

interface FlameGraph3DViewProps {
  className?: string;
}

export const FlameGraph3DView: React.FC<FlameGraph3DViewProps> = ({ className }) => {
  const [zSpacing, setZSpacing] = useState(25);
  const [minCount, setMinCount] = useState(10);
  const [maxDepth, setMaxDepth] = useState(8);
  const [colorScheme, setColorScheme] = useState('hot-cold');
  const [threadFilter, setThreadFilter] = useState('all');
  
  const { data } = useFlameGraphStore();

  const resetCamera = () => {
    console.log('Resetting camera to default position');
  };

  const fitAll = () => {
    console.log('Fitting all elements in view');
  };

  return (
    <div className={`flex flex-col h-full ${className || ''}`}>
      {/* Main 3D Viewport */}
      <div className="flex-1 bg-gray-900 relative overflow-hidden">
        <div className="w-full h-full flex items-center justify-center">
          <div className="text-white text-center">
            <h3 className="text-xl font-semibold mb-3">3D Flame Graph</h3>
            <div className="bg-gray-800 p-4 rounded">
              <p className="mb-2">Data samples: {Object.keys(data).length}</p>
              <p className="text-sm text-gray-400">3D visualization will be rendered here</p>
            </div>
          </div>
        </div>
      </div>

      {/* 3D Controls Panel */}
      <div className="bg-gray-800 border-t border-gray-700 p-3">
        <h4 className="text-sm font-medium text-white mb-3">3D Controls</h4>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Parameters */}
          <div className="space-y-2">
            <h5 className="text-xs font-medium text-gray-300">Parameters</h5>
            
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs text-gray-300">Z-Spacing</label>
                <span className="text-xs text-white">{zSpacing}</span>
              </div>
              <input 
                type="range" 
                min="10" 
                max="50" 
                value={zSpacing}
                onChange={(e) => setZSpacing(Number(e.target.value))}
                className="w-full h-1 bg-gray-600 rounded appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs text-gray-300">Min Count</label>
                <span className="text-xs text-white">{minCount}</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="100" 
                value={minCount}
                onChange={(e) => setMinCount(Number(e.target.value))}
                className="w-full h-1 bg-gray-600 rounded appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-xs text-gray-300">Max Depth</label>
                <span className="text-xs text-white">{maxDepth}</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="20" 
                value={maxDepth}
                onChange={(e) => setMaxDepth(Number(e.target.value))}
                className="w-full h-1 bg-gray-600 rounded appearance-none cursor-pointer"
              />
            </div>
          </div>

          {/* Display Options */}
          <div className="space-y-2">
            <h5 className="text-xs font-medium text-gray-300">Display</h5>
            
            <div>
              <label className="block text-xs text-gray-300 mb-1">Color:</label>
              <select 
                value={colorScheme}
                onChange={(e) => setColorScheme(e.target.value)}
                className="w-full bg-gray-600 text-white rounded px-2 py-1 text-xs"
              >
                <option value="hot-cold">Hot/Cold</option>
                <option value="thread-based">Thread-based</option>
                <option value="function-based">Function-based</option>
              </select>
            </div>
            
            <div>
              <label className="block text-xs text-gray-300 mb-1">Thread:</label>
              <select 
                value={threadFilter}
                onChange={(e) => setThreadFilter(e.target.value)}
                className="w-full bg-gray-600 text-white rounded px-2 py-1 text-xs"
              >
                <option value="all">All Threads</option>
                <option value="main">Main Thread</option>
                <option value="worker">Worker Threads</option>
              </select>
            </div>
          </div>

          {/* Camera Controls */}
          <div className="space-y-2">
            <h5 className="text-xs font-medium text-gray-300">Camera</h5>
            
            <div className="flex flex-col space-y-1">
              <button 
                onClick={resetCamera}
                className="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded text-xs border border-gray-600"
              >
                Reset Camera
              </button>
              
              <button 
                onClick={fitAll}
                className="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded text-xs border border-gray-600"
              >
                Fit All
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 