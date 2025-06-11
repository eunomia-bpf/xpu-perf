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
            <h3 className="text-2xl font-bold mb-4">🔥 3D Flame Graph</h3>
            <p className="text-gray-400 mb-4">Interactive 3D flame stack visualization</p>
            <div className="bg-gray-800 p-6 rounded-lg">
              <p className="text-lg mb-2">Data samples: {Object.keys(data).length}</p>
              <p className="text-sm text-gray-500">3D visualization will be rendered here</p>
            </div>
          </div>
        </div>
        
        {/* Selection Info Overlay */}
        <div className="absolute bottom-4 left-4 bg-gray-800/90 backdrop-blur-md rounded-lg p-3 border border-white/10 max-w-md">
          <span className="text-gray-300 text-sm">
            Selection Info: Click on a function block to see details here
          </span>
        </div>
      </div>

      {/* Integrated 3D Controls Panel - View-Specific Controls */}
      <div className="bg-gray-800 border-t border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-white font-semibold">3D Controls</h4>
          <div className="text-sm text-gray-400">
            Mouse: Rotate • Scroll: Zoom • Click: Select
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Sliders Section */}
          <div className="space-y-3">
            <h5 className="text-sm font-medium text-gray-300">View Parameters</h5>
            
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-sm text-gray-300">Z-Spacing</label>
                <span className="text-sm text-white">{zSpacing}</span>
              </div>
              <input 
                type="range" 
                min="10" 
                max="50" 
                value={zSpacing}
                onChange={(e) => setZSpacing(Number(e.target.value))}
                className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-sm text-gray-300">Min Count</label>
                <span className="text-sm text-white">{minCount}</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="100" 
                value={minCount}
                onChange={(e) => setMinCount(Number(e.target.value))}
                className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-sm text-gray-300">Max Depth</label>
                <span className="text-sm text-white">{maxDepth}</span>
              </div>
              <input 
                type="range" 
                min="1" 
                max="20" 
                value={maxDepth}
                onChange={(e) => setMaxDepth(Number(e.target.value))}
                className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>

          {/* Dropdowns Section */}
          <div className="space-y-3">
            <h5 className="text-sm font-medium text-gray-300">Display Options</h5>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Color Scheme:</label>
              <select 
                value={colorScheme}
                onChange={(e) => setColorScheme(e.target.value)}
                className="w-full bg-gray-600 text-white rounded px-3 py-2 text-sm"
              >
                <option value="hot-cold">Hot/Cold</option>
                <option value="thread-based">Thread-based</option>
                <option value="function-based">Function-based</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Thread Filter:</label>
              <select 
                value={threadFilter}
                onChange={(e) => setThreadFilter(e.target.value)}
                className="w-full bg-gray-600 text-white rounded px-3 py-2 text-sm"
              >
                <option value="all">All Threads</option>
                <option value="main">Main Thread</option>
                <option value="worker">Worker Threads</option>
              </select>
            </div>
          </div>

          {/* Camera Controls Section */}
          <div className="space-y-3">
            <h5 className="text-sm font-medium text-gray-300">Camera Controls</h5>
            
            <div className="flex flex-col space-y-2">
              <button 
                onClick={resetCamera}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm transition-colors flex items-center justify-center space-x-2"
              >
                <span>🎯</span>
                <span>Reset Camera</span>
              </button>
              
              <button 
                onClick={fitAll}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm transition-colors flex items-center justify-center space-x-2"
              >
                <span>📐</span>
                <span>Fit All</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 