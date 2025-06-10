import React from 'react';
import { useFlameGraphStore } from '@/stores';
import { getColorSchemeNames } from '@/utils/colorSchemes';

interface ControlPanelProps {
  className?: string;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({ className }) => {
  const {
    config,
    resetView,
    toggleAutoRotate,
    changeColorScheme,
    updateZSpacing,
    updateMinCount,
    updateMaxDepth,
    loadSampleData
  } = useFlameGraphStore();

  const colorSchemeNames = getColorSchemeNames();

  return (
    <div className={`absolute top-3 right-3 text-white bg-black/70 backdrop-blur-md p-4 rounded-lg min-w-52 font-sans border border-white/10 z-30 ${className || ''}`}>
      <div className="my-3">
        <button 
          className="bg-gray-700 text-white border-none rounded px-3 py-2 cursor-pointer text-sm font-medium w-full transition-colors hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          onClick={resetView}
        >
          Reset View
        </button>
      </div>
      
      <div className="my-3">
        <button 
          className="bg-gray-700 text-white border-none rounded px-3 py-2 cursor-pointer text-sm font-medium w-full transition-colors hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          onClick={toggleAutoRotate}
        >
          {config.autoRotate ? 'Stop Rotation' : 'Auto Rotate'}
        </button>
      </div>
      
      <div className="my-3">
        <button 
          className="bg-gray-700 text-white border-none rounded px-3 py-2 cursor-pointer text-sm font-medium w-full transition-colors hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          onClick={changeColorScheme}
        >
          Colors: {colorSchemeNames[config.colorSchemeIndex]}
        </button>
      </div>

      <div className="my-3">
        <button 
          className="bg-gray-700 text-white border-none rounded px-3 py-2 cursor-pointer text-sm font-medium w-full transition-colors hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          onClick={loadSampleData}
        >
          Load Sample Data
        </button>
      </div>
      
      <div className="my-3">
        <label className="block text-sm mb-1">
          Z-Spacing: {config.zSpacing}
          <input
            className="w-full mt-1 accent-green-500"
            type="range"
            min="5"
            max="50"
            value={config.zSpacing}
            onChange={(e) => updateZSpacing(Number(e.target.value))}
          />
        </label>
      </div>
      
      <div className="my-3">
        <label className="block text-sm mb-1">
          Min Count: {config.minCount}
          <input
            className="w-full mt-1 accent-green-500"
            type="range"
            min="1"
            max="100"
            value={config.minCount}
            onChange={(e) => updateMinCount(Number(e.target.value))}
          />
        </label>
      </div>
      
      <div className="my-3">
        <label className="block text-sm mb-1">
          Max Depth: {config.maxDepth}
          <input
            className="w-full mt-1 accent-green-500"
            type="range"
            min="3"
            max="15"
            value={config.maxDepth}
            onChange={(e) => updateMaxDepth(Number(e.target.value))}
          />
        </label>
      </div>
    </div>
  );
}; 