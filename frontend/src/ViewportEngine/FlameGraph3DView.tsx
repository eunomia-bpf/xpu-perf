import React, { useState } from 'react';
import { useDataSourceStore } from '@/DataManager/DataStore/dataSourceStore';

interface FlameGraph3DViewProps {
  className?: string;
}

export const FlameGraph3DView: React.FC<FlameGraph3DViewProps> = ({ className }) => {
  const [colorScheme, setColorScheme] = useState('hot-cold');
  
  const { currentDataContext } = useDataSourceStore();

  const data = currentDataContext.resolvedData;
  const isFlameGraphCompatible = currentDataContext.format === 'flamegraph';

  return (
    <div className={`flex flex-col h-full ${className || ''}`}>
      {/* Main 3D Viewport */}
      <div className="flex-1 bg-gray-900 relative overflow-hidden">
        <div className="w-full h-full flex items-center justify-center">
          <div className="text-white text-center">
            <h3 className="text-xl font-semibold mb-3">3D Flame Graph</h3>
            <div className="bg-gray-800 p-6 rounded-lg">
              <p className="mb-2">Data format: {currentDataContext.format}</p>
              <p className="mb-2">Data samples: {Object.keys(data).length}</p>
              {isFlameGraphCompatible ? (
                <>
                  <p className="text-green-400 text-sm mb-4">✓ Compatible flamegraph data detected</p>
                  <p className="text-sm text-gray-400 mb-4">3D visualization will be rendered here</p>
                </>
              ) : (
                <>
                  <p className="text-yellow-400 text-sm mb-4">⚠ Current data format ({currentDataContext.format}) may not be optimal for flame graph visualization</p>
                  <p className="text-sm text-gray-400 mb-4">Showing available data visualization</p>
                </>
              )}
              <div className="text-xs text-gray-500">
                Color scheme: {colorScheme}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Simple Controls Panel */}
      <div className="bg-gray-800 border-t border-gray-700 p-3">
        <h4 className="text-sm font-medium text-white mb-3">Display Options</h4>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-300 mb-1">Color Scheme:</label>
            <select 
              value={colorScheme}
              onChange={(e) => setColorScheme(e.target.value)}
              className="w-full bg-gray-600 text-white rounded px-2 py-1.5 text-sm border border-gray-500 focus:border-blue-400 focus:outline-none"
            >
              <option value="hot-cold">Hot/Cold</option>
              <option value="thread-based">Thread-based</option>
              <option value="function-based">Function-based</option>
            </select>
          </div>
          
          <div className="text-xs text-gray-400">
            <div>Sources: {currentDataContext.sources.length}</div>
            <div>Fields: {currentDataContext.fields.join(', ') || 'none'}</div>
            <div className={isFlameGraphCompatible ? 'text-green-400' : 'text-yellow-400'}>
              {isFlameGraphCompatible ? 'Optimal format' : 'Non-optimal format'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 