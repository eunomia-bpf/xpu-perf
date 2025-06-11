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
      <div className="flex-1 bg-white relative overflow-hidden border-b border-gray-200">
        <div className="w-full h-full flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-blue-600 rounded-xl flex items-center justify-center mx-auto mb-4">
              <span className="text-2xl text-white">🎯</span>
            </div>
            <h3 className="text-xl font-semibold text-gray-800 mb-4">3D Flame Graph</h3>
            
            <div className="bg-gray-50 p-4 rounded-lg border border-gray-200 max-w-sm mx-auto">
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <span className="text-gray-500">Format:</span>
                  <div className="font-mono text-xs bg-white px-2 py-1 rounded border mt-1">
                    {currentDataContext.format}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Samples:</span>
                  <div className="font-semibold text-gray-800 mt-1">{Object.keys(data).length}</div>
                </div>
              </div>
              
              {isFlameGraphCompatible ? (
                <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-xs text-green-700">
                  ✅ Compatible data format
                </div>
              ) : (
                <div className="mt-3 p-2 bg-amber-50 border border-amber-200 rounded text-xs text-amber-700">
                  ⚠️ Non-optimal format for flame graph
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white border-t border-gray-200 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium text-gray-700">Color Scheme:</span>
            <select 
              value={colorScheme}
              onChange={(e) => setColorScheme(e.target.value)}
              className="profiler-select text-sm"
            >
              <option value="hot-cold">Hot/Cold</option>
              <option value="thread-based">Thread</option>
              <option value="function-based">Function</option>
            </select>
          </div>
          <div className="flex items-center space-x-3 text-xs text-gray-500">
            <span>Sources: {currentDataContext.sources.length}</span>
            <div className={`w-2 h-2 rounded-full ${isFlameGraphCompatible ? 'bg-green-400' : 'bg-amber-400'}`}></div>
          </div>
        </div>
      </div>
    </div>
  );
}; 