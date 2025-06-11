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
      <div className="flex-1 bg-white relative overflow-hidden border-b border-gray-200">
        <div className="w-full h-full flex items-center justify-center">
          <div className="text-center max-w-md">
            <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <span className="text-3xl text-white">🎯</span>
            </div>
            <h3 className="text-2xl font-bold text-gray-800 mb-4">3D Flame Graph</h3>
            <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 text-left">
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="font-medium text-gray-600">Data format:</span>
                  <span className="text-gray-800 font-mono bg-white px-2 py-1 rounded text-sm border border-gray-200">
                    {currentDataContext.format}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium text-gray-600">Data samples:</span>
                  <span className="text-gray-800 font-semibold">{Object.keys(data).length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium text-gray-600">Color scheme:</span>
                  <span className="text-gray-800 font-mono bg-white px-2 py-1 rounded text-sm border border-gray-200">
                    {colorScheme}
                  </span>
                </div>
              </div>
              
              {isFlameGraphCompatible ? (
                <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-green-800 text-sm flex items-center space-x-2">
                    <span>✅</span>
                    <span>Compatible flamegraph data detected</span>
                  </p>
                  <p className="text-green-700 text-xs mt-1">3D visualization will be rendered here</p>
                </div>
              ) : (
                <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <p className="text-amber-800 text-sm flex items-center space-x-2">
                    <span>⚠️</span>
                    <span>Current data format ({currentDataContext.format}) may not be optimal for flame graph visualization</span>
                  </p>
                  <p className="text-amber-700 text-xs mt-1">Showing available data visualization</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Controls Panel */}
      <div className="bg-white border-t border-gray-200 p-4">
        <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center space-x-2">
          <span>🎨</span>
          <span>Display Options</span>
        </h4>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Color Scheme</label>
            <select 
              value={colorScheme}
              onChange={(e) => setColorScheme(e.target.value)}
              className="profiler-select w-full"
            >
              <option value="hot-cold">Hot/Cold</option>
              <option value="thread-based">Thread-based</option>
              <option value="function-based">Function-based</option>
            </select>
          </div>
          
          <div className="lg:col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-2">Data Info</label>
            <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
              <div className="grid grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Sources:</span>
                  <span className="ml-1 font-semibold text-gray-800">{currentDataContext.sources.length}</span>
                </div>
                <div>
                  <span className="text-gray-600">Fields:</span>
                  <span className="ml-1 font-mono text-xs text-gray-700">
                    {currentDataContext.fields.join(', ') || 'none'}
                  </span>
                </div>
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${
                    isFlameGraphCompatible ? 'bg-green-500' : 'bg-amber-500'
                  }`}></div>
                  <span className={`text-xs font-medium ${
                    isFlameGraphCompatible ? 'text-green-700' : 'text-amber-700'
                  }`}>
                    {isFlameGraphCompatible ? 'Optimal format' : 'Non-optimal format'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 