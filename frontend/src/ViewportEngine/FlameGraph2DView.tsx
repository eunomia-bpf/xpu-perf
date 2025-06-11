import React, { useState } from 'react';
import { useFlameGraphStore } from '@/stores';

interface FlameGraph2DViewProps {
  className?: string;
}

export const FlameGraph2DView: React.FC<FlameGraph2DViewProps> = ({ className }) => {
  const [showControls, setShowControls] = useState(false);
  const { data } = useFlameGraphStore();

  return (
    <div className={`relative w-full h-full bg-gray-900 ${className || ''}`}>
      {/* 2D Flame Graph Canvas */}
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-white text-center">
          <h3 className="text-2xl font-bold mb-4">2D Flame Graph</h3>
          <p className="text-gray-400 mb-4">Traditional horizontal flame graph visualization</p>
          <div className="bg-gray-800 p-4 rounded">
            Data samples: {Object.keys(data).length}
          </div>
        </div>
      </div>

      {/* View-Specific Controls Panel */}
      <div className="absolute bottom-4 right-4 flex flex-col space-y-2">
        {/* Controls Toggle Button */}
        <button
          className="bg-gray-800/90 backdrop-blur-md text-white p-2 rounded-lg border border-white/10 hover:bg-gray-700/90 transition-colors"
          onClick={() => setShowControls(!showControls)}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
          </svg>
        </button>

        {/* Expandable Controls Panel */}
        {showControls && (
          <div className="bg-black/80 backdrop-blur-md text-white p-4 rounded-lg min-w-64 border border-white/10 space-y-3">
            <h4 className="text-lg font-semibold mb-3">2D Controls</h4>
            
            {/* 2D-specific controls */}
            <div className="space-y-3">
              <div>
                <label className="block text-sm mb-1">
                  Scale: 100%
                  <input
                    className="w-full mt-1 accent-blue-500"
                    type="range"
                    min="50"
                    max="200"
                    defaultValue="100"
                  />
                </label>
              </div>
              
              <div>
                <label className="block text-sm mb-1">
                  Min Width: 1px
                  <input
                    className="w-full mt-1 accent-blue-500"
                    type="range"
                    min="1"
                    max="10"
                    defaultValue="1"
                  />
                </label>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <button className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm transition-colors">
                Zoom In
              </button>
              <button className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm transition-colors">
                Zoom Out
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 