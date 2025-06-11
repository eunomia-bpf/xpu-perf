import React from 'react';

interface FlameGraph2DViewProps {
  className?: string;
}

export const FlameGraph2DView: React.FC<FlameGraph2DViewProps> = ({ className }) => {
  return (
    <div className={`relative w-full h-full bg-gray-900 ${className || ''}`}>
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-white text-center">
          <div className="text-4xl mb-4">📊</div>
          <h2 className="text-2xl font-bold mb-2">2D Flame Graph</h2>
          <p className="text-gray-400">Traditional horizontal flame graph view</p>
          <p className="text-sm text-gray-500 mt-4">Mock implementation</p>
        </div>
      </div>
      
      {/* Simple Controls Panel */}
      <div className="absolute bottom-4 right-4">
        <div className="bg-black/80 backdrop-blur-md text-white p-4 rounded-lg border border-white/10">
          <h4 className="text-lg font-semibold mb-3">2D Controls</h4>
          <div className="space-y-2">
            <button className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm">Zoom In</button>
            <button className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm">Zoom Out</button>
          </div>
        </div>
      </div>
    </div>
  );
}; 