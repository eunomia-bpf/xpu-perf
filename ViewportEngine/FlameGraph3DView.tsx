import React from 'react';
import { FlameGraphContent } from './FlameGraph3D';

interface FlameGraph3DViewProps {
  className?: string;
}

export const FlameGraph3DView: React.FC<FlameGraph3DViewProps> = ({ className }) => {
  return (
    <div className={`relative w-full h-full ${className || ''}`}>
      {/* Mock 3D Content */}
      <FlameGraphContent />
      
      {/* Simple Controls Panel */}
      <div className="absolute bottom-4 right-4">
        <div className="bg-black/80 backdrop-blur-md text-white p-4 rounded-lg border border-white/10">
          <h4 className="text-lg font-semibold mb-3">3D Controls</h4>
          <div className="space-y-2">
            <button className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm">Reset View</button>
            <button className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm">Auto Rotate</button>
          </div>
        </div>
      </div>
    </div>
  );
}; 