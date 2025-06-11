import React from 'react';
import { AnalyzerControls, ViewControls } from '@/ControlCenter';

export const FlameGraphAnalyzer: React.FC = () => {
  return (
    <div className="p-4 space-y-4">
      {/* Analyzer Control */}
      <div>
        <h3 className="text-sm font-medium mb-2 text-white">Analyzer</h3>
        <AnalyzerControls />
      </div>

      {/* View Selector */}
      <div>
        <h3 className="text-sm font-medium mb-2 text-white">View Type</h3>
        <ViewControls />
      </div>
    </div>
  );
}; 