import React from 'react';
import { AnalyzerControls, ViewControls } from '@/ControlCenter';

export const FlameGraphAnalyzer: React.FC = () => {
  return (
    <div className="p-4 space-y-6">
      {/* Analyzer Control - MVP */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-white">Flame Graph Analyzer</h3>
        <AnalyzerControls />
      </div>

      {/* View Selector - MVP */}
      <div>
        <h3 className="text-lg font-semibold mb-4 text-white">View Type</h3>
        <ViewControls />
      </div>
    </div>
  );
}; 