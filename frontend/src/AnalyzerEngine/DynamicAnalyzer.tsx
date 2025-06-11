import React from 'react';
import { DynamicAnalyzerControls } from '@/ControlCenter/DynamicAnalyzerControls';

export const DynamicAnalyzer: React.FC = () => {
  return (
    <div className="p-4 space-y-4">
      {/* Dynamic Analyzer Control */}
      <div>
        <h3 className="text-sm font-medium mb-2 text-white">Analyzer</h3>
        <DynamicAnalyzerControls />
      </div>
    </div>
  );
}; 