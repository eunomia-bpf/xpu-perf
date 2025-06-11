import React from 'react';
import { DynamicAnalyzerControls } from '@/ControlCenter/DynamicAnalyzerControls';

export const DynamicAnalyzer: React.FC = () => {
  return (
    <div className="space-y-4">
      {/* Dynamic Analyzer Control */}
      <div>
        <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center space-x-2">
          <span>🔬</span>
          <span>Analyzer</span>
        </h3>
        <DynamicAnalyzerControls />
      </div>
    </div>
  );
}; 