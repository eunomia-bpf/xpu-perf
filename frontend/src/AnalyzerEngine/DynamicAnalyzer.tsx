import React from 'react';
import { DynamicAnalyzerControls } from '@/ControlCenter/DynamicAnalyzerControls';

export const DynamicAnalyzer: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="text-center pb-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Analyzer Configuration</h3>
        <p className="text-sm text-gray-600">Configure and manage your profiling analyzers</p>
      </div>
      <DynamicAnalyzerControls />
    </div>
  );
}; 