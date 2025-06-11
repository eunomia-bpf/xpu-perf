import React from 'react';
import { DynamicViewControls } from './DynamicViewControls';

export const ViewSelector: React.FC = () => {
  return (
    <div className="space-y-4">
      <div className="text-center pb-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">View Selection</h3>
        <p className="text-sm text-gray-600">Choose how to visualize your profiling data</p>
      </div>
      <DynamicViewControls />
    </div>
  );
}; 