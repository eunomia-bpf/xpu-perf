import React from 'react';
import { DynamicViewControls } from './DynamicViewControls';

export const ViewSelector: React.FC = () => {
  return (
    <div>
      <h3 className="text-sm font-semibold text-gray-800 mb-3 flex items-center space-x-2">
        <span>🎯</span>
        <span>View Type</span>
      </h3>
      <DynamicViewControls />
    </div>
  );
}; 