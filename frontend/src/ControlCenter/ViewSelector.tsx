import React from 'react';
import { DynamicViewControls } from './DynamicViewControls';

export const ViewSelector: React.FC = () => {
  return (
    <div className="p-4">
      <h3 className="text-sm font-medium mb-2 text-white">View Type</h3>
      <DynamicViewControls />
    </div>
  );
}; 