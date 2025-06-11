import React from 'react';

export const ViewControls: React.FC = () => {
  return (
    <div className="bg-gray-700 rounded-lg p-4 space-y-3">
      <label className="flex items-center space-x-3 cursor-pointer">
        <input 
          type="radio" 
          name="viewType" 
          value="3d-flame" 
          defaultChecked 
          className="text-blue-500"
        />
        <div>
          <div className="text-white font-medium">● 3D Flame Graph</div>
          <div className="text-sm text-gray-400">Interactive 3D visualization</div>
        </div>
      </label>
      
      <label className="flex items-center space-x-3 cursor-pointer">
        <input 
          type="radio" 
          name="viewType" 
          value="data-table" 
          className="text-blue-500"
        />
        <div>
          <div className="text-gray-300 font-medium">○ Data Table</div>
          <div className="text-sm text-gray-400">Raw data exploration</div>
        </div>
      </label>
    </div>
  );
}; 