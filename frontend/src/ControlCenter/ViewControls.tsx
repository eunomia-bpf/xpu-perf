import React from 'react';
import { useViewContext } from '@/LayoutManager/ViewContext';

export const ViewControls: React.FC = () => {
  const { currentView, setCurrentView } = useViewContext();

  return (
    <div className="bg-gray-700 rounded-lg p-4 space-y-3">
      <label 
        className="flex items-center space-x-3 cursor-pointer"
        onClick={() => setCurrentView('3d-flame')}
      >
        <input 
          type="radio" 
          name="viewType" 
          value="3d-flame" 
          checked={currentView === '3d-flame'}
          onChange={() => setCurrentView('3d-flame')}
          className="text-blue-500"
        />
        <div>
          <div className="text-white font-medium">● 3D Flame Graph</div>
          <div className="text-sm text-gray-400">Interactive 3D visualization</div>
        </div>
      </label>
      
      <label 
        className="flex items-center space-x-3 cursor-pointer"
        onClick={() => setCurrentView('data-table')}
      >
        <input 
          type="radio" 
          name="viewType" 
          value="data-table" 
          checked={currentView === 'data-table'}
          onChange={() => setCurrentView('data-table')}
          className="text-blue-500"
        />
        <div>
          <div className="text-white font-medium">○ Data Table</div>
          <div className="text-sm text-gray-400">Raw data exploration</div>
        </div>
      </label>
    </div>
  );
}; 