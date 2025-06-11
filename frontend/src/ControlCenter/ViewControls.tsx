import React from 'react';
import { useViewContext } from '@/LayoutManager/ViewContext';

export const ViewControls: React.FC = () => {
  const { currentView, setCurrentView } = useViewContext();

  return (
    <div className="bg-gray-700 rounded p-3 space-y-2">
      <label className="flex items-center space-x-2 cursor-pointer">
        <input 
          type="radio" 
          name="viewType" 
          value="3d-flame" 
          checked={currentView === '3d-flame'}
          onChange={() => setCurrentView('3d-flame')}
        />
        <span className="text-sm text-white">3D Flame Graph</span>
      </label>
      
      <label className="flex items-center space-x-2 cursor-pointer">
        <input 
          type="radio" 
          name="viewType" 
          value="data-table" 
          checked={currentView === 'data-table'}
          onChange={() => setCurrentView('data-table')}
        />
        <span className="text-sm text-white">Data Table</span>
      </label>
    </div>
  );
}; 