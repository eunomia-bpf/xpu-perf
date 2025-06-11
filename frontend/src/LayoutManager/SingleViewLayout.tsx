import React, { useState } from 'react';
import { FlameGraph3DView } from '@/ViewportEngine/FlameGraph3DView';
import { DataTableView } from '@/ViewportEngine/DataTableView';

type ViewType = '3d-flame' | 'data-table';

export const SingleViewLayout: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewType>('3d-flame');

  const renderCurrentView = () => {
    switch (currentView) {
      case '3d-flame':
        return <FlameGraph3DView className="w-full h-full" />;
      case 'data-table':
        return <DataTableView className="w-full h-full" />;
      default:
        return <FlameGraph3DView className="w-full h-full" />;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Main Viewport - Single View Display */}
      <div className="flex-1 overflow-hidden">
        {renderCurrentView()}
      </div>

      {/* View-Specific Controls Panel */}
      <div className="bg-gray-800 border-t border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <span className="text-sm text-gray-400">Current View:</span>
            <div className="flex space-x-2">
              <button
                onClick={() => setCurrentView('3d-flame')}
                className={`px-3 py-1.5 rounded text-sm transition-colors ${
                  currentView === '3d-flame'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                🔥 3D Flame Graph
              </button>
              <button
                onClick={() => setCurrentView('data-table')}
                className={`px-3 py-1.5 rounded text-sm transition-colors ${
                  currentView === 'data-table'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                📊 Data Table
              </button>
            </div>
          </div>

          {/* View-Specific Control Hints */}
          <div className="text-sm text-gray-400">
            {currentView === '3d-flame' && (
              <span>Mouse: Rotate • Scroll: Zoom • Click: Select</span>
            )}
            {currentView === 'data-table' && (
              <span>Click headers to sort • Use search to filter</span>
            )}
          </div>
        </div>

        {/* Selection Info Panel */}
        <div className="mt-3 p-3 bg-gray-700 rounded text-sm">
          <span className="text-gray-300">
            Selection Info: Click on a function block to see details here
          </span>
        </div>
      </div>
    </div>
  );
}; 