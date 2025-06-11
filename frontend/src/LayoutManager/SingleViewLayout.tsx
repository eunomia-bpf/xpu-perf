import React from 'react';
import { FlameGraph3DView } from '@/ViewportEngine/FlameGraph3DView';
import { DataTableView } from '@/ViewportEngine/DataTableView';
import { useViewContext } from './ViewContext';

export const SingleViewLayout: React.FC = () => {
  const { currentView, setCurrentView } = useViewContext();

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
      {/* View Selector Header */}
      <div className="bg-gray-800 border-b border-gray-700 p-3">
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
        </div>
      </div>

      {/* Main Viewport - Full Height */}
      <div className="flex-1 overflow-hidden">
        {renderCurrentView()}
      </div>
    </div>
  );
}; 