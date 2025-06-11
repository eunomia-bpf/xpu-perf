import React from 'react';
import { useViewContext } from './ViewContext';

interface AppShellProps {
  children: React.ReactNode;
  sidebar: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children, sidebar }) => {
  const { currentView } = useViewContext();
  
  const getViewDisplayName = (view: string) => {
    switch (view) {
      case '3d-flame':
        return '3D Flame Graph';
      case 'data-table':
        return 'Data Table';
      default:
        return '3D Flame Graph';
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <h1 className="text-lg font-semibold">Zero-Instrument Profiler</h1>
          </div>
          <div className="flex items-center space-x-2">
            <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded text-sm border border-gray-600">
              Export
            </button>
            <button className="px-3 py-1 bg-gray-800 hover:bg-gray-700 text-gray-200 rounded text-sm border border-gray-600">
              Settings
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-80 bg-gray-800 border-r border-gray-700 overflow-y-auto">
          {sidebar}
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {children}
        </main>
      </div>

      {/* Status Bar */}
      <footer className="bg-gray-800 border-t border-gray-700 px-4 py-2">
        <div className="flex items-center space-x-4 text-sm text-gray-300">
          <span>Status: Ready</span>
          <span>Data: 2 samples</span>
          <span>View: {getViewDisplayName(currentView)}</span>
        </div>
      </footer>
    </div>
  );
}; 