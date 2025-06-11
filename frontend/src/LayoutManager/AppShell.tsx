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
        return 'Data View';
      default:
        return '3D Flame Graph';
    }
  };

  return (
    <div className="profiler-layout">
      {/* Header */}
      <header className="profiler-header px-4 py-3 lg:px-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-white bg-opacity-20 rounded-lg flex items-center justify-center">
              <span className="text-lg">🔥</span>
            </div>
            <h1 className="text-lg lg:text-xl font-semibold">Zero-Instrument Profiler</h1>
          </div>
          <div className="flex items-center space-x-2">
            <button className="px-3 py-1.5 bg-white bg-opacity-20 hover:bg-opacity-30 text-white rounded text-sm transition-all duration-200">
              Export
            </button>
            <button className="px-3 py-1.5 bg-white bg-opacity-20 hover:bg-opacity-30 text-white rounded text-sm transition-all duration-200">
              Settings
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="profiler-layout-body">
        {/* Sidebar */}
        <aside className="w-80 lg:w-96 profiler-sidebar">
          {sidebar}
        </aside>

        {/* Main Content */}
        <main className="profiler-main">
          {children}
        </main>
      </div>

      {/* Status Bar */}
      <footer className="bg-white border-t border-gray-200 px-4 py-2 lg:px-6 flex-shrink-0">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span>Ready</span>
            </div>
            <span>•</span>
            <span>Data: 2 samples</span>
            <span>•</span>
            <span className="hidden sm:inline">View: {getViewDisplayName(currentView)}</span>
          </div>
          <div className="text-xs text-gray-500">
            v1.0.0
          </div>
        </div>
      </footer>
    </div>
  );
}; 