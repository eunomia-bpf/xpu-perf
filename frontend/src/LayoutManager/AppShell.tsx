import React from 'react';

interface AppShellProps {
  children: React.ReactNode;
  sidebar: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children, sidebar }) => {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      {/* Simplified Header - MVP Design */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">🔥</span>
            <h1 className="text-xl font-bold">Zero-Instrument Profiler</h1>
          </div>
          <div className="flex items-center space-x-4">
            <button 
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm flex items-center space-x-1 transition-colors"
              title="Export Data"
            >
              <span>📤</span>
              <span>Export</span>
            </button>
            <button 
              className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm flex items-center space-x-1 transition-colors"
              title="Settings"
            >
              <span>⚙️</span>
              <span>Settings</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Area - MVP Layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Control Panel */}
        <aside className="w-80 bg-gray-800 border-r border-gray-700 overflow-y-auto">
          {sidebar}
        </aside>

        {/* Main Viewport */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {children}
        </main>
      </div>

      {/* Status Bar - MVP Design */}
      <footer className="bg-gray-800 border-t border-gray-700 px-4 py-2">
        <div className="flex items-center space-x-6 text-sm text-gray-400">
          <span className="flex items-center space-x-1">
            <span className="w-2 h-2 bg-green-500 rounded-full"></span>
            <span>Status: Ready</span>
          </span>
          <span>Samples: 0</span>
          <span>Duration: 00:00</span>
          <span>View: 3D Flame Graph</span>
        </div>
      </footer>
    </div>
  );
}; 