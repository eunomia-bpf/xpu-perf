import React from 'react';

interface NavigationHeaderProps {
  className?: string;
}

export const NavigationHeader: React.FC<NavigationHeaderProps> = ({ className }) => {
  return (
    <header className={`bg-gray-800 border-b border-gray-700 px-6 py-4 flex items-center justify-between ${className || ''}`}>
      {/* Left: Brand/Logo */}
      <div className="flex items-center space-x-4">
        <div className="text-xl font-bold text-white">
          Zero-Instrument Profiler
        </div>
        <div className="text-sm text-gray-400">
          3D Flame Graph Visualizer
        </div>
      </div>

      {/* Center: Navigation Items */}
      <nav className="hidden md:flex items-center space-x-6">
        <button className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">
          Dashboard
        </button>
        <button className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">
          Sessions
        </button>
        <button className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">
          Analytics
        </button>
        <button className="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition-colors">
          Settings
        </button>
      </nav>

      {/* Right: Actions */}
      <div className="flex items-center space-x-4">
        <button className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors">
          New Session
        </button>
        <button className="text-gray-300 hover:text-white p-2 rounded-md transition-colors">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707" />
          </svg>
        </button>
      </div>
    </header>
  );
}; 