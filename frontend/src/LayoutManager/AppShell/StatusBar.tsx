import React from 'react';

interface StatusBarProps {
  className?: string;
}

export const StatusBar: React.FC<StatusBarProps> = ({ className }) => {
  return (
    <footer className={`bg-gray-800 border-t border-gray-700 px-6 py-3 flex items-center justify-between text-sm ${className || ''}`}>
      {/* Left: Status Information */}
      <div className="flex items-center space-x-6">
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span className="text-gray-300">Ready</span>
        </div>
        <div className="text-gray-400">
          Session: <span className="text-gray-300">sample_data</span>
        </div>
        <div className="text-gray-400">
          Threads: <span className="text-gray-300">5</span>
        </div>
      </div>

      {/* Center: Mini Analytics */}
      <div className="hidden md:flex items-center space-x-6">
        <div className="text-gray-400">
          Total Samples: <span className="text-gray-300">4,567</span>
        </div>
        <div className="text-gray-400">
          Max Depth: <span className="text-gray-300">12</span>
        </div>
        <div className="text-gray-400">
          Render Time: <span className="text-gray-300">16ms</span>
        </div>
      </div>

      {/* Right: System Information */}
      <div className="flex items-center space-x-4">
        <div className="text-gray-400">
          Memory: <span className="text-gray-300">256MB</span>
        </div>
        <div className="text-gray-400">
          FPS: <span className="text-gray-300">60</span>
        </div>
        <button className="text-gray-400 hover:text-gray-300 transition-colors">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </button>
      </div>
    </footer>
  );
}; 