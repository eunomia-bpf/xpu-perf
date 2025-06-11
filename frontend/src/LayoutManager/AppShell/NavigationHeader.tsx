import React from 'react';

interface NavigationHeaderProps {
  className?: string;
}

export const NavigationHeader: React.FC<NavigationHeaderProps> = ({ className }) => {
  return (
    <header className={`bg-gray-900 border-b border-gray-700 px-6 py-4 flex items-center justify-between ${className || ''}`}>
      {/* Left: Simplified menu */}
      <div className="flex items-center">
        <div className="text-lg font-semibold text-white">
          menu
        </div>
      </div>

      {/* Right: Action buttons */}
      <div className="flex items-center space-x-2">
        <button className="px-3 py-1 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors text-sm">
          Help
        </button>
        <button className="px-3 py-1 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors text-sm">
          Export
        </button>
        <button className="px-3 py-1 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors text-sm">
          Settings
        </button>
      </div>
    </header>
  );
}; 