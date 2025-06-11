import React, { useState } from 'react';

interface SidebarProps {
  children: React.ReactNode;
  position?: 'left' | 'right';
  width?: string;
  collapsible?: boolean;
  className?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  children, 
  position = 'left',
  width = 'w-80',
  collapsible = true,
  className 
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleToggle = () => {
    if (collapsible) {
      setIsCollapsed(!isCollapsed);
    }
  };

  return (
    <aside 
      className={`
        ${isCollapsed ? 'w-12' : width} 
        bg-gray-800 border-gray-700 transition-all duration-300 ease-in-out flex flex-col
        ${position === 'left' ? 'border-r' : 'border-l'}
        ${className || ''}
      `}
    >
      {/* Sidebar Header with Toggle */}
      {collapsible && (
        <div className="p-4 border-b border-gray-700">
          <button
            onClick={handleToggle}
            className="w-full flex items-center justify-between text-gray-300 hover:text-white transition-colors"
          >
            {!isCollapsed && (
              <span className="text-sm font-medium">Control Panel</span>
            )}
            <svg 
              className={`w-4 h-4 transition-transform duration-300 ${
                isCollapsed 
                  ? (position === 'left' ? 'rotate-0' : 'rotate-180')
                  : (position === 'left' ? 'rotate-180' : 'rotate-0')
              }`}
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        </div>
      )}

      {/* Sidebar Content */}
      <div className={`flex-1 overflow-hidden ${isCollapsed ? 'hidden' : 'block'}`}>
        {children}
      </div>

      {/* Collapsed Indicator */}
      {isCollapsed && (
        <div className="p-2 text-center">
          <div className="w-2 h-2 bg-blue-500 rounded-full mx-auto"></div>
        </div>
      )}
    </aside>
  );
}; 