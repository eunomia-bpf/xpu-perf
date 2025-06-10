import React from 'react';

interface MainViewportProps {
  children: React.ReactNode;
  className?: string;
}

export const MainViewport: React.FC<MainViewportProps> = ({ children, className }) => {
  return (
    <main className={`flex-1 flex flex-col bg-gray-900 relative overflow-hidden ${className || ''}`}>
      {/* Visualization Area */}
      <div className="flex-1 relative">
        {children}
      </div>
    </main>
  );
}; 