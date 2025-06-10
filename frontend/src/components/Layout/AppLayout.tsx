import React from 'react';

interface AppLayoutProps {
  children: React.ReactNode;
  className?: string;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children, className }) => {
  return (
    <div className={`relative w-screen h-screen bg-gray-900 overflow-hidden ${className || ''}`}>
      {children}
    </div>
  );
}; 