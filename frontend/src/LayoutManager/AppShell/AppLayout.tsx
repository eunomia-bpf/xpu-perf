import React from 'react';
import { NavigationHeader } from './NavigationHeader';
import { StatusBar } from './StatusBar';
import { Sidebar } from './Sidebar';
import { MainViewport } from './MainViewport';

interface AppLayoutProps {
  children: React.ReactNode;
  sidebar?: React.ReactNode;
  className?: string;
}

export const AppLayout: React.FC<AppLayoutProps> = ({ children, sidebar, className }) => {
  return (
    <div className={`w-screen h-screen bg-gray-900 flex flex-col overflow-hidden ${className || ''}`}>
      {/* Navigation Header */}
      <NavigationHeader />
      
      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        {sidebar && (
          <Sidebar position="left" width="w-80">
            {sidebar}
          </Sidebar>
        )}
        
        {/* Main Viewport */}
        <MainViewport>
          {children}
        </MainViewport>
      </div>
      
      {/* Status Bar */}
      <StatusBar />
    </div>
  );
}; 