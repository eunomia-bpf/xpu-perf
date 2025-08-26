import React from 'react';
import { ProfileDataView } from '@/ViewportEngine/ProfileDataView';
import { DataTableView } from '@/ViewportEngine/DataTableView';
import { useViewContext } from './ViewContext';

export const SingleViewLayout: React.FC = () => {
  const { currentView } = useViewContext();

  const renderCurrentView = () => {
    switch (currentView) {
      case '3d-flame':
        return <ProfileDataView className="w-full h-full" />;
      case 'data-table':
        return <DataTableView className="w-full h-full" />;
      default:
        return <ProfileDataView className="w-full h-full" />;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Main Viewport - Full Height */}
      <div className="flex-1 overflow-hidden">
        {renderCurrentView()}
      </div>
    </div>
  );
}; 