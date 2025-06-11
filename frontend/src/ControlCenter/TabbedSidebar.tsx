import React, { useState } from 'react';
import { DynamicAnalyzer } from '@/AnalyzerEngine';
import { DataSourceSelector } from './DataSourceSelector';
import { ViewSelector } from './ViewSelector';

type TabId = 'analyzer' | 'data' | 'view';

interface Tab {
  id: TabId;
  label: string;
  icon: string;
  component: React.ReactNode;
}

export const TabbedSidebar: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('analyzer');

  const tabs: Tab[] = [
    {
      id: 'analyzer',
      label: 'Analyzer',
      icon: '🔬',
      component: <DynamicAnalyzer />
    },
    {
      id: 'data',
      label: 'Data',
      icon: '📊',
      component: <DataSourceSelector />
    },
    {
      id: 'view',
      label: 'View',
      icon: '🎯',
      component: <ViewSelector />
    }
  ];

  const activeTabData = tabs.find(tab => tab.id === activeTab);

  return (
    <div className="flex flex-col h-full">
      {/* Tab Headers */}
      <div className="flex border-b border-gray-200 bg-gray-50">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-3 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-blue-600 border-b-2 border-blue-600 bg-white'
                : 'text-gray-600 hover:text-gray-800 hover:bg-gray-100'
            }`}
          >
            <div className="flex flex-col items-center space-y-1">
              <span className="text-lg">{tab.icon}</span>
              <span>{tab.label}</span>
            </div>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {activeTabData?.component}
      </div>
    </div>
  );
}; 