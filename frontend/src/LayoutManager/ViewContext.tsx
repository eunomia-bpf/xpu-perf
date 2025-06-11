import React, { createContext, useContext, useEffect } from 'react';
import { useAnalyzerStore } from '@/DataManager/DataStore/analyzerStore';

export type ViewType = '3d-flame' | 'data-table';

interface ViewContextType {
  currentView: ViewType;
  setCurrentView: (view: ViewType) => void;
}

const ViewContext = createContext<ViewContextType | undefined>(undefined);

export const useViewContext = () => {
  const context = useContext(ViewContext);
  if (!context) {
    throw new Error('useViewContext must be used within a ViewProvider');
  }
  return context;
};

export const ViewProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { selectedViewId, selectView } = useAnalyzerStore();
  
  // Convert dynamic view ID to legacy ViewType for backward compatibility
  const currentView: ViewType = selectedViewId === 'data-table' ? 'data-table' : '3d-flame';
  
  const setCurrentView = (view: ViewType) => {
    selectView(view);
  };

  // Initialize default view if none selected
  useEffect(() => {
    if (!selectedViewId) {
      selectView('3d-flame');
    }
  }, [selectedViewId, selectView]);
  
  return (
    <ViewContext.Provider value={{ currentView, setCurrentView }}>
      {children}
    </ViewContext.Provider>
  );
}; 