import React from 'react';
import { useAnalyzerStore } from '@/DataManager/DataStore/analyzerStore';
import { useDataSourceStore } from '@/DataManager/DataStore/dataSourceStore';

export const DynamicViewControls: React.FC = () => {
  const {
    selectedViewId,
    getAvailableViews,
    selectView
  } = useAnalyzerStore();

  const { currentDataContext } = useDataSourceStore();

  const availableViews = getAvailableViews();

  // Filter views based on current data format compatibility
  const compatibleViews = availableViews.filter(view => {
    const viewRequirements = view.dataRequirements;
    const currentData = currentDataContext;

    // Check format compatibility
    if (viewRequirements.format === 'any') return true;
    if (viewRequirements.format === currentData.format) return true;
    
    // Check if required fields are available
    const hasRequiredFields = viewRequirements.requiredFields.every(field =>
      currentData.fields.includes(field)
    );
    
    return hasRequiredFields;
  });

  return (
    <div className="profiler-panel p-4 space-y-3">
      {compatibleViews.map(view => (
        <label key={view.id} className="flex items-center space-x-3 cursor-pointer group">
          <input 
            type="radio" 
            name="viewType" 
            value={view.id}
            checked={selectedViewId === view.id}
            onChange={() => selectView(view.id)}
            className="w-4 h-4 text-blue-600 bg-white border-gray-300 focus:ring-blue-500 focus:ring-2"
          />
          <div className="flex-1">
            <span className="text-sm font-medium text-gray-800">{view.icon} {view.displayName}</span>
            <div className="text-xs text-gray-600 mt-0.5">{view.description}</div>
          </div>
        </label>
      ))}
      
      {compatibleViews.length === 0 && (
        <div className="text-sm text-gray-500 italic py-3 text-center">
          No compatible views for format: {currentDataContext.format}
        </div>
      )}
      
      <div className="text-xs text-gray-500 border-t border-gray-200 pt-3">
        <div>Format: {currentDataContext.format}</div>
        <div>Fields: {currentDataContext.fields.join(', ') || 'none'}</div>
      </div>
    </div>
  );
}; 