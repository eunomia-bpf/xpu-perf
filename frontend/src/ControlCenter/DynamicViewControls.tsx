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
    <div className="bg-gray-700 rounded p-3 space-y-3">
      {/* View Type Selector */}
      <div className="space-y-2">
        {compatibleViews.map(view => (
          <label key={view.id} className="flex items-center space-x-2 cursor-pointer group">
            <input 
              type="radio" 
              name="viewType" 
              value={view.id}
              checked={selectedViewId === view.id}
              onChange={() => selectView(view.id)}
              className="w-4 h-4 text-blue-600 bg-gray-600 border-gray-500 focus:ring-blue-500 focus:ring-2"
            />
            <div className="flex-1">
              <div className="flex items-center space-x-2">
                <span className="text-sm text-white">{view.icon} {view.displayName}</span>
              </div>
              <p className="text-xs text-gray-400 ml-6">{view.description}</p>
            </div>
          </label>
        ))}
      </div>
      
      {compatibleViews.length === 0 && (
        <div className="text-xs text-gray-400 italic">
          No compatible views for current data format ({currentDataContext.format})
        </div>
      )}
      
      <div className="text-xs text-gray-500 border-t border-gray-600 pt-2">
        <div>Current data format: {currentDataContext.format}</div>
        <div>Available fields: {currentDataContext.fields.join(', ') || 'none'}</div>
      </div>
    </div>
  );
}; 