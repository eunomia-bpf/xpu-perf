import React from 'react';
import { useAnalyzerStore } from '@/DataManager/DataStore/analyzerStore';

export const DynamicViewControls: React.FC = () => {
  const {
    selectedAnalyzerId,
    selectedViewId,
    getAvailableViews,
    selectView
  } = useAnalyzerStore();

  const availableViews = getAvailableViews(selectedAnalyzerId || undefined);

  // Filter views based on current analyzer's supported views
  const filteredViews = selectedAnalyzerId 
    ? availableViews.filter(view => {
        const analyzer = useAnalyzerStore.getState().registry.analyzers[selectedAnalyzerId];
        return analyzer?.supportedViews.includes(view.id);
      })
    : availableViews;

  return (
    <div className="bg-gray-700 rounded p-3 space-y-3">
      {/* View Type Selector */}
      <div className="space-y-2">
        {filteredViews.map(view => (
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
      
      {filteredViews.length === 0 && (
        <div className="text-xs text-gray-400 italic">
          No views available for selected analyzer
        </div>
      )}
    </div>
  );
}; 