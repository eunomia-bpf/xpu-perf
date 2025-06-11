import React from 'react';
import { useDataSourceStore } from '@/DataManager/DataStore/dataSourceStore';

interface DataTableViewProps {
  className?: string;
}

export const DataTableView: React.FC<DataTableViewProps> = ({ className }) => {
  const { currentDataContext } = useDataSourceStore();

  // Convert data to simple text display
  const data = currentDataContext.resolvedData;
  const dataEntries = Object.entries(data);
  const dataCount = dataEntries.length;

  return (
    <div className={`flex flex-col h-full ${className || ''}`}>
      {/* Main Content */}
      <div className="flex-1 bg-gray-900 overflow-auto p-6">
        <div className="max-w-4xl">
          <h3 className="text-xl font-semibold text-white mb-4">Data View</h3>
          
          <div className="bg-gray-800 rounded-lg p-4 mb-4">
            <h4 className="text-sm font-medium text-white mb-2">Data Summary</h4>
            <p className="text-gray-300">Format: {currentDataContext.format}</p>
            <p className="text-gray-300">Fields: {currentDataContext.fields.join(', ') || 'none'}</p>
            <p className="text-gray-300">Total entries: {dataCount}</p>
            <p className="text-gray-300">Sources: {currentDataContext.sources.length}</p>
            {dataCount > 0 && (
              <p className="text-gray-300">Data available for visualization</p>
            )}
          </div>

          {dataCount > 0 ? (
            <div className="bg-gray-800 rounded-lg p-4">
              <h4 className="text-sm font-medium text-white mb-3">Raw Data</h4>
              <div className="space-y-2 text-sm font-mono">
                {dataEntries.slice(0, 10).map(([key, value], index) => (
                  <div key={index} className="text-gray-300">
                    <span className="text-blue-400">{key}:</span> <span className="text-green-400">{JSON.stringify(value)}</span>
                  </div>
                ))}
                {dataCount > 10 && (
                  <div className="text-gray-500 italic">
                    ... and {dataCount - 10} more entries
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="bg-gray-800 rounded-lg p-8 text-center">
              <p className="text-gray-400">No data available</p>
              <p className="text-xs text-gray-500 mt-2">Select data sources to see data here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}; 