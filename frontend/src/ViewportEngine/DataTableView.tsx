import React from 'react';
import { useDataSourceStore } from '@/DataManager/DataStore/dataSourceStore';

interface DataTableViewProps {
  className?: string;
}

export const DataTableView: React.FC<DataTableViewProps> = ({ className }) => {
  const { currentDataContext } = useDataSourceStore();

  const data = currentDataContext.resolvedData;
  const dataEntries = Object.entries(data);
  const dataCount = dataEntries.length;

  return (
    <div className={`flex flex-col h-full ${className || ''}`}>
      <div className="flex-1 bg-gray-50 overflow-auto p-6">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center space-x-3 mb-6">
            <span className="text-2xl">📊</span>
            <h3 className="text-xl font-semibold text-gray-800">Data View</h3>
          </div>
          
          <div className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm mb-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Format:</span>
                <div className="font-mono text-xs bg-gray-100 px-2 py-1 rounded mt-1">{currentDataContext.format}</div>
              </div>
              <div>
                <span className="text-gray-500">Entries:</span>
                <div className="font-semibold text-gray-800 mt-1">{dataCount}</div>
              </div>
              <div>
                <span className="text-gray-500">Sources:</span>
                <div className="font-semibold text-gray-800 mt-1">{currentDataContext.sources.length}</div>
              </div>
              <div>
                <span className="text-gray-500">Fields:</span>
                <div className="text-xs text-gray-600 mt-1">{currentDataContext.fields.join(', ') || 'none'}</div>
              </div>
            </div>
          </div>

          {dataCount > 0 ? (
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
              <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                <h4 className="font-medium text-gray-800">Raw Data</h4>
              </div>
              <div className="p-4">
                <div className="bg-gray-900 rounded p-3 overflow-x-auto max-h-96 overflow-y-auto">
                  <pre className="text-sm text-gray-300 font-mono">
                    {JSON.stringify(data, null, 2)}
                  </pre>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg p-8 text-center border border-gray-200 shadow-sm">
              <span className="text-4xl text-gray-300 block mb-2">📊</span>
              <p className="text-gray-500">No data available</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}; 