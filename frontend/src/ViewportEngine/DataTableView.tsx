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

  // Format JSON with syntax highlighting
  const formatValue = (value: any): React.JSX.Element => {
    const jsonString = JSON.stringify(value, null, 2);
    return (
      <span 
        className="font-mono text-sm"
        dangerouslySetInnerHTML={{
          __html: jsonString
            .replace(/"([^"]+)":/g, '<span class="code-key">"$1"</span>:')
            .replace(/:\s*"([^"]+)"/g, ': <span class="code-string">"$1"</span>')
            .replace(/:\s*(\d+)/g, ': <span class="code-number">$1</span>')
            .replace(/([{}[\]])/g, '<span class="code-bracket">$1</span>')
        }}
      />
    );
  };

  return (
    <div className={`flex flex-col h-full ${className || ''}`}>
      {/* Main Content */}
      <div className="flex-1 bg-gray-50 overflow-auto p-6">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <span className="text-white text-lg">📊</span>
            </div>
            <h3 className="text-2xl font-bold text-gray-800">Data View</h3>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
              <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center space-x-2">
                <span>📈</span>
                <span>Data Summary</span>
              </h4>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="font-medium text-gray-600">Format:</span>
                  <span className="text-gray-800 font-mono bg-gray-100 px-2 py-1 rounded text-sm">{currentDataContext.format}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium text-gray-600">Fields:</span>
                  <span className="text-gray-800">{currentDataContext.fields.join(', ') || 'none'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium text-gray-600">Total entries:</span>
                  <span className="text-gray-800 font-semibold">{dataCount}</span>
                </div>
                <div className="flex justify-between">
                  <span className="font-medium text-gray-600">Sources:</span>
                  <span className="text-gray-800 font-semibold">{currentDataContext.sources.length}</span>
                </div>
              </div>
              {dataCount > 0 && (
                <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                  <p className="text-green-800 text-sm flex items-center space-x-2">
                    <span>✅</span>
                    <span>Data available for visualization</span>
                  </p>
                </div>
              )}
            </div>

            <div className="bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
              <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center space-x-2">
                <span>🔍</span>
                <span>Data Sources</span>
              </h4>
              {currentDataContext.sources.length > 0 ? (
                <div className="space-y-2">
                  {currentDataContext.sources.map((source) => (
                    <div key={source.id} className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                      <div className="font-medium text-gray-800 text-sm">{source.name}</div>
                      <div className="text-gray-600 text-xs mt-1">{source.type} • {source.format}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500 text-sm">No data sources selected</p>
              )}
            </div>
          </div>

          {dataCount > 0 ? (
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
              <div className="bg-gray-50 px-6 py-4 border-b border-gray-200">
                <h4 className="text-lg font-semibold text-gray-800 flex items-center space-x-2">
                  <span>💾</span>
                  <span>Raw Data</span>
                </h4>
              </div>
              <div className="p-6">
                <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                  <div className="space-y-2 text-sm">
                    {dataEntries.slice(0, 10).map(([key, value], index) => (
                      <div key={index} className="flex items-start space-x-3">
                        <span className="code-key text-blue-400 font-mono">{key}:</span>
                        <div className="flex-1 text-gray-300">
                          {formatValue(value)}
                        </div>
                      </div>
                    ))}
                    {dataCount > 10 && (
                      <div className="text-gray-500 italic text-center py-2 border-t border-gray-700">
                        ... and {dataCount - 10} more entries
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-lg p-12 text-center border border-gray-200 shadow-sm">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl text-gray-400">📊</span>
              </div>
              <h4 className="text-lg font-semibold text-gray-600 mb-2">No data available</h4>
              <p className="text-gray-500">Select data sources to see data here</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}; 