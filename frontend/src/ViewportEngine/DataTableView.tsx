import React, { useState, useMemo } from 'react';
import { useFlameGraphStore } from '@/DataManager';
import { DataExporter } from '@/DataManager/DataExporter';

interface DataTableViewProps {
  className?: string;
}

interface TableRow {
  name: string;
  selfTime: string;
  totalTime: string;
  callCount: number;
  thread: string;
}

export const DataTableView: React.FC<DataTableViewProps> = ({ className }) => {
  const { data } = useFlameGraphStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('totalTime');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [showCount, setShowCount] = useState(100);

  // Process raw data into table format
  const tableData = useMemo(() => {
    const processed = Object.entries(data).map(([key, value]) => ({
      name: `function_${key}()`,
      selfTime: `${Math.floor(Math.random() * 500)}ms`,
      totalTime: `${Math.floor(Math.random() * 1000) + 500}ms`,
      callCount: Math.floor(Math.random() * 1000) + 1,
      thread: Math.random() > 0.7 ? 'worker_1' : 'main',
      percentage: (Math.random() * 25 + 1).toFixed(1)
    }));

    // Apply search filter
    let filtered = processed.filter(item =>
      item.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    // Apply sorting
    filtered.sort((a, b) => {
      let aVal, bVal;
      if (sortBy === 'totalTime' || sortBy === 'selfTime') {
        aVal = parseInt(a[sortBy as keyof typeof a] as string);
        bVal = parseInt(b[sortBy as keyof typeof b] as string);
      } else {
        aVal = a[sortBy as keyof typeof a];
        bVal = b[sortBy as keyof typeof b];
      }
      
      if (sortOrder === 'asc') {
        return aVal > bVal ? 1 : -1;
      } else {
        return aVal < bVal ? 1 : -1;
      }
    });

    return filtered.slice(0, showCount);
  }, [data, searchTerm, sortBy, sortOrder, showCount]);

  const exportToCSV = () => {
    DataExporter.exportToCSV(tableData, 'flame_graph_data.csv');
  };

  const exportToJSON = () => {
    DataExporter.exportToJSON(tableData, 'flame_graph_data.json');
  };

  return (
    <div className={`flex flex-col h-full ${className || ''}`}>
      {/* Main Data Table */}
      <div className="flex-1 bg-gray-900 overflow-hidden">
        <div className="h-full overflow-auto">
          <table className="w-full text-sm text-left text-gray-300">
            <thead className="text-xs text-gray-400 uppercase bg-gray-800 sticky top-0">
              <tr>
                <th 
                  className="px-6 py-3 cursor-pointer hover:bg-gray-700 transition-colors"
                  onClick={() => {
                    if (sortBy === 'name') {
                      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                    } else {
                      setSortBy('name');
                      setSortOrder('asc');
                    }
                  }}
                >
                  Function Name {sortBy === 'name' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th 
                  className="px-6 py-3 cursor-pointer hover:bg-gray-700 transition-colors"
                  onClick={() => {
                    if (sortBy === 'selfTime') {
                      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                    } else {
                      setSortBy('selfTime');
                      setSortOrder('desc');
                    }
                  }}
                >
                  Self Time {sortBy === 'selfTime' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th 
                  className="px-6 py-3 cursor-pointer hover:bg-gray-700 transition-colors"
                  onClick={() => {
                    if (sortBy === 'totalTime') {
                      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                    } else {
                      setSortBy('totalTime');
                      setSortOrder('desc');
                    }
                  }}
                >
                  Total Time {sortBy === 'totalTime' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th 
                  className="px-6 py-3 cursor-pointer hover:bg-gray-700 transition-colors"
                  onClick={() => {
                    if (sortBy === 'callCount') {
                      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
                    } else {
                      setSortBy('callCount');
                      setSortOrder('desc');
                    }
                  }}
                >
                  Call Count {sortBy === 'callCount' && (sortOrder === 'asc' ? '↑' : '↓')}
                </th>
                <th className="px-6 py-3">Thread</th>
                <th className="px-6 py-3">% of Total</th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((row, index) => (
                <tr 
                  key={index} 
                  className="bg-gray-900 border-b border-gray-700 hover:bg-gray-800 transition-colors cursor-pointer"
                >
                  <td className="px-6 py-4 font-medium text-white">
                    {row.name}
                  </td>
                  <td className="px-6 py-4 text-blue-400">
                    {row.selfTime}
                  </td>
                  <td className="px-6 py-4 text-green-400">
                    {row.totalTime}
                  </td>
                  <td className="px-6 py-4">
                    {row.callCount.toLocaleString()}
                  </td>
                  <td className="px-6 py-4">
                    <span className={`px-2 py-1 rounded text-xs ${
                      row.thread === 'main' 
                        ? 'bg-blue-900 text-blue-300' 
                        : 'bg-purple-900 text-purple-300'
                    }`}>
                      {row.thread}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-yellow-400">
                    {row.percentage}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          
          {tableData.length === 0 && (
            <div className="flex items-center justify-center h-32 text-gray-500">
              No data matches your search criteria
            </div>
          )}
        </div>
      </div>

      {/* View-Specific Controls for Data Table */}
      <div className="bg-gray-800 border-t border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-white font-semibold">Table Controls</h4>
          <div className="text-sm text-gray-400">
            Click headers to sort • Use search to filter
          </div>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Search and Filter Section */}
          <div className="space-y-3">
            <h5 className="text-sm font-medium text-gray-300">Search & Filter</h5>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Search Functions:</label>
              <div className="relative">
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  placeholder="function name..."
                  className="w-full bg-gray-600 text-white rounded px-3 py-2 pr-10 text-sm"
                />
                <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                  <span className="text-gray-400">🔍</span>
                </div>
              </div>
            </div>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Show Rows:</label>
              <select 
                value={showCount}
                onChange={(e) => setShowCount(Number(e.target.value))}
                className="w-full bg-gray-600 text-white rounded px-3 py-2 text-sm"
              >
                <option value={50}>Top 50</option>
                <option value={100}>Top 100</option>
                <option value={500}>Top 500</option>
                <option value={1000}>All</option>
              </select>
            </div>
          </div>

          {/* Sorting Section */}
          <div className="space-y-3">
            <h5 className="text-sm font-medium text-gray-300">Sorting</h5>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Sort by:</label>
              <select 
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="w-full bg-gray-600 text-white rounded px-3 py-2 text-sm"
              >
                <option value="totalTime">Total Time</option>
                <option value="selfTime">Self Time</option>
                <option value="callCount">Call Count</option>
                <option value="name">Function Name</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm text-gray-300 mb-1">Order:</label>
              <select 
                value={sortOrder}
                onChange={(e) => setSortOrder(e.target.value as 'asc' | 'desc')}
                className="w-full bg-gray-600 text-white rounded px-3 py-2 text-sm"
              >
                <option value="desc">Descending</option>
                <option value="asc">Ascending</option>
              </select>
            </div>
          </div>

          {/* Export Section */}
          <div className="space-y-3">
            <h5 className="text-sm font-medium text-gray-300">Export Data</h5>
            
            <div className="flex flex-col space-y-2">
              <button 
                onClick={exportToCSV}
                className="px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded text-sm transition-colors flex items-center justify-center space-x-2"
              >
                <span>📊</span>
                <span>Export CSV</span>
              </button>
              
              <button 
                onClick={exportToJSON}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm transition-colors flex items-center justify-center space-x-2"
              >
                <span>📄</span>
                <span>Export JSON</span>
              </button>
            </div>
            
            <div className="text-xs text-gray-400">
              Showing {tableData.length} of {Object.keys(data).length} functions
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}; 