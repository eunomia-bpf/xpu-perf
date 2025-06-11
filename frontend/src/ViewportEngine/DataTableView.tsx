import React, { useState, useMemo } from 'react';
import { useFlameGraphStore } from '@/DataManager';

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
  const [showControls, setShowControls] = useState(false);
  const [sortBy, setSortBy] = useState<'name' | 'time' | 'count'>('time');
  const [filterText, setFilterText] = useState('');
  const [showLimit, setShowLimit] = useState<'100' | '500' | 'all'>('100');
  const { data } = useFlameGraphStore();

  // Process flame graph data into table format
  const tableData = useMemo(() => {
    const rows: TableRow[] = [];
    
    Object.entries(data).forEach(([threadName, threadData]) => {
      if (Array.isArray(threadData)) {
        threadData.forEach(entry => {
          if (entry.stack && Array.isArray(entry.stack)) {
            entry.stack.forEach((funcName) => {
              if (funcName && funcName.trim()) {
                const existingRow = rows.find(row => row.name === funcName && row.thread === threadName);
                if (existingRow) {
                  existingRow.callCount += 1;
                } else {
                  rows.push({
                    name: funcName,
                    selfTime: `${entry.count}ms`,
                    totalTime: `${entry.count}ms`,
                    callCount: 1,
                    thread: threadName
                  });
                }
              }
            });
          }
        });
      }
    });

    return rows;
  }, [data]);

  // Filter and sort data
  const processedData = useMemo(() => {
    let filtered = tableData.filter(row => 
      row.name.toLowerCase().includes(filterText.toLowerCase())
    );

    // Sort data
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'count':
          return b.callCount - a.callCount;
        case 'time':
        default:
          return parseInt(b.totalTime) - parseInt(a.totalTime);
      }
    });

    // Apply limit
    if (showLimit !== 'all') {
      const limit = parseInt(showLimit);
      filtered = filtered.slice(0, limit);
    }

    return filtered;
  }, [tableData, filterText, sortBy, showLimit]);

  const exportToCSV = () => {
    const csvContent = [
      'Function Name,Self Time,Total Time,Call Count,Thread',
      ...processedData.map(row => 
        `"${row.name}","${row.selfTime}","${row.totalTime}",${row.callCount},"${row.thread}"`
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'flame_graph_data.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className={`relative w-full h-full bg-gray-900 ${className || ''}`}>
      {/* Data Table */}
      <div className="w-full h-full overflow-auto">
        <div className="p-4">
          <h3 className="text-2xl font-bold text-white mb-4">📊 Data Table</h3>
          
          {/* Search and Filter Controls */}
          <div className="mb-4 flex space-x-4">
            <input
              type="text"
              placeholder="Search functions..."
              className="flex-1 bg-gray-800 text-white px-3 py-2 rounded border border-gray-600"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
            />
            <select
              className="bg-gray-800 text-white px-3 py-2 rounded border border-gray-600"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as 'name' | 'time' | 'count')}
            >
              <option value="time">Sort by Time</option>
              <option value="name">Sort by Name</option>
              <option value="count">Sort by Count</option>
            </select>
            <select
              className="bg-gray-800 text-white px-3 py-2 rounded border border-gray-600"
              value={showLimit}
              onChange={(e) => setShowLimit(e.target.value as '100' | '500' | 'all')}
            >
              <option value="100">Top 100</option>
              <option value="500">Top 500</option>
              <option value="all">Show All</option>
            </select>
          </div>

          {/* Table */}
          <div className="bg-gray-800 rounded-lg overflow-hidden">
            <table className="w-full text-white">
              <thead className="bg-gray-700">
                <tr>
                  <th className="text-left p-3 border-b border-gray-600">Function Name</th>
                  <th className="text-left p-3 border-b border-gray-600">Self Time</th>
                  <th className="text-left p-3 border-b border-gray-600">Total Time</th>
                  <th className="text-left p-3 border-b border-gray-600">Call Count</th>
                  <th className="text-left p-3 border-b border-gray-600">Thread</th>
                </tr>
              </thead>
              <tbody>
                {processedData.map((row, index) => (
                  <tr key={index} className="hover:bg-gray-700 transition-colors">
                    <td className="p-3 border-b border-gray-700 font-mono text-sm">{row.name}</td>
                    <td className="p-3 border-b border-gray-700">{row.selfTime}</td>
                    <td className="p-3 border-b border-gray-700">{row.totalTime}</td>
                    <td className="p-3 border-b border-gray-700">{row.callCount}</td>
                    <td className="p-3 border-b border-gray-700 text-gray-400">{row.thread}</td>
                  </tr>
                ))}
                {processedData.length === 0 && (
                  <tr>
                    <td colSpan={5} className="p-8 text-center text-gray-400">
                      No data available. Start profiling to see results.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Stats */}
          <div className="mt-4 text-sm text-gray-400">
            Showing {processedData.length} of {tableData.length} functions
          </div>
        </div>
      </div>

      {/* View-Specific Controls Panel */}
      <div className="absolute bottom-4 right-4 flex flex-col space-y-2">
        {/* Controls Toggle Button */}
        <button
          className="bg-gray-800/90 backdrop-blur-md text-white p-2 rounded-lg border border-white/10 hover:bg-gray-700/90 transition-colors"
          onClick={() => setShowControls(!showControls)}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
          </svg>
        </button>

        {/* Expandable Controls Panel */}
        {showControls && (
          <div className="bg-gray-800/90 backdrop-blur-md rounded-lg p-4 border border-white/10 space-y-3 min-w-[200px]">
            <h4 className="text-white font-medium text-sm">Table Controls</h4>
            
            <div className="space-y-2">
              <button 
                className="w-full px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded"
                onClick={exportToCSV}
              >
                📊 Export CSV
              </button>
              <button className="w-full px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded">
                📋 Copy Data
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 