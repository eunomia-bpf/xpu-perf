import React, { useState, useMemo } from 'react';
import { useFlameGraphStore } from '@/DataManager/DataStore';

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
      threadData.forEach(entry => {
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
      });
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
      ['Function Name', 'Self Time', 'Total Time', 'Call Count', 'Thread'].join(','),
      ...processedData.map(row => 
        [row.name, row.selfTime, row.totalTime, row.callCount, row.thread].join(',')
      )
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'flame-graph-data.csv';
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className={`relative w-full h-full bg-gray-900 ${className || ''}`}>
      {/* Data Table */}
      <div className="w-full h-full overflow-auto p-4">
        <div className="bg-gray-800 rounded-lg overflow-hidden">
          <table className="w-full text-white">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left cursor-pointer hover:bg-gray-600" onClick={() => setSortBy('name')}>
                  Function Name {sortBy === 'name' && '↓'}
                </th>
                <th className="px-4 py-3 text-left">Self Time</th>
                <th className="px-4 py-3 text-left cursor-pointer hover:bg-gray-600" onClick={() => setSortBy('time')}>
                  Total Time {sortBy === 'time' && '↓'}
                </th>
                <th className="px-4 py-3 text-left cursor-pointer hover:bg-gray-600" onClick={() => setSortBy('count')}>
                  Call Count {sortBy === 'count' && '↓'}
                </th>
                <th className="px-4 py-3 text-left">Thread</th>
              </tr>
            </thead>
            <tbody>
              {processedData.map((row, index) => (
                <tr key={`${row.name}-${row.thread}-${index}`} className="border-t border-gray-600 hover:bg-gray-700/50">
                  <td className="px-4 py-3 font-mono">{row.name}</td>
                  <td className="px-4 py-3">{row.selfTime}</td>
                  <td className="px-4 py-3">{row.totalTime}</td>
                  <td className="px-4 py-3">{row.callCount}</td>
                  <td className="px-4 py-3">{row.thread}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {processedData.length === 0 && (
          <div className="text-center text-gray-400 mt-8">
            {Object.keys(data).length === 0 ? 'No data available. Start the analyzer to collect data.' : 'No functions match the filter criteria'}
          </div>
        )}
      </div>

      {/* View-Specific Controls Panel */}
      <div className="absolute bottom-4 right-4 flex flex-col space-y-2">
        {/* Controls Toggle Button */}
        <button
          className="bg-gray-800/90 backdrop-blur-md text-white p-2 rounded-lg border border-white/10 hover:bg-gray-700/90 transition-colors"
          onClick={() => setShowControls(!showControls)}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
          </svg>
        </button>

        {/* Expandable Controls Panel */}
        {showControls && (
          <div className="bg-black/80 backdrop-blur-md text-white p-4 rounded-lg min-w-64 border border-white/10 space-y-3">
            <h4 className="text-lg font-semibold mb-3">Table Controls</h4>
            
            {/* Search Filter */}
            <div>
              <label className="block text-sm mb-1">
                Search Functions
                <input
                  type="text"
                  className="w-full mt-1 px-3 py-2 bg-gray-700 rounded border border-gray-600 text-white"
                  placeholder="function name..."
                  value={filterText}
                  onChange={(e) => setFilterText(e.target.value)}
                />
              </label>
            </div>

            {/* Sort Options */}
            <div>
              <label className="block text-sm mb-1">
                Sort by
                <select 
                  className="w-full mt-1 px-3 py-2 bg-gray-700 rounded border border-gray-600 text-white"
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value as any)}
                >
                  <option value="name">Function Name</option>
                  <option value="time">Total Time</option>
                  <option value="count">Call Count</option>
                </select>
              </label>
            </div>

            {/* Display Options */}
            <div>
              <label className="block text-sm mb-1">
                Show
                <select 
                  className="w-full mt-1 px-3 py-2 bg-gray-700 rounded border border-gray-600 text-white"
                  value={showLimit}
                  onChange={(e) => setShowLimit(e.target.value as any)}
                >
                  <option value="100">Top 100</option>
                  <option value="500">Top 500</option>
                  <option value="all">All Functions</option>
                </select>
              </label>
            </div>

            {/* Export */}
            <button 
              className="w-full bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm transition-colors"
              onClick={exportToCSV}
            >
              Export CSV
            </button>
          </div>
        )}
      </div>
    </div>
  );
}; 