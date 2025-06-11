import React, { useState } from 'react';
import { useFlameGraphStore } from '@/stores';

interface DataTableViewProps {
  className?: string;
}

export const DataTableView: React.FC<DataTableViewProps> = ({ className }) => {
  const [showControls, setShowControls] = useState(false);
  const [sortBy, setSortBy] = useState<'name' | 'time' | 'count'>('time');
  const [filterText, setFilterText] = useState('');
  const { data } = useFlameGraphStore();

  // Mock data for demonstration
  const tableData = [
    { name: 'main()', selfTime: '45ms', totalTime: '1,234ms', callCount: 1, thread: 'main' },
    { name: 'processData()', selfTime: '234ms', totalTime: '567ms', callCount: 3, thread: 'main' },
    { name: 'parseInput()', selfTime: '123ms', totalTime: '123ms', callCount: 15, thread: 'main' },
    { name: 'validateData()', selfTime: '89ms', totalTime: '210ms', callCount: 15, thread: 'main' },
    { name: 'compute()', selfTime: '345ms', totalTime: '456ms', callCount: 1, thread: 'worker_1' },
    { name: 'optimizeAlgo()', selfTime: '456ms', totalTime: '456ms', callCount: 5, thread: 'worker_2' },
  ];

  const filteredData = tableData.filter(row => 
    row.name.toLowerCase().includes(filterText.toLowerCase())
  );

  return (
    <div className={`relative w-full h-full bg-gray-900 ${className || ''}`}>
      {/* Data Table */}
      <div className="w-full h-full overflow-auto p-4">
        <div className="bg-gray-800 rounded-lg overflow-hidden">
          <table className="w-full text-white">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left">Function Name</th>
                <th className="px-4 py-3 text-left">Self Time</th>
                <th className="px-4 py-3 text-left">Total Time</th>
                <th className="px-4 py-3 text-left">Call Count</th>
                <th className="px-4 py-3 text-left">Thread</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.map((row, index) => (
                <tr key={index} className="border-t border-gray-600 hover:bg-gray-700/50">
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
        
        {filteredData.length === 0 && (
          <div className="text-center text-gray-400 mt-8">
            No functions match the filter criteria
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
                <select className="w-full mt-1 px-3 py-2 bg-gray-700 rounded border border-gray-600 text-white">
                  <option value="100">Top 100</option>
                  <option value="500">Top 500</option>
                  <option value="all">All Functions</option>
                </select>
              </label>
            </div>

            {/* Export */}
            <button className="w-full bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm transition-colors">
              Export CSV
            </button>
          </div>
        )}
      </div>
    </div>
  );
}; 