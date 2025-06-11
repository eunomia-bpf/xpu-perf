import React from 'react';

interface DataTableViewProps {
  className?: string;
}

export const DataTableView: React.FC<DataTableViewProps> = ({ className }) => {
  return (
    <div className={`relative w-full h-full bg-gray-900 ${className || ''}`}>
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-white text-center">
          <div className="text-4xl mb-4">📋</div>
          <h2 className="text-2xl font-bold mb-2">Data Table</h2>
          <p className="text-gray-400">Tabular data view with search and export</p>
          <p className="text-sm text-gray-500 mt-4">Mock implementation</p>
        </div>
      </div>
      
      {/* Simple Controls Panel */}
      <div className="absolute bottom-4 right-4">
        <div className="bg-black/80 backdrop-blur-md text-white p-4 rounded-lg border border-white/10">
          <h4 className="text-lg font-semibold mb-3">Table Controls</h4>
          <div className="space-y-2">
            <button className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm">Filter</button>
            <button className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm">Export CSV</button>
          </div>
        </div>
      </div>
    </div>
  );
}; 