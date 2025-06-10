import React from 'react';
import { ControlPanel } from './ControlPanel';

interface ControlPanelLayoutProps {
  className?: string;
}

export const ControlPanelLayout: React.FC<ControlPanelLayoutProps> = ({ className }) => {
  return (
    <div className={`flex flex-col h-full ${className || ''}`}>
      {/* Data Controls Section */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Data Controls</h3>
        <ControlPanel className="!relative !top-0 !right-0 !bg-transparent !backdrop-blur-none !p-0 !border-none !min-w-full" />
      </div>

      {/* View Controls Section */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">View Controls</h3>
        <div className="space-y-3">
          <button className="w-full bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded text-sm transition-colors">
            2D View
          </button>
          <button className="w-full bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded text-sm transition-colors">
            3D View
          </button>
          <button className="w-full bg-gray-700 hover:bg-gray-600 text-white px-3 py-2 rounded text-sm transition-colors">
            Timeline View
          </button>
        </div>
      </div>

      {/* Filters Section */}
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Filters</h3>
        <div className="space-y-3">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Search Functions</label>
            <input 
              type="text" 
              placeholder="Filter by function name..."
              className="w-full bg-gray-700 text-white text-sm px-3 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs text-gray-400 mb-1">Thread Filter</label>
            <select className="w-full bg-gray-700 text-white text-sm px-3 py-2 rounded border border-gray-600 focus:border-blue-500 focus:outline-none">
              <option value="">All Threads</option>
              <option value="worker_0">worker_0</option>
              <option value="worker_1">worker_1</option>
              <option value="worker_2">worker_2</option>
            </select>
          </div>
        </div>
      </div>

      {/* Sessions Section */}
      <div className="flex-1 p-4">
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Recent Sessions</h3>
        <div className="space-y-2">
          <div className="bg-gray-700 p-3 rounded text-sm">
            <div className="text-gray-300 font-medium">sample_data</div>
            <div className="text-gray-400 text-xs">Active • 5 threads</div>
          </div>
          <div className="bg-gray-800 p-3 rounded text-sm border border-gray-700">
            <div className="text-gray-400">previous_session</div>
            <div className="text-gray-500 text-xs">2 hours ago</div>
          </div>
        </div>
      </div>
    </div>
  );
}; 