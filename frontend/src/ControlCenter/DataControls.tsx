import React from 'react';

export const DataControls: React.FC = () => {
  return (
    <div className="bg-gray-700 rounded-lg p-4 space-y-4">
      {/* Sliders */}
      <div className="space-y-3">
        <div>
          <label className="block text-sm text-gray-300 mb-1">Z-Spacing: 25</label>
          <input 
            type="range" 
            min="10" 
            max="50" 
            defaultValue="25" 
            className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
          />
        </div>
        
        <div>
          <label className="block text-sm text-gray-300 mb-1">Min Count: 10</label>
          <input 
            type="range" 
            min="1" 
            max="100" 
            defaultValue="10" 
            className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
          />
        </div>
        
        <div>
          <label className="block text-sm text-gray-300 mb-1">Max Depth: 8</label>
          <input 
            type="range" 
            min="1" 
            max="20" 
            defaultValue="8" 
            className="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      {/* Dropdowns */}
      <div className="space-y-2">
        <div>
          <label className="block text-sm text-gray-300 mb-1">Color:</label>
          <select className="w-full bg-gray-600 text-white rounded px-3 py-2">
            <option>Hot/Cold</option>
            <option>Thread-based</option>
            <option>Function-based</option>
          </select>
        </div>
        
        <div>
          <label className="block text-sm text-gray-300 mb-1">Threads:</label>
          <select className="w-full bg-gray-600 text-white rounded px-3 py-2">
            <option>All</option>
            <option>Main Thread</option>
            <option>Worker Threads</option>
          </select>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex space-x-2">
        <button className="flex-1 px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm transition-colors flex items-center justify-center space-x-1">
          <span>🎯</span>
          <span>Reset Camera</span>
        </button>
        <button className="flex-1 px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded text-sm transition-colors flex items-center justify-center space-x-1">
          <span>📐</span>
          <span>Fit All</span>
        </button>
      </div>
    </div>
  );
}; 