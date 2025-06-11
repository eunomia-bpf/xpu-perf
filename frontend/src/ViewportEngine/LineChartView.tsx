import React, { useState } from 'react';
import { useFlameGraphStore } from '@/stores';

interface LineChartViewProps {
  className?: string;
}

export const LineChartView: React.FC<LineChartViewProps> = ({ className }) => {
  const [showControls, setShowControls] = useState(false);
  const [timeRange, setTimeRange] = useState('all');
  const [chartType, setChartType] = useState<'line' | 'area' | 'bar'>('line');
  const { data } = useFlameGraphStore();

  return (
    <div className={`relative w-full h-full bg-gray-900 ${className || ''}`}>
      {/* Chart Canvas */}
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-white text-center">
          <h3 className="text-2xl font-bold mb-4">Line Chart View</h3>
          <p className="text-gray-400 mb-4">Time-series performance visualization</p>
          
          {/* Mock Chart Area */}
          <div className="bg-gray-800 p-8 rounded-lg w-96 h-64 flex items-center justify-center">
            <svg className="w-full h-full" viewBox="0 0 300 150">
              {/* Grid lines */}
              <defs>
                <pattern id="grid" width="30" height="15" patternUnits="userSpaceOnUse">
                  <path d="M 30 0 L 0 0 0 15" fill="none" stroke="#374151" strokeWidth="1"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
              
              {/* Sample line chart */}
              <path
                d="M 10 120 Q 50 80 90 100 T 170 60 T 250 90 L 290 70"
                fill="none"
                stroke="#3B82F6"
                strokeWidth="2"
              />
              
              {/* Sample data points */}
              <circle cx="50" cy="100" r="3" fill="#3B82F6" />
              <circle cx="90" cy="80" r="3" fill="#3B82F6" />
              <circle cx="130" cy="60" r="3" fill="#3B82F6" />
              <circle cx="170" cy="90" r="3" fill="#3B82F6" />
              <circle cx="210" cy="70" r="3" fill="#3B82F6" />
              <circle cx="250" cy="90" r="3" fill="#3B82F6" />
            </svg>
          </div>
          
          <div className="text-sm text-gray-400 mt-4">
            Data points: {Object.keys(data).length} | Type: {chartType} | Range: {timeRange}
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
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
        </button>

        {/* Expandable Controls Panel */}
        {showControls && (
          <div className="bg-black/80 backdrop-blur-md text-white p-4 rounded-lg min-w-64 border border-white/10 space-y-3">
            <h4 className="text-lg font-semibold mb-3">Chart Controls</h4>
            
            {/* Chart Type */}
            <div>
              <label className="block text-sm mb-1">
                Chart Type
                <select 
                  className="w-full mt-1 px-3 py-2 bg-gray-700 rounded border border-gray-600 text-white"
                  value={chartType}
                  onChange={(e) => setChartType(e.target.value as any)}
                >
                  <option value="line">Line Chart</option>
                  <option value="area">Area Chart</option>
                  <option value="bar">Bar Chart</option>
                </select>
              </label>
            </div>

            {/* Time Range */}
            <div>
              <label className="block text-sm mb-1">
                Time Range
                <select 
                  className="w-full mt-1 px-3 py-2 bg-gray-700 rounded border border-gray-600 text-white"
                  value={timeRange}
                  onChange={(e) => setTimeRange(e.target.value)}
                >
                  <option value="all">All Time</option>
                  <option value="1h">Last Hour</option>
                  <option value="1d">Last Day</option>
                  <option value="1w">Last Week</option>
                </select>
              </label>
            </div>

            {/* Y-Axis Scale */}
            <div>
              <label className="block text-sm mb-1">
                Y-Axis Scale
                <select className="w-full mt-1 px-3 py-2 bg-gray-700 rounded border border-gray-600 text-white">
                  <option value="linear">Linear</option>
                  <option value="log">Logarithmic</option>
                </select>
              </label>
            </div>

            {/* Navigation Controls */}
            <div className="grid grid-cols-3 gap-2">
              <button className="bg-gray-700 hover:bg-gray-600 px-2 py-2 rounded text-sm transition-colors">
                ◀◀
              </button>
              <button className="bg-gray-700 hover:bg-gray-600 px-2 py-2 rounded text-sm transition-colors">
                ⏸
              </button>
              <button className="bg-gray-700 hover:bg-gray-600 px-2 py-2 rounded text-sm transition-colors">
                ▶▶
              </button>
            </div>

            {/* Export */}
            <button className="w-full bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm transition-colors">
              Export Chart
            </button>
          </div>
        )}
      </div>
    </div>
  );
}; 