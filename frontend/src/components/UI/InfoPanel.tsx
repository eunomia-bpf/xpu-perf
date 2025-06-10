import React from 'react';
import { useFlameGraphStore } from '@/stores';

interface InfoPanelProps {
  className?: string;
}

export const InfoPanel: React.FC<InfoPanelProps> = ({ className }) => {
  const { hoveredBlock, stats, data } = useFlameGraphStore();

  const getTotalSamples = (threadName: string): number => {
    const threadData = data[threadName] || [];
    return threadData.reduce((sum, entry) => sum + entry.count, 0);
  };

  const getPercentage = (count: number, threadName: string): string => {
    const total = getTotalSamples(threadName);
    return total > 0 ? ((count / total) * 100).toFixed(1) : '0.0';
  };

  return (
    <div className={`absolute top-3 left-3 text-white bg-black/70 backdrop-blur-md p-4 rounded-lg min-w-72 font-sans border border-white/10 z-30 ${className || ''}`}>
      <h3 className="text-lg font-semibold text-white mb-4 mt-0">3D Flame Graph Visualizer</h3>
      
      <div className="mb-4 leading-relaxed">
        {hoveredBlock ? (
          <>
            <div className="my-1"><strong>Thread:</strong> {hoveredBlock.threadName}</div>
            <div className="my-1"><strong>Function:</strong> {hoveredBlock.funcName}</div>
            <div className="my-1"><strong>Count:</strong> {hoveredBlock.count} ({getPercentage(hoveredBlock.count, hoveredBlock.threadName)}%)</div>
            <div className="my-1"><strong>Depth:</strong> {hoveredBlock.depth}</div>
            <div className="my-1"><strong>Width:</strong> {hoveredBlock.width.toFixed(1)}</div>
          </>
        ) : (
          <div>Hover over blocks for details</div>
        )}
      </div>

      <div className="text-xs text-gray-300 leading-relaxed">
        <strong>Thread Statistics:</strong>
        {Object.entries(stats).map(([threadName, stat]) => (
          <div key={threadName} className="my-1">
            <strong>{threadName}:</strong> {stat.totalSamples} samples, {stat.maxDepth} max depth
          </div>
        ))}
      </div>
    </div>
  );
}; 