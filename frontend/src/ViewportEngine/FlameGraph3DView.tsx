import React, { Suspense, useRef, useEffect, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useFlameGraphStore } from '@/DataManager/DataStore';
import { FlameGraphDataLoader } from '@/DataManager/DataProcessor/flameDataLoader';
import { FlameGraphContent } from './FlameGraph3D/FlameGraphContent';
import { LightingSystem } from './FlameGraph3D/LightingSystem';
import { getColorSchemeNames } from '@/DataManager/DataProcessor/colorSchemes';

interface FlameGraph3DViewProps {
  className?: string;
}

export const FlameGraph3DView: React.FC<FlameGraph3DViewProps> = ({ className }) => {
  const controlsRef = useRef<any>(null);
  const [showControls, setShowControls] = useState(false);
  
  const {
    data,
    config,
    hoveredBlock,
    stats,
    setHoveredBlock,
    updateStats,
    resetView,
    toggleAutoRotate,
    changeColorScheme,
    updateZSpacing,
    updateMinCount,
    updateMaxDepth,
    loadSampleData
  } = useFlameGraphStore();

  const colorSchemeNames = getColorSchemeNames();

  // Update statistics when data changes
  useEffect(() => {
    if (Object.keys(data).length > 0) {
      const loader = new FlameGraphDataLoader();
      const stats = loader.getSummaryStats(data);
      updateStats(stats);
    }
  }, [data, updateStats]);

  // Handle auto-rotation
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.autoRotate = config.autoRotate;
      controlsRef.current.autoRotateSpeed = 0.5;
    }
  }, [config.autoRotate]);

  const handlePointerMissed = () => {
    setHoveredBlock(null);
  };

  const getTotalSamples = (threadName: string): number => {
    const threadData = data[threadName] || [];
    return threadData.reduce((sum, entry) => sum + entry.count, 0);
  };

  const getPercentage = (count: number, threadName: string): string => {
    const total = getTotalSamples(threadName);
    return total > 0 ? ((count / total) * 100).toFixed(1) : '0.0';
  };

  const cameraPosition = new THREE.Vector3(60, 40, 60);

  return (
    <div className={`relative w-full h-full ${className || ''}`} style={{ background: '#111111' }}>
      {/* 3D Canvas */}
      <Canvas
        camera={{ 
          position: cameraPosition,
          fov: 75,
          near: 0.1,
          far: 1000
        }}
        shadows
        onPointerMissed={handlePointerMissed}
      >
        <Suspense fallback={null}>
          {/* Lighting System */}
          <LightingSystem />
          
          {/* Controls */}
          <OrbitControls
            ref={controlsRef}
            enableDamping
            dampingFactor={0.05}
            maxDistance={200}
            minDistance={10}
            autoRotate={config.autoRotate}
            autoRotateSpeed={0.5}
          />
          
          {/* Flame Graph Content */}
          <FlameGraphContent />
        </Suspense>
      </Canvas>

      {/* Hover Information Panel */}
      <div className="absolute top-4 left-4 text-white bg-black/70 backdrop-blur-md p-4 rounded-lg min-w-72 font-sans border border-white/10 z-30">
        <h3 className="text-lg font-semibold text-white mb-4 mt-0">3D Flame Graph</h3>
        
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
          <div className="bg-black/80 backdrop-blur-md text-white p-4 rounded-lg min-w-64 border border-white/10 space-y-3">
            <h4 className="text-lg font-semibold mb-3">3D Controls</h4>
            
            {/* Quick Actions */}
            <div className="grid grid-cols-2 gap-2">
              <button 
                className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm transition-colors"
                onClick={resetView}
              >
                Reset View
              </button>
              <button 
                className="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm transition-colors"
                onClick={toggleAutoRotate}
              >
                {config.autoRotate ? 'Stop Rotation' : 'Auto Rotate'}
              </button>
            </div>

            {/* Color Scheme */}
            <div>
              <button 
                className="w-full bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm transition-colors"
                onClick={changeColorScheme}
              >
                Colors: {colorSchemeNames[config.colorSchemeIndex]}
              </button>
            </div>

            {/* Sliders */}
            <div className="space-y-3">
              <div>
                <label className="block text-sm mb-1">
                  Z-Spacing: {config.zSpacing}
                  <input
                    className="w-full mt-1 accent-blue-500"
                    type="range"
                    min="5"
                    max="50"
                    value={config.zSpacing}
                    onChange={(e) => updateZSpacing(Number(e.target.value))}
                  />
                </label>
              </div>
              
              <div>
                <label className="block text-sm mb-1">
                  Min Count: {config.minCount}
                  <input
                    className="w-full mt-1 accent-blue-500"
                    type="range"
                    min="1"
                    max="100"
                    value={config.minCount}
                    onChange={(e) => updateMinCount(Number(e.target.value))}
                  />
                </label>
              </div>
              
              <div>
                <label className="block text-sm mb-1">
                  Max Depth: {config.maxDepth}
                  <input
                    className="w-full mt-1 accent-blue-500"
                    type="range"
                    min="3"
                    max="15"
                    value={config.maxDepth}
                    onChange={(e) => updateMaxDepth(Number(e.target.value))}
                  />
                </label>
              </div>
            </div>

            {/* Data Actions */}
            <div>
              <button 
                className="w-full bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm transition-colors"
                onClick={loadSampleData}
              >
                Reload Data
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 