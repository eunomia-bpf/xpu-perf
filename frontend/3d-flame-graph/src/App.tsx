import React, { useEffect } from 'react';
import { FlameGraph3D } from '@/components/FlameGraph3D/FlameGraph3D';
import { InfoPanel } from '@/components/UI/InfoPanel';
import { ControlPanel } from '@/components/Controls/ControlPanel';
import { useFlameGraphStore } from '@/stores/flameGraphStore';

function App() {
  const { loadSampleData, isLoading, error } = useFlameGraphStore();

  // Load sample data on app startup
  useEffect(() => {
    loadSampleData();
  }, [loadSampleData]);

  return (
    <div className="relative w-screen h-screen bg-gray-900 overflow-hidden">
      {/* Loading indicator */}
      {isLoading && (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white text-lg z-50 bg-black/80 px-10 py-5 rounded-lg text-center backdrop-blur-md">
          Loading 3D Flame Graph...
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-red-400 text-base z-50 bg-black/90 px-10 py-5 rounded-lg text-center border border-red-400 max-w-4/5">
          Error: {error}
        </div>
      )}

      {/* Main 3D visualization */}
      <FlameGraph3D />

      {/* UI overlays */}
      <InfoPanel />
      <ControlPanel />
    </div>
  );
}

export default App;
