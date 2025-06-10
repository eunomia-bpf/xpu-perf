import { useEffect } from 'react';
import { FlameGraph3D } from '@/components/FlameGraph3D/FlameGraph3D';
import { InfoPanel } from '@/components/UI/InfoPanel';
import { ControlPanel } from '@/components/Controls/ControlPanel';
import { AppLayout } from '@/components/Layout';
import { LoadingSpinner, ErrorDisplay } from '@/components/UI/shared';
import { useFlameGraphStore } from '@/stores';

function App() {
  const { loadSampleData, isLoading, error } = useFlameGraphStore();

  // Load sample data on app startup
  useEffect(() => {
    loadSampleData();
  }, [loadSampleData]);

  return (
    <AppLayout>
      {/* Loading indicator */}
      {isLoading && <LoadingSpinner message="Loading 3D Flame Graph..." />}

      {/* Error display */}
      {error && <ErrorDisplay error={error} />}

      {/* Main 3D visualization */}
      <FlameGraph3D />

      {/* UI overlays */}
      <InfoPanel />
      <ControlPanel />
    </AppLayout>
  );
}

export default App;
