import { useEffect } from 'react';
import { FlameGraph3D } from '@/components/FlameGraph3D/FlameGraph3D';
import { AppLayout } from '@/components/Layout';
import { ControlPanelLayout } from '@/components/Controls/ControlPanelLayout';
import { InfoPanelLayout } from '@/components/UI/InfoPanelLayout';
import { LoadingSpinner, ErrorDisplay } from '@/components/UI/shared';
import { useFlameGraphStore } from '@/stores';

function App() {
  const { loadSampleData, isLoading, error } = useFlameGraphStore();

  // Load sample data on app startup
  useEffect(() => {
    loadSampleData();
  }, [loadSampleData]);

  return (
    <AppLayout sidebar={<ControlPanelLayout />}>
      {/* Loading indicator */}
      {isLoading && <LoadingSpinner message="Loading 3D Flame Graph..." />}

      {/* Error display */}
      {error && <ErrorDisplay error={error} />}

      {/* Main 3D visualization */}
      <FlameGraph3D />

      {/* Info panel overlay */}
      <InfoPanelLayout />
    </AppLayout>
  );
}

export default App;
