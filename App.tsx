import { useEffect } from 'react';
import { AppLayout } from '@/LayoutManager/AppShell';
import { AnalyzerControlPanel } from '@/ControlCenter/AnalyzerControls';
import { ViewportContainer } from '@/ViewportEngine';
import { LoadingSpinner, ErrorDisplay } from '@/components/UI/shared';
import { useFlameGraphStore } from '@/DataManager/DataStore';

function App() {
  const { loadSampleData, isLoading, error } = useFlameGraphStore();

  // Load sample data on app startup
  useEffect(() => {
    loadSampleData();
  }, [loadSampleData]);

  return (
    <AppLayout sidebar={<AnalyzerControlPanel />}>
      {/* Loading indicator */}
      {isLoading && <LoadingSpinner message="Loading profiler data..." />}

      {/* Error display */}
      {error && <ErrorDisplay error={error} />}

      {/* Main viewport with self-contained views */}
      <ViewportContainer />
    </AppLayout>
  );
}

export default App; 