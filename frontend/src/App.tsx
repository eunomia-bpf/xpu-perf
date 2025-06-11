import { useEffect } from 'react';
import { AppLayout } from '@/components/Layout';
import { AnalyzerControlPanel } from '@/components/analyzers/AnalyzerControlPanel';
import { ViewportContainer } from '@/components/views/ViewportContainer';
import { LoadingSpinner, ErrorDisplay } from '@/components/UI/shared';
import { useFlameGraphStore } from '@/stores';

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
