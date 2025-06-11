import { useEffect } from 'react';
import { AppShell, ViewProvider } from '@/LayoutManager';
import { DynamicAnalyzer } from '@/AnalyzerEngine';
import { DataSourceSelector, ViewSelector } from '@/ControlCenter';
import { ViewportManager } from '@/ViewportEngine';
import { useFlameGraphStore, useDataSourceStore } from '@/DataManager';

function App() {
  const { loadSampleData, isLoading, error, data } = useFlameGraphStore();
  const { setCurrentDataDirect } = useDataSourceStore();

  // Load sample data on app startup and set it as current data
  useEffect(() => {
    loadSampleData();
  }, [loadSampleData]);

  // Set sample data as current data context when available
  useEffect(() => {
    if (Object.keys(data).length > 0) {
      setCurrentDataDirect(data, 'sample', ['key', 'value']);
    }
  }, [data, setCurrentDataDirect]);

  return (
    <ViewProvider>
      <AppShell 
        sidebar={
          <div className="space-y-0">
            <DynamicAnalyzer />
            <DataSourceSelector />
            <ViewSelector />
          </div>
        }
      >
        {/* Error display */}
        {error && (
          <div className="fixed top-4 right-4 bg-red-600 text-white p-4 rounded-lg shadow-lg z-50 max-w-md">
            <div className="flex items-start space-x-3">
              <span className="text-xl">⚠️</span>
              <div className="flex-1">
                <h4 className="font-semibold mb-1">Error</h4>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Loading indicator */}
        {isLoading && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-gray-800 rounded-lg p-6 flex flex-col items-center space-y-4">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
              <span className="text-white text-lg">Loading profiler data...</span>
            </div>
          </div>
        )}

        {/* Main viewport with data-focused design */}
        <ViewportManager />
      </AppShell>
    </ViewProvider>
  );
}

export default App;
