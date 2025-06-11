// Mock data store for architecture-first implementation
interface DataState {
  isLoading: boolean;
  error: string | null;
  data: any;
}

interface DataActions {
  loadSampleData: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

// Simple mock implementation
export const useDataStore = (): DataState & DataActions => ({
  isLoading: false,
  error: null,
  data: {},
  
  loadSampleData: () => {
    console.log('Mock: Loading sample data...');
  },
  
  setLoading: (loading: boolean) => {
    console.log('Mock: Set loading', loading);
  },
  
  setError: (error: string | null) => {
    console.log('Mock: Set error', error);
  }
}); 