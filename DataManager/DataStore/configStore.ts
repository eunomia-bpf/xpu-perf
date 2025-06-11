// Mock config store for architecture-first implementation
interface ConfigState {
  config: any;
}

interface ConfigActions {
  updateConfig: (config: any) => void;
}

// Simple mock implementation
export const useConfigStore = (): ConfigState & ConfigActions => ({
  config: {},
  
  updateConfig: (config: any) => {
    console.log('Mock: Update config', config);
  }
}); 