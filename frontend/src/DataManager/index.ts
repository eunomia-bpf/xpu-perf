// Store composition for backward compatibility
import { useDataStore } from './DataStore/dataStore';
import { useConfigStore } from './DataStore/configStore';
import { useInteractionStore } from './DataStore/interactionStore';

// Composite hook that maintains the original flameGraphStore interface
export const useFlameGraphStore = () => {
  const dataStore = useDataStore();
  const configStore = useConfigStore();
  const interactionStore = useInteractionStore();
  
  return {
    // Data state and actions
    ...dataStore,
    
    // Config state and actions
    ...configStore,
    
    // Interaction state and actions
    ...interactionStore
  };
};

// Export individual stores for new modular usage
export { useDataStore } from './DataStore/dataStore';
export { useConfigStore } from './DataStore/configStore';
export { useInteractionStore } from './DataStore/interactionStore';

// Export new dynamic analyzer store
export { useAnalyzerStore } from './DataStore/analyzerStore'; 