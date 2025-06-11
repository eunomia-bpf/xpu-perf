// Store composition for backward compatibility
import { useDataStore } from './dataStore';
import { useConfigStore } from './configStore';
import { useInteractionStore } from './interactionStore';

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
export { useDataStore } from './dataStore';
export { useConfigStore } from './configStore';
export { useInteractionStore } from './interactionStore'; 