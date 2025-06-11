// Store composition for backward compatibility
import { useDataStore } from './core/dataStore';
import { useConfigStore } from './core/configStore';
import { useInteractionStore } from './ui/interactionStore';

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
export { useDataStore } from './core/dataStore';
export { useConfigStore } from './core/configStore';
export { useInteractionStore } from './ui/interactionStore'; 