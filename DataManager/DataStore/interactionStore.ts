// Mock interaction store for architecture-first implementation
interface InteractionState {
  hoveredBlock: any;
}

interface InteractionActions {
  setHoveredBlock: (block: any) => void;
}

// Simple mock implementation
export const useInteractionStore = (): InteractionState & InteractionActions => ({
  hoveredBlock: null,
  
  setHoveredBlock: (block: any) => {
    console.log('Mock: Set hovered block', block);
  }
}); 