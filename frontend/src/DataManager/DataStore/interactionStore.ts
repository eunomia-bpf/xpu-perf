import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { FlameBlockMetadata } from '@/types/flame.types';

interface InteractionState {
  hoveredBlock: FlameBlockMetadata | null;
}

interface InteractionActions {
  setHoveredBlock: (block: FlameBlockMetadata | null) => void;
  resetView: () => void;
}

const initialState: InteractionState = {
  hoveredBlock: null
};

export const useInteractionStore = create<InteractionState & InteractionActions>()(
  devtools(
    (set, _get) => ({
      ...initialState,

      setHoveredBlock: (hoveredBlock: FlameBlockMetadata | null) => set({ hoveredBlock }),

      resetView: () => set({ hoveredBlock: null })
    }),
    {
      name: 'interaction-store'
    }
  )
); 