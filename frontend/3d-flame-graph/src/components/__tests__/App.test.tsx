import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from '../../App';

// Mock Three.js components
vi.mock('@react-three/fiber', () => ({
  Canvas: ({ children }: { children: React.ReactNode }) => <div data-testid="canvas">{children}</div>,
  useFrame: vi.fn(),
}));

vi.mock('@react-three/drei', () => ({
  OrbitControls: () => <div data-testid="orbit-controls" />,
  Environment: () => <div data-testid="environment" />,
  Text: ({ children }: { children: React.ReactNode }) => <div data-testid="text">{children}</div>,
}));

// Mock the store
vi.mock('../../stores/flameGraphStore', () => ({
  useFlameGraphStore: () => ({
    loadSampleData: vi.fn(),
    isLoading: false,
    error: null,
    data: {},
    config: {
      zSpacing: 25,
      minCount: 10,
      maxDepth: 8,
      colorSchemeIndex: 0,
      autoRotate: false
    },
    stats: {},
    hoveredBlock: null,
    setHoveredBlock: vi.fn(),
    resetView: vi.fn(),
    toggleAutoRotate: vi.fn(),
    changeColorScheme: vi.fn(),
    updateZSpacing: vi.fn(),
    updateMinCount: vi.fn(),
    updateMaxDepth: vi.fn(),
    updateStats: vi.fn()
  })
}));

describe('App', () => {
  it('should render without crashing', () => {
    render(<App />);
    
    expect(screen.getByTestId('canvas')).toBeInTheDocument();
  });

  it('should render control panels', () => {
    render(<App />);
    
    expect(screen.getByText('3D Flame Graph Visualizer')).toBeInTheDocument();
    expect(screen.getByText('Reset View')).toBeInTheDocument();
    expect(screen.getByText('Load Sample Data')).toBeInTheDocument();
  });

  it('should render range controls', () => {
    render(<App />);
    
    expect(screen.getByText(/Z-Spacing:/)).toBeInTheDocument();
    expect(screen.getByText(/Min Count:/)).toBeInTheDocument();
    expect(screen.getByText(/Max Depth:/)).toBeInTheDocument();
  });
}); 