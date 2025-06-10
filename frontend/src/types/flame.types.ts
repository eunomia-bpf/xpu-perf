export interface FlameStackEntry {
  stack: string[];
  count: number;
}

export interface FlameTreeNode {
  count: number;
  children: Record<string, FlameTreeNode>;
}

export interface FlameData {
  [threadName: string]: FlameStackEntry[];
}

export interface FileInfo {
  path: string;
  name: string;
}

export interface ThreadStats {
  totalSamples: number;
  maxCount: number;
  avgCount: number;
  stackTraces: number;
  uniqueFunctions: number;
  maxDepth: number;
}

export interface FlameBlockMetadata {
  funcName: string;
  count: number;
  threadName: string;
  depth: number;
  originalColor: THREE.Color;
  width: number;
}

export interface ColorScheme {
  name: string;
  colors: string[];
}

export interface FlameGraphConfig {
  zSpacing: number;
  minCount: number;
  maxDepth: number;
  colorSchemeIndex: number;
  autoRotate: boolean;
}

export interface CameraPosition {
  x: number;
  y: number;
  z: number;
}

export interface FlameGraphState {
  data: FlameData;
  config: FlameGraphConfig;
  stats: Record<string, ThreadStats>;
  isLoading: boolean;
  error: string | null;
  hoveredBlock: FlameBlockMetadata | null;
} 