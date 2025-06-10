import { ColorScheme } from '@/types/flame.types';

export const COLOR_SCHEMES: ColorScheme[] = [
  {
    name: 'Warm',
    colors: ['#ff6b6b', '#ffa726', '#ffee58', '#66bb6a', '#42a5f5', '#ab47bc', '#ec407a', '#26a69a']
  },
  {
    name: 'Cool',
    colors: ['#26c6da', '#29b6f6', '#5c6bc0', '#7e57c2', '#8d6e63', '#78909c', '#546e7a', '#5d4037']
  },
  {
    name: 'Vibrant',
    colors: ['#f44336', '#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#00bcd4', '#009688']
  },
  {
    name: 'Pastel',
    colors: ['#ffcdd2', '#f8bbd9', '#e1bee7', '#d1c4e9', '#c5cae9', '#bbdefb', '#b3e5fc', '#b2dfdb']
  }
];

/**
 * Get color from scheme based on function name hash
 */
export function getColorForFunction(functionName: string, schemeIndex: number): string {
  const scheme = COLOR_SCHEMES[schemeIndex] ?? COLOR_SCHEMES[0]!;
  const hash = Math.abs(
    functionName.split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0);
      return a & a;
    }, 0)
  );
  return scheme.colors[hash % scheme.colors.length] ?? scheme.colors[0]!;
}

/**
 * Get all available color scheme names
 */
export function getColorSchemeNames(): string[] {
  return COLOR_SCHEMES.map(scheme => scheme.name);
}

/**
 * Get color scheme by index
 */
export function getColorScheme(index: number): ColorScheme {
  return COLOR_SCHEMES[index] || COLOR_SCHEMES[0]!;
} 