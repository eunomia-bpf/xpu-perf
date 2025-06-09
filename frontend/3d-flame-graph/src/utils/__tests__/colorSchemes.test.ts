import { describe, it, expect } from 'vitest';
import { getColorForFunction, getColorSchemeNames, getColorScheme, COLOR_SCHEMES } from '../colorSchemes';

describe('colorSchemes', () => {
  describe('getColorForFunction', () => {
    it('should return consistent color for same function name', () => {
      const color1 = getColorForFunction('test_function', 0);
      const color2 = getColorForFunction('test_function', 0);
      
      expect(color1).toBe(color2);
    });

    it('should return different colors for different functions', () => {
      const color1 = getColorForFunction('function1', 0);
      const color2 = getColorForFunction('function2', 0);
      
      expect(color1).not.toBe(color2);
    });

    it('should return valid hex color', () => {
      const color = getColorForFunction('test_function', 0);
      
      expect(color).toMatch(/^#[0-9A-Fa-f]{6}$/);
    });

    it('should handle invalid scheme index gracefully', () => {
      const color = getColorForFunction('test_function', 999);
      
      expect(color).toMatch(/^#[0-9A-Fa-f]{6}$/);
    });
  });

  describe('getColorSchemeNames', () => {
    it('should return array of scheme names', () => {
      const names = getColorSchemeNames();
      
      expect(Array.isArray(names)).toBe(true);
      expect(names.length).toBeGreaterThan(0);
      expect(names).toContain('Warm');
      expect(names).toContain('Cool');
    });
  });

  describe('getColorScheme', () => {
    it('should return valid color scheme for valid index', () => {
      const scheme = getColorScheme(0);
      
      expect(scheme).toHaveProperty('name');
      expect(scheme).toHaveProperty('colors');
      expect(Array.isArray(scheme.colors)).toBe(true);
    });

    it('should return first scheme for invalid index', () => {
      const scheme = getColorScheme(999);
      const firstScheme = COLOR_SCHEMES[0];
      
      expect(scheme).toEqual(firstScheme);
    });
  });

  describe('COLOR_SCHEMES', () => {
    it('should have expected structure', () => {
      expect(Array.isArray(COLOR_SCHEMES)).toBe(true);
      expect(COLOR_SCHEMES.length).toBeGreaterThan(0);
      
      COLOR_SCHEMES.forEach(scheme => {
        expect(scheme).toHaveProperty('name');
        expect(scheme).toHaveProperty('colors');
        expect(typeof scheme.name).toBe('string');
        expect(Array.isArray(scheme.colors)).toBe(true);
        expect(scheme.colors.length).toBeGreaterThan(0);
        
        scheme.colors.forEach(color => {
          expect(color).toMatch(/^#[0-9A-Fa-f]{6}$/);
        });
      });
    });
  });
}); 