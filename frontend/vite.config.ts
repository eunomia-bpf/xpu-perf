import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: './',
  resolve: {
    alias: {
      '@': '/src',
      '@/AnalyzerEngine': '/src/AnalyzerEngine',
      '@/ViewportEngine': '/src/ViewportEngine',
      '@/ControlCenter': '/src/ControlCenter',
      '@/DataManager': '/src/DataManager',
      '@/LayoutManager': '/src/LayoutManager',
      '@/types': '/src/types'
    }
  },
  server: {
    port: 3000,
    open: true
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          'three': ['three'],
          'react-three': ['@react-three/fiber', '@react-three/drei'],
          'vendor': ['react', 'react-dom'],
          'utils': ['d3-scale', 'd3-array', 'zustand']
        }
      }
    }
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['src/test/setup.ts'],
    globals: true
  }
})
