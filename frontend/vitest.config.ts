import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
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
  test: {
    environment: 'jsdom',
    setupFiles: ['src/test/setup.ts'],
    globals: true
  }
}) 