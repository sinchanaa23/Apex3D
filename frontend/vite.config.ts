import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { mockApi } from './dev/mockApi'

// `npm run dev`       proxies /api to Person C's FastAPI service.
// `npm run dev:mock`  answers /api from dev/mockApi.ts. The plugin only runs on
//                     the dev server, so test responses cannot reach `npm run build`.
export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss(), ...(mode === 'mock' ? [mockApi()] : [])],
  server: {
    host: true,
    proxy: mode === 'mock' ? undefined : { '/api': process.env.APEX3D_API ?? 'http://127.0.0.1:8000' },
  },
  build: {
    chunkSizeWarningLimit: 1200,
  },
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
}))
