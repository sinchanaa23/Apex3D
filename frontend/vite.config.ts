import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, type Plugin } from 'vitest/config'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { mockApi } from './dev/mockApi'

// Person A's saved result files, from the repo's results/ folder. Only these are exposed.
const RESULT_FILES = ['classifier_metrics.json', 'optimization_metrics.json']
const RESULTS_DIR = resolve(__dirname, '../results')

/** Serves the result files in development and copies them into the build, so the Research Lab can read them. */
function savedResults(): Plugin {
  return {
    name: 'apex3d-saved-results',
    configureServer(server) {
      server.middlewares.use('/results', (req, res, next) => {
        const name = (req.url ?? '').replace(/^\//, '').split('?')[0]
        const file = resolve(RESULTS_DIR, name)
        if (!RESULT_FILES.includes(name) || !existsSync(file)) return next()
        res.setHeader('content-type', 'application/json')
        res.end(readFileSync(file))
      })
    },
    closeBundle() {
      const out = resolve(__dirname, 'dist/results')
      for (const name of RESULT_FILES) {
        const file = resolve(RESULTS_DIR, name)
        if (!existsSync(file)) continue
        mkdirSync(out, { recursive: true })
        copyFileSync(file, resolve(out, name))
      }
    },
  }
}

// `npm run dev`       proxies /api to Person C's FastAPI service.
// `npm run dev:mock`  answers /api from dev/mockApi.ts. The plugin only runs on
//                     the dev server, so test responses cannot reach `npm run build`.
export default defineConfig(({ mode }) => ({
  plugins: [react(), tailwindcss(), savedResults(), ...(mode === 'mock' ? [mockApi()] : [])],
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
