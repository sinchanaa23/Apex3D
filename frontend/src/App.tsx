import { Suspense, lazy, type ReactNode } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Layout } from './components/Layout'
import { PageShell } from './components/hud/PageShell'
import { EmptyState, PageLoading } from './components/States'
import Classical from './pages/Classical'
import Home from './pages/Home'
import Identity from './pages/Identity'
import References from './pages/References'
import Upload from './pages/Upload'

// three.js is the bulk of the bundle, so phones only download it when they open the studio.
const Studio = lazy(() => import('./pages/Studio'))
const Lab = lazy(() => import('./pages/Lab'))

const shell = (page: ReactNode) => <PageShell>{page}</PageShell>

export default function App() {
  const { pathname } = useLocation()
  return (
    <Layout>
      <ErrorBoundary key={pathname} title="This page hit a problem">
        <Suspense fallback={<PageLoading label="Loading page" />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={shell(<Upload />)} />
            <Route path="/identity" element={shell(<Identity />)} />
            <Route path="/classical" element={shell(<Classical />)} />
            <Route path="/references" element={shell(<References />)} />
            <Route path="/studio" element={shell(<Studio />)} />
            <Route path="/studio/:jobId" element={shell(<Studio />)} />
            <Route path="/lab" element={shell(<Lab />)} />
            <Route
              path="*"
              element={<EmptyState title="Page not found" body="That address isn't part of Apex3D." action={{ to: '/', label: 'Go to home' }} />}
            />
          </Routes>
        </Suspense>
      </ErrorBoundary>
    </Layout>
  )
}
