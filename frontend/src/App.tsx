import { Suspense, lazy } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Layout } from './components/Layout'
import { EmptyState, PageLoading } from './components/States'
import Classical from './pages/Classical'
import Home from './pages/Home'
import Identity from './pages/Identity'
import References from './pages/References'
import Upload from './pages/Upload'

// three.js is the bulk of the bundle, so phones only download it when they open the studio.
const Studio = lazy(() => import('./pages/Studio'))
const Lab = lazy(() => import('./pages/Lab'))

export default function App() {
  const { pathname } = useLocation()
  return (
    <Layout>
      <ErrorBoundary key={pathname} title="This page hit a problem">
        <Suspense fallback={<PageLoading label="Loading page" />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/identity" element={<Identity />} />
            <Route path="/classical" element={<Classical />} />
            <Route path="/references" element={<References />} />
            <Route path="/studio" element={<Studio />} />
            <Route path="/studio/:jobId" element={<Studio />} />
            <Route path="/lab" element={<Lab />} />
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
