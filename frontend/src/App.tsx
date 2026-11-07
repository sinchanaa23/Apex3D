import { Suspense, lazy, type ReactNode } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { ErrorBoundary } from './components/ErrorBoundary'
import { Layout } from './components/Layout'
import { PageShell } from './components/hud/PageShell'
import { EmptyState, PageLoading } from './components/States'
import Auth from './pages/Auth'
import Classical from './pages/Classical'
import Home from './pages/Home'
import Identity from './pages/Identity'
import References from './pages/References'
import Upload from './pages/Upload'
import { useAuth } from './state/auth'

// three.js is the bulk of the bundle, so phones only download it when they open the studio.
const Studio = lazy(() => import('./pages/Studio'))
const Lab = lazy(() => import('./pages/Lab'))

const shell = (page: ReactNode) => <PageShell>{page}</PageShell>

/** Pages that run the models need a login. The landing page, the Research Lab and the account pages stay open. */
function RequireAuth({ children }: { children: ReactNode }) {
  const { loading, user, loginRequired } = useAuth()
  const location = useLocation()
  if (loading) return <PageLoading label="Checking your session" />
  if (loginRequired && !user) return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />
  return <>{children}</>
}
const guarded = (page: ReactNode) => shell(<RequireAuth>{page}</RequireAuth>)

export default function App() {
  const { pathname } = useLocation()
  return (
    <Layout>
      <ErrorBoundary key={pathname} title="This page hit a problem">
        <Suspense fallback={<PageLoading label="Loading page" />}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={shell(<Auth mode="login" />)} />
            <Route path="/signup" element={shell(<Auth mode="signup" />)} />
            <Route path="/upload" element={guarded(<Upload />)} />
            <Route path="/identity" element={guarded(<Identity />)} />
            <Route path="/classical" element={guarded(<Classical />)} />
            <Route path="/references" element={guarded(<References />)} />
            <Route path="/studio" element={guarded(<Studio />)} />
            <Route path="/studio/:jobId" element={guarded(<Studio />)} />
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
