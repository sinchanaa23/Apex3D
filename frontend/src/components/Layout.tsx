import { useEffect, type ReactNode } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { Box, FlaskConical, Images, LogOut, ScanLine, ScanSearch, Upload } from 'lucide-react'
import { useAuth } from '../state/auth'

const NAV = [
  { to: '/upload', label: 'Upload', icon: Upload },
  { to: '/identity', label: 'Identity', icon: ScanSearch },
  { to: '/classical', label: 'Classical', icon: ScanLine },
  { to: '/references', label: 'References', icon: Images },
  { to: '/studio', label: '3D Studio', short: '3D', icon: Box },
  { to: '/lab', label: 'Research Lab', short: 'Lab', icon: FlaskConical },
]

export function Logo() {
  return (
    <Link to="/" className="flex min-h-11 items-center gap-2 font-semibold tracking-tight">
      <svg aria-hidden viewBox="0 0 32 32" className="size-6">
        <path d="M5 25C13 25 13 7 18 7s4 9 9 11" fill="none" stroke="var(--color-accent)" strokeWidth="3" strokeLinecap="round" />
      </svg>
      Apex3D
    </Link>
  )
}

/** Log in and sign up, or the user's name and a log out button. Hidden while the server answers, and when no login is required. */
function Account() {
  const { loading, user, loginRequired, logout } = useAuth()
  const link = 'rounded-lg px-3 py-2 text-sm transition-colors hover:text-ink'
  if (loading || !loginRequired) return <div className="w-10 md:w-0" />
  if (user) {
    return (
      <div className="flex items-center gap-1">
        <span className="hidden max-w-36 truncate text-sm text-ink-2 sm:inline" title={user.email}>
          {user.name}
        </span>
        <button type="button" onClick={() => void logout()} className={`${link} inline-flex min-h-10 items-center gap-1.5 text-ink-2`}>
          <LogOut aria-hidden className="size-4" />
          Log out
        </button>
      </div>
    )
  }
  return (
    <div className="flex items-center gap-1">
      <NavLink to="/login" className={({ isActive }) => `${link} ${isActive ? 'bg-raised text-ink' : 'text-ink-2'}`}>
        Log in
      </NavLink>
      <NavLink to="/signup" className={({ isActive }) => `${link} border border-line-strong ${isActive ? 'bg-raised text-ink' : 'text-ink'}`}>
        Sign up
      </NavLink>
    </div>
  )
}

export function Layout({ children }: { children: ReactNode }) {
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])

  return (
    <div className="flex min-h-dvh flex-col">
      {import.meta.env.MODE === 'mock' && (
        <p className="bg-warning px-4 py-1 text-center text-xs font-medium text-page">Dev mock: every result here is a test response.</p>
      )}
      <header className="sticky top-0 z-30 border-b border-line bg-page/90 backdrop-blur">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 px-4 sm:px-6">
          <Logo />
          <nav aria-label="Main" className="hidden gap-1 md:flex">
            {NAV.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-2 text-sm transition-colors hover:text-ink ${isActive ? 'bg-raised text-ink' : 'text-ink-2'}`
                }
              >
                {label}
              </NavLink>
            ))}
          </nav>
          <Account />
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8 pb-28 sm:px-6 md:pb-12">{children}</main>

      <nav
        aria-label="Main"
        className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-6 border-t border-line bg-page/95 pb-[env(safe-area-inset-bottom)] backdrop-blur md:hidden"
      >
        {NAV.map(({ to, label, short, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex min-h-14 flex-col items-center justify-center gap-1 text-[11px] ${isActive ? 'text-accent' : 'text-ink-3'}`
            }
          >
            <Icon aria-hidden className="size-5" strokeWidth={1.75} />
            <span className="max-w-full truncate px-0.5">{short ?? (label === 'References' ? 'Refs' : label)}</span>
          </NavLink>
        ))}
      </nav>
    </div>
  )
}
