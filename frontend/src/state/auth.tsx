import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { fetchMe, logIn, logOut, signUp, UNAUTHORIZED_EVENT, type User } from '../api/auth'

interface AuthState {
  /** True until the first answer from the server. */
  loading: boolean
  user: User | null
  /** False when the server doesn't require a login, so every page is open. */
  loginRequired: boolean
  /** True when the user was signed out by the server and not by choice, so the login page can say so. */
  expired: boolean
  login(email: string, password: string): Promise<void>
  signup(name: string, email: string, password: string): Promise<void>
  logout(): Promise<void>
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(true)
  const [user, setUser] = useState<User | null>(null)
  const [loginRequired, setLoginRequired] = useState(true)
  const [expired, setExpired] = useState(false)

  useEffect(() => {
    let live = true
    fetchMe()
      .then((me) => {
        if (!live) return
        setUser(me.user)
        setLoginRequired(me.loginRequired)
      })
      .catch(() => undefined) // Server down: stay logged out. Pages show their own offline message.
      .finally(() => live && setLoading(false))
    return () => {
      live = false
    }
  }, [])

  useEffect(() => {
    const onUnauthorized = () => {
      setUser((current) => {
        if (current) setExpired(true)
        return null
      })
    }
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized)
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized)
  }, [])

  const login = useCallback(async (email: string, password: string) => {
    setUser(await logIn({ email, password }))
    setExpired(false)
  }, [])
  const signup = useCallback(async (name: string, email: string, password: string) => {
    setUser(await signUp({ name, email, password }))
    setExpired(false)
  }, [])
  const logout = useCallback(async () => {
    await logOut()
    setUser(null)
    setExpired(false)
  }, [])

  const value = useMemo(() => ({ loading, user, loginRequired, expired, login, signup, logout }), [loading, user, loginRequired, expired, login, signup, logout])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth(): AuthState {
  const value = useContext(AuthContext)
  if (!value) throw new Error('useAuth must be used inside AuthProvider')
  return value
}
