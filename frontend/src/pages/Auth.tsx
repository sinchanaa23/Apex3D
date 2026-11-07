import { useId, useState, type FormEvent } from 'react'
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom'
import { Eye, EyeOff, LoaderCircle, ShieldCheck } from 'lucide-react'
import { AuthError } from '../api/auth'
import { PageLoading, hudPrimary } from '../components/States'
import { emailProblem, MIN_PASSWORD, nameProblem, passwordProblem, passwordStrength } from '../lib/validate'
import { useAuth } from '../state/auth'

type Mode = 'login' | 'signup'

const inputClass =
  'min-h-11 w-full rounded-lg border border-line-strong bg-raised px-3 text-ink placeholder:text-ink-3 focus:border-accent aria-[invalid=true]:border-warning'

function Field({
  label,
  hint,
  error,
  children,
}: {
  label: string
  hint?: string
  error?: string | null
  children: (props: { id: string; describedBy: string | undefined; invalid: boolean }) => React.ReactNode
}) {
  const id = useId()
  const describedBy = error ? `${id}-error` : hint ? `${id}-hint` : undefined
  return (
    <div>
      <label htmlFor={id} className="mb-1.5 block text-sm text-ink-2">
        {label}
      </label>
      {children({ id, describedBy, invalid: Boolean(error) })}
      {error ? (
        <p id={`${id}-error`} className="mt-1.5 text-sm text-warning">
          {error}
        </p>
      ) : (
        hint && (
          <p id={`${id}-hint`} className="mt-1.5 text-xs text-ink-3">
            {hint}
          </p>
        )
      )}
    </div>
  )
}

function StrengthMeter({ password }: { password: string }) {
  if (!password) return null
  const level = passwordStrength(password)
  const labels = ['Too weak', 'Weak', 'Fair', 'Good', 'Strong']
  return (
    <div className="mt-2" aria-live="polite">
      <div className="flex gap-1" aria-hidden>
        {[1, 2, 3, 4].map((step) => (
          <span key={step} className={`h-1 flex-1 rounded-full ${step <= level ? 'bg-accent' : 'bg-line-strong'}`} />
        ))}
      </div>
      <p className="mt-1 text-xs text-ink-3">Password strength: {labels[level]}</p>
    </div>
  )
}

export default function Auth({ mode }: { mode: Mode }) {
  const { loading, user, loginRequired, expired, login, signup } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const from = (location.state as { from?: string } | null)?.from ?? '/upload'
  const isSignup = mode === 'signup'

  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [show, setShow] = useState(false)
  const [touched, setTouched] = useState(false)
  const [busy, setBusy] = useState(false)
  const [serverError, setServerError] = useState<string | null>(null)

  if (loading) return <PageLoading label="Checking your session" />
  if (user || !loginRequired) return <Navigate to={from} replace />

  const problems = {
    name: isSignup ? nameProblem(name) : null,
    email: emailProblem(email),
    password: isSignup ? passwordProblem(password) : password ? null : 'Enter your password.',
    confirm: isSignup && confirm !== password ? 'The two passwords do not match.' : null,
  }
  const show_ = (key: keyof typeof problems) => (touched ? problems[key] : null)
  const valid = !problems.name && !problems.email && !problems.password && !problems.confirm

  async function submit(event: FormEvent) {
    event.preventDefault()
    setTouched(true)
    setServerError(null)
    if (!valid) return
    setBusy(true)
    try {
      if (isSignup) await signup(name.trim(), email.trim(), password)
      else await login(email.trim(), password)
      navigate(from, { replace: true })
    } catch (error) {
      setServerError(error instanceof AuthError ? error.message : 'Something went wrong. Try again.')
      setBusy(false)
    }
  }

  return (
    <div className="mx-auto max-w-md pt-4">
      <p className="mb-3 flex items-center gap-2 font-mono text-xs tracking-[0.22em] text-accent uppercase">
        <span aria-hidden className="h-px w-8 bg-accent/60" />
        {isSignup ? 'Create account' : 'Welcome back'}
      </p>
      <h1 className="text-3xl font-semibold tracking-tight">{isSignup ? 'Sign up for Apex3D' : 'Log in to Apex3D'}</h1>
      <p className="mt-2 text-ink-2">
        {isSignup ? 'An account lets you identify cars from photos and build 3D models.' : 'Log in to identify cars from photos and build 3D models.'}
      </p>

      {expired && !serverError && (
        <p role="status" className="mt-5 rounded-lg border border-line-strong bg-raised px-4 py-3 text-sm text-ink-2">
          Your session ended. Log in again to continue.
        </p>
      )}

      <form onSubmit={submit} noValidate className="hud-panel mt-6 space-y-5 p-5 sm:p-6">
        {isSignup && (
          <Field label="Name" error={show_('name')}>
            {({ id, describedBy, invalid }) => (
              <input id={id} name="name" type="text" autoComplete="name" required maxLength={60} value={name} onChange={(e) => setName(e.target.value)} aria-invalid={invalid} aria-describedby={describedBy} className={inputClass} />
            )}
          </Field>
        )}

        <Field label="Email" error={show_('email')}>
          {({ id, describedBy, invalid }) => (
            <input id={id} name="email" type="email" inputMode="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} aria-invalid={invalid} aria-describedby={describedBy} className={inputClass} />
          )}
        </Field>

        <Field label="Password" error={show_('password')} hint={isSignup ? `At least ${MIN_PASSWORD} characters.` : undefined}>
          {({ id, describedBy, invalid }) => (
            <>
              <div className="relative">
                <input
                  id={id}
                  name="password"
                  type={show ? 'text' : 'password'}
                  autoComplete={isSignup ? 'new-password' : 'current-password'}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  aria-invalid={invalid}
                  aria-describedby={describedBy}
                  className={`${inputClass} pr-12`}
                />
                <button
                  type="button"
                  onClick={() => setShow((value) => !value)}
                  aria-label={show ? 'Hide password' : 'Show password'}
                  aria-pressed={show}
                  className="absolute inset-y-0 right-0 flex w-11 items-center justify-center text-ink-3 hover:text-ink"
                >
                  {show ? <EyeOff aria-hidden className="size-4" /> : <Eye aria-hidden className="size-4" />}
                </button>
              </div>
              {isSignup && <StrengthMeter password={password} />}
            </>
          )}
        </Field>

        {isSignup && (
          <Field label="Confirm password" error={show_('confirm')}>
            {({ id, describedBy, invalid }) => (
              <input id={id} name="confirm" type={show ? 'text' : 'password'} autoComplete="new-password" required value={confirm} onChange={(e) => setConfirm(e.target.value)} aria-invalid={invalid} aria-describedby={describedBy} className={inputClass} />
            )}
          </Field>
        )}

        <div aria-live="assertive">
          {serverError && (
            <p role="alert" className="rounded-lg border border-warning/50 bg-warning/10 px-4 py-3 text-sm text-ink">
              {serverError}
            </p>
          )}
        </div>

        <button type="submit" disabled={busy} className={`${hudPrimary} w-full`}>
          {busy && <LoaderCircle aria-hidden className="size-4 animate-spin motion-reduce:animate-none" />}
          {busy ? (isSignup ? 'Creating account' : 'Logging in') : isSignup ? 'Create account' : 'Log in'}
        </button>
      </form>

      <p className="mt-5 text-sm text-ink-2">
        {isSignup ? 'Already have an account? ' : 'New here? '}
        <Link to={isSignup ? '/login' : '/signup'} state={location.state} className="text-ink underline underline-offset-4">
          {isSignup ? 'Log in' : 'Create an account'}
        </Link>
      </p>
      <p className="mt-6 flex items-start gap-2 text-xs text-ink-3">
        <ShieldCheck aria-hidden className="mt-0.5 size-4 shrink-0" />
        Passwords are stored as salted hashes, never as text. You stay logged in on this device for 7 days, or until you log out.
      </p>
    </div>
  )
}
