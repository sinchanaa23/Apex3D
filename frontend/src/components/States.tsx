import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { CircleAlert, LoaderCircle } from 'lucide-react'
import { humanise, milliseconds } from '../lib/format'

export const primaryButton =
  'inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-accent px-5 py-2.5 font-medium text-page transition-colors hover:bg-accent-hover disabled:opacity-40 disabled:hover:bg-accent'
export const secondaryButton =
  'inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-line-strong px-5 py-2.5 font-medium text-ink transition-colors hover:bg-raised disabled:opacity-40 disabled:hover:bg-transparent'

export function PageHeader({ title, intro, children }: { title: string; intro?: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h1>
        {children}
      </div>
      {intro && <p className="mt-3 max-w-2xl text-ink-2">{intro}</p>}
    </header>
  )
}

export function Spinner({ label }: { label: string }) {
  return (
    <span role="status" className="inline-flex items-center gap-2 text-ink-2">
      <LoaderCircle aria-hidden className="size-4 animate-spin motion-reduce:animate-none" />
      {label}
    </span>
  )
}

export function PageLoading({ label }: { label: string }) {
  return (
    <div className="py-24 text-center">
      <Spinner label={label} />
    </div>
  )
}

export function ErrorNotice({ title, message, onRetry }: { title: string; message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded-xl border border-critical/50 bg-critical/8 p-4">
      <div className="flex gap-3">
        <CircleAlert aria-hidden className="mt-0.5 size-5 shrink-0 text-critical" />
        <div className="min-w-0">
          <p className="font-medium">{title}</p>
          <p className="mt-1 text-sm text-ink-2">{message}</p>
          {onRetry && (
            <button type="button" onClick={onRetry} className={`${secondaryButton} mt-3 text-sm`}>
              Try again
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export function EmptyState({ title, body, action }: { title: string; body: string; action?: { to: string; label: string } }) {
  return (
    <div className="mx-auto max-w-md py-20 text-center">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      <p className="mt-3 text-ink-2">{body}</p>
      {action && (
        <Link to={action.to} className={`${primaryButton} mt-6`}>
          {action.label}
        </Link>
      )}
    </div>
  )
}

export function Card({ title, children, className = '' }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-line bg-surface p-4 sm:p-5 ${className}`}>
      {title && <h2 className="mb-3 font-medium">{title}</h2>}
      {children}
    </section>
  )
}

export function Timings({ timings }: { timings: Record<string, number> }) {
  return (
    <dl className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-ink-3">
      {Object.entries(timings).map(([key, value]) => (
        <div key={key} className="flex gap-1.5">
          <dt>{humanise(key)}</dt>
          <dd className="num text-ink-2">{milliseconds(value)}</dd>
        </div>
      ))}
    </dl>
  )
}
