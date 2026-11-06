import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { CircleAlert, LoaderCircle } from 'lucide-react'
import { humanise, milliseconds } from '../lib/format'

// The landing page uses primaryButton, secondaryButton and Spinner exactly as they are. Don't restyle them here.
export const primaryButton =
  'inline-flex min-h-11 items-center justify-center gap-2 rounded-lg bg-accent px-5 py-2.5 font-medium text-page transition-colors hover:bg-accent-hover disabled:opacity-40 disabled:hover:bg-accent'
export const secondaryButton =
  'inline-flex min-h-11 items-center justify-center gap-2 rounded-lg border border-line-strong px-5 py-2.5 font-medium text-ink transition-colors hover:bg-raised disabled:opacity-40 disabled:hover:bg-transparent'

/** The same buttons with a glow and a light sweep, for the inner pages. */
export const hudPrimary = `${primaryButton} hud-btn`
export const hudSecondary = `${secondaryButton} hud-btn`

export function Spinner({ label }: { label: string }) {
  return (
    <span role="status" className="inline-flex items-center gap-2 text-ink-2">
      <LoaderCircle aria-hidden className="size-4 animate-spin motion-reduce:animate-none" />
      {label}
    </span>
  )
}

export function PageHeader({ title, eyebrow, intro, children }: { title: string; eyebrow?: string; intro?: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-8">
      {eyebrow && (
        <p className="mb-3 flex items-center gap-2 font-mono text-xs tracking-[0.22em] text-accent uppercase">
          <span aria-hidden className="inline-block h-px w-6 bg-accent" />
          {eyebrow}
        </p>
      )}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h1>
        {children}
      </div>
      <motion.div
        aria-hidden
        className="mt-4 h-px origin-left bg-gradient-to-r from-accent via-accent/30 to-transparent"
        initial={{ scaleX: 0 }}
        animate={{ scaleX: 1 }}
        transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1], delay: 0.15 }}
      />
      {intro && <p className="mt-4 max-w-2xl text-ink-2">{intro}</p>}
    </header>
  )
}

export function PageLoading({ label }: { label: string }) {
  return (
    <div className="mx-auto max-w-xs py-24 text-center">
      <Spinner label={label} />
      <div className="scanbar mt-4" />
    </div>
  )
}

export function ErrorNotice({ title, message, onRetry }: { title: string; message: string; onRetry?: () => void }) {
  return (
    <motion.div
      role="alert"
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
      className="relative overflow-hidden rounded-xl border border-critical/50 bg-critical/8 p-4"
    >
      <span aria-hidden className="absolute inset-y-0 left-0 w-0.5 bg-critical shadow-[0_0_12px_var(--color-critical)]" />
      <div className="flex gap-3">
        <CircleAlert aria-hidden className="mt-0.5 size-5 shrink-0 text-critical" />
        <div className="min-w-0">
          <p className="font-medium">{title}</p>
          <p className="mt-1 text-sm text-ink-2">{message}</p>
          {onRetry && (
            <button type="button" onClick={onRetry} className={`${hudSecondary} mt-3 text-sm`}>
              Try again
            </button>
          )}
        </div>
      </div>
    </motion.div>
  )
}

export function EmptyState({ title, body, action }: { title: string; body: string; action?: { to: string; label: string } }) {
  return (
    <div className="mx-auto max-w-md py-20 text-center">
      <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
      <p className="mt-3 text-ink-2">{body}</p>
      {action && (
        <Link to={action.to} className={`${hudPrimary} mt-6`}>
          {action.label}
        </Link>
      )}
    </div>
  )
}

export function Card({ title, children, className = '' }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`hud-panel p-4 sm:p-5 ${className}`}>
      {title && (
        <h2 className="mb-3 flex items-center gap-2 font-medium">
          <span aria-hidden className="size-1.5 rounded-full bg-accent shadow-[0_0_8px_var(--color-accent)]" />
          {title}
        </h2>
      )}
      {children}
    </section>
  )
}

export function Timings({ timings }: { timings: Record<string, number> }) {
  return (
    <dl className="flex flex-wrap gap-2 font-mono text-xs text-ink-3">
      {Object.entries(timings).map(([key, value]) => (
        <div key={key} className="flex gap-1.5 rounded border border-line bg-surface/60 px-2 py-1">
          <dt className="uppercase">{humanise(key)}</dt>
          <dd className="num text-ink-2">{milliseconds(value)}</dd>
        </div>
      ))}
    </dl>
  )
}
