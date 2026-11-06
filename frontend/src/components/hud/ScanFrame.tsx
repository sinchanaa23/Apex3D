import type { ReactNode } from 'react'

/**
 * A frame with HUD corner marks and a scanning line.
 * `active` scans continuously, for work in progress. Otherwise it scans once, `delay` seconds after it appears.
 */
export function ScanFrame({
  children,
  active = false,
  scan = true,
  delay = 0,
  className = '',
}: {
  children: ReactNode
  active?: boolean
  scan?: boolean
  delay?: number
  className?: string
}) {
  return (
    <div className={`relative overflow-hidden rounded-xl border border-line bg-surface ${className}`}>
      {children}
      <span aria-hidden className="hud-corner tl" />
      <span aria-hidden className="hud-corner tr" />
      <span aria-hidden className="hud-corner bl" />
      <span aria-hidden className="hud-corner br" />
      {scan && <span aria-hidden className={`scan-line ${active ? '' : 'once'}`} style={{ animationDelay: `${delay}s` }} />}
    </div>
  )
}
