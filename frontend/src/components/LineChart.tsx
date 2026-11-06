import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { motion } from 'framer-motion'

export interface Series {
  name: string
  /** Series colour. Identity only: text stays in ink colours. */
  color: string
  /** Fractions from 0 to 1, one per entry in `xs`. */
  values: number[]
}

interface Props {
  title: string
  xLabel: string
  xs: number[]
  series: Series[]
  formatX?(x: number): string
  /** Marks one x position with a hairline and a label. */
  marker?: { x: number; label: string } | null
  height?: number
}

const MARGIN = { top: 14, right: 16, bottom: 36, left: 40 }
const TICKS = [0, 0.25, 0.5, 0.75, 1]

function useWidth() {
  const ref = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(320)
  useEffect(() => {
    if (!ref.current) return
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(ref.current)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

/** Line chart on one 0 to 100% axis, with a crosshair and tooltip for pointer and keyboard. */
export function LineChart({ title, xLabel, xs, series, formatX = String, marker, height = 200 }: Props) {
  const [ref, width] = useWidth()
  const [active, setActive] = useState<number | null>(null)

  const innerW = Math.max(width - MARGIN.left - MARGIN.right, 10)
  const innerH = height - MARGIN.top - MARGIN.bottom
  const min = Math.min(...xs)
  const max = Math.max(...xs)
  const x = (value: number) => MARGIN.left + (max === min ? innerW / 2 : ((value - min) / (max - min)) * innerW)
  const y = (value: number) => MARGIN.top + (1 - value) * innerH

  function nearest(px: number) {
    let best = 0
    xs.forEach((value, index) => {
      if (Math.abs(x(value) - px) < Math.abs(x(xs[best]) - px)) best = index
    })
    return best
  }

  function onKey(event: KeyboardEvent) {
    if (event.key === 'ArrowRight') setActive((a) => Math.min((a ?? -1) + 1, xs.length - 1))
    else if (event.key === 'ArrowLeft') setActive((a) => Math.max((a ?? 1) - 1, 0))
    else return
    event.preventDefault()
  }

  const tooltipOnLeft = active != null && x(xs[active]) > width / 2

  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 text-sm text-ink-2">{title}</figcaption>
      {series.length > 1 && (
        <ul className="mb-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
          {series.map(({ name, color }) => (
            <li key={name} className="flex items-center gap-1.5">
              <span aria-hidden className="h-0.5 w-3.5 rounded-full" style={{ background: color }} />
              {name}
            </li>
          ))}
        </ul>
      )}
      <div
        ref={ref}
        role="img"
        tabIndex={0}
        aria-label={`${title}. Use left and right arrow keys to read values. A table view follows.`}
        className="relative touch-pan-y"
        onKeyDown={onKey}
        onBlur={() => setActive(null)}
        onPointerMove={(event) => setActive(nearest(event.clientX - event.currentTarget.getBoundingClientRect().left))}
        onPointerLeave={() => setActive(null)}
      >
        <svg width={width} height={height} className="block" aria-hidden>
          {TICKS.map((tick) => (
            <g key={tick}>
              <line x1={MARGIN.left} x2={width - MARGIN.right} y1={y(tick)} y2={y(tick)} stroke={tick === 0 ? 'var(--color-line-strong)' : 'var(--color-line)'} />
              <text x={MARGIN.left - 8} y={y(tick) + 4} textAnchor="end" fontSize="11" fill="var(--color-ink-3)" className="num">
                {tick * 100}%
              </text>
            </g>
          ))}
          {xs.map((value, index) => {
            // Thin the labels when points are dense, keeping the first and last.
            const every = Math.ceil(xs.length / Math.max(Math.floor(innerW / 44), 1))
            if (index % every !== 0 && index !== xs.length - 1) return null
            return (
              <text key={value} x={x(value)} y={height - MARGIN.bottom + 16} textAnchor="middle" fontSize="11" fill="var(--color-ink-3)" className="num">
                {formatX(value)}
              </text>
            )
          })}
          <text x={MARGIN.left + innerW / 2} y={height - 4} textAnchor="middle" fontSize="11" fill="var(--color-ink-3)">
            {xLabel}
          </text>
          {marker && (
            <g>
              <line x1={x(marker.x)} x2={x(marker.x)} y1={MARGIN.top} y2={MARGIN.top + innerH} stroke="var(--color-ink-3)" />
              <text x={x(marker.x) + 5} y={MARGIN.top + 10} fontSize="11" fill="var(--color-ink-2)">
                {marker.label}
              </text>
            </g>
          )}
          {series.map(({ name, color, values }) => (
            <motion.polyline
              key={name}
              fill="none"
              stroke={color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              style={{ filter: `drop-shadow(0 0 4px ${color})` }}
              points={values.map((value, index) => `${x(xs[index])},${y(value)}`).join(' ')}
              initial={{ pathLength: 0, opacity: 0 }}
              whileInView={{ pathLength: 1, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.2, ease: 'easeInOut' }}
            />
          ))}
          {active != null && (
            <g>
              <line x1={x(xs[active])} x2={x(xs[active])} y1={MARGIN.top} y2={MARGIN.top + innerH} stroke="var(--color-line-strong)" />
              {series.map(({ name, color, values }) => (
                <circle key={name} cx={x(xs[active])} cy={y(values[active])} r="4.5" fill={color} stroke="var(--color-surface)" strokeWidth="2" />
              ))}
            </g>
          )}
        </svg>
        {active != null && (
          <div
            className="pointer-events-none absolute z-10 rounded-md border border-line-strong bg-page px-2 py-1 text-xs whitespace-nowrap shadow-lg"
            style={{
              top: MARGIN.top,
              ...(tooltipOnLeft ? { right: width - x(xs[active]) + 8 } : { left: x(xs[active]) + 8 }),
            }}
          >
            <p className="text-ink-3">
              {xLabel} {formatX(xs[active])}
            </p>
            {series.map(({ name, color, values }) => (
              <p key={name} className="flex items-center gap-1.5">
                <span aria-hidden className="size-2 rounded-full" style={{ background: color }} />
                {series.length > 1 && <span className="text-ink-2">{name}</span>}
                <span className="num font-mono">{(values[active] * 100).toFixed(1)}%</span>
              </p>
            ))}
          </div>
        )}
      </div>
    </figure>
  )
}
