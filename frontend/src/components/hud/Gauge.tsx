import { motion } from 'framer-motion'
import { CountUp } from './CountUp'

const RADIUS = 34
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

/** A radial gauge for a 0 to 1 value, with tick marks. The number inside is always shown, so colour is never the only signal. */
export function Gauge({ value, color, label, size = 96 }: { value: number; color: string; label: string; size?: number }) {
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }} role="img" aria-label={`${label}: ${(value * 100).toFixed(1)}%`}>
      <svg viewBox="0 0 80 80" className="size-full" aria-hidden>
        {Array.from({ length: 40 }, (_, i) => {
          const angle = (i / 40) * Math.PI * 2
          const major = i % 10 === 0
          const r1 = major ? 36.5 : 38
          return (
            <line
              key={i}
              x1={40 + Math.cos(angle) * r1}
              y1={40 + Math.sin(angle) * r1}
              x2={40 + Math.cos(angle) * 39.5}
              y2={40 + Math.sin(angle) * 39.5}
              stroke={major ? 'var(--color-ink-3)' : 'var(--color-line-strong)'}
              strokeWidth="0.6"
            />
          )
        })}
        <circle cx="40" cy="40" r={RADIUS} fill="none" stroke="var(--color-line)" strokeWidth="4" />
        <motion.circle
          cx="40"
          cy="40"
          r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          initial={{ strokeDashoffset: CIRCUMFERENCE }}
          animate={{ strokeDashoffset: CIRCUMFERENCE * (1 - Math.min(Math.max(value, 0), 1)) }}
          transition={{ duration: 1.1, ease: [0.22, 1, 0.36, 1], delay: 0.2 }}
          transform="rotate(-90 40 40)"
          style={{ filter: `drop-shadow(0 0 5px ${color})` }}
        />
      </svg>
      <div className="absolute inset-0 grid place-items-center">
        <CountUp value={value} format={(v) => `${(v * 100).toFixed(0)}%`} delay={0.2} className="font-mono text-lg font-semibold" />
      </div>
    </div>
  )
}
