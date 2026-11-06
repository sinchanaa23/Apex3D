import { useEffect, useState } from 'react'
import { animate, useReducedMotion } from 'framer-motion'
import { percent } from '../../lib/format'

/** A number that counts up to its value. The final value is exact, and it shows immediately with reduced motion. */
export function CountUp({
  value,
  format = (v: number) => percent(v),
  duration = 0.9,
  delay = 0,
  className = '',
}: {
  value: number
  format?(v: number): string
  duration?: number
  delay?: number
  className?: string
}) {
  const reduce = useReducedMotion()
  const [current, setCurrent] = useState(reduce ? value : 0)

  useEffect(() => {
    if (reduce) {
      setCurrent(value)
      return
    }
    const controls = animate(0, value, { duration, delay, ease: [0.22, 1, 0.36, 1], onUpdate: setCurrent, onComplete: () => setCurrent(value) })
    return () => controls.stop()
  }, [value, duration, delay, reduce])

  return <span className={`num ${className}`}>{format(current)}</span>
}
