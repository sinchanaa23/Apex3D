import { useEffect, useState } from 'react'
import { useReducedMotion } from 'framer-motion'

const GLYPHS = 'ABCDEFGHJKLMNPRSTUVWXYZ0123456789#%&'

/**
 * Text that resolves from scrambled characters into the real string.
 * The real text always stays in the layout (invisible while decoding), so nothing reflows,
 * and screen readers read the real text rather than the scramble.
 */
export function DecodeText({ text, className = '' }: { text: string; className?: string }) {
  const reduce = useReducedMotion()
  const [shown, setShown] = useState<string | null>(reduce ? null : '')

  useEffect(() => {
    if (reduce) {
      setShown(null)
      return
    }
    const steps = 20
    let step = 0
    const timer = setInterval(() => {
      step++
      if (step >= steps) {
        clearInterval(timer)
        setShown(null)
        return
      }
      const revealed = Math.floor((text.length * step) / steps)
      setShown(
        [...text].map((char, i) => (char === ' ' || i < revealed ? char : GLYPHS[Math.floor(Math.random() * GLYPHS.length)])).join(''),
      )
    }, 32)
    return () => clearInterval(timer)
  }, [text, reduce])

  return (
    <span className={`relative inline-block ${className}`}>
      <span className={shown === null ? '' : 'opacity-0'}>{text}</span>
      {shown !== null && (
        <span aria-hidden className="absolute inset-0 text-accent">
          {shown}
        </span>
      )}
    </span>
  )
}
