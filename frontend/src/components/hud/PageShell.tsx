import type { ReactNode } from 'react'
import { motion } from 'framer-motion'

/** Ambient grid and glow behind an inner page, plus its entrance transition. Not used on the landing page. */
export function PageShell({ children }: { children: ReactNode }) {
  return (
    <div className="relative">
      <div aria-hidden className="hud-bg" />
      <motion.div
        initial={{ opacity: 0, y: 16, filter: 'blur(6px)' }}
        animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      >
        {children}
      </motion.div>
    </div>
  )
}

/** Container and child variants for a staggered reveal. */
export const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07, delayChildren: 0.05 } },
}
export const rise = {
  hidden: { opacity: 0, y: 14 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] as const } },
}
