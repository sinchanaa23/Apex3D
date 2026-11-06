import { CircleCheck, CircleHelp, CircleX } from 'lucide-react'
import type { ConfidenceState } from '../api/types'

export const CONFIDENCE = {
  confident: { label: 'Confident', icon: CircleCheck, color: 'text-good', border: 'border-good/50' },
  ambiguous: { label: 'Ambiguous', icon: CircleHelp, color: 'text-warning', border: 'border-warning/50' },
  rejected: { label: 'Rejected', icon: CircleX, color: 'text-critical', border: 'border-critical/50' },
} as const

export function ConfidenceBadge({ state }: { state: ConfidenceState }) {
  const { label, icon: Icon, color, border } = CONFIDENCE[state]
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-sm ${border}`}>
      <Icon aria-hidden className={`size-4 ${color}`} />
      {label}
    </span>
  )
}
