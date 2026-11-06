import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ChevronDown, TriangleAlert } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import type { ClassPrediction, ConfidenceState } from '../api/types'
import { ConfidenceBadge } from '../components/ConfidenceBadge'
import { DetectionImage } from '../components/DetectionImage'
import { EmptyState, Timings, primaryButton, secondaryButton } from '../components/States'
import { percent } from '../lib/format'
import { useSession } from '../state/session'

const HEADLINE: Record<ConfidenceState, string> = {
  confident: 'Best match',
  ambiguous: 'Too close to call. Pick the one you think is right.',
  rejected: 'This may not be one of the 196 classes the model knows.',
}

export default function Identity() {
  const { photo, recognition, decideIdentity } = useSession()
  const navigate = useNavigate()
  const state = recognition?.confidence.state
  const [selected, setSelected] = useState<number | null>(state === 'confident' ? (recognition?.top5[0]?.class_id ?? null) : null)
  const [open, setOpen] = useState(state !== 'confident')

  if (!photo || !recognition || !state) {
    return (
      <EmptyState
        title="No photo yet"
        body="Upload a photo to see the detected vehicle and what the classifier thinks it is. Photos aren't kept if you reload the page."
        action={{ to: '/upload', label: 'Upload a photo' }}
      />
    )
  }

  const { confidence, detection, top5, timings_ms } = recognition
  const best = top5[0]
  const chosen = top5.find((p) => p.class_id === selected) ?? null

  function proceed(identity: ClassPrediction | null) {
    decideIdentity(identity)
    navigate('/references')
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr] lg:items-start lg:gap-8">
      <div className="space-y-3">
        <DetectionImage src={photo.url} recognition={recognition} name={chosen?.label ?? best?.label ?? 'Vehicle'} />
        {!detection && (
          <p className="flex gap-2 rounded-lg border border-warning/50 bg-warning/8 p-3 text-sm">
            <TriangleAlert aria-hidden className="mt-0.5 size-4 shrink-0 text-warning" />
            No vehicle was detected, so the whole photo was classified. Treat this result with caution.
          </p>
        )}
        <Timings timings={timings_ms} />
      </div>

      <div>
        <div className="flex flex-wrap items-center gap-3">
          <ConfidenceBadge state={state} />
          <p className="num font-mono text-sm text-ink-3">
            top {percent(confidence.top1)} · margin {percent(confidence.margin)}
          </p>
        </div>
        <p className="mt-4 text-sm text-ink-2">{HEADLINE[state]}</p>
        {best && <h1 className="mt-1 text-3xl font-semibold tracking-tight sm:text-4xl">{chosen?.label ?? best.label}</h1>}
        {state !== 'confident' && !chosen && <p className="mt-1 text-sm text-ink-3">Shown: the model's top guess, not confirmed.</p>}
        {confidence.reason && <p className="mt-3 text-sm text-ink-2">{confidence.reason}</p>}

        <div className="mt-5 rounded-xl border border-line bg-surface">
          <button
            type="button"
            aria-expanded={open}
            aria-controls="top5"
            onClick={() => setOpen(!open)}
            className="flex min-h-12 w-full items-center justify-between px-4 text-left font-medium"
          >
            Top 5 predictions
            <ChevronDown aria-hidden className={`size-4 transition-transform ${open ? 'rotate-180' : ''}`} />
          </button>
          <AnimatePresence initial={false}>
            {open && (
              <motion.fieldset
                id="top5"
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <legend className="sr-only">Choose the correct identity</legend>
                <ul className="border-t border-line px-2 py-2">
                  {top5.map((prediction) => (
                    <li key={prediction.class_id}>
                      <label className="flex min-h-11 cursor-pointer items-center gap-3 rounded-lg px-2 py-2 hover:bg-raised has-checked:bg-raised">
                        <input
                          type="radio"
                          name="identity"
                          className="size-4 shrink-0 accent-accent"
                          checked={selected === prediction.class_id}
                          onChange={() => setSelected(prediction.class_id)}
                        />
                        <span className="min-w-0 flex-1">
                          <span className="flex items-baseline justify-between gap-3 text-sm">
                            <span>{prediction.label}</span>
                            <span className="num font-mono text-xs text-ink-2">{percent(prediction.probability)}</span>
                          </span>
                          <span className="mt-1 block h-1 rounded-full bg-line">
                            <span className="block h-full rounded-full bg-series" style={{ width: `${prediction.probability * 100}%` }} />
                          </span>
                        </span>
                      </label>
                    </li>
                  ))}
                </ul>
              </motion.fieldset>
            )}
          </AnimatePresence>
        </div>

        <div className="mt-5 flex flex-wrap gap-3">
          <button type="button" disabled={!chosen} className={primaryButton} onClick={() => proceed(chosen)}>
            {chosen ? 'Confirm and find references' : 'Select a prediction to confirm'}
          </button>
          <button type="button" className={secondaryButton} onClick={() => proceed(null)}>
            None of these
          </button>
        </div>
        <p className="mt-2 text-xs text-ink-3">"None of these" skips automatic references. You can still upload your own and generate a mesh.</p>

        <div className="mt-6 flex flex-wrap gap-x-5 gap-y-2 border-t border-line pt-4 text-sm">
          <Link to="/classical" className="text-ink-2 underline underline-offset-4 hover:text-ink">
            See the classical CV stages
          </Link>
          <Link to="/upload" className="text-ink-2 underline underline-offset-4 hover:text-ink">
            Try another photo
          </Link>
        </div>
        <p className="mt-4 text-xs text-ink-3">The model always answers from its 196 classes, up to 2012. Newer or unlisted cars may get a confident but wrong result.</p>
      </div>
    </div>
  )
}
