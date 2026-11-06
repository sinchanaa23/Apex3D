import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ChevronDown, TriangleAlert } from 'lucide-react'
import { AnimatePresence, motion } from 'framer-motion'
import type { ClassPrediction, ConfidenceState } from '../api/types'
import { ConfidenceBadge } from '../components/ConfidenceBadge'
import { DetectionImage } from '../components/DetectionImage'
import { CountUp } from '../components/hud/CountUp'
import { DecodeText } from '../components/hud/DecodeText'
import { Gauge } from '../components/hud/Gauge'
import { rise, stagger } from '../components/hud/PageShell'
import { PhotoUpload } from '../components/PhotoUpload'
import { PageHeader, Timings, hudPrimary, hudSecondary } from '../components/States'
import { percent } from '../lib/format'
import { useSession } from '../state/session'

const HEADLINE: Record<ConfidenceState, string> = {
  confident: 'Best match',
  ambiguous: 'Too close to call. Pick the one you think is right.',
  rejected: 'This may not be one of the 196 classes the model knows.',
}
const GAUGE_COLOUR: Record<ConfidenceState, string> = {
  confident: 'var(--color-good)',
  ambiguous: 'var(--color-warning)',
  rejected: 'var(--color-critical)',
}

export default function Identity() {
  const { photo, recognition, decideIdentity } = useSession()
  const navigate = useNavigate()
  const state = recognition?.confidence.state
  const [selected, setSelected] = useState<number | null>(null)
  const [open, setOpen] = useState(false)

  // A photo can arrive while this page is open (uploaded in place), so set up the choice whenever a new result appears.
  useEffect(() => {
    if (!recognition) return
    setSelected(recognition.confidence.state === 'confident' ? (recognition.top5[0]?.class_id ?? null) : null)
    setOpen(recognition.confidence.state !== 'confident')
  }, [recognition])

  if (!photo || !recognition || !state) {
    return (
      <div className="mx-auto max-w-3xl">
        <PageHeader eyebrow="02 / Identity" title="Vehicle identity" intro="Add a photo to find the car in it, see how sure the classifier is, and confirm or correct the answer." />
        <PhotoUpload />
      </div>
    )
  }

  const { confidence, detection, top5, timings_ms } = recognition
  const best = top5[0]
  const chosen = top5.find((p) => p.class_id === selected) ?? null
  const shownName = chosen?.label ?? best?.label ?? 'Vehicle'

  function proceed(identity: ClassPrediction | null) {
    decideIdentity(identity)
    navigate('/references')
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr] lg:items-start lg:gap-8">
      <div className="space-y-3">
        <DetectionImage src={photo.url} recognition={recognition} name={shownName} />
        {!detection && (
          <p className="flex gap-2 rounded-lg border border-warning/50 bg-warning/8 p-3 text-sm">
            <TriangleAlert aria-hidden className="mt-0.5 size-4 shrink-0 text-warning" />
            No vehicle was detected, so the whole photo was classified. Treat this result with caution.
          </p>
        )}
        <Timings timings={timings_ms} />
      </div>

      <motion.div variants={stagger} initial="hidden" animate="show">
        <motion.p variants={rise} className="mb-3 flex items-center gap-2 font-mono text-xs tracking-[0.22em] text-accent uppercase">
          <span aria-hidden className="inline-block h-px w-6 bg-accent" />
          02 / Identity
        </motion.p>

        <motion.div variants={rise} className="flex items-center gap-4">
          <Gauge value={confidence.top1} color={GAUGE_COLOUR[state]} label="Top prediction confidence" />
          <div className="min-w-0">
            <ConfidenceBadge state={state} />
            <p className="num mt-2 font-mono text-sm text-ink-3">margin {percent(confidence.margin)}</p>
          </div>
        </motion.div>

        <motion.p variants={rise} className="mt-4 text-sm text-ink-2">
          {HEADLINE[state]}
        </motion.p>
        {best && (
          <motion.h1 variants={rise} className="mt-1 text-3xl font-semibold tracking-tight sm:text-4xl">
            <DecodeText key={shownName} text={shownName} />
          </motion.h1>
        )}
        {state !== 'confident' && !chosen && <p className="mt-1 text-sm text-ink-3">Shown: the model's top guess, not confirmed.</p>}
        {confidence.reason && <p className="mt-3 text-sm text-ink-2">{confidence.reason}</p>}

        <motion.div variants={rise} className="hud-panel mt-5">
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
                transition={{ duration: 0.25 }}
                className="overflow-hidden"
              >
                <legend className="sr-only">Choose the correct identity</legend>
                <ul className="border-t border-line px-2 py-2">
                  {top5.map((prediction, index) => (
                    <li key={prediction.class_id}>
                      <label className="flex min-h-11 cursor-pointer items-center gap-3 rounded-lg px-2 py-2 transition-colors hover:bg-raised has-checked:bg-raised">
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
                            <CountUp value={prediction.probability} delay={0.1 + index * 0.08} className="font-mono text-xs text-ink-2" />
                          </span>
                          <span className="mt-1 block h-1 overflow-hidden rounded-full bg-line">
                            <motion.span
                              className={`block h-full origin-left rounded-full ${index === 0 ? 'bg-accent shadow-[0_0_10px_var(--color-accent)]' : 'bg-series'}`}
                              style={{ width: `${prediction.probability * 100}%` }}
                              initial={{ scaleX: 0 }}
                              animate={{ scaleX: 1 }}
                              transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1], delay: 0.1 + index * 0.08 }}
                            />
                          </span>
                        </span>
                      </label>
                    </li>
                  ))}
                </ul>
              </motion.fieldset>
            )}
          </AnimatePresence>
        </motion.div>

        <motion.div variants={rise} className="mt-5 flex flex-wrap gap-3">
          <button type="button" disabled={!chosen} className={hudPrimary} onClick={() => proceed(chosen)}>
            {chosen ? 'Confirm and find references' : 'Select a prediction to confirm'}
          </button>
          <button type="button" className={hudSecondary} onClick={() => proceed(null)}>
            None of these
          </button>
        </motion.div>
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
      </motion.div>
    </div>
  )
}
