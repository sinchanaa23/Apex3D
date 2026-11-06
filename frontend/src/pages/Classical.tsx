import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { ApiError, classical } from '../api/client'
import { ScanFrame } from '../components/hud/ScanFrame'
import { rise, stagger } from '../components/hud/PageShell'
import { PhotoUpload } from '../components/PhotoUpload'
import { ErrorNotice, PageHeader, Timings } from '../components/States'
import { humanise } from '../lib/format'
import { useSession } from '../state/session'

function Histogram({ values }: { values: number[] }) {
  const max = Math.max(...values, Number.EPSILON)
  return (
    <div aria-hidden className="flex h-8 w-full max-w-64 items-end gap-px">
      {values.map((value, index) => (
        <span
          key={index}
          className="bar-rise min-h-px flex-1 rounded-t-[1px] bg-series"
          style={{ height: `${(value / max) * 100}%`, animationDelay: `${Math.min(index * 14, 600)}ms` }}
        />
      ))}
    </div>
  )
}

function FeatureValue({ value }: { value: number | string | number[] }) {
  if (Array.isArray(value)) {
    return (
      <div className="flex flex-col items-end gap-1">
        {value.length <= 64 && <Histogram values={value} />}
        <span className="num font-mono text-xs text-ink-3">{value.length} values</span>
      </div>
    )
  }
  return <span className="num font-mono text-sm">{typeof value === 'number' ? Number(value.toPrecision(4)) : value}</span>
}

export default function Classical() {
  const { photo, classical: result, setClassical } = useSession()
  const [error, setError] = useState<string | null>(null)
  const [attempt, setAttempt] = useState(0)
  const hasResult = result != null

  useEffect(() => {
    if (!photo || hasResult) return
    const controller = new AbortController()
    setError(null)
    classical(photo.file, controller.signal).then(setClassical, (reason: unknown) => {
      if (controller.signal.aborted) return
      setError(reason instanceof ApiError ? reason.message : 'The classical pipeline failed.')
    })
    return () => controller.abort()
  }, [photo, hasResult, attempt, setClassical])

  if (!photo) {
    return (
      <div className="mx-auto max-w-3xl">
        <PageHeader eyebrow="03 / Classical CV" title="Classical CV" intro="Add a photo to watch each hand-built image-processing stage run on it." />
        <PhotoUpload />
      </div>
    )
  }

  return (
    <div>
      <PageHeader
        eyebrow="03 / Classical CV"
        title="Classical CV"
        intro="What hand-built image processing sees in your photo, computed on the server for this image. These features feed the classical baseline and the hybrid experiments."
      />

      {error && <ErrorNotice title="Classical view unavailable" message={error} onRetry={() => setAttempt((a) => a + 1)} />}

      {!error && !result && (
        <div role="status" aria-label="Running classical pipeline" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {Array.from({ length: 6 }, (_, i) => (
            <ScanFrame key={i} active delay={i * 0.18} className="aspect-[4/3]">
              <span className="absolute inset-0 grid place-items-center font-mono text-xs tracking-widest text-ink-3 uppercase">Processing</span>
            </ScanFrame>
          ))}
        </div>
      )}

      {result && (
        <>
          <motion.ol variants={stagger} initial="hidden" animate="show" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {result.steps.map((step, index) => (
              <motion.li key={step.key} variants={rise}>
                <figure>
                  <ScanFrame delay={0.15 + index * 0.22}>
                    <img src={step.image} alt={`${step.title} stage output`} loading="lazy" className="aspect-[4/3] w-full bg-surface object-contain" />
                  </ScanFrame>
                  <figcaption className="mt-2 text-sm">
                    <span className="num mr-1.5 font-mono text-xs text-accent">{String(index + 1).padStart(2, '0')}</span>
                    {step.title}
                    {step.caption && <span className="mt-0.5 block text-xs text-ink-3">{step.caption}</span>}
                  </figcaption>
                </figure>
              </motion.li>
            ))}
          </motion.ol>

          <h2 className="mt-10 mb-3 flex items-center gap-2 text-lg font-medium">
            <span aria-hidden className="size-1.5 rounded-full bg-accent shadow-[0_0_8px_var(--color-accent)]" />
            Extracted features
          </h2>
          <motion.dl
            className="hud-panel divide-y divide-line px-4"
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-40px' }}
            transition={{ duration: 0.5 }}
          >
            {Object.entries(result.features).map(([name, value]) => (
              <div key={name} className="flex items-center justify-between gap-4 py-2.5">
                <dt className="text-sm text-ink-2">{humanise(name)}</dt>
                <dd className="flex min-w-0 flex-1 justify-end">
                  <FeatureValue value={value} />
                </dd>
              </div>
            ))}
          </motion.dl>
          {result.timings_ms && (
            <div className="mt-3">
              <Timings timings={result.timings_ms} />
            </div>
          )}
        </>
      )}
    </div>
  )
}
