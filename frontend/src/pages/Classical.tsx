import { useEffect, useState } from 'react'
import { ApiError, classical } from '../api/client'
import { EmptyState, ErrorNotice, PageHeader, Timings } from '../components/States'
import { humanise } from '../lib/format'
import { useSession } from '../state/session'

function Histogram({ values }: { values: number[] }) {
  const max = Math.max(...values, Number.EPSILON)
  return (
    <div aria-hidden className="flex h-8 w-full max-w-64 items-end gap-px">
      {values.map((value, index) => (
        <span key={index} className="min-h-px flex-1 rounded-t-[1px] bg-series" style={{ height: `${(value / max) * 100}%` }} />
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
      <EmptyState
        title="No photo yet"
        body="Upload a photo to see each classical computer vision stage run on it."
        action={{ to: '/upload', label: 'Upload a photo' }}
      />
    )
  }

  return (
    <div>
      <PageHeader
        title="Classical CV"
        intro="What hand-built image processing sees in your photo, computed on the server for this image. These features feed the classical baseline and the hybrid experiments."
      />

      {error && <ErrorNotice title="Classical view unavailable" message={error} onRetry={() => setAttempt((a) => a + 1)} />}

      {!error && !result && (
        <div role="status" aria-label="Running classical pipeline" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          {Array.from({ length: 8 }, (_, i) => (
            <div key={i} className="skeleton aspect-[4/3] rounded-xl" />
          ))}
        </div>
      )}

      {result && (
        <>
          <ol className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {result.steps.map((step, index) => (
              <li key={step.key}>
                <figure>
                  <img src={step.image} alt={`${step.title} stage output`} loading="lazy" className="aspect-[4/3] w-full rounded-xl border border-line bg-surface object-contain" />
                  <figcaption className="mt-2 text-sm">
                    <span className="num mr-1.5 font-mono text-xs text-ink-3">{index + 1}</span>
                    {step.title}
                    {step.caption && <span className="mt-0.5 block text-xs text-ink-3">{step.caption}</span>}
                  </figcaption>
                </figure>
              </li>
            ))}
          </ol>

          <h2 className="mt-10 mb-3 text-lg font-medium">Extracted features</h2>
          <dl className="divide-y divide-line rounded-xl border border-line bg-surface px-4">
            {Object.entries(result.features).map(([name, value]) => (
              <div key={name} className="flex items-center justify-between gap-4 py-2.5">
                <dt className="text-sm text-ink-2">{humanise(name)}</dt>
                <dd className="flex min-w-0 flex-1 justify-end">
                  <FeatureValue value={value} />
                </dd>
              </div>
            ))}
          </dl>
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
