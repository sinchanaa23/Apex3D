import { useId, useState } from 'react'
import { motion } from 'framer-motion'
import { Link, useNavigate } from 'react-router-dom'
import { Check, ImagePlus, Plus, TriangleAlert } from 'lucide-react'
import { ApiError, getReferences, uploadReferences } from '../api/client'
import type { Reference, Viewpoint } from '../api/types'
import { Attribution } from '../components/Attribution'
import { Card, ErrorNotice, PageHeader, Spinner, hudPrimary as primaryButton, hudSecondary as secondaryButton } from '../components/States'
import { checkFile, normaliseImage } from '../lib/image'
import { useResource } from '../lib/useResource'
import { useSession } from '../state/session'

type Group = Viewpoint | 'unsorted'

const GROUPS: { key: Group; label: string }[] = [
  { key: 'front', label: 'Front' },
  { key: 'front_three_quarter', label: 'Front three-quarter' },
  { key: 'side', label: 'Side' },
  { key: 'rear_three_quarter', label: 'Rear three-quarter' },
  { key: 'rear', label: 'Rear' },
  { key: 'unsorted', label: 'Unsorted' },
]

function ReferenceCard({ reference, group, onMove, index }: { reference: Reference; group: Group; onMove(group: Group): void; index: number }) {
  const { input, setInput, pack, togglePack } = useSession()
  const isInput = input?.kind === 'reference' && input.reference.id === reference.id
  const inPack = pack.some((r) => r.id === reference.id)
  const selectId = useId()

  return (
    <motion.li
      initial={{ opacity: 0, y: 18, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1], delay: Math.min(index * 0.05, 0.5) }}
      whileHover={{ y: -3 }}
      className={`hud-panel flex flex-col overflow-hidden ${isInput ? 'border-accent shadow-[0_0_30px_-8px_rgb(255_106_43/0.55)]' : ''}`}
    >
      <div className="relative">
        <img src={reference.thumb_url ?? reference.image_url} alt={`Reference image from ${reference.source}`} loading="lazy" className="aspect-[4/3] w-full bg-raised object-cover" />
        {reference.match === 'confirmed' && (
          <span className="absolute top-2 left-2 inline-flex items-center gap-1 rounded bg-page/85 px-1.5 py-0.5 text-xs">
            <Check aria-hidden className="size-3 text-good" /> Title matches model
          </span>
        )}
        {reference.match === 'approximate' && (
          <span className="absolute top-2 left-2 inline-flex items-center gap-1 rounded bg-page/85 px-1.5 py-0.5 text-xs">
            <TriangleAlert aria-hidden className="size-3 text-warning" /> Approximate reference
          </span>
        )}
      </div>
      <div className="flex flex-1 flex-col gap-3 p-3">
        <Attribution reference={reference} />
        <div className="mt-auto">
          <label htmlFor={selectId} className="sr-only">
            Viewpoint group
          </label>
          <select
            id={selectId}
            value={group}
            onChange={(event) => onMove(event.target.value as Group)}
            className="mb-2 min-h-9 w-full rounded-md border border-line-strong bg-page px-2 text-sm text-ink-2"
          >
            {GROUPS.map(({ key, label }) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <button
              type="button"
              aria-pressed={isInput}
              onClick={() => setInput({ kind: 'reference', reference })}
              className={`min-h-10 rounded-lg border px-2 ${isInput ? 'border-accent bg-accent/12 text-ink' : 'border-line-strong hover:bg-raised'}`}
            >
              {isInput ? 'Input view' : 'Use as input'}
            </button>
            <button
              type="button"
              aria-pressed={inPack}
              onClick={() => togglePack(reference)}
              className={`inline-flex min-h-10 items-center justify-center gap-1 rounded-lg border px-2 ${inPack ? 'border-series bg-series/12' : 'border-line-strong hover:bg-raised'}`}
            >
              {inPack ? <Check aria-hidden className="size-4" /> : <Plus aria-hidden className="size-4" />}
              {inPack ? 'In pack' : 'Add to pack'}
            </button>
          </div>
        </div>
      </div>
    </motion.li>
  )
}

function UploadOwn() {
  const inputId = useId()
  const { addUploadedReferences, setInput } = useSession()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function upload(list: FileList | null) {
    const files = Array.from(list ?? [])
    if (!files.length) return
    setError(null)
    const problem = files.map(checkFile).find(Boolean)
    if (problem) return setError(problem)
    setBusy(true)
    try {
      const normalised = await Promise.all(files.map(normaliseImage))
      normalised.forEach(({ url }) => URL.revokeObjectURL(url))
      const { candidates } = await uploadReferences(normalised.map(({ file }) => file))
      addUploadedReferences(candidates)
      if (candidates[0]) setInput({ kind: 'reference', reference: candidates[0] })
    } catch (reason) {
      setError(reason instanceof ApiError || reason instanceof Error ? reason.message : 'The upload failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      <label htmlFor={inputId} className={`${secondaryButton} ${busy ? 'pointer-events-none opacity-50' : ''}`}>
        <ImagePlus aria-hidden className="size-4" />
        {busy ? 'Uploading' : 'Upload your own'}
        <input
          id={inputId}
          type="file"
          accept="image/*"
          multiple
          disabled={busy}
          className="sr-only"
          onChange={(event) => {
            void upload(event.target.files)
            event.target.value = ''
          }}
        />
      </label>
      {error && (
        <div className="mt-3">
          <ErrorNotice title="Upload failed" message={error} />
        </div>
      )}
    </div>
  )
}

export default function References() {
  const { photo, identity, identityDecided, input, setInput, uploadedReferences } = useSession()
  const navigate = useNavigate()
  const classId = identity?.class_id ?? null
  const retrieval = useResource((signal) => (classId == null ? Promise.resolve({ candidates: [] as Reference[] }) : getReferences(classId, signal)), classId)
  const [moved, setMoved] = useState<Record<string, Group>>({})

  const all = [...uploadedReferences, ...(retrieval.data?.candidates ?? [])]
  const groupOf = (reference: Reference): Group => moved[reference.id] ?? reference.viewpoint ?? 'unsorted'
  const inputUrl = input?.kind === 'photo' ? photo?.url : (input?.reference.thumb_url ?? input?.reference.image_url)

  return (
    <div>
      <PageHeader
        eyebrow="04 / References"
        title="Reference Studio"
        intro={
          identity ? (
            <>
              Openly licensed photos for <span className="text-ink">{identity.label}</span>. Keep the ones worth comparing against, and pick the single view to reconstruct from.
            </>
          ) : (
            'Upload your own reference photos, then pick the single view to reconstruct from.'
          )
        }
      >
        <UploadOwn />
      </PageHeader>

      {!identity && (
        <p className="mb-6 rounded-lg border border-line bg-surface p-3 text-sm text-ink-2">
          {identityDecided ? (
            'You continued without a confirmed identity, so no references are searched for automatically.'
          ) : (
            <>
              No confirmed identity yet, so no references are searched for.{' '}
              <Link to={photo ? '/identity' : '/upload'} className="text-ink underline underline-offset-4">
                {photo ? 'Confirm the identity' : 'Upload a photo'}
              </Link>{' '}
              to search automatically.
            </>
          )}
        </p>
      )}

      <Card className="mb-8">
        <div className="flex flex-wrap items-center gap-4">
          {inputUrl ? (
            <img src={inputUrl} alt="Selected reconstruction input" className="h-16 w-24 rounded-md bg-raised object-cover" />
          ) : (
            <div className="grid h-16 w-24 place-items-center rounded-md border border-dashed border-line-strong text-xs text-ink-3">No view</div>
          )}
          <div className="min-w-0 flex-1 basis-48">
            <h2 className="font-medium">Reconstruction input</h2>
            <p className="text-sm text-ink-2">
              {input?.kind === 'photo' && 'Your uploaded photo.'}
              {input?.kind === 'reference' && 'The reference marked below.'}
              {!input && 'Upload a reference, or upload a photo first.'}
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            {photo && input?.kind !== 'photo' && (
              <button type="button" className={secondaryButton} onClick={() => setInput({ kind: 'photo' })}>
                Use my photo
              </button>
            )}
            <button type="button" disabled={!input} className={primaryButton} onClick={() => navigate('/studio')}>
              Continue to 3D Studio
            </button>
          </div>
        </div>
        <p className="mt-3 border-t border-line pt-3 text-xs text-ink-3">
          TripoSR uses one image. Other references aren't fused into the geometry: they sit beside the mesh for comparison. References show the same model, not the same physical car, so trim, generation and bodywork may differ.
        </p>
      </Card>

      {retrieval.status === 'loading' && classId != null && (
        <div className="mb-6">
          <Spinner label="Searching Wikimedia Commons" />
        </div>
      )}
      {retrieval.status === 'error' && (
        <div className="mb-6">
          <ErrorNotice title="Automatic search failed" message={`${retrieval.error} You can still upload your own references.`} onRetry={retrieval.reload} />
        </div>
      )}
      {retrieval.status === 'ready' && classId != null && retrieval.data.candidates.length === 0 && (
        <p className="mb-6 rounded-lg border border-line bg-surface p-3 text-sm text-ink-2">
          No openly licensed references passed the checks for this model. You can upload your own.
        </p>
      )}

      {GROUPS.map(({ key, label }) => {
        const items = all.filter((reference) => groupOf(reference) === key)
        if (!items.length) return null
        return (
          <section key={key} className="mb-8">
            <h2 className="mb-3 flex items-baseline gap-2 text-lg font-medium">
              {label}
              <span className="num rounded border border-line px-1.5 font-mono text-xs font-normal text-accent">{String(items.length).padStart(2, '0')}</span>
            </h2>
            <ul className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {items.map((reference, index) => (
                <ReferenceCard
                  key={reference.id}
                  index={index}
                  reference={reference}
                  group={key}
                  onMove={(group) => setMoved((previous) => ({ ...previous, [reference.id]: group }))}
                />
              ))}
            </ul>
          </section>
        )
      })}
    </div>
  )
}
