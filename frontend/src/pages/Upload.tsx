import { useEffect, useId, useRef, useState, type DragEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Camera, ImageUp } from 'lucide-react'
import { ApiError, recognize } from '../api/client'
import { ErrorNotice, PageHeader, Spinner, primaryButton, secondaryButton } from '../components/States'
import { checkFile, normaliseImage, type NormalisedImage } from '../lib/image'
import { useSession } from '../state/session'

type Stage = { name: 'idle' } | { name: 'preparing' } | { name: 'ready'; photo: NormalisedImage } | { name: 'analysing'; photo: NormalisedImage }

export default function Upload() {
  const inputId = useId()
  const cameraId = useId()
  const navigate = useNavigate()
  const { startWithPhoto } = useSession()
  const [stage, setStage] = useState<Stage>({ name: 'idle' })
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const abort = useRef<AbortController | null>(null)

  useEffect(() => {
    if (stage.name !== 'analysing') return
    setElapsed(0)
    const started = Date.now()
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [stage.name])

  useEffect(() => () => abort.current?.abort(), [])

  async function choose(file: File | undefined) {
    if (!file) return
    setError(null)
    const problem = checkFile(file)
    if (problem) return setError(problem)
    setStage({ name: 'preparing' })
    try {
      setStage({ name: 'ready', photo: await normaliseImage(file) })
    } catch (reason) {
      setStage({ name: 'idle' })
      setError(reason instanceof Error ? reason.message : "This photo couldn't be read.")
    }
  }

  async function analyse(photo: NormalisedImage) {
    setError(null)
    setStage({ name: 'analysing', photo })
    abort.current = new AbortController()
    try {
      const recognition = await recognize(photo.file, abort.current.signal)
      startWithPhoto(photo, recognition)
      navigate('/identity')
    } catch (reason) {
      setStage({ name: 'ready', photo })
      if (reason instanceof ApiError && reason.kind === 'aborted') return
      setError(reason instanceof ApiError ? reason.message : 'Something went wrong while analysing the photo.')
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault()
    setDragging(false)
    void choose(event.dataTransfer.files[0])
  }

  function reset() {
    if (stage.name === 'ready') URL.revokeObjectURL(stage.photo.url)
    setError(null)
    setStage({ name: 'idle' })
  }

  const photo = stage.name === 'ready' || stage.name === 'analysing' ? stage.photo : null

  return (
    <div className="mx-auto max-w-3xl">
      <PageHeader title="Upload a photo" intro="One car, reasonably in frame, works best. JPEG, PNG or WebP, up to 20 MB." />

      {error && (
        <div className="mb-4">
          <ErrorNotice title="That didn't work" message={error} />
        </div>
      )}

      {photo ? (
        <div>
          <div className="overflow-hidden rounded-xl border border-line bg-surface">
            <img src={photo.url} alt="The photo you chose" className="mx-auto block max-h-[60dvh] w-auto max-w-full" />
          </div>
          {stage.name === 'analysing' ? (
            <div className="mt-4 flex flex-wrap items-center justify-between gap-3" aria-live="polite">
              <Spinner label={`Detecting and classifying on the server, ${elapsed} s`} />
              <button type="button" className={secondaryButton} onClick={() => abort.current?.abort()}>
                Cancel
              </button>
            </div>
          ) : (
            <div className="mt-4 flex flex-wrap gap-3">
              <button type="button" className={primaryButton} onClick={() => void analyse(photo)}>
                Identify this car
              </button>
              <button type="button" className={secondaryButton} onClick={reset}>
                Choose another
              </button>
            </div>
          )}
        </div>
      ) : (
        <div
          onDragOver={(event) => {
            event.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={`rounded-xl border border-dashed px-6 py-16 text-center transition-colors ${
            dragging ? 'border-accent bg-accent/8' : 'border-line-strong bg-surface'
          }`}
        >
          {stage.name === 'preparing' ? (
            <Spinner label="Preparing photo" />
          ) : (
            <>
              <ImageUp aria-hidden className="mx-auto size-8 text-ink-3" strokeWidth={1.5} />
              <p className="mt-3 text-ink-2 touch:hidden">Drop a photo here, or</p>
              <div className="mt-4 flex flex-wrap justify-center gap-3">
                <label htmlFor={inputId} className={`${primaryButton} focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-accent`}>
                  Choose a photo
                  <input
                    id={inputId}
                    type="file"
                    accept="image/*"
                    className="sr-only"
                    onChange={(event) => {
                      void choose(event.target.files?.[0])
                      event.target.value = ''
                    }}
                  />
                </label>
                <label htmlFor={cameraId} className={`${secondaryButton} hidden touch:inline-flex`}>
                  <Camera aria-hidden className="size-4" />
                  Take a photo
                  <input
                    id={cameraId}
                    type="file"
                    accept="image/*"
                    capture="environment"
                    className="sr-only"
                    onChange={(event) => {
                      void choose(event.target.files?.[0])
                      event.target.value = ''
                    }}
                  />
                </label>
              </div>
            </>
          )}
        </div>
      )}

      <p className="mt-4 text-xs text-ink-3">
        Your photo goes to the team's host laptop for processing and is deleted afterwards. Photos may show people or number plates, so choose accordingly.
      </p>
    </div>
  )
}
