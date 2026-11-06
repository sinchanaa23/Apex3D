import { useEffect, useId, useRef, useState, type DragEvent } from 'react'
import { Camera, ImageUp } from 'lucide-react'
import { ApiError, recognize } from '../api/client'
import { ScanFrame } from './hud/ScanFrame'
import { ErrorNotice, Spinner, hudPrimary, hudSecondary } from './States'
import { checkFile, normaliseImage, type NormalisedImage } from '../lib/image'
import { useSession } from '../state/session'

type Stage = { name: 'idle' } | { name: 'preparing' } | { name: 'ready'; photo: NormalisedImage } | { name: 'analysing'; photo: NormalisedImage }

/**
 * Choose a photo and identify it. Pages that need a photo show this in place, so nobody is sent to another page to upload.
 * When identification succeeds the photo and result are in the session, and `onDone` runs.
 */
export function PhotoUpload({ onDone, className = '' }: { onDone?: () => void; className?: string }) {
  const inputId = useId()
  const cameraId = useId()
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
      const recognition = await recognize(photo.file, { width: photo.width, height: photo.height }, abort.current.signal)
      startWithPhoto(photo, recognition)
      onDone?.()
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
  const analysing = stage.name === 'analysing'

  return (
    <div className={className}>
      {error && (
        <div className="mb-4">
          <ErrorNotice title="That didn't work" message={error} />
        </div>
      )}

      {photo ? (
        <div>
          <ScanFrame active={analysing} scan={analysing}>
            <img src={photo.url} alt="The photo you chose" className={`mx-auto block max-h-[60dvh] w-auto max-w-full transition-[filter] duration-500 ${analysing ? 'brightness-75' : ''}`} />
            {analysing && (
              <span className="absolute top-3 left-3 rounded border border-accent/60 bg-page/80 px-2 py-1 font-mono text-xs tracking-widest text-accent uppercase">
                Analysing
              </span>
            )}
          </ScanFrame>
          {analysing ? (
            <div className="mt-4" aria-live="polite">
              <div className="scanbar mb-3" />
              <div className="flex flex-wrap items-center justify-between gap-3">
                <Spinner label={`Detecting and classifying on the server, ${elapsed} s`} />
                <button type="button" className={hudSecondary} onClick={() => abort.current?.abort()}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="mt-4 flex flex-wrap gap-3">
              <button type="button" className={hudPrimary} onClick={() => void analyse(photo)}>
                Identify this car
              </button>
              <button type="button" className={hudSecondary} onClick={reset}>
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
          className={`relative overflow-hidden rounded-xl border border-dashed px-6 py-14 text-center transition-all duration-300 ${
            dragging ? 'border-accent bg-accent/10 shadow-[0_0_40px_-8px_rgb(255_106_43/0.5)]' : 'border-line-strong bg-surface'
          }`}
        >
          <span aria-hidden className="hud-corner tl" />
          <span aria-hidden className="hud-corner tr" />
          <span aria-hidden className="hud-corner bl" />
          <span aria-hidden className="hud-corner br" />
          {stage.name === 'preparing' ? (
            <div className="mx-auto max-w-xs">
              <Spinner label="Preparing photo" />
              <div className="scanbar mt-4" />
            </div>
          ) : (
            <>
              <div className="relative mx-auto grid size-24 place-items-center">
                <span aria-hidden className="radar-ring" />
                <span aria-hidden className="radar-ring" style={{ animationDelay: '1.6s' }} />
                <span aria-hidden className="radar-sweep" />
                <ImageUp aria-hidden className="relative size-8 text-accent" strokeWidth={1.5} />
              </div>
              <p className="mt-4 text-ink-2 touch:hidden">Drop a photo here, or</p>
              <div className="mt-4 flex flex-wrap justify-center gap-3">
                <label htmlFor={inputId} className={`${hudPrimary} focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-accent`}>
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
                <label htmlFor={cameraId} className={`${hudSecondary} hidden touch:inline-flex`}>
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
        Your photo goes to the team's host laptop for processing and is deleted afterwards. It stays in this tab's memory only, so reloading the page clears it. Photos may show people or number plates, so choose accordingly.
      </p>
    </div>
  )
}
