import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { CircleCheck, CircleMinus, Download, Library, TriangleAlert } from 'lucide-react'
import { ApiError, assetUrl, getJob, reconstruct } from '../api/client'
import type { Job, JobResult, ValidationReport } from '../api/types'
import { Attribution } from '../components/Attribution'
import MeshViewer from '../components/MeshViewer'
import { Card, EmptyState, ErrorNotice, PageHeader, PageLoading, Spinner, hudPrimary as primaryButton, hudSecondary as secondaryButton } from '../components/States'
import { PhotoUpload } from '../components/PhotoUpload'
import { ScanFrame } from '../components/hud/ScanFrame'
import { rise, stagger } from '../components/hud/PageShell'
import { bytes } from '../lib/format'
import { useSession } from '../state/session'

function useStartJob() {
  const { photo, input, identity } = useSession()
  const navigate = useNavigate()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const classId = identity?.class_id ?? null
  const source =
    input?.kind === 'photo' && photo ? { image: photo.file, classId } : input?.kind === 'reference' ? { reference: input.reference, classId } : null

  async function start() {
    if (!source) return
    setBusy(true)
    setError(null)
    try {
      const { job_id } = await reconstruct(source)
      navigate(`/studio/${encodeURIComponent(job_id)}`)
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : 'The job could not be started.')
    } finally {
      setBusy(false)
    }
  }
  return { canStart: source != null, start, busy, error }
}

/** Says what the mesh was made from, so a ready-made model is never mistaken for a reconstruction of the user's photo. */
function SourceNote({ source, label }: { source?: Job['mesh_source']; label?: string }) {
  const car = label ?? 'the matched car'
  if (source === 'catalog') {
    return (
      <p className="flex gap-2 text-sm text-ink-2">
        <Library aria-hidden className="mt-0.5 size-4 shrink-0 text-accent" />
        <span>Catalog model of {car}. It was made in advance, not built from your photo.</span>
      </p>
    )
  }
  return (
    <p className="flex gap-2 text-sm text-ink-2">
      <TriangleAlert aria-hidden className="mt-0.5 size-4 shrink-0 text-warning" />
      <span>
        {source === 'reference' ? `Built from a reference photo of ${car}, not from your photo. ` : ''}This is a single-view reconstruction. The sides the photo doesn't show are inferred, proportions aren't
        measured, and the scale is unknown.
      </span>
    </p>
  )
}

function SingleViewNote() {
  return <SourceNote />
}

function StartPanel() {
  const { photo, identity, input } = useSession()
  const { canStart, start, busy, error } = useStartJob()
  const inputUrl = input?.kind === 'photo' ? photo?.url : input?.reference.image_url

  if (!canStart || !inputUrl) {
    return (
      <div className="mx-auto max-w-3xl">
        <PageHeader eyebrow="05 / 3D Studio" title="3D Studio" intro="Add a car photo to build a draft mesh from. You can also start from a reference photo instead." />
        <PhotoUpload />
        <p className="mt-6 text-sm text-ink-3">
          Prefer a reference photo?{' '}
          <Link to="/references" className="text-ink underline underline-offset-4">
            Open Reference Studio
          </Link>
        </p>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-3xl">
      <PageHeader eyebrow="05 / 3D Studio" title="3D Studio" intro={identity ? `Generate a draft mesh of ${identity.label} from the view below.` : 'Generate a draft mesh from the view below.'} />
      <ScanFrame>
        <img src={inputUrl} alt="Reconstruction input" className="mx-auto block max-h-[50dvh] w-auto max-w-full" />
      </ScanFrame>
      {input?.kind === 'reference' && (
        <div className="mt-2">
          <Attribution reference={input.reference} />
        </div>
      )}
      <div className="mt-4">
        <SingleViewNote />
      </div>
      {error && (
        <div className="mt-4">
          <ErrorNotice title="Couldn't start the job" message={error} />
        </div>
      )}
      <div className="mt-5 flex flex-wrap gap-3">
        <button type="button" disabled={busy} className={primaryButton} onClick={() => void start()}>
          {busy ? 'Starting' : 'Generate mesh'}
        </button>
        <Link to="/references" className={secondaryButton}>
          Choose a different view
        </Link>
      </div>
    </div>
  )
}

/** Polls the job until it finishes. Keeps going through brief network drops, because the job continues on the server. */
function useJob(id: string) {
  const [job, setJob] = useState<Job | null>(null)
  const [lost, setLost] = useState(false)
  const [fatal, setFatal] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    let timer: ReturnType<typeof setTimeout>
    setJob(null)
    setFatal(null)
    async function poll() {
      try {
        const next = await getJob(id, controller.signal)
        setJob(next)
        setLost(false)
        if (next.status === 'queued' || next.status === 'running') timer = setTimeout(poll, 1500)
      } catch (reason) {
        if (controller.signal.aborted) return
        if (reason instanceof ApiError && (reason.kind === 'parse' || (reason.kind === 'http' && reason.status !== undefined && reason.status < 500))) {
          setFatal(reason.message)
        } else {
          setLost(true)
          timer = setTimeout(poll, 3000)
        }
      }
    }
    void poll()
    return () => {
      controller.abort()
      clearTimeout(timer)
    }
  }, [id])

  return { job, lost, fatal }
}

function JobProgress({ job, lost }: { job: Job; lost: boolean }) {
  return (
    <Card>
      <div aria-live="polite" className="flex flex-wrap items-center justify-between gap-4">
        {job.status === 'queued' ? (
          <Spinner label={job.queue_position != null ? `In the queue, position ${job.queue_position}` : 'In the queue'} />
        ) : (
          <Spinner label={job.stage ?? 'Generating mesh'} />
        )}
        {job.status === 'queued' && job.queue_position != null && (
          <p className="num font-mono text-4xl font-semibold text-accent" aria-hidden>
            #{job.queue_position}
          </p>
        )}
      </div>
      <div className="scanbar mt-4" />
      {lost && <p className="mt-2 text-sm text-warning">Connection lost. Retrying. The job carries on while this page reconnects.</p>}
      <JobLog logs={job.logs} />
    </Card>
  )
}

function JobLog({ logs }: { logs?: string[] }) {
  if (!logs?.length) return null
  return (
    <ol className="mt-4 max-h-56 overflow-y-auto rounded-lg border border-line bg-page p-3 font-mono text-xs text-ink-2">
      {logs.map((line, index) => (
        <motion.li
          key={index}
          initial={{ opacity: 0, x: -6 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.25 }}
          className={`py-0.5 ${index === logs.length - 1 ? 'cursor-blink text-ink' : ''}`}
        >
          <span aria-hidden className="mr-2 text-accent">
            ›
          </span>
          {line}
        </motion.li>
      ))}
    </ol>
  )
}

function Check({ label, value, good }: { label: string; value: string; good?: boolean | null }) {
  const Icon = good ? CircleCheck : CircleMinus
  return (
    <motion.div variants={rise} className="flex items-center justify-between gap-4 py-2 text-sm">
      <dt className="flex items-center gap-2 text-ink-2">
        {good != null && <Icon aria-hidden className={`size-4 ${good ? 'text-good' : 'text-warning'}`} />}
        {label}
      </dt>
      <dd className="num font-mono text-right">{value}</dd>
    </motion.div>
  )
}

const NOT_REPORTED = 'Not reported'
const count = (value?: number | null) => (value == null ? NOT_REPORTED : value.toLocaleString())
const yesNo = (value?: boolean | null) => (value == null ? NOT_REPORTED : value ? 'Yes' : 'No')
const zeroIsGood = (value?: number | null) => (value == null ? null : value === 0)

const READINESS: Record<string, string> = { draft: 'Draft', needs_repair: 'Needs repair', unvalidated: 'Unvalidated' }

function ValidationRows({ report, seconds }: { report: ValidationReport | null | undefined; seconds?: number | null }) {
  return (
    <motion.dl variants={stagger} initial="hidden" animate="show" className="divide-y divide-line">
      <Check label="Asset readiness" value={report?.readiness ? (READINESS[report.readiness] ?? report.readiness) : NOT_REPORTED} />
      <Check label="Watertight" value={yesNo(report?.watertight)} good={report?.watertight} />
      <Check label="Manifold" value={yesNo(report?.manifold)} good={report?.manifold} />
      <Check label="Degenerate faces" value={count(report?.degenerate_faces)} good={zeroIsGood(report?.degenerate_faces)} />
      <Check
        label="Connected components"
        value={count(report?.connected_components)}
        good={report?.connected_components == null ? null : report.connected_components === 1}
      />
      <Check label="Vertices" value={count(report?.vertices)} />
      <Check label="Faces" value={count(report?.faces)} />
      <Check label="Texture" value={yesNo(report?.has_texture)} />
      <Check label="Vertex colours" value={yesNo(report?.has_vertex_colors)} />
      <Check label="Generation time" value={seconds == null ? NOT_REPORTED : `${seconds.toFixed(1)} s`} />
      <Check label="Real-world scale" value="Unknown" />
    </motion.dl>
  )
}

function Validation({ result }: { result: JobResult }) {
  return (
    <Card title="Mesh validation">
      <ValidationRows report={result.validation} seconds={result.generation_seconds} />
      {result.validation?.notes && result.validation.notes.length > 0 && (
        <ul className="mt-3 space-y-1 border-t border-line pt-3 text-xs text-ink-2">
          {result.validation.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
      <p className="mt-3 text-xs text-ink-3">
        These checks describe the file, not how accurate the shape is. No ground-truth 3D exists to compare against, so nothing here makes the mesh print-ready. Games need retopology, UVs and LODs. Printing needs repair, scaling and wall-thickness checks.
      </p>
    </Card>
  )
}

const FILE_LABEL: Record<string, string> = { glb: 'GLB', obj: 'OBJ', stl: 'STL', fbx: 'FBX', json: 'Validation report' }

function Downloads({ result }: { result: JobResult }) {
  return (
    <Card title="Downloads">
      <ul className="grid gap-2">
        {result.files.map((file) => (
          <li key={file.name}>
            <a href={assetUrl(result.asset_id, file.name)} download={file.name} className={`${secondaryButton} w-full justify-between`}>
              <span className="inline-flex items-center gap-2">
                <Download aria-hidden className="size-4" />
                {FILE_LABEL[file.format] ?? file.format.toUpperCase()}
              </span>
              {file.size_bytes != null && <span className="num font-mono text-xs font-normal text-ink-3">{bytes(file.size_bytes)}</span>}
            </a>
          </li>
        ))}
      </ul>
    </Card>
  )
}

function Compare() {
  const { photo, input, pack } = useSession()
  const inputUrl = input?.kind === 'photo' ? photo?.url : input?.reference.image_url
  if (!inputUrl && !pack.length) return null
  return (
    <section>
      <h2 className="mb-3 font-medium">Compare with your references</h2>
      <ul className="grid grid-cols-2 gap-3 md:grid-cols-4">
        {inputUrl && (
          <li>
            <img src={inputUrl} alt="The input view" className="aspect-[4/3] w-full rounded-lg bg-raised object-cover" />
            <p className="mt-1 text-xs text-ink-3">Input view</p>
          </li>
        )}
        {pack.map((reference) => (
          <li key={reference.id}>
            <img src={reference.thumb_url ?? reference.image_url} alt={`Reference from ${reference.source}`} className="aspect-[4/3] w-full rounded-lg bg-raised object-cover" />
            <div className="mt-1">
              <Attribution reference={reference} />
            </div>
          </li>
        ))}
      </ul>
    </section>
  )
}

function JobView({ id }: { id: string }) {
  const { identity } = useSession()
  const { job, lost, fatal } = useJob(id)
  const { canStart, start, busy, error } = useStartJob()

  if (fatal) {
    return <EmptyState title="Job not available" body={`${fatal} Generated files are removed by a cleanup step, so older links stop working.`} action={{ to: '/studio', label: 'Back to 3D Studio' }} />
  }
  if (!job) {
    return lost ? (
      <ErrorNotice title="Can't reach the server" message="Retrying. The host laptop may be offline." />
    ) : (
      <PageLoading label="Checking job" />
    )
  }

  const glb = job.result?.files.find((file) => file.format === 'glb')

  return (
    <div>
      <PageHeader eyebrow="05 / 3D Studio" title="3D Studio" intro={identity ? (job.mesh_source === 'catalog' ? `Catalog model of ${identity.label}.` : `Draft mesh of ${identity.label}.`) : undefined} />

      {(job.status === 'queued' || job.status === 'running') && <JobProgress job={job} lost={lost} />}

      {job.status === 'failed' && (
        <div className="space-y-4">
          <ErrorNotice title="Generation failed" message={job.error ?? 'The server did not report a reason.'} />
          <JobLog logs={job.logs} />
          {error && <ErrorNotice title="Couldn't start the job" message={error} />}
          <div className="flex flex-wrap gap-3">
            {canStart && (
              <button type="button" disabled={busy} className={primaryButton} onClick={() => void start()}>
                {busy ? 'Starting' : 'Try again'}
              </button>
            )}
            <Link to="/references" className={secondaryButton}>
              Choose a different view
            </Link>
          </div>
        </div>
      )}

      {job.status === 'succeeded' && job.result && (
        <div className="grid gap-6 lg:grid-cols-[1.7fr_1fr] lg:items-start">
          <div className="space-y-5">
            {glb ? (
              <MeshViewer url={assetUrl(job.result.asset_id, glb.name)} />
            ) : (
              <ErrorNotice title="No GLB to display" message="The job finished without a GLB file. Any other formats are listed under Downloads." />
            )}
            {glb && <p className="font-mono text-xs text-ink-3">{job.mesh_source === 'catalog' ? 'Catalog file' : 'Generated file'}: {glb.name}</p>}
            <SourceNote source={job.mesh_source} label={identity?.label} />
            <Compare />
          </div>
          <div className="space-y-5">
            <Downloads result={job.result} />
            <Validation result={job.result} />
          </div>
        </div>
      )}
    </div>
  )
}

export default function Studio() {
  const { jobId } = useParams()
  return jobId ? <JobView key={jobId} id={jobId} /> : <StartPanel />
}
