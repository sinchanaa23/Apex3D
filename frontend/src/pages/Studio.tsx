import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { CircleCheck, CircleMinus, Download, TriangleAlert } from 'lucide-react'
import { ApiError, assetUrl, getJob, getValidation, reconstruct } from '../api/client'
import type { Job, JobResult, ValidationReport } from '../api/types'
import { Attribution } from '../components/Attribution'
import MeshViewer from '../components/MeshViewer'
import { Card, EmptyState, ErrorNotice, PageHeader, PageLoading, Spinner, primaryButton, secondaryButton } from '../components/States'
import { bytes } from '../lib/format'
import { useResource } from '../lib/useResource'
import { useSession } from '../state/session'

function useStartJob() {
  const { photo, identity, input } = useSession()
  const navigate = useNavigate()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const source = input?.kind === 'photo' && photo ? { image: photo.file } : input?.kind === 'reference' ? { referenceId: input.reference.id } : null

  async function start() {
    if (!source) return
    setBusy(true)
    setError(null)
    try {
      const { job_id } = await reconstruct(source, identity?.class_id)
      navigate(`/studio/${encodeURIComponent(job_id)}`)
    } catch (reason) {
      setError(reason instanceof ApiError ? reason.message : 'The job could not be started.')
    } finally {
      setBusy(false)
    }
  }
  return { canStart: source != null, start, busy, error }
}

function SingleViewNote() {
  return (
    <p className="flex gap-2 text-sm text-ink-2">
      <TriangleAlert aria-hidden className="mt-0.5 size-4 shrink-0 text-warning" />
      <span>This is a single-view reconstruction. The sides the photo doesn't show are inferred, proportions aren't measured, and the scale is unknown.</span>
    </p>
  )
}

function StartPanel() {
  const { photo, identity, input } = useSession()
  const { canStart, start, busy, error } = useStartJob()
  const inputUrl = input?.kind === 'photo' ? photo?.url : input?.reference.image_url

  if (!canStart || !inputUrl) {
    return (
      <EmptyState
        title="Nothing to reconstruct yet"
        body="Upload a car photo, or add a reference, and choose the view to build a mesh from."
        action={{ to: '/upload', label: 'Upload a photo' }}
      />
    )
  }

  return (
    <div className="mx-auto max-w-3xl">
      <PageHeader title="3D Studio" intro={identity ? `Generate a draft mesh of ${identity.label} from the view below.` : 'Generate a draft mesh from the view below.'} />
      <div className="overflow-hidden rounded-xl border border-line bg-surface">
        <img src={inputUrl} alt="Reconstruction input" className="mx-auto block max-h-[50dvh] w-auto max-w-full" />
      </div>
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
        if (reason instanceof ApiError && (reason.kind === 'http' || reason.kind === 'parse') && reason.status !== undefined && reason.status < 500) {
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
      <div aria-live="polite">
        {job.status === 'queued' ? (
          <Spinner label={job.queue_position != null ? `In the queue, position ${job.queue_position}` : 'In the queue'} />
        ) : (
          <Spinner label={job.stage ?? 'Generating mesh'} />
        )}
      </div>
      {lost && <p className="mt-2 text-sm text-warning">Connection lost. Retrying. The job carries on while this page reconnects.</p>}
      <JobLog logs={job.logs} />
    </Card>
  )
}

function JobLog({ logs }: { logs?: string[] }) {
  if (!logs?.length) return null
  return (
    <ol className="mt-4 max-h-56 overflow-y-auto rounded-lg bg-page p-3 font-mono text-xs text-ink-2">
      {logs.map((line, index) => (
        <li key={index} className="py-0.5">
          {line}
        </li>
      ))}
    </ol>
  )
}

function Check({ label, value, good }: { label: string; value: string; good?: boolean | null }) {
  const Icon = good ? CircleCheck : CircleMinus
  return (
    <div className="flex items-center justify-between gap-4 py-2 text-sm">
      <dt className="flex items-center gap-2 text-ink-2">
        {good != null && <Icon aria-hidden className={`size-4 ${good ? 'text-good' : 'text-warning'}`} />}
        {label}
      </dt>
      <dd className="num font-mono text-right">{value}</dd>
    </div>
  )
}

const NOT_REPORTED = 'Not reported'
const count = (value?: number | null) => (value == null ? NOT_REPORTED : value.toLocaleString())
const yesNo = (value?: boolean | null) => (value == null ? NOT_REPORTED : value ? 'Yes' : 'No')
const zeroIsGood = (value?: number | null) => (value == null ? null : value === 0)

function ValidationRows({ report, seconds }: { report: ValidationReport | null; seconds?: number | null }) {
  return (
    <dl className="divide-y divide-line">
      <Check label="Watertight" value={yesNo(report?.watertight)} good={report?.watertight} />
      <Check label="Non-manifold edges" value={count(report?.non_manifold_edges)} good={zeroIsGood(report?.non_manifold_edges)} />
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
    </dl>
  )
}

function Validation({ result }: { result: JobResult }) {
  const reportFile = result.files.find((file) => file.format === 'json')
  const report = useResource(
    (signal) => (reportFile ? getValidation(result.asset_id, reportFile.name, signal) : Promise.resolve(null)),
    `${result.asset_id}/${reportFile?.name}`,
  )
  return (
    <Card title="Mesh validation">
      {report.status === 'loading' && <Spinner label="Reading validation report" />}
      {report.status === 'error' && <ErrorNotice title="Report unavailable" message={report.error} onRetry={report.reload} />}
      {report.status === 'ready' && <ValidationRows report={report.data} seconds={result.generation_seconds} />}
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
            <img src={reference.image_url} alt={`Reference from ${reference.source}`} className="aspect-[4/3] w-full rounded-lg bg-raised object-cover" />
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
      <PageHeader title="3D Studio" intro={identity ? `Draft mesh of ${identity.label}.` : undefined} />

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
            {glb && <p className="font-mono text-xs text-ink-3">Generated file: {glb.name}</p>}
            <SingleViewNote />
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
