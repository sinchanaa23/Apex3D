import {
  adaptClassical,
  adaptJob,
  adaptMetrics,
  adaptRecognition,
  adaptReference,
  adaptReferences,
  withMeasuredCalibration,
  type WireClassical,
  type WireJob,
  type WireMetrics,
  type WireRecognition,
  type WireReference,
} from './adapter'
import { UNAUTHORIZED_EVENT } from './auth'
import { normaliseImage } from '../lib/image'
import type { MetricsResponse, Reference } from './types'

export type ApiErrorKind = 'network' | 'timeout' | 'http' | 'parse' | 'aborted'

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status?: number

  constructor(kind: ApiErrorKind, message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
  }
}

const OFFLINE = "Can't reach the Apex3D server. The host laptop may be offline, or the public link may have changed."
const UNREADABLE = 'The server sent a response this page could not read.'

/** Turns a failed response into the message shown to the user. */
export function errorMessage(status: number, body: string): string {
  try {
    const { detail } = JSON.parse(body) as { detail?: unknown }
    if (typeof detail === 'string' && detail) return detail
    if (Array.isArray(detail)) {
      const parts = detail.map((d: { msg?: string }) => d?.msg).filter(Boolean)
      if (parts.length) return parts.join('. ')
    }
  } catch {
    // Not JSON. The tunnel returns HTML pages when the laptop stops answering.
  }
  if (status === 401) return 'Please log in to continue.'
  if (status === 413) return 'That file is too large for the server to accept.'
  if (status === 415) return 'The server does not accept that file type.'
  if (status === 404) return 'The server could not find that item. It may have been cleaned up.'
  if (status === 429) return 'The server is busy. Wait a moment and try again.'
  // 502 to 504 from any proxy, 520 and up from Cloudflare when the tunnel can't reach the laptop.
  if ((status >= 502 && status <= 504) || status >= 520) return OFFLINE
  return `The server returned an error (${status}).`
}

interface Options {
  signal?: AbortSignal
  timeoutMs?: number
}

async function request<T>(path: string, init: RequestInit, { signal, timeoutMs = 30_000 }: Options): Promise<T> {
  const controller = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, timeoutMs)
  const onAbort = () => controller.abort()
  signal?.addEventListener('abort', onAbort)

  try {
    const response = await fetch(path, { ...init, signal: controller.signal })
    const body = await response.text()
    // The session ended or there is none. The app listens for this and sends the user to the login page.
    if (response.status === 401) window.dispatchEvent(new Event(UNAUTHORIZED_EVENT))
    if (!response.ok) throw new ApiError('http', errorMessage(response.status, body), response.status)
    try {
      return JSON.parse(body) as T
    } catch {
      throw new ApiError('parse', UNREADABLE, response.status)
    }
  } catch (error) {
    if (error instanceof ApiError) throw error
    if (timedOut) throw new ApiError('timeout', 'The server took too long to respond. Try again.')
    if (signal?.aborted) throw new ApiError('aborted', 'Cancelled.')
    throw new ApiError('network', OFFLINE)
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

/** Runs a translation, reporting a response in the wrong shape as a readable error. */
function adapted<W, T>(wire: W, adapt: (wire: W) => T): T {
  try {
    return adapt(wire)
  } catch {
    throw new ApiError('parse', UNREADABLE)
  }
}

function fileForm(file: Blob, name = 'photo.jpg'): FormData {
  const form = new FormData()
  form.append('file', file, name)
  return form
}

export async function recognize(image: Blob, size: { width: number; height: number }, signal?: AbortSignal) {
  const wire = await request<WireRecognition>('/api/recognize', { method: 'POST', body: fileForm(image) }, { signal, timeoutMs: 90_000 })
  return adapted(wire, (w) => adaptRecognition(w, size))
}

export async function classical(image: Blob, signal?: AbortSignal) {
  const wire = await request<WireClassical>('/api/classical', { method: 'POST', body: fileForm(image) }, { signal, timeoutMs: 60_000 })
  return adapted(wire, adaptClassical)
}

export async function getReferences(classId: number, signal?: AbortSignal) {
  const wire = await request<{ query?: string | null; candidates: WireReference[] }>(`/api/references?class_id=${classId}`, {}, { signal, timeoutMs: 90_000 })
  return adapted(wire, adaptReferences)
}

/** The endpoint takes one file at a time. */
export async function uploadReferences(files: File[], signal?: AbortSignal) {
  const candidates = await Promise.all(
    files.map(async (file) => {
      const wire = await request<{ candidate: WireReference }>(
        '/api/references/upload',
        { method: 'POST', body: fileForm(file, file.name) },
        { signal, timeoutMs: 90_000 },
      )
      return adapted(wire.candidate, adaptReference)
    }),
  )
  return { candidates }
}

export type ReconstructInput = ({ image: Blob } | { reference: Reference }) & {
  /** The class the user confirmed. The server uses it to find a catalog model or an earlier mesh for the same car. */
  classId?: number | null
}

async function reconstructForm(input: ReconstructInput): Promise<FormData> {
  const form = await imageForm(input)
  if (input.classId != null) form.append('class_id', String(input.classId))
  form.append('source', 'reference' in input ? 'reference' : 'photo')
  return form
}

async function imageForm(input: ReconstructInput): Promise<FormData> {
  if ('image' in input) return fileForm(input.image, 'input.jpg')

  // The browser fetches the reference and uploads the file itself, so the server never has to fetch a web address on our behalf.
  const response = await fetch(input.reference.image_url).catch(() => null)
  if (!response?.ok) throw new ApiError('network', "That reference image couldn't be loaded. Choose another, or upload your own.")
  const blob = await response.blob()
  const prepared = await normaliseImage(new File([blob], 'reference.jpg', { type: blob.type || 'image/jpeg' })).catch(() => null)
  if (!prepared) throw new ApiError('parse', "That reference image couldn't be read. Choose another, or upload your own.")
  URL.revokeObjectURL(prepared.url)
  return fileForm(prepared.file, 'input.jpg')
}

export async function reconstruct(input: ReconstructInput, signal?: AbortSignal) {
  return request<{ job_id: string }>('/api/reconstruct', { method: 'POST', body: await reconstructForm(input) }, { signal, timeoutMs: 60_000 })
}

export async function getJob(id: string, signal?: AbortSignal) {
  const wire = await request<WireJob>(`/api/jobs/${encodeURIComponent(id)}`, {}, { signal, timeoutMs: 15_000 })
  return adapted(wire, adaptJob)
}

export function assetUrl(assetId: string, file: string) {
  return `/api/assets/${encodeURIComponent(assetId)}/${encodeURIComponent(file)}`
}

/**
 * Fetches a saved result file that was copied next to the app when it was built (see vite.config.ts).
 * Returns null for anything that isn't JSON, because the server answers unknown paths with the app's own page.
 */
async function savedResult(name: string, signal?: AbortSignal): Promise<Record<string, unknown> | null> {
  try {
    const response = await fetch(`/results/${name}`, { signal })
    if (!response.ok) return null
    const parsed: unknown = JSON.parse(await response.text())
    return typeof parsed === 'object' && parsed !== null ? (parsed as Record<string, unknown>) : null
  } catch {
    return null
  }
}

const hasSection = (metrics: MetricsResponse) => Object.entries(metrics).some(([key, value]) => key !== 'message' && value != null && value !== 'not available')

/**
 * The backend's /api/metrics is the first choice. Any section it has no data for is filled from the result files
 * Person A saved under results/, so the Research Lab shows the trained model's real numbers even while the backend
 * does not read that folder. Nothing is invented: a section with no file behind it stays "not available".
 */
export async function getMetrics(signal?: AbortSignal): Promise<MetricsResponse> {
  let fromApi: MetricsResponse | null = null
  let apiError: unknown = null
  try {
    fromApi = adapted(await request<WireMetrics>('/api/metrics', {}, { signal }), adaptMetrics)
  } catch (error) {
    if (error instanceof ApiError && error.kind === 'aborted') throw error
    apiError = error
  }

  const [classifier, optimisation] = await Promise.all([savedResult('classifier_metrics.json', signal), savedResult('optimization_metrics.json', signal)])
  if (!classifier && !optimisation) {
    if (fromApi) return fromApi
    throw apiError
  }

  const fromFiles = adapted(
    { status: 'available', source_file: 'results/', metrics: { classifier_metrics: classifier, optimization_metrics: optimisation } } as WireMetrics,
    adaptMetrics,
  )
  const merged: MetricsResponse = { ...fromFiles, message: fromApi?.message }
  for (const [key, value] of Object.entries(fromApi ?? {})) {
    if (key !== 'message' && value != null && value !== 'not available') (merged as Record<string, unknown>)[key] = value
  }
  ;(merged as Record<string, unknown>).calibration = withMeasuredCalibration(merged.calibration, merged.confidence)
  return hasSection(merged) ? merged : (fromApi ?? merged)
}
