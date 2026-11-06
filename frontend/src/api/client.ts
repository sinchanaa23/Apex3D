import type {
  ClassicalResponse,
  Job,
  MetricsResponse,
  RecognizeResponse,
  ReferencesResponse,
  ValidationReport,
} from './types'

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
    if (!response.ok) throw new ApiError('http', errorMessage(response.status, body), response.status)
    try {
      return JSON.parse(body) as T
    } catch {
      throw new ApiError('parse', 'The server sent a response this page could not read.', response.status)
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

function imageForm(file: Blob): FormData {
  const form = new FormData()
  form.append('image', file, 'photo.jpg')
  return form
}

export function recognize(image: Blob, signal?: AbortSignal) {
  return request<RecognizeResponse>('/api/recognize', { method: 'POST', body: imageForm(image) }, { signal, timeoutMs: 90_000 })
}

export function classical(image: Blob, signal?: AbortSignal) {
  return request<ClassicalResponse>('/api/classical', { method: 'POST', body: imageForm(image) }, { signal, timeoutMs: 60_000 })
}

export function getReferences(classId: number, signal?: AbortSignal) {
  return request<ReferencesResponse>(`/api/references?class_id=${classId}`, {}, { signal, timeoutMs: 90_000 })
}

export function uploadReferences(files: File[], signal?: AbortSignal) {
  const form = new FormData()
  for (const file of files) form.append('images', file, file.name)
  return request<ReferencesResponse>('/api/references/upload', { method: 'POST', body: form }, { signal, timeoutMs: 90_000 })
}

export type ReconstructInput = { image: Blob } | { referenceId: string }

export function reconstruct(input: ReconstructInput, classId?: number | null, signal?: AbortSignal) {
  const form = new FormData()
  if ('image' in input) form.append('image', input.image, 'input.jpg')
  else form.append('reference_id', input.referenceId)
  if (classId != null) form.append('class_id', String(classId))
  return request<{ job_id: string }>('/api/reconstruct', { method: 'POST', body: form }, { signal, timeoutMs: 60_000 })
}

export function getJob(id: string, signal?: AbortSignal) {
  return request<Job>(`/api/jobs/${encodeURIComponent(id)}`, {}, { signal, timeoutMs: 15_000 })
}

export function assetUrl(assetId: string, file: string) {
  return `/api/assets/${encodeURIComponent(assetId)}/${encodeURIComponent(file)}`
}

export function getValidation(assetId: string, file: string, signal?: AbortSignal) {
  return request<ValidationReport>(assetUrl(assetId, file), {}, { signal })
}

export function getMetrics(signal?: AbortSignal) {
  return request<MetricsResponse>('/api/metrics', {}, { signal })
}
