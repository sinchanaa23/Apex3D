import {
  adaptClassical,
  adaptJob,
  adaptMetrics,
  adaptRecognition,
  adaptReference,
  adaptReferences,
  type WireClassical,
  type WireJob,
  type WireMetrics,
  type WireRecognition,
  type WireReference,
} from './adapter'
import type { Reference } from './types'

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

export type ReconstructInput = { image: Blob } | { reference: Reference }

async function reconstructForm(input: ReconstructInput): Promise<FormData> {
  if ('image' in input) return fileForm(input.image, 'input.jpg')

  const { image_url: url, id } = input.reference
  if (url.startsWith('/')) {
    // A user's upload is served by our own API, so the server can't fetch it by that relative path.
    const response = await fetch(url).catch(() => null)
    if (!response?.ok) throw new ApiError('network', "That reference couldn't be loaded. It may have been cleaned up, so upload it again.")
    return fileForm(await response.blob(), 'input.jpg')
  }
  const form = new FormData()
  form.append('image_url', url)
  form.append('reference_id', id)
  return form
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

export async function getMetrics(signal?: AbortSignal) {
  const wire = await request<WireMetrics>('/api/metrics', {}, { signal })
  return adapted(wire, adaptMetrics)
}
