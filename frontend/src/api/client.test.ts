import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, errorMessage, getJob, reconstruct } from './client'

describe('errorMessage', () => {
  it('uses the FastAPI detail string', () => {
    expect(errorMessage(400, JSON.stringify({ detail: 'No car found.' }))).toBe('No car found.')
  })
  it('joins FastAPI validation errors', () => {
    expect(errorMessage(422, JSON.stringify({ detail: [{ msg: 'Field required' }, { msg: 'Bad type' }] }))).toBe('Field required. Bad type')
  })
  it('explains tunnel failures that return HTML', () => {
    expect(errorMessage(502, '<html>Bad gateway</html>')).toMatch(/host laptop/)
    expect(errorMessage(530, '<html>Cloudflare error</html>')).toMatch(/host laptop/)
  })
  it('falls back to the status code', () => {
    expect(errorMessage(500, 'oops')).toMatch(/500/)
  })
})

describe('request errors', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('reports a network failure as offline', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('failed')))
    await expect(getJob('x')).rejects.toMatchObject({ kind: 'network' })
  })
  it('reports unreadable success bodies rather than returning them', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>')))
    await expect(getJob('x')).rejects.toBeInstanceOf(ApiError)
  })
  it('escapes job ids in the URL', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}'))
    vi.stubGlobal('fetch', fetchMock)
    await getJob('a/b')
    expect(fetchMock.mock.calls[0][0]).toBe('/api/jobs/a%2Fb')
  })
})

describe('reconstruct from a reference', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('never asks the server to fetch a web address', async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('blocked'))
    vi.stubGlobal('fetch', fetchMock)
    const reference = { id: 'wm_1', image_url: 'https://upload.wikimedia.org/a.jpg', source: 'Wikimedia Commons', match: 'confirmed' as const }
    await expect(reconstruct({ reference })).rejects.toThrow(/couldn't be loaded/)
    // The only request made is the browser's own download of the image. Nothing was posted to /api/reconstruct.
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe('https://upload.wikimedia.org/a.jpg')
  })
})

// ---- getMetrics: the API first, then the saved result files ----
import classifierFile from './fixtures/person_a_classifier_metrics.json'
import optimisationFile from './fixtures/person_a_optimization_metrics.json'
import { getMetrics } from './client'

const json = (value: unknown) => new Response(JSON.stringify(value))
const SPA_PAGE = '<!doctype html><html><body>the app</body></html>'

/** Answers each URL from a table, and 404s anything else. */
function serve(routes: Record<string, () => Response | Promise<Response>>) {
  const fetchMock = vi.fn((url: string) => (routes[url] ? Promise.resolve(routes[url]()) : Promise.resolve(new Response('nope', { status: 404 }))))
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}
const files = {
  '/results/classifier_metrics.json': () => json(classifierFile),
  '/results/optimization_metrics.json': () => json(optimisationFile),
}
const notAvailable = { '/api/metrics': () => json({ status: 'not available', message: 'No metrics yet', metrics: null }) }
const filled = (section: unknown) => section != null && section !== 'not available'

describe('getMetrics', () => {
  afterEach(() => vi.unstubAllGlobals())

  it("fills a backend that has no data from Person A's saved files", async () => {
    serve({ ...notAvailable, ...files })
    const metrics = await getMetrics()
    expect(filled(metrics.classifier)).toBe(true)
    expect(filled(metrics.optimisation)).toBe(true)
    expect(filled(metrics.states)).toBe(true)
    expect(filled(metrics.robustness)).toBe(false) // nobody has produced this, so it stays not available
    expect(metrics.message).toBe('No metrics yet')
  })
  it('keeps what the backend does return, and fills only the gaps', async () => {
    serve({
      '/api/metrics': () => json({ status: 'available', source_file: 'metrics.json', metrics: { test_top1_acc: 0.5, test_top5_acc: 0.7, robustness: { blur_top1: 0.3 } } }),
      ...files,
    })
    const metrics = await getMetrics()
    expect((metrics.classifier as { data: { top1: number } }).data.top1).toBe(0.5) // the backend's number wins
    expect(filled(metrics.robustness)).toBe(true)
    expect(filled(metrics.optimisation)).toBe(true) // filled from the saved file
  })
  it('still shows the saved files when the backend is unreachable', async () => {
    serve({ '/api/metrics': () => Promise.reject(new TypeError('offline')) as unknown as Response, ...files })
    expect(filled((await getMetrics()).classifier)).toBe(true)
  })
  it('reports the backend error when there is nothing else to show', async () => {
    serve({ '/api/metrics': () => new Response('{"detail":"Metrics files could not be read."}', { status: 500 }) })
    await expect(getMetrics()).rejects.toThrow(/Metrics files could not be read/)
  })
  it("treats the app's own page, returned for a missing file, as no file at all", async () => {
    serve({ ...notAvailable, '/results/classifier_metrics.json': () => new Response(SPA_PAGE), '/results/optimization_metrics.json': () => new Response(SPA_PAGE) })
    const metrics = await getMetrics()
    expect(filled(metrics.classifier)).toBe(false)
    expect(metrics.message).toBe('No metrics yet')
  })
})
