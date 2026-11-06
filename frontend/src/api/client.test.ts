import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, errorMessage, getJob } from './client'

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
