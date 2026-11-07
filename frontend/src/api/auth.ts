/** Account calls. The session lives in an HttpOnly cookie the server sets, so nothing secret is kept in the page. */

/** Fired when a call comes back 401, for example after the session expired. */
export const UNAUTHORIZED_EVENT = 'apex3d:unauthorized'

export interface User {
  id: number
  name: string
  email: string
}

export interface Me {
  user: User | null
  /** False when the server runs with the login requirement switched off (local development). */
  loginRequired: boolean
}

export class AuthError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
  }
}

const OFFLINE = "Can't reach the server. Check your connection and try again."

async function call(path: string, init?: RequestInit): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(path, { credentials: 'same-origin', ...init })
  } catch {
    throw new AuthError(OFFLINE, 0)
  }
  const text = await response.text()
  let body: unknown = null
  try {
    body = JSON.parse(text)
  } catch {
    // The tunnel answers with an HTML page when the laptop stops responding.
  }
  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail
    const message =
      typeof detail === 'string' ? detail : response.status >= 500 || response.status === 0 ? OFFLINE : `Something went wrong (${response.status}).`
    throw new AuthError(message, response.status)
  }
  return body
}

const post = (path: string, body?: unknown) =>
  call(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })

export async function fetchMe(): Promise<Me> {
  const body = (await call('/api/auth/me')) as { user?: User | null; login_required?: boolean } | null
  return { user: body?.user ?? null, loginRequired: body?.login_required !== false }
}

export async function signUp(input: { name: string; email: string; password: string }): Promise<User> {
  return ((await post('/api/auth/signup', input)) as { user: User }).user
}

export async function logIn(input: { email: string; password: string }): Promise<User> {
  return ((await post('/api/auth/login', input)) as { user: User }).user
}

export async function logOut(): Promise<void> {
  await post('/api/auth/logout').catch(() => undefined)
}
