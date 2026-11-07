/** The same rules the server enforces (backend/auth.py), so people see the problem before they submit. The server stays the judge. */

export const MIN_PASSWORD = 8
export const MAX_PASSWORD = 128

const EMAIL = /^[^@\s]+@[^@\s]+\.[^@\s]+$/
const COMMON = new Set(['password', 'password1', '12345678', '123456789', '1234567890', 'qwertyuiop', 'qwerty123', 'iloveyou', '11111111', 'abcd1234', 'admin123', 'letmein123'])

export function emailProblem(value: string): string | null {
  const email = value.trim()
  return email.length <= 254 && EMAIL.test(email) ? null : 'Enter a valid email address.'
}

export function nameProblem(value: string): string | null {
  const name = value.trim().replace(/\s+/g, ' ')
  return name && name.length <= 60 ? null : 'Enter your name (up to 60 characters).'
}

export function passwordProblem(value: string): string | null {
  if (value.length < MIN_PASSWORD) return `Use at least ${MIN_PASSWORD} characters for the password.`
  if (value.length > MAX_PASSWORD) return `Use at most ${MAX_PASSWORD} characters for the password.`
  if (COMMON.has(value.toLowerCase()) || new Set(value).size === 1) return 'That password is too common. Choose another one.'
  return null
}

/** Rough strength for the meter: 0 (too weak to submit) to 4. Length and variety only, so it never claims more than that. */
export function passwordStrength(value: string): 0 | 1 | 2 | 3 | 4 {
  if (passwordProblem(value)) return 0
  const kinds = [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/].filter((pattern) => pattern.test(value)).length
  const score = (value.length >= 12 ? 2 : value.length >= 10 ? 1 : 0) + (kinds >= 3 ? 2 : kinds === 2 ? 1 : 0)
  return Math.max(1, Math.min(4, score)) as 1 | 2 | 3 | 4
}
