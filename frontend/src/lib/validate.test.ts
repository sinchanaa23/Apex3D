import { describe, expect, it } from 'vitest'
import { emailProblem, nameProblem, passwordProblem, passwordStrength } from './validate'

describe('sign-up rules', () => {
  it('accepts normal emails and rejects the rest', () => {
    expect(emailProblem('asha@example.com')).toBeNull()
    expect(emailProblem('  asha@example.com ')).toBeNull()
    for (const bad of ['', 'asha', 'asha@', 'asha@example', 'as ha@example.com']) expect(emailProblem(bad)).not.toBeNull()
    expect(emailProblem(`${'a'.repeat(250)}@example.com`)).not.toBeNull()
  })

  it('needs a name of reasonable length', () => {
    expect(nameProblem('Asha Rao')).toBeNull()
    expect(nameProblem('   ')).not.toBeNull()
    expect(nameProblem('x'.repeat(61))).not.toBeNull()
  })

  it('checks password length and the most common choices', () => {
    expect(passwordProblem('correct horse 42')).toBeNull()
    expect(passwordProblem('short')).toMatch(/at least 8/)
    expect(passwordProblem('x'.repeat(129))).toMatch(/at most 128/)
    expect(passwordProblem('Password')).toMatch(/too common/)
    expect(passwordProblem('aaaaaaaaaa')).toMatch(/too common/)
  })

  it('rates strength by length and variety only', () => {
    expect(passwordStrength('short')).toBe(0)
    expect(passwordStrength('abcdefgh')).toBe(1)
    expect(passwordStrength('Abcdefg1!xyz')).toBeGreaterThanOrEqual(3)
    expect(passwordStrength('correct horse battery staple')).toBeGreaterThan(passwordStrength('abcdefgh'))
  })
})
