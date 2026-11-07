import { describe, expect, it } from 'vitest'
import { dominantColour, proportionsFor } from './adapt'

/** An RGBA pixel buffer filled with one colour, with a little of another mixed in. */
function pixels(main: [number, number, number], count = 400, extra?: { colour: [number, number, number]; count: number }) {
  const data: number[] = []
  const push = ([r, g, b]: [number, number, number], n: number) => {
    for (let i = 0; i < n; i++) data.push(r, g, b, 255)
  }
  push(main, count)
  if (extra) push(extra.colour, extra.count)
  return data
}

describe('dominantColour', () => {
  it('finds a red car', () => {
    const colour = dominantColour(pixels([200, 20, 30]))
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(colour.slice(i, i + 2), 16))
    expect(r).toBeGreaterThan(150)
    expect(g).toBeLessThan(60)
    expect(b).toBeLessThan(60)
  })
  it('ignores a little grey road around a blue car', () => {
    const colour = dominantColour(pixels([20, 60, 200], 300, { colour: [120, 120, 120], count: 700 }))
    expect(parseInt(colour.slice(5, 7), 16)).toBeGreaterThan(150)
  })
  it('reports white, grey and black cars as neutrals', () => {
    expect(dominantColour(pixels([240, 240, 242]))).toBe('#d9dbde')
    expect(dominantColour(pixels([130, 132, 135]))).toBe('#8a8d92')
    expect(dominantColour(pixels([20, 20, 22]))).toBe('#17181b')
  })
  it('does not fail on an empty image', () => {
    expect(dominantColour([])).toBe('#8a8d92')
  })
})

describe('proportionsFor', () => {
  it('reads the body style from a Stanford Cars label', () => {
    expect(proportionsFor('Audi R8 Coupe 2012').basis).toBe('Coupe body style')
    expect(proportionsFor('Chevrolet Tahoe Hybrid SUV 2012').height).toBeGreaterThan(1.1)
    expect(proportionsFor('Ford F-450 Super Duty Crew Cab 2012').basis).toBe('Pickup body style')
    expect(proportionsFor('Dodge Caravan Minivan 1997').basis).toBe('Van body style')
  })
  it('keeps the showcase shape when the style is unknown', () => {
    expect(proportionsFor('Some Car 2012')).toMatchObject({ width: 1, height: 1, length: 1 })
    expect(proportionsFor(null).length).toBe(1)
  })
})

import classNames from '../api/fixtures/class_names.json'

describe('showcase car for all 196 classes', () => {
  const labels = (classNames as { name: string }[]).map((row) => row.name)

  it('has all 196 classes to cover', () => {
    expect(labels).toHaveLength(196)
  })
  it('gives every class a real body style, never the unknown fallback', () => {
    const fallback = labels.filter((label) => proportionsFor(label).basis.includes('own shape'))
    expect(fallback).toEqual([])
  })
  it('keeps every class within sensible proportions', () => {
    for (const label of labels) {
      const p = proportionsFor(label)
      for (const value of [p.width, p.height, p.length]) {
        expect(value).toBeGreaterThan(0.8)
        expect(value).toBeLessThan(1.4)
      }
    }
  })
  it('reads the trim-only names correctly', () => {
    expect(proportionsFor('Chevrolet Corvette ZR1 2012').basis).toBe('Coupe body style')
    expect(proportionsFor('Chevrolet TrailBlazer SS 2009').basis).toBe('SUV body style')
    expect(proportionsFor('Dodge Charger SRT-8 2009').basis).toBe('Sedan body style')
  })
})
