import { describe, expect, it } from 'vitest'
import { describeVehicle } from './vehicle'
import classNames from '../api/fixtures/class_names.json'

describe('describeVehicle', () => {
  it('splits a label into manufacturer, model, body style and year', () => {
    expect(describeVehicle('Audi R8 Coupe 2012')).toEqual({ make: 'Audi', model: 'R8', year: 2012, bodyStyle: 'Coupe' })
    expect(describeVehicle('Aston Martin V8 Vantage Convertible 2012')).toEqual({ make: 'Aston Martin', model: 'V8 Vantage', year: 2012, bodyStyle: 'Convertible' })
    expect(describeVehicle('AM General Hummer SUV 2000')).toMatchObject({ make: 'AM General', model: 'Hummer', year: 2000, bodyStyle: 'SUV' })
  })
  it('uses the backend fields when the response has them', () => {
    expect(describeVehicle('x 1999', { make: 'Make', model: 'Model', year: 2005, body_style: 'Sedan' })).toEqual({ make: 'Make', model: 'Model', year: 2005, bodyStyle: 'Sedan' })
  })
  it('keeps a model when the label has no body style', () => {
    expect(describeVehicle('Chevrolet Corvette ZR1 2012')).toMatchObject({ make: 'Chevrolet', model: 'Corvette ZR1', year: 2012, bodyStyle: null })
  })
})

describe('every one of the 196 classes shows a manufacturer, a model and a year', () => {
  const labels = (classNames as { name: string }[]).map((row) => row.name)
  it('has a make, a model and a year for all of them', () => {
    const incomplete = labels.filter((label) => {
      const facts = describeVehicle(label)
      return !facts.make || !facts.model || facts.year == null || facts.year < 1990 || facts.year > 2012
    })
    expect(incomplete).toEqual([])
  })
})
