import { describe, expect, it } from 'vitest'
import { MAX_LONG_SIDE, MAX_UPLOAD_BYTES, checkFile, fitWithin } from './image'

describe('checkFile', () => {
  it('accepts images and files with no reported type', () => {
    expect(checkFile({ type: 'image/jpeg', size: 1000 })).toBeNull()
    expect(checkFile({ type: '', size: 1000 })).toBeNull()
  })
  it('rejects non-images, empty files and oversized files', () => {
    expect(checkFile({ type: 'application/pdf', size: 1000 })).toMatch(/not an image/)
    expect(checkFile({ type: 'image/png', size: 0 })).toMatch(/empty/)
    expect(checkFile({ type: 'image/png', size: MAX_UPLOAD_BYTES + 1 })).toMatch(/limit/)
  })
})

describe('fitWithin', () => {
  it('leaves small images alone and never upscales', () => {
    expect(fitWithin(640, 480)).toEqual({ width: 640, height: 480 })
  })
  it('scales the long side down and keeps the aspect ratio', () => {
    expect(fitWithin(4000, 3000)).toEqual({ width: MAX_LONG_SIDE, height: 1536 })
    expect(fitWithin(3000, 4000)).toEqual({ width: 1536, height: MAX_LONG_SIDE })
  })
})
