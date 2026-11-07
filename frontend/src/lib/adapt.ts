// What the showcase car takes from the user's photo. Both pieces are measured or read from the app's own results, nothing is guessed.

/** The paint colour of the car in a photo, from its pixels. Saturated colours win. Otherwise the car is white, silver, grey or black. */
export function dominantColour(pixels: ArrayLike<number>): string {
  const BUCKETS = 24
  const weight = new Array<number>(BUCKETS).fill(0)
  const sum = Array.from({ length: BUCKETS }, () => [0, 0, 0])
  let saturated = 0
  let total = 0
  let luma = 0

  for (let i = 0; i + 3 < pixels.length; i += 4) {
    const r = pixels[i] / 255
    const g = pixels[i + 1] / 255
    const b = pixels[i + 2] / 255
    if (pixels[i + 3] < 128) continue
    total += 1
    luma += 0.2126 * r + 0.7152 * g + 0.0722 * b
    const max = Math.max(r, g, b)
    const min = Math.min(r, g, b)
    const saturation = max === 0 ? 0 : (max - min) / max
    // Very dark pixels are shadows and tyres, very pale ones are sky and glare, and grey ones carry no hue.
    if (saturation < 0.28 || max < 0.18 || min > 0.9) continue
    saturated += 1
    const delta = max - min
    let hue = max === r ? ((g - b) / delta) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4
    if (hue < 0) hue += 6
    const bucket = Math.min(BUCKETS - 1, Math.floor((hue / 6) * BUCKETS))
    const w = saturation * max
    weight[bucket] += w
    sum[bucket][0] += r * w
    sum[bucket][1] += g * w
    sum[bucket][2] += b * w
  }

  if (total === 0) return '#8a8d92'
  if (saturated / total < 0.08) {
    const mean = luma / total
    return mean > 0.68 ? '#d9dbde' : mean < 0.24 ? '#17181b' : '#8a8d92'
  }
  let best = 0
  for (let i = 1; i < BUCKETS; i++) if (weight[i] > weight[best]) best = i
  const [r, g, b] = sum[best].map((value) => Math.round((value / weight[best]) * 255))
  return `#${[r, g, b].map((value) => value.toString(16).padStart(2, '0')).join('')}`
}

/** Reads the car's colour from the middle of its box in the photo, where the bodywork is. Returns null if the image can't be read. */
export async function colourFromImage(url: string, box?: [number, number, number, number] | null, imageSize?: { width: number; height: number } | null): Promise<string | null> {
  try {
    const image = new Image()
    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve()
      image.onerror = () => reject(new Error('load'))
      image.src = url
    })
    const w = image.naturalWidth
    const h = image.naturalHeight
    let [x1, y1, x2, y2] = [0, 0, w, h]
    if (box && imageSize?.width && imageSize.height) {
      const sx = w / imageSize.width
      const sy = h / imageSize.height
      ;[x1, y1, x2, y2] = [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
    }
    // The middle 60% of the box, nudged up, avoids wheels, road and sky.
    const cw = (x2 - x1) * 0.6
    const ch = (y2 - y1) * 0.5
    const cx = x1 + (x2 - x1) * 0.2
    const cy = y1 + (y2 - y1) * 0.15
    const canvas = document.createElement('canvas')
    canvas.width = 64
    canvas.height = 64
    const context = canvas.getContext('2d', { willReadFrequently: true })
    if (!context) return null
    context.drawImage(image, cx, cy, Math.max(cw, 1), Math.max(ch, 1), 0, 0, 64, 64)
    return dominantColour(context.getImageData(0, 0, 64, 64).data)
  } catch {
    return null
  }
}

export interface Proportions {
  /** Scale along the car's width, height and length, relative to the showcase car. */
  width: number
  height: number
  length: number
  /** What the proportions came from, shown to the user. */
  basis: string
}

const STYLES: { match: RegExp; name: string; width: number; height: number; length: number }[] = [
  { match: /\b(minivan|van)\b/i, name: 'Van', width: 1.04, height: 1.3, length: 1.04 },
  { match: /\b(extended cab|regular cab|supercab|crew cab|quad cab|club cab)\b/i, name: 'Pickup', width: 1.06, height: 1.22, length: 1.1 },
  { match: /\bsuv\b/i, name: 'SUV', width: 1.04, height: 1.2, length: 0.95 },
  { match: /\bhatchback\b/i, name: 'Hatchback', width: 0.97, height: 1.04, length: 0.86 },
  { match: /\bwagon\b/i, name: 'Wagon', width: 1, height: 1.06, length: 1 },
  { match: /\b(convertible|coupe)\b/i, name: 'Coupe', width: 1.02, height: 0.9, length: 0.96 },
  { match: /\bsedan\b/i, name: 'Sedan', width: 1, height: 1, length: 1 },
]

/**
 * A few Stanford Cars labels name a trim instead of a body style, such as "Chevrolet Corvette ZR1 2012".
 * These are the 14 such labels, with the body style the real car has.
 */
const BY_MODEL: [RegExp, string][] = [
  [/acura tl type-s/i, 'Sedan'],
  [/acura integra type r/i, 'Coupe'],
  [/buick regal gs/i, 'Sedan'],
  [/chevrolet corvette/i, 'Coupe'],
  [/chevrolet hhr ss/i, 'Wagon'],
  [/chevrolet cobalt ss/i, 'Coupe'],
  [/chevrolet trailblazer ss/i, 'SUV'],
  [/chrysler 300 srt-8/i, 'Sedan'],
  [/dodge challenger srt8/i, 'Coupe'],
  [/dodge charger srt-8/i, 'Sedan'],
  [/fiat 500 abarth/i, 'Hatchback'],
  [/jaguar xk xkr/i, 'Coupe'],
  [/lamborghini gallardo/i, 'Coupe'],
]

/** Proportions for the body style in a Stanford Cars label such as "Audi R8 Coupe 2012". Unknown styles keep the showcase car's own shape. */
export function proportionsFor(label?: string | null): Proportions {
  const named = label ? BY_MODEL.find(([pattern]) => pattern.test(label)) : undefined
  const style = label ? (named ? STYLES.find((entry) => entry.name === named[1]) : STYLES.find((entry) => entry.match.test(label))) : undefined
  if (!style) return { width: 1, height: 1, length: 1, basis: 'the showcase car’s own shape' }
  return { width: style.width, height: style.height, length: style.length, basis: `${style.name} body style` }
}
