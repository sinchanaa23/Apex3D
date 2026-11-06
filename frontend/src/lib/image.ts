export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024
/** Long-side limit for uploads. Far above what the classifier uses, but keeps uploads quick over the tunnel. */
export const MAX_LONG_SIDE = 2048

/** Returns a message if the file can't be used, or null if it is worth trying to decode. */
export function checkFile(file: { type: string; size: number }): string | null {
  if (file.type && !file.type.startsWith('image/')) {
    return 'That file is not an image. Choose a JPEG, PNG or WebP photo.'
  }
  if (file.size === 0) return 'That file is empty. Choose another photo.'
  if (file.size > MAX_UPLOAD_BYTES) {
    return `That file is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is ${MAX_UPLOAD_BYTES / 1024 / 1024} MB.`
  }
  return null
}

export function fitWithin(width: number, height: number, maxSide = MAX_LONG_SIDE) {
  const scale = Math.min(1, maxSide / Math.max(width, height))
  return { width: Math.round(width * scale), height: Math.round(height * scale) }
}

export interface NormalisedImage {
  file: File
  url: string
  width: number
  height: number
}

/**
 * Redraws the photo onto a canvas before upload. Phone cameras store rotation as
 * EXIF metadata, which browsers apply and many server decoders don't, so without
 * this the server's bounding box would land on a rotated image.
 */
export async function normaliseImage(source: File): Promise<NormalisedImage> {
  const sourceUrl = URL.createObjectURL(source)
  try {
    const img = new Image()
    try {
      // The load event is used because img.decode() can stall in embedded browsers and background tabs.
      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = () => reject(new Error('decode failed'))
        img.src = sourceUrl
      })
    } catch {
      throw new Error("This photo couldn't be read. It may be damaged, or a HEIC file this browser can't open. Try a JPEG or PNG.")
    }
    const { width, height } = fitWithin(img.naturalWidth, img.naturalHeight)
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const context = canvas.getContext('2d')
    if (!context) throw new Error("This browser couldn't prepare the photo for upload.")
    context.fillStyle = '#fff' // Transparent PNGs would otherwise flatten onto black.
    context.fillRect(0, 0, width, height)
    context.drawImage(img, 0, 0, width, height)
    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92))
    if (!blob) throw new Error("This browser couldn't prepare the photo for upload.")
    const file = new File([blob], 'photo.jpg', { type: 'image/jpeg' })
    return { file, url: URL.createObjectURL(file), width, height }
  } finally {
    URL.revokeObjectURL(sourceUrl)
  }
}
