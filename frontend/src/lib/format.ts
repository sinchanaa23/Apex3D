export function percent(fraction: number, digits = 1) {
  return `${(fraction * 100).toFixed(digits)}%`
}

export function bytes(size: number) {
  if (size < 1024) return `${size} B`
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(0)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

export function milliseconds(ms: number) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`
}

/** "detection_ms" or "low_light" to "Detection", "Low light". */
export function humanise(key: string) {
  const text = key.replace(/_ms$/, '').replace(/[_-]+/g, ' ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}
