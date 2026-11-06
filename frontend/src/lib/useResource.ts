import { useCallback, useEffect, useState } from 'react'
import { ApiError } from '../api/client'

export type Resource<T> =
  | { status: 'loading'; data?: undefined; error?: undefined }
  | { status: 'ready'; data: T; error?: undefined }
  | { status: 'error'; data?: undefined; error: string }

/** Runs a request on mount and whenever `key` changes, and cancels it if the page goes away. */
export function useResource<T>(load: (signal: AbortSignal) => Promise<T>, key: unknown): Resource<T> & { reload(): void } {
  const [state, setState] = useState<Resource<T>>({ status: 'loading' })
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setState({ status: 'loading' })
    load(controller.signal).then(
      (data) => {
        if (!controller.signal.aborted) setState({ status: 'ready', data })
      },
      (error: unknown) => {
        if (controller.signal.aborted) return
        setState({ status: 'error', error: error instanceof ApiError ? error.message : 'Something went wrong.' })
      },
    )
    return () => controller.abort()
    // `load` closes over `key`, so `key` is the real dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, attempt])

  const reload = useCallback(() => setAttempt((a) => a + 1), [])
  return { ...state, reload }
}
