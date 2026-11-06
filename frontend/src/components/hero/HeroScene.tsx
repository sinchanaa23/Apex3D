import { Component, Suspense, lazy, useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { useReducedMotion } from 'framer-motion'
import { Spinner } from '../States'

// The 3D scene is a separate download, so the headline and buttons appear before three.js has loaded.
const HeroCar = lazy(() => import('./HeroCar'))

function hasWebGL() {
  try {
    return document.createElement('canvas').getContext('webgl2') != null
  } catch {
    return false
  }
}

/** If the scene can't run, the hero quietly falls back to text only. */
class SceneBoundary extends Component<{ onFail(): void; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  componentDidCatch() {
    this.props.onFail()
  }
  render() {
    return this.state.failed ? null : this.props.children
  }
}

export function HeroScene({ className = '' }: { className?: string }) {
  const reduced = useReducedMotion() ?? false
  const box = useRef<HTMLDivElement>(null)
  const pointer = useRef({ x: 0, y: 0 })
  const [supported, setSupported] = useState(hasWebGL)
  const [visible, setVisible] = useState(true)
  const [ready, setReady] = useState(false)
  const onReady = useCallback(() => setReady(true), [])

  // Stops rendering while the hero is scrolled out of view.
  useEffect(() => {
    if (!box.current) return
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting), { rootMargin: '80px' })
    observer.observe(box.current)
    return () => observer.disconnect()
  }, [])

  // Mouse only. Touch and pen scrolling is never intercepted.
  useEffect(() => {
    if (reduced) return
    const move = (event: PointerEvent) => {
      if (event.pointerType !== 'mouse') return
      pointer.current.x = (event.clientX / window.innerWidth) * 2 - 1
      pointer.current.y = (event.clientY / window.innerHeight) * 2 - 1
    }
    window.addEventListener('pointermove', move, { passive: true })
    return () => window.removeEventListener('pointermove', move)
  }, [reduced])

  if (!supported) return null

  return (
    <div ref={box} role="img" aria-label="A rotating 3D sports car with a glowing orange wireframe" className={`relative min-w-0 ${className}`}>
      <div aria-hidden className="pointer-events-none absolute inset-[8%] rounded-full bg-[radial-gradient(closest-side,rgb(255_106_43/0.16),transparent)]" />
      {!ready && (
        <div className="absolute inset-0 grid place-items-center text-sm">
          <Spinner label="Loading 3D model" />
        </div>
      )}
      {/* The mask feathers the canvas edges, so no particle, line or glow ends in a hard cut. */}
      <div
        className={`absolute inset-0 [mask-image:radial-gradient(ellipse_closest-side,#000_78%,transparent)] transition-opacity duration-1000 motion-reduce:transition-none ${ready ? 'opacity-100' : 'opacity-0'}`}
      >
        <SceneBoundary onFail={() => setSupported(false)}>
          <Suspense fallback={null}>
            <HeroCar reducedMotion={reduced} visible={visible} pointer={pointer} onReady={onReady} />
          </Suspense>
        </SceneBoundary>
      </div>
    </div>
  )
}
