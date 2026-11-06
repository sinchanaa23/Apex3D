import { Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useLoader, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { Grid3x3, RotateCcw } from 'lucide-react'
import { Box3, DoubleSide, Mesh, MeshStandardMaterial, Vector3, type Material } from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { ErrorBoundary } from './ErrorBoundary'
import { Spinner } from './States'

const TARGET_SIZE = 2
const START_POSITION: [number, number, number] = [2.4, 1.3, 2.8]

function Model({ url, wireframe, onReady }: { url: string; wireframe: boolean; onReady(): void }) {
  const { scene } = useLoader(GLTFLoader, url)
  const invalidate = useThree((state) => state.invalidate)

  // Fits any mesh into the same view. Display only: TripoSR's output has no real-world scale.
  const { scale, offset, floor } = useMemo(() => {
    const box = new Box3().setFromObject(scene)
    const size = box.getSize(new Vector3())
    const center = box.getCenter(new Vector3())
    const fit = TARGET_SIZE / Math.max(size.x, size.y, size.z, Number.EPSILON)
    return { scale: fit, offset: center.multiplyScalar(-1), floor: (-size.y * fit) / 2 }
  }, [scene])

  const materials = useMemo(() => {
    const found = new Set<Material>()
    scene.traverse((node) => {
      if (node instanceof Mesh) {
        const list = Array.isArray(node.material) ? node.material : [node.material]
        list.forEach((material: Material) => found.add(material))
      }
    })
    return [...found]
  }, [scene])

  useLayoutEffect(() => {
    for (const material of materials) {
      // Generated meshes can have flipped faces, so show both sides rather than holes.
      material.side = DoubleSide
      if (material instanceof MeshStandardMaterial) material.wireframe = wireframe
      material.needsUpdate = true
    }
    invalidate()
  }, [materials, wireframe, invalidate])

  // The canvas only draws on demand, to save phone batteries, so ask for a frame once
  // the mesh has been attached. Without this it stays blank until the first drag.
  useEffect(() => {
    onReady()
    const frame = requestAnimationFrame(() => invalidate())
    return () => cancelAnimationFrame(frame)
  }, [scene, invalidate, onReady])

  return (
    <>
      <group scale={scale}>
        <primitive object={scene} position={offset} />
      </group>
      <gridHelper args={[6, 12, '#b04a20', '#33271f']} position={[0, floor, 0]} />
    </>
  )
}

export default function MeshViewer({ url }: { url: string }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const [wireframe, setWireframe] = useState(false)
  const [ready, setReady] = useState(false)
  const onReady = useCallback(() => setReady(true), [])

  // Set explicitly: OrbitControls.reset() returns to the state saved before the camera was positioned.
  function resetView() {
    const orbit = controls.current
    if (!orbit) return
    orbit.object.position.set(...START_POSITION)
    orbit.target.set(0, 0, 0)
    orbit.update()
  }

  return (
    <ErrorBoundary title="Viewer unavailable" message="This browser may not support WebGL, or the file may not have downloaded fully. The download links still work.">
      <div className="relative aspect-[4/3] overflow-hidden rounded-xl border border-line bg-[radial-gradient(ellipse_at_50%_85%,rgb(255_106_43/0.1),#141413_62%)] lg:aspect-[16/10]">
        <Canvas frameloop="demand" dpr={[1, 2]} camera={{ position: START_POSITION, fov: 40 }} aria-label="Interactive 3D view of the generated mesh">
          <hemisphereLight args={['#ffffff', '#3a3a36', 1.6]} />
          <directionalLight position={[4, 6, 3]} intensity={2} />
          <directionalLight position={[-4, 2, -3]} intensity={0.7} />
          <Suspense fallback={null}>
            <Model url={url} wireframe={wireframe} onReady={onReady} />
          </Suspense>
          {/* No damping: leftover momentum would carry the camera away from a reset. */}
          <OrbitControls ref={controls} makeDefault enableDamping={false} minDistance={0.8} maxDistance={12} />
        </Canvas>
        {!ready && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center">
            <div className="w-40 text-center">
              <Spinner label="Loading mesh" />
              <div className="scanbar mt-3" />
            </div>
          </div>
        )}
        <span aria-hidden className="hud-corner tl" />
        <span aria-hidden className="hud-corner tr" />
        <span aria-hidden className="hud-corner bl" />
        <span aria-hidden className="hud-corner br" />
        {ready && <span aria-hidden className="scan-line once" />}
        <p className="pointer-events-none absolute top-4 left-5 hidden items-center gap-2 font-mono sm:flex text-[11px] tracking-[0.2em] text-accent uppercase">
          <span aria-hidden className="size-1.5 rounded-full bg-accent shadow-[0_0_8px_var(--color-accent)]" />
          Generated mesh
        </p>
        <div className="absolute top-3 right-3 flex gap-2">
          <button
            type="button"
            aria-pressed={wireframe}
            onClick={() => setWireframe(!wireframe)}
            className={`inline-flex min-h-10 items-center gap-1.5 rounded-lg border px-3 text-sm backdrop-blur ${wireframe ? 'border-accent bg-accent/15' : 'border-line-strong bg-page/70 hover:bg-raised'}`}
          >
            <Grid3x3 aria-hidden className="size-4" />
            Wireframe
          </button>
          <button
            type="button"
            onClick={resetView}
            className="inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-line-strong bg-page/70 px-3 text-sm backdrop-blur hover:bg-raised"
          >
            <RotateCcw aria-hidden className="size-4" />
            Reset view
          </button>
        </div>
        <p className="pointer-events-none absolute bottom-3 left-8 text-xs text-ink-3">
          <span className="touch:hidden">Drag to rotate · scroll to zoom · right-drag to pan</span>
          <span className="hidden touch:inline">Drag to rotate · pinch to zoom · two fingers to pan</span>
        </p>
      </div>
    </ErrorBoundary>
  )
}
