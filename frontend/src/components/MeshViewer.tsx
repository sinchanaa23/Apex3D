import { Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber'
import { ContactShadows, Environment, Lightformer, OrbitControls } from '@react-three/drei'
import { useReducedMotion } from 'framer-motion'
import { Grid3x3, RotateCcw, Sparkles } from 'lucide-react'
import { Box3, DoubleSide, EdgesGeometry, LineSegments, Mesh, MeshPhysicalMaterial, MeshStandardMaterial, Vector3, type Material } from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { ErrorBoundary } from './ErrorBoundary'
import { Spinner } from './States'
import { createEdgeMaterial, createFloorMaterial, createHoloUniforms, patchSurface, type HoloUniforms } from './hero/shaders'

/** The same size the hero car is drawn at, so the contour grid and floor look the same here. */
const TARGET_SIZE = 4.4
const START_POSITION: [number, number, number] = [5.2, 2.6, 6.6]
const SCAN_PERIOD = 5.5
const PULSE_PERIOD = 8.5
const PULSE_LENGTH = 2.8
/** Edge lines on a very dense mesh cost more than they add. */
const MAX_EDGE_FACES = 300_000

type Look = 'holo' | 'natural'

function Model({ url, look, wireframe, uniforms, reducedMotion, onReady }: { url: string; look: Look; wireframe: boolean; uniforms: HoloUniforms; reducedMotion: boolean; onReady(): void }) {
  const { scene } = useLoader(GLTFLoader, url)
  const invalidate = useThree((state) => state.invalidate)

  // Fits any mesh into the same view. Display only: TripoSR's output has no real-world scale.
  const { scale, offset, floor, height } = useMemo(() => {
    const box = new Box3().setFromObject(scene)
    const size = box.getSize(new Vector3())
    const center = box.getCenter(new Vector3())
    const fit = TARGET_SIZE / Math.max(size.x, size.y, size.z, Number.EPSILON)
    return { scale: fit, offset: center.multiplyScalar(-1), floor: (-size.y * fit) / 2, height: size.y * fit }
  }, [scene])

  const meshes = useMemo(() => {
    const found: Mesh[] = []
    scene.traverse((node) => {
      if (node instanceof Mesh) {
        // Generated meshes often have no normals. Without them the surface shader cannot compile and three.js falls back to flat shading.
        if (!node.geometry.attributes.normal) node.geometry.computeVertexNormals()
        found.push(node)
      }
    })
    return found
  }, [scene])

  // The file's own materials, kept so "Natural" can bring them back.
  const originals = useMemo(() => new Map(meshes.map((mesh) => [mesh, mesh.material as Material | Material[]])), [meshes])

  const holoMaterial = useMemo(() => {
    const material = new MeshPhysicalMaterial({ color: '#1a1512', metalness: 0.85, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.12, envMapIntensity: 1.2, side: DoubleSide })
    patchSurface(material, uniforms, { grid: 1, rim: 1 })
    return material
  }, [uniforms])

  const edges = useMemo(() => {
    const faces = meshes.reduce((sum, mesh) => sum + (mesh.geometry.index ? mesh.geometry.index.count / 3 : mesh.geometry.attributes.position.count / 3), 0)
    if (faces > MAX_EDGE_FACES) return { material: null, items: [] }
    const material = createEdgeMaterial(uniforms)
    return {
      material,
      items: meshes.map((mesh) => {
        const lines = new LineSegments(new EdgesGeometry(mesh.geometry, 55), material)
        lines.renderOrder = 2
        return { mesh, lines }
      }),
    }
  }, [meshes, uniforms])

  useEffect(
    () => () => {
      holoMaterial.dispose()
      edges.material?.dispose()
      for (const { lines } of edges.items) lines.geometry.dispose()
    },
    [holoMaterial, edges],
  )

  useLayoutEffect(() => {
    for (const mesh of meshes) {
      mesh.material = look === 'holo' ? holoMaterial : (originals.get(mesh) as Material | Material[])
      const list = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
      for (const material of list) {
        // Generated meshes can have flipped faces, so show both sides rather than holes.
        material.side = DoubleSide
        if (material instanceof MeshStandardMaterial) material.wireframe = wireframe
        material.needsUpdate = true
      }
    }
    for (const { mesh, lines } of edges.items) {
      if (look === 'holo' && !wireframe) mesh.add(lines)
      else mesh.remove(lines)
    }
    invalidate()
  }, [meshes, originals, holoMaterial, edges, look, wireframe, invalidate])

  // Keeps the grid fixed to the mesh, and runs the scan and pulse like the landing page.
  useFrame((state) => {
    const t = reducedMotion ? 0 : state.clock.elapsedTime
    uniforms.uTime.value = t
    if (reducedMotion) {
      uniforms.uScan.value = floor + height * 0.7
      uniforms.uPulse.value = -1
      return
    }
    uniforms.uScan.value = floor - 0.15 + (t % SCAN_PERIOD) * (height / 3.2)
    const pulse = (t % PULSE_PERIOD) / PULSE_LENGTH
    uniforms.uPulse.value = pulse <= 1 ? pulse : -1
  })

  // The canvas only draws on demand with reduced motion, so ask for a frame once the mesh is attached.
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
      <group position={[0, floor, 0]}>
        <Floor uniforms={uniforms} />
        <ContactShadows position={[0, 0.01, 0]} opacity={0.8} scale={9} blur={2.6} far={1.7} resolution={256} frames={reducedMotion ? 3 : Infinity} color="#000000" />
      </group>
    </>
  )
}

function Floor({ uniforms }: { uniforms: HoloUniforms }) {
  const material = useMemo(() => createFloorMaterial(uniforms), [uniforms])
  useEffect(() => () => material.dispose(), [material])
  return (
    <mesh rotation-x={-Math.PI / 2} material={material} renderOrder={-1}>
      <planeGeometry args={[14, 14]} />
    </mesh>
  )
}

export default function MeshViewer({ url }: { url: string }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const reducedMotion = useReducedMotion() ?? false
  const uniforms = useMemo(createHoloUniforms, [])
  const [look, setLook] = useState<Look>('holo')
  const [wireframe, setWireframe] = useState(false)
  const [ready, setReady] = useState(false)
  const resume = useRef<number>(0)
  const onReady = useCallback(() => setReady(true), [])

  useEffect(() => () => window.clearTimeout(resume.current), [])

  // Set explicitly: OrbitControls.reset() returns to the state saved before the camera was positioned.
  function resetView() {
    const orbit = controls.current
    if (!orbit) return
    orbit.object.position.set(...START_POSITION)
    orbit.target.set(0, 0, 0)
    orbit.update()
  }

  // The slow turntable stops while the person is dragging and starts again a few seconds later.
  function pauseTurntable() {
    window.clearTimeout(resume.current)
    if (controls.current) controls.current.autoRotate = false
  }
  function resumeTurntable() {
    window.clearTimeout(resume.current)
    resume.current = window.setTimeout(() => {
      if (controls.current && !reducedMotion) controls.current.autoRotate = true
    }, 4000)
  }

  return (
    <ErrorBoundary title="Viewer unavailable" message="This browser may not support WebGL, or the file may not have downloaded fully. The download links still work.">
      <div className="relative aspect-[4/3] overflow-hidden rounded-xl border border-line bg-[radial-gradient(ellipse_at_50%_85%,rgb(255_106_43/0.12),#0f0f0e_65%)] lg:aspect-[16/10]">
        <Canvas
          frameloop={reducedMotion ? 'demand' : 'always'}
          dpr={[1, 1.75]}
          camera={{ position: START_POSITION, fov: 32, near: 0.1, far: 80 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          onCreated={({ gl }) => {
            gl.toneMappingExposure = 1.15
          }}
          aria-label="Interactive 3D view of the generated mesh"
        >
          <Environment resolution={256} frames={1}>
            <Lightformer form="rect" intensity={2.4} position={[0, 5, 0]} rotation-x={Math.PI / 2} scale={[10, 10, 1]} />
            <Lightformer form="rect" intensity={1.6} color="#ffffff" position={[-6, 1.5, 1]} rotation-y={Math.PI / 2} scale={[8, 1.2, 1]} />
            <Lightformer form="rect" intensity={1.2} color="#ffffff" position={[6, 1, -2]} rotation-y={-Math.PI / 2} scale={[8, 1, 1]} />
            <Lightformer form="rect" intensity={3} color="#ff6a2b" position={[-2, 0.8, -6]} scale={[10, 1.4, 1]} />
            <Lightformer form="ring" intensity={2.2} color="#ff7a3b" position={[5, 2.5, -4]} scale={4} />
          </Environment>
          <pointLight position={[-3.5, 1.6, -3.5]} color="#ff6a2b" intensity={40} distance={12} decay={2} />
          <pointLight position={[3.5, 3, 3]} color="#ffffff" intensity={14} distance={12} decay={2} />
          <Suspense fallback={null}>
            <Model url={url} look={look} wireframe={wireframe} uniforms={uniforms} reducedMotion={reducedMotion} onReady={onReady} />
          </Suspense>
          {/* No damping: leftover momentum would carry the camera away from a reset. */}
          <OrbitControls
            ref={controls}
            makeDefault
            enableDamping={false}
            minDistance={3}
            maxDistance={20}
            autoRotate={!reducedMotion}
            autoRotateSpeed={0.9}
            onStart={pauseTurntable}
            onEnd={resumeTurntable}
          />
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
            aria-pressed={look === 'natural'}
            onClick={() => setLook(look === 'holo' ? 'natural' : 'holo')}
            className={`inline-flex min-h-10 items-center gap-1.5 rounded-lg border px-3 text-sm backdrop-blur ${look === 'natural' ? 'border-accent bg-accent/15' : 'border-line-strong bg-page/70 hover:bg-raised'}`}
          >
            <Sparkles aria-hidden className="size-4" />
            Original colours
          </button>
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
