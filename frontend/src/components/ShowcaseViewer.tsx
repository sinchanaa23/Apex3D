import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { ContactShadows, Environment, Lightformer, OrbitControls, useGLTF } from '@react-three/drei'
import { useReducedMotion } from 'framer-motion'
import { Palette, RotateCcw } from 'lucide-react'
import { Box3, Color, EdgesGeometry, Group, LineSegments, Material, Mesh, MeshPhysicalMaterial, Vector3, type Object3D } from 'three'
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib'
import { ErrorBoundary } from './ErrorBoundary'
import { Spinner } from './States'
import { createEdgeMaterial, createFloorMaterial, createHoloUniforms, patchSurface, type HoloUniforms } from './hero/shaders'
import type { Proportions } from '../lib/adapt'

const MODEL_URL = '/models/car.glb'
const START_POSITION: [number, number, number] = [4.1, 1.9, 5.4]
const TARGET: [number, number, number] = [0, 0.5, 0]
const SCAN_PERIOD = 5.5
const PULSE_PERIOD = 8.5
const PULSE_LENGTH = 2.8

type Part = 'glass' | 'headlight' | 'taillight' | 'signal' | 'rim' | 'disc' | 'pad' | 'trim' | 'accent' | 'body'

// Names in the Khronos Car Concept model, grouped by what they are made of.
function classify(name: string): Part {
  if (/^(BodyWindshield|BodyRearwindow|BodyWindowsRearSides|BodyDoor[LR]Window)$/.test(name)) return 'glass'
  if (name === 'BodyHeadlights') return 'headlight'
  if (name === 'BodyTaillights') return 'taillight'
  if (name === 'BodyTurnsignalsRear') return 'signal'
  if (/Rim$/.test(name)) return 'rim'
  if (/BrakeDisc$/.test(name)) return 'disc'
  if (/BrakePad$/.test(name)) return 'pad'
  if (/Color2$/.test(name)) return 'accent'
  if (/^Body/.test(name) && !/Gasket|Handle|Topgrill/.test(name)) return 'body'
  return 'trim'
}

function useMaterials(uniforms: HoloUniforms, paint: string | null) {
  const materials = useMemo(() => {
    const physical = (params: ConstructorParameters<typeof MeshPhysicalMaterial>[0]) => new MeshPhysicalMaterial(params)
    const m: Record<Part, MeshPhysicalMaterial> = {
      body: physical({ color: '#121315', metalness: 0.92, roughness: 0.27, clearcoat: 1, clearcoatRoughness: 0.07, envMapIntensity: 1.3 }),
      accent: physical({ color: '#2b2d31', metalness: 0.95, roughness: 0.3, clearcoat: 1, clearcoatRoughness: 0.1, envMapIntensity: 1.3 }),
      trim: physical({ color: '#0b0b0c', metalness: 0.35, roughness: 0.55, envMapIntensity: 0.8 }),
      rim: physical({ color: '#303236', metalness: 1, roughness: 0.28, envMapIntensity: 1.4 }),
      disc: physical({ color: '#4a4c50', metalness: 1, roughness: 0.4, envMapIntensity: 1.1 }),
      glass: physical({ color: '#05070a', metalness: 0.2, roughness: 0.05, transparent: true, opacity: 0.62, depthWrite: false, envMapIntensity: 2 }),
      pad: physical({ color: '#ff6a2b', emissive: '#ff5a14', emissiveIntensity: 1.8, metalness: 0.2, roughness: 0.5 }),
      headlight: physical({ color: '#ffffff', emissive: '#fff4e6', emissiveIntensity: 2.6, roughness: 0.2 }),
      taillight: physical({ color: '#ff3a1a', emissive: '#ff2a0a', emissiveIntensity: 2.4, roughness: 0.3 }),
      signal: physical({ color: '#ff8a2b', emissive: '#ff7a14', emissiveIntensity: 1.8, roughness: 0.3 }),
    }
    const lit: [Part, number, number][] = [['body', 1, 1], ['accent', 1, 1], ['trim', 0.35, 0.4], ['rim', 0.7, 0.8], ['disc', 0.5, 0.5], ['glass', 0.45, 1.2]]
    for (const [part, grid, rim] of lit) patchSurface(m[part], uniforms, { grid, rim })
    return m
  }, [uniforms])

  // The photo's colour goes on the body panels. Without it the body keeps the landing page's dark finish.
  useEffect(() => {
    if (paint) {
      materials.body.color = new Color(paint)
      materials.body.metalness = 0.65
      materials.body.roughness = 0.24
    } else {
      materials.body.color = new Color('#121315')
      materials.body.metalness = 0.92
      materials.body.roughness = 0.27
    }
  }, [materials, paint])

  useEffect(() => () => Object.values(materials).forEach((material: Material) => material.dispose()), [materials])
  return materials
}

interface Wheel {
  mesh: Mesh
  speed: number
}

function Car({ uniforms, paint, proportions, reducedMotion, onReady, onModel }: { uniforms: HoloUniforms; paint: string | null; proportions: Proportions; reducedMotion: boolean; onReady(): void; onModel?(group: Group | null): void }) {
  const outer = useRef<Group>(null)
  const { scene } = useGLTF(MODEL_URL, false, false) as unknown as { scene: Group }
  const materials = useMaterials(uniforms, paint)
  const edgeMaterial = useMemo(() => createEdgeMaterial(uniforms), [uniforms])

  const built = useMemo(() => {
    const root = scene.clone(true)
    const edges: EdgesGeometry[] = []
    const wheels: Wheel[] = []
    root.traverse((node: Object3D) => {
      if (!(node instanceof Mesh)) return
      const part = classify(node.name)
      node.material = materials[part]
      node.frustumCulled = false
      if (part === 'body' || part === 'accent') {
        const geometry = new EdgesGeometry(node.geometry, 32)
        edges.push(geometry)
        const lines = new LineSegments(geometry, edgeMaterial)
        lines.frustumCulled = false
        node.add(lines)
      }
      if (part === 'rim' || part === 'disc') {
        node.geometry.computeBoundingBox()
        node.userData.centre = node.geometry.boundingBox!.getCenter(new Vector3())
        wheels.push({ mesh: node, speed: -2.2 })
      }
    })
    const box = new Box3().setFromObject(root)
    const centre = box.getCenter(new Vector3())
    // The tyres rest on y = 0, which is where the floor grid is.
    return { root, edges, wheels, offset: new Vector3(-centre.x, -box.min.y, -centre.z) }
  }, [scene, materials, edgeMaterial])

  useEffect(
    () => () => {
      built.edges.forEach((geometry) => geometry.dispose())
      edgeMaterial.dispose()
    },
    [built, edgeMaterial],
  )

  useFrame((state, delta) => {
    const t = reducedMotion ? 0 : state.clock.elapsedTime
    uniforms.uTime.value = t
    if (reducedMotion) {
      uniforms.uScan.value = 0.9
      uniforms.uPulse.value = -1
      return
    }
    uniforms.uScan.value = -0.1 + (t % SCAN_PERIOD) * 0.5
    const pulse = (t % PULSE_PERIOD) / PULSE_LENGTH
    uniforms.uPulse.value = pulse <= 1 ? pulse : -1
    const dt = Math.min(delta, 0.05)
    for (const { mesh, speed } of built.wheels) {
      const c = mesh.userData.centre as Vector3
      mesh.rotation.x += speed * dt
      mesh.position.set(0, 0, 0)
      mesh.position.y = c.y - (c.y * Math.cos(mesh.rotation.x) - c.z * Math.sin(mesh.rotation.x))
      mesh.position.z = c.z - (c.y * Math.sin(mesh.rotation.x) + c.z * Math.cos(mesh.rotation.x))
    }
  })

  useEffect(() => {
    onReady()
  }, [onReady])

  // Hands the car to the page so it can be downloaded exactly as shown.
  useEffect(() => {
    onModel?.(outer.current)
    return () => onModel?.(null)
  }, [onModel, built])

  // Scaled about the floor, so a taller car grows upwards from the same ground.
  return (
    <group ref={outer} scale={[proportions.width, proportions.height, proportions.length]}>
      <group position={built.offset}>
        <primitive object={built.root} />
      </group>
    </group>
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

/** A designed car in the landing page's style, adapted to the user's photo. Not a reconstruction of their car. */
export default function ShowcaseViewer({ paint, proportions, onModel }: { paint: string | null; proportions: Proportions; onModel?(group: Group | null): void }) {
  const controls = useRef<OrbitControlsImpl>(null)
  const reducedMotion = useReducedMotion() ?? false
  const uniforms = useMemo(createHoloUniforms, [])
  const [usePhotoColour, setUsePhotoColour] = useState(true)
  const [ready, setReady] = useState(false)
  const resume = useRef(0)
  const onReady = useCallback(() => setReady(true), [])

  useEffect(() => () => window.clearTimeout(resume.current), [])

  function resetView() {
    const orbit = controls.current
    if (!orbit) return
    orbit.object.position.set(...START_POSITION)
    orbit.target.set(...TARGET)
    orbit.update()
  }
  function pause() {
    window.clearTimeout(resume.current)
    if (controls.current) controls.current.autoRotate = false
  }
  function later() {
    window.clearTimeout(resume.current)
    resume.current = window.setTimeout(() => {
      if (controls.current && !reducedMotion) controls.current.autoRotate = true
    }, 4000)
  }

  return (
    <ErrorBoundary title="Viewer unavailable" message="This browser may not support WebGL.">
      <div className="relative aspect-[4/3] overflow-hidden rounded-xl border border-line bg-[radial-gradient(ellipse_at_50%_85%,rgb(255_106_43/0.12),#0f0f0e_65%)] lg:aspect-[16/10]">
        <Canvas
          frameloop={reducedMotion ? 'demand' : 'always'}
          dpr={[1, 1.75]}
          camera={{ position: START_POSITION, fov: 32, near: 0.1, far: 80 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          onCreated={({ gl }) => {
            gl.toneMappingExposure = 1.15
          }}
          aria-label="Interactive 3D showcase car adapted to your photo"
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
          <Floor uniforms={uniforms} />
          <ContactShadows position={[0, 0.01, 0]} opacity={0.85} scale={9} blur={2.6} far={1.7} resolution={256} frames={reducedMotion ? 3 : Infinity} color="#000000" />
          <Suspense fallback={null}>
            <Car uniforms={uniforms} paint={usePhotoColour ? paint : null} proportions={proportions} reducedMotion={reducedMotion} onReady={onReady} onModel={onModel} />
          </Suspense>
          <OrbitControls
            ref={controls}
            makeDefault
            enableDamping={false}
            target={TARGET}
            minDistance={3}
            maxDistance={16}
            maxPolarAngle={Math.PI / 2 - 0.05}
            autoRotate={!reducedMotion}
            autoRotateSpeed={1.1}
            onStart={pause}
            onEnd={later}
          />
        </Canvas>
        {!ready && (
          <div className="pointer-events-none absolute inset-0 grid place-items-center">
            <div className="w-40 text-center">
              <Spinner label="Loading car" />
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
          Showcase model
        </p>
        <div className="absolute top-3 right-3 flex gap-2">
          {paint && (
            <button
              type="button"
              aria-pressed={usePhotoColour}
              onClick={() => setUsePhotoColour(!usePhotoColour)}
              className={`inline-flex min-h-10 items-center gap-1.5 rounded-lg border px-3 text-sm backdrop-blur ${usePhotoColour ? 'border-accent bg-accent/15' : 'border-line-strong bg-page/70 hover:bg-raised'}`}
            >
              <Palette aria-hidden className="size-4" />
              Photo colour
            </button>
          )}
          <button type="button" onClick={resetView} className="inline-flex min-h-10 items-center gap-1.5 rounded-lg border border-line-strong bg-page/70 px-3 text-sm backdrop-blur hover:bg-raised">
            <RotateCcw aria-hidden className="size-4" />
            Reset view
          </button>
        </div>
        <p className="pointer-events-none absolute bottom-3 left-8 text-xs text-ink-3">
          <span className="touch:hidden">Drag to rotate · scroll to zoom</span>
          <span className="hidden touch:inline">Drag to rotate · pinch to zoom</span>
        </p>
      </div>
    </ErrorBoundary>
  )
}
