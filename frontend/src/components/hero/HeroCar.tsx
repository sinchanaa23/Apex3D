import { useEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { ContactShadows, Environment, Lightformer, useGLTF } from '@react-three/drei'
import {
  BufferGeometry,
  Float32BufferAttribute,
  Group,
  LineSegments,
  EdgesGeometry,
  Mesh,
  MeshPhysicalMaterial,
  Box3,
  Vector3,
  type Material,
  type Object3D,
} from 'three'
import {
  createEdgeMaterial,
  createFloorMaterial,
  createHoloUniforms,
  createParticleMaterial,
  patchSurface,
  type HoloUniforms,
} from './shaders'

const MODEL_URL = '/models/car.glb'
const FLOOR_Y = -1.0
const SCAN_PERIOD = 5.5
const PULSE_PERIOD = 8.5
const PULSE_LENGTH = 2.8

interface Props {
  reducedMotion: boolean
  /** False while the hero is scrolled out of view, so nothing renders off screen. */
  visible: boolean
  /** Pointer position as -1 to 1 across the window. Updated by the parent, read every frame. */
  pointer: { current: { x: number; y: number } }
  onReady(): void
}

type Part = 'glass' | 'headlight' | 'taillight' | 'signal' | 'rim' | 'disc' | 'pad' | 'trim' | 'accent' | 'body'

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

function useMaterials(uniforms: HoloUniforms) {
  const materials = useMemo(() => {
    const physical = (params: ConstructorParameters<typeof MeshPhysicalMaterial>[0]) => new MeshPhysicalMaterial(params)
    const m: Record<Part, Material> = {
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
    for (const [part, grid, rim] of lit) patchSurface(m[part] as MeshPhysicalMaterial, uniforms, { grid, rim })
    return m
  }, [uniforms])

  useEffect(() => () => Object.values(materials).forEach((material) => material.dispose()), [materials])
  return materials
}

interface Wheel {
  mesh: Mesh
  /** Spin speed in radians per second. */
  speed: number
}

/** Builds the displayed car from the cached glTF without altering the cached copy. */
function Car({ uniforms, reducedMotion, onReady, carRef }: { uniforms: HoloUniforms; reducedMotion: boolean; onReady(): void; carRef: React.RefObject<Group | null> }) {
  const { scene } = useGLTF(MODEL_URL, false, false) as unknown as { scene: Group }
  const invalidate = useThree((state) => state.invalidate)
  const materials = useMaterials(uniforms)
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
        // Geometry is already in car space, so spin about the wheel's own centre, with the axle along x.
        node.geometry.computeBoundingBox()
        const centre = node.geometry.boundingBox!.getCenter(new Vector3())
        node.userData.centre = centre
        wheels.push({ mesh: node, speed: part === 'rim' ? -2.2 : -2.2 })
      }
    })
    const box = new Box3().setFromObject(root)
    const centre = box.getCenter(new Vector3())
    return { root, edges, wheels, offset: new Vector3(-centre.x, -box.min.y - 0.575, -centre.z) }
  }, [scene, materials, edgeMaterial])

  useEffect(() => {
    return () => {
      built.edges.forEach((geometry) => geometry.dispose())
      edgeMaterial.dispose()
    }
  }, [built, edgeMaterial])

  // Wheels turn about their own centre without moving their geometry: translate, rotate, translate back.
  useFrame((_, delta) => {
    if (reducedMotion) return
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
    const frame = requestAnimationFrame(() => invalidate())
    return () => cancelAnimationFrame(frame)
  }, [onReady, invalidate])

  return (
    <group ref={carRef}>
      <group position={built.offset}>
        <primitive object={built.root} />
      </group>
    </group>
  )
}

function Particles({ uniforms }: { uniforms: HoloUniforms }) {
  const dpr = useThree((state) => state.viewport.dpr)
  const geometry = useMemo(() => {
    const count = 240
    const positions = new Float32Array(count * 3)
    const seeds = new Float32Array(count)
    // A fixed sequence, so the scene looks the same on every load.
    let s = 7
    const rand = () => ((s = (s * 16807) % 2147483647) - 1) / 2147483646
    for (let i = 0; i < count; i++) {
      const radius = 1.5 + rand() * 3.2
      const angle = rand() * Math.PI * 2
      positions.set([Math.cos(angle) * radius, -1.1 + rand() * 3.8, Math.sin(angle) * radius], i * 3)
      seeds[i] = rand()
    }
    const g = new BufferGeometry()
    g.setAttribute('position', new Float32BufferAttribute(positions, 3))
    g.setAttribute('aSeed', new Float32BufferAttribute(seeds, 1))
    return g
  }, [])
  const material = useMemo(() => createParticleMaterial(uniforms, dpr), [uniforms, dpr])
  useEffect(() => () => { geometry.dispose(); material.dispose() }, [geometry, material])
  return <points geometry={geometry} material={material} frustumCulled={false} />
}

function Floor({ uniforms }: { uniforms: HoloUniforms }) {
  const material = useMemo(() => createFloorMaterial(uniforms), [uniforms])
  useEffect(() => () => material.dispose(), [material])
  return (
    <mesh position={[0, FLOOR_Y, 0]} rotation-x={-Math.PI / 2} material={material} renderOrder={-1}>
      <planeGeometry args={[14, 14]} />
    </mesh>
  )
}

/** Drives the animation: rotation, bobbing, scan, pulses, and the camera's response to the pointer. */
function Director({ uniforms, carRef, reducedMotion, pointer }: { uniforms: HoloUniforms; carRef: React.RefObject<Group | null>; reducedMotion: boolean; pointer: Props['pointer'] }) {
  const camera = useThree((state) => state.camera)
  const size = useThree((state) => state.size)
  const invalidate = useThree((state) => state.invalidate)
  const clock = useRef(0)
  const look = useMemo(() => new Vector3(), [])

  // Narrow or tall frames need the camera further back to keep the whole car in view.
  const aspect = size.width / Math.max(size.height, 1)
  // The car's bounding sphere has a radius of about 2.6, so this keeps it inside the frame at every angle.
  const distance = 8.9 * Math.max(1, 1.05 / aspect)

  useEffect(() => {
    if (!reducedMotion) return
    // A still, well-lit pose: scan resting partway up so the grid is visible.
    uniforms.uScan.value = 0.95
    uniforms.uPulse.value = -1
    uniforms.uTime.value = 0
    if (carRef.current) {
      carRef.current.rotation.y = -0.75
      carRef.current.updateMatrixWorld(true)
      uniforms.uInvCar.value.copy(carRef.current.matrixWorld).invert()
    }
    camera.position.set(distance * 0.58, distance * 0.27, distance * 0.77)
    camera.lookAt(0, -0.1, 0)
    invalidate()
  }, [reducedMotion, uniforms, carRef, camera, distance, invalidate])

  useFrame((_, delta) => {
    if (reducedMotion) return
    const dt = Math.min(delta, 0.05)
    clock.current += dt
    const t = clock.current

    const car = carRef.current
    if (car) {
      car.rotation.y = -0.75 + t * 0.28
      car.rotation.z = Math.sin(t * 0.6) * 0.012
      car.position.y = Math.sin(t * 0.9) * 0.05
      car.updateMatrixWorld(true)
      uniforms.uInvCar.value.copy(car.matrixWorld).invert()
    }

    uniforms.uTime.value = t
    uniforms.uScan.value = -0.25 + (t % SCAN_PERIOD) * 0.62
    const pulse = (t % PULSE_PERIOD) / PULSE_LENGTH
    uniforms.uPulse.value = pulse <= 1 ? pulse : -1

    // The camera leans a little towards the pointer and eases back, without capturing any input.
    const target = new Vector3(distance * 0.58 + pointer.current.x * 0.7, distance * 0.27 - pointer.current.y * 0.35, distance * 0.77)
    camera.position.lerp(target, 1 - Math.exp(-dt * 3))
    camera.lookAt(look.set(0, -0.1, 0))
  })
  return null
}

export default function HeroCar({ reducedMotion, visible, pointer, onReady }: Props) {
  const uniforms = useMemo(createHoloUniforms, [])
  const carRef = useRef<Group>(null)

  // Frees the cached model's GPU buffers and removes it from the loader cache when the hero goes away.
  useEffect(
    () => () => {
      const cached = useGLTF as unknown as { clear(url: string): void }
      cached.clear(MODEL_URL)
    },
    [],
  )

  return (
    <Canvas
      frameloop={reducedMotion ? 'demand' : visible ? 'always' : 'never'}
      dpr={[1, 1.75]}
      camera={{ fov: 32, near: 0.1, far: 60, position: [5.16, 2.4, 6.85] }}
      gl={{ alpha: true, antialias: true, powerPreference: 'high-performance' }}
      onCreated={({ gl }) => {
        gl.toneMappingExposure = 1.15
      }}
      // Nothing in the scene takes input, so the page keeps scrolling normally over it.
      style={{ pointerEvents: 'none' }}
      aria-hidden
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
      <ContactShadows position={[0, FLOOR_Y + 0.01, 0]} opacity={0.85} scale={9} blur={2.6} far={1.7} resolution={256} frames={reducedMotion ? 3 : Infinity} color="#000000" />
      <Car uniforms={uniforms} reducedMotion={reducedMotion} onReady={onReady} carRef={carRef} />
      {!reducedMotion && <Particles uniforms={uniforms} />}
      <Director uniforms={uniforms} carRef={carRef} reducedMotion={reducedMotion} pointer={pointer} />
    </Canvas>
  )
}
