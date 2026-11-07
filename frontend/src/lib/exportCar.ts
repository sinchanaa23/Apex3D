import { Group, LineSegments, Mesh, Scene } from 'three'
import type { Object3D } from 'three'

export type ExportFormat = 'glb' | 'obj' | 'stl'

/** A copy of the car as it is shown, minus the glowing edge lines and with the wheels back at rest, ready to write to a file. */
function cleanCopy(source: Group): Scene {
  const scene = new Scene()
  const copy = source.clone(true)
  const lines: Object3D[] = []
  copy.traverse((node) => {
    if (node instanceof LineSegments) lines.push(node)
    if (node instanceof Mesh && node.userData.centre) {
      node.rotation.set(0, 0, 0)
      node.position.set(0, 0, 0)
    }
  })
  lines.forEach((line) => line.parent?.remove(line))
  scene.add(copy)
  scene.updateMatrixWorld(true)
  return scene
}

function save(data: BlobPart, type: string, filename: string) {
  const url = URL.createObjectURL(new Blob([data], { type }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

/** Writes the showcase car, with its photo colour and proportions, as a GLB, OBJ or STL download. */
export async function exportCar(source: Group, format: ExportFormat, name: string) {
  const scene = cleanCopy(source)
  const base = `apex3d_${name}`
  if (format === 'glb') {
    const { GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js')
    const result = await new GLTFExporter().parseAsync(scene, { binary: true })
    save(result as ArrayBuffer, 'model/gltf-binary', `${base}.glb`)
  } else if (format === 'obj') {
    const { OBJExporter } = await import('three/addons/exporters/OBJExporter.js')
    save(new OBJExporter().parse(scene), 'text/plain', `${base}.obj`)
  } else {
    const { STLExporter } = await import('three/addons/exporters/STLExporter.js')
    save(new STLExporter().parse(scene, { binary: true }) as unknown as BlobPart, 'model/stl', `${base}.stl`)
  }
}

/** A file-name-safe version of a car's name, such as audi-r8-coupe-2012. */
export function slug(label?: string | null) {
  return (label ?? 'showcase-car').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'showcase-car'
}
