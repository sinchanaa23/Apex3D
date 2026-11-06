// DEV ONLY. Test responses for `npm run dev:mock`, so the UI can be built before
// Person C's endpoints exist. Registered only on the dev server, never in `npm run build`.
// Change a scenario from the browser: GET /__mock/set?recognize=ambiguous
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin } from 'vite'

const scenario: Record<string, string> = {
  recognize: 'confident', // confident | ambiguous | rejected | nocar | error
  classical: 'ok', // ok | error
  references: 'ok', // ok | empty | error
  job: 'ok', // ok | fail
  metrics: 'partial', // partial | full | none | malformed | error
}

const placeholder = (text: string, hue = 24, w = 640, h = 420) =>
  'data:image/svg+xml,' +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><rect width="100%" height="100%" fill="hsl(${hue} 18% 18%)"/><text x="50%" y="50%" fill="#c3c2b7" font-family="monospace" font-size="24" text-anchor="middle">MOCK: ${text}</text></svg>`,
  )

const top5 = [
  { class_id: 11, label: 'MOCK Coupe Alpha 2011', probability: 0.71 },
  { class_id: 12, label: 'MOCK Coupe Alpha 2009', probability: 0.14 },
  { class_id: 40, label: 'MOCK Sedan Beta 2012', probability: 0.06 },
  { class_id: 77, label: 'MOCK Hatchback Gamma 2010', probability: 0.03 },
  { class_id: 103, label: 'MOCK Convertible Delta 2007', probability: 0.02 },
]

function recognizeBody(state: string) {
  const base = {
    image_size: { width: 1000, height: 1000 },
    detection: { box: [120, 280, 880, 760], label: 'car', score: 0.98 },
    timings_ms: { detection: 412, classification: 96, total: 540 },
  }
  if (state === 'nocar') {
    return { ...base, detection: null, top5, confidence: { state: 'rejected', top1: 0.22, margin: 0.03, reason: 'No vehicle detected. The full frame was classified.' } }
  }
  if (state === 'ambiguous') {
    return {
      ...base,
      top5: [{ ...top5[0], probability: 0.41 }, { ...top5[1], probability: 0.38 }, ...top5.slice(2)],
      confidence: { state, top1: 0.41, margin: 0.03, reason: 'The top two classes are too close to separate.' },
    }
  }
  if (state === 'rejected') {
    return {
      ...base,
      top5: top5.map((p, i) => ({ ...p, probability: [0.18, 0.12, 0.09, 0.07, 0.05][i] })),
      confidence: { state, top1: 0.18, margin: 0.06, reason: 'Confidence is below the rejection threshold.' },
    }
  }
  return { ...base, top5, confidence: { state: 'confident', top1: 0.71, margin: 0.57, reason: null } }
}

const classicalBody = {
  steps: ['Original', 'CLAHE', 'Canny edges', 'Contours', 'ROI', 'HSV', 'HOG', 'LBP'].map((title, i) => ({
    key: title.toLowerCase(),
    title,
    image: placeholder(title, i * 40),
    caption: i === 1 ? 'Contrast limited equalisation on LAB luminance.' : null,
  })),
  features: {
    aspect_ratio: 1.58,
    extent: 0.71,
    solidity: 0.88,
    hu_moment_1: 0.00213,
    hsv_histogram: Array.from({ length: 32 }, (_, i) => Math.abs(Math.sin(i / 3))),
    hog_descriptor: Array.from({ length: 3780 }, () => 0.1),
    lbp_histogram: Array.from({ length: 26 }, (_, i) => (i % 7) / 7),
  },
  timings_ms: { total: 180 },
}

const views = ['front', 'front_three_quarter', 'side', 'rear', 'rear_three_quarter', null]
const references = Array.from({ length: 8 }, (_, i) => ({
  id: `ref-${i}`,
  image_url: placeholder(`reference ${i}`, 200 + i * 15),
  source: 'Wikimedia Commons',
  source_url: 'https://commons.wikimedia.org/',
  author: `Mock author ${i}`,
  licence: i % 2 ? 'CC BY-SA 4.0' : 'CC BY 2.0',
  licence_url: 'https://creativecommons.org/licenses/',
  viewpoint: views[i % views.length],
  match: i % 3 === 2 ? 'approximate' : 'confirmed',
}))

const section = <T,>(file: string, data: T) => ({ source: `notebooks/outputs/${file}`, generated_at: '2026-11-06T00:00:00Z', data })
const perturbations = ['brightness', 'low_light', 'blur', 'noise', 'low_resolution', 'occlusion']
const fullMetrics = {
  classifier: section('classifier.json', { model: 'MOCK model', top1: 0.5, top5: 0.75, top1_detector_crop: 0.48, top5_detector_crop: 0.73, test_images: 100 }),
  detection: section('detection.json', { rows: [{ model: 'MOCK R-CNN A', mean_iou: 0.5, map: 0.4, cpu_ms: 900 }, { model: 'MOCK R-CNN B', mean_iou: 0.45, map: 0.35, cpu_ms: 300 }] }),
  comparison: section('comparison.json', { rows: [{ approach: 'MOCK Classical', top1: 0.2, top5: 0.4, inference_ms: 90, size_mb: 12 }, { approach: 'MOCK Deep', top1: 0.5, top5: 0.75, inference_ms: 60, size_mb: 110 }, { approach: 'MOCK Hybrid', top1: 0.52, top5: 0.76, inference_ms: 150, size_mb: 122 }] }),
  optimisation: section('optimisation.json', { rows: [{ variant: 'MOCK Original', size_mb: 110, cpu_ms: 180, gpu_ms: 12, top1: 0.5 }, { variant: 'MOCK INT8 ONNX', size_mb: 29, cpu_ms: 70, gpu_ms: null, top1: 0.49 }] }),
  robustness: section('robustness.json', { rows: perturbations.flatMap((perturbation, p) => [0, 1, 2, 3].map((severity) => ({ perturbation, severity, top1: 0.5 - severity * (0.03 + p * 0.015) }))) }),
  confidence: section('confidence.json', { chosen_threshold: 0.5, rows: Array.from({ length: 10 }, (_, i) => ({ threshold: i / 10, coverage: 1 - i * 0.09, accuracy: 0.5 + i * 0.04 })) }),
  ablation: section('ablation.json', { baseline_top1: 0.5, rows: [{ removed: 'MOCK Detector crop', top1: 0.45 }, { removed: 'MOCK CLAHE', top1: 0.49 }, { removed: 'MOCK Augmentation', top1: 0.42, note: 'Fewer epochs' }] }),
  confused_pairs: section('confusion.json', { rows: [{ a: 'MOCK Coupe 2011', b: 'MOCK Coupe 2009', count: 9 }, { a: 'MOCK Sedan 2012', b: 'MOCK Sedan 2010', count: 6 }] }),
  failure_cases: section('failures.json', { cases: [{ title: 'MOCK heavy occlusion', image: placeholder('failure 1'), predicted: 'MOCK Sedan', actual: 'MOCK Coupe', explanation: 'Mock explanation.' }] }),
}

function metricsBody(state: string) {
  if (state === 'none') return Object.fromEntries(Object.keys(fullMetrics).map((key) => [key, 'not available']))
  if (state === 'malformed') return { ...fullMetrics, comparison: section('comparison.json', { nope: true }), robustness: section('r.json', { rows: 4 }) }
  if (state === 'partial') return { ...fullMetrics, ablation: 'not available', failure_cases: 'not available' }
  return fullMetrics
}

/** A blocky GLB with vertex colours and no normals, the same kind of file trimesh writes for TripoSR. */
function mockGlb(): Buffer {
  const boxes = [
    { min: [-1, 0.2, -0.45], max: [1, 0.55, 0.45], color: [0.75, 0.3, 0.15] },
    { min: [-0.5, 0.55, -0.38], max: [0.45, 0.85, 0.38], color: [0.55, 0.6, 0.65] },
    { min: [-0.8, 0, -0.5], max: [-0.45, 0.28, 0.5], color: [0.08, 0.08, 0.08] },
    { min: [0.45, 0, -0.5], max: [0.8, 0.28, 0.5], color: [0.08, 0.08, 0.08] },
  ]
  const faces = [[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6], [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]]
  const positions: number[] = []
  const colors: number[] = []
  const indices: number[] = []
  for (const { min, max, color } of boxes) {
    const start = positions.length / 3
    for (const z of [min[2], max[2]]) {
      for (const [x, y] of [[min[0], min[1]], [max[0], min[1]], [max[0], max[1]], [min[0], max[1]]]) {
        positions.push(x, y, z)
        colors.push(...color)
      }
    }
    for (const face of faces) indices.push(...face.map((v) => v + start))
  }
  const indexBytes = Buffer.from(new Uint16Array(indices).buffer)
  const pad = (4 - (indexBytes.length % 4)) % 4
  const positionBytes = Buffer.from(new Float32Array(positions).buffer)
  const colorBytes = Buffer.from(new Float32Array(colors).buffer)
  const bin = Buffer.concat([indexBytes, Buffer.alloc(pad), positionBytes, colorBytes])
  const positionOffset = indexBytes.length + pad
  const json = {
    asset: { version: '2.0' },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 1, COLOR_0: 2 }, indices: 0 }] }],
    buffers: [{ byteLength: bin.length }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: indexBytes.length, target: 34963 },
      { buffer: 0, byteOffset: positionOffset, byteLength: positionBytes.length, target: 34962 },
      { buffer: 0, byteOffset: positionOffset + positionBytes.length, byteLength: colorBytes.length, target: 34962 },
    ],
    accessors: [
      { bufferView: 0, componentType: 5123, count: indices.length, type: 'SCALAR' },
      { bufferView: 1, componentType: 5126, count: positions.length / 3, type: 'VEC3', min: [-1, 0, -0.5], max: [1, 0.85, 0.5] },
      { bufferView: 2, componentType: 5126, count: colors.length / 3, type: 'VEC3' },
    ],
  }
  const text = Buffer.from(JSON.stringify(json))
  const jsonChunk = Buffer.concat([text, Buffer.alloc((4 - (text.length % 4)) % 4, 0x20)])
  const binChunk = Buffer.concat([bin, Buffer.alloc((4 - (bin.length % 4)) % 4)])
  const header = Buffer.alloc(12)
  header.writeUInt32LE(0x46546c67, 0)
  header.writeUInt32LE(2, 4)
  header.writeUInt32LE(12 + 8 + jsonChunk.length + 8 + binChunk.length, 8)
  const chunkHeader = (length: number, type: number) => {
    const b = Buffer.alloc(8)
    b.writeUInt32LE(length, 0)
    b.writeUInt32LE(type, 4)
    return b
  }
  return Buffer.concat([header, chunkHeader(jsonChunk.length, 0x4e4f534a), jsonChunk, chunkHeader(binChunk.length, 0x004e4942), binChunk])
}

const jobs = new Map<string, number>()
function jobBody(id: string) {
  const age = Date.now() - (jobs.get(id) ?? 0)
  const logs = ['Job accepted']
  if (age < 2500) return { id, status: 'queued', queue_position: age < 1200 ? 2 : 1, logs }
  logs.push('Removing background', 'Running reconstruction')
  if (age < 5500) return { id, status: 'running', stage: 'Reconstructing mesh', logs }
  if (scenario.job === 'fail') return { id, status: 'failed', logs, error: 'GPU ran out of memory while generating the mesh.' }
  logs.push('Validating mesh', 'Exporting GLB, OBJ and STL')
  return {
    id,
    status: 'succeeded',
    stage: 'Done',
    logs,
    result: {
      asset_id: id,
      generation_seconds: 3.1,
      files: [
        { name: 'model.glb', format: 'glb', size_bytes: mockGlb().length },
        { name: 'model.obj', format: 'obj', size_bytes: 2100 },
        { name: 'model.stl', format: 'stl', size_bytes: 1900 },
        { name: 'validation.json', format: 'json', size_bytes: 180 },
      ],
    },
  }
}

const readBody = (req: IncomingMessage) =>
  new Promise<Buffer>((resolve) => {
    const chunks: Buffer[] = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', () => resolve(Buffer.concat(chunks)))
  })

function send(res: ServerResponse, status: number, body: unknown, delay = 400) {
  setTimeout(() => {
    res.statusCode = status
    res.setHeader('content-type', 'application/json')
    res.end(JSON.stringify(body))
  }, delay)
}

export function mockApi(): Plugin {
  return {
    name: 'apex3d-mock-api',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url ?? '/', 'http://mock')
        const path = url.pathname
        if (path === '/__mock/set') {
          url.searchParams.forEach((value, key) => (scenario[key] = value))
          return send(res, 200, scenario, 0)
        }
        if (!path.startsWith('/api/')) return next()
        const body = req.method === 'POST' ? (await readBody(req)).toString('latin1') : ''
        const has = (field: string) => body.includes(`name="${field}"`)

        if (path === '/api/recognize' && req.method === 'POST') {
          if (!has('image')) return send(res, 422, { detail: 'image field missing' })
          if (scenario.recognize === 'error') return send(res, 415, { detail: 'The server could not decode that image.' })
          return send(res, 200, recognizeBody(scenario.recognize), 1200)
        }
        if (path === '/api/classical' && req.method === 'POST') {
          if (!has('image')) return send(res, 422, { detail: 'image field missing' })
          if (scenario.classical === 'error') return send(res, 500, { detail: 'Classical pipeline failed on this image.' })
          return send(res, 200, classicalBody, 900)
        }
        if (path === '/api/references' && req.method === 'GET') {
          if (scenario.references === 'error') return send(res, 503, { detail: 'Wikimedia Commons could not be reached.' })
          return send(res, 200, { query: 'MOCK query', candidates: scenario.references === 'empty' ? [] : references }, 900)
        }
        if (path === '/api/references/upload' && req.method === 'POST') {
          if (!has('images')) return send(res, 422, { detail: 'images field missing' })
          const id = `user-${Date.now()}`
          return send(res, 200, { candidates: [{ id, image_url: placeholder('your upload', 120), source: 'Your upload', match: 'user', viewpoint: null }] })
        }
        if (path === '/api/reconstruct' && req.method === 'POST') {
          if (!has('image') && !has('reference_id')) return send(res, 422, { detail: 'image or reference_id required' })
          const id = `job-${Date.now()}`
          jobs.set(id, Date.now())
          return send(res, 200, { job_id: id })
        }
        const job = path.match(/^\/api\/jobs\/(.+)$/)
        if (job) {
          if (!jobs.has(job[1])) return send(res, 404, { detail: 'Job not found. It may have been cleaned up.' })
          return send(res, 200, jobBody(job[1]), 100)
        }
        const asset = path.match(/^\/api\/assets\/([^/]+)\/(.+)$/)
        if (asset) {
          if (asset[2] === 'validation.json') {
            return send(res, 200, { watertight: false, non_manifold_edges: 3, degenerate_faces: 0, connected_components: 4, vertices: 32, faces: 48, has_texture: false, has_vertex_colors: true })
          }
          if (asset[2] === 'model.glb') {
            res.setHeader('content-type', 'model/gltf-binary')
            return res.end(mockGlb())
          }
          res.setHeader('content-type', 'text/plain')
          return res.end(`MOCK ${asset[2]}`)
        }
        if (path === '/api/metrics') {
          if (scenario.metrics === 'error') return send(res, 500, { detail: 'Metrics files could not be read.' })
          return send(res, 200, metricsBody(scenario.metrics))
        }
        return send(res, 404, { detail: 'Not in mock' })
      })
    },
  }
}
