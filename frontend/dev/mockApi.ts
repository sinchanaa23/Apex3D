// DEV ONLY. Test responses for `npm run dev:mock`, in the same format as Person C's API
// (docs/API_CONTRACT.md), so the UI's states can be exercised without models or a GPU.
// Registered only on the dev server, never in `npm run build`.
// Change a scenario from the browser: GET /__mock/set?recognize=ambiguous
import { readFileSync } from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import type { Plugin } from 'vite'

const scenario: Record<string, string> = {
  recognize: 'confident', // confident | ambiguous | rejected | nocar | error
  classical: 'ok', // ok | error
  references: 'ok', // ok | empty | error
  job: 'ok', // ok | fail
  metrics: 'partial', // partial | full | persona | none | malformed | error
}

const placeholder = (text: string, hue = 24, w = 640, h = 420) =>
  'data:image/svg+xml,' +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}"><rect width="100%" height="100%" fill="hsl(${hue} 18% 18%)"/><text x="50%" y="50%" fill="#c3c2b7" font-family="monospace" font-size="24" text-anchor="middle">MOCK: ${text}</text></svg>`,
  )

const predictions = [
  { class_id: 11, class_name: 'MOCK Coupe Alpha 2011', probability: 0.71 },
  { class_id: 12, class_name: 'MOCK Coupe Alpha 2009', probability: 0.14 },
  { class_id: 40, class_name: 'MOCK Sedan Beta 2012', probability: 0.06 },
  { class_id: 77, class_name: 'MOCK Hatchback Gamma 2010', probability: 0.03 },
  { class_id: 103, class_name: 'MOCK Convertible Delta 2007', probability: 0.02 },
].map((p) => ({ ...p, make: 'MOCK', model: 'MOCK', year: 2011, body_style: 'Coupe' }))

function recognizeBody(state: string) {
  const base = {
    success: true,
    car_detected: true,
    crop_used: 'detected_bbox',
    bounding_box: { x1: 120, y1: 280, x2: 880, y2: 760, confidence: 0.98, label: 'car' },
    all_boxes: [],
    timings_ms: { detection: 412, classification: 96, total: 540 },
  }
  const probabilities = (values: number[]) => predictions.map((p, i) => ({ ...p, probability: values[i] }))
  if (state === 'nocar') {
    return {
      ...base,
      car_detected: false,
      crop_used: 'full_frame',
      bounding_box: null,
      predictions: probabilities([0.18, 0.12, 0.09, 0.07, 0.05]),
      confidence_state: 'rejected',
      confidence_score: 0.18,
      margin: 0.06,
      message: 'Low confidence: Vehicle may be outside the taxonomy, or poorly visible.',
    }
  }
  if (state === 'ambiguous') {
    return {
      ...base,
      predictions: probabilities([0.41, 0.38, 0.06, 0.03, 0.02]),
      confidence_state: state,
      confidence_score: 0.41,
      margin: 0.03,
      message: 'Ambiguous classification: Multiple closely matching model years or trim levels detected.',
    }
  }
  if (state === 'rejected') {
    return {
      ...base,
      predictions: probabilities([0.18, 0.12, 0.09, 0.07, 0.05]),
      confidence_state: state,
      confidence_score: 0.18,
      margin: 0.06,
      message: 'Low confidence: Vehicle may be outside the taxonomy, or poorly visible.',
    }
  }
  return { ...base, predictions, confidence_state: 'confident', confidence_score: 0.71, margin: 0.57, message: 'High-confidence vehicle identification.' }
}

const classicalBody = {
  success: true,
  intermediate_images: Object.fromEntries(['original', 'clahe', 'edges', 'contours', 'largest_contour_roi', 'hog_visualization'].map((key, i) => [key, placeholder(key, i * 50)])),
  features: {
    geometric: { aspect_ratio: 1.58, extent: 0.71, solidity: 0.88, contour_area: 1000, perimeter: 500, hu_moments: [0.2, 0.01, 0.002, 0.001, 0, 0, 0] },
    hog: { vector_length: 3780, mean: 0.05, std: 0.08, feature_sample: [0.1, 0.04, 0, 0.2, 0.15] },
    lbp: { num_points: 24, radius: 3, histogram: Array.from({ length: 26 }, (_, i) => (i % 7) / 7) },
    color_hsv: { mean_hue: 24, mean_saturation: 142, mean_value: 185, hue_histogram: Array.from({ length: 32 }, (_, i) => Math.abs(Math.sin(i / 3))) },
  },
  timings_ms: { clahe: 8, hog: 24, total: 180 },
}

const views = ['front', 'front_three_quarter', 'side', 'rear', 'rear_three_quarter', 'unknown']
const references = Array.from({ length: 8 }, (_, i) => ({
  id: `wm_${i}`,
  title: `File:Mock reference ${i}.jpg`,
  image_url: placeholder(`reference ${i}`, 200 + i * 15),
  thumb_url: placeholder(`reference ${i}`, 200 + i * 15),
  source: 'wikimedia_commons',
  license: i % 2 ? 'CC BY-SA 4.0' : 'CC BY 2.0',
  license_url: 'https://creativecommons.org/licenses/',
  author: `Mock author ${i}`,
  viewpoint: views[i % views.length],
  viewpoint_confidence: 0.5,
  match_score: 0.85,
  is_approximate: i % 3 === 2,
}))

const perturbations = ['brightness', 'low_light', 'blur', 'noise', 'low_resolution', 'occlusion']
const flatMetrics = {
  test_top1_acc: 0.5,
  test_top5_acc: 0.75,
  detector_mAP50: 0.4,
  detector_mean_iou: 0.5,
  model_size_mb: 110,
  quantized_size_mb: 29,
  inference_time_gpu_ms: 12,
  inference_time_cpu_ms: 180,
  inference_time_quantized_cpu_ms: 70,
  robustness: { noise_gaussian_top1: 0.4, blur_motion_top1: 0.38, brightness_low_top1: 0.42, occlusion_synthetic_top1: 0.3 },
  comparison: { classical_svm_top1: 0.2, deep_convnext_top1: 0.5, hybrid_fusion_top1: 0.52 },
}
const fullMetrics = {
  ...flatMetrics,
  model_name: 'MOCK model',
  test_images: 100,
  test_top1_acc_detector_crop: 0.48,
  test_top5_acc_detector_crop: 0.73,
  generated_at: '2026-11-06T00:00:00Z',
  robustness: { rows: perturbations.flatMap((perturbation, p) => [0, 1, 2, 3].map((severity) => ({ perturbation, severity, top1: 0.5 - severity * (0.03 + p * 0.015) }))) },
  confidence: { chosen_threshold: 0.5, rows: Array.from({ length: 10 }, (_, i) => ({ threshold: i / 10, coverage: 1 - i * 0.09, accuracy: 0.5 + i * 0.04 })) },
  ablation: { baseline_top1: 0.5, rows: [{ removed: 'MOCK Detector crop', top1: 0.45 }, { removed: 'MOCK CLAHE', top1: 0.49 }, { removed: 'MOCK Augmentation', top1: 0.42, note: 'Fewer epochs' }] },
  confused_pairs: { rows: [{ a: 'MOCK Coupe 2011', b: 'MOCK Coupe 2009', count: 9 }, { a: 'MOCK Sedan 2012', b: 'MOCK Sedan 2010', count: 6 }] },
  failure_cases: { cases: [{ title: 'MOCK heavy occlusion', image: placeholder('failure 1'), predicted: 'MOCK Sedan', actual: 'MOCK Coupe', explanation: 'Mock explanation.' }] },
}

function metricsBody(state: string) {
  // Person A's real saved files, served the way the backend would serve them once metrics.json combines both.
  if (state === 'persona') {
    const read = (name: string) => JSON.parse(readFileSync(new URL(`../src/api/fixtures/${name}`, import.meta.url), 'utf-8'))
    return {
      status: 'available',
      source_file: 'metrics.json',
      message: 'MOCK',
      metrics: { classifier_metrics: read('person_a_classifier_metrics.json'), optimization_metrics: read('person_a_optimization_metrics.json') },
    }
  }
  if (state === 'none') return { status: 'not available', message: 'MOCK: metrics have not been generated yet.', metrics: null }
  const metrics = state === 'full' ? fullMetrics : state === 'malformed' ? { ...fullMetrics, ablation: { rows: 4 }, confidence: { nope: true } } : flatMetrics
  return { status: 'available', source_file: 'metrics.json', message: 'MOCK', metrics }
}

/** A blocky GLB with vertex colours and no normals, the same kind of file trimesh writes. */
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
  const created = jobs.get(id) ?? 0
  const age = Date.now() - created
  const iso = (ms: number) => new Date(ms).toISOString()
  const base = { job_id: id, created_at: iso(created), error: null, assets: null, validation: null }
  const logs = ['Job accepted']
  if (age < 2500) return { ...base, status: 'queued', stage: 'waiting', progress: 0, queue_position: age < 1200 ? 2 : 1, logs }
  logs.push('Removing background', 'Running reconstruction')
  if (age < 5500) return { ...base, status: 'running', stage: 'triposr_inference', progress: 0.6, queue_position: 0, logs, started_at: iso(created + 2500) }
  if (scenario.job === 'fail') return { ...base, status: 'failed', stage: 'failed', progress: 0, queue_position: 0, logs, error: 'GPU ran out of memory while generating the mesh.' }
  logs.push('Validating mesh', 'Exporting GLB, OBJ and STL')
  return {
    ...base,
    status: 'completed',
    stage: 'done',
    progress: 1,
    queue_position: 0,
    logs,
    started_at: iso(created + 2500),
    completed_at: iso(created + 5500),
    assets: {
      glb_url: `/api/assets/${id}/mesh.glb`,
      obj_url: `/api/assets/${id}/mesh.obj`,
      stl_url: `/api/assets/${id}/mesh.stl`,
      validation_report_url: `/api/assets/${id}/validation.json`,
    },
    validation: {
      is_watertight: false,
      is_manifold: false,
      vertex_count: 32,
      face_count: 48,
      degenerate_faces: 0,
      connected_components: 4,
      has_vertex_colors: true,
      has_uv_texture: false,
      bounding_box_extents: [2, 0.85, 1],
      asset_readiness: 'needs_repair',
      scale_status: 'unknown',
      notes: ['MOCK note: unseen sides were inferred from a single view.'],
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
          if (!has('file')) return send(res, 422, { detail: 'file field missing' })
          if (scenario.recognize === 'error') return send(res, 422, { detail: 'Failed to decode image file. File may be corrupted.' })
          return send(res, 200, recognizeBody(scenario.recognize), 1200)
        }
        if (path === '/api/classical' && req.method === 'POST') {
          if (!has('file')) return send(res, 422, { detail: 'file field missing' })
          if (scenario.classical === 'error') return send(res, 500, { detail: 'Classical pipeline failed on this image.' })
          return send(res, 200, classicalBody, 900)
        }
        if (path === '/api/references' && req.method === 'GET') {
          if (scenario.references === 'error') return send(res, 503, { detail: 'Wikimedia Commons could not be reached.' })
          const candidates = scenario.references === 'empty' ? [] : references
          return send(res, 200, { success: true, query: 'MOCK query', total_candidates: candidates.length, candidates }, 900)
        }
        if (path === '/api/references/upload' && req.method === 'POST') {
          if (!has('file')) return send(res, 422, { detail: 'file field missing' })
          const candidate = {
            id: `upload_${Date.now()}`,
            title: 'photo.jpg',
            image_url: placeholder('your upload', 120),
            thumb_url: placeholder('your upload', 120),
            source: 'manual_upload',
            license: 'User Contributed (Personal)',
            author: 'Current User',
            viewpoint: 'unknown',
            is_approximate: false,
          }
          return send(res, 200, { success: true, candidate })
        }
        if (path === '/api/reconstruct' && req.method === 'POST') {
          if (!has('file') && !has('image_url')) return send(res, 400, { detail: "Either a 'file' upload or an 'image_url' must be supplied." })
          const id = `job-${Date.now()}`
          jobs.set(id, Date.now())
          return send(res, 202, { job_id: id, status: 'queued', queue_position: 2 })
        }
        const job = path.match(/^\/api\/jobs\/(.+)$/)
        if (job) {
          if (!jobs.has(job[1])) return send(res, 404, { detail: `Job '${job[1]}' not found.` })
          return send(res, 200, jobBody(job[1]), 100)
        }
        const asset = path.match(/^\/api\/assets\/([^/]+)\/(.+)$/)
        if (asset) {
          if (asset[2] === 'mesh.glb') {
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
