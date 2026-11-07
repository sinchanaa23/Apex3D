// Translates Person C's API (docs/API_CONTRACT.md, backend/models.py) into the shapes in
// types.ts that the pages use. If the backend's formats change, this is the file to update.
import { humanise } from '../lib/format'
import type {
  ClassicalResponse,
  Job,
  JobStatus,
  MaybeSection,
  MetricsResponse,
  OptimisationMetrics,
  RecognizeResponse,
  Reference,
  ReferencesResponse,
  Viewpoint,
} from './types'

export interface WireBox {
  x1: number
  y1: number
  x2: number
  y2: number
  confidence: number
  label: string
}

export interface WireRecognition {
  car_detected: boolean
  bounding_box?: WireBox | null
  predictions: { class_id: number; class_name: string; probability: number; make?: string; model?: string; year?: number; body_style?: string }[]
  confidence_state: RecognizeResponse['confidence']['state']
  confidence_score: number
  margin: number
  message?: string | null
  timings_ms: Record<string, number>
}

/** The backend doesn't return the image size, but the box is in pixels of the image we sent. */
export function adaptRecognition(wire: WireRecognition, size: { width: number; height: number }): RecognizeResponse {
  const box = wire.bounding_box
  return {
    image_size: size,
    detection: wire.car_detected && box ? { box: [box.x1, box.y1, box.x2, box.y2], label: box.label, score: box.confidence } : null,
    top5: wire.predictions.map((p) => ({ class_id: p.class_id, label: p.class_name, probability: p.probability, make: p.make, model: p.model, year: p.year, body_style: p.body_style })),
    confidence: {
      state: wire.confidence_state,
      top1: wire.confidence_score,
      margin: wire.margin,
      // The confident badge already says this, and the state's wording shouldn't add a claim of its own.
      reason: wire.confidence_state === 'confident' ? null : (wire.message ?? null),
    },
    timings_ms: wire.timings_ms,
  }
}

const STAGE_TITLES: Record<string, string> = {
  original: 'Original',
  clahe: 'CLAHE',
  edges: 'Canny edges',
  contours: 'Contours',
  largest_contour_roi: 'Largest contour ROI',
  hog_visualization: 'HOG',
}

const FEATURE_PREFIX: Record<string, string> = { geometric: '', hog: 'HOG_', lbp: 'LBP_', color_hsv: 'HSV_' }

export interface WireClassical {
  intermediate_images: Record<string, string>
  features: Record<string, Record<string, number | string | number[]>>
  timings_ms?: Record<string, number>
}

/** Person B's backend/classical_cv/api.py returns the page's own format, so it passes straight through. */
interface NativeClassical {
  steps: { key?: unknown; title?: unknown; image?: unknown; caption?: unknown }[]
  features?: Record<string, number | string | number[]>
  timings_ms?: Record<string, number>
}

export function adaptClassical(wire: WireClassical | NativeClassical): ClassicalResponse {
  if ('steps' in wire && Array.isArray(wire.steps)) {
    return {
      steps: wire.steps.flatMap((step) =>
        typeof step.key === 'string' && typeof step.title === 'string' && typeof step.image === 'string'
          ? [{ key: step.key, title: step.title, image: step.image, caption: typeof step.caption === 'string' ? step.caption : null }]
          : [],
      ),
      features: wire.features ?? {},
      timings_ms: wire.timings_ms,
    }
  }
  wire = wire as WireClassical
  const features: ClassicalResponse['features'] = {}
  for (const [group, values] of Object.entries(wire.features)) {
    const prefix = FEATURE_PREFIX[group] ?? `${group}_`
    for (const [name, value] of Object.entries(values)) features[`${prefix}${name}`] = value
  }
  return {
    steps: Object.entries(wire.intermediate_images).map(([key, image]) => ({ key, image, title: STAGE_TITLES[key] ?? humanise(key) })),
    features,
    timings_ms: wire.timings_ms,
  }
}

export interface WireReference {
  id: string
  title?: string | null
  image_url: string
  thumb_url?: string | null
  source: string
  license?: string | null
  license_url?: string | null
  author?: string | null
  viewpoint?: string | null
  is_approximate?: boolean
}

const VIEWPOINTS = ['front', 'front_three_quarter', 'side', 'rear', 'rear_three_quarter']
const SOURCE_NAMES: Record<string, string> = {
  wikimedia_commons: 'Wikimedia Commons',
  manual_upload: 'Your upload',
  local_stanford: 'Stanford Cars (local only)',
}

export function adaptReference(wire: WireReference): Reference {
  const isUser = wire.source === 'manual_upload'
  const isCommonsFile = wire.source === 'wikimedia_commons' && wire.title?.startsWith('File:')
  return {
    id: wire.id,
    image_url: wire.image_url,
    thumb_url: wire.thumb_url,
    source: SOURCE_NAMES[wire.source] ?? wire.source,
    source_url: isCommonsFile ? `https://commons.wikimedia.org/wiki/${encodeURIComponent(wire.title!.replace(/ /g, '_'))}` : null,
    author: isUser ? null : wire.author,
    licence: isUser ? null : wire.license,
    licence_url: isUser ? null : wire.license_url,
    viewpoint: VIEWPOINTS.includes(wire.viewpoint ?? '') ? (wire.viewpoint as Viewpoint) : null,
    match: isUser ? 'user' : wire.is_approximate ? 'approximate' : 'confirmed',
  }
}

export function adaptReferences(wire: { query?: string | null; candidates: WireReference[] }): ReferencesResponse {
  return { query: wire.query, candidates: wire.candidates.map(adaptReference) }
}

export interface WireJob {
  job_id: string
  status: string
  stage?: string | null
  queue_position?: number | null
  started_at?: string | null
  completed_at?: string | null
  logs?: string[]
  error?: string | null
  mesh_source?: string | null
  class_name?: string | null
  assets?: Record<string, string> | null
  validation?: {
    is_watertight?: boolean
    is_manifold?: boolean
    vertex_count?: number
    face_count?: number
    degenerate_faces?: number
    connected_components?: number
    has_vertex_colors?: boolean
    has_uv_texture?: boolean
    asset_readiness?: string
    notes?: string[]
  } | null
}

const FILE_ORDER = ['glb', 'obj', 'stl', 'json']

export function adaptJob(wire: WireJob): Job {
  const status: JobStatus = wire.status === 'completed' ? 'succeeded' : (wire.status as JobStatus)
  const files = Object.values(wire.assets ?? {})
    .map((url) => {
      const name = url.split('/').pop() ?? url
      return { name, format: name.split('.').pop()?.toLowerCase() ?? '' }
    })
    .sort((a, b) => FILE_ORDER.indexOf(a.format) - FILE_ORDER.indexOf(b.format))
  const started = wire.started_at ? Date.parse(wire.started_at) : NaN
  const finished = wire.completed_at ? Date.parse(wire.completed_at) : NaN
  const report = wire.validation

  return {
    id: wire.job_id,
    status,
    stage: wire.stage ? humanise(wire.stage).replace(/triposr/i, 'TripoSR') : null,
    // The backend sends 0 for "not waiting". The app counts from 1.
    queue_position: wire.queue_position ? wire.queue_position : null,
    logs: wire.logs,
    error: wire.error,
    mesh_source: wire.mesh_source === 'catalog' || wire.mesh_source === 'reference' || wire.mesh_source === 'photo' ? wire.mesh_source : null,
    class_name: wire.class_name ?? null,
    result:
      status === 'succeeded' && files.length
        ? {
            asset_id: wire.job_id,
            files,
            generation_seconds: finished > started ? (finished - started) / 1000 : null,
            validation: report
              ? {
                  watertight: report.is_watertight,
                  manifold: report.is_manifold,
                  vertices: report.vertex_count,
                  faces: report.face_count,
                  degenerate_faces: report.degenerate_faces,
                  connected_components: report.connected_components,
                  has_vertex_colors: report.has_vertex_colors,
                  has_texture: report.has_uv_texture,
                  readiness: report.asset_readiness,
                  notes: report.notes,
                }
              : null,
          }
        : null,
  }
}

export interface WireMetrics {
  status: string
  message?: string | null
  source_file?: string | null
  metrics?: Record<string, unknown> | null
}

const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const orNull = (value: unknown) => (isNumber(value) ? value : null)
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null
const rec = (value: unknown) => (isRecord(value) ? value : null)
const text = (value: unknown) => (typeof value === 'string' && value ? value : null)

/** Finds a document by shape, either at the top level or under one of a few expected names. */
function pickDoc(m: Record<string, unknown>, test: (d: Record<string, unknown>) => boolean, keys: string[]) {
  if (test(m)) return m
  for (const key of keys) {
    const d = rec(m[key])
    if (d && test(d)) return d
  }
  return null
}
/** Person A's classifier notebook saves a document with a nested `test: {top1, top5, ...}` block. */
const isClassifierDoc = (d: Record<string, unknown>) => isNumber(rec(d.test)?.top1) && isNumber(rec(d.test)?.top5)
/** Person A's export notebook saves `variants: {name: {size_mb, test_top1, ...}}`. */
const isOptimisationDoc = (d: Record<string, unknown>) => rec(d.variants) != null

const VARIANT_NAMES: Record<string, string> = {
  pytorch_fp32: 'PyTorch FP32',
  onnx_fp32: 'ONNX FP32',
  onnx_fp16: 'ONNX FP16',
  onnx_int8_dynamic: 'ONNX INT8, dynamic',
  onnx_int8_static_all: 'ONNX INT8, static (all layers)',
  onnx_int8_static_matmul: 'ONNX INT8, static (MatMul only)',
}
const variantName = (key: string) =>
  VARIANT_NAMES[key] ?? humanise(key).replace(/onnx/i, 'ONNX').replace(/int8/i, 'INT8').replace(/fp(\d+)/i, 'FP$1')

function optimisationFromDoc(doc: Record<string, unknown>) {
  const variants = rec(doc.variants) ?? {}
  const rows = Object.entries(variants).flatMap(([key, value]) => {
    const v = rec(value)
    if (!v || !isNumber(v.size_mb)) return []
    return [
      {
        variant: variantName(key),
        size_mb: v.size_mb,
        cpu_ms: orNull(v.cpu_latency_ms_b1),
        gpu_ms: null,
        top1: orNull(v.test_top1),
        top5: orNull(v.test_top5),
        agreement: orNull(v.agreement_with_pytorch_fp32),
        note: doc.deploy_variant === key ? 'Deployed' : null,
      },
    ]
  })
  return rows.length ? { rows } : null
}

function classifierFromDoc(doc: Record<string, unknown>) {
  const test = rec(doc.test)!
  const validation = rec(doc.validation)
  return {
    model: text(rec(doc.model)?.arch),
    top1: test.top1 as number,
    top5: test.top5 as number,
    test_images: orNull(test.n),
    make_top1: orNull(test.make_top1),
    macro_f1: orNull(test.macro_f1),
    val_top1: orNull(validation?.top1),
    val_top5: orNull(validation?.top5),
    top1_detector_crop: null,
    top5_detector_crop: null,
  }
}

function statesFromDoc(doc: Record<string, unknown>) {
  const states = rec(rec(doc.test)?.states)
  if (!states) return null
  const rows = (['confident', 'ambiguous', 'rejected'] as const).flatMap((state) => {
    const s = rec(states[state])
    return s && isNumber(s.share) ? [{ state, share: s.share, accuracy: orNull(s.accuracy) }] : []
  })
  if (!rows.length) return null
  const t = rec(doc.thresholds)
  const rule =
    t && isNumber(t.confident_p1) && isNumber(t.confident_margin)
      ? `Confident when the top probability is at least ${t.confident_p1} and it leads the runner-up by at least ${t.confident_margin}.` +
        (isNumber(t.reject_below_p1) ? ` Rejected when the top probability is below ${t.reject_below_p1}.` : '') +
        ' The thresholds were chosen on the validation set, and these shares and accuracies are measured on the test set.'
      : null
  return { rows, rule }
}

/** `measured` is the deployed model's own test-set calibration (results/confidence.json), when the metrics file has it. */
/**
 * The calibration table comes from Person A's saved file, which has no test-set "before" values. The backend's metrics
 * carry the deployed model's own test ECE and NLL, so those rows are replaced with them when they are there.
 */
export function withMeasuredCalibration(calibration: unknown, confidence: unknown): unknown {
  const cal = rec(calibration)
  const data = rec(cal?.data)
  const measured = rec(rec(rec(confidence)?.data)?.test_calibration)
  if (!cal || !data || !Array.isArray(data.rows) || !measured) return calibration
  const rows = (data.rows as { metric: string; before: number | null; after: number | null }[]).map((row) =>
    row.metric === 'Test ECE'
      ? { ...row, before: orNull(measured.ece_before) ?? row.before, after: orNull(measured.ece_after) ?? row.after }
      : row.metric === 'Test NLL'
        ? { ...row, before: orNull(measured.nll_before) ?? row.before, after: orNull(measured.nll_after) ?? row.after }
        : row,
  )
  return { ...cal, data: { ...data, rows } }
}

export function calibrationFromDoc(doc: Record<string, unknown>, measured?: Record<string, unknown> | null) {
  const c = rec(doc.calibration)
  const test = rec(doc.test)
  if (!c || !isNumber(c.temperature)) return null
  const m = measured ?? {}
  const rows = [
    { metric: 'Validation ECE', before: orNull(c.val_ece_before), after: orNull(c.val_ece_after) },
    { metric: 'Validation NLL', before: orNull(c.val_nll_before), after: orNull(c.val_nll_after) },
    { metric: 'Test ECE', before: orNull(m.ece_before), after: orNull(m.ece_after ?? test?.ece_calibrated) },
    { metric: 'Test NLL', before: orNull(m.nll_before), after: orNull(m.nll_after ?? test?.nll_calibrated) },
  ].filter((row) => row.before != null || row.after != null)
  return { temperature: c.temperature, rows }
}

function runFromDoc(doc: Record<string, unknown>) {
  const model = rec(doc.model)
  const dataset = rec(doc.dataset)
  const training = rec(doc.training)
  const env = rec(doc.environment)
  const rows: { label: string; value: string }[] = []
  const add = (label: string, value: string | null) => value && rows.push({ label, value })
  add('Model', text(model?.arch))
  add('Parameters', isNumber(model?.params_m) ? `${model.params_m.toFixed(1)} M` : null)
  add('Checkpoint size', isNumber(model?.checkpoint_mb) ? `${model.checkpoint_mb.toFixed(0)} MB` : null)
  add('Input', text(dataset?.input))
  add(
    'Data split',
    isNumber(dataset?.train) && isNumber(dataset?.val) && isNumber(dataset?.test)
      ? `${dataset.train.toLocaleString()} train, ${dataset.val.toLocaleString()} validation, ${dataset.test.toLocaleString()} test`
      : null,
  )
  add('Data source', text(dataset?.source))
  add(
    'Training',
    isNumber(training?.epochs_run) ? `${training.epochs_run} epochs` + (isNumber(training?.best_epoch) ? `, best at epoch ${training.best_epoch}` : '') : null,
  )
  add('Precision', text(training?.amp))
  add('Trained on', text(env?.gpu) ? `${env!.gpu}${isNumber(env?.gpu_vram_gib) ? `, ${env.gpu_vram_gib} GB` : ''}` : null)
  return rows.length ? { rows } : null
}

/** Reads `{noise_gaussian_top1: 0.8}` as rows, skipping anything that isn't a number. */
function top1Entries(value: unknown) {
  if (!isRecord(value)) return []
  return Object.entries(value)
    .filter((entry): entry is [string, number] => isNumber(entry[1]))
    .map(([key, top1]) => ({ name: key.replace(/_top1$/, ''), top1 }))
}

/** A notebook may also save the fuller `{rows: [...]}` form, which the charts can use. */
const hasRows = (value: unknown) => isRecord(value) && Array.isArray(value.rows)

/** A section that is already `{source, generated_at, data}`, as Person B's eval_P5_metrics.json writes them. */
const isNativeSection = (value: unknown): value is { source: string; generated_at?: string | null; data: Record<string, unknown> } =>
  isRecord(value) && typeof value.source === 'string' && isRecord(value.data)

export function adaptMetrics(wire: WireMetrics): MetricsResponse {
  const m = wire.metrics
  if (wire.status !== 'available' || !isRecord(m)) return { message: wire.message }

  const source = wire.source_file ?? 'metrics file'
  const generated_at = typeof m.generated_at === 'string' ? m.generated_at : null
  const section = <T,>(data: T | null): MaybeSection<T> => (data == null ? 'not available' : { source, generated_at, data })
  /** A section from one of Person A's documents names that notebook as its source. */
  const sectionFrom = <T,>(doc: Record<string, unknown> | null, data: T | null): MaybeSection<T> =>
    data == null || !doc ? 'not available' : { source: text(doc.generated_by) ?? source, generated_at: text(doc.generated_at) ?? generated_at, data }
  const passThrough = <T,>(value: unknown): MaybeSection<T> => (isRecord(value) ? section(value as T) : 'not available')

  const robustness = top1Entries(m.robustness)
  const comparison = top1Entries(m.comparison)
  const classifierDoc = pickDoc(m, isClassifierDoc, ['classifier_metrics', 'classifier'])
  const optimisationDoc = pickDoc(m, isOptimisationDoc, ['optimization_metrics', 'optimisation_metrics', 'optimization', 'optimisation'])

  const flatClassifier =
    isNumber(m.test_top1_acc) && isNumber(m.test_top5_acc)
      ? {
          model: typeof m.model_name === 'string' ? m.model_name : null,
          top1: m.test_top1_acc,
          top5: m.test_top5_acc,
          top1_detector_crop: orNull(m.test_top1_acc_detector_crop),
          top5_detector_crop: orNull(m.test_top5_acc_detector_crop),
          test_images: orNull(m.test_images),
        }
      : null
  const flatOptimisation = isNumber(m.model_size_mb)
    ? {
        rows: [
          { variant: 'Original', size_mb: m.model_size_mb, cpu_ms: orNull(m.inference_time_cpu_ms), gpu_ms: orNull(m.inference_time_gpu_ms) },
          ...(isNumber(m.quantized_size_mb)
            ? [{ variant: 'Quantized', size_mb: m.quantized_size_mb, cpu_ms: orNull(m.inference_time_quantized_cpu_ms) }]
            : []),
        ],
      }
    : null

  /** A native section from the file wins. Otherwise use what was worked out from the other formats. */
  const prefer = <T,>(key: string, computed: MaybeSection<T>): MaybeSection<T> => (isNativeSection(m[key]) ? (m[key] as unknown as MaybeSection<T>) : computed)
  /** Gives optimisation rows readable names when the file uses the notebook's keys, such as onnx_int8_dynamic. */
  const nativeOptimisation = (): MaybeSection<OptimisationMetrics> => {
    const native = m.optimisation
    if (!isNativeSection(native) || !Array.isArray(native.data.rows)) return 'not available'
    const rows = (native.data.rows as { variant?: unknown }[]).map((row) => (typeof row.variant === 'string' && VARIANT_NAMES[row.variant] ? { ...row, variant: VARIANT_NAMES[row.variant] } : row))
    return { ...native, data: { ...native.data, rows } } as unknown as MaybeSection<OptimisationMetrics>
  }

  return {
    classifier: prefer('classifier', flatClassifier ? section(flatClassifier) : classifierDoc ? sectionFrom(classifierDoc, classifierFromDoc(classifierDoc)) : 'not available'),
    detection: prefer(
      'detection',
      section(
        isNumber(m.detector_mean_iou) && isNumber(m.detector_mAP50)
          ? { rows: [{ model: typeof m.detector_name === 'string' ? m.detector_name : 'Detector', mean_iou: m.detector_mean_iou, map: m.detector_mAP50 }] }
          : null,
      ),
    ),
    optimisation: isNativeSection(m.optimisation)
      ? nativeOptimisation()
      : flatOptimisation
        ? section(flatOptimisation)
        : optimisationDoc
          ? sectionFrom(optimisationDoc, optimisationFromDoc(optimisationDoc))
          : 'not available',
    run: sectionFrom(classifierDoc, classifierDoc ? runFromDoc(classifierDoc) : null),
    calibration: sectionFrom(classifierDoc, classifierDoc ? calibrationFromDoc(classifierDoc, rec(rec(rec(m.confidence)?.data)?.test_calibration)) : null),
    states: sectionFrom(classifierDoc, classifierDoc ? statesFromDoc(classifierDoc) : null),
    out_of_taxonomy: prefer('out_of_taxonomy', 'not available'),
    robustness: prefer(
      'robustness',
      hasRows(m.robustness)
        ? passThrough(m.robustness)
        : section(robustness.length ? { rows: robustness.map(({ name, top1 }) => ({ perturbation: name, severity: null, top1 })) } : null),
    ),
    comparison: prefer(
      'comparison',
      hasRows(m.comparison)
        ? passThrough(m.comparison)
        : section(comparison.length ? { rows: comparison.map(({ name, top1 }) => ({ approach: humanise(name), top1 })) } : null),
    ),
    confidence: prefer('confidence', passThrough(m.confidence)),
    ablation: prefer('ablation', passThrough(m.ablation)),
    confused_pairs: prefer('confused_pairs', passThrough(m.confused_pairs)),
    failure_cases: prefer('failure_cases', passThrough(m.failure_cases)),
  }
}
