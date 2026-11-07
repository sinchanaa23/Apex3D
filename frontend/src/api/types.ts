// The shapes the pages use. adapter.ts translates Person C's API into these, so a backend
// format change only touches that file. All probabilities and rates are fractions from 0 to 1.

export type ConfidenceState = 'confident' | 'ambiguous' | 'rejected'

export interface ClassPrediction {
  class_id: number
  /** Stanford Cars label, e.g. "Audi TT Hatchback 2011". */
  label: string
  probability: number
  /** Read from the backend. When absent, the page reads them from the label. */
  make?: string | null
  model?: string | null
  year?: number | null
  body_style?: string | null
}

export interface RecognizeResponse {
  /** Size of the image the box coordinates refer to. */
  image_size: { width: number; height: number }
  /** Null when no vehicle was found and the full frame was classified instead. */
  detection: { box: [number, number, number, number]; label: string; score: number } | null
  top5: ClassPrediction[]
  confidence: { state: ConfidenceState; top1: number; margin: number; reason?: string | null }
  timings_ms: Record<string, number>
}

export interface ClassicalStep {
  key: string
  title: string
  /** Data URL or /api path, usable directly as an img src. */
  image: string
  caption?: string | null
}

export interface ClassicalResponse {
  steps: ClassicalStep[]
  /** Scalars such as aspect ratio, or arrays such as an HSV histogram. */
  features: Record<string, number | string | number[]>
  timings_ms?: Record<string, number>
}

export type Viewpoint = 'front' | 'front_three_quarter' | 'side' | 'rear' | 'rear_three_quarter'

export interface Reference {
  id: string
  image_url: string
  thumb_url?: string | null
  source: string
  source_url?: string | null
  author?: string | null
  licence?: string | null
  licence_url?: string | null
  viewpoint?: Viewpoint | null
  /** "confirmed": the backend matched it to the model. Anything else is approximate. */
  match: 'confirmed' | 'approximate' | 'user'
}

export interface ReferencesResponse {
  query?: string | null
  candidates: Reference[]
}

export interface AssetFile {
  name: string
  /** glb, obj, stl, json (validation report), or anything else the backend adds. */
  format: string
  size_bytes?: number | null
}

export interface JobResult {
  asset_id: string
  files: AssetFile[]
  generation_seconds?: number | null
  validation?: ValidationReport | null
}

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export interface Job {
  id: string
  status: JobStatus
  stage?: string | null
  /** 1 means next. Null or absent when not queued. */
  queue_position?: number | null
  logs?: string[]
  error?: string | null
  result?: JobResult | null
  /** What the mesh was made from: a licensed catalog model, a reference photo of the matched car, or the user's own photo. */
  mesh_source?: 'catalog' | 'reference' | 'photo' | null
  /** The matched car's name, when the job was started with a confirmed class. */
  class_name?: string | null
}

export interface ValidationReport {
  watertight?: boolean | null
  manifold?: boolean | null
  readiness?: string | null
  notes?: string[] | null
  degenerate_faces?: number | null
  connected_components?: number | null
  vertices?: number | null
  faces?: number | null
  has_texture?: boolean | null
  has_vertex_colors?: boolean | null
}

/** One saved notebook file. The API sends the string "not available" when the file is absent. */
export interface MetricSection<T> {
  /** Path of the JSON file these numbers were read from. */
  source: string
  generated_at?: string | null
  data: T
}
export type MaybeSection<T> = MetricSection<T> | 'not available' | null | undefined

export interface ClassifierMetrics {
  model?: string | null
  top1: number
  top5: number
  top1_detector_crop?: number | null
  top5_detector_crop?: number | null
  test_images?: number | null
  make_top1?: number | null
  macro_f1?: number | null
  val_top1?: number | null
  val_top5?: number | null
}
export interface DetectionMetrics {
  rows: { model: string; mean_iou: number; map: number; cpu_ms?: number | null }[]
}
export interface ComparisonMetrics {
  rows: { approach: string; top1: number; top5?: number | null; inference_ms?: number | null; size_mb?: number | null }[]
}
export interface AblationMetrics {
  rows: { removed: string; top1: number; note?: string | null }[]
  baseline_top1?: number | null
}
export interface OptimisationMetrics {
  rows: {
    variant: string
    size_mb: number
    cpu_ms?: number | null
    gpu_ms?: number | null
    top1?: number | null
    top5?: number | null
    /** How often this variant's answer matches the original model's. */
    agreement?: number | null
    note?: string | null
  }[]
}
export interface CalibrationMetrics {
  temperature: number
  rows: { metric: string; before: number | null; after: number | null }[]
}
export interface StateMetrics {
  rows: { state: ConfidenceState; share: number; accuracy: number | null }[]
  rule?: string | null
}
export interface OutOfTaxonomyMetrics {
  /** What happened to photos of things that are not one of the 196 cars. */
  rows: { group: string; images: number; rejected: number; ambiguous: number; confident: number }[]
  /** How well the top probability alone separates in-class photos from outside ones. */
  auroc?: number | null
  limits?: string[] | null
}
export interface RunInfo {
  rows: { label: string; value: string }[]
}
export interface RobustnessMetrics {
  /** Severity is null when the file gives one result per perturbation. */
  rows: { perturbation: string; severity: number | null; top1: number }[]
}
export interface ConfidenceMetrics {
  rows: { threshold: number; coverage: number; accuracy: number }[]
  chosen_threshold?: number | null
}
export interface ConfusedPairsMetrics {
  rows: { a: string; b: string; count: number }[]
}
export interface FailureCasesMetrics {
  cases: { title: string; image?: string | null; predicted?: string | null; actual?: string | null; explanation: string }[]
}

export interface MetricsResponse {
  /** The backend's explanation when no metrics file exists yet. */
  message?: string | null
  classifier?: MaybeSection<ClassifierMetrics>
  detection?: MaybeSection<DetectionMetrics>
  comparison?: MaybeSection<ComparisonMetrics>
  optimisation?: MaybeSection<OptimisationMetrics>
  robustness?: MaybeSection<RobustnessMetrics>
  confidence?: MaybeSection<ConfidenceMetrics>
  ablation?: MaybeSection<AblationMetrics>
  confused_pairs?: MaybeSection<ConfusedPairsMetrics>
  failure_cases?: MaybeSection<FailureCasesMetrics>
  calibration?: MaybeSection<CalibrationMetrics>
  states?: MaybeSection<StateMetrics>
  out_of_taxonomy?: MaybeSection<OutOfTaxonomyMetrics>
  run?: MaybeSection<RunInfo>
}
