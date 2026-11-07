// Inputs follow the examples in Person C's docs/API_CONTRACT.md.
import { describe, expect, it } from 'vitest'
import { adaptClassical, adaptJob, adaptMetrics, adaptRecognition, adaptReference, calibrationFromDoc } from './adapter'

const box = { x1: 42, y1: 115, x2: 612, y2: 450, confidence: 0.94, label: 'car' }
const prediction = { class_id: 0, class_name: 'AM General Hummer SUV 2000', probability: 0.88 }
const recognition = {
  car_detected: true,
  bounding_box: box,
  predictions: [prediction],
  confidence_state: 'confident' as const,
  confidence_score: 0.88,
  margin: 0.82,
  message: 'High-confidence vehicle identification.',
  timings_ms: { total: 81 },
}

describe('adaptRecognition', () => {
  const size = { width: 800, height: 600 }

  it('maps the box and predictions, keeping class id 0', () => {
    const result = adaptRecognition(recognition, size)
    expect(result.detection).toEqual({ box: [42, 115, 612, 450], label: 'car', score: 0.94 })
    expect(result.top5[0]).toEqual({ class_id: 0, label: prediction.class_name, probability: 0.88 })
    expect(result.image_size).toEqual(size)
  })
  it('reports no detection when no car was found', () => {
    expect(adaptRecognition({ ...recognition, car_detected: false, bounding_box: null }, size).detection).toBeNull()
  })
  it('only passes the message through for uncertain results', () => {
    expect(adaptRecognition(recognition, size).confidence.reason).toBeNull()
    expect(adaptRecognition({ ...recognition, confidence_state: 'rejected', message: 'Low confidence' }, size).confidence.reason).toBe('Low confidence')
  })
})

describe('adaptClassical', () => {
  it('turns images into ordered steps and flattens feature groups', () => {
    const result = adaptClassical({
      intermediate_images: { clahe: 'data:a', some_new_stage: 'data:b' },
      features: { geometric: { aspect_ratio: 2.1 }, hog: { vector_length: 3780 } },
    })
    expect(result.steps.map((s) => s.title)).toEqual(['CLAHE', 'Some new stage'])
    expect(result.features).toEqual({ aspect_ratio: 2.1, HOG_vector_length: 3780 })
  })
})

describe('adaptReference', () => {
  const wikimedia = {
    id: 'wm_1',
    title: 'File:Audi A4 front.jpg',
    image_url: 'https://upload.wikimedia.org/a.jpg',
    source: 'wikimedia_commons',
    license: 'CC BY-SA 4.0',
    license_url: 'https://creativecommons.org/x',
    author: 'Name',
    viewpoint: 'side',
    is_approximate: false,
  }

  it('keeps attribution and links back to the Commons page', () => {
    const result = adaptReference(wikimedia)
    expect(result.licence).toBe('CC BY-SA 4.0')
    expect(result.author).toBe('Name')
    expect(result.source).toBe('Wikimedia Commons')
    expect(result.source_url).toBe('https://commons.wikimedia.org/wiki/File%3AAudi_A4_front.jpg')
  })
  it('marks approximate matches and unknown viewpoints', () => {
    const result = adaptReference({ ...wikimedia, is_approximate: true, viewpoint: 'unknown' })
    expect(result.match).toBe('approximate')
    expect(result.viewpoint).toBeNull()
  })
  it('treats manual uploads as the user\'s own', () => {
    const result = adaptReference({ ...wikimedia, source: 'manual_upload', license: 'User Contributed (Personal)', author: 'Current User' })
    expect(result).toMatchObject({ match: 'user', source: 'Your upload', licence: null, author: null })
  })
})

describe('adaptJob', () => {
  it('reports a queued job from position 1 and a running one with no position', () => {
    expect(adaptJob({ job_id: 'a', status: 'queued', queue_position: 1 }).queue_position).toBe(1)
    const running = adaptJob({ job_id: 'a', status: 'running', queue_position: 0, stage: 'triposr_inference' })
    expect(running.queue_position).toBeNull()
    expect(running.stage).toBe('TripoSR inference')
  })
  it('maps a completed job to files, time and validation', () => {
    const job = adaptJob({
      job_id: 'a',
      status: 'completed',
      started_at: '2026-11-06T12:00:02Z',
      completed_at: '2026-11-06T12:00:26Z',
      assets: {
        validation_report_url: '/api/assets/a/validation.json',
        glb_url: '/api/assets/a/mesh.glb',
        stl_url: '/api/assets/a/mesh.stl',
      },
      validation: { is_watertight: true, vertex_count: 10, has_uv_texture: false, asset_readiness: 'draft', notes: ['Note'] },
    })
    expect(job.status).toBe('succeeded')
    expect(job.result?.files.map((f) => f.format)).toEqual(['glb', 'stl', 'json'])
    expect(job.result?.generation_seconds).toBe(24)
    expect(job.result?.validation).toMatchObject({ watertight: true, vertices: 10, has_texture: false, readiness: 'draft', notes: ['Note'] })
  })
  it('passes failures through without a result', () => {
    const job = adaptJob({ job_id: 'a', status: 'failed', error: 'Out of memory' })
    expect(job).toMatchObject({ status: 'failed', error: 'Out of memory', result: null })
  })
})

describe('adaptMetrics', () => {
  it('carries the message when no file exists, with no sections', () => {
    expect(adaptMetrics({ status: 'not available', message: 'Not trained', metrics: null })).toEqual({ message: 'Not trained' })
  })
  it('maps the contract\'s flat file and names its source on every section', () => {
    const result = adaptMetrics({
      status: 'available',
      source_file: 'metrics.json',
      metrics: {
        test_top1_acc: 0.5,
        test_top5_acc: 0.7,
        detector_mAP50: 0.4,
        detector_mean_iou: 0.6,
        model_size_mb: 100,
        quantized_size_mb: 25,
        inference_time_quantized_cpu_ms: 18,
        robustness: { blur_motion_top1: 0.3 },
        comparison: { classical_svm_top1: 0.2 },
      },
    })
    expect(result.classifier).toMatchObject({ source: 'metrics.json', data: { top1: 0.5, top5: 0.7 } })
    expect(result.detection).toMatchObject({ data: { rows: [{ mean_iou: 0.6, map: 0.4 }] } })
    expect(result.optimisation).toMatchObject({ data: { rows: [{ size_mb: 100 }, { variant: 'Quantized', size_mb: 25, cpu_ms: 18 }] } })
    expect(result.robustness).toMatchObject({ data: { rows: [{ perturbation: 'blur_motion', severity: null, top1: 0.3 }] } })
    expect(result.comparison).toMatchObject({ data: { rows: [{ approach: 'Classical svm', top1: 0.2 }] } })
  })
  it('never invents a value: missing or non-numeric results are not available', () => {
    const result = adaptMetrics({ status: 'available', metrics: { test_top1_acc: 'high', robustness: { blur_top1: 'low' } } })
    expect(result.classifier).toBe('not available')
    expect(result.robustness).toBe('not available')
    expect(result.ablation).toBe('not available')
  })
  it('passes richer notebook sections through', () => {
    const result = adaptMetrics({ status: 'available', metrics: { ablation: { baseline_top1: 0.5, rows: [] }, robustness: { rows: [{ perturbation: 'blur', severity: 1, top1: 0.4 }] } } })
    expect(result.ablation).toMatchObject({ data: { baseline_top1: 0.5 } })
    expect(result.robustness).toMatchObject({ data: { rows: [{ severity: 1 }] } })
  })
})

// The two files below are Person A's real saved outputs (results/*.json on the person-a branch), so these tests check the real numbers.
import classifierDoc from './fixtures/person_a_classifier_metrics.json'
import optimisationDoc from './fixtures/person_a_optimization_metrics.json'

const available = <T>(section: unknown) => {
  expect(section).not.toBe('not available')
  return section as { source: string; generated_at: string | null; data: T }
}
const wire = (metrics: unknown) => ({ status: 'available', source_file: 'metrics.json', metrics: metrics as Record<string, unknown> })

describe("adaptMetrics with Person A's classifier file", () => {
  const result = adaptMetrics(wire(classifierDoc))

  it('reads the nested test block and names the notebook as the source', () => {
    const section = available<{ top1: number; top5: number; test_images: number; make_top1: number; macro_f1: number; val_top1: number; model: string }>(result.classifier)
    expect(section.data.top1).toBeCloseTo(0.92202, 4)
    expect(section.data.top5).toBeCloseTo(0.98421, 4)
    expect(section.data).toMatchObject({ test_images: 8041, model: 'convnext_tiny.fb_in22k_ft_in1k' })
    expect(section.data.make_top1).toBeCloseTo(0.97152, 4)
    expect(section.data.macro_f1).toBeCloseTo(0.92077, 4)
    expect(section.data.val_top1).toBeCloseTo(0.92226, 4)
    expect(section.source).toBe('notebooks/01_classifier_training.ipynb')
  })
  it('reads the confidence states and explains the thresholds', () => {
    const { rows, rule } = available<{ rows: { state: string; share: number; accuracy: number }[]; rule: string }>(result.states).data
    expect(rows.map((r) => r.state)).toEqual(['confident', 'ambiguous', 'rejected'])
    expect(rows.reduce((sum, r) => sum + r.share, 0)).toBeCloseTo(1, 3)
    expect(rows[0].accuracy).toBeCloseTo(0.9504, 3)
    expect(rule).toContain('0.52')
    expect(rule).toContain('0.34')
  })
  it('reads calibration, leaving out the values that were not measured', () => {
    const { temperature, rows } = available<{ temperature: number; rows: { metric: string; before: number | null; after: number | null }[] }>(result.calibration).data
    expect(temperature).toBeCloseTo(0.6166, 3)
    expect(rows.find((r) => r.metric === 'Validation ECE')).toMatchObject({ before: expect.closeTo(0.2075, 3), after: expect.closeTo(0.034, 3) })
    expect(rows.find((r) => r.metric === 'Test ECE')).toMatchObject({ before: null })
  })
  it('uses the deployed model\'s measured test calibration when the file has it', () => {
    const doc = { calibration: { temperature: 0.6, val_ece_before: 0.2, val_ece_after: 0.03 }, test: {} }
    const measured = { ece_before: 0.175, ece_after: 0.023, nll_before: 0.547, nll_after: 0.364 }
    const out = calibrationFromDoc(doc, measured)!
    expect(out.rows.find((r) => r.metric === 'Test ECE')).toMatchObject({ before: 0.175, after: 0.023 })
    expect(out.rows.find((r) => r.metric === 'Test NLL')).toMatchObject({ before: 0.547, after: 0.364 })
  })
  it('lists the training run from recorded fields only', () => {
    const rows = available<{ rows: { label: string; value: string }[] }>(result.run).data.rows
    const byLabel = Object.fromEntries(rows.map((r) => [r.label, r.value]))
    expect(byLabel['Training']).toBe('32 epochs, best at epoch 28')
    expect(byLabel['Data split']).toBe('6,922 train, 1,222 validation, 8,041 test')
    expect(byLabel['Trained on']).toContain('RTX 3050')
    expect(rows.every((r) => r.value.length > 0 && !/undefined|NaN|null/.test(r.value))).toBe(true)
  })
  it('does not invent sections the file does not contain', () => {
    expect(result.optimisation).toBe('not available')
    expect(result.robustness).toBe('not available')
    expect(result.comparison).toBe('not available')
    expect(result.detection).toBe('not available')
  })
})

describe("adaptMetrics with Person A's optimisation file", () => {
  const result = adaptMetrics(wire(optimisationDoc))

  it('turns every variant into a row, with readable names', () => {
    const { rows } = available<{ rows: { variant: string; size_mb: number; cpu_ms: number; top1: number; top5: number; agreement: number; note: string | null }[] }>(result.optimisation).data
    expect(rows.map((r) => r.variant)).toEqual([
      'PyTorch FP32',
      'ONNX FP32',
      'ONNX FP16',
      'ONNX INT8, dynamic',
      'ONNX INT8, static (all layers)',
      'ONNX INT8, static (MatMul only)',
    ])
    const dynamic = rows[3]
    expect(dynamic.cpu_ms).toBeCloseTo(151.34, 2) // slower than FP32, and the page must show that as it is
    expect(dynamic.size_mb).toBeCloseTo(28.36, 2)
  })
  it('marks only the variant the file says was deployed', () => {
    const { rows } = available<{ rows: { variant: string; note: string | null; top1: number; cpu_ms: number }[] }>(result.optimisation).data
    const deployed = rows.filter((r) => r.note === 'Deployed')
    expect(deployed).toHaveLength(1)
    expect(deployed[0]).toMatchObject({ variant: 'ONNX INT8, static (MatMul only)', cpu_ms: 19.4 })
    expect(deployed[0].top1).toBeCloseTo(0.92178, 4)
  })
  it('does not produce classifier sections', () => {
    expect(result.classifier).toBe('not available')
    expect(result.states).toBe('not available')
  })
})

describe('adaptMetrics with both of Person A files in one document', () => {
  it('finds each under its own name', () => {
    const result = adaptMetrics(wire({ classifier_metrics: classifierDoc, optimization_metrics: optimisationDoc }))
    expect(available(result.classifier)).toBeTruthy()
    expect(available(result.optimisation)).toBeTruthy()
    expect(available(result.states)).toBeTruthy()
    expect(available<{ rows: unknown[] }>(result.optimisation).data.rows).toHaveLength(6)
  })
  it('shows confused pairs added at the top level of the same file', () => {
    const pairs = { rows: [{ a: 'Audi V8 Sedan 1994', b: 'Audi 100 Sedan 1994', count: 20 }] }
    const result = adaptMetrics(wire({ classifier_metrics: classifierDoc, optimization_metrics: optimisationDoc, confused_pairs: pairs }))
    expect(available<typeof pairs>(result.confused_pairs).data.rows[0].count).toBe(20)
  })
  it('prefers the flat format when a file has both', () => {
    const result = adaptMetrics(wire({ test_top1_acc: 0.5, test_top5_acc: 0.7, ...classifierDoc }))
    expect(available<{ top1: number }>(result.classifier).data.top1).toBe(0.5)
  })
  it('refuses a test block that is not numeric', () => {
    expect(adaptMetrics(wire({ test: { top1: 'high', top5: 'higher' } })).classifier).toBe('not available')
  })
})

// Person B's eval_P5_metrics.json is the file the backend serves first. It is already in the page's own format
// (every section is {source, generated_at, data}). The images in the fixture are shortened.
import personBFile from './fixtures/person_b_app_metrics.json'

describe("adaptMetrics with Person B's eval_P5_metrics.json", () => {
  const result = adaptMetrics(wire(personBFile))

  it('uses every section the file has, as it is', () => {
    expect(available<{ top1: number }>(result.classifier).data.top1).toBeCloseTo(0.92202, 4)
    const detection = available<{ rows: { mean_iou: number; map: number; model: string }[] }>(result.detection).data.rows
    expect(detection).toHaveLength(2)
    expect(detection[0].mean_iou).toBeCloseTo(0.9314, 3)
    expect(available<{ rows: unknown[] }>(result.comparison).data.rows).toHaveLength(2)
    expect(available<{ rows: unknown[] }>(result.ablation).data.rows).toHaveLength(4)
    expect(available<{ rows: unknown[] }>(result.confused_pairs).data.rows).toHaveLength(15)
    expect(available<{ cases: unknown[] }>(result.failure_cases).data.cases).toHaveLength(5)
  })
  it('keeps the source file named on each section', () => {
    expect(available(result.comparison).source).toBe('results/classical_baseline.json')
    expect(available(result.detection).source).toContain('detector_resnet50')
  })
  it("gives the notebook's variant keys readable names", () => {
    const rows = available<{ rows: { variant: string }[] }>(result.optimisation).data.rows
    expect(rows[0].variant).toBe('PyTorch FP32')
    expect(rows.map((r) => r.variant)).not.toContain('pytorch_fp32')
  })
  it('keeps a section the file says is not available as not available', () => {
    expect(result.robustness).toBe('not available')
  })
})

describe("adaptClassical with Person B's format", () => {
  const native = {
    steps: [
      { key: 'original', title: 'Original', image: 'data:image/jpeg;base64,AAAA', caption: 'The photo, resized for analysis.' },
      { key: 'hog', title: 'HOG', image: 'data:image/jpeg;base64,BBBB', caption: 'Edge directions.' },
      { key: 'broken', title: 'No image here' },
    ],
    features: { aspect_ratio: 1.8, hu_moments: [0.2, 0.01], low_light: 'no', mean_brightness: 112.5 },
    timings_ms: { total: 88.1 },
  }

  it('passes the stages and features straight through', () => {
    const result = adaptClassical(native)
    expect(result.steps.map((s) => s.key)).toEqual(['original', 'hog'])
    expect(result.steps[0].caption).toBe('The photo, resized for analysis.')
    expect(result.features).toEqual(native.features)
    expect(result.timings_ms).toEqual({ total: 88.1 })
  })
  it("still reads Person C's format", () => {
    const result = adaptClassical({ intermediate_images: { clahe: 'data:a' }, features: { geometric: { aspect_ratio: 2 } } })
    expect(result.steps[0].title).toBe('CLAHE')
    expect(result.features).toEqual({ aspect_ratio: 2 })
  })
})

describe('adaptMetrics with an out-of-taxonomy section', () => {
  const section = {
    source: 'results/out_of_taxonomy.json',
    generated_at: '2026-11-07T01:00:00+00:00',
    data: {
      rows: [
        { group: 'Other vehicles', images: 600, rejected: 0.7, ambiguous: 0.2, confident: 0.1 },
        { group: 'Stanford Cars photos (in the 196 classes), uncropped', images: 1200, rejected: 0.05, ambiguous: 0.1, confident: 0.85 },
      ],
      auroc: 0.93,
      limits: ['Whole photos with no detector crop.'],
    },
  }

  it('passes the section through as it is', () => {
    const result = adaptMetrics(wire({ out_of_taxonomy: section }))
    const found = available<{ rows: { confident: number }[]; auroc: number }>(result.out_of_taxonomy)
    expect(found.data.rows).toHaveLength(2)
    expect(found.data.auroc).toBeCloseTo(0.93)
    expect(found.source).toBe('results/out_of_taxonomy.json')
  })
  it('says not available when the file has not been produced', () => {
    expect(adaptMetrics(wire({})).out_of_taxonomy).toBe('not available')
  })
})
