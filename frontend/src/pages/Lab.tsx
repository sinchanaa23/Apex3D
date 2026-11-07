import type { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { CountUp } from '../components/hud/CountUp'
import { getMetrics } from '../api/client'
import type { MaybeSection, MetricsResponse } from '../api/types'
import { ConfidenceBadge } from '../components/ConfidenceBadge'
import { ErrorBoundary } from '../components/ErrorBoundary'
import { LineChart } from '../components/LineChart'
import { ErrorNotice, PageHeader } from '../components/States'
import { humanise, milliseconds, percent } from '../lib/format'
import { useResource } from '../lib/useResource'

const SERIES_1 = '#3987e5'
const SERIES_2 = '#d95926'

/** Its own component so a file in the wrong format throws inside the section's error boundary, not above it. */
function SectionBody<T>({ render, data }: { render(data: T): ReactNode; data: T }) {
  return render(data)
}

function Section<T>({ title, why, value, children }: { title: string; why: string; value: MaybeSection<T>; children(data: T): ReactNode }) {
  const available = value != null && typeof value === 'object'
  return (
    <motion.section
      className="border-t border-line py-8"
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
    >
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 className="text-xl font-medium tracking-tight">{title}</h2>
        {available ? (
          <p className="font-mono text-xs text-ink-3">
            {value.source}
            {value.generated_at && ` · ${value.generated_at}`}
          </p>
        ) : (
          <p className="rounded-full border border-line-strong px-2.5 py-0.5 text-xs text-ink-2">Not available</p>
        )}
      </div>
      <p className="mb-4 max-w-2xl text-sm text-ink-2">{why}</p>
      {available ? (
        <ErrorBoundary title="This file couldn't be displayed" message={`${value.source} isn't in the format this page expects, so its numbers aren't shown.`}>
          <SectionBody render={children} data={value.data} />
        </ErrorBoundary>
      ) : (
        <p className="text-sm text-ink-3">The notebook that produces this result hasn't saved its file yet.</p>
      )}
    </motion.section>
  )
}

function Stat({ label, value, delay = 0 }: { label: string; value: number; delay?: number }) {
  return (
    <div className="hud-panel p-4">
      <dt className="text-sm text-ink-2">{label}</dt>
      <dd className="mt-1 text-3xl font-semibold tracking-tight">
        <CountUp value={value} delay={delay} duration={1.1} />
      </dd>
    </div>
  )
}

/** A value with a thin bar. The number stays readable on its own. */
function Rate({ value }: { value?: number | null }) {
  if (value == null) return <span className="text-ink-3">Not measured</span>
  return (
    <span className="flex items-center justify-end gap-2">
      <span aria-hidden className="hidden h-1.5 w-20 rounded-full bg-line sm:block">
        <motion.span
          className="block h-full origin-left rounded-r-full"
          style={{ width: `${value * 100}%`, background: SERIES_1, boxShadow: `0 0 8px ${SERIES_1}` }}
          initial={{ scaleX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
        />
      </span>
      <span className="num font-mono">{percent(value)}</span>
    </span>
  )
}

function Plain({ children }: { children: ReactNode }) {
  return children == null ? <span className="text-ink-3">Not measured</span> : <span className="num font-mono">{children}</span>
}

function Table({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <div className="hud-panel overflow-x-auto">
      <table className="w-full min-w-max text-sm">
        <thead>
          <tr className="border-b border-line text-ink-3">
            {head.map((heading, index) => (
              <th key={heading} scope="col" className={`px-4 py-2.5 font-normal ${index === 0 ? 'text-left' : 'text-right'}`}>
                {heading}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) => (
                <td key={c} className={`px-4 py-2.5 ${c === 0 ? 'text-left' : 'text-right'}`}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function TableView({ children }: { children: ReactNode }) {
  return (
    <details className="mt-3">
      <summary className="min-h-8 text-sm text-ink-2 hover:text-ink">Table view</summary>
      <div className="mt-2">{children}</div>
    </details>
  )
}

type Robustness = NonNullable<Extract<MetricsResponse['robustness'], { data: unknown }>['data']>

function RobustnessCharts({ rows }: Robustness) {
  // One result per perturbation can't make a curve, so show it as a table.
  if (rows.every((row) => row.severity == null)) {
    return <Table head={['Perturbation', 'Top-1']} rows={rows.map((row) => [humanise(row.perturbation), <Rate value={row.top1} />])} />
  }

  const groups = [...new Set(rows.map((row) => row.perturbation))]
  const severities = [...new Set(rows.map((row) => row.severity as number))].sort((a, b) => a - b)
  return (
    <>
      <div className="grid gap-x-6 gap-y-6 hud-panel p-4 sm:grid-cols-2 lg:grid-cols-3">
        {groups.map((group) => {
          const points = rows.filter((row) => row.perturbation === group).sort((a, b) => (a.severity as number) - (b.severity as number))
          return (
            <LineChart
              key={group}
              title={humanise(group)}
              xLabel="Severity"
              xs={points.map((p) => p.severity as number)}
              series={[{ name: 'Top-1', color: SERIES_1, values: points.map((p) => p.top1) }]}
              height={170}
            />
          )
        })}
      </div>
      <TableView>
        <Table
          head={['Perturbation', ...severities.map((s) => `Severity ${s}`)]}
          rows={groups.map((group) => [
            humanise(group),
            ...severities.map((s) => {
              const row = rows.find((r) => r.perturbation === group && r.severity === s)
              return <Plain key={s}>{row ? percent(row.top1) : null}</Plain>
            }),
          ])}
        />
      </TableView>
    </>
  )
}

function Skeleton() {
  return (
    <div role="status" aria-label="Loading saved results" className="space-y-4">
      <div className="skeleton h-28 rounded-xl" />
      <div className="skeleton h-48 rounded-xl" />
    </div>
  )
}

export default function Lab() {
  const metrics = useResource(getMetrics, 'metrics')

  return (
    <div>
      <PageHeader
        eyebrow="06 / Research Lab"
        title="Research Lab"
        intro="Every number here is read from JSON files saved by the training and evaluation notebooks. Nothing is written into the page, and a missing file is reported as not available."
      />

      {metrics.status === 'loading' && <Skeleton />}
      {metrics.status === 'error' && <ErrorNotice title="Results unavailable" message={metrics.error} onRetry={metrics.reload} />}

      {metrics.status === 'ready' && (
        <>
          {metrics.data.message && <p className="mb-6 rounded-lg border border-line bg-surface p-3 text-sm text-ink-2">{metrics.data.message}</p>}
          <Section title="Classifier accuracy" why="Top-1 and Top-5 on the unseen test set. Ground-truth crops use the annotated box. Detector crops, where measured, use the box the detector found." value={metrics.data.classifier}>
            {(d) => (
              <>
                <dl className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                  <Stat label="Top-1, ground-truth crop" value={d.top1} />
                  <Stat label="Top-5, ground-truth crop" value={d.top5} delay={0.1} />
                  {d.top1_detector_crop != null && <Stat label="Top-1, detector crop" value={d.top1_detector_crop} delay={0.2} />}
                  {d.top5_detector_crop != null && <Stat label="Top-5, detector crop" value={d.top5_detector_crop} delay={0.3} />}
                  {d.make_top1 != null && <Stat label="Make-level Top-1" value={d.make_top1} delay={0.4} />}
                  {d.macro_f1 != null && <Stat label="Macro F1" value={d.macro_f1} delay={0.5} />}
                </dl>
                {d.val_top1 != null && (
                  <p className="num mt-3 font-mono text-sm text-ink-3">
                    validation Top-1 {percent(d.val_top1)}
                    {d.val_top5 != null && ` · Top-5 ${percent(d.val_top5)}`}
                  </p>
                )}
                {(d.model || d.test_images != null) && (
                  <p className="mt-3 text-sm text-ink-3">
                    {d.model}
                    {d.model && d.test_images != null && ' · '}
                    {d.test_images != null && `${d.test_images.toLocaleString()} test images`}
                  </p>
                )}
              </>
            )}
          </Section>

          <Section title="Training run" why="How the classifier was trained and on what, as recorded by its notebook." value={metrics.data.run}>
            {(d) => (
              <dl className="hud-panel divide-y divide-line px-4">
                {d.rows.map((row) => (
                  <div key={row.label} className="flex flex-wrap justify-between gap-x-6 gap-y-1 py-2.5">
                    <dt className="text-sm text-ink-2">{row.label}</dt>
                    <dd className="num text-right font-mono text-sm">{row.value}</dd>
                  </div>
                ))}
              </dl>
            )}
          </Section>

          <Section title="Detection" why="IoU and COCO-style mAP on the Stanford boxes. Each image has one annotated box, so other visible cars can count as false positives." value={metrics.data.detection}>
            {(d) => (
              <Table
                head={['Detector', 'Mean IoU', 'mAP', 'CPU latency']}
                rows={d.rows.map((r) => [r.model, <Plain>{r.mean_iou.toFixed(3)}</Plain>, <Plain>{r.map.toFixed(3)}</Plain>, <Plain>{r.cpu_ms == null ? null : milliseconds(r.cpu_ms)}</Plain>])}
              />
            )}
          </Section>

          <Section title="Size and speed" why="The optimisation result: model size, inference time and accuracy before and after. Times are per image, on the machine that ran the notebook." value={metrics.data.optimisation}>
            {(d) => {
              const hasGpu = d.rows.some((r) => r.gpu_ms != null)
              const hasTop5 = d.rows.some((r) => r.top5 != null)
              const hasAgreement = d.rows.some((r) => r.agreement != null)
              return (
                <Table
                  head={['Variant', 'Size', 'CPU', ...(hasGpu ? ['GPU'] : []), 'Top-1', ...(hasTop5 ? ['Top-5'] : []), ...(hasAgreement ? ['Agrees with original'] : [])]}
                  rows={d.rows.map((r) => [
                    <span key="v" className="flex flex-wrap items-center gap-2">
                      {r.variant}
                      {r.note && <span className="rounded-full border border-accent/60 px-2 py-0.5 text-xs text-accent">{r.note}</span>}
                    </span>,
                    <Plain>{r.size_mb.toFixed(1)} MB</Plain>,
                    <Plain>{r.cpu_ms == null ? null : milliseconds(r.cpu_ms)}</Plain>,
                    ...(hasGpu ? [<Plain>{r.gpu_ms == null ? null : milliseconds(r.gpu_ms)}</Plain>] : []),
                    <Rate value={r.top1} />,
                    ...(hasTop5 ? [<Rate value={r.top5} />] : []),
                    ...(hasAgreement ? [<Rate value={r.agreement} />] : []),
                  ])}
                />
              )
            }}
          </Section>

          <Section title="Robustness" why="Top-1 as brightness, low light, blur, noise, low resolution and occlusion get worse." value={metrics.data.robustness}>
            {(d) => <RobustnessCharts rows={d.rows} />}
          </Section>

          <Section title="Confidence and rejection" why="Raising the threshold rejects more images and is more accurate on those it keeps. The chosen threshold decides the confident state on the identity page." value={metrics.data.confidence}>
            {(d) => {
              const rows = [...d.rows].sort((a, b) => a.threshold - b.threshold)
              // One threshold is a single point, which cannot be drawn as a curve, so show it as a table.
              if (rows.length < 2) {
                return (
                  <>
                    <Table
                      head={['Threshold', 'Coverage', 'Accuracy']}
                      rows={rows.map((r) => [<Plain>{r.threshold.toFixed(2)}</Plain>, <Plain>{percent(r.coverage)}</Plain>, <Plain>{percent(r.accuracy)}</Plain>])}
                    />
                    {d.chosen_threshold != null && <p className="mt-3 text-sm text-ink-3">Chosen on the validation set: {d.chosen_threshold}.</p>}
                  </>
                )
              }
              return (
                <>
                  <div className="hud-panel p-4">
                    <LineChart
                      title="Coverage and accuracy by confidence threshold"
                      xLabel="Threshold"
                      xs={rows.map((r) => r.threshold)}
                      formatX={(x) => x.toFixed(2)}
                      series={[
                        { name: 'Accuracy on kept images', color: SERIES_1, values: rows.map((r) => r.accuracy) },
                        { name: 'Coverage', color: SERIES_2, values: rows.map((r) => r.coverage) },
                      ]}
                      marker={d.chosen_threshold != null ? { x: d.chosen_threshold, label: 'Chosen' } : null}
                      height={260}
                    />
                  </div>
                  <TableView>
                    <Table
                      head={['Threshold', 'Coverage', 'Accuracy']}
                      rows={rows.map((r) => [<Plain>{r.threshold.toFixed(2)}</Plain>, <Plain>{percent(r.coverage)}</Plain>, <Plain>{percent(r.accuracy)}</Plain>])}
                    />
                  </TableView>
                </>
              )
            }}
          </Section>

          <Section title="Confidence states" why="What the identity page does with each answer, and how often each state is right. A confident answer should be right far more often than an ambiguous or rejected one." value={metrics.data.states}>
            {(d) => (
              <>
                <Table
                  head={['State', 'Share of test images', 'Accuracy']}
                  rows={d.rows.map((r) => [<ConfidenceBadge key={r.state} state={r.state} />, <Rate value={r.share} />, <Rate value={r.accuracy} />])}
                />
                {d.rule && <p className="mt-3 max-w-2xl text-sm text-ink-2">{d.rule}</p>}
              </>
            )}
          </Section>

          <Section
            title="Photos outside the 196 cars"
            why="The classifier always names one of its 196 classes, so only the confidence rule protects against inputs it was never taught. This counts how often that rule fires on airplanes, motorbikes, boats, animals and objects, against Stanford Cars photos scored the same way."
            value={metrics.data.out_of_taxonomy}
          >
            {(d) => (
              <>
                <Table
                  head={['Photos', 'Images', 'Rejected', 'Ambiguous', 'Confident']}
                  rows={d.rows.map((r) => [r.group, <Plain>{r.images.toLocaleString()}</Plain>, <Rate value={r.rejected} />, <Rate value={r.ambiguous} />, <Rate value={r.confident} />])}
                />
                {d.auroc != null && (
                  <p className="mt-3 max-w-2xl text-sm text-ink-2">
                    Telling in-class photos from outside ones by the top probability alone scores an AUROC of <span className="num text-ink">{d.auroc.toFixed(3)}</span> (1 is perfect, 0.5 is guessing).
                  </p>
                )}
                {d.limits && d.limits.length > 0 && (
                  <ul className="mt-3 max-w-2xl list-disc space-y-1 pl-5 text-sm text-ink-3">
                    {d.limits.map((limit) => (
                      <li key={limit}>{limit}</li>
                    ))}
                  </ul>
                )}
              </>
            )}
          </Section>

          <Section title="Calibration" why="A calibrated model's stated confidence matches how often it is right. Temperature scaling, fitted on the validation set, corrects it. ECE is the average gap between the two, and lower is better." value={metrics.data.calibration}>
            {(d) => (
              <>
                <dl className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
                  <div className="hud-panel p-4">
                    <dt className="text-sm text-ink-2">Temperature</dt>
                    <dd className="num mt-1 text-3xl font-semibold tracking-tight">{d.temperature.toFixed(3)}</dd>
                  </div>
                </dl>
                <Table
                  head={['Measure', 'Before', 'After']}
                  rows={d.rows.map((r) => [r.metric, <Plain>{r.before?.toFixed(3)}</Plain>, <Plain>{r.after?.toFixed(3)}</Plain>])}
                />
              </>
            )}
          </Section>

          <Section title="Classical vs deep vs hybrid" why="All three on the same split. If fusion doesn't help, this table shows that." value={metrics.data.comparison}>
            {(d) => (
              <Table
                head={['Approach', 'Top-1', 'Top-5', 'Inference', 'Size']}
                rows={d.rows.map((r) => [
                  r.approach,
                  <Rate value={r.top1} />,
                  <Rate value={r.top5} />,
                  <Plain>{r.inference_ms == null ? null : milliseconds(r.inference_ms)}</Plain>,
                  <Plain>{r.size_mb == null ? null : `${r.size_mb.toFixed(1)} MB`}</Plain>,
                ])}
              />
            )}
          </Section>

          <Section title="Ablation study" why="One component removed at a time, to show what each contributes. Retraining runs may use fewer epochs, noted beside the row." value={metrics.data.ablation}>
            {(d) => (
              <>
                {d.baseline_top1 != null && <p className="mb-3 text-sm text-ink-2">Baseline Top-1: <span className="num font-mono text-ink">{percent(d.baseline_top1)}</span></p>}
                <Table
                  head={['Removed', 'Top-1', 'Change', 'Note']}
                  rows={d.rows.map((r) => [
                    r.removed,
                    <Rate value={r.top1} />,
                    <Plain>{d.baseline_top1 == null ? null : `${((r.top1 - d.baseline_top1) * 100).toFixed(1)} pts`}</Plain>,
                    <span className="text-ink-3">{r.note}</span>,
                  ])}
                />
              </>
            )}
          </Section>

          <Section title="Most-confused pairs" why="Which similar classes the model mixes up most, usually neighbouring model years." value={metrics.data.confused_pairs}>
            {(d) => <Table head={['Class', 'Confused with', 'Errors']} rows={d.rows.map((r) => [r.a, <span className="block text-left sm:text-right">{r.b}</span>, <Plain>{r.count}</Plain>])} />}
          </Section>

          <Section title="Failure cases" why="Examples the system gets wrong, and why." value={metrics.data.failure_cases}>
            {(d) => (
              <ul className="grid gap-4 md:grid-cols-3">
                {d.cases.map((item) => (
                  <li key={item.title} className="hud-panel overflow-hidden">
                    {item.image && <img src={item.image} alt={item.title} loading="lazy" className="aspect-[4/3] w-full object-cover" />}
                    <div className="p-4">
                      <h3 className="font-medium">{item.title}</h3>
                      {(item.predicted || item.actual) && (
                        <dl className="mt-2 text-xs text-ink-2">
                          {item.predicted && <div className="flex gap-1.5"><dt className="text-ink-3">Predicted</dt><dd>{item.predicted}</dd></div>}
                          {item.actual && <div className="flex gap-1.5"><dt className="text-ink-3">Actual</dt><dd>{item.actual}</dd></div>}
                        </dl>
                      )}
                      <p className="mt-2 text-sm text-ink-2">{item.explanation}</p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Section>
        </>
      )}
    </div>
  )
}
