import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { primaryButton, secondaryButton } from '../components/States'

const STAGES = [
  { title: 'Recognise', body: 'An R-CNN finds the car. A fine-tuned classifier suggests make, model and year from 196 classes, and says when it is unsure.' },
  { title: 'Confirm', body: 'You accept or correct the identity. The classical CV view shows what the image-processing stages saw.' },
  { title: 'Reference', body: 'Openly licensed photos of that model are checked by the classifier and grouped by viewpoint, with attribution kept.' },
  { title: 'Reconstruct', body: 'TripoSR makes a mesh from one view. You inspect it, read its validation report, and export GLB, OBJ or STL.' },
]

const LIMITS = [
  'Only 196 classes, up to 2012 model years. Newer cars may be misidentified or rejected.',
  'A mesh comes from a single image, so the unseen sides are inferred and the scale is unknown.',
  'References show the same model, not the same physical car.',
  'Meshes are drafts. They need work before use in games or 3D printing.',
]

export default function Home() {
  return (
    <div>
      <section className="pt-6 sm:pt-14">
        <h1 className="max-w-3xl text-5xl leading-[1.02] font-semibold tracking-tight sm:text-7xl">
          Recognise.
          <br />
          Reconstruct.
          <br />
          <span className="text-accent">Create.</span>
        </h1>
        <p className="mt-6 max-w-xl text-lg text-ink-2">
          Upload a photo of a car. Apex3D works out what it is, shows how sure it is, and helps you turn that identity into a draft 3D model.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <Link to="/upload" className={primaryButton}>
            Identify a car
            <ArrowRight aria-hidden className="size-4" />
          </Link>
          <Link to="/lab" className={secondaryButton}>
            See the evaluation
          </Link>
        </div>
      </section>

      <section aria-label="How it works" className="mt-14 sm:mt-20">
        <motion.svg
          aria-hidden
          viewBox="0 0 1200 120"
          preserveAspectRatio="none"
          className="hidden h-24 w-full md:block"
          initial={{ clipPath: 'inset(0 100% 0 0)' }}
          animate={{ clipPath: 'inset(0 0% 0 0)' }}
          transition={{ duration: 1.4, ease: 'easeInOut' }}
        >
          <path
            d="M0 112 C 220 112, 300 104, 450 62 S 640 8, 760 26 S 1000 96, 1200 100"
            fill="none"
            stroke="var(--color-accent)"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />
        </motion.svg>
        <ol className="grid gap-6 border-l-2 border-accent pl-5 md:grid-cols-4 md:border-l-0 md:pl-0">
          {STAGES.map(({ title, body }, index) => (
            <li key={title}>
              <p className="num font-mono text-xs text-ink-3">0{index + 1}</p>
              <h2 className="mt-1 text-lg font-medium">{title}</h2>
              <p className="mt-1.5 text-sm text-ink-2">{body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-14 border-t border-line pt-8 sm:mt-20">
        <h2 className="text-lg font-medium">What it won't claim</h2>
        <ul className="mt-4 grid gap-x-10 gap-y-3 text-sm text-ink-2 md:grid-cols-2">
          {LIMITS.map((limit) => (
            <li key={limit} className="border-l border-line-strong pl-3">
              {limit}
            </li>
          ))}
        </ul>
        <p className="mt-6 text-xs text-ink-3">
          A research demonstration. Uploaded photos pass through a tunnel to a team laptop and are deleted after processing.
        </p>
      </section>
    </div>
  )
}
