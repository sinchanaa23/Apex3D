import { motion } from 'framer-motion'
import type { RecognizeResponse } from '../api/types'
import { ScanFrame } from './hud/ScanFrame'
import { CountUp } from './hud/CountUp'

/** The photo with the detector's box. Positions are percentages, so the box stays right at any display size. */
export function DetectionImage({ src, recognition, name }: { src: string; recognition: RecognizeResponse; name: string }) {
  const { detection, image_size: size } = recognition
  const box = detection?.box
  const alt = detection ? `Your photo, with a box around the detected vehicle, labelled ${name}` : 'Your photo. No vehicle was detected.'
  return (
    <ScanFrame>
      <img src={src} alt={alt} className="block h-auto w-full" />
      {box && (
        <motion.div
          initial={{ opacity: 0, scale: 1.06 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1], delay: 0.9 }}
          className="absolute border border-accent/70 shadow-[0_0_0_1px_rgb(0_0_0/0.5),0_0_24px_rgb(255_106_43/0.35),inset_0_0_24px_rgb(255_106_43/0.12)]"
          style={{
            left: `${(box[0] / size.width) * 100}%`,
            top: `${(box[1] / size.height) * 100}%`,
            width: `${((box[2] - box[0]) / size.width) * 100}%`,
            height: `${((box[3] - box[1]) / size.height) * 100}%`,
          }}
        >
          {/* Heavier corner marks make the box read as a lock-on. */}
          <span aria-hidden className="absolute -top-px -left-px size-3.5 border-t-[3px] border-l-[3px] border-accent" />
          <span aria-hidden className="absolute -top-px -right-px size-3.5 border-t-[3px] border-r-[3px] border-accent" />
          <span aria-hidden className="absolute -bottom-px -left-px size-3.5 border-b-[3px] border-l-[3px] border-accent" />
          <span aria-hidden className="absolute -right-px -bottom-px size-3.5 border-r-[3px] border-b-[3px] border-accent" />
          <span className="absolute top-0 left-0 max-w-full truncate rounded-br bg-accent px-1.5 py-0.5 text-xs font-medium text-page">{name}</span>
          <span className="absolute right-0 bottom-0 rounded-tl bg-page/85 px-1.5 py-0.5 font-mono text-xs text-ink">
            detector <CountUp value={detection.score} format={(v) => `${(v * 100).toFixed(0)}%`} delay={1} />
          </span>
        </motion.div>
      )}
    </ScanFrame>
  )
}
