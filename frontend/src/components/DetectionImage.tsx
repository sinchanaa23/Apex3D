import { motion } from 'framer-motion'
import type { RecognizeResponse } from '../api/types'
import { percent } from '../lib/format'

/** The photo with the detector's box. Positions are percentages, so the box stays right at any display size. */
export function DetectionImage({ src, recognition, name }: { src: string; recognition: RecognizeResponse; name: string }) {
  const { detection, image_size: size } = recognition
  const box = detection?.box
  const alt = detection ? `Your photo, with a box around the detected vehicle, labelled ${name}` : 'Your photo. No vehicle was detected.'
  return (
    <figure className="relative overflow-hidden rounded-xl border border-line bg-surface">
      <img src={src} alt={alt} className="block h-auto w-full" />
      {box && (
        <motion.div
          initial={{ opacity: 0, scale: 1.04 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.35, ease: 'easeOut' }}
          className="absolute rounded-sm border-2 border-accent shadow-[0_0_0_1px_rgb(0_0_0/0.5)]"
          style={{
            left: `${(box[0] / size.width) * 100}%`,
            top: `${(box[1] / size.height) * 100}%`,
            width: `${((box[2] - box[0]) / size.width) * 100}%`,
            height: `${((box[3] - box[1]) / size.height) * 100}%`,
          }}
        >
          <span className="absolute top-0 left-0 max-w-full truncate rounded-br bg-accent px-1.5 py-0.5 text-xs font-medium text-page">
            {name}
          </span>
          <span className="num absolute right-0 bottom-0 rounded-tl bg-page/85 px-1.5 py-0.5 font-mono text-xs text-ink">
            detector {percent(detection.score, 0)}
          </span>
        </motion.div>
      )}
    </figure>
  )
}
