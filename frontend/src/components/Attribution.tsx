import type { Reference } from '../api/types'

const linkClass = 'underline underline-offset-2 hover:text-ink'

/** Source, author and licence. CC BY-SA requires these to stay beside the image wherever it appears. */
export function Attribution({ reference }: { reference: Reference }) {
  const { source, source_url, author, licence, licence_url } = reference
  return (
    <p className="text-xs text-ink-3">
      {source_url ? (
        <a href={source_url} target="_blank" rel="noreferrer" className={linkClass}>
          {source}
        </a>
      ) : (
        source
      )}
      {author && ` · ${author}`}
      {licence && ' · '}
      {licence &&
        (licence_url ? (
          <a href={licence_url} target="_blank" rel="noreferrer" className={linkClass}>
            {licence}
          </a>
        ) : (
          licence
        ))}
    </p>
  )
}
