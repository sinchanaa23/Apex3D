import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'
import type { ClassPrediction, ClassicalResponse, RecognizeResponse, Reference } from '../api/types'
import type { NormalisedImage } from '../lib/image'

/** The single view that goes into reconstruction: the user's photo or one reference. */
export type InputChoice = { kind: 'photo' } | { kind: 'reference'; reference: Reference }

interface Session {
  photo: NormalisedImage | null
  recognition: RecognizeResponse | null
  classical: ClassicalResponse | null
  /** The class the user confirmed. Null if they haven't, or chose to continue without one. */
  identity: ClassPrediction | null
  /** True once the user has made the confirm-or-skip decision. */
  identityDecided: boolean
  input: InputChoice | null
  pack: Reference[]
  uploadedReferences: Reference[]
}

const EMPTY: Session = {
  photo: null,
  recognition: null,
  classical: null,
  identity: null,
  identityDecided: false,
  input: null,
  pack: [],
  uploadedReferences: [],
}

interface SessionApi extends Session {
  startWithPhoto(photo: NormalisedImage, recognition: RecognizeResponse): void
  setClassical(result: ClassicalResponse): void
  decideIdentity(identity: ClassPrediction | null): void
  setInput(input: InputChoice): void
  togglePack(reference: Reference): void
  addUploadedReferences(references: Reference[]): void
}

const SessionContext = createContext<SessionApi | null>(null)

// Held in memory only. The proposal promises uploads aren't retained, so nothing goes in browser storage.
export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session>(EMPTY)

  const startWithPhoto = useCallback((photo: NormalisedImage, recognition: RecognizeResponse) => {
    setSession((previous) => {
      if (previous.photo) URL.revokeObjectURL(previous.photo.url)
      return { ...EMPTY, photo, recognition, input: { kind: 'photo' } }
    })
  }, [])
  const setClassical = useCallback((classical: ClassicalResponse) => setSession((s) => ({ ...s, classical })), [])
  const decideIdentity = useCallback(
    (identity: ClassPrediction | null) => setSession((s) => ({ ...s, identity, identityDecided: true })),
    [],
  )
  const setInput = useCallback((input: InputChoice) => setSession((s) => ({ ...s, input })), [])
  const togglePack = useCallback(
    (reference: Reference) =>
      setSession((s) => ({
        ...s,
        pack: s.pack.some((r) => r.id === reference.id) ? s.pack.filter((r) => r.id !== reference.id) : [...s.pack, reference],
      })),
    [],
  )
  const addUploadedReferences = useCallback(
    (references: Reference[]) => setSession((s) => ({ ...s, uploadedReferences: [...references, ...s.uploadedReferences] })),
    [],
  )

  const value = useMemo(
    () => ({ ...session, startWithPhoto, setClassical, decideIdentity, setInput, togglePack, addUploadedReferences }),
    [session, startWithPhoto, setClassical, decideIdentity, setInput, togglePack, addUploadedReferences],
  )
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>
}

export function useSession() {
  const context = useContext(SessionContext)
  if (!context) throw new Error('useSession must be used inside SessionProvider')
  return context
}
