import { useNavigate } from 'react-router-dom'
import { PhotoUpload } from '../components/PhotoUpload'
import { PageHeader } from '../components/States'

export default function Upload() {
  const navigate = useNavigate()
  return (
    <div className="mx-auto max-w-3xl">
      <PageHeader eyebrow="01 / Upload" title="Upload a photo" intro="One car, reasonably in frame, works best. JPEG, PNG or WebP, up to 20 MB." />
      <PhotoUpload onDone={() => navigate('/identity')} />
    </div>
  )
}
