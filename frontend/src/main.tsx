import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { MotionConfig } from 'framer-motion'
import App from './App'
import { SessionProvider } from './state/session'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <MotionConfig reducedMotion="user">
        <SessionProvider>
          <App />
        </SessionProvider>
      </MotionConfig>
    </BrowserRouter>
  </StrictMode>,
)
