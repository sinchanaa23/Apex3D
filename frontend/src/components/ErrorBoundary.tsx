import { Component, type ReactNode } from 'react'
import { ErrorNotice } from './States'

interface Props {
  title: string
  /** Shown instead of the default message when something inside fails to render. */
  message?: string
  children: ReactNode
}

/** Stops one broken section, such as a metrics file in an unexpected format, from blanking the whole page. */
export class ErrorBoundary extends Component<Props, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() {
    return { failed: true }
  }

  render() {
    if (!this.state.failed) return this.props.children
    return (
      <ErrorNotice
        title={this.props.title}
        message={this.props.message ?? 'Reload the page to try again. If it keeps happening, start again from a new photo.'}
      />
    )
  }
}
