import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import DevPanel from '../../components/DevPanel'

/**
 * DevPanel Tests
 *
 * Tests the developer tools panel component.
 */

describe('DevPanel', () => {
  describe('Closed State', () => {
    it('returns null when closed', () => {
      const { container } = render(<DevPanel isOpen={false} onClose={vi.fn()} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Open State', () => {
    it('renders when open', () => {
      const { container } = render(<DevPanel isOpen={true} onClose={vi.fn()} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows Developer Panel title', () => {
      render(<DevPanel isOpen={true} onClose={vi.fn()} />)
      expect(screen.getByText('Developer Panel')).toBeInTheDocument()
    })

    it('shows description text', () => {
      render(<DevPanel isOpen={true} onClose={vi.fn()} />)
      expect(screen.getByText('Developer tools and debugging options')).toBeInTheDocument()
    })

    it('renders close button', () => {
      const { container } = render(<DevPanel isOpen={true} onClose={vi.fn()} />)
      const button = container.querySelector('button')
      expect(button).toBeInTheDocument()
    })
  })

  describe('Interactions', () => {
    it('calls onClose when close button clicked', () => {
      const onClose = vi.fn()
      const { container } = render(<DevPanel isOpen={true} onClose={onClose} />)

      const closeButton = container.querySelector('button')
      if (closeButton) {
        fireEvent.click(closeButton)
      }

      expect(onClose).toHaveBeenCalledTimes(1)
    })
  })

  describe('Overlay', () => {
    it('has fixed positioning overlay', () => {
      const { container } = render(<DevPanel isOpen={true} onClose={vi.fn()} />)
      const overlay = container.firstChild as HTMLElement
      expect(overlay).toHaveClass('fixed', 'inset-0')
    })

    it('has z-50 for overlay', () => {
      const { container } = render(<DevPanel isOpen={true} onClose={vi.fn()} />)
      const overlay = container.firstChild as HTMLElement
      expect(overlay).toHaveClass('z-50')
    })
  })
})
