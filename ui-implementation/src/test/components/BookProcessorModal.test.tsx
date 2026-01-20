import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { BookProcessorModal } from '../../components/BookProcessorModal'

/**
 * BookProcessorModal Tests
 *
 * Tests the book/document processing modal.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({
      books: [],
    }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('BookProcessorModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Closed State', () => {
    it('returns null when closed', () => {
      const { container } = render(<BookProcessorModal {...defaultProps} isOpen={false} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Open State', () => {
    it('renders when open', () => {
      const { container } = render(<BookProcessorModal {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows Document Processor title', () => {
      render(<BookProcessorModal {...defaultProps} />)
      expect(screen.getByText('Document Processor')).toBeInTheDocument()
    })
  })

  describe('Close Behavior', () => {
    it('calls onClose when close button clicked', () => {
      const { container } = render(<BookProcessorModal {...defaultProps} />)

      const closeButton = container.querySelector('button')
      if (closeButton) {
        closeButton.click()
      }

      expect(defaultProps.onClose).toHaveBeenCalled()
    })
  })
})
