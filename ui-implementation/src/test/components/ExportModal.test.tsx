import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ExportModal } from '../../components/ExportModal'

/**
 * ExportModal Tests
 *
 * Tests the data export modal component.
 */

// Mock fetch
global.fetch = vi.fn().mockResolvedValue({
  ok: true,
  json: () => Promise.resolve({
    total_mb: 25.5,
    breakdown: {
      sessions_mb: 10.0,
      memory_mb: 8.5,
      books_mb: 5.0,
      knowledge_mb: 2.0,
    },
    file_counts: {
      sessions: 15,
      memory: 100,
      books: 3,
      knowledge: 50,
    },
  }),
})

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('ExportModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Closed State', () => {
    it('returns null when closed', () => {
      const { container } = render(<ExportModal {...defaultProps} isOpen={false} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Open State', () => {
    it('renders when open', () => {
      const { container } = render(<ExportModal {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows Export Data title', () => {
      render(<ExportModal {...defaultProps} />)
      expect(screen.getByText('Export Data')).toBeInTheDocument()
    })

    it('shows export options', () => {
      render(<ExportModal {...defaultProps} />)
      expect(screen.getByText('Conversations')).toBeInTheDocument()
      expect(screen.getByText('Memory (ChromaDB)')).toBeInTheDocument()
      expect(screen.getByText('Books & Documents')).toBeInTheDocument()
      expect(screen.getByText('Knowledge & Learning')).toBeInTheDocument()
    })

    it('has all checkboxes checked by default', () => {
      render(<ExportModal {...defaultProps} />)
      const checkboxes = screen.getAllByRole('checkbox')
      checkboxes.forEach(checkbox => {
        expect(checkbox).toBeChecked()
      })
    })

    it('shows Select All / Deselect All button', () => {
      render(<ExportModal {...defaultProps} />)
      expect(screen.getByText('Deselect All')).toBeInTheDocument()
    })
  })

  describe('Checkbox Toggling', () => {
    it('toggles checkbox when clicked', () => {
      render(<ExportModal {...defaultProps} />)

      const sessionLabel = screen.getByText('Conversations').closest('label')
      const checkbox = sessionLabel?.querySelector('input[type="checkbox"]')

      if (checkbox) {
        fireEvent.click(checkbox)
        expect(checkbox).not.toBeChecked()
      }
    })

    it('shows Select All when some are unchecked', () => {
      render(<ExportModal {...defaultProps} />)

      const sessionLabel = screen.getByText('Conversations').closest('label')
      const checkbox = sessionLabel?.querySelector('input[type="checkbox"]')

      if (checkbox) {
        fireEvent.click(checkbox)
        expect(screen.getByText('Select All')).toBeInTheDocument()
      }
    })
  })

  describe('Export Button', () => {
    it('shows export button', () => {
      render(<ExportModal {...defaultProps} />)
      expect(screen.getByText('Export Selected Data')).toBeInTheDocument()
    })

    it('disables export button when none selected', () => {
      render(<ExportModal {...defaultProps} />)

      // Deselect all
      fireEvent.click(screen.getByText('Deselect All'))

      const exportButton = screen.getByText('Export Selected Data').closest('button')
      expect(exportButton).toBeDisabled()
    })
  })

  describe('Close Behavior', () => {
    it('calls onClose when close button clicked', () => {
      const { container } = render(<ExportModal {...defaultProps} />)

      // Find the close button (first button in header)
      const closeButton = container.querySelector('button[title="Close"]')
      if (closeButton) {
        fireEvent.click(closeButton)
        expect(defaultProps.onClose).toHaveBeenCalledTimes(1)
      }
    })

    it('calls onClose when overlay clicked', () => {
      const { container } = render(<ExportModal {...defaultProps} />)

      // Click on the overlay (first child)
      const overlay = container.firstChild as HTMLElement
      fireEvent.click(overlay)
      expect(defaultProps.onClose).toHaveBeenCalledTimes(1)
    })

    it('does not close when modal content clicked', () => {
      render(<ExportModal {...defaultProps} />)

      fireEvent.click(screen.getByText('Export Data'))
      expect(defaultProps.onClose).not.toHaveBeenCalled()
    })
  })
})
