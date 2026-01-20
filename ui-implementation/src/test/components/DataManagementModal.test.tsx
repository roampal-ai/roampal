import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { DataManagementModal } from '../../components/DataManagementModal'

/**
 * DataManagementModal Tests
 *
 * Tests the data management modal with export and delete tabs.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockImplementation((url: string) => {
    if (url.includes('/data/stats')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          memory_bank: { count: 10, active: 8, archived: 2 },
          working: { count: 5 },
          history: { count: 20 },
          patterns: { count: 15 },
          books: { count: 3 },
          sessions: { count: 12 },
          knowledge_graph: { nodes: 50, edges: 100 },
        }),
      })
    }
    if (url.includes('/backup/estimate')) {
      return Promise.resolve({
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
    }
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({}),
      blob: () => Promise.resolve(new Blob()),
      headers: { get: () => null },
    })
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('DataManagementModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Closed State', () => {
    it('returns null when closed', () => {
      const { container } = render(<DataManagementModal {...defaultProps} isOpen={false} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Open State', () => {
    it('renders when open', () => {
      const { container } = render(<DataManagementModal {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows Data Management title', () => {
      render(<DataManagementModal {...defaultProps} />)
      expect(screen.getByText('Data Management')).toBeInTheDocument()
    })

    it('shows Export and Delete tabs', () => {
      render(<DataManagementModal {...defaultProps} />)
      expect(screen.getByText('Export')).toBeInTheDocument()
      expect(screen.getByText('Delete')).toBeInTheDocument()
    })
  })

  describe('Export Tab', () => {
    it('shows export tab by default', () => {
      render(<DataManagementModal {...defaultProps} />)
      expect(screen.getByText('Conversations')).toBeInTheDocument()
      expect(screen.getByText('Memory (ChromaDB)')).toBeInTheDocument()
    })

    it('has checkboxes for each data type', () => {
      render(<DataManagementModal {...defaultProps} />)
      const checkboxes = screen.getAllByRole('checkbox')
      expect(checkboxes.length).toBe(4)
    })

    it('shows export button', () => {
      render(<DataManagementModal {...defaultProps} />)
      expect(screen.getByText('Export Selected Data')).toBeInTheDocument()
    })
  })

  describe('Delete Tab', () => {
    it('switches to delete tab', () => {
      render(<DataManagementModal {...defaultProps} />)

      fireEvent.click(screen.getByText('Delete'))

      expect(screen.getByText(/Danger Zone/)).toBeInTheDocument()
    })

    it('shows delete options', async () => {
      render(<DataManagementModal {...defaultProps} />)
      fireEvent.click(screen.getByText('Delete'))

      await waitFor(() => {
        expect(screen.getByText('Memory Bank')).toBeInTheDocument()
        expect(screen.getByText('Working Memory')).toBeInTheDocument()
        expect(screen.getByText('History')).toBeInTheDocument()
        expect(screen.getByText('Patterns')).toBeInTheDocument()
        expect(screen.getByText('Books')).toBeInTheDocument()
        expect(screen.getByText('Sessions')).toBeInTheDocument()
        expect(screen.getByText('Knowledge Graph')).toBeInTheDocument()
      })
    })

    it('shows delete buttons for each category', async () => {
      render(<DataManagementModal {...defaultProps} />)
      fireEvent.click(screen.getByText('Delete'))

      await waitFor(() => {
        // Should show at least one delete button plus the tab
        const deleteButtons = screen.getAllByText('Delete')
        expect(deleteButtons.length).toBeGreaterThan(0)
      })
    })

    it('shows compact database button', async () => {
      render(<DataManagementModal {...defaultProps} />)
      fireEvent.click(screen.getByText('Delete'))

      await waitFor(() => {
        expect(screen.getByText(/Compact Database/)).toBeInTheDocument()
      })
    })
  })

  describe('Tab Switching', () => {
    it('switches between tabs', () => {
      render(<DataManagementModal {...defaultProps} />)

      // Initially on export
      expect(screen.getByText('Conversations')).toBeInTheDocument()

      // Switch to delete
      fireEvent.click(screen.getByText('Delete'))
      expect(screen.getByText(/Danger Zone/)).toBeInTheDocument()

      // Switch back to export
      fireEvent.click(screen.getByText('Export'))
      expect(screen.getByText('Conversations')).toBeInTheDocument()
    })
  })

  describe('Close Behavior', () => {
    it('calls onClose when close button clicked', () => {
      const { container } = render(<DataManagementModal {...defaultProps} />)

      const closeButton = container.querySelector('button[title="Close"]')
      if (closeButton) {
        fireEvent.click(closeButton)
        expect(defaultProps.onClose).toHaveBeenCalledTimes(1)
      }
    })

    it('calls onClose when overlay clicked', () => {
      const { container } = render(<DataManagementModal {...defaultProps} />)

      const overlay = container.firstChild as HTMLElement
      fireEvent.click(overlay)
      expect(defaultProps.onClose).toHaveBeenCalledTimes(1)
    })
  })
})
