import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryBankModal } from '../../components/MemoryBankModal'

/**
 * MemoryBankModal Tests
 *
 * Tests the memory bank management modal.
 * Note: Uses virtualization (react-window) - some tests are structural.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({
      memories: [],
      archived: [],
      stats: { total_memories: 0, active: 0, archived: 0, unique_tags: 0, tags: [] },
    }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('MemoryBankModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Closed State', () => {
    it('returns null when closed', () => {
      const { container } = render(<MemoryBankModal {...defaultProps} isOpen={false} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Open State', () => {
    it('renders when open', () => {
      const { container } = render(<MemoryBankModal {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows Memory Bank title', () => {
      render(<MemoryBankModal {...defaultProps} />)
      expect(screen.getByText('Memory Bank')).toBeInTheDocument()
    })
  })

  describe('View Tabs', () => {
    it('shows Active tab', () => {
      render(<MemoryBankModal {...defaultProps} />)
      expect(screen.getByText(/Active \(\d+\)/)).toBeInTheDocument()
    })

    it('shows Archived tab', () => {
      render(<MemoryBankModal {...defaultProps} />)
      expect(screen.getByText(/Archived \(\d+\)/)).toBeInTheDocument()
    })

    it('shows Stats tab', () => {
      render(<MemoryBankModal {...defaultProps} />)
      expect(screen.getByText('Stats')).toBeInTheDocument()
    })

    it('switches tabs when clicked', () => {
      render(<MemoryBankModal {...defaultProps} />)

      const archivedTab = screen.getByText(/Archived \(\d+\)/)
      fireEvent.click(archivedTab)

      // Tab should be clickable without error
      expect(archivedTab).toBeInTheDocument()
    })
  })

  describe('Search', () => {
    it('shows search input', () => {
      render(<MemoryBankModal {...defaultProps} />)
      expect(screen.getByPlaceholderText(/Search/)).toBeInTheDocument()
    })
  })
})
