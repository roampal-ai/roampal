import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ModelContextSettings } from '../../components/ModelContextSettings'

/**
 * ModelContextSettings Tests
 *
 * Tests the context window configuration panel.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({
      models: [
        { name: 'llama3:latest', provider: 'ollama' },
      ],
    }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

vi.mock('../services/modelContextService', () => ({
  modelContextService: {
    getModelContext: vi.fn().mockResolvedValue({
      default: 4096,
      max: 8192,
      current: 4096,
      is_override: false,
    }),
  },
}))

describe('ModelContextSettings', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Closed State', () => {
    it('returns null when closed', () => {
      const { container } = render(<ModelContextSettings {...defaultProps} isOpen={false} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Open State', () => {
    it('renders when open', () => {
      const { container } = render(<ModelContextSettings {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows Context Window Settings title', () => {
      render(<ModelContextSettings {...defaultProps} />)
      expect(screen.getByText('Context Window Settings')).toBeInTheDocument()
    })
  })

  describe('Close Behavior', () => {
    it('calls onClose when close button clicked', () => {
      const { container } = render(<ModelContextSettings {...defaultProps} />)

      const closeButton = container.querySelector('button')
      if (closeButton) {
        closeButton.click()
      }

      expect(defaultProps.onClose).toHaveBeenCalled()
    })
  })
})
