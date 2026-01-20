import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ConversationBadges } from '../../components/ConversationBadges'

/**
 * ConversationBadges Tests
 *
 * Tests the conversation list sidebar component.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockRejectedValue(new Error('API not available')),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('ConversationBadges', () => {
  const defaultProps = {
    onSelectConversation: vi.fn(),
    onDeleteConversation: vi.fn(),
    onNewConversation: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Initial Render', () => {
    it('renders the component', () => {
      const { container } = render(<ConversationBadges {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows conversations header', () => {
      render(<ConversationBadges {...defaultProps} />)
      expect(screen.getByText('Conversations')).toBeInTheDocument()
    })

    it('shows new conversation button', () => {
      render(<ConversationBadges {...defaultProps} />)
      expect(screen.getByText('+ New')).toBeInTheDocument()
    })

    it('shows loading spinner initially', () => {
      const { container } = render(<ConversationBadges {...defaultProps} />)
      // Loading spinner has animate-spin class
      expect(container.querySelector('.animate-spin')).toBeInTheDocument()
    })
  })

  describe('After Loading', () => {
    it('shows mock conversations after loading', async () => {
      render(<ConversationBadges {...defaultProps} />)

      // Wait for loading to complete (mock data loads)
      await waitFor(() => {
        expect(screen.getByText('API Optimization')).toBeInTheDocument()
      })
    })

    it('shows multiple conversations', async () => {
      render(<ConversationBadges {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('API Optimization')).toBeInTheDocument()
        expect(screen.getByText('Debug Session')).toBeInTheDocument()
        expect(screen.getByText('Code Review')).toBeInTheDocument()
      })
    })

    it('shows message counts', async () => {
      render(<ConversationBadges {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('12')).toBeInTheDocument() // API Optimization
      })
    })

    it('shows preview text', async () => {
      render(<ConversationBadges {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('Implementing caching strategy...')).toBeInTheDocument()
      })
    })
  })

  describe('Interactions', () => {
    it('calls onNewConversation when clicking new button', () => {
      render(<ConversationBadges {...defaultProps} />)

      fireEvent.click(screen.getByText('+ New'))

      expect(defaultProps.onNewConversation).toHaveBeenCalledTimes(1)
    })

    it('calls onSelectConversation when clicking a conversation', async () => {
      render(<ConversationBadges {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('API Optimization')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByText('API Optimization'))

      expect(defaultProps.onSelectConversation).toHaveBeenCalledWith('c1')
    })
  })

  describe('Active Conversation', () => {
    it('highlights active conversation', async () => {
      render(
        <ConversationBadges
          {...defaultProps}
          activeConversationId="c1"
        />
      )

      await waitFor(() => {
        expect(screen.getByText('API Optimization')).toBeInTheDocument()
      })

      // Active conversation has different styling with blue color
      const title = screen.getByText('API Optimization')
      expect(title).toHaveClass('text-blue-400')
    })

    it('shows pulse indicator for active conversation', async () => {
      const { container } = render(
        <ConversationBadges
          {...defaultProps}
          activeConversationId="c1"
        />
      )

      await waitFor(() => {
        expect(screen.getByText('API Optimization')).toBeInTheDocument()
      })

      // Should have pulse animation for active indicator
      const pulseIndicators = container.querySelectorAll('.animate-pulse')
      expect(pulseIndicators.length).toBeGreaterThan(0)
    })
  })

  describe('Footer Stats', () => {
    it('shows conversation count', async () => {
      render(<ConversationBadges {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText(/3 conversations/)).toBeInTheDocument()
      })
    })

    it('shows total message count', async () => {
      render(<ConversationBadges {...defaultProps} />)

      // 12 + 8 + 15 = 35 total messages
      await waitFor(() => {
        expect(screen.getByText(/35 total messages/)).toBeInTheDocument()
      })
    })
  })
})
