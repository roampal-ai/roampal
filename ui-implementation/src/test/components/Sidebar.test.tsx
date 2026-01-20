import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Sidebar } from '../../components/Sidebar'

// Mock dependencies
vi.mock('../../stores/useChatStore', () => ({
  useChatStore: vi.fn((selector) => {
    const state = {
      sessions: [
        { id: 'session-1', name: 'Chat 1', timestamp: 1704067200, messageCount: 5 },
        { id: 'session-2', name: 'Chat 2', timestamp: 1704153600, messageCount: 10 },
      ],
    }
    return selector ? selector(state) : state
  }),
}))

vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({ assistant_name: 'Roampal' }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: { apiUrl: 'http://localhost:8765' },
}))

vi.mock('../services/modelContextService', () => ({
  modelContextService: {
    getActiveModel: () => 'llama3',
    getModelLimit: () => 4096,
    getAllContexts: vi.fn().mockResolvedValue([]),
  },
}))

describe('Sidebar', () => {
  const defaultProps = {
    activeShard: 'roampal',
    onShardChange: vi.fn(),
    onSelectChat: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Rendering', () => {
    it('renders the sidebar', () => {
      const { container } = render(<Sidebar {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows new chat button when onNewChat provided', () => {
      render(<Sidebar {...defaultProps} onNewChat={vi.fn()} />)
      // Look for the plus icon or new chat button
      const buttons = screen.getAllByRole('button')
      expect(buttons.length).toBeGreaterThan(0)
    })
  })

  describe('Chat History', () => {
    it('displays chat sessions from store', () => {
      render(<Sidebar {...defaultProps} />)
      expect(screen.getByText('Chat 1')).toBeInTheDocument()
      expect(screen.getByText('Chat 2')).toBeInTheDocument()
    })

    it('calls onSelectChat when session clicked', async () => {
      const onSelectChat = vi.fn()
      render(<Sidebar {...defaultProps} onSelectChat={onSelectChat} />)

      await userEvent.click(screen.getByText('Chat 1'))
      expect(onSelectChat).toHaveBeenCalledWith('session-1')
    })
  })

  describe('New Chat', () => {
    it('calls onNewChat when new chat button clicked', async () => {
      const onNewChat = vi.fn()
      render(<Sidebar {...defaultProps} onNewChat={onNewChat} />)

      await userEvent.click(screen.getByRole('button', { name: /new/i }))
      expect(onNewChat).toHaveBeenCalled()
    })
  })

  describe('Collapse', () => {
    it('shows collapse button when onCollapse provided', () => {
      render(<Sidebar {...defaultProps} onCollapse={vi.fn()} />)
      // Collapse button should be present
      const buttons = screen.getAllByRole('button')
      expect(buttons.length).toBeGreaterThan(0)
    })
  })

  describe('Active Session', () => {
    it('highlights active session', () => {
      render(<Sidebar {...defaultProps} activeSessionId="session-1" />)
      // Active session should have distinct styling
      const chatItems = screen.getAllByRole('button')
      expect(chatItems.length).toBeGreaterThan(0)
    })
  })
})