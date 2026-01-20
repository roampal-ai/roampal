import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { EnhancedChatMessage } from '../../components/EnhancedChatMessage'

/**
 * EnhancedChatMessage Tests
 *
 * Tests the enhanced chat message component with user/assistant display.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({
      content: 'identity:\n  name: "TestBot"',
    }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

vi.mock('../../components/EnhancedMessageDisplay', () => ({
  EnhancedMessageDisplay: ({ content }: { content: string }) => (
    <div data-testid="enhanced-display">{content}</div>
  ),
}))

vi.mock('../../components/ToolExecutionDisplay', () => ({
  ToolExecutionDisplay: ({ executions }: { executions: any[] }) => (
    <div data-testid="tool-display">{executions.length} tools</div>
  ),
}))

describe('EnhancedChatMessage', () => {
  const baseMessage = {
    id: 'msg-1',
    sender: 'user' as const,
    content: 'Hello',
    timestamp: new Date(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('User Messages', () => {
    it('renders user message', () => {
      render(<EnhancedChatMessage message={baseMessage} />)
      expect(screen.getByText('Hello')).toBeInTheDocument()
    })

    it('shows "You" as sender name', () => {
      render(<EnhancedChatMessage message={baseMessage} />)
      expect(screen.getByText('You')).toBeInTheDocument()
    })

    it('has user styling', () => {
      const { container } = render(<EnhancedChatMessage message={baseMessage} />)
      expect(container.querySelector('.bg-cyan-950\\/20')).toBeInTheDocument()
    })

    it('shows timestamp', () => {
      const message = { ...baseMessage, timestamp: new Date('2024-01-01T14:30:00') }
      render(<EnhancedChatMessage message={message} />)
      // Timestamp format depends on locale
      expect(screen.getByText(/\d{1,2}:\d{2}/)).toBeInTheDocument()
    })
  })

  describe('Assistant Messages', () => {
    const assistantMessage = {
      ...baseMessage,
      sender: 'assistant' as const,
      content: 'Hello, how can I help?',
    }

    it('renders assistant message', () => {
      render(<EnhancedChatMessage message={assistantMessage} />)
      expect(screen.getByTestId('enhanced-display')).toBeInTheDocument()
    })

    it('shows assistant name', async () => {
      render(<EnhancedChatMessage message={assistantMessage} />)

      await waitFor(() => {
        expect(screen.getByText('TestBot')).toBeInTheDocument()
      })
    })

    it('uses EnhancedMessageDisplay for content', () => {
      render(<EnhancedChatMessage message={assistantMessage} />)
      expect(screen.getByTestId('enhanced-display')).toBeInTheDocument()
      expect(screen.getByText('Hello, how can I help?')).toBeInTheDocument()
    })
  })

  describe('Tool Executions', () => {
    const messageWithTools = {
      ...baseMessage,
      sender: 'assistant' as const,
      content: 'Found memories',
      toolExecutions: [
        { tool: 'search', status: 'completed' as const, description: 'Searched memory' },
      ],
    }

    it('shows tool executions', () => {
      render(<EnhancedChatMessage message={messageWithTools} />)
      expect(screen.getByTestId('tool-display')).toBeInTheDocument()
      expect(screen.getByText('1 tools')).toBeInTheDocument()
    })
  })

  describe('Streaming State', () => {
    it('shows typing indicator when streaming', () => {
      const streamingMessage = {
        ...baseMessage,
        sender: 'assistant' as const,
        streaming: true,
      }
      render(<EnhancedChatMessage message={streamingMessage} />)
      expect(screen.getByText('• Typing')).toBeInTheDocument()
    })
  })

  describe('Model Name', () => {
    it('shows model name badge for assistant', () => {
      const messageWithModel = {
        ...baseMessage,
        sender: 'assistant' as const,
        content: 'Response',
        metadata: { model_name: 'llama3:latest' },
      }
      render(<EnhancedChatMessage message={messageWithModel} />)
      expect(screen.getByText('llama3')).toBeInTheDocument()
    })
  })

  describe('Avatar', () => {
    it('shows user avatar', () => {
      const { container } = render(<EnhancedChatMessage message={baseMessage} />)
      expect(container.querySelector('.bg-blue-600')).toBeInTheDocument()
    })

    it('shows assistant avatar', () => {
      const assistantMessage = { ...baseMessage, sender: 'assistant' as const }
      const { container } = render(<EnhancedChatMessage message={assistantMessage} />)
      expect(container.querySelector('.bg-zinc-800')).toBeInTheDocument()
    })
  })
})
