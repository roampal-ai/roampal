import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MessageGroup } from '../../components/MessageGroup'
import type { Message } from '../../components/MessageThread'

/**
 * MessageGroup Tests
 *
 * Tests the message grouping component that displays
 * user and assistant messages with attachments and citations.
 */

// Helper to create test messages with required fields
const createMessage = (overrides: Partial<Message> = {}): Message => ({
  id: '1',
  content: 'Test content',
  sender: 'user',
  timestamp: new Date(),
  ...overrides,
})

describe('MessageGroup', () => {
  const baseProps = {
    onMemoryClick: vi.fn(),
    onCommandClick: vi.fn(),
    timestamp: new Date(),
  }

  describe('User Messages', () => {
    it('renders user message', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="user"
          messages={[createMessage({ id: '1', content: 'Hello world' })]}
        />
      )
      expect(screen.getByText('Hello world')).toBeInTheDocument()
    })

    it('applies user styling (right aligned)', () => {
      const { container } = render(
        <MessageGroup
          {...baseProps}
          sender="user"
          messages={[createMessage({ id: '1', content: 'Hello world' })]}
        />
      )
      expect(container.querySelector('.justify-end')).toBeInTheDocument()
    })

    it('shows timestamp for user messages', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="user"
          messages={[createMessage({ id: '1', content: 'Hello world' })]}
        />
      )
      // User timestamp appears at bottom right
      const timeElements = screen.getAllByText(/\d+:\d+/)
      expect(timeElements.length).toBeGreaterThan(0)
    })
  })

  describe('Assistant Messages', () => {
    it('renders assistant message', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({ id: '1', content: 'Hello user' })]}
        />
      )
      expect(screen.getByText('Hello user')).toBeInTheDocument()
    })

    it('shows Roampal header', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({ id: '1', content: 'Hello user' })]}
        />
      )
      expect(screen.getByText('Roampal')).toBeInTheDocument()
    })

    it('shows R avatar', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({ id: '1', content: 'Hello user' })]}
        />
      )
      expect(screen.getByText('R')).toBeInTheDocument()
    })

    it('applies assistant styling (left aligned)', () => {
      const { container } = render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({ id: '1', content: 'Hello user' })]}
        />
      )
      expect(container.querySelector('.justify-start')).toBeInTheDocument()
    })
  })

  describe('System Messages', () => {
    it('renders system message as centered badge', () => {
      const { container } = render(
        <MessageGroup
          {...baseProps}
          sender="system"
          messages={[createMessage({ id: '1', content: 'Session started' })]}
        />
      )
      expect(screen.getByText('Session started')).toBeInTheDocument()
      expect(container.querySelector('.justify-center')).toBeInTheDocument()
    })

    it('applies minimal badge styling', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="system"
          messages={[createMessage({ id: '1', content: 'Session started' })]}
        />
      )
      const badge = screen.getByText('Session started')
      expect(badge).toHaveClass('text-xs', 'text-zinc-500')
    })
  })

  describe('Multiple Messages', () => {
    it('renders multiple messages in a group', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[
            createMessage({ id: '1', content: 'First message' }),
            createMessage({ id: '2', content: 'Second message' }),
          ]}
        />
      )
      expect(screen.getByText('First message')).toBeInTheDocument()
      expect(screen.getByText('Second message')).toBeInTheDocument()
    })
  })

  describe('Attachments', () => {
    it('renders attachment badges', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="user"
          messages={[createMessage({
            id: '1',
            content: 'Check this file',
            attachments: [
              { id: 'a1', name: 'document.pdf', size: 1024, type: 'application/pdf' },
            ],
          })]}
        />
      )
      expect(screen.getByText('document.pdf')).toBeInTheDocument()
    })

    it('shows file size', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="user"
          messages={[createMessage({
            id: '1',
            content: 'Check this file',
            attachments: [
              { id: 'a1', name: 'document.pdf', size: 1024, type: 'application/pdf' },
            ],
          })]}
        />
      )
      expect(screen.getByText('1.0KB')).toBeInTheDocument()
    })

    it('shows attachment icon', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="user"
          messages={[createMessage({
            id: '1',
            content: 'Check this file',
            attachments: [
              { id: 'a1', name: 'document.pdf', size: 500, type: 'application/pdf' },
            ],
          })]}
        />
      )
      expect(screen.getByText('📎')).toBeInTheDocument()
    })
  })

  describe('Citations', () => {
    it('renders citations section', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({
            id: '1',
            content: 'Based on memory',
            citations: [
              { citation_id: 1, source: 'Memory Source', confidence: 0.9 },
            ],
          })]}
        />
      )
      expect(screen.getByText('Sources:')).toBeInTheDocument()
    })

    it('shows citation source', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({
            id: '1',
            content: 'Based on memory',
            citations: [
              { citation_id: 1, source: 'Important Memory', confidence: 0.9 },
            ],
          })]}
        />
      )
      expect(screen.getByText(/Important Memory/)).toBeInTheDocument()
    })

    it('shows confidence percentage', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({
            id: '1',
            content: 'Based on memory',
            citations: [
              { citation_id: 1, source: 'Memory', confidence: 0.85 },
            ],
          })]}
        />
      )
      expect(screen.getByText('85% match')).toBeInTheDocument()
    })

    it('shows collection name', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({
            id: '1',
            content: 'Based on memory',
            citations: [
              { citation_id: 1, source: 'Memory', confidence: 0.9, collection: 'books' },
            ],
          })]}
        />
      )
      expect(screen.getByText(/\(books\)/)).toBeInTheDocument()
    })
  })

  describe('Code Blocks', () => {
    it('renders code blocks', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({
            id: '1',
            content: '```javascript\nconst x = 1;\n```',
          })]}
        />
      )
      expect(screen.getByText('const x = 1;')).toBeInTheDocument()
    })
  })

  describe('Command Handling', () => {
    it('renders commands as badges', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({
            id: '1',
            content: '/help\nUse this command',
          })]}
        />
      )
      expect(screen.getByText('/help')).toBeInTheDocument()
    })

    it('calls onCommandClick when command clicked', () => {
      const onCommandClick = vi.fn()
      render(
        <MessageGroup
          {...baseProps}
          onCommandClick={onCommandClick}
          sender="assistant"
          messages={[createMessage({
            id: '1',
            content: '/help\nUse this command',
          })]}
        />
      )

      fireEvent.click(screen.getByText('/help'))
      expect(onCommandClick).toHaveBeenCalledWith('/help')
    })
  })

  describe('Bold Text', () => {
    it('renders bold text with strong tags', () => {
      render(
        <MessageGroup
          {...baseProps}
          sender="assistant"
          messages={[createMessage({
            id: '1',
            content: 'This is **important** text',
          })]}
        />
      )
      expect(screen.getByText('important')).toHaveClass('font-semibold')
    })
  })
})
