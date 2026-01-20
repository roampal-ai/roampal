import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MessageThread } from '../../components/MessageThread'

/**
 * MessageThread Tests
 *
 * Tests the message thread component that groups and displays messages.
 */

// Mock MessageGroup to simplify testing
vi.mock('../../components/MessageGroup', () => ({
  MessageGroup: ({ messages, sender }: { messages: any[]; sender: string }) => (
    <div data-testid={`group-${sender}`}>
      {messages.map((m: any) => (
        <div key={m.id}>{m.content}</div>
      ))}
    </div>
  ),
}))

describe('MessageThread', () => {
  const defaultProps = {
    onMemoryClick: vi.fn(),
    onCommandClick: vi.fn(),
  }

  describe('Empty State', () => {
    it('renders empty thread', () => {
      const { container } = render(
        <MessageThread {...defaultProps} messages={[]} />
      )
      expect(container.firstChild).not.toBeNull()
    })

    it('includes scroll anchor', () => {
      const { container } = render(
        <MessageThread {...defaultProps} messages={[]} />
      )
      expect(container.querySelector('#messages-end')).toBeInTheDocument()
    })
  })

  describe('Single Message', () => {
    it('renders single message', () => {
      const messages = [
        { id: '1', sender: 'user' as const, content: 'Hello', timestamp: new Date() },
      ]

      render(<MessageThread {...defaultProps} messages={messages} />)

      expect(screen.getByText('Hello')).toBeInTheDocument()
    })
  })

  describe('Message Grouping', () => {
    it('groups consecutive user messages', () => {
      const now = new Date()
      const messages = [
        { id: '1', sender: 'user' as const, content: 'Hello', timestamp: now },
        { id: '2', sender: 'user' as const, content: 'World', timestamp: now },
      ]

      const { container } = render(
        <MessageThread {...defaultProps} messages={messages} />
      )

      // Should only have one user group
      const groups = container.querySelectorAll('[data-testid="group-user"]')
      expect(groups.length).toBe(1)
    })

    it('separates user and assistant messages', () => {
      const now = new Date()
      const messages = [
        { id: '1', sender: 'user' as const, content: 'Hello', timestamp: now },
        { id: '2', sender: 'assistant' as const, content: 'Hi there', timestamp: now },
      ]

      const { container } = render(
        <MessageThread {...defaultProps} messages={messages} />
      )

      expect(container.querySelector('[data-testid="group-user"]')).toBeInTheDocument()
      expect(container.querySelector('[data-testid="group-assistant"]')).toBeInTheDocument()
    })

    it('creates new group after 5 minute gap', () => {
      const now = new Date()
      const sixMinutesLater = new Date(now.getTime() + 6 * 60 * 1000)
      const messages = [
        { id: '1', sender: 'user' as const, content: 'Hello', timestamp: now },
        { id: '2', sender: 'user' as const, content: 'Delayed', timestamp: sixMinutesLater },
      ]

      const { container } = render(
        <MessageThread {...defaultProps} messages={messages} />
      )

      // Should have two user groups due to time gap
      const groups = container.querySelectorAll('[data-testid="group-user"]')
      expect(groups.length).toBe(2)
    })
  })

  describe('Content Rendering', () => {
    it('renders all messages', () => {
      const now = new Date()
      const messages = [
        { id: '1', sender: 'user' as const, content: 'First', timestamp: now },
        { id: '2', sender: 'assistant' as const, content: 'Second', timestamp: now },
        { id: '3', sender: 'user' as const, content: 'Third', timestamp: now },
      ]

      render(<MessageThread {...defaultProps} messages={messages} />)

      expect(screen.getByText('First')).toBeInTheDocument()
      expect(screen.getByText('Second')).toBeInTheDocument()
      expect(screen.getByText('Third')).toBeInTheDocument()
    })
  })

  describe('System Messages', () => {
    it('renders system messages in own group', () => {
      const now = new Date()
      const messages = [
        { id: '1', sender: 'system' as const, content: 'Session started', timestamp: now },
      ]

      const { container } = render(
        <MessageThread {...defaultProps} messages={messages} />
      )

      expect(container.querySelector('[data-testid="group-system"]')).toBeInTheDocument()
    })
  })
})
