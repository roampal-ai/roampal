import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { EnhancedMessageDisplay } from '../../components/EnhancedMessageDisplay'

/**
 * EnhancedMessageDisplay Tests
 *
 * Tests the enhanced message display with markdown and citations.
 */

// Mock clipboard
Object.assign(navigator, {
  clipboard: {
    writeText: vi.fn().mockResolvedValue(undefined),
  },
})

describe('EnhancedMessageDisplay', () => {
  describe('Content Rendering', () => {
    it('renders text content', () => {
      render(<EnhancedMessageDisplay content="Hello world" />)
      expect(screen.getByText('Hello world')).toBeInTheDocument()
    })

    it('renders markdown content', () => {
      render(<EnhancedMessageDisplay content="**Bold text**" />)
      const boldElement = screen.getByText('Bold text')
      expect(boldElement.tagName).toBe('STRONG')
    })

    it('renders inline code', () => {
      render(<EnhancedMessageDisplay content="Use `console.log()` for debugging" />)
      expect(screen.getByText('console.log()')).toBeInTheDocument()
    })
  })

  describe('Citations', () => {
    const mockCitations = [
      {
        citation_id: 1,
        source: 'Memory Bank',
        confidence: 0.95,
        collection: 'memory_bank',
        text: 'Citation text',
      },
    ]

    it('shows citation button when citations provided', () => {
      render(
        <EnhancedMessageDisplay
          content="Content with citation"
          citations={mockCitations}
        />
      )
      expect(screen.getByText('1 Citation')).toBeInTheDocument()
    })

    it('expands citations when clicked', () => {
      render(
        <EnhancedMessageDisplay
          content="Content with citation"
          citations={mockCitations}
        />
      )

      fireEvent.click(screen.getByText('1 Citation'))

      // After expanding, should show citation info
      expect(screen.getByText(/Memory Bank/)).toBeInTheDocument()
    })

    it('shows citation text', () => {
      render(
        <EnhancedMessageDisplay
          content="Content with citation"
          citations={mockCitations}
        />
      )

      fireEvent.click(screen.getByText('1 Citation'))

      expect(screen.getByText('Citation text')).toBeInTheDocument()
    })

    it('shows plural text for multiple citations', () => {
      const multipleCitations = [
        { citation_id: 1, source: 'Source 1', confidence: 0.9, collection: 'books' },
        { citation_id: 2, source: 'Source 2', confidence: 0.8, collection: 'history' },
      ]

      render(
        <EnhancedMessageDisplay
          content="Content"
          citations={multipleCitations}
        />
      )

      expect(screen.getByText('2 Citations')).toBeInTheDocument()
    })

    it('does not show citations button when no citations', () => {
      render(<EnhancedMessageDisplay content="Content" />)
      expect(screen.queryByText(/Citation/)).not.toBeInTheDocument()
    })
  })

  describe('Code Blocks', () => {
    it('renders code blocks with syntax highlighting', () => {
      const codeContent = '```javascript\nconst x = 1;\n```'
      const { container } = render(<EnhancedMessageDisplay content={codeContent} />)

      // The code block should be rendered (ReactMarkdown handles this)
      expect(container.querySelector('code')).toBeInTheDocument()
    })

    it('renders code content', () => {
      const codeContent = '```javascript\nconst x = 1;\n```'
      const { container } = render(<EnhancedMessageDisplay content={codeContent} />)

      // Code element should exist
      expect(container.querySelector('code')).toBeInTheDocument()
    })
  })

  describe('Structure', () => {
    it('renders with prose styling', () => {
      const { container } = render(<EnhancedMessageDisplay content="Hello" />)
      expect(container.querySelector('.prose')).toBeInTheDocument()
    })

    it('renders with break-words class', () => {
      const { container } = render(<EnhancedMessageDisplay content="Hello" />)
      expect(container.querySelector('.break-words')).toBeInTheDocument()
    })
  })
})
