import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryCitation } from '../../components/MemoryCitation'

/**
 * MemoryCitation Tests
 *
 * Tests the memory citation display component that shows
 * inline citations and expandable citation lists.
 */

describe('MemoryCitation', () => {
  const mockCitations = [
    {
      citation_id: 1,
      source: 'Test Book',
      confidence: 0.95,
      collection: 'books' as const,
      text: 'Some citation text',
    },
    {
      citation_id: 2,
      source: 'History Entry',
      confidence: 0.75,
      collection: 'history' as const,
    },
  ]

  describe('No Citations', () => {
    it('renders message without citations when none provided', () => {
      render(<MemoryCitation message="Hello world" citations={[]} />)
      expect(screen.getByText('Hello world')).toBeInTheDocument()
    })

    it('does not show expand button when no citations', () => {
      render(<MemoryCitation message="Hello world" citations={[]} />)
      expect(screen.queryByText(/Used/)).not.toBeInTheDocument()
    })
  })

  describe('With Citations', () => {
    it('renders message with citation markers', () => {
      const message = 'This is a fact [1] and another [2]'
      render(<MemoryCitation message={message} citations={mockCitations} />)
      // The message content should be in the document
      expect(screen.getByText(/This is a fact/)).toBeInTheDocument()
    })

    it('shows memory count in expand button', () => {
      render(<MemoryCitation message="Test [1] [2]" citations={mockCitations} />)
      expect(screen.getByText(/Used 2 memories/)).toBeInTheDocument()
    })

    it('shows singular "memory" for single citation', () => {
      render(
        <MemoryCitation
          message="Test [1]"
          citations={[mockCitations[0]]}
        />
      )
      expect(screen.getByText(/Used 1 memory/)).toBeInTheDocument()
    })
  })

  describe('Expand/Collapse', () => {
    it('expands citations when button clicked', () => {
      render(<MemoryCitation message="Test [1]" citations={[mockCitations[0]]} />)

      const expandButton = screen.getByText(/Used 1 memory/)
      fireEvent.click(expandButton)

      // Should show source after expanding
      expect(screen.getByText('Test Book')).toBeInTheDocument()
    })

    it('shows confidence percentage', () => {
      render(<MemoryCitation message="Test [1]" citations={[mockCitations[0]]} />)

      fireEvent.click(screen.getByText(/Used 1 memory/))

      expect(screen.getByText('95% confidence')).toBeInTheDocument()
    })

    it('shows collection name', () => {
      render(<MemoryCitation message="Test [1]" citations={[mockCitations[0]]} />)

      fireEvent.click(screen.getByText(/Used 1 memory/))

      expect(screen.getByText(/Collection: books/)).toBeInTheDocument()
    })

    it('shows citation text when available', () => {
      render(<MemoryCitation message="Test [1]" citations={[mockCitations[0]]} />)

      fireEvent.click(screen.getByText(/Used 1 memory/))

      expect(screen.getByText(/"Some citation text"/)).toBeInTheDocument()
    })
  })

  describe('Confidence Colors', () => {
    it('shows green for high confidence (>=0.9)', () => {
      const highConfCitation = { ...mockCitations[0], confidence: 0.95 }
      render(<MemoryCitation message="Test [1]" citations={[highConfCitation]} />)

      fireEvent.click(screen.getByText(/Used 1 memory/))

      const confidence = screen.getByText('95% confidence')
      expect(confidence).toHaveClass('text-green-500')
    })

    it('shows yellow for medium confidence (>=0.7)', () => {
      const medConfCitation = { ...mockCitations[0], confidence: 0.75 }
      render(<MemoryCitation message="Test [1]" citations={[medConfCitation]} />)

      fireEvent.click(screen.getByText(/Used 1 memory/))

      const confidence = screen.getByText('75% confidence')
      expect(confidence).toHaveClass('text-yellow-500')
    })

    it('shows orange for low confidence (<0.7)', () => {
      const lowConfCitation = { ...mockCitations[0], confidence: 0.5 }
      render(<MemoryCitation message="Test [1]" citations={[lowConfCitation]} />)

      fireEvent.click(screen.getByText(/Used 1 memory/))

      const confidence = screen.getByText('50% confidence')
      expect(confidence).toHaveClass('text-orange-500')
    })
  })

  describe('Collection Icons', () => {
    it('renders without crashing for books collection', () => {
      const booksCitation = { ...mockCitations[0], collection: 'books' as const }
      const { container } = render(
        <MemoryCitation message="Test [1]" citations={[booksCitation]} />
      )
      expect(container.firstChild).not.toBeNull()
    })

    it('renders without crashing for working collection', () => {
      const workingCitation = { ...mockCitations[0], collection: 'working' as const }
      const { container } = render(
        <MemoryCitation message="Test [1]" citations={[workingCitation]} />
      )
      expect(container.firstChild).not.toBeNull()
    })

    it('renders without crashing for history collection', () => {
      const historyCitation = { ...mockCitations[0], collection: 'history' as const }
      const { container } = render(
        <MemoryCitation message="Test [1]" citations={[historyCitation]} />
      )
      expect(container.firstChild).not.toBeNull()
    })

    it('renders without crashing for patterns collection', () => {
      const patternsCitation = { ...mockCitations[0], collection: 'patterns' as const }
      const { container } = render(
        <MemoryCitation message="Test [1]" citations={[patternsCitation]} />
      )
      expect(container.firstChild).not.toBeNull()
    })
  })

  describe('Long Text Truncation', () => {
    it('truncates long citation text', () => {
      const longText = 'A'.repeat(300)
      const longCitation = { ...mockCitations[0], text: longText }
      render(<MemoryCitation message="Test [1]" citations={[longCitation]} />)

      fireEvent.click(screen.getByText(/Used 1 memory/))

      // Should show truncated text with ellipsis
      expect(screen.getByText(/\.\.\."/)).toBeInTheDocument()
    })
  })
})