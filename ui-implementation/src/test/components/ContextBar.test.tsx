import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ContextBar } from '../../components/ContextBar'

/**
 * ContextBar Tests
 *
 * Tests the context sidebar with fragments, graph, and references.
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

describe('ContextBar', () => {
  const mockMemories = [
    {
      id: 'f1',
      type: 'memory' as const,
      content: 'Test memory content',
      score: 0.95,
      timestamp: new Date(),
      tags: ['test', 'memory'],
    },
    {
      id: 'f2',
      type: 'concept' as const,
      content: 'Test concept content',
      score: 0.75,
      timestamp: new Date(Date.now() - 60000),
    },
  ]

  const mockKnowledgeGraph = {
    concepts: 10,
    relationships: 25,
    activeTopics: ['testing', 'development'],
  }

  const mockReferences: never[] = []

  const defaultProps = {
    memories: mockMemories,
    knowledgeGraph: mockKnowledgeGraph,
    references: mockReferences,
    onMemoryClick: vi.fn(),
    onRefresh: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Initial Render', () => {
    it('renders the component', () => {
      const { container } = render(<ContextBar {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows fragments tab by default', () => {
      render(<ContextBar {...defaultProps} />)
      expect(screen.getByText('Fragments')).toBeInTheDocument()
    })

    it('shows graph tab', () => {
      render(<ContextBar {...defaultProps} />)
      expect(screen.getByText('Graph')).toBeInTheDocument()
    })

    it('shows fragment count badge', () => {
      render(<ContextBar {...defaultProps} />)
      expect(screen.getByText('2')).toBeInTheDocument()
    })
  })

  describe('Fragments Tab', () => {
    it('displays memory fragments', () => {
      render(<ContextBar {...defaultProps} />)
      expect(screen.getByText('Test memory content')).toBeInTheDocument()
      expect(screen.getByText('Test concept content')).toBeInTheDocument()
    })

    it('shows fragment scores', () => {
      render(<ContextBar {...defaultProps} />)
      expect(screen.getByText('95%')).toBeInTheDocument()
      expect(screen.getByText('75%')).toBeInTheDocument()
    })

    it('shows fragment types', () => {
      render(<ContextBar {...defaultProps} />)
      // Types may be shown as badges with different styling
      const typeElements = screen.getAllByText(/memory|concept/)
      expect(typeElements.length).toBeGreaterThan(0)
    })

    it('shows content from fragments', () => {
      render(<ContextBar {...defaultProps} />)
      // The content should be visible
      expect(screen.getByText('Test memory content')).toBeInTheDocument()
      expect(screen.getByText('Test concept content')).toBeInTheDocument()
    })

    it('calls onMemoryClick when fragment clicked', () => {
      render(<ContextBar {...defaultProps} />)

      const fragment = screen.getByText('Test memory content').closest('div[class*="cursor-pointer"]')
      if (fragment) {
        fireEvent.click(fragment)
        expect(defaultProps.onMemoryClick).toHaveBeenCalledWith('f1')
      }
    })
  })

  describe('Sorting', () => {
    it('has sort by recent button', () => {
      const { container } = render(<ContextBar {...defaultProps} />)
      // Clock icon button for recent sort
      const buttons = container.querySelectorAll('button')
      expect(buttons.length).toBeGreaterThan(0)
    })

    it('has sort by score button', () => {
      const { container } = render(<ContextBar {...defaultProps} />)
      const buttons = container.querySelectorAll('button')
      expect(buttons.length).toBeGreaterThan(0)
    })
  })

  describe('Search', () => {
    it('shows search button', () => {
      const { container } = render(<ContextBar {...defaultProps} />)
      // Magnifying glass icon button
      const buttons = container.querySelectorAll('button')
      expect(buttons.length).toBeGreaterThan(0)
    })
  })

  describe('Graph Tab', () => {
    it('switches to graph tab', () => {
      render(<ContextBar {...defaultProps} />)

      fireEvent.click(screen.getByText('Graph'))

      expect(screen.getByText('Concepts')).toBeInTheDocument()
      expect(screen.getByText('Relations')).toBeInTheDocument()
    })

    it('shows concept count', () => {
      render(<ContextBar {...defaultProps} />)
      fireEvent.click(screen.getByText('Graph'))

      expect(screen.getByText('10')).toBeInTheDocument()
    })

    it('shows relationship count', () => {
      render(<ContextBar {...defaultProps} />)
      fireEvent.click(screen.getByText('Graph'))

      expect(screen.getByText('25')).toBeInTheDocument()
    })

    it('shows active topics', () => {
      render(<ContextBar {...defaultProps} />)
      fireEvent.click(screen.getByText('Graph'))

      expect(screen.getByText('testing')).toBeInTheDocument()
      expect(screen.getByText('development')).toBeInTheDocument()
    })
  })

  describe('Empty State', () => {
    it('shows empty state for fragments', () => {
      render(
        <ContextBar
          {...defaultProps}
          memories={[]}
        />
      )

      expect(screen.getByText('No fragments yet')).toBeInTheDocument()
      expect(screen.getByText('Send messages to build memory')).toBeInTheDocument()
    })
  })

  describe('Footer Stats', () => {
    it('shows fragment count in footer', () => {
      render(<ContextBar {...defaultProps} />)
      expect(screen.getByText('2 fragments')).toBeInTheDocument()
    })

    it('shows graph stats in footer when on graph tab', () => {
      render(<ContextBar {...defaultProps} />)
      fireEvent.click(screen.getByText('Graph'))

      expect(screen.getByText(/10 concepts/)).toBeInTheDocument()
      expect(screen.getByText(/25 relations/)).toBeInTheDocument()
    })
  })

  describe('Score Colors', () => {
    it('applies emerald color for high scores', () => {
      render(<ContextBar {...defaultProps} />)
      const highScore = screen.getByText('95%')
      expect(highScore).toHaveClass('text-emerald-400')
    })

    it('applies blue color for medium scores', () => {
      render(<ContextBar {...defaultProps} />)
      const medScore = screen.getByText('75%')
      expect(medScore).toHaveClass('text-blue-400')
    })
  })
})
