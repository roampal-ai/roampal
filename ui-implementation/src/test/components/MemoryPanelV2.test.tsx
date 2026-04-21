import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import MemoryPanelV2 from '../../components/MemoryPanelV2'

/**
 * MemoryPanelV2 Tests
 *
 * Tests the memory panel v2 component including tag filtering.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({
      fragments: [],
      stats: { total: 0, by_collection: {} },
    }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

// Mock KnowledgeGraph component
vi.mock('../../components/KnowledgeGraph', () => ({
  default: () => <div data-testid="knowledge-graph" />,
}))

const makeMemory = (overrides: Record<string, any> = {}) => ({
  id: `mem-${Math.random().toString(36).slice(2, 8)}`,
  text: 'Test memory content',
  content: 'Test memory content',
  type: 'working',
  timestamp: new Date(),
  score: 0.7,
  tags: [],
  ...overrides,
})

describe('MemoryPanelV2', () => {
  const defaultProps = {
    memories: [],
    knowledgeGraph: { nodes: [], edges: [] },
    onMemoryClick: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Initial Render', () => {
    it('renders the component', () => {
      const { container } = render(<MemoryPanelV2 {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows empty state when no memories', () => {
      render(<MemoryPanelV2 {...defaultProps} />)
      expect(screen.getByText('No memories yet')).toBeInTheDocument()
    })

    it('renders memories', () => {
      const memories = [makeMemory({ text: 'Hello world', content: 'Hello world' })]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)
      expect(screen.getByText('Hello world')).toBeInTheDocument()
    })
  })

  describe('Tag Display', () => {
    it('shows tags on memory cards', () => {
      const memories = [makeMemory({ tags: ['python', 'asyncio'] })]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)
      // Tags appear in both the cloud and the card — just check they exist
      expect(screen.getAllByText('python').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('asyncio').length).toBeGreaterThanOrEqual(1)
    })

    it('limits tags to 4 per card with overflow count', () => {
      const memories = [makeMemory({ tags: ['a', 'b', 'c', 'd', 'e', 'f'] })]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)
      // Should show 4 tags + overflow
      expect(screen.getByText('+2')).toBeInTheDocument()
    })

    it('does not show tag cloud when memories have no tags', () => {
      const memories = [makeMemory({ tags: [] })]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)
      // No Clear button = no tag cloud rendered
      expect(screen.queryByText('Clear')).not.toBeInTheDocument()
    })
  })

  describe('Tag Cloud', () => {
    it('shows tag cloud when memories have tags', () => {
      const memories = [
        makeMemory({ tags: ['python', 'api'] }),
        makeMemory({ tags: ['python', 'testing'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)
      // Tag cloud shows counts — python appears twice
      const pythonButtons = screen.getAllByText(/python/)
      expect(pythonButtons.length).toBeGreaterThan(0)
    })

    it('does not show tag cloud when no tags exist', () => {
      const memories = [makeMemory({ tags: [] }), makeMemory({ tags: [] })]
      const { container } = render(<MemoryPanelV2 {...defaultProps} memories={memories} />)
      // No Clear button should exist
      expect(screen.queryByText('Clear')).not.toBeInTheDocument()
    })

    // v0.3.2: Rewritten for the Substack-style tag input — counts now live
    // in the typeahead dropdown, not a static cloud.
    it('shows tag counts in the typeahead suggestions', () => {
      const memories = [
        makeMemory({ tags: ['python'] }),
        makeMemory({ tags: ['python'] }),
        makeMemory({ tags: ['rust'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)
      const input = screen.getByPlaceholderText('Filter by tag...')
      fireEvent.change(input, { target: { value: 'py' } })
      fireEvent.focus(input)
      // The typeahead dropdown renders a button per matching tag with the count.
      const suggestionButtons = screen.getAllByRole('button').filter(
        btn => btn.textContent?.includes('python') && btn.textContent?.includes('2')
      )
      expect(suggestionButtons.length).toBeGreaterThan(0)
    })
  })

  // v0.3.2: Tag cloud click-to-filter was replaced by Substack-style tag input
  // with typeahead suggestions in v0.3.1. Tests rewritten against the new
  // input/pill UI (MemoryPanelV2.tsx ~L280-327).
  describe('Tag Filtering', () => {
    it('filters memories when a tag suggestion is picked', () => {
      // Need 2+ memories per tag — component hides tags with count < 2.
      const memories = [
        makeMemory({ id: 'mem-1', text: 'Python memory', content: 'Python memory', tags: ['python'] }),
        makeMemory({ id: 'mem-py2', text: 'More python', content: 'More python', tags: ['python'] }),
        makeMemory({ id: 'mem-2', text: 'Rust memory', content: 'Rust memory', tags: ['rust'] }),
        makeMemory({ id: 'mem-rust2', text: 'More rust', content: 'More rust', tags: ['rust'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      // Both memories visible before filtering
      expect(screen.getByText('Python memory')).toBeInTheDocument()
      expect(screen.getByText('Rust memory')).toBeInTheDocument()

      const input = screen.getByPlaceholderText('Filter by tag...')
      fireEvent.change(input, { target: { value: 'py' } })
      fireEvent.focus(input)
      // Click the python suggestion in the typeahead dropdown.
      const suggestion = screen.getAllByRole('button').find(
        btn => btn.textContent?.includes('python') && /\d+/.test(btn.textContent || '')
      )
      if (suggestion) fireEvent.mouseDown(suggestion)

      // Only Python memory should remain visible
      expect(screen.getByText('Python memory')).toBeInTheDocument()
      expect(screen.queryByText('Rust memory')).not.toBeInTheDocument()
    })

    it('shows the Clear X button once a tag pill is added', () => {
      const memories = [
        makeMemory({ tags: ['python'] }),
        makeMemory({ tags: ['python'] }),
        makeMemory({ tags: ['rust'] }),
        makeMemory({ tags: ['rust'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      // Not present before anything is picked.
      expect(screen.queryByTitle('Clear all tags')).not.toBeInTheDocument()

      const input = screen.getByPlaceholderText('Filter by tag...')
      fireEvent.change(input, { target: { value: 'py' } })
      fireEvent.focus(input)
      const suggestion = screen.getAllByRole('button').find(
        btn => btn.textContent?.includes('python') && /\d+/.test(btn.textContent || '')
      )
      if (suggestion) fireEvent.mouseDown(suggestion)

      // X button surfaces once a pill is in place.
      expect(screen.getByTitle('Clear all tags')).toBeInTheDocument()
    })

    it('clears all tag filters when the X button is clicked', () => {
      const memories = [
        makeMemory({ id: 'mem-1', text: 'Python memory', content: 'Python memory', tags: ['python'] }),
        makeMemory({ id: 'mem-py2', text: 'More python', content: 'More python', tags: ['python'] }),
        makeMemory({ id: 'mem-2', text: 'Rust memory', content: 'Rust memory', tags: ['rust'] }),
        makeMemory({ id: 'mem-rust2', text: 'More rust', content: 'More rust', tags: ['rust'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      const input = screen.getByPlaceholderText('Filter by tag...')
      fireEvent.change(input, { target: { value: 'py' } })
      fireEvent.focus(input)
      const suggestion = screen.getAllByRole('button').find(
        btn => btn.textContent?.includes('python') && /\d+/.test(btn.textContent || '')
      )
      if (suggestion) fireEvent.mouseDown(suggestion)

      expect(screen.queryByText('Rust memory')).not.toBeInTheDocument()

      fireEvent.click(screen.getByTitle('Clear all tags'))

      // Both visible again after clearing.
      expect(screen.getByText('Python memory')).toBeInTheDocument()
      expect(screen.getByText('Rust memory')).toBeInTheDocument()
    })

    it('applies AND logic when two tag pills are picked', () => {
      // Need count >= 2 for each tag to show in the typeahead.
      const memories = [
        makeMemory({ id: 'mem-1', text: 'Both tags', content: 'Both tags', tags: ['python', 'api'] }),
        makeMemory({ id: 'mem-1b', text: 'Both again', content: 'Both again', tags: ['python', 'api'] }),
        makeMemory({ id: 'mem-2', text: 'Python only', content: 'Python only', tags: ['python'] }),
        makeMemory({ id: 'mem-3', text: 'API only', content: 'API only', tags: ['api'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      const input = screen.getByPlaceholderText('Filter by tag...')
      fireEvent.change(input, { target: { value: 'py' } })
      fireEvent.focus(input)
      let suggestion = screen.getAllByRole('button').find(
        btn => btn.textContent?.includes('python') && /\d+/.test(btn.textContent || '')
      )
      if (suggestion) fireEvent.mouseDown(suggestion)

      fireEvent.change(input, { target: { value: 'ap' } })
      fireEvent.focus(input)
      suggestion = screen.getAllByRole('button').find(
        btn => btn.textContent?.includes('api') && /\d+/.test(btn.textContent || '')
      )
      if (suggestion) fireEvent.mouseDown(suggestion)

      // Only the memory with BOTH tags should remain.
      expect(screen.getByText('Both tags')).toBeInTheDocument()
      expect(screen.queryByText('Python only')).not.toBeInTheDocument()
      expect(screen.queryByText('API only')).not.toBeInTheDocument()
    })
  })

  describe('Search with Tags', () => {
    it('search matches tag content', () => {
      const memories = [
        makeMemory({ id: 'mem-1', text: 'Some content', content: 'Some content', tags: ['python'] }),
        makeMemory({ id: 'mem-2', text: 'Other content', content: 'Other content', tags: ['rust'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      const searchInput = screen.getByPlaceholderText('Search memories...')
      fireEvent.change(searchInput, { target: { value: 'python' } })

      // Should find memory by tag match even though content doesn't contain "python"
      expect(screen.getByText('Some content')).toBeInTheDocument()
      expect(screen.queryByText('Other content')).not.toBeInTheDocument()
    })
  })

  describe('Type Filter', () => {
    it('filters by collection type', () => {
      const memories = [
        makeMemory({ id: 'mem-1', text: 'Working mem', content: 'Working mem', type: 'working' }),
        makeMemory({ id: 'mem-2', text: 'History mem', content: 'History mem', type: 'history' }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      // Select working filter
      const select = screen.getByDisplayValue('All Types')
      fireEvent.change(select, { target: { value: 'working' } })

      expect(screen.getByText('Working mem')).toBeInTheDocument()
      expect(screen.queryByText('History mem')).not.toBeInTheDocument()
    })
  })

  describe('Memory Detail Modal', () => {
    it('shows tags in detail modal', () => {
      const memories = [makeMemory({
        text: 'Click me',
        content: 'Click me',
        tags: ['python', 'asyncio', 'testing'],
      })]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      // Click memory to open modal
      fireEvent.click(screen.getByText('Click me'))

      // All tags should be visible in modal (not limited to 4)
      expect(screen.getAllByText('python').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('asyncio').length).toBeGreaterThanOrEqual(1)
      expect(screen.getAllByText('testing').length).toBeGreaterThanOrEqual(1)
    })

    it('shows memory_type in detail modal', () => {
      const memories = [makeMemory({
        text: 'Fact memory',
        content: 'Fact memory',
        memory_type: 'fact',
      })]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      fireEvent.click(screen.getByText('Fact memory'))

      expect(screen.getByText('fact')).toBeInTheDocument()
    })
  })
})
