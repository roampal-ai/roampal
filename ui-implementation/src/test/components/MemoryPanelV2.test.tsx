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

    it('shows tag counts in the cloud', () => {
      const memories = [
        makeMemory({ tags: ['python'] }),
        makeMemory({ tags: ['python'] }),
        makeMemory({ tags: ['rust'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)
      // python count = 2 should appear somewhere
      expect(screen.getByText('2')).toBeInTheDocument()
    })
  })

  describe('Tag Filtering', () => {
    it('filters memories when tag is clicked in cloud', () => {
      const memories = [
        makeMemory({ id: 'mem-1', text: 'Python memory', content: 'Python memory', tags: ['python'] }),
        makeMemory({ id: 'mem-2', text: 'Rust memory', content: 'Rust memory', tags: ['rust'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      // Both memories visible
      expect(screen.getByText('Python memory')).toBeInTheDocument()
      expect(screen.getByText('Rust memory')).toBeInTheDocument()

      // Click python tag in the cloud (first button matching "python")
      const tagButtons = screen.getAllByRole('button')
      const pythonButton = tagButtons.find(btn => btn.textContent?.includes('python'))
      if (pythonButton) fireEvent.click(pythonButton)

      // Only Python memory should be visible
      expect(screen.getByText('Python memory')).toBeInTheDocument()
      expect(screen.queryByText('Rust memory')).not.toBeInTheDocument()
    })

    it('shows Clear button when tags selected', () => {
      const memories = [
        makeMemory({ tags: ['python'] }),
        makeMemory({ tags: ['rust'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      // No Clear initially
      expect(screen.queryByText('Clear')).not.toBeInTheDocument()

      // Click a tag
      const tagButtons = screen.getAllByRole('button')
      const pythonButton = tagButtons.find(btn => btn.textContent?.includes('python'))
      if (pythonButton) fireEvent.click(pythonButton)

      // Clear should appear
      expect(screen.getByText('Clear')).toBeInTheDocument()
    })

    it('clears tag filter when Clear clicked', () => {
      const memories = [
        makeMemory({ id: 'mem-1', text: 'Python memory', content: 'Python memory', tags: ['python'] }),
        makeMemory({ id: 'mem-2', text: 'Rust memory', content: 'Rust memory', tags: ['rust'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      // Click python tag
      const tagButtons = screen.getAllByRole('button')
      const pythonButton = tagButtons.find(btn => btn.textContent?.includes('python'))
      if (pythonButton) fireEvent.click(pythonButton)

      // Only Python visible
      expect(screen.queryByText('Rust memory')).not.toBeInTheDocument()

      // Click Clear
      fireEvent.click(screen.getByText('Clear'))

      // Both visible again
      expect(screen.getByText('Python memory')).toBeInTheDocument()
      expect(screen.getByText('Rust memory')).toBeInTheDocument()
    })

    it('applies AND logic for multiple tags', () => {
      const memories = [
        makeMemory({ id: 'mem-1', text: 'Both tags', content: 'Both tags', tags: ['python', 'api'] }),
        makeMemory({ id: 'mem-2', text: 'Python only', content: 'Python only', tags: ['python'] }),
        makeMemory({ id: 'mem-3', text: 'API only', content: 'API only', tags: ['api'] }),
      ]
      render(<MemoryPanelV2 {...defaultProps} memories={memories} />)

      // Click python tag
      const tagButtons = screen.getAllByRole('button')
      const pythonButton = tagButtons.find(btn => btn.textContent?.includes('python'))
      if (pythonButton) fireEvent.click(pythonButton)

      // Click api tag on the memory card (it has an onClick handler too)
      const apiButton = screen.getAllByRole('button').find(btn => btn.textContent?.includes('api'))
      if (apiButton) fireEvent.click(apiButton)

      // Only "Both tags" should remain (has both python AND api)
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
