import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render } from '@testing-library/react'
import MemoryPanelV2 from '../../components/MemoryPanelV2'

/**
 * MemoryPanelV2 Tests
 *
 * Tests the memory panel v2 component.
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

    it('accepts onMemoryClick prop', () => {
      const { container } = render(<MemoryPanelV2 {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })
  })
})
