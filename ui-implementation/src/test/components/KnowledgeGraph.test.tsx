import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render } from '@testing-library/react'
import KnowledgeGraph from '../../components/KnowledgeGraph'

/**
 * KnowledgeGraph Tests
 *
 * Tests the knowledge graph visualization component.
 * Note: Complex D3 visualization tests are limited due to canvas/svg rendering.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({
      nodes: [
        { id: 'n1', label: 'Testing', type: 'concept', usage_count: 10, success_rate: 0.85 },
        { id: 'n2', label: 'Development', type: 'concept', usage_count: 5, success_rate: 0.9 },
      ],
      edges: [
        { source: 'n1', target: 'n2', weight: 5 },
      ],
    }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('KnowledgeGraph', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Initial Render', () => {
    it('renders the component', () => {
      const { container } = render(<KnowledgeGraph />)
      expect(container.firstChild).not.toBeNull()
    })

    it('accepts searchQuery prop', () => {
      const { container } = render(<KnowledgeGraph searchQuery="test" />)
      expect(container.firstChild).not.toBeNull()
    })
  })
})
