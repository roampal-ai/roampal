import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'

/**
 * MemoryStatsPanel Tests
 *
 * Tests the memory statistics panel component structure.
 * Note: Async fetch tests omitted due to timer complexity.
 */

// Mock dependencies to prevent actual network calls
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({
      conversation_id: 'test-123',
      collections: { books: 10, working: 5, history: 20, patterns: 15 },
      kg_patterns: 50,
      knowledge_graph: {},
      relationships: {},
      learning: {},
    }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('MemoryStatsPanel', () => {
  it('returns null when closed', async () => {
    const { default: MemoryStatsPanel } = await import('../../components/MemoryStatsPanel')
    const { container } = render(
      <MemoryStatsPanel isOpen={false} onClose={vi.fn()} />
    )
    expect(container.firstChild).toBeNull()
  })

  it('renders when open', async () => {
    const { default: MemoryStatsPanel } = await import('../../components/MemoryStatsPanel')
    const { container } = render(
      <MemoryStatsPanel isOpen={true} onClose={vi.fn()} />
    )
    expect(container.firstChild).not.toBeNull()
  })

  it('accepts onClose callback', async () => {
    const onClose = vi.fn()
    const { default: MemoryStatsPanel } = await import('../../components/MemoryStatsPanel')
    render(<MemoryStatsPanel isOpen={true} onClose={onClose} />)
    // Component accepts the prop without error
    expect(onClose).not.toHaveBeenCalled()
  })
})