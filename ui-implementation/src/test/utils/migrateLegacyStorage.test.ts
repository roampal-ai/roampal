import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

/**
 * MigrateLegacyStorage Tests
 *
 * Tests the localStorage migration utility.
 */

describe('migrateLegacyStorage', () => {
  let originalLocalStorage: Storage

  beforeEach(() => {
    // Mock localStorage
    originalLocalStorage = global.localStorage

    const store: Record<string, string> = {}
    const mockStorage = {
      getItem: vi.fn((key: string) => store[key] || null),
      setItem: vi.fn((key: string, value: string) => {
        store[key] = value
      }),
      removeItem: vi.fn((key: string) => {
        delete store[key]
      }),
      clear: vi.fn(() => {
        Object.keys(store).forEach((key) => delete store[key])
      }),
      length: 0,
      key: vi.fn(),
    }

    Object.defineProperty(global, 'localStorage', {
      value: mockStorage,
      writable: true,
    })

    // Reset modules to re-run migration
    vi.resetModules()
  })

  afterEach(() => {
    Object.defineProperty(global, 'localStorage', {
      value: originalLocalStorage,
      writable: true,
    })
  })

  describe('Migration Check', () => {
    it('skips migration if already complete', async () => {
      localStorage.setItem('roampal_migration_v2_complete', 'true')

      await import('../../utils/migrateLegacyStorage')

      // Should not have set any new items (except the marker)
      expect(localStorage.setItem).toHaveBeenCalledTimes(1)
    })

    it('marks migration as complete', async () => {
      await import('../../utils/migrateLegacyStorage')

      expect(localStorage.setItem).toHaveBeenCalledWith('roampal_migration_v2_complete', 'true')
    })
  })

  describe('Chat History Migration', () => {
    it('migrates old chat_history format', async () => {
      const oldData = {
        session_id: 'old-session',
        active_shard: 'test-shard',
        messages: [{ content: 'hello' }],
      }
      localStorage.setItem('chat_history', JSON.stringify(oldData))

      await import('../../utils/migrateLegacyStorage')

      expect(localStorage.removeItem).toHaveBeenCalledWith('chat_history')
    })
  })

  describe('Feature Flag Cleanup', () => {
    it('removes old feature flags', async () => {
      localStorage.setItem('VITE_ENABLE_NEURAL_UI', 'true')
      localStorage.setItem('VITE_ENABLE_MOCK_MODE', 'true')

      await import('../../utils/migrateLegacyStorage')

      expect(localStorage.removeItem).toHaveBeenCalledWith('VITE_ENABLE_NEURAL_UI')
      expect(localStorage.removeItem).toHaveBeenCalledWith('VITE_ENABLE_MOCK_MODE')
    })
  })
})