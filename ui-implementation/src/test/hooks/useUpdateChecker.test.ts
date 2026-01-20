import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useUpdateChecker } from '../../hooks/useUpdateChecker'

/**
 * useUpdateChecker Tests
 *
 * Tests the update checker hook structure.
 * Note: Timing-dependent tests are simplified to avoid flakiness.
 */

// Mock dependencies
vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

vi.mock('@tauri-apps/api/shell', () => ({
  open: vi.fn(),
}))

describe('useUpdateChecker', () => {
  const originalFetch = global.fetch

  beforeEach(() => {
    vi.clearAllMocks()
    global.fetch = vi.fn().mockResolvedValue({
      json: () => Promise.resolve({ available: false }),
    })
  })

  afterEach(() => {
    global.fetch = originalFetch
  })

  describe('Hook Structure', () => {
    it('returns expected properties', () => {
      const { result } = renderHook(() => useUpdateChecker())

      expect(result.current).toHaveProperty('updateInfo')
      expect(result.current).toHaveProperty('checking')
      expect(result.current).toHaveProperty('dismiss')
      expect(result.current).toHaveProperty('openDownload')
    })

    it('starts with null updateInfo', () => {
      const { result } = renderHook(() => useUpdateChecker())
      expect(result.current.updateInfo).toBeNull()
    })

    it('starts with checking as false', () => {
      const { result } = renderHook(() => useUpdateChecker())
      expect(result.current.checking).toBe(false)
    })

    it('dismiss is a function', () => {
      const { result } = renderHook(() => useUpdateChecker())
      expect(result.current.dismiss).toBeInstanceOf(Function)
    })

    it('openDownload is a function', () => {
      const { result } = renderHook(() => useUpdateChecker())
      expect(result.current.openDownload).toBeInstanceOf(Function)
    })
  })

  describe('Dismiss Function', () => {
    it('can be called without error', () => {
      const { result } = renderHook(() => useUpdateChecker())

      expect(() => {
        act(() => {
          result.current.dismiss()
        })
      }).not.toThrow()
    })
  })

  describe('OpenDownload Function', () => {
    it('can be called without error', async () => {
      const { result } = renderHook(() => useUpdateChecker())

      await expect(result.current.openDownload()).resolves.not.toThrow()
    })
  })
})