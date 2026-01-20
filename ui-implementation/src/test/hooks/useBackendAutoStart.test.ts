import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'

/**
 * useBackendAutoStart Tests
 *
 * Tests the backend auto-start hook.
 * Note: Complex async behavior is mocked for testing.
 */

// Mock all dependencies upfront
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({ ok: true }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

vi.mock('@tauri-apps/api/tauri', () => ({
  invoke: vi.fn().mockResolvedValue(true),
}))

describe('useBackendAutoStart', () => {
  const originalTauri = (window as any).__TAURI__

  beforeEach(() => {
    vi.clearAllMocks()
    // Ensure not in Tauri mode for dev tests
    delete (window as any).__TAURI__
  })

  afterEach(() => {
    ;(window as any).__TAURI__ = originalTauri
  })

  describe('Hook Structure', () => {
    it('returns backendStatus and errorMessage', async () => {
      const { useBackendAutoStart } = await import('../../hooks/useBackendAutoStart')
      const { result } = renderHook(() => useBackendAutoStart())

      expect(result.current).toHaveProperty('backendStatus')
      expect(result.current).toHaveProperty('errorMessage')
    })

    it('errorMessage is initially null', async () => {
      const { useBackendAutoStart } = await import('../../hooks/useBackendAutoStart')
      const { result } = renderHook(() => useBackendAutoStart())

      expect(result.current.errorMessage).toBeNull()
    })

    it('sets ready status when not in Tauri', async () => {
      delete (window as any).__TAURI__
      vi.resetModules()

      const { useBackendAutoStart } = await import('../../hooks/useBackendAutoStart')
      const { result } = renderHook(() => useBackendAutoStart())

      await waitFor(() => {
        expect(result.current.backendStatus).toBe('ready')
      })
    })
  })
})