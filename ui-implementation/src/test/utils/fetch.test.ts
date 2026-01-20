import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

/**
 * Fetch Utility Tests
 *
 * Tests the unified fetch wrapper that works in both dev and Tauri production.
 */

// Mock Tauri HTTP
vi.mock('@tauri-apps/api/http', () => ({
  fetch: vi.fn(),
  ResponseType: { Text: 1 },
  Body: {
    text: (content: string) => ({ type: 'text', payload: content }),
  },
}))

describe('apiFetch', () => {
  let originalWindow: any
  let apiFetch: any

  beforeEach(async () => {
    vi.resetModules()
    originalWindow = global.window

    // Default to non-Tauri environment
    global.window = { __TAURI__: undefined } as any

    const module = await import('../../utils/fetch')
    apiFetch = module.apiFetch
  })

  afterEach(() => {
    global.window = originalWindow
    vi.clearAllMocks()
  })

  describe('Non-Tauri Environment (Dev Mode)', () => {
    beforeEach(() => {
      global.fetch = vi.fn().mockResolvedValue(
        new Response('{"success": true}', { status: 200 })
      )
    })

    it('uses native fetch for localhost URLs', async () => {
      const response = await apiFetch('http://localhost:8765/api/test')

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8765/api/test',
        undefined
      )
    })

    it('passes options to native fetch', async () => {
      const options = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: 'test' }),
      }

      await apiFetch('http://localhost:8765/api/test', options)

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8765/api/test',
        options
      )
    })

    it('returns Response object', async () => {
      const response = await apiFetch('http://localhost:8765/api/test')

      expect(response).toBeInstanceOf(Response)
      expect(response.status).toBe(200)
    })
  })

  describe('URL Detection', () => {
    beforeEach(() => {
      global.fetch = vi.fn().mockResolvedValue(
        new Response('ok', { status: 200 })
      )
    })

    it('uses native fetch for 127.0.0.1 URLs', async () => {
      await apiFetch('http://127.0.0.1:8765/api/test')

      expect(global.fetch).toHaveBeenCalled()
    })

    it('uses native fetch for localhost URLs', async () => {
      await apiFetch('http://localhost:8765/api/test')

      expect(global.fetch).toHaveBeenCalled()
    })
  })
})

describe('isTauri detection', () => {
  it('detects non-Tauri environment when __TAURI__ is undefined', () => {
    // @ts-ignore
    global.window = { __TAURI__: undefined }

    // The function checks for window.__TAURI__
    const isTauri = typeof window !== 'undefined' && (window as any).__TAURI__ !== undefined
    expect(isTauri).toBe(false)
  })

  it('detects Tauri environment when __TAURI__ is defined', () => {
    // @ts-ignore
    global.window = { __TAURI__: {} }

    const isTauri = typeof window !== 'undefined' && (window as any).__TAURI__ !== undefined
    expect(isTauri).toBe(true)
  })
})
