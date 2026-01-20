import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { isTauri, getPlatformBadge } from '../../utils/tauri'

/**
 * Tauri Utils Tests
 *
 * Tests the Tauri utility functions.
 */

describe('tauri utils', () => {
  const originalTauri = (window as any).__TAURI__

  afterEach(() => {
    // Restore original
    if (originalTauri !== undefined) {
      (window as any).__TAURI__ = originalTauri
    } else {
      delete (window as any).__TAURI__
    }
  })

  describe('isTauri', () => {
    it('returns true when __TAURI__ is defined', () => {
      (window as any).__TAURI__ = {}
      expect(isTauri()).toBe(true)
    })

    it('returns false when __TAURI__ is undefined', () => {
      delete (window as any).__TAURI__
      expect(isTauri()).toBe(false)
    })
  })

  describe('getPlatformBadge', () => {
    it('returns Desktop when in Tauri', () => {
      (window as any).__TAURI__ = {}
      expect(getPlatformBadge()).toBe('Desktop')
    })

    it('returns null when not in Tauri', () => {
      delete (window as any).__TAURI__
      expect(getPlatformBadge()).toBeNull()
    })
  })
})