import { describe, it, expect, vi, beforeEach } from 'vitest'
import { setupTauriEventListeners, showNotification } from '../../utils/tauriEvents'

/**
 * TauriEvents Tests
 *
 * Tests the Tauri events utility functions.
 */

describe('tauriEvents', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('setupTauriEventListeners', () => {
    it('is a function', () => {
      expect(setupTauriEventListeners).toBeInstanceOf(Function)
    })

    it('does not throw when called', () => {
      expect(() => setupTauriEventListeners()).not.toThrow()
    })
  })

  describe('showNotification', () => {
    it('is a function', () => {
      expect(showNotification).toBeInstanceOf(Function)
    })

    it('logs notification to console', () => {
      const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {})

      showNotification('Test Title', 'Test Body')

      expect(consoleSpy).toHaveBeenCalledWith('Notification:', 'Test Title', 'Test Body')

      consoleSpy.mockRestore()
    })

    it('does not throw when called', () => {
      expect(() => showNotification('Title', 'Body')).not.toThrow()
    })
  })
})