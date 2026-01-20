import { describe, it, expect, beforeEach, vi } from 'vitest'
import { act } from '@testing-library/react'
import { usePreferenceStore } from '../../stores/usePreferenceStore'

/**
 * Preference Store Tests
 *
 * Tests user preference management including transparency settings.
 */

describe('usePreferenceStore', () => {
  beforeEach(() => {
    // Reset to defaults before each test
    act(() => {
      usePreferenceStore.getState().resetToDefaults()
    })
  })

  describe('Default Values', () => {
    it('starts with summary transparency level', () => {
      expect(usePreferenceStore.getState().transparencyLevel).toBe('summary')
    })

    it('starts with autoExpandThinking false', () => {
      expect(usePreferenceStore.getState().autoExpandThinking).toBe(false)
    })

    it('starts with inline thinking position', () => {
      expect(usePreferenceStore.getState().thinkingPosition).toBe('inline')
    })

    it('starts with showConfidence true', () => {
      expect(usePreferenceStore.getState().showConfidence).toBe(true)
    })

    it('starts with showAlternatives true', () => {
      expect(usePreferenceStore.getState().showAlternatives).toBe(true)
    })
  })

  describe('Transparency Level', () => {
    it('can set to none', () => {
      act(() => {
        usePreferenceStore.getState().setTransparencyLevel('none')
      })
      expect(usePreferenceStore.getState().transparencyLevel).toBe('none')
    })

    it('can set to summary', () => {
      act(() => {
        usePreferenceStore.getState().setTransparencyLevel('summary')
      })
      expect(usePreferenceStore.getState().transparencyLevel).toBe('summary')
    })

    it('can set to detailed', () => {
      act(() => {
        usePreferenceStore.getState().setTransparencyLevel('detailed')
      })
      expect(usePreferenceStore.getState().transparencyLevel).toBe('detailed')
    })
  })

  describe('Auto Expand Thinking', () => {
    it('can enable auto expand', () => {
      act(() => {
        usePreferenceStore.getState().setAutoExpand(true)
      })
      expect(usePreferenceStore.getState().autoExpandThinking).toBe(true)
    })

    it('can disable auto expand', () => {
      act(() => {
        usePreferenceStore.getState().setAutoExpand(true)
        usePreferenceStore.getState().setAutoExpand(false)
      })
      expect(usePreferenceStore.getState().autoExpandThinking).toBe(false)
    })
  })

  describe('Thinking Position', () => {
    it('can set to inline', () => {
      act(() => {
        usePreferenceStore.getState().setPosition('inline')
      })
      expect(usePreferenceStore.getState().thinkingPosition).toBe('inline')
    })

    it('can set to sidebar', () => {
      act(() => {
        usePreferenceStore.getState().setPosition('sidebar')
      })
      expect(usePreferenceStore.getState().thinkingPosition).toBe('sidebar')
    })
  })

  describe('Show Confidence', () => {
    it('can disable confidence display', () => {
      act(() => {
        usePreferenceStore.getState().setShowConfidence(false)
      })
      expect(usePreferenceStore.getState().showConfidence).toBe(false)
    })

    it('can enable confidence display', () => {
      act(() => {
        usePreferenceStore.getState().setShowConfidence(false)
        usePreferenceStore.getState().setShowConfidence(true)
      })
      expect(usePreferenceStore.getState().showConfidence).toBe(true)
    })
  })

  describe('Show Alternatives', () => {
    it('can disable alternatives display', () => {
      act(() => {
        usePreferenceStore.getState().setShowAlternatives(false)
      })
      expect(usePreferenceStore.getState().showAlternatives).toBe(false)
    })
  })

  describe('Reset to Defaults', () => {
    it('resets all preferences to defaults', () => {
      // Change all settings
      act(() => {
        usePreferenceStore.getState().setTransparencyLevel('detailed')
        usePreferenceStore.getState().setAutoExpand(true)
        usePreferenceStore.getState().setPosition('sidebar')
        usePreferenceStore.getState().setShowConfidence(false)
        usePreferenceStore.getState().setShowAlternatives(false)
      })

      // Verify they changed
      expect(usePreferenceStore.getState().transparencyLevel).toBe('detailed')
      expect(usePreferenceStore.getState().autoExpandThinking).toBe(true)

      // Reset
      act(() => {
        usePreferenceStore.getState().resetToDefaults()
      })

      // Verify reset
      expect(usePreferenceStore.getState().transparencyLevel).toBe('summary')
      expect(usePreferenceStore.getState().autoExpandThinking).toBe(false)
      expect(usePreferenceStore.getState().thinkingPosition).toBe('inline')
      expect(usePreferenceStore.getState().showConfidence).toBe(true)
      expect(usePreferenceStore.getState().showAlternatives).toBe(true)
    })
  })

  describe('Action Stability', () => {
    it('actions are stable references', () => {
      const state1 = usePreferenceStore.getState()
      const state2 = usePreferenceStore.getState()

      expect(state1.setTransparencyLevel).toBe(state2.setTransparencyLevel)
      expect(state1.setAutoExpand).toBe(state2.setAutoExpand)
      expect(state1.resetToDefaults).toBe(state2.resetToDefaults)
    })
  })
})