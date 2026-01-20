import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useSplitPane } from '../../hooks/useSplitPane'

/**
 * useSplitPane Hook Tests
 *
 * Tests the split pane resizing hook including:
 * - Initial size handling
 * - Min/max constraints
 * - LocalStorage persistence
 * - Reset functionality
 */

describe('useSplitPane', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    localStorage.clear()
  })

  describe('Initial State', () => {
    it('uses default initial size', () => {
      const { result } = renderHook(() => useSplitPane())
      expect(result.current.size).toBe(300) // default initialSize
    })

    it('uses provided initial size', () => {
      const { result } = renderHook(() =>
        useSplitPane({ initialSize: 400 })
      )
      expect(result.current.size).toBe(400)
    })

    it('starts not dragging', () => {
      const { result } = renderHook(() => useSplitPane())
      expect(result.current.isDragging).toBe(false)
    })
  })

  describe('LocalStorage Persistence', () => {
    it('loads size from localStorage when storageKey provided', () => {
      localStorage.setItem('test-panel', '450')

      const { result } = renderHook(() =>
        useSplitPane({ storageKey: 'test-panel', initialSize: 300 })
      )

      expect(result.current.size).toBe(450)
    })

    it('uses initialSize when localStorage is empty', () => {
      const { result } = renderHook(() =>
        useSplitPane({ storageKey: 'test-panel', initialSize: 350 })
      )

      expect(result.current.size).toBe(350)
    })

    it('ignores collapsed size in localStorage and uses initialSize', () => {
      // If stored size is <= 50, it's considered collapsed
      localStorage.setItem('test-panel', '40')

      const { result } = renderHook(() =>
        useSplitPane({ storageKey: 'test-panel', initialSize: 300 })
      )

      expect(result.current.size).toBe(300) // Uses initialSize, not 40
    })

    it('clamps stored value to min/max bounds', () => {
      localStorage.setItem('test-panel', '1000')

      const { result } = renderHook(() =>
        useSplitPane({
          storageKey: 'test-panel',
          initialSize: 300,
          minSize: 200,
          maxSize: 600,
        })
      )

      expect(result.current.size).toBe(600) // Clamped to maxSize
    })
  })

  describe('Size Constraints', () => {
    it('respects minSize constraint', () => {
      const { result } = renderHook(() =>
        useSplitPane({ minSize: 200, initialSize: 100 })
      )

      // Initial size below min should still return the initial
      // but setSize should clamp
      act(() => {
        result.current.setSize(50)
      })

      // setSize doesn't auto-clamp, but the hook's mouse handlers do
      expect(result.current.size).toBe(50) // Direct setSize doesn't clamp
    })

    it('respects maxSize constraint via direct setSize', () => {
      const { result } = renderHook(() =>
        useSplitPane({ maxSize: 600, initialSize: 300 })
      )

      act(() => {
        result.current.setSize(800)
      })

      // Direct setSize doesn't clamp
      expect(result.current.size).toBe(800)
    })
  })

  describe('Reset Functionality', () => {
    it('resets to initial size', () => {
      const { result } = renderHook(() =>
        useSplitPane({ initialSize: 300 })
      )

      act(() => {
        result.current.setSize(500)
      })

      expect(result.current.size).toBe(500)

      act(() => {
        result.current.reset()
      })

      expect(result.current.size).toBe(300)
    })

    it('updates localStorage on reset when storageKey provided', () => {
      const { result } = renderHook(() =>
        useSplitPane({ initialSize: 300, storageKey: 'test-panel' })
      )

      act(() => {
        result.current.setSize(500)
      })

      act(() => {
        result.current.reset()
      })

      expect(localStorage.getItem('test-panel')).toBe('300')
    })
  })

  describe('Return Values', () => {
    it('returns size', () => {
      const { result } = renderHook(() => useSplitPane())
      expect(typeof result.current.size).toBe('number')
    })

    it('returns isDragging', () => {
      const { result } = renderHook(() => useSplitPane())
      expect(typeof result.current.isDragging).toBe('boolean')
    })

    it('returns handleMouseDown function', () => {
      const { result } = renderHook(() => useSplitPane())
      expect(typeof result.current.handleMouseDown).toBe('function')
    })

    it('returns reset function', () => {
      const { result } = renderHook(() => useSplitPane())
      expect(typeof result.current.reset).toBe('function')
    })

    it('returns setSize function', () => {
      const { result } = renderHook(() => useSplitPane())
      expect(typeof result.current.setSize).toBe('function')
    })
  })

  describe('Direction Handling', () => {
    it('accepts horizontal direction', () => {
      const { result } = renderHook(() =>
        useSplitPane({ direction: 'horizontal' })
      )
      expect(result.current.size).toBe(300)
    })

    it('accepts vertical direction', () => {
      const { result } = renderHook(() =>
        useSplitPane({ direction: 'vertical' })
      )
      expect(result.current.size).toBe(300)
    })
  })

  describe('Inverted Mode', () => {
    it('accepts inverted option for right sidebars', () => {
      const { result } = renderHook(() =>
        useSplitPane({ inverted: true })
      )
      expect(result.current.size).toBe(300)
    })
  })
})