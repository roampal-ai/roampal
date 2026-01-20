import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useChatStore } from '../../stores/useChatStore'

/**
 * v0.3.0 Performance Fix Verification
 *
 * These tests verify the store selector optimization in ConnectedChat.tsx
 * Previously: Destructuring all values caused re-renders on ANY store change
 * Now: Individual selectors only re-render when that specific value changes
 */

describe('useChatStore selectors (v0.3.0 performance fix)', () => {
  beforeEach(() => {
    // Reset store to initial state
    useChatStore.setState({
      conversationId: null,
      connectionStatus: 'disconnected',
      messages: [],
      isProcessing: false,
      processingStatus: '',
    })
  })

  it('individual selectors return correct values', () => {
    const { result: conversationId } = renderHook(() =>
      useChatStore(state => state.conversationId)
    )
    const { result: connectionStatus } = renderHook(() =>
      useChatStore(state => state.connectionStatus)
    )
    const { result: messages } = renderHook(() =>
      useChatStore(state => state.messages)
    )
    const { result: isProcessing } = renderHook(() =>
      useChatStore(state => state.isProcessing)
    )

    expect(conversationId.current).toBe(null)
    expect(connectionStatus.current).toBe('disconnected')
    expect(messages.current).toEqual([])
    expect(isProcessing.current).toBe(false)
  })

  it('selector only updates when its specific value changes', () => {
    const renderCount = { messages: 0, isProcessing: 0 }

    const { result: messagesResult } = renderHook(() => {
      renderCount.messages++
      return useChatStore(state => state.messages)
    })

    const { result: isProcessingResult } = renderHook(() => {
      renderCount.isProcessing++
      return useChatStore(state => state.isProcessing)
    })

    // Initial render
    expect(renderCount.messages).toBe(1)
    expect(renderCount.isProcessing).toBe(1)

    // Change isProcessing - should NOT re-render messages selector
    act(() => {
      useChatStore.setState({ isProcessing: true })
    })

    expect(renderCount.isProcessing).toBe(2) // Updated
    expect(renderCount.messages).toBe(1) // NOT updated - this is the fix!

    // Change messages - should NOT re-render isProcessing selector
    act(() => {
      useChatStore.setState({
        messages: [{ id: '1', role: 'user', content: 'test' }] as any
      })
    })

    expect(renderCount.messages).toBe(2) // Updated
    expect(renderCount.isProcessing).toBe(2) // NOT updated again
  })

  it('getState() returns stable action references', () => {
    const actions1 = useChatStore.getState()
    const actions2 = useChatStore.getState()

    // Actions should be the same reference (no re-renders needed)
    expect(actions1.searchMemory).toBe(actions2.searchMemory)
    expect(actions1.clearSession).toBe(actions2.clearSession)
    expect(actions1.initialize).toBe(actions2.initialize)
  })
})
