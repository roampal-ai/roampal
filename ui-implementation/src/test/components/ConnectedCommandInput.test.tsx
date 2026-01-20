import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ConnectedCommandInput } from '../../components/ConnectedCommandInput'

/**
 * ConnectedCommandInput Tests
 *
 * Tests the connected command input component.
 */

// Mock the chat store
const mockSendMessage = vi.fn()
const mockSearchMemory = vi.fn()
const mockCancelProcessing = vi.fn()
const mockClearSession = vi.fn()

vi.mock('../../stores/useChatStore', () => ({
  useChatStore: Object.assign(
    () => ({
      sendMessage: mockSendMessage,
      searchMemory: mockSearchMemory,
      isProcessing: false,
      processingStatus: null,
      cancelProcessing: mockCancelProcessing,
    }),
    {
      getState: () => ({
        clearSession: mockClearSession,
      }),
    }
  ),
}))

vi.mock('../../utils/logger', () => ({
  default: {
    debug: vi.fn(),
    info: vi.fn(),
  },
}))

describe('ConnectedCommandInput', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Initial Render', () => {
    it('renders the component', () => {
      const { container } = render(<ConnectedCommandInput />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows placeholder text', () => {
      render(<ConnectedCommandInput />)
      expect(screen.getByPlaceholderText(/Ready when you are/)).toBeInTheDocument()
    })

    it('shows send button', () => {
      const { container } = render(<ConnectedCommandInput />)
      const sendButton = container.querySelector('button')
      expect(sendButton).toBeInTheDocument()
    })
  })

  describe('Input Handling', () => {
    it('updates text when typing', () => {
      render(<ConnectedCommandInput />)

      const textarea = screen.getByPlaceholderText(/Ready when you are/)
      fireEvent.change(textarea, { target: { value: 'Hello' } })

      expect(textarea).toHaveValue('Hello')
    })

    it('clears input after sending', async () => {
      render(<ConnectedCommandInput />)

      const textarea = screen.getByPlaceholderText(/Ready when you are/)
      fireEvent.change(textarea, { target: { value: 'Hello' } })
      fireEvent.keyDown(textarea, { key: 'Enter' })

      expect(textarea).toHaveValue('')
    })
  })

  describe('Command Palette', () => {
    it('shows command palette when typing /', () => {
      render(<ConnectedCommandInput />)

      const textarea = screen.getByPlaceholderText(/Ready when you are/)
      fireEvent.change(textarea, { target: { value: '/' } })

      expect(screen.getByText('memory search [query]')).toBeInTheDocument()
    })

    it('hides command palette when typing regular text', () => {
      render(<ConnectedCommandInput />)

      const textarea = screen.getByPlaceholderText(/Ready when you are/)
      fireEvent.change(textarea, { target: { value: '/' } })
      fireEvent.change(textarea, { target: { value: 'hello' } })

      expect(screen.queryByText('memory search [query]')).not.toBeInTheDocument()
    })

    it('shows available commands', () => {
      render(<ConnectedCommandInput />)

      const textarea = screen.getByPlaceholderText(/Ready when you are/)
      fireEvent.change(textarea, { target: { value: '/' } })

      expect(screen.getByText('memory search [query]')).toBeInTheDocument()
      expect(screen.getByText('clear')).toBeInTheDocument()
      expect(screen.getByText('help')).toBeInTheDocument()
    })
  })

  describe('Keyboard Shortcuts', () => {
    it('sends message on Enter', () => {
      render(<ConnectedCommandInput />)

      const textarea = screen.getByPlaceholderText(/Ready when you are/)
      fireEvent.change(textarea, { target: { value: 'Hello' } })
      fireEvent.keyDown(textarea, { key: 'Enter' })

      expect(mockSendMessage).toHaveBeenCalledWith('Hello')
    })

    it('allows new line with Shift+Enter', () => {
      render(<ConnectedCommandInput />)

      const textarea = screen.getByPlaceholderText(/Ready when you are/)
      fireEvent.change(textarea, { target: { value: 'Line 1' } })
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true })

      // Should not call sendMessage
      expect(mockSendMessage).not.toHaveBeenCalled()
    })

    it('navigates commands with arrow keys', () => {
      render(<ConnectedCommandInput />)

      const textarea = screen.getByPlaceholderText(/Ready when you are/)
      fireEvent.change(textarea, { target: { value: '/' } })

      // Navigate down
      fireEvent.keyDown(textarea, { key: 'ArrowDown' })

      // The second command should be highlighted (tested via visual state)
      // This tests that arrow key navigation doesn't crash
      expect(screen.getByText('memory search [query]')).toBeInTheDocument()
    })

    it('closes palette on Escape', () => {
      render(<ConnectedCommandInput />)

      const textarea = screen.getByPlaceholderText(/Ready when you are/)
      fireEvent.change(textarea, { target: { value: '/' } })

      expect(screen.getByText('memory search [query]')).toBeInTheDocument()

      fireEvent.keyDown(textarea, { key: 'Escape' })

      expect(screen.queryByText('memory search [query]')).not.toBeInTheDocument()
    })
  })

  describe('No Model State', () => {
    it('shows disabled message when no model', () => {
      render(<ConnectedCommandInput hasChatModel={false} />)

      const textarea = screen.getByPlaceholderText(/Install a chat model/)
      expect(textarea).toBeDisabled()
    })
  })

  describe('Helper Text', () => {
    it('shows keyboard shortcut hints', () => {
      render(<ConnectedCommandInput />)
      expect(screen.getByText('Shift+Enter for new line')).toBeInTheDocument()
      expect(screen.getByText('⌘+Enter to send')).toBeInTheDocument()
    })
  })
})
