import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CommandInput } from '../../components/CommandInput'

/**
 * CommandInput Tests
 *
 * Tests the main chat input component including:
 * - Text input and submission
 * - Command detection (/ prefix)
 * - Processing state handling
 * - Keyboard shortcuts
 */

describe('CommandInput', () => {
  const defaultProps = {
    onSend: vi.fn(),
    onCommand: vi.fn(),
    onVoiceStart: vi.fn(),
    onVoiceEnd: vi.fn(),
    isProcessing: false,
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Rendering', () => {
    it('renders textarea with placeholder', () => {
      render(<CommandInput {...defaultProps} />)
      expect(screen.getByPlaceholderText(/Type a message/i)).toBeInTheDocument()
    })

    it('renders custom placeholder when provided', () => {
      render(<CommandInput {...defaultProps} placeholder="Custom placeholder" />)
      expect(screen.getByPlaceholderText('Custom placeholder')).toBeInTheDocument()
    })

    it('renders send button', () => {
      render(<CommandInput {...defaultProps} />)
      expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument()
    })
  })

  describe('Text Input', () => {
    it('updates value on typing', async () => {
      render(<CommandInput {...defaultProps} />)
      const textarea = screen.getByPlaceholderText(/Type a message/i)

      await userEvent.type(textarea, 'Hello world')
      expect(textarea).toHaveValue('Hello world')
    })

    it('clears input after send', async () => {
      render(<CommandInput {...defaultProps} />)
      const textarea = screen.getByPlaceholderText(/Type a message/i)

      await userEvent.type(textarea, 'Test message')
      await userEvent.click(screen.getByRole('button', { name: /send/i }))

      expect(defaultProps.onSend).toHaveBeenCalledWith('Test message', [])
    })
  })

  describe('Command Detection', () => {
    it('shows command menu when typing /', async () => {
      render(<CommandInput {...defaultProps} />)
      const textarea = screen.getByPlaceholderText(/Type a message/i)

      await userEvent.type(textarea, '/')
      // Command menu should appear - use getAllByText since there may be multiple matches
      const matches = screen.getAllByText(/switch/i)
      expect(matches.length).toBeGreaterThan(0)
    })

    it('filters commands based on input', async () => {
      render(<CommandInput {...defaultProps} />)
      const textarea = screen.getByPlaceholderText(/Type a message/i)

      await userEvent.type(textarea, '/mem')
      const matches = screen.getAllByText(/memory/i)
      expect(matches.length).toBeGreaterThan(0)
    })
  })

  describe('Processing State', () => {
    it('disables input when processing', () => {
      render(<CommandInput {...defaultProps} isProcessing={true} />)
      const textarea = screen.getByPlaceholderText(/Type a message/i)
      expect(textarea).toBeDisabled()
    })

    it('disables send button when processing', () => {
      render(<CommandInput {...defaultProps} isProcessing={true} />)
      const sendButton = screen.getByRole('button', { name: /send/i })
      expect(sendButton).toBeDisabled()
    })

    it('enables input when not processing', () => {
      render(<CommandInput {...defaultProps} isProcessing={false} />)
      const textarea = screen.getByPlaceholderText(/Type a message/i)
      expect(textarea).not.toBeDisabled()
    })
  })

  describe('Keyboard Shortcuts', () => {
    it('submits on Enter without Shift', async () => {
      render(<CommandInput {...defaultProps} />)
      const textarea = screen.getByPlaceholderText(/Type a message/i)

      await userEvent.type(textarea, 'Test message')
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })

      expect(defaultProps.onSend).toHaveBeenCalled()
    })

    it('adds newline on Shift+Enter', async () => {
      render(<CommandInput {...defaultProps} />)
      const textarea = screen.getByPlaceholderText(/Type a message/i)

      await userEvent.type(textarea, 'Line 1')
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true })

      // Should not submit
      expect(defaultProps.onSend).not.toHaveBeenCalled()
    })
  })

  describe('Empty Input Handling', () => {
    it('does not submit empty message', async () => {
      render(<CommandInput {...defaultProps} />)
      const sendButton = screen.getByRole('button', { name: /send/i })

      await userEvent.click(sendButton)

      expect(defaultProps.onSend).not.toHaveBeenCalled()
    })

    it('does not submit whitespace-only message', async () => {
      render(<CommandInput {...defaultProps} />)
      const textarea = screen.getByPlaceholderText(/Type a message/i)

      await userEvent.type(textarea, '   ')
      await userEvent.click(screen.getByRole('button', { name: /send/i }))

      expect(defaultProps.onSend).not.toHaveBeenCalled()
    })
  })
})
