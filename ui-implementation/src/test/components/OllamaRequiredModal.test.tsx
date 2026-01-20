import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { OllamaRequiredModal } from '../../components/OllamaRequiredModal'

/**
 * OllamaRequiredModal Tests
 *
 * Tests the Ollama setup modal with LLM and MCP tabs.
 */

// Mock Tauri shell API
vi.mock('@tauri-apps/api/shell', () => ({
  open: vi.fn().mockResolvedValue(undefined),
}))

describe('OllamaRequiredModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    onOpenIntegrations: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Closed State', () => {
    it('returns null when closed', () => {
      const { container } = render(
        <OllamaRequiredModal {...defaultProps} isOpen={false} />
      )
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Open State', () => {
    it('renders when open', () => {
      const { container } = render(<OllamaRequiredModal {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows welcome title', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      expect(screen.getByText('Welcome to Roampal')).toBeInTheDocument()
    })

    it('shows both tab options', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      expect(screen.getByText('Setup LLM Provider')).toBeInTheDocument()
      expect(screen.getByText('MCP Integration (Optional)')).toBeInTheDocument()
    })
  })

  describe('LLM Tab (Default)', () => {
    it('shows LLM provider information', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      expect(screen.getByText('Recommended Provider:')).toBeInTheDocument()
    })

    it('shows Ollama as recommended', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      expect(screen.getByText('Ollama')).toBeInTheDocument()
    })

    it('shows LM Studio option', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      expect(screen.getByText('LM Studio')).toBeInTheDocument()
    })

    it('shows download buttons', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      expect(screen.getByText('Download Ollama')).toBeInTheDocument()
      expect(screen.getByText('Download LM Studio')).toBeInTheDocument()
    })

    it('explains why local providers', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      expect(screen.getByText('Why local providers?')).toBeInTheDocument()
      expect(screen.getByText(/No data leaves your computer/)).toBeInTheDocument()
    })
  })

  describe('MCP Tab', () => {
    it('switches to MCP tab when clicked', () => {
      render(<OllamaRequiredModal {...defaultProps} />)

      fireEvent.click(screen.getByText('MCP Integration (Optional)'))

      // The actual text includes emoji: "✨ What is MCP?"
      expect(screen.getByText(/What is MCP/)).toBeInTheDocument()
    })

    it('shows MCP explanation', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      fireEvent.click(screen.getByText('MCP Integration (Optional)'))

      expect(screen.getByText(/Model Context Protocol/)).toBeInTheDocument()
    })

    it('shows quick setup steps', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      fireEvent.click(screen.getByText('MCP Integration (Optional)'))

      expect(screen.getByText('Quick Setup:')).toBeInTheDocument()
    })

    it('shows compatible tools list', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      fireEvent.click(screen.getByText('MCP Integration (Optional)'))

      expect(screen.getByText('Works with any MCP client:')).toBeInTheDocument()
      expect(screen.getByText('Claude Desktop')).toBeInTheDocument()
      expect(screen.getByText('Cursor')).toBeInTheDocument()
    })

    it('shows open integrations button', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      fireEvent.click(screen.getByText('MCP Integration (Optional)'))

      expect(screen.getByText('Open Integrations Settings')).toBeInTheDocument()
    })

    it('calls onOpenIntegrations when button clicked', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      fireEvent.click(screen.getByText('MCP Integration (Optional)'))
      fireEvent.click(screen.getByText('Open Integrations Settings'))

      expect(defaultProps.onClose).toHaveBeenCalled()
      expect(defaultProps.onOpenIntegrations).toHaveBeenCalled()
    })
  })

  describe('Get Started Button', () => {
    it('shows get started button', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      expect(screen.getByText('Get Started')).toBeInTheDocument()
    })

    it('calls onClose when get started clicked', () => {
      render(<OllamaRequiredModal {...defaultProps} />)
      fireEvent.click(screen.getByText('Get Started'))

      expect(defaultProps.onClose).toHaveBeenCalledTimes(1)
    })
  })
})
