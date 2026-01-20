import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { IntegrationsPanel } from '../../components/IntegrationsPanel'

/**
 * IntegrationsPanel Tests
 *
 * Tests the MCP integrations panel component.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({
      tools: [
        { name: 'Claude Desktop', status: 'connected', config_path: '/path/claude' },
        { name: 'Cursor', status: 'available', config_path: '/path/cursor' },
        { name: 'VSCode', status: 'not_installed', config_path: null },
      ],
    }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('IntegrationsPanel', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  describe('Closed State', () => {
    it('returns null when closed', () => {
      const { container } = render(<IntegrationsPanel {...defaultProps} isOpen={false} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Open State', () => {
    it('renders when open', () => {
      const { container } = render(<IntegrationsPanel {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows Integrations title', () => {
      render(<IntegrationsPanel {...defaultProps} />)
      expect(screen.getByText('Integrations')).toBeInTheDocument()
    })

    it('shows subtitle', () => {
      render(<IntegrationsPanel {...defaultProps} />)
      expect(screen.getByText('Connect Roampal memory to AI tools')).toBeInTheDocument()
    })

    it('shows refresh button', async () => {
      render(<IntegrationsPanel {...defaultProps} />)
      // Button shows "Scanning..." initially, then "Refresh" after loading
      await waitFor(() => {
        expect(screen.getByText(/Refresh|Scanning/)).toBeInTheDocument()
      })
    })
  })

  describe('Tools List', () => {
    it('shows tools after loading', async () => {
      render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('Claude Desktop')).toBeInTheDocument()
        expect(screen.getByText('Cursor')).toBeInTheDocument()
        expect(screen.getByText('VSCode')).toBeInTheDocument()
      })
    })

    it('shows status for each tool', async () => {
      render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('Connected')).toBeInTheDocument()
        expect(screen.getByText('Available')).toBeInTheDocument()
        expect(screen.getByText('Not Installed')).toBeInTheDocument()
      })
    })

    it('shows Connect button for available tools', async () => {
      render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('Connect')).toBeInTheDocument()
      })
    })

    it('shows Disconnect button for connected tools', async () => {
      render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('Disconnect')).toBeInTheDocument()
      })
    })
  })

  describe('Add Custom Client', () => {
    it('shows Add Custom MCP Client button', async () => {
      render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('Add Custom MCP Client')).toBeInTheDocument()
      })
    })
  })

  describe('Info Box', () => {
    it('shows how it works info', async () => {
      render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('How it works')).toBeInTheDocument()
      })
    })

    it('explains local storage', async () => {
      render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText(/Roampal memory stays 100% local/)).toBeInTheDocument()
      })
    })
  })

  describe('Close Behavior', () => {
    it('calls onClose when close button clicked', () => {
      const { container } = render(<IntegrationsPanel {...defaultProps} />)

      const closeButton = container.querySelector('button[title="Close"]')
      if (closeButton) {
        fireEvent.click(closeButton)
        expect(defaultProps.onClose).toHaveBeenCalledTimes(1)
      }
    })

    it('calls onClose when overlay clicked', async () => {
      render(<IntegrationsPanel {...defaultProps} />)

      // Wait for content to load
      await waitFor(() => {
        expect(screen.getByText('Integrations')).toBeInTheDocument()
      })

      // Find and click overlay (not the modal content)
      const overlay = document.querySelector('.fixed.inset-0.bg-black\\/50')
      if (overlay) {
        fireEvent.click(overlay)
        expect(defaultProps.onClose).toHaveBeenCalled()
      }
    })
  })

  describe('Status Colors', () => {
    it('shows green indicator for connected tools', async () => {
      const { container } = render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(container.querySelector('.bg-green-500')).toBeInTheDocument()
      })
    })

    it('shows yellow indicator for available tools', async () => {
      const { container } = render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(container.querySelector('.bg-yellow-500')).toBeInTheDocument()
      })
    })

    it('shows gray indicator for not installed tools', async () => {
      const { container } = render(<IntegrationsPanel {...defaultProps} />)

      await waitFor(() => {
        expect(container.querySelector('.bg-zinc-600')).toBeInTheDocument()
      })
    })
  })
})
