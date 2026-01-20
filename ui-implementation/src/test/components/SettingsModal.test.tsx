import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { SettingsModal } from '../../components/SettingsModal'

/**
 * SettingsModal Tests
 *
 * Tests the main settings modal component.
 */

// Mock dependencies
vi.mock('@tauri-apps/api/tauri', () => ({
  invoke: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockResolvedValue({
    ok: true,
    json: () => Promise.resolve({
      current_model: 'llama3:latest',
      provider: 'ollama',
      providers: [],
    }),
  }),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

// Mock child modals
vi.mock('../../components/DataManagementModal', () => ({
  DataManagementModal: () => <div data-testid="data-management-modal" />,
}))

vi.mock('../../components/MemoryBankModal', () => ({
  MemoryBankModal: () => <div data-testid="memory-bank-modal" />,
}))

vi.mock('../../components/ModelContextSettings', () => ({
  ModelContextSettings: () => <div data-testid="model-context-settings" />,
}))

vi.mock('../../components/IntegrationsPanel', () => ({
  IntegrationsPanel: () => <div data-testid="integrations-panel" />,
}))

vi.mock('../../components/MCPServersPanel', () => ({
  MCPServersPanel: () => <div data-testid="mcp-servers-panel" />,
}))

describe('SettingsModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
  }

  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Closed State', () => {
    it('returns null when closed', () => {
      const { container } = render(<SettingsModal {...defaultProps} isOpen={false} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Open State', () => {
    it('renders when open', () => {
      const { container } = render(<SettingsModal {...defaultProps} />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows Settings title', () => {
      render(<SettingsModal {...defaultProps} />)
      expect(screen.getByText('Settings')).toBeInTheDocument()
    })

    it('shows close button', () => {
      const { container } = render(<SettingsModal {...defaultProps} />)
      const closeButton = container.querySelector('button')
      expect(closeButton).toBeInTheDocument()
    })
  })

  describe('Menu Items', () => {
    it('shows Model Context Settings option', async () => {
      render(<SettingsModal {...defaultProps} />)
      await waitFor(() => {
        expect(screen.getByText('Model Context Settings')).toBeInTheDocument()
      })
    })

    it('shows Memory Bank option', async () => {
      render(<SettingsModal {...defaultProps} />)
      await waitFor(() => {
        expect(screen.getByText('Memory Bank')).toBeInTheDocument()
      })
    })

    it('shows Integrations option', async () => {
      render(<SettingsModal {...defaultProps} />)
      await waitFor(() => {
        expect(screen.getByText('Integrations')).toBeInTheDocument()
      })
    })

    it('shows Data Management option', async () => {
      render(<SettingsModal {...defaultProps} />)
      await waitFor(() => {
        expect(screen.getByText('Data Management')).toBeInTheDocument()
      })
    })

    it('shows MCP Tool Servers option', async () => {
      render(<SettingsModal {...defaultProps} />)
      await waitFor(() => {
        expect(screen.getByText('MCP Tool Servers')).toBeInTheDocument()
      })
    })
  })

  describe('Opening Sub-modals', () => {
    it('opens Data Management modal when clicked', async () => {
      render(<SettingsModal {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('Data Management')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByText('Data Management'))

      expect(screen.getByTestId('data-management-modal')).toBeInTheDocument()
    })

    it('opens Memory Bank modal when clicked', async () => {
      render(<SettingsModal {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('Memory Bank')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByText('Memory Bank'))

      expect(screen.getByTestId('memory-bank-modal')).toBeInTheDocument()
    })
  })

  describe('Initial Tab', () => {
    it('opens integrations when initialTab is integrations', async () => {
      render(<SettingsModal {...defaultProps} initialTab="integrations" />)

      await waitFor(() => {
        expect(screen.getByTestId('integrations-panel')).toBeInTheDocument()
      })
    })
  })

  describe('Close Behavior', () => {
    it('calls onClose when close button clicked', async () => {
      render(<SettingsModal {...defaultProps} />)

      await waitFor(() => {
        expect(screen.getByText('Settings')).toBeInTheDocument()
      })

      const closeButtons = screen.getAllByRole('button')
      // First button should be close button
      fireEvent.click(closeButtons[0])

      expect(defaultProps.onClose).toHaveBeenCalled()
    })
  })
})
