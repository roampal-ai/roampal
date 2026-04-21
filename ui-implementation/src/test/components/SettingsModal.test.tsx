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

vi.mock('../../components/SidecarReloadWarningModal', () => ({
  SidecarReloadWarningModal: ({ isOpen, chatModel, sidecarModel, onCancel, onConfirm }: any) =>
    isOpen ? (
      <div data-testid="sidecar-reload-warning">
        <span data-testid="warn-chat">{chatModel}</span>
        <span data-testid="warn-sidecar">{sidecarModel}</span>
        <button data-testid="warn-cancel" onClick={onCancel}>Cancel</button>
        <button data-testid="warn-confirm" onClick={onConfirm}>Confirm</button>
      </div>
    ) : null,
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

  // v0.3.2 (0f): Advanced disclosure hides sidecar controls by default.
  describe('Advanced Settings (v0.3.2)', () => {
    it('hides the sidecar mirror toggle until Advanced is expanded', async () => {
      render(<SettingsModal {...defaultProps} />)
      await waitFor(() => {
        expect(screen.getByTestId('advanced-settings-toggle')).toBeInTheDocument()
      })
      expect(screen.queryByTestId('advanced-settings-panel')).not.toBeInTheDocument()
      expect(screen.queryByTestId('sidecar-mirror-toggle')).not.toBeInTheDocument()
    })

    it('reveals the sidecar mirror toggle after clicking Advanced', async () => {
      render(<SettingsModal {...defaultProps} />)
      await waitFor(() => {
        expect(screen.getByTestId('advanced-settings-toggle')).toBeInTheDocument()
      })
      fireEvent.click(screen.getByTestId('advanced-settings-toggle'))
      expect(screen.getByTestId('advanced-settings-panel')).toBeInTheDocument()
      expect(screen.getByTestId('sidecar-mirror-toggle')).toBeInTheDocument()
    })

    // v0.3.2: the chat-header sidecar badge deep-links in with initialFocus='advanced'.
    it('auto-expands Advanced when initialFocus="advanced"', async () => {
      render(<SettingsModal {...defaultProps} initialFocus="advanced" />)
      await waitFor(() => {
        expect(screen.getByTestId('advanced-settings-panel')).toBeInTheDocument()
      })
      expect(screen.getByTestId('sidecar-mirror-toggle')).toBeInTheDocument()
    })
  })

  describe('Sidecar Reload Warning (v0.3.2 0f)', () => {
    it('does NOT show warning when sidecar pick matches chat model', async () => {
      const { apiFetch } = await import('../../utils/fetch')
      // Chat model = llama3:latest; sidecar status returns mirror_chat=false, model=other
      ;(apiFetch as any).mockImplementation((url: string) => {
        if (url.endsWith('/api/model/current')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ current_model: 'llama3:latest', provider: 'ollama' }),
          })
        }
        if (url.endsWith('/api/model/sidecar/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ model: 'qwen2.5:7b', provider: 'ollama', mirror_chat: false }),
          })
        }
        if (url.endsWith('/api/model/providers/all/models')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ providers: { ollama: [{ name: 'llama3:latest' }] } }),
          })
        }
        if (url.endsWith('/api/model/gpu-info')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve({ gpu_count: 1 }) })
        }
        // /sidecar/set POST
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) })
      })

      render(<SettingsModal {...defaultProps} />)
      await waitFor(() => screen.getByTestId('advanced-settings-toggle'))
      fireEvent.click(screen.getByTestId('advanced-settings-toggle'))
      // Pick a sidecar model that matches chat — no warning should fire.
      const select = screen.getByTestId('sidecar-model-select')
      fireEvent.change(select, { target: { value: 'llama3:latest' } })
      expect(screen.queryByTestId('sidecar-reload-warning')).not.toBeInTheDocument()
    })

    it('shows warning when sidecar pick differs from chat on single-GPU system', async () => {
      const { apiFetch } = await import('../../utils/fetch')
      ;(apiFetch as any).mockImplementation((url: string) => {
        if (url.endsWith('/api/model/current')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ current_model: 'llama3:latest', provider: 'ollama' }),
          })
        }
        if (url.endsWith('/api/model/sidecar/status')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ model: '', provider: 'ollama', mirror_chat: false }),
          })
        }
        if (url.endsWith('/api/model/providers/all/models')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              providers: { ollama: [{ name: 'llama3:latest' }, { name: 'gpt-oss:20b' }] },
            }),
          })
        }
        if (url.endsWith('/api/model/gpu-info')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve({ gpu_count: 1 }) })
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) })
      })

      render(<SettingsModal {...defaultProps} />)
      await waitFor(() => screen.getByTestId('advanced-settings-toggle'))
      fireEvent.click(screen.getByTestId('advanced-settings-toggle'))
      const select = screen.getByTestId('sidecar-model-select')
      fireEvent.change(select, { target: { value: 'gpt-oss:20b' } })
      expect(screen.getByTestId('sidecar-reload-warning')).toBeInTheDocument()
      expect(screen.getByTestId('warn-chat').textContent).toBe('llama3:latest')
      expect(screen.getByTestId('warn-sidecar').textContent).toBe('gpt-oss:20b')
    })
  })
})
