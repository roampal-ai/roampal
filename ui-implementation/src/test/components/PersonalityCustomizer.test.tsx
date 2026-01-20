import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { PersonalityCustomizer } from '../../components/PersonalityCustomizer'

/**
 * PersonalityCustomizer Tests
 *
 * Tests the personality customization component.
 */

// Mock dependencies
global.fetch = vi.fn()
  .mockImplementation((url: string) => {
    if (url.includes('/presets')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          presets: ['default', 'professional', 'friendly'],
        }),
      })
    }
    if (url.includes('/current')) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          template_id: 'default',
          name: 'Default',
          content: `identity:
  name: "Roampal"
  role: "Assistant"
communication:
  tone: "warm"`,
          is_preset: true,
        }),
      })
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
  })

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('PersonalityCustomizer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Initial Render', () => {
    it('renders the component', async () => {
      const { container } = render(<PersonalityCustomizer />)
      await waitFor(() => {
        expect(container.firstChild).not.toBeNull()
      })
    })

    it('shows preset selector', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Choose a Starting Point')).toBeInTheDocument()
      })
    })

    it('shows mode toggle buttons', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Quick Settings')).toBeInTheDocument()
        expect(screen.getByText('Advanced')).toBeInTheDocument()
      })
    })
  })

  describe('Quick Settings Mode', () => {
    it('shows quick settings by default', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Assistant Name')).toBeInTheDocument()
      })
    })

    it('shows conversation style setting', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Conversation Style')).toBeInTheDocument()
      })
    })

    it('shows response length setting', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Response Length')).toBeInTheDocument()
      })
    })

    it('shows memory references setting', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Memory References')).toBeInTheDocument()
      })
    })
  })

  describe('Advanced Mode', () => {
    it('switches to advanced mode', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Advanced')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByText('Advanced'))

      await waitFor(() => {
        expect(screen.getByText('Configuration File')).toBeInTheDocument()
      })
    })

    it('shows YAML editor in advanced mode', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Advanced')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByText('Advanced'))

      await waitFor(() => {
        const textarea = document.querySelector('textarea')
        expect(textarea).toBeInTheDocument()
      })
    })

    it('shows Load Example button', async () => {
      render(<PersonalityCustomizer />)

      fireEvent.click(screen.getByText('Advanced'))

      await waitFor(() => {
        expect(screen.getByText('Load Example')).toBeInTheDocument()
      })
    })
  })

  describe('Actions', () => {
    it('shows Save Changes button', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Save Changes')).toBeInTheDocument()
      })
    })

    it('shows Export button', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Export')).toBeInTheDocument()
      })
    })

    it('Save button is disabled when no changes', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        const saveButton = screen.getByText('Save Changes').closest('button')
        expect(saveButton).toBeDisabled()
      })
    })
  })

  describe('Info Panel', () => {
    it('shows info button', async () => {
      const { container } = render(<PersonalityCustomizer />)

      await waitFor(() => {
        const infoButton = container.querySelector('button[title="Show information"]')
        expect(infoButton).toBeInTheDocument()
      })
    })

    it('toggles info panel when clicked', async () => {
      const { container } = render(<PersonalityCustomizer />)

      await waitFor(() => {
        const infoButton = container.querySelector('button[title="Show information"]')
        expect(infoButton).toBeInTheDocument()
      })

      const infoButton = container.querySelector('button[title="Show information"]')
      if (infoButton) {
        fireEvent.click(infoButton)

        await waitFor(() => {
          expect(screen.getByText('How it Works')).toBeInTheDocument()
        })
      }
    })
  })

  describe('Presets', () => {
    it('loads presets into dropdown', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        const select = document.querySelector('select')
        expect(select).toBeInTheDocument()
        // The default option should be visible
        expect(screen.getByText(/Default/)).toBeInTheDocument()
      })
    })
  })

  describe('Toggle Settings', () => {
    it('shows toggle buttons for boolean settings', async () => {
      render(<PersonalityCustomizer />)

      await waitFor(() => {
        expect(screen.getByText('Use Analogies')).toBeInTheDocument()
        expect(screen.getByText('Use Examples')).toBeInTheDocument()
        expect(screen.getByText('Use Humor')).toBeInTheDocument()
      })
    })
  })
})
