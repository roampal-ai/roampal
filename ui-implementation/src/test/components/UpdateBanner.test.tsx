import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { UpdateBanner } from '../../components/UpdateBanner'

// Mock the hook
vi.mock('../../hooks/useUpdateChecker', () => ({
  useUpdateChecker: vi.fn(),
}))

import { useUpdateChecker } from '../../hooks/useUpdateChecker'

describe('UpdateBanner', () => {
  describe('No Update', () => {
    it('returns null when no update available', () => {
      ;(useUpdateChecker as any).mockReturnValue({
        updateInfo: null,
        dismiss: vi.fn(),
        openDownload: vi.fn(),
      })

      const { container } = render(<UpdateBanner />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Normal Update', () => {
    beforeEach(() => {
      ;(useUpdateChecker as any).mockReturnValue({
        updateInfo: {
          version: '0.3.1',
          notes: 'Bug fixes',
          is_critical: false,
        },
        dismiss: vi.fn(),
        openDownload: vi.fn(),
      })
    })

    it('shows update available message', () => {
      render(<UpdateBanner />)
      expect(screen.getByText('Update Available')).toBeInTheDocument()
    })

    it('displays version number', () => {
      render(<UpdateBanner />)
      expect(screen.getByText(/0\.3\.1/)).toBeInTheDocument()
    })

    it('shows Download button', () => {
      render(<UpdateBanner />)
      expect(screen.getByRole('button', { name: /download/i })).toBeInTheDocument()
    })

    it('shows Later button for non-critical', () => {
      render(<UpdateBanner />)
      expect(screen.getByRole('button', { name: /later/i })).toBeInTheDocument()
    })

    it('calls openDownload when Download clicked', async () => {
      const openDownload = vi.fn()
      ;(useUpdateChecker as any).mockReturnValue({
        updateInfo: { version: '0.3.1', is_critical: false },
        dismiss: vi.fn(),
        openDownload,
      })

      render(<UpdateBanner />)
      await userEvent.click(screen.getByRole('button', { name: /download/i }))
      expect(openDownload).toHaveBeenCalled()
    })

    it('calls dismiss when Later clicked', async () => {
      const dismiss = vi.fn()
      ;(useUpdateChecker as any).mockReturnValue({
        updateInfo: { version: '0.3.1', is_critical: false },
        dismiss,
        openDownload: vi.fn(),
      })

      render(<UpdateBanner />)
      await userEvent.click(screen.getByRole('button', { name: /later/i }))
      expect(dismiss).toHaveBeenCalled()
    })
  })

  describe('Critical Update', () => {
    beforeEach(() => {
      ;(useUpdateChecker as any).mockReturnValue({
        updateInfo: {
          version: '0.3.1',
          notes: 'Security fix',
          is_critical: true,
        },
        dismiss: vi.fn(),
        openDownload: vi.fn(),
      })
    })

    it('shows critical update message', () => {
      render(<UpdateBanner />)
      expect(screen.getByText('Critical Update Required')).toBeInTheDocument()
    })

    it('does not show Later button for critical', () => {
      render(<UpdateBanner />)
      expect(screen.queryByRole('button', { name: /later/i })).not.toBeInTheDocument()
    })

    it('uses red styling for critical', () => {
      const { container } = render(<UpdateBanner />)
      expect(container.querySelector('.bg-red-900\\/90')).toBeInTheDocument()
    })
  })
})