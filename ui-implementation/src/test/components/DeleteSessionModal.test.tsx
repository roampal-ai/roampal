import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DeleteSessionModal } from '../../components/DeleteSessionModal'

describe('DeleteSessionModal', () => {
  const defaultProps = {
    isOpen: true,
    sessionTitle: 'Test Chat',
    onConfirm: vi.fn(),
    onCancel: vi.fn(),
  }

  describe('Visibility', () => {
    it('returns null when closed', () => {
      const { container } = render(
        <DeleteSessionModal {...defaultProps} isOpen={false} />
      )
      expect(container.firstChild).toBeNull()
    })

    it('renders when open', () => {
      render(<DeleteSessionModal {...defaultProps} />)
      expect(screen.getByText('Delete Conversation')).toBeInTheDocument()
    })
  })

  describe('Content', () => {
    it('shows session title', () => {
      render(<DeleteSessionModal {...defaultProps} />)
      expect(screen.getByText(/"Test Chat"/)).toBeInTheDocument()
    })

    it('shows warning message', () => {
      render(<DeleteSessionModal {...defaultProps} />)
      expect(screen.getByText(/cannot be undone/i)).toBeInTheDocument()
    })

    it('shows permanent deletion warning', () => {
      render(<DeleteSessionModal {...defaultProps} />)
      expect(screen.getByText(/permanently deleted/i)).toBeInTheDocument()
    })
  })

  describe('Actions', () => {
    it('has delete button', () => {
      render(<DeleteSessionModal {...defaultProps} />)
      expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument()
    })

    it('has cancel button', () => {
      render(<DeleteSessionModal {...defaultProps} />)
      expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument()
    })

    it('calls onConfirm when delete clicked', async () => {
      const onConfirm = vi.fn()
      render(<DeleteSessionModal {...defaultProps} onConfirm={onConfirm} />)

      await userEvent.click(screen.getByRole('button', { name: /delete/i }))
      expect(onConfirm).toHaveBeenCalled()
    })

    it('calls onCancel when cancel clicked', async () => {
      const onCancel = vi.fn()
      render(<DeleteSessionModal {...defaultProps} onCancel={onCancel} />)

      await userEvent.click(screen.getByRole('button', { name: /cancel/i }))
      expect(onCancel).toHaveBeenCalled()
    })

    it('calls onCancel when X button clicked', async () => {
      const onCancel = vi.fn()
      render(<DeleteSessionModal {...defaultProps} onCancel={onCancel} />)

      // Click the X button (close button)
      const buttons = screen.getAllByRole('button')
      const closeButton = buttons.find(b => !b.textContent?.toLowerCase().includes('delete') && !b.textContent?.toLowerCase().includes('cancel'))
      if (closeButton) await userEvent.click(closeButton)

      expect(onCancel).toHaveBeenCalled()
    })
  })
})
