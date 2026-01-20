import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { DeleteConfirmationModal } from '../../components/DeleteConfirmationModal'

describe('DeleteConfirmationModal', () => {
  const defaultProps = {
    isOpen: true,
    onClose: vi.fn(),
    onConfirm: vi.fn(),
    title: 'Delete Item',
    message: 'Are you sure you want to delete this?',
  }

  describe('Visibility', () => {
    it('returns null when closed', () => {
      const { container } = render(
        <DeleteConfirmationModal {...defaultProps} isOpen={false} />
      )
      expect(container.firstChild).toBeNull()
    })

    it('renders when open', () => {
      render(<DeleteConfirmationModal {...defaultProps} />)
      expect(screen.getByText('Delete Item')).toBeInTheDocument()
    })
  })

  describe('Content', () => {
    it('displays the title', () => {
      render(<DeleteConfirmationModal {...defaultProps} />)
      expect(screen.getByText('Delete Item')).toBeInTheDocument()
    })

    it('displays the message', () => {
      render(<DeleteConfirmationModal {...defaultProps} />)
      expect(screen.getByText('Are you sure you want to delete this?')).toBeInTheDocument()
    })

    it('shows item count when provided', () => {
      render(<DeleteConfirmationModal {...defaultProps} itemCount={5} />)
      expect(screen.getByText('5')).toBeInTheDocument()
    })

    it('shows collection name when provided', () => {
      render(
        <DeleteConfirmationModal
          {...defaultProps}
          itemCount={5}
          collectionName="patterns"
        />
      )
      expect(screen.getByText('patterns')).toBeInTheDocument()
    })
  })

  describe('Confirmation Flow', () => {
    it('requires typing DELETE to enable button', () => {
      render(<DeleteConfirmationModal {...defaultProps} />)

      const deleteButton = screen.getByRole('button', { name: /delete permanently/i })
      expect(deleteButton).toBeDisabled()
    })

    it('enables button after typing DELETE', async () => {
      render(<DeleteConfirmationModal {...defaultProps} />)

      const input = screen.getByPlaceholderText(/type delete/i)
      await userEvent.type(input, 'DELETE')

      const deleteButton = screen.getByRole('button', { name: /delete permanently/i })
      expect(deleteButton).not.toBeDisabled()
    })

    it('does not enable with wrong text', async () => {
      render(<DeleteConfirmationModal {...defaultProps} />)

      const input = screen.getByPlaceholderText(/type delete/i)
      await userEvent.type(input, 'delete') // lowercase

      const deleteButton = screen.getByRole('button', { name: /delete permanently/i })
      expect(deleteButton).toBeDisabled()
    })

    it('calls onConfirm when confirmed', async () => {
      const onConfirm = vi.fn().mockResolvedValue(undefined)
      render(<DeleteConfirmationModal {...defaultProps} onConfirm={onConfirm} />)

      const input = screen.getByPlaceholderText(/type delete/i)
      await userEvent.type(input, 'DELETE')

      const deleteButton = screen.getByRole('button', { name: /delete permanently/i })
      await userEvent.click(deleteButton)

      expect(onConfirm).toHaveBeenCalled()
    })
  })

  describe('Cancel', () => {
    it('has cancel button', () => {
      render(<DeleteConfirmationModal {...defaultProps} />)
      expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument()
    })

    it('calls onClose when cancel clicked', async () => {
      const onClose = vi.fn()
      render(<DeleteConfirmationModal {...defaultProps} onClose={onClose} />)

      await userEvent.click(screen.getByRole('button', { name: /cancel/i }))
      expect(onClose).toHaveBeenCalled()
    })

    it('clears input on close', async () => {
      const { rerender } = render(<DeleteConfirmationModal {...defaultProps} />)

      const input = screen.getByPlaceholderText(/type delete/i)
      await userEvent.type(input, 'DELETE')
      expect(input).toHaveValue('DELETE')

      await userEvent.click(screen.getByRole('button', { name: /cancel/i }))

      // Rerender to simulate reopening
      rerender(<DeleteConfirmationModal {...defaultProps} />)
      expect(screen.getByPlaceholderText(/type delete/i)).toHaveValue('')
    })
  })

  describe('Warning', () => {
    it('shows permanent action warning', () => {
      render(<DeleteConfirmationModal {...defaultProps} />)
      expect(screen.getByText(/this action is permanent/i)).toBeInTheDocument()
    })
  })
})