import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Toast } from '../../components/Toast'

describe('Toast', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  describe('Rendering', () => {
    it('displays the message', () => {
      render(<Toast message="Test message" type="info" onClose={vi.fn()} />)
      expect(screen.getByText('Test message')).toBeInTheDocument()
    })

    it('renders close button', () => {
      render(<Toast message="Test" type="info" onClose={vi.fn()} />)
      expect(screen.getByRole('button', { name: /close/i })).toBeInTheDocument()
    })
  })

  describe('Types', () => {
    it('renders success type with green styling', () => {
      const { container } = render(<Toast message="Success!" type="success" onClose={vi.fn()} />)
      expect(container.querySelector('.border-green-500\\/30')).toBeInTheDocument()
    })

    it('renders error type with red styling', () => {
      const { container } = render(<Toast message="Error!" type="error" onClose={vi.fn()} />)
      expect(container.querySelector('.border-red-500\\/30')).toBeInTheDocument()
    })

    it('renders info type with blue styling', () => {
      const { container } = render(<Toast message="Info" type="info" onClose={vi.fn()} />)
      expect(container.querySelector('.border-blue-500\\/30')).toBeInTheDocument()
    })
  })

  describe('Auto-dismiss', () => {
    it('calls onClose after default duration', () => {
      const onClose = vi.fn()
      render(<Toast message="Test" type="info" onClose={onClose} />)

      expect(onClose).not.toHaveBeenCalled()
      vi.advanceTimersByTime(4000)
      expect(onClose).toHaveBeenCalledTimes(1)
    })

    it('uses custom duration', () => {
      const onClose = vi.fn()
      render(<Toast message="Test" type="info" onClose={onClose} duration={2000} />)

      vi.advanceTimersByTime(1999)
      expect(onClose).not.toHaveBeenCalled()
      vi.advanceTimersByTime(1)
      expect(onClose).toHaveBeenCalledTimes(1)
    })
  })

  describe('Manual Close', () => {
    it('calls onClose when close button clicked', async () => {
      vi.useRealTimers() // Need real timers for userEvent
      const onClose = vi.fn()
      render(<Toast message="Test" type="info" onClose={onClose} />)

      await userEvent.click(screen.getByRole('button', { name: /close/i }))
      expect(onClose).toHaveBeenCalledTimes(1)
    })
  })
})