import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ConnectionStatus } from '../../components/ConnectionStatus'

/**
 * ConnectionStatus Tests
 *
 * Tests the connection status indicator component.
 * Verifies correct display for all connection states.
 */

describe('ConnectionStatus', () => {
  describe('Connected State', () => {
    it('displays "Connected" text', () => {
      render(<ConnectionStatus status="connected" />)
      expect(screen.getByText('Connected')).toBeInTheDocument()
    })

    it('shows green indicator', () => {
      const { container } = render(<ConnectionStatus status="connected" />)
      const indicator = container.querySelector('.bg-green-500')
      expect(indicator).toBeInTheDocument()
    })

    it('does not have pulse animation', () => {
      const { container } = render(<ConnectionStatus status="connected" />)
      const indicator = container.querySelector('.animate-pulse')
      expect(indicator).not.toBeInTheDocument()
    })
  })

  describe('Connecting State', () => {
    it('displays "Connecting..." text', () => {
      render(<ConnectionStatus status="connecting" />)
      expect(screen.getByText('Connecting...')).toBeInTheDocument()
    })

    it('shows yellow indicator', () => {
      const { container } = render(<ConnectionStatus status="connecting" />)
      const indicator = container.querySelector('.bg-yellow-500')
      expect(indicator).toBeInTheDocument()
    })

    it('has pulse animation', () => {
      const { container } = render(<ConnectionStatus status="connecting" />)
      const indicator = container.querySelector('.animate-pulse')
      expect(indicator).toBeInTheDocument()
    })
  })

  describe('Disconnected State', () => {
    it('displays "Disconnected" text', () => {
      render(<ConnectionStatus status="disconnected" />)
      expect(screen.getByText('Disconnected')).toBeInTheDocument()
    })

    it('shows zinc/gray indicator', () => {
      const { container } = render(<ConnectionStatus status="disconnected" />)
      const indicator = container.querySelector('.bg-zinc-500')
      expect(indicator).toBeInTheDocument()
    })
  })

  describe('Error State', () => {
    it('displays "Connection Error" text', () => {
      render(<ConnectionStatus status="error" />)
      expect(screen.getByText('Connection Error')).toBeInTheDocument()
    })

    it('shows red indicator', () => {
      const { container } = render(<ConnectionStatus status="error" />)
      const indicator = container.querySelector('.bg-red-500')
      expect(indicator).toBeInTheDocument()
    })
  })

  describe('Structure', () => {
    it('renders indicator dot', () => {
      const { container } = render(<ConnectionStatus status="connected" />)
      const dot = container.querySelector('.rounded-full')
      expect(dot).toBeInTheDocument()
    })

    it('renders status text with correct styling', () => {
      render(<ConnectionStatus status="connected" />)
      const text = screen.getByText('Connected')
      expect(text).toHaveClass('text-xs', 'text-zinc-400')
    })
  })
})