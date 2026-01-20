import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ToolExecutionDisplay } from '../../components/ToolExecutionDisplay'

/**
 * ToolExecutionDisplay Tests
 *
 * Tests the tool execution status display component.
 */

describe('ToolExecutionDisplay', () => {
  describe('Empty State', () => {
    it('returns null when no executions', () => {
      const { container } = render(<ToolExecutionDisplay executions={[]} />)
      expect(container.firstChild).toBeNull()
    })

    it('returns null when executions is undefined', () => {
      const { container } = render(<ToolExecutionDisplay executions={undefined as any} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Running Status', () => {
    it('shows running execution', () => {
      render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'search',
            status: 'running',
            description: 'Searching memory',
          }]}
        />
      )
      expect(screen.getByText('Searching memory')).toBeInTheDocument()
    })

    it('shows Running... label', () => {
      render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'search',
            status: 'running',
            description: 'Searching memory',
          }]}
        />
      )
      expect(screen.getByText('Running...')).toBeInTheDocument()
    })

    it('shows spinner for running status', () => {
      const { container } = render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'search',
            status: 'running',
            description: 'Searching memory',
          }]}
        />
      )
      expect(container.querySelector('.animate-spin')).toBeInTheDocument()
    })
  })

  describe('Completed Status', () => {
    it('shows completed execution', () => {
      render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'add_memory',
            status: 'completed',
            description: 'Added to memory bank',
          }]}
        />
      )
      expect(screen.getByText('Added to memory bank')).toBeInTheDocument()
    })

    it('shows green check for completed', () => {
      const { container } = render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'add_memory',
            status: 'completed',
            description: 'Added to memory bank',
          }]}
        />
      )
      expect(container.querySelector('.text-green-500')).toBeInTheDocument()
    })

    it('does not show Running... for completed', () => {
      render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'add_memory',
            status: 'completed',
            description: 'Added to memory bank',
          }]}
        />
      )
      expect(screen.queryByText('Running...')).not.toBeInTheDocument()
    })
  })

  describe('Failed Status', () => {
    it('shows failed execution', () => {
      render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'search',
            status: 'failed',
            description: 'Search failed',
          }]}
        />
      )
      expect(screen.getByText('Search failed')).toBeInTheDocument()
    })

    it('shows red X for failed', () => {
      const { container } = render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'search',
            status: 'failed',
            description: 'Search failed',
          }]}
        />
      )
      expect(container.querySelector('.text-red-500')).toBeInTheDocument()
    })
  })

  describe('Detail Text', () => {
    it('shows detail when provided', () => {
      render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'search',
            status: 'completed',
            description: 'Search completed',
            detail: 'Found 5 results',
          }]}
        />
      )
      expect(screen.getByText('Found 5 results')).toBeInTheDocument()
    })

    it('does not show detail section when not provided', () => {
      render(
        <ToolExecutionDisplay
          executions={[{
            tool: 'search',
            status: 'completed',
            description: 'Search completed',
          }]}
        />
      )
      // Only description should be visible
      const items = screen.getAllByText(/Search completed/)
      expect(items.length).toBe(1)
    })
  })

  describe('Multiple Executions', () => {
    it('renders multiple executions', () => {
      render(
        <ToolExecutionDisplay
          executions={[
            { tool: 'search', status: 'completed', description: 'Searched memory' },
            { tool: 'add', status: 'running', description: 'Adding to bank' },
            { tool: 'score', status: 'failed', description: 'Scoring failed' },
          ]}
        />
      )

      expect(screen.getByText('Searched memory')).toBeInTheDocument()
      expect(screen.getByText('Adding to bank')).toBeInTheDocument()
      expect(screen.getByText('Scoring failed')).toBeInTheDocument()
    })

    it('shows correct status icons for each', () => {
      const { container } = render(
        <ToolExecutionDisplay
          executions={[
            { tool: 'search', status: 'completed', description: 'Searched memory' },
            { tool: 'add', status: 'running', description: 'Adding to bank' },
            { tool: 'score', status: 'failed', description: 'Scoring failed' },
          ]}
        />
      )

      expect(container.querySelector('.text-green-500')).toBeInTheDocument()
      expect(container.querySelector('.animate-spin')).toBeInTheDocument()
      expect(container.querySelector('.text-red-500')).toBeInTheDocument()
    })
  })
})
