import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { CodeChangePreview } from '../../components/CodeChangePreview'

/**
 * CodeChangePreview Tests
 *
 * Tests the code change preview component with apply/skip actions.
 */

describe('CodeChangePreview', () => {
  const mockChanges = [
    {
      change_id: 'c1',
      file_path: 'src/app.ts',
      diff: '+const x = 1;\n-const y = 2;',
      description: 'Add constant',
      status: 'pending' as const,
      lines_added: 1,
      lines_removed: 1,
    },
    {
      change_id: 'c2',
      file_path: 'src/utils.ts',
      diff: '+function helper() {}',
      description: 'Add helper function',
      status: 'pending' as const,
      lines_added: 1,
    },
  ]

  const defaultProps = {
    changes: mockChanges,
    onApply: vi.fn(),
    onSkip: vi.fn(),
    onApplyAll: vi.fn(),
  }

  describe('Empty State', () => {
    it('returns null when no changes', () => {
      const { container } = render(
        <CodeChangePreview {...defaultProps} changes={[]} />
      )
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Header', () => {
    it('shows Code Changes Preview title', () => {
      render(<CodeChangePreview {...defaultProps} />)
      expect(screen.getByText('Code Changes Preview')).toBeInTheDocument()
    })

    it('shows pending count', () => {
      render(<CodeChangePreview {...defaultProps} />)
      expect(screen.getByText('(2 pending)')).toBeInTheDocument()
    })

    it('shows Apply All button', () => {
      render(<CodeChangePreview {...defaultProps} />)
      expect(screen.getByText('Apply All')).toBeInTheDocument()
    })
  })

  describe('Changes List', () => {
    it('shows file paths', () => {
      render(<CodeChangePreview {...defaultProps} />)
      expect(screen.getByText('src/app.ts')).toBeInTheDocument()
      expect(screen.getByText('src/utils.ts')).toBeInTheDocument()
    })

    it('shows descriptions', () => {
      render(<CodeChangePreview {...defaultProps} />)
      expect(screen.getByText('Add constant')).toBeInTheDocument()
      expect(screen.getByText('Add helper function')).toBeInTheDocument()
    })

    it('shows lines added/removed', () => {
      render(<CodeChangePreview {...defaultProps} />)
      // Both changes have lines_added: 1, so we should find multiple +1 elements
      const addedElements = screen.getAllByText('+1')
      expect(addedElements.length).toBeGreaterThan(0)
    })
  })

  describe('Actions', () => {
    it('shows Apply button for each change', () => {
      render(<CodeChangePreview {...defaultProps} />)
      const applyButtons = screen.getAllByText('Apply')
      expect(applyButtons.length).toBe(2)
    })

    it('shows Skip button for each change', () => {
      render(<CodeChangePreview {...defaultProps} />)
      const skipButtons = screen.getAllByText('Skip')
      expect(skipButtons.length).toBe(2)
    })

    it('calls onApply when Apply clicked', () => {
      render(<CodeChangePreview {...defaultProps} />)

      const applyButtons = screen.getAllByText('Apply')
      fireEvent.click(applyButtons[0])

      expect(defaultProps.onApply).toHaveBeenCalledWith('c1')
    })

    it('calls onSkip when Skip clicked', () => {
      render(<CodeChangePreview {...defaultProps} />)

      const skipButtons = screen.getAllByText('Skip')
      fireEvent.click(skipButtons[0])

      expect(defaultProps.onSkip).toHaveBeenCalledWith('c1')
    })

    it('calls onApplyAll when Apply All clicked', () => {
      render(<CodeChangePreview {...defaultProps} />)

      fireEvent.click(screen.getByText('Apply All'))

      expect(defaultProps.onApplyAll).toHaveBeenCalled()
    })
  })

  describe('Expand/Collapse', () => {
    it('expands diff view when change clicked', () => {
      render(<CodeChangePreview {...defaultProps} />)

      // Click on file path to expand
      fireEvent.click(screen.getByText('src/app.ts'))

      // Diff content should be visible
      expect(screen.getByText(/const x = 1/)).toBeInTheDocument()
    })
  })

  describe('Applied/Skipped States', () => {
    it('shows Applied label after applying', () => {
      render(<CodeChangePreview {...defaultProps} />)

      const applyButtons = screen.getAllByText('Apply')
      fireEvent.click(applyButtons[0])

      expect(screen.getByText('Applied')).toBeInTheDocument()
    })

    it('shows Skipped label after skipping', () => {
      render(<CodeChangePreview {...defaultProps} />)

      const skipButtons = screen.getAllByText('Skip')
      fireEvent.click(skipButtons[0])

      expect(screen.getByText('Skipped')).toBeInTheDocument()
    })
  })

  describe('Config Warning', () => {
    it('shows warning for config files', () => {
      const configChanges = [{
        ...mockChanges[0],
        file_path: '.env.local',
      }]

      render(<CodeChangePreview {...defaultProps} changes={configChanges} />)

      expect(screen.getByText('Review configuration changes carefully')).toBeInTheDocument()
    })
  })
})
