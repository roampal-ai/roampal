import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ContextIndicator } from '../../components/ContextIndicator'

describe('ContextIndicator', () => {
  describe('No Context', () => {
    it('returns null when no context', () => {
      const { container } = render(<ContextIndicator />)
      expect(container.firstChild).toBeNull()
    })

    it('returns null for empty context', () => {
      const { container } = render(<ContextIndicator context={{}} />)
      expect(container.firstChild).toBeNull()
    })
  })

  describe('Current File', () => {
    it('shows current file when provided', () => {
      render(<ContextIndicator context={{ current_file: 'app.tsx' }} />)
      expect(screen.getByText('app.tsx')).toBeInTheDocument()
    })
  })

  describe('Error', () => {
    it('shows error indicator when last_error present', () => {
      render(<ContextIndicator context={{ last_error: 'Some error' }} />)
      expect(screen.getByText('Error present')).toBeInTheDocument()
    })
  })

  describe('Code', () => {
    it('shows code ready when has_code is true', () => {
      render(<ContextIndicator context={{ has_code: true }} />)
      expect(screen.getByText('Code ready')).toBeInTheDocument()
    })

    it('hides code indicator when has_code is false', () => {
      render(<ContextIndicator context={{ has_code: false }} />)
      expect(screen.queryByText('Code ready')).not.toBeInTheDocument()
    })
  })

  describe('Memories', () => {
    it('shows memory count when greater than 0', () => {
      render(<ContextIndicator context={{ memories_retrieved: 5 }} />)
      expect(screen.getByText('5 memories')).toBeInTheDocument()
    })

    it('hides memory count when 0', () => {
      render(<ContextIndicator context={{ memories_retrieved: 0 }} />)
      expect(screen.queryByText(/memories/)).not.toBeInTheDocument()
    })
  })

  describe('Multiple Indicators', () => {
    it('shows multiple indicators together', () => {
      render(
        <ContextIndicator
          context={{
            current_file: 'test.ts',
            has_code: true,
            memories_retrieved: 3,
          }}
        />
      )
      expect(screen.getByText('test.ts')).toBeInTheDocument()
      expect(screen.getByText('Code ready')).toBeInTheDocument()
      expect(screen.getByText('3 memories')).toBeInTheDocument()
    })
  })

  describe('Custom className', () => {
    it('applies custom className', () => {
      const { container } = render(
        <ContextIndicator context={{ has_code: true }} className="custom-class" />
      )
      expect(container.firstChild).toHaveClass('custom-class')
    })
  })
})
