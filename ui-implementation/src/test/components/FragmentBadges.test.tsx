import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { FragmentBadges } from '../../components/FragmentBadges'

/**
 * FragmentBadges Tests
 *
 * Tests the memory fragments sidebar component.
 */

// Mock dependencies
vi.mock('../../utils/fetch', () => ({
  apiFetch: vi.fn().mockRejectedValue(new Error('API not available')),
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('FragmentBadges', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('Initial Render', () => {
    it('renders the component', () => {
      const { container } = render(<FragmentBadges />)
      expect(container.firstChild).not.toBeNull()
    })

    it('shows memory fragments header', () => {
      render(<FragmentBadges />)
      expect(screen.getByText('Memory Fragments')).toBeInTheDocument()
    })

    it('shows sort buttons', () => {
      render(<FragmentBadges />)
      expect(screen.getByText('Recent')).toBeInTheDocument()
      expect(screen.getByText('Score')).toBeInTheDocument()
    })

    it('shows loading spinner initially', () => {
      const { container } = render(<FragmentBadges />)
      expect(container.querySelector('.animate-spin')).toBeInTheDocument()
    })
  })

  describe('After Loading', () => {
    it('shows mock fragments after loading', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        expect(screen.getByText(/DataLoader pattern/)).toBeInTheDocument()
      })
    })

    it('shows multiple fragments', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        expect(screen.getByText(/DataLoader pattern/)).toBeInTheDocument()
        expect(screen.getByText(/Redis caching/)).toBeInTheDocument()
      })
    })

    it('shows score badges', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        expect(screen.getByText('95%')).toBeInTheDocument()
        expect(screen.getByText('88%')).toBeInTheDocument()
      })
    })

    it('shows type badges (global/private)', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        const globalBadges = screen.getAllByText('global')
        const privateBadges = screen.getAllByText('private')
        expect(globalBadges.length).toBeGreaterThan(0)
        expect(privateBadges.length).toBeGreaterThan(0)
      })
    })

    it('shows content after loading', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        // Should show fragment content from mock data
        expect(screen.getByText(/DataLoader pattern/)).toBeInTheDocument()
      })
    })
  })

  describe('Sorting', () => {
    it('recent sort is active by default', () => {
      render(<FragmentBadges />)

      const recentButton = screen.getByText('Recent').closest('button')
      expect(recentButton).toHaveClass('text-blue-400')
    })

    it('switches to score sort when clicked', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        expect(screen.getByText(/DataLoader pattern/)).toBeInTheDocument()
      })

      fireEvent.click(screen.getByText('Score'))

      const scoreButton = screen.getByText('Score').closest('button')
      expect(scoreButton).toHaveClass('text-blue-400')
    })
  })

  describe('Click Handler', () => {
    it('calls onFragmentClick when clicking a fragment', async () => {
      const onFragmentClick = vi.fn()
      render(<FragmentBadges onFragmentClick={onFragmentClick} />)

      await waitFor(() => {
        expect(screen.getByText(/DataLoader pattern/)).toBeInTheDocument()
      })

      // Find and click the fragment
      const fragment = screen.getByText(/DataLoader pattern/).closest('div[class*="cursor-pointer"]')
      if (fragment) {
        fireEvent.click(fragment)
        expect(onFragmentClick).toHaveBeenCalledWith('f1')
      }
    })
  })

  describe('Footer Stats', () => {
    it('shows global count', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        expect(screen.getByText(/3 global/)).toBeInTheDocument()
      })
    })

    it('shows private count', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        expect(screen.getByText(/2 private/)).toBeInTheDocument()
      })
    })

    it('shows total count', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        expect(screen.getByText(/5 total/)).toBeInTheDocument()
      })
    })
  })

  describe('Score Colors', () => {
    it('renders high score badges with green color class', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        const highScore = screen.getByText('95%')
        expect(highScore).toHaveClass('text-green-400')
      })
    })

    it('renders medium score badges with blue color class', async () => {
      render(<FragmentBadges />)

      await waitFor(() => {
        const medScore = screen.getByText('88%')
        expect(medScore).toHaveClass('text-blue-400')
      })
    })
  })
})
