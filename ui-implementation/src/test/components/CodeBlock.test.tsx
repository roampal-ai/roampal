import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { CodeBlock } from '../../components/CodeBlock'

describe('CodeBlock', () => {
  beforeEach(() => {
    // Mock clipboard API
    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    })
  })

  describe('Rendering', () => {
    it('displays the code content', () => {
      render(<CodeBlock code="console.log('hello')" />)
      expect(screen.getByText("console.log('hello')")).toBeInTheDocument()
    })

    it('shows language label', () => {
      render(<CodeBlock code="print('hello')" language="python" />)
      expect(screen.getByText('python')).toBeInTheDocument()
    })

    it('shows default language when not specified', () => {
      render(<CodeBlock code="some code" />)
      expect(screen.getByText('plaintext')).toBeInTheDocument()
    })

    it('shows filename when provided', () => {
      render(<CodeBlock code="code" language="js" filename="app.js" />)
      expect(screen.getByText('app.js')).toBeInTheDocument()
    })
  })

  describe('Copy Functionality', () => {
    it('copies code to clipboard when copy button clicked', async () => {
      render(<CodeBlock code="test code" />)

      const copyButton = screen.getByRole('button', { name: /copy/i })
      await userEvent.click(copyButton)

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith('test code')
    })
  })

  describe('Run Button', () => {
    it('shows run button for Python', () => {
      render(<CodeBlock code="print('hi')" language="python" onRun={vi.fn()} />)
      expect(screen.getByRole('button', { name: /run/i })).toBeInTheDocument()
    })

    it('shows run button for JavaScript', () => {
      render(<CodeBlock code="console.log()" language="javascript" onRun={vi.fn()} />)
      expect(screen.getByRole('button', { name: /run/i })).toBeInTheDocument()
    })

    it('hides run button when onRun not provided', () => {
      render(<CodeBlock code="console.log()" language="javascript" />)
      expect(screen.queryByRole('button', { name: /run/i })).not.toBeInTheDocument()
    })

    it('calls onRun with code and language', async () => {
      const onRun = vi.fn()
      render(<CodeBlock code="test code" language="python" onRun={onRun} />)

      await userEvent.click(screen.getByRole('button', { name: /run/i }))
      expect(onRun).toHaveBeenCalledWith('test code', 'python')
    })
  })

  describe('Apply Button', () => {
    it('shows apply button when filename and onApplyToFile provided', () => {
      render(
        <CodeBlock
          code="code"
          filename="test.js"
          onApplyToFile={vi.fn()}
        />
      )
      expect(screen.getByRole('button', { name: /apply/i })).toBeInTheDocument()
    })

    it('calls onApplyToFile with filename and code', async () => {
      const onApply = vi.fn()
      render(
        <CodeBlock
          code="new code"
          filename="app.js"
          onApplyToFile={onApply}
        />
      )

      await userEvent.click(screen.getByRole('button', { name: /apply/i }))
      expect(onApply).toHaveBeenCalledWith('app.js', 'new code')
    })
  })
})