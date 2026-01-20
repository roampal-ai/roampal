import { describe, it, expect, vi, beforeEach } from 'vitest'
import { validateFile, formatFileSize, getFileIconType } from '../../utils/fileUpload'

/**
 * FileUpload Tests
 *
 * Tests the file upload utility functions.
 */

// Mock dependencies
vi.mock('./logger', () => ({
  default: {
    log: vi.fn(),
    error: vi.fn(),
    warn: vi.fn(),
    info: vi.fn(),
    debug: vi.fn(),
  },
}))

vi.mock('../../config/roampal', () => ({
  ROAMPAL_CONFIG: {
    apiUrl: 'http://localhost:8765',
  },
}))

describe('fileUpload', () => {
  describe('validateFile', () => {
    it('accepts .txt files', () => {
      const file = new File(['content'], 'test.txt', { type: 'text/plain' })
      const result = validateFile(file)
      expect(result.valid).toBe(true)
    })

    it('accepts .md files', () => {
      const file = new File(['# Title'], 'readme.md', { type: 'text/markdown' })
      const result = validateFile(file)
      expect(result.valid).toBe(true)
    })

    it('rejects unsupported file types', () => {
      const file = new File(['binary'], 'image.png', { type: 'image/png' })
      const result = validateFile(file)
      expect(result.valid).toBe(false)
      expect(result.error).toContain('Only')
    })

    it('rejects files over 10MB', () => {
      // Create a file larger than 10MB
      const content = new Array(11 * 1024 * 1024).fill('a').join('')
      const file = new File([content], 'large.txt', { type: 'text/plain' })
      const result = validateFile(file)
      expect(result.valid).toBe(false)
      expect(result.error).toContain('too large')
    })

    it('accepts files under 10MB', () => {
      const content = 'Small file content'
      const file = new File([content], 'small.txt', { type: 'text/plain' })
      const result = validateFile(file)
      expect(result.valid).toBe(true)
    })
  })

  describe('formatFileSize', () => {
    it('formats 0 bytes', () => {
      expect(formatFileSize(0)).toBe('0 B')
    })

    it('formats bytes', () => {
      expect(formatFileSize(500)).toBe('500.0 B')
    })

    it('formats kilobytes', () => {
      expect(formatFileSize(1024)).toBe('1.0 KB')
    })

    it('formats megabytes', () => {
      expect(formatFileSize(1024 * 1024)).toBe('1.0 MB')
    })

    it('formats gigabytes', () => {
      expect(formatFileSize(1024 * 1024 * 1024)).toBe('1.0 GB')
    })

    it('handles fractional sizes', () => {
      expect(formatFileSize(1536)).toBe('1.5 KB')
    })
  })

  describe('getFileIconType', () => {
    it('returns image for image files', () => {
      const file = new File([''], 'photo.jpg', { type: 'image/jpeg' })
      expect(getFileIconType(file)).toBe('image')
    })

    it('returns text for text files', () => {
      const file = new File([''], 'notes.txt', { type: 'text/plain' })
      expect(getFileIconType(file)).toBe('text')
    })

    it('returns text for markdown files', () => {
      const file = new File([''], 'readme.md', { type: 'text/markdown' })
      expect(getFileIconType(file)).toBe('text')
    })

    it('returns document for other files', () => {
      const file = new File([''], 'data.pdf', { type: 'application/pdf' })
      expect(getFileIconType(file)).toBe('document')
    })
  })
})