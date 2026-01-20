import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'

/**
 * Logger Tests
 *
 * Tests the development-only logger utility.
 */

describe('logger', () => {
  const originalEnv = process.env.NODE_ENV
  const originalConsole = {
    log: console.log,
    error: console.error,
    warn: console.warn,
    info: console.info,
    debug: console.debug,
  }

  beforeEach(() => {
    // Mock console methods
    console.log = vi.fn()
    console.error = vi.fn()
    console.warn = vi.fn()
    console.info = vi.fn()
    console.debug = vi.fn()

    // Reset module cache to re-import with new env
    vi.resetModules()
  })

  afterEach(() => {
    // Restore console
    console.log = originalConsole.log
    console.error = originalConsole.error
    console.warn = originalConsole.warn
    console.info = originalConsole.info
    console.debug = originalConsole.debug

    // Restore env
    process.env.NODE_ENV = originalEnv
  })

  describe('Development Mode', () => {
    beforeEach(() => {
      process.env.NODE_ENV = 'development'
    })

    it('log calls console.log in dev mode', async () => {
      const { logger } = await import('../../utils/logger')
      logger.log('test message')
      expect(console.log).toHaveBeenCalledWith('test message')
    })

    it('warn calls console.warn in dev mode', async () => {
      const { logger } = await import('../../utils/logger')
      logger.warn('warning message')
      expect(console.warn).toHaveBeenCalledWith('warning message')
    })

    it('info calls console.info in dev mode', async () => {
      const { logger } = await import('../../utils/logger')
      logger.info('info message')
      expect(console.info).toHaveBeenCalledWith('info message')
    })

    it('debug calls console.debug in dev mode', async () => {
      const { logger } = await import('../../utils/logger')
      logger.debug('debug message')
      expect(console.debug).toHaveBeenCalledWith('debug message')
    })

    it('error calls console.error in dev mode', async () => {
      const { logger } = await import('../../utils/logger')
      logger.error('error message')
      expect(console.error).toHaveBeenCalledWith('error message')
    })
  })

  describe('Production Mode', () => {
    beforeEach(() => {
      process.env.NODE_ENV = 'production'
    })

    it('log does not call console.log in production', async () => {
      const { logger } = await import('../../utils/logger')
      logger.log('test message')
      expect(console.log).not.toHaveBeenCalled()
    })

    it('warn does not call console.warn in production', async () => {
      const { logger } = await import('../../utils/logger')
      logger.warn('warning message')
      expect(console.warn).not.toHaveBeenCalled()
    })

    it('error still logs error messages in production', async () => {
      const { logger } = await import('../../utils/logger')
      logger.error('error message')
      expect(console.error).toHaveBeenCalledWith('error message')
    })
  })

  describe('Logger Export', () => {
    it('exports logger as default', async () => {
      const loggerModule = await import('../../utils/logger')
      expect(loggerModule.default).toBeDefined()
      expect(loggerModule.default.log).toBeInstanceOf(Function)
    })

    it('exports logger as named export', async () => {
      const { logger } = await import('../../utils/logger')
      expect(logger).toBeDefined()
      expect(logger.log).toBeInstanceOf(Function)
    })
  })
})