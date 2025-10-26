// Development-only logger to prevent console spam in production
const isDev = process.env.NODE_ENV === 'development';

export const logger = {
  log: (...args: any[]) => {
    if (isDev) {
      console.log(...args);
    }
  },

  error: (...args: any[]) => {
    // Always log errors, but with stack trace only in dev
    if (isDev) {
      console.error(...args);
    } else if (args[0]) {
      console.error(args[0]?.message || args[0]);
    }
  },

  warn: (...args: any[]) => {
    if (isDev) {
      console.warn(...args);
    }
  },

  info: (...args: any[]) => {
    if (isDev) {
      console.info(...args);
    }
  },

  debug: (...args: any[]) => {
    if (isDev) {
      console.debug(...args);
    }
  }
};

// Export as default for easier import
export default logger;