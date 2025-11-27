import { ROAMPAL_CONFIG } from '../config/roampal';
/**
 * Service for fetching and caching model context window information
 * Uses the new /api/model/contexts endpoint to get dynamic context limits
 */

const API_URL = ROAMPAL_CONFIG.apiUrl;

interface ModelContextInfo {
  current: number;
  default: number;
  max: number;
  is_override: boolean;
}

interface AllModelContexts {
  status: string;
  contexts: Record<string, {
    default: number;
    max: number;
  }>;
}

class ModelContextService {
  private cache: Map<string, ModelContextInfo> = new Map();
  private allContextsCache: Record<string, { default: number; max: number }> | null = null;
  private cacheExpiry: number = Date.now();
  private readonly CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

  /**
   * Get context info for all models
   */
  async getAllContexts(): Promise<Record<string, { default: number; max: number }>> {
    // Check cache
    if (this.allContextsCache && Date.now() < this.cacheExpiry) {
      return this.allContextsCache;
    }

    try {
      const response = await fetch(`${API_URL}/api/model/contexts`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        console.error('Failed to fetch model contexts:', response.statusText);
        return this.getFallbackContexts();
      }

      const data: AllModelContexts = await response.json();
      this.allContextsCache = data.contexts;
      this.cacheExpiry = Date.now() + this.CACHE_DURATION;
      return data.contexts;
    } catch (error) {
      console.error('Error fetching model contexts:', error);
      return this.getFallbackContexts();
    }
  }

  /**
   * Get context info for a specific model
   */
  async getModelContext(modelName: string): Promise<ModelContextInfo> {
    // Check individual cache
    const cached = this.cache.get(modelName);
    if (cached && Date.now() < this.cacheExpiry) {
      return cached;
    }

    try {
      const response = await fetch(`${API_URL}/api/model/context/${encodeURIComponent(modelName)}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        console.error(`Failed to fetch context for ${modelName}:`, response.statusText);
        return this.getFallbackContextForModel(modelName);
      }

      const data = await response.json();
      const contextInfo: ModelContextInfo = {
        current: data.current,
        default: data.default,
        max: data.max,
        is_override: data.is_override,
      };

      this.cache.set(modelName, contextInfo);
      return contextInfo;
    } catch (error) {
      console.error(`Error fetching context for ${modelName}:`, error);
      return this.getFallbackContextForModel(modelName);
    }
  }

  /**
   * Get the current context size for a model (what's actually being used)
   */
  async getContextSize(modelName: string): Promise<number> {
    const info = await this.getModelContext(modelName);
    return info.current;
  }

  /**
   * Update context size for a model
   */
  async setContextSize(modelName: string, contextSize: number): Promise<boolean> {
    try {
      const response = await fetch(`${API_URL}/api/model/context/${encodeURIComponent(modelName)}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ context_size: contextSize }),
      });

      if (!response.ok) {
        console.error(`Failed to set context for ${modelName}:`, response.statusText);
        return false;
      }

      // Clear cache for this model
      this.cache.delete(modelName);
      return true;
    } catch (error) {
      console.error(`Error setting context for ${modelName}:`, error);
      return false;
    }
  }

  /**
   * Reset context size to default for a model
   */
  async resetContextSize(modelName: string): Promise<boolean> {
    try {
      const response = await fetch(`${API_URL}/api/model/context/${encodeURIComponent(modelName)}`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        console.error(`Failed to reset context for ${modelName}:`, response.statusText);
        return false;
      }

      // Clear cache for this model
      this.cache.delete(modelName);
      return true;
    } catch (error) {
      console.error(`Error resetting context for ${modelName}:`, error);
      return false;
    }
  }

  /**
   * Clear all caches
   */
  clearCache() {
    this.cache.clear();
    this.allContextsCache = null;
    this.cacheExpiry = Date.now();
  }

  /**
   * Get fallback contexts when API is unavailable
   * These match the defaults in config/model_contexts.py
   */
  private getFallbackContexts(): Record<string, { default: number; max: number }> {
    return {
      'gpt-oss': { default: 32768, max: 128000 },
      'llama3.1': { default: 32768, max: 131072 },
      'llama3.2': { default: 32768, max: 131072 },
      'llama3.3': { default: 32768, max: 131072 },
      'qwen2.5': { default: 32768, max: 32768 },
      'qwen3': { default: 32768, max: 32768 },
      'mistral': { default: 16384, max: 32768 },
      'mixtral': { default: 32768, max: 32768 },
      'phi3': { default: 32768, max: 128000 },
      'phi4': { default: 16384, max: 128000 },
      'phi': { default: 16384, max: 128000 },
      'dolphin3': { default: 16384, max: 32768 },
      'firefunction': { default: 16384, max: 32768 },
      'command-r': { default: 32768, max: 128000 },
      'gemma3': { default: 32768, max: 128000 },
    };
  }

  /**
   * Get fallback context for a specific model
   */
  private getFallbackContextForModel(modelName: string): ModelContextInfo {
    const modelLower = modelName.toLowerCase();
    const fallbacks = this.getFallbackContexts();

    // Find matching model prefix
    for (const [prefix, config] of Object.entries(fallbacks)) {
      if (modelLower.includes(prefix)) {
        return {
          current: config.default,
          default: config.default,
          max: config.max,
          is_override: false,
        };
      }
    }

    // Ultimate fallback
    return {
      current: 8192,
      default: 8192,
      max: 32768,
      is_override: false,
    };
  }
}

// Export singleton instance
export const modelContextService = new ModelContextService();