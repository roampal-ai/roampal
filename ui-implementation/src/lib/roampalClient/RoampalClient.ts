import { ROAMPAL_CONFIG } from '../../config/roampal';
import {
  SendMessageRequest,
  SendMessageResponse,
  SendMessageResponseSchema,
  MemorySearchResponse,
  MemorySearchResponseSchema,
  MemoryAddResponseSchema,
  ShardsListResponse,
  ShardsListResponseSchema,
  ShardSwitchResponseSchema,
  HealthCheckResponse,
  HealthCheckResponseSchema,
  WSMessage,
  WSMessageSchema,
} from './schemas';

/**
 * RoampalClient - Type-safe SDK for backend communication
 * Handles WebSocket/SSE streaming, retries, and fallback to mock mode
 */
export class RoampalClient {
  private ws: WebSocket | null = null;
  private eventSource: EventSource | null = null;
  private reconnectAttempts = 0;
  private idempotencyCounter = 0;
  private requestCallbacks = new Map<string, (data: any) => void>();
  private mockMode = false;
  
  constructor(private config = ROAMPAL_CONFIG) {
    if (config.ENABLE_DEBUG_LOGGING) {
      console.log('[RoampalClient] Initialized with config:', config);
    }
  }
  
  /**
   * Generate unique idempotency key for requests
   */
  private generateIdempotencyKey(): string {
    return `${Date.now()}-${++this.idempotencyCounter}-${Math.random().toString(36).substr(2, 9)}`;
  }
  
  /**
   * Exponential backoff with jitter
   */
  private getRetryDelay(attempt: number): number {
    const baseDelay = this.config.RETRY_DELAY;
    const maxDelay = baseDelay * Math.pow(2, attempt);
    const jitter = Math.random() * 1000;
    return Math.min(maxDelay + jitter, 30000);
  }
  
  /**
   * HTTP request with retries and type safety
   */
  private async request<T>(
    path: string,
    options: RequestInit = {},
    schema: any,
    retries = 0
  ): Promise<T> {
    const url = `${this.config.API_BASE}${path}`;
    const idempotencyKey = this.generateIdempotencyKey();
    
    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          'X-Idempotency-Key': idempotencyKey,
          ...options.headers,
        },
        signal: AbortSignal.timeout(this.config.REQUEST_TIMEOUT),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      const parsed = schema.safeParse(data);
      
      if (!parsed.success) {
        console.warn('[RoampalClient] Schema validation warning:', parsed.error);
        // Return original data with defaults applied
        return schema.parse(data);
      }
      
      return parsed.data;
    } catch (error) {
      if (retries < this.config.MAX_RETRIES) {
        const delay = this.getRetryDelay(retries);
        console.log(`[RoampalClient] Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        return this.request<T>(path, options, schema, retries + 1);
      }
      
      console.error('[RoampalClient] Request failed:', error);
      throw error;
    }
  }
  
  /**
   * Health check
   */
  async checkHealth(): Promise<HealthCheckResponse> {
    try {
      const response = await fetch(`${this.config.API_BASE}${this.config.HEALTH}`);
      if (response.ok) {
        // Normalize any successful response to our contract
        return { status: 'healthy', timestamp: new Date().toISOString() };
      }
      throw new Error(`Health check failed: ${response.status}`);
    } catch (error) {
      console.warn('[RoampalClient] Health check failed, entering mock mode');
      this.mockMode = true;
      return { status: 'unhealthy', timestamp: new Date().toISOString() };
    }
  }
  
  /**
   * Send message
   */
  async sendMessage(input: {
    text: string;
    shard?: string;
    meta?: any;
  }): Promise<SendMessageResponse> {
    if (this.mockMode) {
      return this.mockSendMessage(input);
    }
    
    const shardId = input.shard || 'roampal';
    const path = this.config.CHAT.SEND.replace('{shard_id}', shardId);
    
    // Backend expects 'input' field, not 'message'
    const payload = {
      input: input.text,
      user_id: input.meta?.user_id || 'default-user',
      session: input.meta?.session_id || `session-${Date.now()}`
    };
    
    try {
      const response = await fetch(`${this.config.API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        throw new Error(`Send message failed: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Normalize Roampal backend response to our contract
      const responseText = typeof data === 'string' ? data : (data.response || data.message || data.output || '');
      
      // Convert injected_fragments to citations if present
      const citations = data.citations || (data.injected_fragments ? 
        data.injected_fragments.map((f: any) => ({
          id: f.chunk_id || `citation-${Date.now()}`,
          title: f.source || 'Memory',
          snippet: f.text || '',
        })) : []);
      
      return {
        id: data.session_id || `msg-${Date.now()}`,
        text: responseText,
        shard: shardId,
        citations,
        request_id: data.session_id || `req-${Date.now()}`,
      };
    } catch (error) {
      console.error('[RoampalClient] Send message error:', error);
      throw error;
    }
  }
  
  /**
   * Stream response via WebSocket or SSE
   */
  async streamResponse(
    requestId: string,
    onToken: (token: string) => void,
    onProcessing?: (state: any) => void,
    onComplete?: () => void
  ): Promise<void> {
    if (this.mockMode) {
      return this.mockStreamResponse(requestId, onToken, onComplete);
    }
    
    if (this.config.TRANSPORT === 'ws') {
      return this.streamViaWebSocket(requestId, onToken, onProcessing, onComplete);
    } else {
      return this.streamViaSSE(requestId, onToken, onProcessing, onComplete);
    }
  }
  
  /**
   * WebSocket streaming
   */
  private async streamViaWebSocket(
    requestId: string,
    onToken: (token: string) => void,
    onProcessing?: (state: any) => void,
    onComplete?: () => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        this.connectWebSocket();
      }
      
      const messageHandler = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          const parsed = WSMessageSchema.safeParse(data);
          
          if (!parsed.success) {
            console.warn('[RoampalClient] Invalid WS message:', parsed.error);
            return;
          }
          
          const message = parsed.data;
          
          if (message.request_id !== requestId) return;
          
          switch (message.type) {
            case 'token':
              onToken(message.payload);
              break;
            case 'processing':
              onProcessing?.(message.payload);
              break;
            case 'complete':
              onComplete?.();
              this.ws?.removeEventListener('message', messageHandler);
              resolve();
              break;
            case 'error':
              console.error('[RoampalClient] Stream error:', message.payload);
              this.ws?.removeEventListener('message', messageHandler);
              reject(new Error(message.payload));
              break;
          }
        } catch (error) {
          console.error('[RoampalClient] WS message parse error:', error);
        }
      };
      
      this.ws?.addEventListener('message', messageHandler);
      
      // Send request ID to subscribe to this stream
      this.ws?.send(JSON.stringify({
        type: 'subscribe',
        request_id: requestId,
      }));
    });
  }
  
  /**
   * SSE streaming
   */
  private async streamViaSSE(
    requestId: string,
    onToken: (token: string) => void,
    onProcessing?: (state: any) => void,
    onComplete?: () => void
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = `${this.config.API_BASE}${this.config.CHAT.STREAM}?request_id=${requestId}`;
      this.eventSource = new EventSource(url);
      
      this.eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'token') {
            onToken(data.payload);
          } else if (data.type === 'processing') {
            onProcessing?.(data.payload);
          } else if (data.type === 'complete') {
            onComplete?.();
            this.eventSource?.close();
            resolve();
          }
        } catch (error) {
          console.error('[RoampalClient] SSE parse error:', error);
        }
      };
      
      this.eventSource.onerror = (error) => {
        console.error('[RoampalClient] SSE error:', error);
        this.eventSource?.close();
        reject(error);
      };
    });
  }
  
  /**
   * Connect WebSocket with reconnection logic
   */
  private connectWebSocket(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;
    
    this.ws = new WebSocket(this.config.WS_URL);
    
    this.ws.onopen = () => {
      console.log('[RoampalClient] WebSocket connected');
      this.reconnectAttempts = 0;
    };
    
    this.ws.onclose = () => {
      if (this.reconnectAttempts < this.config.WS_MAX_RECONNECT_ATTEMPTS) {
        const delay = this.getRetryDelay(this.reconnectAttempts++);
        console.log(`[RoampalClient] WebSocket reconnecting in ${delay}ms...`);
        setTimeout(() => this.connectWebSocket(), delay);
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('[RoampalClient] WebSocket error:', error);
    };
  }
  
  /**
   * Search memory
   */
  async searchMemory(query: string, shardId: string = 'roampal'): Promise<MemorySearchResponse> {
    if (this.mockMode) {
      return this.mockSearchMemory(query);
    }
    
    const path = this.config.MEMORY.SEARCH.replace('{shard_id}', shardId);
    
    try {
      const response = await fetch(`${this.config.API_BASE}${path}`);
      
      if (!response.ok) {
        throw new Error(`Memory search failed: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Normalize memory-viz response to our contract
      const fragments = data.fragments || data.items || [];
      const items = fragments.map((item: any, idx: number) => ({
        id: item.id || item.chunk_id || `mem-${Date.now()}-${idx}`,
        title: item.source || item.title || 'Memory',
        snippet: item.text || item.content || item.snippet || '',
        relevance: item.relevance || item.confidence || 0.5,
        timestamp: item.timestamp || item.created_at || new Date().toISOString(),
      }));
      
      return { items };
    } catch (error) {
      console.error('[RoampalClient] Memory search error:', error);
      return { items: [] };
    }
  }
  
  /**
   * Add memory
   */
  async addMemory(payload: { text: string; tags?: string[]; shard_id?: string }) {
    if (this.mockMode) {
      return { id: `mock-${Date.now()}`, success: true };
    }
    
    return await this.request(
      this.config.MEMORY.ADD,
      {
        method: 'POST',
        body: JSON.stringify(payload),
      },
      MemoryAddResponseSchema
    );
  }
  
  /**
   * List shards
   */
  async listShards(): Promise<ShardsListResponse> {
    if (this.mockMode) {
      return { shards: ['roampal', 'dev', 'creative', 'analyst', 'coach'], active: 'roampal' };
    }
    
    try {
      // Try the shard management endpoint first
      const response = await fetch(`${this.config.API_BASE}${this.config.SHARDS.LIST}`);
      
      if (!response.ok) {
        // Fallback to memory-viz endpoint
        const fallbackResponse = await fetch(`${this.config.API_BASE}${this.config.MEMORY.AVAILABLE_SHARDS}`);
        if (fallbackResponse.ok) {
          const data = await fallbackResponse.json();
          return {
            shards: data.shards || data || [],
            active: data.active || 'roampal',
          };
        }
        throw new Error(`List shards failed: ${response.status}`);
      }
      
      const data = await response.json();
      return {
        shards: data.shards || data || [],
        active: data.active || 'roampal',
      };
    } catch (error) {
      console.error('[RoampalClient] List shards error:', error);
      return { shards: ['roampal'], active: 'roampal' };
    }
  }
  
  /**
   * Switch shard
   */
  async switchShard(name: string) {
    if (this.mockMode) {
      return { active: name, success: true };
    }
    
    const path = this.config.SHARDS.SWITCH.replace('{shard_name}', name);
    
    try {
      const response = await fetch(`${this.config.API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (!response.ok) {
        throw new Error(`Switch shard failed: ${response.status}`);
      }
      
      const data = await response.json();
      return {
        active: data.active || name,
        success: data.success !== false,
      };
    } catch (error) {
      console.error('[RoampalClient] Switch shard error:', error);
      return { active: name, success: false };
    }
  }
  
  /**
   * Cleanup
   */
  disconnect(): void {
    this.ws?.close();
    this.eventSource?.close();
    this.requestCallbacks.clear();
  }
  
  // Mock implementations for fallback mode
  private mockSendMessage(input: any): SendMessageResponse {
    return {
      id: `mock-${Date.now()}`,
      text: `Mock response to: ${input.text}`,
      shard: input.shard || 'roampal',
      request_id: `req-${Date.now()}`,
    };
  }
  
  private async mockStreamResponse(
    requestId: string,
    onToken: (token: string) => void,
    onComplete?: () => void
  ): Promise<void> {
    const mockText = 'This is a mock streamed response. The backend is currently unavailable.';
    const words = mockText.split(' ');
    
    for (const word of words) {
      await new Promise(resolve => setTimeout(resolve, 100));
      onToken(word + ' ');
    }
    
    onComplete?.();
  }
  
  private mockSearchMemory(query: string): MemorySearchResponse {
    return {
      items: [
        {
          id: 'mock-1',
          snippet: `Mock memory result for: ${query}`,
          title: 'Mock Memory',
        },
      ],
    };
  }
}

// Singleton instance
let clientInstance: RoampalClient | null = null;

export function getRoampalClient(): RoampalClient {
  if (!clientInstance) {
    clientInstance = new RoampalClient();
  }
  return clientInstance;
}