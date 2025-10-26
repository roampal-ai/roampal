import { z } from 'zod';

/**
 * Zod schemas for type-safe API communication
 * All schemas use safeParse with sensible defaults
 */

// Base message schema
export const MessageSchema = z.object({
  id: z.string().default(() => `msg-${Date.now()}`),
  text: z.string().default(''),
  shard: z.string().optional(),
  sender: z.enum(['user', 'assistant', 'system']).default('assistant'),
  timestamp: z.string().or(z.date()).transform(val => 
    typeof val === 'string' ? new Date(val) : val
  ).default(() => new Date()),
  citations: z.array(z.object({
    id: z.string(),
    title: z.string().optional(),
    snippet: z.string().optional(),
    url: z.string().optional(),
  })).optional(),
  meta: z.record(z.any()).optional(),
});

export type Message = z.infer<typeof MessageSchema>;

// Send message request
export const SendMessageRequestSchema = z.object({
  message: z.string(),
  shard: z.string().optional(),
  session_id: z.string().optional(),
  meta: z.record(z.any()).optional(),
});

export type SendMessageRequest = z.infer<typeof SendMessageRequestSchema>;

// Send message response
export const SendMessageResponseSchema = z.object({
  id: z.string(),
  text: z.string().optional(),
  shard: z.string().optional(),
  citations: z.array(z.object({
    id: z.string(),
    title: z.string().optional(),
    snippet: z.string().optional(),
  })).optional(),
  meta: z.record(z.any()).optional(),
  request_id: z.string().optional(),
});

export type SendMessageResponse = z.infer<typeof SendMessageResponseSchema>;

// Memory search
export const MemorySearchRequestSchema = z.object({
  q: z.string(),
  limit: z.number().optional().default(10),
  shard_id: z.string().optional(),
});

export const MemorySearchResponseSchema = z.object({
  items: z.array(z.object({
    id: z.string(),
    title: z.string().optional(),
    snippet: z.string(),
    relevance: z.number().optional(),
    timestamp: z.string().optional(),
  })).default([]),
});

export type MemorySearchResponse = z.infer<typeof MemorySearchResponseSchema>;

// Memory add
export const MemoryAddRequestSchema = z.object({
  text: z.string(),
  tags: z.array(z.string()).optional(),
  shard_id: z.string().optional(),
});

export const MemoryAddResponseSchema = z.object({
  id: z.string(),
  success: z.boolean().default(true),
});

// Shards
export const ShardsListResponseSchema = z.object({
  shards: z.array(z.string()).default([]),
  active: z.string().optional(),
});

export type ShardsListResponse = z.infer<typeof ShardsListResponseSchema>;

export const ShardSwitchRequestSchema = z.object({
  name: z.string(),
});

export const ShardSwitchResponseSchema = z.object({
  active: z.string(),
  success: z.boolean().default(true),
});

// Health check
export const HealthCheckResponseSchema = z.object({
  status: z.enum(['healthy', 'degraded', 'unhealthy']).default('healthy'),
  version: z.string().optional(),
  timestamp: z.string().optional(),
});

export type HealthCheckResponse = z.infer<typeof HealthCheckResponseSchema>;

// WebSocket messages
export const WSMessageSchema = z.object({
  type: z.enum(['token', 'message', 'processing', 'memory_update', 'error', 'complete']),
  payload: z.any(),
  request_id: z.string().optional(),
});

export type WSMessage = z.infer<typeof WSMessageSchema>;

// Processing state
export const ProcessingStateSchema = z.object({
  stage: z.enum(['thinking', 'memory_retrieval', 'web_search', 'generating']),
  message: z.string().optional(),
  progress: z.number().optional(),
});

export type ProcessingState = z.infer<typeof ProcessingStateSchema>;