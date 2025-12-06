/**
 * Enhancement patch for useChatStore to handle new streaming events
 * This should be integrated into useChatStore.ts
 */

import { create } from 'zustand';
import { useChatStore } from './useChatStore';

interface ToolExecution {
  tool: string;
  status: 'running' | 'completed' | 'failed';
  description: string;
  detail?: string;
  metadata?: Record<string, any>;
}

interface EnhancedMessage {
  id: string;
  sender: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  streaming?: boolean;
  thinking?: string;
  toolExecutions?: ToolExecution[];
  codeBlocks?: Array<{ language: string; code: string }>;
  citations?: Array<{
    citation_id: number;
    source: string;
    confidence: number;
    collection: string;
    text?: string;
  }>;
  toolsUsed?: Array<{
    tool: string;
    success: boolean;
    result?: string;
  }>;
}

// Handler for new event types in the streaming response
export function handleEnhancedStreamingEvent(
  event: any,
  assistantMsgId: string,
  messages: EnhancedMessage[]
): EnhancedMessage[] {
  const updatedMessages = [...messages];
  const assistantMsg = updatedMessages.find(
    m => m.id === assistantMsgId && m.sender === 'assistant'
  );

  if (!assistantMsg) return updatedMessages;

  switch (event.type) {
    // thinking case removed (v0.2.5) - feature deprecated

    case 'tool_execution':
      // Add tool execution to the list
      if (!assistantMsg.toolExecutions) {
        assistantMsg.toolExecutions = [];
      }
      assistantMsg.toolExecutions.push({
        tool: event.tool,
        status: event.status,
        description: event.description,
        detail: event.detail,
        metadata: event.metadata
      });
      break;

    case 'response_with_code':
      // Response with structured code blocks
      assistantMsg.content = event.content;
      assistantMsg.codeBlocks = event.code_blocks;
      assistantMsg.thinking = undefined; // Clear thinking
      assistantMsg.toolExecutions = undefined; // Clear tool executions
      break;

    case 'token':
      // Regular text response
      assistantMsg.content = event.content;
      assistantMsg.thinking = undefined; // Clear thinking
      break;

    case 'complete':
      // Add metadata when complete
      assistantMsg.streaming = false;
      assistantMsg.citations = event.citations;
      assistantMsg.toolsUsed = event.tools_used;
      assistantMsg.toolExecutions = undefined; // Clear execution display
      break;

    default:
      break;
  }

  return updatedMessages;
}

// Example integration into existing sendMessage function:
/*
// In useChatStore.ts, modify the stream parsing:

for (const line of lines) {
  if (line.startsWith('data: ')) {
    const data = line.slice(6);
    if (data === '[DONE]') continue;

    try {
      const event = JSON.parse(data);

      // Use the enhanced handler
      set((state) => ({
        messages: handleEnhancedStreamingEvent(
          event,
          assistantMsgId,
          state.messages
        )
      }));

    } catch (error) {
      console.error('Failed to parse SSE event:', error);
    }
  }
}
*/

// Export a hook for using enhanced message display
export function useEnhancedMessageDisplay(messageId: string) {
  const message = useChatStore((state: any) =>
    state.messages.find((m: any) => m.id === messageId)
  ) as EnhancedMessage | undefined;

  return {
    content: message?.content || '',
    thinking: message?.thinking,
    toolExecutions: message?.toolExecutions || [],
    codeBlocks: message?.codeBlocks || [],
    citations: message?.citations || [],
    toolsUsed: message?.toolsUsed || [],
    isStreaming: message?.streaming || false
  };
}