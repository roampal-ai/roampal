import { create } from 'zustand';
import { flushSync } from 'react-dom';
import { useTauri } from '../config/roampal';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

// Get Tauri functions once at module level
const { readFile, writeFile, listFiles, isTauri } = useTauri();

// localStorage key for persisting active conversation across page refresh
const CONVERSATION_ID_KEY = 'roampal_active_conversation';

// Import Tauri API functions only when needed
let tauriInvoke: any = null;
let tauriListen: any = null;

if (isTauri && typeof window !== 'undefined') {
  import('@tauri-apps/api/tauri').then(module => {
    tauriInvoke = module.invoke;
  });
  import('@tauri-apps/api/event').then(module => {
    tauriListen = module.listen;
  });
}

export interface ChatSession {
  id: string;
  name: string;
  timestamp: number;  // Unix timestamp in seconds
  messageCount: number;
  createdAt?: number;  // Unix timestamp in milliseconds
  lastActivity?: number;  // Unix timestamp in milliseconds
}

interface ChatState {
  // Conversation state (STANDARDIZED)
  conversationId: string | null;  // Primary identifier

  // Messages
  messages: any[];
  sessions: ChatSession[];  // Clean array, no wrapper object

  // Processing state
  isProcessing: boolean;
  processingStage: string;
  processingStatus: string | null;  // New: for transparency status line
  isStreaming: boolean;  // NEW: for token-by-token streaming

  // Transparency state
  thinkingMessage: string | null;
  citations: Array<{
    citation_id: number;
    source: string;
    confidence: number;
    collection: string;
    text?: string;
    doc_id?: string;
  }>;

  // Connection state
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
  websocket: WebSocket | null;
  actionStatuses: Map<string, any>;
  abortController: AbortController | null;

  // Memory state
  activeMemories: any[];
  memories: any[];
  lastMemoryUpdate?: number;

  // v0.2.9: Race condition guard
  _currentSwitchId: number;

  // Mode system removed - RoamPal always uses memory

  // Actions
  sendMessage: (text: string) => Promise<void>;
  createConversation: () => Promise<void>;
  switchConversation: (newId: string) => Promise<void>;
  searchMemory: (query: string) => Promise<void>;
  clearSession: () => Promise<void>;
  initialize: () => Promise<void>;
  getCurrentState: () => any;
  getCurrentMessages: () => any[];
  getCurrentProcessingState: () => { isProcessing: boolean; processingStage: string };
  loadSessions: () => Promise<void>;
  loadSession: (conversationId: string) => Promise<void>;
  deleteSession: (conversationId: string) => Promise<void>;
  updateSessionTitle: (sessionId: string, title: string) => Promise<void>;
  initWebSocket: () => void;
  closeWebSocket: () => void;
  cancelProcessing: () => void;
}

// Quick intent detection for immediate UI feedback
const getQuickIntent = (text: string): string => {
  const lower = text.toLowerCase();
  if (lower.includes('find') || lower.includes('search') || lower.includes('look for') || lower.includes('where')) {
    return 'Searching...';
  }
  if (lower.includes('fix') || lower.includes('repair') || lower.includes('solve')) {
    return 'Fixing issue...';
  }
  if (lower.includes('create') || lower.includes('make') || lower.includes('new') || lower.includes('add')) {
    return 'Creating...';
  }
  if (lower.includes('edit') || lower.includes('change') || lower.includes('modify') || lower.includes('update')) {
    return 'Making changes...';
  }
  if (lower.includes('run') || lower.includes('execute')) {
    return 'Running...';
  }
  if (lower.includes('delete') || lower.includes('remove')) {
    return 'Removing...';
  }
  if (lower.includes('refactor')) {
    return 'Refactoring code...';
  }
  if (lower.includes('list') || lower.includes('show')) {
    return 'Gathering information...';
  }
  if (lower.includes('analyze') || lower.includes('explain') || lower.includes('understand')) {
    return 'Analyzing...';
  }
  if (lower.includes('test')) {
    return 'Testing...';
  }
  if (lower.includes('debug')) {
    return 'Debugging...';
  }
  if (lower.includes('install')) {
    return 'Installing...';
  }
  if (lower.includes('deploy') || lower.includes('publish')) {
    return 'Deploying...';
  }
  if (lower.includes('optimize') || lower.includes('improve')) {
    return 'Optimizing...';
  }
  return 'Processing...';
};

export const useChatStore = create<ChatState>()((set, get) => ({
  // Initial state
  conversationId: null,
  messages: [],
  sessions: [],
  isProcessing: false,
  thinkingMessage: null,
  citations: [],
  processingStage: 'idle',
  processingStatus: null,
  isStreaming: false,
  connectionStatus: 'disconnected',
  websocket: null,
  actionStatuses: new Map(),
  abortController: null,
  activeMemories: [],
  memories: [],
  lastMemoryUpdate: undefined,
  _currentSwitchId: 0,  // v0.2.9: Guard against stale conversation switches

  // Create new conversation (server-generated ID)
  createConversation: async () => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/chat/create-conversation`, {
        method: 'POST'
      });

      if (response.ok) {
        const data = await response.json();
        const newConversationId = data.conversation_id;

        // Immediately add the new conversation to the sessions list
        const state = get();
        const currentSessions = state.sessions;
        const newSession: ChatSession = {
          id: newConversationId,
          name: 'New Chat',
          timestamp: Date.now() / 1000,  // Unix timestamp in seconds
          messageCount: 0,
          createdAt: Date.now(),
          lastActivity: Date.now()
        };

        // Add new session at the beginning of the list
        set({
          sessions: [newSession, ...currentSessions]
        });

        // Switch to the new conversation
        await get().switchConversation(newConversationId);

        return newConversationId;
      }
    } catch (error) {
      console.error('[createConversation] Failed to create conversation:', error);
      // Fallback to client-generated ID
      const fallbackId = `conv_${Date.now()}_client`;
      set({ conversationId: fallbackId });
      return fallbackId;
    }
  },

  // Switch conversations (with memory promotion)
  switchConversation: async (newConversationId: string) => {
    const state = get();
    const oldConversationId = state.conversationId;

    // Don't switch to same conversation
    if (oldConversationId === newConversationId) return;

    // v0.2.9: Generate unique switch ID to guard against race conditions
    const switchId = Date.now();
    set({ _currentSwitchId: switchId });

    try {
      // Cancel any active streaming before switch
      if (state.abortController) {
        state.abortController.abort();

        // Mark current streaming message as incomplete
        set((state) => {
          const messages = [...state.messages];
          const lastMsg = messages[messages.length - 1];
          if (lastMsg && lastMsg.streaming) {
            messages[messages.length - 1] = {
              ...lastMsg,
              streaming: false,
              content: (lastMsg.content || '') + '\n\n_[Conversation switched during streaming]_'
            };
          }
          return { messages, abortController: null };
        });
      }

      // Close WebSocket before switch
      get().closeWebSocket();

      // Notify backend of conversation switch for memory promotion
      if (oldConversationId) {
        const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/chat/switch-conversation`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            old_conversation_id: oldConversationId,
            new_conversation_id: newConversationId
          })
        });

        if (response.ok) {
          const data = await response.json();
        }
      }

      // Load messages for the new conversation
      let loadedMessages: any[] = [];
      try {
        const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/sessions/${newConversationId}`);
        if (response.ok) {
          const data = await response.json();

          // Map backend messages to UI format (same logic as loadSession)
          loadedMessages = data.messages.map((msg: any) => {
            let thinking = null;
            let content = msg.content || '';

            // Extract thinking from various formats
            if (msg.metadata?.hybridEvents) {
              const thinkingEvents = msg.metadata.hybridEvents.filter((e: any) => e.type === 'thinking');
              if (thinkingEvents.length > 0) {
                thinking = thinkingEvents.map((e: any) => e.content).join('\n');
              }
            }

            if (!thinking && msg.metadata?.thinking) {
              thinking = msg.metadata.thinking;
            }

            if (!thinking && msg.role === 'assistant' && content) {
              const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
              if (thinkMatch) {
                thinking = thinkMatch[1].trim();
                content = content.replace(/<think>[\s\S]*?<\/think>/, '').trim();
              }
            }

            // Parse timestamp - backend sends local time without timezone info
            // Replace 'T' with space to force browser to parse as local time, not UTC
            const parsedTimestamp = msg.timestamp
              ? new Date(msg.timestamp.replace('T', ' '))
              : new Date();

            // Extract toolExecutions from metadata.toolResults
            let toolExecutions = undefined;
            if (msg.metadata?.toolResults && Array.isArray(msg.metadata.toolResults)) {
              toolExecutions = msg.metadata.toolResults.map((result: any) => ({
                tool: result.tool || 'search_memory',
                status: 'completed',  // All loaded tools are already completed
                resultCount: result.result_count,
                arguments: result.arguments
              }));
            }

            return {
              id: `msg-${msg.timestamp || Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              sender: msg.role === 'user' ? 'user' : 'assistant',
              content: content,
              timestamp: parsedTimestamp,
              thinking: thinking,
              toolExecutions: toolExecutions,
              streaming: false
            };
          });
        }
      } catch (loadError) {
        console.error('[switchConversation] Failed to load messages:', loadError);
        // Continue with empty messages if loading fails
      }

      // v0.2.9: Guard against stale switch - another switch may have started
      if (get()._currentSwitchId !== switchId) {
        console.log('[switchConversation] Stale switch detected, discarding results');
        return;  // Another switch happened, don't overwrite with old data
      }

      // Update local state with loaded messages
      set({
        conversationId: newConversationId,
        messages: loadedMessages,
        processingStatus: null,
        thinkingMessage: null,
        processingStage: 'idle',
        isProcessing: false
      });

      // Persist conversation ID to localStorage
      localStorage.setItem(CONVERSATION_ID_KEY, newConversationId);

      // Reinitialize WebSocket with new conversation ID
      get().initWebSocket();

    } catch (error) {
      console.error('[switchConversation] Error switching conversation:', error);
      // v0.2.9: Also check for stale switch in error case
      if (get()._currentSwitchId !== switchId) {
        console.log('[switchConversation] Stale switch detected in error handler');
        return;
      }
      // Still update local state even if backend fails
      set({
        conversationId: newConversationId,
        messages: [],
        isProcessing: false
      });
    }
  },

  // WebSocket Management (UNIFIED ENDPOINT)
  initWebSocket: () => {
    const state = get();

    // Close existing connection if any
    if (state.websocket) {
      state.websocket.close();
      set({ websocket: null });
    }

    // Skip WebSocket init if no conversation yet (lazy creation - wait for first message)
    if (!state.conversationId) {
      console.log('[WebSocket] Skipping init - no conversation yet (will init on first message)');
      set({ connectionStatus: 'disconnected' });
      return;
    }

    const conversationId = state.conversationId;
    // Use single unified WebSocket endpoint
    const ws = new WebSocket(`${ROAMPAL_CONFIG.WS_URL}/ws/conversation/${conversationId}`);

    // Setup heartbeat to keep connection alive
    let heartbeatInterval: NodeJS.Timeout | null = null;

    ws.onopen = () => {
      set({
        websocket: ws,
        connectionStatus: 'connected',
        actionStatuses: new Map()
      });

      // Send initial handshake
      ws.send(JSON.stringify({
        type: 'handshake',
        conversation_id: conversationId
      }));

      // Start heartbeat - ping every 30 seconds
      heartbeatInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        } else {
          if (heartbeatInterval) clearInterval(heartbeatInterval);
        }
      }, 30000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        switch (data.type) {
          case 'title':
            // Backend has generated a title - refresh sessions to show it
            console.log('[WebSocket] Title generated by backend:', data.title);
            get().loadSessions().then(() => {
              console.log('[WebSocket] Sessions refreshed after title generation');
            });
            break;

          case 'action-status-update':
            // Handle LLM explanations during tool chaining
            if (data.action_type === 'llm_explanation') {
              // Add the LLM explanation as a temporary processing message
              set({
                processingStatus: data.description,
                processingStage: 'processing'
              });

              // Also dispatch for any components listening
              window.dispatchEvent(new CustomEvent('llm-explanation', {
                detail: {
                  description: data.description,
                  metadata: data.metadata
                }
              }));
            } else {
              // Handle other action status updates
              window.dispatchEvent(new CustomEvent('action-status-update', {
                detail: {
                  id: data.action_id,
                  type: data.status,
                  action: data.action_type,
                  details: data.detail || data.description,
                  metadata: data.metadata
                }
              }));
            }
            break;

          case 'action_status':
            const statuses = new Map(get().actionStatuses);
            statuses.set(data.action, data);
            set({ actionStatuses: statuses });

            // Update processing status for transparency
            if (data.action === 'search_memory') {
              set({ processingStatus: `searching memory...${data.detail ? ` for "${data.detail}"` : ''}` });
            } else if (data.action === 'web_search') {
              set({ processingStatus: 'searching web...' });
            } else if (data.action === 'execute_code') {
              set({ processingStatus: 'running code...' });
            } else if (data.action === 'read_file') {
              set({ processingStatus: 'reading file...' });
            } else if (data.action === 'write_file') {
              set({ processingStatus: 'writing file...' });
            } else if (data.action === 'auto_promotion') {
              set({ processingStatus: 'promoting valuable memories...' });
            } else if (data.action === 'memory_promotion') {
              set({ processingStatus: 'switching conversations...' });
            }

            // Dispatch custom event for ActionStatus component
            window.dispatchEvent(new CustomEvent('action-status-update', {
              detail: {
                id: data.action_id || `${data.action}_${Date.now()}`,
                type: data.status || 'executing',
                action: data.action,
                details: data.detail || data.description
              }
            }));

            if (data.status === 'completed' || data.status === 'failed') {
              const allComplete = Array.from(statuses.values()).every(
                s => s.status === 'completed' || s.status === 'failed'
              );
              if (allComplete) {
                set({
                  processingStage: 'idle',
                  isProcessing: false,
                  processingStatus: null  // Clear status when all actions complete
                });
              }
            }
            break;

          case 'memory_update':
            // Real-time memory updates
            set({
              memories: data.memories || [],
              lastMemoryUpdate: Date.now()
            });
            break;

          case 'conversation_switched':
            // Handle backend-initiated conversation switch
            set({ conversationId: data.new_conversation_id });
            break;

          case 'error':
            console.error('[WebSocket] Error:', data.message);
            break;

          // NEW: Handle streaming message types from backend
          case 'status':
            console.log('[WebSocket] Status:', data.status, data.message);
            set({
              processingStage: data.status === 'thinking' ? 'thinking' : 'idle',
              processingStatus: data.message || null
            });
            break;

          case 'content':
            console.log('[WebSocket] Response content received');
            // Update the last assistant message with the response
            set((state) => {
              const messages = [...state.messages];
              const lastMessage = messages[messages.length - 1];
              if (lastMessage && lastMessage.sender === 'assistant') {
                lastMessage.content = data.content;
                lastMessage.citations = data.citations;
              }
              return {
                messages,
                processingStage: 'idle',
                isProcessing: false,
                processingStatus: null
              };
            });
            break;

          case 'complete':
            console.log('[WebSocket] Generation complete');
            set({
              processingStage: 'idle',
              isProcessing: false,
              processingStatus: null
            });
            break;

          // NEW STREAMING MESSAGES (2025-10-16)
          case 'stream_start':
            console.log('[WebSocket] Stream starting');
            // Clean up any empty assistant messages, but don't create placeholder yet
            set((state) => {
              const messages = [...state.messages];

              // Remove any empty assistant messages from previous failed attempts
              while (messages.length > 0 &&
                     messages[messages.length - 1]?.sender === 'assistant' &&
                     (!messages[messages.length - 1]?.content || messages[messages.length - 1]?.content === '')) {
                console.log('[WebSocket] Removing empty assistant message from previous attempt');
                messages.pop();
              }

              const lastMsg = messages[messages.length - 1];
              console.log('[WebSocket] Last message after cleanup:', lastMsg?.sender, 'Total messages:', messages.length);

              return {
                messages,
                processingStage: 'streaming',
                processingStatus: 'Thinking...',
                isStreaming: true
              };
            });
            break;

          case 'token':
            // Lazily create assistant message on first token, then append subsequent tokens
            // v0.2.5 RESTORED: Stream tokens for interleaved tool/response display
            console.log('[WebSocket] Received token:', data.content?.substring(0, 50));
            set((state) => {
              const messages = [...state.messages];
              const lastMsg = messages[messages.length - 1];
              const tokenContent = data.content || '';

              if (!tokenContent) {
                return state; // Skip empty tokens
              }

              if (!lastMsg || lastMsg.sender !== 'assistant' || !lastMsg.streaming) {
                console.log('[WebSocket] Creating assistant message on first token');
                // v0.2.5: Initialize events timeline for chronological rendering
                const firstTextEvent = {
                  type: 'text' as const,
                  timestamp: Date.now(),
                  data: { chunk: tokenContent, firstChunk: true }
                };
                messages.push({
                  id: `msg-${Date.now()}`,
                  sender: 'assistant',
                  content: tokenContent,
                  streaming: true,
                  timestamp: new Date(),
                  thinking: null,
                  toolExecutions: [],
                  events: [firstTextEvent]
                });
              } else {
                // Immutable update - preserve all properties including toolExecutions and events
                const newContent = (lastMsg.content || '') + tokenContent;
                // v0.2.5: Add text event to timeline (only if we have events - timeline mode)
                const existingEvents = lastMsg.events || [];
                messages[messages.length - 1] = {
                  ...lastMsg,
                  content: newContent,
                  toolExecutions: lastMsg.toolExecutions,
                  events: existingEvents  // Don't add every token to events (too granular)
                };
                console.log('[WebSocket] Updated message content, now:', newContent.substring(0, 50), 'Length:', newContent.length);
              }

              return {
                messages,
                processingStatus: 'Streaming...'
              };
            });
            break;

          // v0.2.5: Response case - buffered complete response
          case 'response':
            console.log('[WebSocket] Full response received:', data.content?.substring(0, 50));
            set((state) => {
              const messages = [...state.messages];
              let lastMsg = messages[messages.length - 1];

              if (lastMsg && lastMsg.sender === 'assistant') {
                messages[messages.length - 1] = {
                  ...lastMsg,
                  content: data.content,
                  streaming: false
                };
              } else {
                // Edge case: response arrives without prior message
                messages.push({
                  id: `msg-${Date.now()}`,
                  sender: 'assistant' as const,
                  content: data.content,
                  streaming: false,
                  timestamp: new Date(),
                  toolExecutions: []
                });
              }
              return {
                messages,
                isStreaming: false,
                processingStatus: null
              };
            });
            break;

          // v0.2.5: Thinking state events - show "Thinking..." status
          case 'thinking_start':
            console.log('[WebSocket] LLM thinking started');
            set({ processingStatus: 'Thinking...' });
            break;

          case 'thinking_end':
            console.log('[WebSocket] LLM thinking ended');
            set({ processingStatus: 'Streaming...' });
            break;

          case 'tool_start':
            console.log('[WebSocket] Tool starting:', data.tool);
            set((state) => {
              const messages = [...state.messages];
              let lastMsg = messages[messages.length - 1];

              // v0.2.5: Create tool data and timeline event
              const toolData = {
                tool: data.tool,
                status: 'running' as const,
                arguments: data.arguments
              };
              const toolEvent = {
                type: 'tool_execution' as const,
                timestamp: Date.now(),
                data: toolData
              };

              // Lazy message creation: create assistant message if tool_start arrives before first token
              if (!lastMsg || lastMsg.sender !== 'assistant' || !lastMsg.streaming) {
                console.log('[WebSocket] Creating assistant message for tool execution');
                messages.push({
                  id: `msg-${Date.now()}`,
                  sender: 'assistant' as const,
                  content: '',
                  streaming: true,
                  timestamp: new Date(),
                  thinking: null,
                  toolExecutions: [toolData],
                  events: [toolEvent],  // v0.2.5: Initialize timeline with tool event
                  _lastTextEndIndex: 0  // v0.2.5: Track text position for segment capture
                });
              } else {
                // v0.2.5: Capture text segment BEFORE tool starts (true chronological interleaving)
                const currentContent = lastMsg.content || '';
                const lastTextEndIndex = lastMsg._lastTextEndIndex || 0;
                const newTextSegment = currentContent.slice(lastTextEndIndex);

                const existingTools = lastMsg.toolExecutions || [];
                const existingEvents = [...(lastMsg.events || [])];

                // If there's new text since last boundary, add a text_segment event BEFORE the tool
                if (newTextSegment.length > 0) {
                  existingEvents.push({
                    type: 'text_segment' as const,
                    timestamp: Date.now() - 1,  // Just before tool
                    data: { content: newTextSegment }
                  });
                  console.log('[WebSocket] Captured text segment before tool:', newTextSegment.substring(0, 50));
                }

                // Add tool event
                existingEvents.push(toolEvent);

                messages[messages.length - 1] = {
                  ...lastMsg,
                  toolExecutions: [...existingTools, toolData],
                  events: existingEvents,
                  _lastTextEndIndex: currentContent.length  // Update boundary marker
                };
              }

              return {
                messages,
                processingStatus: `Using ${data.tool}...`
              };
            });
            break;

          case 'tool_complete':
            console.log('[WebSocket] Tool completed:', data.tool);
            set((state) => {
              const messages = [...state.messages];
              let lastMsg = messages[messages.length - 1];

              // Ensure assistant message exists (handle edge case where tool_complete arrives first)
              if (!lastMsg || lastMsg.sender !== 'assistant') {
                console.log('[WebSocket] Creating assistant message for tool completion (edge case)');
                messages.push({
                  id: `msg-${Date.now()}`,
                  sender: 'assistant' as const,
                  content: '',
                  streaming: true,
                  timestamp: new Date(),
                  thinking: null,
                  toolExecutions: []
                });
              } else if (lastMsg.toolExecutions) {
                // FIX: Immutable update - map over tools and create new array
                const updatedTools = lastMsg.toolExecutions.map((t: any) =>
                  t.tool === data.tool && t.status === 'running'
                    ? { ...t, status: 'completed', resultCount: data.result_count, resultPreview: data.result_preview }
                    : t
                );
                messages[messages.length - 1] = {
                  ...lastMsg,
                  toolExecutions: updatedTools
                };
              }

              return { messages };
            });
            break;

          case 'validation_error':
            // Handle validation errors without creating messages
            console.log('[WebSocket] Validation error:', data.message);
            set((state) => {
              const messages = [...state.messages];
              // Remove the user message that triggered the validation error
              if (messages.length > 0 && messages[messages.length - 1]?.sender === 'user') {
                messages.pop();
                console.log('[WebSocket] Removed user message after validation error');
              }
              return {
                messages,
                isStreaming: false,
                processingStage: 'idle',
                processingStatus: null
              };
            });
            break;

          case 'stream_complete':
            console.log('[WebSocket] Stream complete');
            set((state) => {
              const messages = [...state.messages];
              const lastMsg = messages[messages.length - 1];
              console.log('[WebSocket] Final message state:', {
                sender: lastMsg?.sender,
                streaming: lastMsg?.streaming,
                contentLength: lastMsg?.content?.length,
                content: lastMsg?.content?.substring(0, 50)
              });

              // Dispatch memory update event if backend signals it (independent of streaming state)
              if (lastMsg?.sender === 'assistant' && data.memory_updated) {
                console.log('[WebSocket] Memory updated, triggering refresh event');
                window.dispatchEvent(new CustomEvent('memoryUpdated', {
                  detail: { timestamp: data.timestamp || new Date().toISOString() }
                }));
              }

              // Handle citations FIRST - they arrive after streaming ends
              if (lastMsg?.sender === 'assistant' && data.citations && data.citations.length > 0) {
                messages[messages.length - 1] = {
                  ...lastMsg,
                  citations: data.citations
                };
                console.log('[WebSocket] Set', data.citations.length, 'citations');
              }

              // Refresh lastMsg reference after citation update
              let updatedLastMsg = messages[messages.length - 1];

              // v0.2.5: Capture trailing text segment (after last tool)
              if (updatedLastMsg?.sender === 'assistant' && updatedLastMsg.events?.length > 0) {
                const currentContent = updatedLastMsg.content || '';
                const lastTextEndIndex = updatedLastMsg._lastTextEndIndex || 0;
                const trailingText = currentContent.slice(lastTextEndIndex);

                if (trailingText.length > 0) {
                  const existingEvents = [...(updatedLastMsg.events || [])];
                  existingEvents.push({
                    type: 'text_segment' as const,
                    timestamp: Date.now(),
                    data: { content: trailingText }
                  });
                  messages[messages.length - 1] = {
                    ...updatedLastMsg,
                    events: existingEvents
                  };
                  updatedLastMsg = messages[messages.length - 1];  // Refresh reference
                  console.log('[WebSocket] Captured trailing text segment:', trailingText.substring(0, 50));
                }
              }

              // THEN handle streaming state and message cleanup
              if (updatedLastMsg?.sender === 'assistant' && updatedLastMsg.streaming) {
                // Only remove message if it has no content AND no tool executions
                const hasContent = updatedLastMsg.content && updatedLastMsg.content.trim() !== '';
                const hasTools = updatedLastMsg.toolExecutions && updatedLastMsg.toolExecutions.length > 0;

                if (!hasContent && !hasTools) {
                  messages.pop();
                  console.log('[WebSocket] Removed empty assistant message');
                } else {
                  // FIX: Immutable update - preserve all properties including toolExecutions AND citations
                  // v0.2.5: Also mark all tools as completed when stream ends (safety net)
                  const completedTools = updatedLastMsg.toolExecutions?.map((t: any) => ({
                    ...t,
                    status: 'completed'  // Force all tools to completed on stream end
                  })) || [];
                  messages[messages.length - 1] = {
                    ...updatedLastMsg,
                    streaming: false,
                    toolExecutions: completedTools
                  };
                  console.log('[WebSocket] Keeping message - content:', updatedLastMsg.content?.length || 0, 'tools:', completedTools.length);
                }
              }

              console.log('[WebSocket] Returning', messages.length, 'messages');
              return {
                messages,
                isStreaming: false,
                isProcessing: false,
                processingStage: 'idle',
                processingStatus: null
              };
            });
            break;
        }
      } catch (e) {
        console.error('[WebSocket] Error parsing message:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
      set({ connectionStatus: 'error' });
    };

    ws.onclose = () => {
      // Clean up heartbeat
      if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = null;
      }

      set({
        websocket: null,
        connectionStatus: 'disconnected'
      });

      // Auto-reconnect after 3 seconds only if no websocket exists
      setTimeout(() => {
        const currentState = get();
        if (currentState.connectionStatus === 'disconnected' &&
            currentState.conversationId &&
            !currentState.websocket) {
          currentState.initWebSocket();
        }
      }, 3000);
    };

    set({ websocket: ws });
  },

  closeWebSocket: () => {
    const state = get();
    if (state.websocket) {
      state.websocket.close();
      set({ websocket: null, connectionStatus: 'disconnected' });
    }
  },

  cancelProcessing: async () => {
    const state = get();

    // 1. Abort HTTP polling
    if (state.abortController) {
      state.abortController.abort();
      set({ abortController: null });
    }

    // 2. Call backend to cancel task
    const conversationId = state.conversationId;
    if (conversationId) {
      try {
        await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/agent/cancel/${conversationId}`, {
          method: 'POST',
        });
        console.log('[Cancel] Backend task cancelled');
      } catch (e) {
        console.warn('[Cancel] Failed to cancel backend task:', e);
      }
    }

    // 3. Reset UI state
    set({
      isProcessing: false,
      processingStage: 'idle',
      processingStatus: null,
      actionStatuses: new Map()
    });

    // 4. Mark streaming messages as complete
    set((state) => {
      const messages = [...state.messages];
      const lastMsg = messages[messages.length - 1];
      if (lastMsg && lastMsg.streaming) {
        lastMsg.streaming = false;
        if (!lastMsg.content) {
          lastMsg.content = '(Cancelled)';
        }
      }
      return { messages };
    });
  },

  // Send message with proper conversation tracking
  sendMessage: async (text: string) => {
    const state = get();

    // Abort any existing request before starting new one (prevents rapid-fire race)
    if (state.abortController) {
      console.log('[sendMessage] Aborting previous request');
      state.abortController.abort();
      set({ abortController: null });
    }

    // Create conversation if this is the first message (lazy creation)
    let conversationId = state.conversationId;
    if (!conversationId) {
      console.log('[sendMessage] First message - creating conversation');
      await get().createConversation();
      conversationId = get().conversationId;

      // Initialize WebSocket now that we have a conversation
      get().initWebSocket();

      // Wait for WebSocket to connect (max 2 seconds)
      let attempts = 0;
      while (get().connectionStatus !== 'connected' && attempts < 20) {
        await new Promise(resolve => setTimeout(resolve, 100));
        attempts++;
      }
      if (get().connectionStatus === 'connected') {
        console.log('[sendMessage] WebSocket connected after', attempts * 100, 'ms');
      } else {
        console.warn('[sendMessage] WebSocket not connected, falling back to polling');
      }
    }

    // Add user message to UI
    const userMessage: any = {
      id: `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      sender: 'user' as const,
      content: text,
      timestamp: new Date(),
    };

    set((state) => ({
      messages: [...state.messages, userMessage],
      isProcessing: true,
      processingStage: 'thinking',
      processingStatus: null,
    }));

    // NON-STREAMING: Simple HTTP request/response
    const assistantMsgId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Add placeholder assistant message
    set((state) => ({
      messages: [...state.messages, {
        id: assistantMsgId,
        sender: 'assistant' as const,
        content: '',
        timestamp: new Date(),
        thinking: null,
        citations: []
      }],
      processingStage: 'thinking',
      processingStatus: getQuickIntent(text),
    }));

    try {
      const abortController = new AbortController();
      set((state) => ({
        ...state,
        abortController
      }));

      const requestBody = {
        message: text,
        conversation_id: conversationId,
        transparency_level: localStorage.getItem('transparencyLevel') || 'summary'
      };

      // Start async generation task
      console.log('[POLLING] Starting async generation');
      const startResponse = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/agent/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: abortController.signal,
      });

      if (!startResponse.ok) {
        throw new Error(`HTTP error! status: ${startResponse.status}`);
      }

      const { conversation_id } = await startResponse.json();
      console.log('[POLLING] Task started, conversation_id:', conversation_id);

      // Poll for progress every 500ms
      const pollInterval = 500;
      let lastStatus = 'thinking';

      const poll = async () => {
        try {
          const progressResponse = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/agent/progress/${conversation_id}`, {
            signal: abortController.signal,
          });

          if (!progressResponse.ok) {
            console.error('[POLLING] Progress check failed:', progressResponse.status);
            return;
          }

          const progress = await progressResponse.json();
          console.log('[POLLING] Progress:', progress.status);

          // Update UI with progress
          set((state) => {
            const messages = [...state.messages];
            const idx = messages.findIndex(m => m.id === assistantMsgId);

            if (idx >= 0) {
              messages[idx] = {
                ...messages[idx],
                thinking: progress.thinking || null,
                toolExecutions: progress.tool_executions || undefined,
              };
            }

            return {
              messages,
              processingStage: progress.status === 'thinking' ? 'thinking' : 
                              progress.status === 'tool_running' ? 'tool_execution' : 'processing',
              thinkingMessage: progress.thinking || null,
            };
          });

          // Check if complete
          if (progress.status === 'complete') {
            console.log('[POLLING] Generation complete');
            
            // Update with final response
            set((state) => {
              const messages = [...state.messages];
              const idx = messages.findIndex(m => m.id === assistantMsgId);

              if (idx >= 0) {
                messages[idx] = {
                  ...messages[idx],
                  content: progress.response || '',
                  thinking: progress.thinking || null,
                  toolExecutions: progress.tool_executions || undefined,
                };
              }

              return {
                messages,
                isProcessing: false,
                processingStage: 'idle',
                processingStatus: null,
                abortController: null,
                thinkingMessage: null,
              };
            });

            // Refresh sessions
            await get().loadSessions();
            
          } else if (progress.status === 'error') {
            console.error('[POLLING] Generation error:', progress.error);
            throw new Error(progress.error || 'Generation failed');
            
          } else {
            // Continue polling
            setTimeout(poll, pollInterval);
          }

        } catch (error: any) {
          if (error.name !== 'AbortError') {
            console.error('[POLLING] Poll error:', error);
            throw error;
          }
        }
      };

      // Start polling
      await poll();

    } catch (error: any) {
      const isAbort = error.name === 'AbortError';

      set({
        isProcessing: false,
        processingStage: 'idle',
        processingStatus: null,
        abortController: null
      });

      if (isAbort) {
        console.log('[sendMessage] Message sending was cancelled');
        return;
      }

      console.error('Failed to send message:', error);

      // Show error message
      set((state) => ({
        messages: [...state.messages, {
          id: `msg-${Date.now()}`,
          sender: 'system' as const,
          content: `Error: ${error.message}`,
          timestamp: new Date(),
        }]
      }));
    }
  },

  // Mode system removed - RoamPal always uses memory

  searchMemory: async (query: string) => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (response.ok) {
        const data = await response.json();
        set({
          memories: data.results || [],
          lastMemoryUpdate: Date.now()
        });
      }
    } catch (error) {
      console.error('[searchMemory] Failed:', error);
    }
  },

  clearSession: async () => {
    const state = get();
    const oldConversationId = state.conversationId;

    // Don't create conversation yet - wait for first message (lazy creation)
    // Conversation will be created automatically in sendMessage() when user sends first message

    // Close WebSocket since we're leaving the conversation
    get().closeWebSocket();

    // Notify backend of switch to promote old conversation's memories (if it exists)
    if (oldConversationId) {
      try {
        await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/chat/switch-conversation`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            old_conversation_id: oldConversationId,
            new_conversation_id: null  // Switching to null (new chat pending)
          })
        });
      } catch (error) {
        console.error('[clearSession] Failed to notify backend of switch:', error);
      }
    }

    // Clear the UI state for the new conversation
    set({
      conversationId: null,  // Null until first message sent
      messages: [],
      isProcessing: false,
      processingStage: 'idle',
      citations: [],
      thinkingMessage: null,
      processingStatus: null
    });

    // Clear persisted conversation ID from localStorage
    localStorage.removeItem(CONVERSATION_ID_KEY);
  },

  initialize: async () => {
    // Get initial feature mode
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/chat/feature-mode`);
      if (response.ok) {
        const data = await response.json();
        // Mode system removed
      }
    } catch (error) {
      console.error('[initialize] Failed to get feature mode:', error);
    }

    set({ connectionStatus: 'connecting' });

    // Load existing sessions
    await get().loadSessions();

    // Restore last active conversation from localStorage
    const lastConversationId = localStorage.getItem(CONVERSATION_ID_KEY);
    if (lastConversationId) {
      console.log('[initialize] Restoring last conversation:', lastConversationId);
      try {
        // Load the conversation (will set conversationId and messages)
        await get().loadSession(lastConversationId);
      } catch (error) {
        console.error('[initialize] Failed to restore conversation:', error);
        // Clear invalid conversation ID
        localStorage.removeItem(CONVERSATION_ID_KEY);
      }
    }

    // Initialize WebSocket only if not already connected
    if (!get().websocket) {
      get().initWebSocket();
    }
  },

  getCurrentState: () => {
    const state = get();
    return {
      conversationId: state.conversationId,
      messages: state.messages,
      memories: state.memories
    };
  },

  getCurrentMessages: () => get().messages,

  getCurrentProcessingState: () => {
    const state = get();
    return {
      isProcessing: state.isProcessing,
      processingStage: state.processingStage,
    };
  },

  loadSessions: async () => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/sessions/list`);
      if (response.ok) {
        const data = await response.json();
        const formattedSessions: ChatSession[] = data.sessions.map((session: any) => ({
          id: session.session_id || session.conversation_id,
          name: session.title || session.first_message || `Session ${session.session_id?.slice(-6)}`,
          timestamp: session.timestamp,  // Keep as unix timestamp (seconds)
          messageCount: session.message_count || 0,
          createdAt: session.timestamp ? session.timestamp * 1000 : undefined,  // Convert to ms for compatibility
          lastActivity: session.last_updated ? session.last_updated * 1000 : undefined,  // Convert to ms
        }));

        // Store sessions as clean array
        set({ sessions: formattedSessions });
      }
    } catch (error) {
      console.error('[loadSessions] Failed:', error);
    }
  },

  loadSession: async (conversationId: string) => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/sessions/${conversationId}`);
      if (response.ok) {
        const data = await response.json();
        const messages = data.messages.map((msg: any) => {
          let content = msg.content;
          let thinking = null;

          // Extract thinking from hybridEvents (new format)
          if (msg.metadata?.hybridEvents) {
            const thinkingEvent = msg.metadata.hybridEvents.find((e: any) => e.type === 'thinking');
            if (thinkingEvent) {
              thinking = thinkingEvent.content;
            }
          }

          // For backward compatibility: extract thinking from metadata.thinking (old format)
          if (!thinking && msg.metadata?.thinking) {
            thinking = msg.metadata.thinking;
          }

          // For backward compatibility: extract thinking from content if it's embedded (very old format)
          if (!thinking && msg.role === 'assistant' && content) {
            const thinkMatch = content.match(/<(?:think|antml:thinking)>([\s\S]*?)<\/(?:think|antml:thinking)>/);
            if (thinkMatch) {
              thinking = thinkMatch[1].trim();
              // Remove thinking tags from content
              content = content.replace(/<(?:think|antml:thinking)>[\s\S]*?<\/(?:think|antml:thinking)>\s*/g, '').trim();
            }
          }

          // Parse timestamp as local time (backend sends local time without TZ info)
          const msgTimestamp = msg.timestamp
            ? new Date(msg.timestamp.replace('T', ' '))
            : new Date();

          return {
            id: `msg-${Date.now()}-${Math.random()}`,
            sender: msg.role === 'user' ? 'user' : 'assistant',
            content: content,
            timestamp: msgTimestamp,
            // Add thinking if found (from hybridEvents, metadata, or extracted from content)
            ...(thinking ? { thinking: thinking } : {}),
            // Preserve any additional metadata - hybridEvents are in metadata
            ...(msg.metadata?.hybridEvents ? { hybridEvents: msg.metadata.hybridEvents } : {}),
            ...(msg.actions ? { actions: msg.actions } : {}),
            ...(msg.citations ? { citations: msg.citations } : {}),
            // FIX: Load toolEvents from session metadata for display after reload
            ...(msg.metadata?.toolEvents ? {
              toolExecutions: msg.metadata.toolEvents.map((e: any) => ({
                tool: e.tool,
                status: 'completed',
                resultCount: e.result_count
              }))
            } : {})
          };
        });

        // Set conversation ID and messages in one update (don't call switchConversation which clears messages!)
        set({
          conversationId: conversationId,
          messages: messages,
          processingStatus: null,
          thinkingMessage: null,
          processingStage: 'idle',
          isProcessing: false
        });

        // Persist conversation ID to localStorage
        localStorage.setItem(CONVERSATION_ID_KEY, conversationId);

        // Reinitialize WebSocket with new conversation ID
        get().closeWebSocket();
        get().initWebSocket();

      }
    } catch (error) {
      console.error('[loadSession] Failed:', error);
    }
  },

  deleteSession: async (conversationId: string) => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/sessions/${conversationId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        // Reload sessions
        await get().loadSessions();
      }
    } catch (error) {
      console.error('[deleteSession] Failed:', error);
    }
  },

  updateSessionTitle: async (sessionId: string, title: string) => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/sessions/${sessionId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: title })
      });

      if (response.ok) {
        await get().loadSessions();
      }
    } catch (error) {
      console.error('[updateSessionTitle] Failed:', error);
    }
  },

  // NOTE: Title generation is now handled entirely by backend during streaming
  // The /generate-title endpoint still exists for manual regeneration if needed
}));