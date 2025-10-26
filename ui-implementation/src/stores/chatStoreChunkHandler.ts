/**
 * Handler for chunked responses in useChatStore
 * Add this to your useChatStore to handle token_chunk events
 */

interface ChunkedMessage {
  chunks: string[];
  isComplete: boolean;
}

const chunkedMessages: Map<string, ChunkedMessage> = new Map();

export function handleChunkedResponse(
  event: any,
  assistantMsgId: string,
  messages: any[]
): any[] {
  const updatedMessages = [...messages];
  const assistantMsg = updatedMessages.find(
    m => m.id === assistantMsgId && m.sender === 'assistant'
  );

  if (!assistantMsg) return updatedMessages;

  if (event.type === 'token_chunk') {
    // Handle chunked response
    let chunkedMsg = chunkedMessages.get(assistantMsgId);

    if (!chunkedMsg) {
      chunkedMsg = { chunks: [], isComplete: false };
      chunkedMessages.set(assistantMsgId, chunkedMsg);
    }

    // Add chunk
    chunkedMsg.chunks.push(event.content);

    // Update message content with all chunks so far
    assistantMsg.content = chunkedMsg.chunks.join('');

    // Check if this is the final chunk
    if (event.final) {
      chunkedMsg.isComplete = true;
      // Clean up
      chunkedMessages.delete(assistantMsgId);
    }
  }

  return updatedMessages;
}

// Add this to your stream parsing logic:
/*
if (event.type === 'token_chunk') {
  set((state) => ({
    messages: handleChunkedResponse(event, assistantMsgId, state.messages)
  }));
} else if (event.type === 'token') {
  // Regular single response
  set((state) => {
    const messages = [...state.messages];
    const assistantMsg = messages.find(m =>
      m.id === assistantMsgId && m.sender === 'assistant'
    );
    if (assistantMsg) {
      assistantMsg.content = event.content;
    }
    return { messages };
  });
}
*/