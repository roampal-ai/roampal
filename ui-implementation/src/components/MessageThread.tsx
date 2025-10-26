import React from 'react';
import { MessageGroup } from './MessageGroup';

interface MessageThreadProps {
  messages: Message[];
  onMemoryClick: (memoryId: string) => void;
  onCommandClick: (command: string) => void;
}

export interface Message {
  id: string;
  sender: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  status?: 'sending' | 'sent' | 'error';
  memories?: MemoryReference[];
  citations?: Citation[];
  attachments?: Attachment[];
  processing?: ProcessingState;
  actions?: Action[];
  code_changes?: CodeChange[];
  streaming?: boolean;
}

interface Action {
  action_id: string;
  action_type: string;
  description: string;
  detail?: string;
  status: 'executing' | 'completed' | 'failed';
  timestamp: string;
  metadata?: Record<string, any>;
}

interface CodeChange {
  change_id: string;
  file_path: string;
  diff: string;
  description: string;
  status: 'pending' | 'applied' | 'skipped';
  risk_level: 'low' | 'medium' | 'high';
}

interface MemoryReference {
  id: string;
  count: number;
  preview?: string[];
}

interface Citation {
  citation_id?: number;
  id?: string;
  title?: string;
  url?: string;
  source?: string;
  confidence?: number;
  collection?: string;
  text?: string;
}

interface Attachment {
  id: string;
  name: string;
  size: number;
  type: string;
}

interface ProcessingState {
  stage: 'memory_retrieval' | 'web_search' | 'thinking' | 'generating';
  message?: string;
}

export const MessageThread: React.FC<MessageThreadProps> = ({
  messages,
  onMemoryClick,
  onCommandClick,
}) => {
  // Group consecutive messages from the same sender
  const groupedMessages = groupMessages(messages);
  
  return (
    <div className="space-y-6">
      {groupedMessages.map((group, index) => (
        <MessageGroup
          key={`group-${index}`}
          messages={group.messages}
          sender={group.sender}
          timestamp={group.timestamp}
          onMemoryClick={onMemoryClick}
          onCommandClick={onCommandClick}
        />
      ))}
      
      {/* Auto-scroll anchor */}
      <div id="messages-end" />
    </div>
  );
};

// Helper function to group consecutive messages
function groupMessages(messages: Message[]) {
  const groups: {
    sender: string;
    timestamp: Date;
    messages: Message[];
  }[] = [];
  
  messages.forEach((message, index) => {
    const lastGroup = groups[groups.length - 1];
    const timeDiff = lastGroup
      ? message.timestamp.getTime() - lastGroup.timestamp.getTime()
      : Infinity;
    
    // Start new group if:
    // - Different sender
    // - More than 5 minutes apart
    // - First message
    if (
      !lastGroup ||
      lastGroup.sender !== message.sender ||
      timeDiff > 5 * 60 * 1000
    ) {
      groups.push({
        sender: message.sender,
        timestamp: message.timestamp,
        messages: [message],
      });
    } else {
      lastGroup.messages.push(message);
    }
  });
  
  return groups;
}