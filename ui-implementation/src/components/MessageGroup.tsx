import React, { useState } from 'react';
// Removed: ProcessingIndicator, ActionSummary (deleted)
import { MemoryCitation } from './MemoryCitation';
import type { Message } from './MessageThread';

interface MessageGroupProps {
  messages: Message[];
  sender: string;
  timestamp: Date;
  onMemoryClick: (memoryId: string) => void;
  onCommandClick: (command: string) => void;
}

export const MessageGroup: React.FC<MessageGroupProps> = ({
  messages,
  sender,
  timestamp,
  onMemoryClick,
  onCommandClick,
}) => {
  const [expandedMessages, setExpandedMessages] = useState<Set<string>>(new Set());

  const isUser = sender === 'user';
  const isSystem = sender === 'system';

  const toggleExpanded = (messageId: string) => {
    const newExpanded = new Set(expandedMessages);
    if (newExpanded.has(messageId)) {
      newExpanded.delete(messageId);
    } else {
      newExpanded.add(messageId);
    }
    setExpandedMessages(newExpanded);
  };

  // System messages - minimal badge style
  if (isSystem) {
    return (
      <div className="flex justify-center my-2">
        <div className="px-3 py-1.5 bg-zinc-900/50 text-zinc-500 text-xs rounded-full border border-zinc-800">
          {messages[0].content}
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-2xl ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Sleek Message Bubble */}
        <div
          className={`relative px-4 py-3 rounded-2xl ${
            isUser
              ? 'bg-gradient-to-br from-blue-600/20 to-blue-700/20 border border-blue-600/30 ml-12'
              : 'bg-zinc-900/50 border border-zinc-800 mr-12'
          }`}
        >
          {/* Assistant Header */}
          {!isUser && (
            <div className="flex items-center gap-2 mb-2">
              <div className="w-6 h-6 rounded-full bg-gradient-to-br from-blue-500 to-green-500 flex items-center justify-center">
                <span className="text-xs font-bold text-white">R</span>
              </div>
              <span className="text-xs font-medium text-zinc-400">Roampal</span>
              <span className="text-xs text-zinc-600">•</span>
              <span className="text-xs text-zinc-600">{formatTime(timestamp)}</span>
            </div>
          )}

          {/* Messages */}
          {messages.map((message, idx) => (
            <div key={message.id} className={idx > 0 ? 'mt-3 pt-3 border-t border-zinc-800' : ''}>
              {/* Message Content */}
              <div className={`text-sm ${isUser ? 'text-zinc-200' : 'text-zinc-300'}`}>
                {renderMessageContent(message.content, onCommandClick)}
              </div>

              {/* Attachments as small badges */}
              {message.attachments && message.attachments.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {message.attachments.map(attachment => (
                    <div
                      key={attachment.id}
                      className="px-2 py-1 bg-zinc-800/50 rounded-lg flex items-center gap-1.5 border border-zinc-700"
                    >
                      <span className="text-xs">📎</span>
                      <span className="text-xs text-zinc-400">{attachment.name}</span>
                      <span className="text-xs text-zinc-600">
                        {formatFileSize(attachment.size)}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Citations - Memory citations from backend */}
              {message.citations && message.citations.length > 0 && (
                <div className="mt-2 space-y-1">
                  <div className="text-xs text-zinc-500">Sources:</div>
                  {message.citations.map((citation, idx) => (
                    <div
                      key={citation.citation_id || citation.id || idx}
                      className="flex items-start gap-2 text-xs"
                    >
                      <span className="text-blue-400">[{citation.citation_id || idx + 1}]</span>
                      <div className="flex-1">
                        <span className="text-zinc-400">
                          {citation.source || 'Memory'}
                          {citation.collection && ` (${citation.collection})`}
                        </span>
                        {citation.confidence !== undefined && (
                          <span className="ml-2 text-zinc-600">
                            {Math.round(citation.confidence * 100)}% match
                          </span>
                        )}
                        {citation.text && (
                          <div className="mt-1 text-zinc-500 italic truncate">
                            "{citation.text}"
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Processing indicator */}
              {/* Removed: Processing indicator */}

              {/* Removed: Action summary */}
              {!isUser && message.actions && message.actions.length > 0 && (
                <div className="mt-3">
                  {/* ActionSummary removed */}
                </div>
              )}
            </div>
          ))}

          {/* User timestamp - subtle */}
          {isUser && (
            <div className="mt-2 text-xs text-zinc-600 text-right">
              {formatTime(timestamp)}
            </div>
          )}
        </div>

        {/* Tail for speech bubble effect */}
        <div
          className={`absolute w-3 h-3 transform rotate-45 ${
            isUser
              ? 'bg-gradient-to-br from-blue-600/20 to-blue-700/20 border-r border-b border-blue-600/30 -right-1.5 bottom-3'
              : 'bg-zinc-900/50 border-l border-t border-zinc-800 -left-1.5 bottom-3'
          }`}
          style={{
            position: 'absolute',
            [isUser ? 'right' : 'left']: '-6px',
            bottom: '16px',
          }}
        />
      </div>
    </div>
  );
};

// Helper functions
function formatTime(date: Date) {
  return new Date(date).toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  });
}

function formatFileSize(bytes: number) {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
}

function renderMessageContent(content: string, onCommandClick: (cmd: string) => void) {
  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let codeBlock: string[] = [];
  let inCodeBlock = false;
  let codeLanguage = '';

  lines.forEach((line, idx) => {
    // Code blocks
    if (line.startsWith('```')) {
      if (!inCodeBlock) {
        inCodeBlock = true;
        codeLanguage = line.slice(3).trim();
        codeBlock = [];
      } else {
        // End code block
        elements.push(
          <pre
            key={`code-${idx}`}
            className="mt-2 p-3 bg-zinc-900 rounded-lg overflow-x-auto border border-zinc-800"
          >
            <code className="text-xs text-zinc-300">{codeBlock.join('\n')}</code>
          </pre>
        );
        inCodeBlock = false;
        codeBlock = [];
      }
      return;
    }

    if (inCodeBlock) {
      codeBlock.push(line);
      return;
    }

    // Commands - styled as badges
    if (line.startsWith('/') || line.startsWith('!')) {
      elements.push(
        <span
          key={idx}
          className="inline-block px-2 py-1 mt-1 text-xs bg-blue-600/20 text-blue-400 rounded-lg cursor-pointer hover:bg-blue-600/30 transition-colors"
          onClick={() => onCommandClick(line)}
        >
          {line}
        </span>
      );
      return;
    }

    // Bold text
    let processedLine: React.ReactNode = line;
    if (line.includes('**')) {
      const parts = line.split('**');
      processedLine = (
        <span key={idx}>
          {parts.map((part, i) =>
            i % 2 === 1 ? <strong key={i} className="font-semibold text-zinc-100">{part}</strong> : part
          )}
        </span>
      );
    } else {
      processedLine = <span key={idx}>{line}</span>;
    }

    elements.push(processedLine);
    if (idx < lines.length - 1) {
      elements.push(<br key={`br-${idx}`} />);
    }
  });

  return <>{elements}</>;
}