import React, { useState, useEffect } from 'react';
import { User, Bot } from 'lucide-react';
import { EnhancedMessageDisplay } from './EnhancedMessageDisplay';
import { ToolExecutionDisplay } from './ToolExecutionDisplay';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface EnhancedChatMessageProps {
  message: {
    id: string;
    sender: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    streaming?: boolean;
    thinking?: string;
    toolExecutions?: Array<{
      tool: string;
      status: 'running' | 'completed' | 'failed';
      description: string;
      detail?: string;
      metadata?: Record<string, any>;
    }>;
    codeBlocks?: Array<{ language: string; code: string }>;
    citations?: Array<{
      citation_id: number;
      source: string;
      confidence: number;
      collection: string;
      text?: string;
    }>;
    metadata?: {
      model_name?: string;
      [key: string]: any;
    };
  };
}

export const EnhancedChatMessage: React.FC<EnhancedChatMessageProps> = ({
  message
}) => {
  const isUser = message.sender === 'user';

  // Debug log for tool executions
  console.log('[EnhancedChatMessage] Rendering message:', {
    id: message.id,
    sender: message.sender,
    hasToolExecutions: !!message.toolExecutions,
    toolExecutionsCount: message.toolExecutions?.length || 0,
    toolExecutions: message.toolExecutions
  });

  if (message.toolExecutions) {
    console.log('[EnhancedChatMessage] Message has toolExecutions:', message.toolExecutions);
  }
  const [assistantName, setAssistantName] = useState('Roampal');

  // Fetch assistant name from personality settings
  useEffect(() => {
    const fetchAssistantName = async () => {
      try {
        const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/personality/current`);
        if (response.ok) {
          const data = await response.json();
          // Parse YAML properly - handle quoted and unquoted values
          const nameMatch = data.content.match(/name:\s*(?:"([^"]+)"|'([^']+)'|([^\n]+))/);
          if (nameMatch) {
            const name = (nameMatch[1] || nameMatch[2] || nameMatch[3] || '').trim();
            if (name) {
              setAssistantName(name);
            }
          }
        }
      } catch (error) {
        console.error('Failed to fetch assistant name:', error);
      }
    };

    if (!isUser) {
      fetchAssistantName();
    }
  }, [isUser]);

  return (
    <div
      className={`flex gap-3 px-4 py-3 ${
        isUser ? 'bg-zinc-900/30' : 'bg-transparent'
      }`}
    >
      {/* Avatar */}
      <div className="flex-shrink-0">
        <div
          className={`w-8 h-8 rounded-lg flex items-center justify-center ${
            isUser ? 'bg-blue-600' : 'bg-zinc-800'
          }`}
        >
          {isUser ? (
            <User className="w-5 h-5 text-white" />
          ) : (
            <Bot className="w-5 h-5 text-zinc-300" />
          )}
        </div>
      </div>

      {/* Message Content */}
      <div className="flex-1 min-w-0">
        {/* Sender name and timestamp */}
        <div className="flex items-center gap-2 mb-1">
          <span className="text-sm font-medium text-zinc-400">
            {isUser ? 'You' : assistantName}
          </span>
          <span className="text-xs text-zinc-600">
            {new Date(message.timestamp).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit'
            })}
          </span>
          {!isUser && message.metadata?.model_name && (
            <span className="text-xs px-1.5 py-0.5 bg-zinc-800/50 text-zinc-500 rounded border border-zinc-700/50">
              {message.metadata.model_name.split(':')[0]}
            </span>
          )}
          {message.streaming && (
            <span className="text-xs text-blue-500 animate-pulse">• Typing</span>
          )}
        </div>

        {/* Message body */}
        {isUser ? (
          // User messages are simple text
          <div className="text-zinc-300 whitespace-pre-wrap">{message.content}</div>
        ) : (
          // Assistant messages have enhanced display with citations only
          <div className="space-y-3">
            {/* Show tool executions if present */}
            {message.toolExecutions && message.toolExecutions.length > 0 && (
              <>
                {console.log('[EnhancedChatMessage] Rendering ToolExecutionDisplay with', message.toolExecutions.length, 'executions')}
                <ToolExecutionDisplay
                  executions={message.toolExecutions}
                />
              </>
            )}

            {/* Show the actual response with enhanced formatting */}
            {message.content && (
              <EnhancedMessageDisplay
                content={message.content}
                codeBlocks={message.codeBlocks}
                citations={message.citations}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};