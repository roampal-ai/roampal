import React, { useEffect, useState } from 'react';
import { TrashIcon, ChatBubbleLeftIcon, ClockIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface Conversation {
  id: string;
  title: string;
  messageCount: number;
  lastMessage: Date;
  preview?: string;
}

interface ConversationBadgesProps {
  activeConversationId?: string;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onNewConversation: () => void;
}

export const ConversationBadges: React.FC<ConversationBadgesProps> = ({
  activeConversationId,
  onSelectConversation,
  onDeleteConversation,
  onNewConversation,
}) => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchConversations();
  }, []);

  const fetchConversations = async () => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/conversations`);
      if (response.ok) {
        const data = await response.json();
        setConversations(data);
      }
    } catch (error) {
      console.error('Failed to fetch conversations:', error);
      // Mock data for development
      setConversations([
        {
          id: 'c1',
          title: 'API Optimization',
          messageCount: 12,
          lastMessage: new Date(Date.now() - 10 * 60000),
          preview: 'Implementing caching strategy...'
        },
        {
          id: 'c2',
          title: 'Debug Session',
          messageCount: 8,
          lastMessage: new Date(Date.now() - 60 * 60000),
          preview: 'Fixed memory leak in...'
        },
        {
          id: 'c3',
          title: 'Code Review',
          messageCount: 15,
          lastMessage: new Date(Date.now() - 2 * 60 * 60000),
          preview: 'Reviewing auth implementation...'
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'now';
    if (minutes < 60) return `${minutes}m`;
    if (hours < 24) return `${hours}h`;
    return `${days}d`;
  };

  return (
    <div className="w-80 h-full flex flex-col bg-zinc-950 border-r border-zinc-800">
      {/* Header */}
      <div className="h-14 px-4 flex items-center justify-between border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
          <span className="text-sm font-medium text-zinc-100">Conversations</span>
        </div>
        <button
          onClick={onNewConversation}
          className="px-3 py-1.5 text-xs bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 rounded-lg border border-blue-600/30 transition-all"
        >
          + New
        </button>
      </div>

      {/* Conversation Badges */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : conversations.length === 0 ? (
          <div className="text-center py-8">
            <ChatBubbleLeftIcon className="w-8 h-8 mx-auto text-zinc-700 mb-2" />
            <p className="text-sm text-zinc-500">No conversations yet</p>
          </div>
        ) : (
          conversations.map((conv) => (
            <div
              key={conv.id}
              onClick={() => onSelectConversation(conv.id)}
              className={`group relative p-3 rounded-xl cursor-pointer transition-all transform hover:scale-[1.02] ${
                activeConversationId === conv.id
                  ? 'bg-blue-600/20 border border-blue-600/40 shadow-lg shadow-blue-600/10'
                  : 'bg-zinc-900 hover:bg-zinc-800 border border-zinc-800'
              }`}
            >
              {/* Badge Content */}
              <div className="flex items-start justify-between mb-2">
                <h3 className={`text-sm font-medium truncate pr-2 ${
                  activeConversationId === conv.id ? 'text-blue-400' : 'text-zinc-200'
                }`}>
                  {conv.title}
                </h3>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(conv.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-zinc-700 rounded transition-all"
                >
                  <TrashIcon className="w-3.5 h-3.5 text-zinc-400 hover:text-red-400" />
                </button>
              </div>

              {/* Preview */}
              {conv.preview && (
                <p className="text-xs text-zinc-500 truncate mb-2">
                  {conv.preview}
                </p>
              )}

              {/* Stats */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {/* Message Count */}
                  <div className="flex items-center gap-1">
                    <ChatBubbleLeftIcon className="w-3 h-3 text-zinc-500" />
                    <span className="text-xs text-zinc-400">{conv.messageCount}</span>
                  </div>

                  {/* Time */}
                  <div className="flex items-center gap-1">
                    <ClockIcon className="w-3 h-3 text-zinc-500" />
                    <span className="text-xs text-zinc-400">{formatTime(conv.lastMessage)}</span>
                  </div>
                </div>

                {/* Active Indicator */}
                {activeConversationId === conv.id && (
                  <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer Stats */}
      <div className="h-12 px-4 flex items-center justify-center border-t border-zinc-800">
        <span className="text-xs text-zinc-500">
          {conversations.length} conversation{conversations.length !== 1 ? 's' : ''} • {
            conversations.reduce((sum, c) => sum + c.messageCount, 0)
          } total messages
        </span>
      </div>
    </div>
  );
};