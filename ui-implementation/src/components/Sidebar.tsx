import React, { useState, useEffect } from 'react';
import { ChevronDownIcon, PlusIcon, ChevronLeftIcon, TrashIcon } from '@heroicons/react/24/outline';
import { useChatStore, ChatSession as StoreChatSession } from '../stores/useChatStore';
import { modelContextService } from '../services/modelContextService';
import { apiFetch } from '../utils/fetch';
import { DeleteSessionModal } from './DeleteSessionModal';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface SidebarProps {
  activeShard: string;
  activeSessionId?: string;
  availableShards?: string[];
  onShardChange: (shard: string) => void;
  onNewChat?: () => void;
  hasChatModel?: boolean;
  // chatHistory removed - Sidebar reads from store directly
  onSelectChat: (sessionId: string) => void;
  onDeleteChat?: (sessionId: string) => void;
  onMemoryPanel?: () => void;
  onManageShards?: () => void;
  onSettings?: () => void;
  onCollapse?: () => void;
  onPersonalityCustomizer?: () => void;
}

// Internal interface for Sidebar's transformed data
interface ChatSession {
  id: string;
  title: string;
  shard: string;
  timestamp: Date;
  messageCount: number;
}

const SHARD_COLORS: Record<string, string> = {
  roampal: 'blue',
  dev: 'green',
  creative: 'amber',
  analyst: 'purple',
  coach: 'pink',
};

export const Sidebar: React.FC<SidebarProps> = ({
  activeShard,
  activeSessionId,
  availableShards,
  onShardChange,
  onNewChat,
  hasChatModel = true,
  onSelectChat,
  onDeleteChat,
  onMemoryPanel,
  onManageShards,
  onSettings,
  onCollapse,
  onPersonalityCustomizer,
}) => {
  const [isShardMenuOpen, setIsShardMenuOpen] = useState(false);
  const [assistantName, setAssistantName] = useState('Roampal');
  const [modelLimits, setModelLimits] = useState<Record<string, number>>({});
  const [sessionToDelete, setSessionToDelete] = useState<{ id: string; title: string } | null>(null);

  // Subscribe to sessions from store - this automatically re-renders when sessions change
  const storeSessions = useChatStore(state => state.sessions);

  // Transform store sessions to Sidebar's format
  // Safety check: ensure storeSessions is an array
  const chatHistory: ChatSession[] = (Array.isArray(storeSessions) ? storeSessions : []).map(s => ({
    id: s.id,
    title: s.name || 'Untitled Session',
    shard: 'loopsmith',  // Single shard system
    timestamp: new Date(s.timestamp * 1000),  // Convert unix seconds to JS Date
    messageCount: s.messageCount
  }));

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

    fetchAssistantName();

    // Refresh name every 5 seconds to pick up changes
    const interval = setInterval(fetchAssistantName, 5000);
    return () => clearInterval(interval);
  }, []);

  // Fetch model context limits from API
  useEffect(() => {
    const fetchModelLimits = async () => {
      const contexts = await modelContextService.getAllContexts();
      const limits: Record<string, number> = {};

      // Convert contexts to simple model name -> current limit mapping
      for (const [modelPrefix, config] of Object.entries(contexts)) {
        limits[modelPrefix] = config.default;
      }

      setModelLimits(limits);
    };

    fetchModelLimits();
  }, []);

  // Mode system removed - RoamPal always uses memory

  const shards = [
    { id: 'roampal', name: 'Roampal', status: 'active' },
    { id: 'dev', name: 'Dev', status: 'idle' },
    { id: 'creative', name: 'Creative', status: 'idle' },
    { id: 'analyst', name: 'Analyst', status: 'idle' },
    { id: 'coach', name: 'Coach', status: 'idle' },
  ];
  
  const groupedHistory = groupChatsByDate(chatHistory);
  
  return (
    <aside className="w-full h-full flex flex-col bg-zinc-950 border-r border-zinc-800">
      {/* Header */}
      <div className="h-12 px-4 flex items-center justify-between border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green-500" />
          <span className="text-sm font-medium text-zinc-100">{assistantName}</span>
        </div>

        {/* Collapse button */}
        {onCollapse && (
          <button
            onClick={onCollapse}
            className="p-1.5 hover:bg-zinc-900 rounded-md transition-colors"
            title="Collapse sidebar (or double-click the drag handle)"
          >
            <ChevronLeftIcon className="w-4 h-4 text-zinc-400" />
          </button>
        )}
      </div>
      
      {/* New Chat Button */}
      <div className="px-4 mt-4">
        <button
          onClick={onNewChat}
          disabled={!hasChatModel || !onNewChat}
          className={`w-full h-10 px-3 py-2 flex items-center justify-center gap-2 rounded-lg transition-colors ${
            hasChatModel && onNewChat
              ? 'bg-blue-600/10 hover:bg-blue-600/20 border border-blue-600/30 cursor-pointer'
              : 'bg-zinc-800/50 border border-zinc-700/50 cursor-not-allowed opacity-50'
          }`}
          title={!hasChatModel ? 'Please install a chat model first' : 'Start a new conversation'}
        >
          <PlusIcon className={`w-4 h-4 ${hasChatModel && onNewChat ? 'text-blue-500' : 'text-zinc-600'}`} />
          <span className={`text-sm font-medium ${hasChatModel && onNewChat ? 'text-blue-500' : 'text-zinc-600'}`}>
            {hasChatModel ? 'New Conversation' : 'No Model Available'}
          </span>
        </button>
      </div>
      
      {/* Chat History */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {Object.entries(groupedHistory).map(([date, sessions]) => (
          <div key={date}>
            <h3 className="text-xs font-medium text-zinc-500 mb-2">{date}</h3>
            <div className="space-y-1">
              {sessions.map(session => (
                <div
                  key={session.id}
                  className="relative w-full h-8 flex items-center rounded-md hover:bg-zinc-900 transition-colors group"
                >
                  <button
                    onClick={() => onSelectChat(session.id)}
                    className="flex-1 h-full px-3 flex items-center gap-2 min-w-0"
                  >
                    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 bg-${SHARD_COLORS[session.shard]}-500`} />
                    <span className="flex-1 text-sm text-zinc-300 text-left truncate min-w-0">
                      {session.title}
                    </span>
                  </button>
                  {/* Timestamp - hidden on hover */}
                  <span className="flex-shrink-0 mr-2 text-xs text-zinc-500 opacity-100 group-hover:opacity-0 transition-opacity pointer-events-none">
                    {formatTime(session.timestamp)}
                  </span>
                  {/* Delete button - shown on hover, positioned over timestamp */}
                  {onDeleteChat && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setSessionToDelete({ id: session.id, title: session.title });
                      }}
                      className="absolute right-2 p-1 opacity-0 group-hover:opacity-100 hover:bg-red-500/10 rounded transition-all"
                      title="Delete conversation"
                    >
                      <TrashIcon className="w-3.5 h-3.5 text-zinc-500 hover:text-red-400" />
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
      
      {/* Quick Actions */}
      <div className="h-auto p-4 border-t border-zinc-800 space-y-1">
        <button
          onClick={onPersonalityCustomizer}
          className="w-full h-8 px-3 text-left text-sm text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 rounded-md transition-colors">
          Personality & Identity
        </button>
        <button
          onClick={onManageShards}
          className="w-full h-8 px-3 text-left text-sm text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 rounded-md transition-colors">
          Document Processor
        </button>
        <button
          onClick={onSettings}
          className="w-full h-8 px-3 text-left text-sm text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 rounded-md transition-colors">
          Settings
        </button>
      </div>

      {/* Delete Session Confirmation Modal */}
      <DeleteSessionModal
        isOpen={sessionToDelete !== null}
        sessionTitle={sessionToDelete?.title || ''}
        onConfirm={() => {
          if (sessionToDelete && onDeleteChat) {
            onDeleteChat(sessionToDelete.id);
          }
        }}
        onCancel={() => setSessionToDelete(null)}
      />

    </aside>
  );
};

// Helper functions
function groupChatsByDate(sessions: ChatSession[]) {
  // Sort sessions by timestamp descending (newest first)
  const sortedSessions = [...sessions].sort((a, b) => {
    const timeA = a.timestamp.getTime();  // timestamp is already a Date object
    const timeB = b.timestamp.getTime();
    return timeB - timeA;
  });

  const groups: Record<string, ChatSession[]> = {};
  const groupOrder: string[] = [];

  sortedSessions.forEach(session => {
    const date = session.timestamp;  // Already a Date object
    let label: string;

    if (isToday(date)) label = 'Today';
    else if (isYesterday(date)) label = 'Yesterday';
    else if (isThisWeek(date)) label = 'This Week';
    else label = formatDate(date);

    if (!groups[label]) {
      groups[label] = [];
      groupOrder.push(label);
    }
    groups[label].push(session);
  });

  // Return ordered groups
  const orderedGroups: Record<string, ChatSession[]> = {};
  groupOrder.forEach(label => {
    orderedGroups[label] = groups[label];
  });

  return orderedGroups;
}

function isToday(date: Date) {
  const today = new Date();
  return date.toDateString() === today.toDateString();
}

function isYesterday(date: Date) {
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  return date.toDateString() === yesterday.toDateString();
}

function isThisWeek(date: Date) {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);
  const weekAgo = new Date(today);
  weekAgo.setDate(weekAgo.getDate() - 7);

  // This week = after week ago, but NOT today or yesterday
  return date > weekAgo && date < yesterday;
}

function formatDate(date: Date) {
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatTime(date: Date) {
  return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}