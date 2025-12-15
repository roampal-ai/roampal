import React, { useEffect, useState, useRef, useCallback } from 'react';
import { VariableSizeList as List } from 'react-window';
import {
  TrashIcon,
  ArchiveBoxIcon,
  ArrowPathIcon,
  MagnifyingGlassIcon,
  TagIcon,
  SparklesIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface Memory {
  id: string;
  text: string;
  tags: string[];
  status: 'active' | 'archived';
  created_at: string;
  archived_at?: string;
  archived_reason?: string;
}

interface MemoryStats {
  total_memories: number;
  active: number;
  archived: number;
  unique_tags: number;
  tags: string[];
}

interface MemoryBankModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// Fixed height for virtualized list items (approximate)
const ITEM_HEIGHT = 90;
const ITEM_WITH_TAGS_HEIGHT = 110;

export const MemoryBankModal: React.FC<MemoryBankModalProps> = ({ isOpen, onClose }) => {
  const [memories, setMemories] = useState<Memory[]>([]);
  const [archivedMemories, setArchivedMemories] = useState<Memory[]>([]);
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<'active' | 'archived' | 'stats'>('active');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [showFilters, setShowFilters] = useState(false);

  // Virtualization refs
  const listRef = useRef<List>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerHeight, setContainerHeight] = useState(400);

  // Measure container height on mount and resize
  useEffect(() => {
    const updateHeight = () => {
      if (containerRef.current) {
        setContainerHeight(containerRef.current.clientHeight);
      }
    };
    updateHeight();
    window.addEventListener('resize', updateHeight);
    return () => window.removeEventListener('resize', updateHeight);
  }, [isOpen]);

  // Reset scroll position when filters change
  useEffect(() => {
    listRef.current?.scrollTo(0);
  }, [searchQuery, selectedTags, view]);

  useEffect(() => {
    if (isOpen) {
      fetchData();
    }
  }, [isOpen]);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [memoriesRes, archivedRes, statsRes] = await Promise.all([
        apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory-bank/list`),
        apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory-bank/archived`),
        apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory-bank/stats`)
      ]);

      if (memoriesRes.ok) {
        const data = await memoriesRes.json();
        setMemories(data.memories || []);
      }
      if (archivedRes.ok) {
        const data = await archivedRes.json();
        setArchivedMemories(data.memories || []);
      }
      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Failed to fetch memory bank data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleArchive = async (id: string) => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory-bank/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          status: 'archived',
          archived_reason: 'user_action'
        })
      });
      if (response.ok) {
        fetchData();
      }
    } catch (error) {
      console.error('Failed to archive memory:', error);
    }
  };

  const handleRestore = async (id: string) => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory-bank/restore/${id}`, {
        method: 'POST'
      });
      if (response.ok) {
        fetchData();
      }
    } catch (error) {
      console.error('Failed to restore memory:', error);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Permanently delete this memory? This cannot be undone.')) return;

    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory-bank/delete/${id}`, {
        method: 'DELETE'
      });
      if (response.ok) {
        fetchData();
      }
    } catch (error) {
      console.error('Failed to delete memory:', error);
    }
  };

  const toggleTag = (tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  };

  const filteredMemories = memories.filter(memory => {
    const matchesSearch = searchQuery === '' ||
      memory.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
      memory.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));

    const matchesTags = selectedTags.length === 0 ||
      selectedTags.every(tag => memory.tags.includes(tag));

    return matchesSearch && matchesTags;
  });

  const allTags = Array.from(new Set(memories.flatMap(m => m.tags)));

  // Calculate item size based on whether it has tags
  const getItemSize = useCallback((index: number) => {
    const memory = filteredMemories[index];
    return memory?.tags?.length > 0 ? ITEM_WITH_TAGS_HEIGHT : ITEM_HEIGHT;
  }, [filteredMemories]);

  const getArchivedItemSize = useCallback((index: number) => {
    const memory = archivedMemories[index];
    return memory?.tags?.length > 0 ? ITEM_WITH_TAGS_HEIGHT : ITEM_HEIGHT;
  }, [archivedMemories]);

  // Active memory row renderer
  const ActiveMemoryRow = useCallback(({ index, style }: { index: number; style: React.CSSProperties }) => {
    const memory = filteredMemories[index];
    if (!memory) return null;

    return (
      <div style={{ ...style, paddingBottom: 4 }}>
        <div className="p-4 bg-zinc-800 border border-zinc-700 rounded-lg hover:border-zinc-600 transition-colors mx-1">
          <div className="flex justify-between items-start mb-3">
            <div className="flex-1">
              <p className="text-sm text-zinc-200 mb-2 line-clamp-3">{memory.text}</p>
              {memory.tags.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {memory.tags.map(tag => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 text-xs bg-zinc-700 text-zinc-400 rounded"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div className="flex gap-1 ml-3 flex-shrink-0">
              <button
                onClick={() => handleArchive(memory.id)}
                className="p-1.5 text-zinc-400 hover:text-yellow-400 hover:bg-zinc-700 rounded transition-colors"
                title="Archive"
              >
                <ArchiveBoxIcon className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleDelete(memory.id)}
                className="p-1.5 text-zinc-400 hover:text-red-400 hover:bg-zinc-700 rounded transition-colors"
                title="Delete"
              >
                <TrashIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="flex gap-4 text-xs text-zinc-500">
            <span>{new Date(memory.created_at).toLocaleDateString()}</span>
          </div>
        </div>
      </div>
    );
  }, [filteredMemories, handleArchive, handleDelete]);

  // Archived memory row renderer
  const ArchivedMemoryRow = useCallback(({ index, style }: { index: number; style: React.CSSProperties }) => {
    const memory = archivedMemories[index];
    if (!memory) return null;

    return (
      <div style={{ ...style, paddingBottom: 4 }}>
        <div className="p-4 bg-zinc-800/50 border border-zinc-700/50 rounded-lg mx-1">
          <div className="flex justify-between items-start mb-3">
            <div className="flex-1">
              <p className="text-sm text-zinc-400 mb-2 line-clamp-3">{memory.text}</p>
              {memory.tags.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {memory.tags.map(tag => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 text-xs bg-zinc-700/50 text-zinc-500 rounded"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div className="flex gap-1 ml-3 flex-shrink-0">
              <button
                onClick={() => handleRestore(memory.id)}
                className="p-1.5 text-zinc-400 hover:text-green-400 hover:bg-zinc-700 rounded transition-colors"
                title="Restore"
              >
                <ArrowPathIcon className="w-4 h-4" />
              </button>
              <button
                onClick={() => handleDelete(memory.id)}
                className="p-1.5 text-zinc-400 hover:text-red-400 hover:bg-zinc-700 rounded transition-colors"
                title="Delete"
              >
                <TrashIcon className="w-4 h-4" />
              </button>
            </div>
          </div>
          <div className="flex gap-4 text-xs text-zinc-600">
            <span>Archived: {new Date(memory.archived_at!).toLocaleDateString()}</span>
            {memory.archived_reason && <span>Reason: {memory.archived_reason}</span>}
          </div>
        </div>
      </div>
    );
  }, [archivedMemories, handleRestore, handleDelete]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-zinc-900 rounded-xl shadow-2xl w-full max-w-4xl h-[80vh] border border-zinc-800 flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex justify-between items-center p-3 border-b border-zinc-800 flex-shrink-0">
          <h2 className="text-lg font-bold">Memory Bank</h2>
          <button
            onClick={onClose}
            className="p-2 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded-lg transition-colors"
            title="Close"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-3 py-1 border-b border-zinc-800 flex-shrink-0">
          <button
            onClick={() => setView('active')}
            className={`px-3 py-1.5 rounded-t-lg transition-colors ${
              view === 'active'
                ? 'bg-zinc-800 text-zinc-100 border-t border-x border-zinc-700'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Active ({memories.length})
          </button>
          <button
            onClick={() => setView('archived')}
            className={`px-3 py-1.5 rounded-t-lg transition-colors ${
              view === 'archived'
                ? 'bg-zinc-800 text-zinc-100 border-t border-x border-zinc-700'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            Archived ({archivedMemories.length})
          </button>
          <button
            onClick={() => setView('stats')}
            className={`px-3 py-1.5 rounded-t-lg transition-colors ${
              view === 'stats'
                ? 'bg-zinc-800 text-zinc-100 border-t border-x border-zinc-700'
                : 'text-zinc-400 hover:text-zinc-200'
            }`}
          >
            <ChartBarIcon className="w-4 h-4 inline mr-1" />
            Stats
          </button>
        </div>

        {/* Search & Filters (only for active view) */}
        {view === 'active' && (
          <div className="p-2 border-b border-zinc-800 space-y-2 flex-shrink-0">
            <div className="flex gap-2">
              <div className="relative flex-1">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
                <input
                  type="text"
                  placeholder="Search memories..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-cyan-600"
                />
              </div>
              {allTags.length > 0 && (
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-zinc-400 hover:text-zinc-200 hover:border-zinc-600 transition-colors flex items-center gap-1"
                >
                  <TagIcon className="w-4 h-4" />
                  {selectedTags.length > 0 && <span className="text-cyan-400">({selectedTags.length})</span>}
                </button>
              )}
            </div>

            {showFilters && allTags.length > 0 && (
              <div className="flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
                {allTags.map(tag => (
                  <button
                    key={tag}
                    onClick={() => toggleTag(tag)}
                    className={`px-2.5 py-1 text-xs rounded flex-shrink-0 transition-colors ${
                      selectedTags.includes(tag)
                        ? 'bg-cyan-600/20 text-cyan-400 border border-cyan-600/30'
                        : 'bg-zinc-800 text-zinc-400 border border-zinc-700 hover:border-zinc-600'
                    }`}
                  >
                    #{tag}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Content - Virtualized Lists */}
        <div ref={containerRef} className="flex-1 overflow-hidden p-4">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : view === 'active' ? (
            filteredMemories.length === 0 ? (
              <div className="text-center py-12">
                <SparklesIcon className="w-12 h-12 mx-auto text-zinc-700 mb-3" />
                <p className="text-zinc-500">
                  {searchQuery || selectedTags.length > 0
                    ? 'No memories match your filters'
                    : 'No memories stored yet'}
                </p>
              </div>
            ) : (
              <List
                ref={listRef}
                height={containerHeight - 32}
                itemCount={filteredMemories.length}
                itemSize={getItemSize}
                width="100%"
                className="scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent"
              >
                {ActiveMemoryRow}
              </List>
            )
          ) : view === 'archived' ? (
            archivedMemories.length === 0 ? (
              <div className="text-center py-12">
                <ArchiveBoxIcon className="w-12 h-12 mx-auto text-zinc-700 mb-3" />
                <p className="text-zinc-500">No archived memories</p>
              </div>
            ) : (
              <List
                height={containerHeight - 32}
                itemCount={archivedMemories.length}
                itemSize={getArchivedItemSize}
                width="100%"
                className="scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent"
              >
                {ArchivedMemoryRow}
              </List>
            )
          ) : (
            // Stats view
            stats && (
              <div className="space-y-6 overflow-y-auto h-full">
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-zinc-800 border border-zinc-700 rounded-lg">
                    <div className="text-2xl font-bold text-cyan-400">{stats.active}</div>
                    <div className="text-sm text-zinc-500">Active Memories</div>
                  </div>
                  <div className="p-4 bg-zinc-800 border border-zinc-700 rounded-lg">
                    <div className="text-2xl font-bold text-zinc-400">{stats.archived}</div>
                    <div className="text-sm text-zinc-500">Archived</div>
                  </div>
                  <div className="p-4 bg-zinc-800 border border-zinc-700 rounded-lg">
                    <div className="text-2xl font-bold text-green-400">{stats.total_memories}</div>
                    <div className="text-sm text-zinc-500">Total Memories</div>
                  </div>
                  <div className="p-4 bg-zinc-800 border border-zinc-700 rounded-lg">
                    <div className="text-2xl font-bold text-blue-400">{stats.unique_tags}</div>
                    <div className="text-sm text-zinc-500">Unique Tags</div>
                  </div>
                </div>

                {stats.tags.length > 0 && (
                  <div className="p-4 bg-zinc-800 border border-zinc-700 rounded-lg">
                    <div className="flex items-center gap-2 mb-3">
                      <TagIcon className="w-4 h-4 text-cyan-400" />
                      <h3 className="text-sm font-medium">All Tags</h3>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {stats.tags.map(tag => (
                        <span
                          key={tag}
                          className="px-2 py-1 text-xs bg-zinc-700 text-zinc-300 rounded"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
};
