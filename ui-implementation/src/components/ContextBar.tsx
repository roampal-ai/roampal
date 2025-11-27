import React, { useState, useEffect } from 'react';
import { ArrowDownIcon, ClockIcon, SparklesIcon, CubeIcon, LinkIcon, MagnifyingGlassIcon, FunnelIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface ContextBarProps {
  memories: MemoryFragment[];
  knowledgeGraph: KnowledgeGraphData;
  references: Reference[];
  onMemoryClick: (memoryId: string) => void;
  onRefresh?: () => void;
}

interface MemoryFragment {
  id: string;
  type: 'memory' | 'concept' | 'relation';
  content: string;
  score: number;
  timestamp: Date;
  session_id?: string;
  usefulness_score?: number;
  sentiment_score?: number;
  tags?: string[];
}

interface KnowledgeGraphData {
  concepts: number;
  relationships: number;
  activeTopics: string[];
  nodes?: KnowledgeNode[];
}

interface KnowledgeNode {
  id: string;
  name: string;
  meaning: string;
  connections: number;
  strength: number;
}

interface Reference {
  id: string;
  title: string;
  url?: string;
  snippet: string;
  timestamp: Date;
}

type SortType = 'recent' | 'score';

export const ContextBar: React.FC<ContextBarProps> = ({
  memories,
  knowledgeGraph,
  references,
  onMemoryClick,
  onRefresh,
}) => {
  const [activeTab, setActiveTab] = useState<'fragments' | 'graph' | 'references'>('fragments');
  const [sortBy, setSortBy] = useState<SortType>('recent');
  const [fragments, setFragments] = useState<MemoryFragment[]>(memories);
  const [graphNodes, setGraphNodes] = useState<KnowledgeNode[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'memory' | 'concept' | 'relation'>('all');
  const [showSearch, setShowSearch] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [offset, setOffset] = useState(0);
  const batchSize = 20;

  useEffect(() => {
    // Don't fetch on mount if we're receiving data from parent
    // Only fetch if onRefresh is not provided (standalone mode)
    if (!onRefresh) {
      fetchFragments();
      fetchKnowledgeGraph();
    }
  }, [onRefresh]);

  useEffect(() => {
    // Update fragments when memories prop changes
    if (memories && memories.length > 0) {
      console.log('[ContextBar] Received memories from parent:', memories.length);
      setFragments(memories);
    }
  }, [memories]);

  const fetchFragments = async (reset = false) => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory/fragments?limit=200&offset=0`);
      if (response.ok) {
        const data = await response.json();
        console.log('[ContextBar] Fetched fragments:', data);
        // The API returns { fragments: [...] }
        if (data && data.fragments) {
          setFragments(data.fragments);
        }
      }
    } catch (error) {
      console.error('Failed to fetch fragments:', error);
    }
  };

  const loadMoreFragments = async () => {
    // Disabled for now since we're loading all at once
    return;
  };

  const fetchKnowledgeGraph = async () => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory/knowledge-graph`);
      if (response.ok) {
        const data = await response.json();
        if (data.nodes) {
          setGraphNodes(data.nodes);
        }
      }
    } catch (error) {
      console.error('Failed to fetch knowledge graph:', error);
    }
  };

  // Filter and search fragments
  const filteredFragments = fragments.filter(fragment => {
    // Type filter
    if (filterType !== 'all' && fragment.type !== filterType) return false;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        fragment.content.toLowerCase().includes(query) ||
        fragment.tags?.some(tag => tag.toLowerCase().includes(query)) ||
        fragment.session_id?.toLowerCase().includes(query)
      );
    }

    return true;
  });

  const sortedFragments = [...filteredFragments].sort((a, b) => {
    if (sortBy === 'score') {
      return b.score - a.score;
    } else {
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    }
  });

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - new Date(date).getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'now';
    if (minutes < 60) return `${minutes}m`;
    if (hours < 24) return `${hours}h`;
    return `${days}d`;
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'text-emerald-400';
    if (score >= 0.7) return 'text-blue-400';
    if (score >= 0.5) return 'text-amber-400';
    return 'text-zinc-500';
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'memory':
        return 'text-blue-400';
      case 'concept':
        return 'text-purple-400';
      case 'relation':
        return 'text-emerald-400';
      default:
        return 'text-zinc-500';
    }
  };

  return (
    <aside className="w-80 flex flex-col bg-zinc-950 border-l border-zinc-800">
      {/* Header with Tabs */}
      <div className="h-14 px-4 flex items-center justify-between border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveTab('fragments')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'fragments'
                ? 'bg-zinc-800 text-zinc-100'
                : 'text-zinc-400 hover:text-zinc-100'
            }`}
          >
            <SparklesIcon className="w-3.5 h-3.5 inline mr-1" />
            Fragments
            {sortedFragments.length > 0 && (
              <span className="ml-2 px-1.5 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded-full">
                {sortedFragments.length}
              </span>
            )}
          </button>

          <button
            onClick={() => setActiveTab('graph')}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
              activeTab === 'graph'
                ? 'bg-zinc-800 text-zinc-100'
                : 'text-zinc-400 hover:text-zinc-100'
            }`}
          >
            <CubeIcon className="w-3.5 h-3.5 inline mr-1" />
            Graph
          </button>
        </div>

        {/* Controls for Fragments */}
        {activeTab === 'fragments' && (
          <div className="flex items-center gap-1">
            <button
              onClick={() => {
                console.log('[ContextBar] Refresh button clicked');
                setOffset(0);
                setHasMore(true);
                fetchFragments(true);
                fetchKnowledgeGraph();
              }}
              className="p-1 rounded text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800"
              title="Refresh"
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
            <button
              onClick={() => setShowSearch(!showSearch)}
              className={`p-1 rounded ${
                showSearch ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
              }`}
              title="Search"
            >
              <MagnifyingGlassIcon className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setSortBy('recent')}
              className={`p-1 rounded ${
                sortBy === 'recent' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
              }`}
              title="Sort by recent"
            >
              <ClockIcon className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setSortBy('score')}
              className={`p-1 rounded ${
                sortBy === 'score' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-500 hover:text-zinc-300'
              }`}
              title="Sort by score"
            >
              <ArrowDownIcon className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-hidden">
        {/* Fragments Tab */}
        {activeTab === 'fragments' && (
          <div className="flex flex-col h-full">
            {/* Search Bar */}
            {showSearch && (
              <div className="px-3 pt-3 pb-2">
                <div className="relative">
                  <MagnifyingGlassIcon className="absolute left-2.5 top-2.5 w-4 h-4 text-zinc-500" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search fragments..."
                    className="w-full pl-8 pr-3 py-2 bg-zinc-900 border border-zinc-800 rounded-lg text-xs text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-zinc-700"
                  />
                </div>
              </div>
            )}

            {/* Search Results Count */}
            {searchQuery && (
              <div className="px-3 pb-2">
                <span className="text-xs text-zinc-500">
                  {sortedFragments.length} results
                </span>
              </div>
            )}

            {/* Fragments List */}
            <div className="flex-1 overflow-y-auto p-3 space-y-2" style={{ maxHeight: 'calc(100vh - 250px)' }}>
              {sortedFragments.length === 0 ? (
              <div className="text-center py-12">
                <div className="relative">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-20 h-20 bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-full blur-xl" />
                  </div>
                  <SparklesIcon className="w-10 h-10 mx-auto text-zinc-600 mb-3 relative" />
                </div>
                <p className="text-sm text-zinc-500">No fragments yet</p>
                <p className="text-xs text-zinc-600 mt-1">Send messages to build memory</p>
              </div>
            ) : (
              sortedFragments.map((fragment) => (
                <div
                  key={fragment.id}
                  onClick={() => onMemoryClick(fragment.id)}
                  className="group relative p-3 rounded-lg cursor-pointer transition-all bg-zinc-900/50 hover:bg-zinc-800/50 border border-zinc-800 hover:border-zinc-700"
                >
                  {/* Type and Score Badges */}
                  <div className="flex items-start justify-between mb-2">
                    <span className={`text-[10px] uppercase tracking-wider font-medium ${getTypeColor(fragment.type)}`}>
                      {fragment.type}
                    </span>
                    <span className={`text-xs font-mono ${getScoreColor(fragment.score)}`}>
                      {Math.round(fragment.score * 100)}%
                    </span>
                  </div>

                  {/* Content */}
                  <p className="text-xs text-zinc-300 line-clamp-3 mb-2.5 leading-relaxed">
                    {fragment.content}
                  </p>

                  {/* Tags */}
                  {fragment.tags && fragment.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-2">
                      {fragment.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="text-[10px] px-1.5 py-0.5 bg-zinc-800/50 text-zinc-500 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Footer */}
                  <div className="flex items-center justify-between">
                    {fragment.session_id && (
                      <span className="text-[10px] text-zinc-600 font-mono">{fragment.session_id.slice(0, 6)}</span>
                    )}
                    <span className="text-[10px] text-zinc-600">
                      {formatTime(fragment.timestamp)}
                    </span>
                  </div>
                </div>
              ))
            )}

              {/* End of list indicator */}
              {fragments.length > 20 && (
                <div className="text-center py-4 text-xs text-zinc-600">
                  {fragments.length} total fragments
                </div>
              )}
            </div>
          </div>
        )}

        {/* Knowledge Graph Tab */}
        {activeTab === 'graph' && (
          <div className="p-4">
            {/* Graph Stats */}
            <div className="grid grid-cols-2 gap-3 mb-4">
              <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-800">
                <div className="flex items-center gap-2 mb-1">
                  <CubeIcon className="w-3.5 h-3.5 text-blue-400" />
                  <span className="text-xs text-zinc-500">Concepts</span>
                </div>
                <p className="text-xl font-semibold text-zinc-200">
                  {knowledgeGraph.concepts || graphNodes.length || 0}
                </p>
              </div>
              <div className="bg-zinc-900/50 rounded-lg p-3 border border-zinc-800">
                <div className="flex items-center gap-2 mb-1">
                  <LinkIcon className="w-3.5 h-3.5 text-purple-400" />
                  <span className="text-xs text-zinc-500">Relations</span>
                </div>
                <p className="text-xl font-semibold text-zinc-200">
                  {knowledgeGraph.relationships || 0}
                </p>
              </div>
            </div>

            {/* Concept List */}
            <div className="space-y-2">
              <h3 className="text-xs font-medium text-zinc-400 mb-2">Active Concepts</h3>
              {graphNodes.length > 0 ? (
                graphNodes.slice(0, 10).map((node) => (
                  <div
                    key={node.id}
                    className="p-2.5 bg-zinc-900/50 rounded-lg hover:bg-zinc-800/50 border border-zinc-800 hover:border-zinc-700 transition-all"
                  >
                    <div className="flex items-start justify-between mb-1">
                      <span className="text-sm font-medium text-zinc-200">{node.name}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-zinc-500">{node.connections} links</span>
                        <div className={`w-2 h-2 rounded-full ${
                          node.strength >= 0.8 ? 'bg-green-400' :
                          node.strength >= 0.5 ? 'bg-yellow-400' :
                          'bg-zinc-400'
                        }`} />
                      </div>
                    </div>
                    <p className="text-xs text-zinc-500 line-clamp-2">{node.meaning}</p>
                  </div>
                ))
              ) : knowledgeGraph.activeTopics && knowledgeGraph.activeTopics.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {knowledgeGraph.activeTopics.map((topic, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 bg-zinc-900 text-zinc-300 text-xs rounded-full"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-zinc-600">No active concepts yet</p>
              )}
            </div>
          </div>
        )}

        {/* References Tab */}
        {activeTab === 'references' && (
          <div className="p-3 space-y-2">
            {references.length === 0 ? (
              <div className="text-center py-8">
                <LinkIcon className="w-8 h-8 mx-auto text-zinc-700 mb-2" />
                <p className="text-sm text-zinc-500">No references yet</p>
              </div>
            ) : (
              references.map((ref) => (
                <a
                  key={ref.id}
                  href={ref.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block p-3 bg-zinc-900 hover:bg-zinc-800 rounded-lg transition-colors"
                >
                  <h4 className="text-sm font-medium text-blue-400 mb-1">{ref.title}</h4>
                  <p className="text-xs text-zinc-500 line-clamp-2 mb-2">{ref.snippet}</p>
                  <span className="text-xs text-zinc-600">{formatTime(ref.timestamp)}</span>
                </a>
              ))
            )}
          </div>
        )}
      </div>

      {/* Footer Stats */}
      <div className="h-10 px-4 flex items-center justify-center border-t border-zinc-800">
        <span className="text-xs text-zinc-500">
          {activeTab === 'fragments' && `${sortedFragments.length} fragment${sortedFragments.length !== 1 ? 's' : ''}`}
          {activeTab === 'graph' && `${knowledgeGraph.concepts || 0} concepts • ${knowledgeGraph.relationships || 0} relations`}
          {activeTab === 'references' && `${references.length} reference${references.length !== 1 ? 's' : ''}`}
        </span>
      </div>
    </aside>
  );
};