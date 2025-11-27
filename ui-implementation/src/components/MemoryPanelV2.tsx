import React, { useState } from 'react';
import KnowledgeGraph from './KnowledgeGraph';

interface MemoryPanelV2Props {
  memories: any[];
  knowledgeGraph: any;
  onMemoryClick?: (memoryId: string) => void;
  onClose?: () => void;
  onRefresh?: () => void;
  isRefreshing?: boolean;
  lastRefresh?: Date | null;
  currentUserId?: string;
  activeShard?: string;
}

const MemoryPanelV2: React.FC<MemoryPanelV2Props> = ({
  memories,
  knowledgeGraph,
  onClose,
  onRefresh,
  isRefreshing,
  lastRefresh,
  activeShard,
  onMemoryClick,
}) => {
  const [activeTab, setActiveTab] = useState<'all' | 'books' | 'working' | 'conversations' | 'patterns' | 'graph'>('all');
  const [selectedMemory, setSelectedMemory] = useState<any>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'recent' | 'score'>('recent');
  const [filterType, setFilterType] = useState<'all' | 'working' | 'history' | 'patterns'>('all');
  const [showTypeInfo, setShowTypeInfo] = useState(false);
  const [showGraphInfo, setShowGraphInfo] = useState(false);

  // Heroicon components - muted colors
  const BrainIcon = () => (
    <svg className="w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
  );

  const UserIcon = () => (
    <svg className="w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
    </svg>
  );

  const ChipIcon = () => (
    <svg className="w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
    </svg>
  );

  const BeakerIcon = () => (
    <svg className="w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
    </svg>
  );

  const getMemoryIcon = (type: string) => {
    switch(type) {
      case 'book': return <BeakerIcon />;  // Reference from books
      case 'pattern': return <ChipIcon />;  // Proven patterns
      case 'working': return <BrainIcon />; // Working memory
      case 'conversation': return <UserIcon />; // Conversation history
      default: return <BrainIcon />;
    }
  };

  return (
    <div className="h-full max-h-screen flex flex-col bg-zinc-950 overflow-hidden">
      {/* TAB BAR - Per wireframe spec */}
      <div className="h-12 px-2 border-b border-zinc-800 flex items-center gap-2 flex-shrink-0">
        <button
          onClick={() => setActiveTab('all')}
          className={`px-4 py-1.5 text-sm font-medium rounded-lg transition-all flex-1 border ${
            activeTab === 'all' || activeTab === 'working' || activeTab === 'conversations' || activeTab === 'patterns'
              ? 'bg-blue-600/10 border-blue-600/30 text-blue-500'
              : 'bg-transparent border-zinc-700/50 text-zinc-400 hover:bg-blue-600/5 hover:border-blue-600/20 hover:text-zinc-300'
          }`}
        >
          Memory
        </button>
        <button
          onClick={() => setActiveTab('graph')}
          className={`px-4 py-1.5 text-sm font-medium rounded-lg transition-all flex-1 border ${
            activeTab === 'graph'
              ? 'bg-blue-600/10 border-blue-600/30 text-blue-500'
              : 'bg-transparent border-zinc-700/50 text-zinc-400 hover:bg-blue-600/5 hover:border-blue-600/20 hover:text-zinc-300'
          }`}
        >
          Knowledge
        </button>
        <div className="ml-auto flex items-center gap-1">
          <button
            onClick={onRefresh}
            disabled={isRefreshing}
            className="p-1.5 hover:bg-zinc-800 rounded transition-colors"
            title="Refresh"
          >
            <svg className={`w-4 h-4 text-zinc-400 ${isRefreshing ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-red-900/50 rounded transition-colors group"
            title="Close Memory Panel"
          >
            <svg className="w-4 h-4 text-zinc-400 group-hover:text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>


      {/* SEARCH AND FILTERS BAR */}
      <div className="px-4 py-2 border-b border-zinc-800 space-y-2">
        {/* Search Bar - Always visible */}
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={
                activeTab === 'graph' ? 'Search concepts...' :
                activeTab === 'books' ? 'Search references...' :
                'Search memories...'
              }
              className="w-full px-3 py-1.5 pl-8 text-xs bg-zinc-900 border border-zinc-800 rounded-lg text-zinc-300 placeholder-zinc-600 focus:outline-none focus:border-zinc-700"
            />
            <svg className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>

          {/* Graph info button - shows inline with search when Graph tab is active */}
          {activeTab === 'graph' && (
            <button
              onClick={() => setShowGraphInfo(true)}
              className="p-1 hover:bg-zinc-800 rounded transition-colors flex-shrink-0"
              title="Learn about the concept routing map"
            >
              <svg className="w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </button>
          )}
        </div>

        {/* Filter Options - Only for memory tabs */}
        {activeTab !== 'graph' && activeTab !== 'books' && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {/* Memory Type Filter */}
              <div className="relative">
                <select
                  value={filterType}
                  onChange={(e) => setFilterType(e.target.value as any)}
                  className={`px-2 py-1 text-xs font-medium rounded-lg border appearance-none pr-6 cursor-pointer focus:outline-none focus:ring-1 focus:ring-zinc-600 ${
                    filterType === 'all' ? 'bg-zinc-800 text-zinc-400 border-zinc-700' :
                    filterType === 'working' ? 'bg-blue-500/10 text-blue-400 border-blue-500/20' :
                    filterType === 'history' ? 'bg-green-500/10 text-green-400 border-green-500/20' :
                    filterType === 'patterns' ? 'bg-purple-500/10 text-purple-400 border-purple-500/20' :
                    'bg-zinc-800 text-zinc-300 border-zinc-700'
                  }`}
                  style={{ borderRadius: '0.5rem' }}
                >
                  <option value="all" className="bg-zinc-900">All Types</option>
                  <option value="working" className="bg-zinc-900">Working</option>
                  <option value="history" className="bg-zinc-900">History</option>
                  <option value="patterns" className="bg-zinc-900">Patterns</option>
                </select>
                {/* Dropdown arrow */}
                <svg className="absolute right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 text-current pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </div>

              {/* Sort Options */}
              <div className="flex items-center gap-1">
                <button
                  onClick={() => setSortBy('recent')}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    sortBy === 'recent'
                      ? 'bg-zinc-800 text-zinc-300'
                      : 'text-zinc-500 hover:text-zinc-400'
                  }`}
                >
                  Recent
                </button>
                <button
                  onClick={() => setSortBy('score')}
                  className={`px-2 py-1 text-xs rounded transition-colors ${
                    sortBy === 'score'
                      ? 'bg-zinc-800 text-zinc-300'
                      : 'text-zinc-500 hover:text-zinc-400'
                  }`}
                >
                  Score
                </button>
              </div>

              {/* Info Button */}
              <button
                onClick={() => setShowTypeInfo(true)}
                className="p-1 hover:bg-zinc-800 rounded transition-colors"
                title="Learn about memory types"
              >
                <svg className="w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </button>
            </div>

            {/* Refresh Button */}
            <button
              onClick={onRefresh}
              disabled={isRefreshing}
              className="p-1 hover:bg-zinc-800 rounded transition-colors"
              title="Refresh memories"
            >
              <svg className={`w-3.5 h-3.5 text-zinc-500 ${isRefreshing ? 'animate-spin' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        )}
      </div>

      {/* CONTENT AREA - Show memories by collection or graph */}
      <div className="flex-1 min-h-0 overflow-y-auto scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent">
        {activeTab === 'graph' ? (
          <div className="h-full p-4">
            <KnowledgeGraph searchQuery={searchQuery} />
          </div>
        ) : (
          <div className="p-4 space-y-1 pb-4">
            {(() => {
              // Filter by collection type
              let filteredMemories = memories.filter(memory => {
                // Filter by active tab
                if (activeTab !== 'all') {
                  const memType = memory.type || memory.collection_type || memory.collection || '';
                  if (activeTab === 'working' && memType !== 'working') return false;
                  if (activeTab === 'conversations' && memType !== 'conversation' && memType !== 'conversations') return false;
                  if (activeTab === 'patterns' && memType !== 'pattern' && memType !== 'patterns') return false;
                  if (activeTab === 'books' && memType !== 'book' && memType !== 'books') return false;
                }

                // Apply type filter dropdown
                if (filterType !== 'all') {
                  const memType = memory.type || memory.collection_type || memory.collection || '';
                  if (filterType === 'working' && memType !== 'working') return false;
                  if (filterType === 'history' && memType !== 'history' && memType !== 'conversation' && memType !== 'conversations') return false;
                  if (filterType === 'patterns' && memType !== 'pattern' && memType !== 'patterns') return false;
                }

                // Search filter
                if (searchQuery) {
                  const content = (memory.text || memory.content || '').toLowerCase();
                  return content.includes(searchQuery.toLowerCase());
                }
                return true;
              });

              // Sort memories
              filteredMemories = [...filteredMemories].sort((a, b) => {
                switch(sortBy) {
                  case 'recent':
                    // Sort by timestamp (newest first)
                    const timeA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
                    const timeB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
                    return timeB - timeA;
                  case 'score':
                    // Sort by score, but only for collections that have meaningful scores
                    const aType = a.type || a.collection_type || a.collection || '';
                    const bType = b.type || b.collection_type || b.collection || '';
                    const aHasScore = aType !== 'book' && aType !== 'books' && aType !== 'memory_bank';
                    const bHasScore = bType !== 'book' && bType !== 'books' && bType !== 'memory_bank';

                    // Items with scores sort before items without scores
                    if (aHasScore && !bHasScore) return -1;
                    if (!aHasScore && bHasScore) return 1;

                    // Both have scores or both don't - sort by score value (or timestamp)
                    if (aHasScore && bHasScore) {
                      return (b.score || 0) - (a.score || 0);
                    } else {
                      // Both are books/memory_bank - sort by timestamp
                      const timeA = a.timestamp ? new Date(a.timestamp).getTime() : 0;
                      const timeB = b.timestamp ? new Date(b.timestamp).getTime() : 0;
                      return timeB - timeA;
                    }
                  default:
                    return 0;
                }
              });

              return filteredMemories.length === 0 ? (
              <div className="text-center py-8">
                <svg className="w-12 h-12 mx-auto text-zinc-700 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="text-sm text-zinc-500">No memories yet</p>
                <p className="text-xs text-zinc-600 mt-1">Memories from your conversations will appear here</p>
              </div>
            ) : (
              <>
                {filteredMemories.slice(0, 100).map((memory, idx) => (
                  <div
                    key={memory.id || idx}
                    className="p-3 rounded-lg bg-zinc-900 border border-zinc-800 hover:border-zinc-700 cursor-pointer transition-all"
                    onClick={() => setSelectedMemory(memory)}
                  >
                    <div className="flex items-start gap-1">
                      <span className="flex-shrink-0 mt-0.5">
                        {getMemoryIcon(memory.type)}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <div className="text-[11px] text-zinc-600">
                            {memory.timestamp ? new Date(memory.timestamp).toLocaleString() : 'Unknown'}
                          </div>
                          {/* Memory Type Badge */}
                          <span className={`inline-flex items-center px-1.5 py-0.5 text-[10px] font-medium rounded-md ${
                            (memory.type === 'pattern' || memory.type === 'patterns') ? 'bg-purple-500/10 text-purple-400 border border-purple-500/20' :
                            (memory.type === 'working') ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20' :
                            (memory.type === 'conversation' || memory.type === 'conversations') ? 'bg-green-500/10 text-green-400 border border-green-500/20' :
                            (memory.type === 'book' || memory.type === 'books') ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20' :
                            'bg-zinc-800 text-zinc-500 border border-zinc-700'
                          }`}>
                            {memory.type || memory.collection_type || memory.collection || 'memory'}
                          </span>
                        </div>
                        <div className="text-xs text-zinc-500 line-clamp-2">
                          {memory.text || memory.content || 'Empty memory'}
                        </div>
                        {memory.score !== undefined &&
                         memory.type !== 'book' &&
                         memory.type !== 'books' &&
                         memory.collection !== 'books' &&
                         memory.type !== 'memory_bank' &&
                         memory.collection !== 'memory_bank' && (
                          <div className="mt-0.5 flex items-center gap-1.5">
                            <div className="flex-1 bg-zinc-800/50 rounded-full h-1">
                              <div
                                className="h-1 rounded-full transition-all bg-zinc-700"
                                style={{ width: `${memory.score * 100}%` }}
                              />
                            </div>
                            <span className={`text-[10px] font-medium ${
                              memory.score >= 0.75 ? 'text-green-500' :
                              memory.score >= 0.5 ? 'text-orange-500' : 'text-red-500'
                            }`}>
                              {Math.round(memory.score * 100)}%
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </>
            )})()}
          </div>
        )}

      </div>

      {/* Memory Detail Modal */}
      {selectedMemory && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedMemory(null)}
        >
          <div
            className="bg-zinc-900 rounded-xl border border-zinc-800 w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col mx-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="flex-shrink-0">
                  {getMemoryIcon(selectedMemory.type)}
                </span>
                <h3 className="text-sm font-medium text-zinc-300">
                  {selectedMemory.type === 'book' || selectedMemory.type === 'books' ? 'Reference Document' :
                   selectedMemory.type === 'pattern' || selectedMemory.type === 'patterns' ? 'Validated Pattern' :
                   selectedMemory.type === 'working' ? 'Working Memory' :
                   selectedMemory.type === 'conversation' || selectedMemory.type === 'conversations' ? 'Conversation' :
                   'Memory Entry'}
                </h3>
              </div>
              <button
                onClick={() => setSelectedMemory(null)}
                className="p-1 hover:bg-zinc-800 rounded transition-colors"
              >
                <svg className="w-4 h-4 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-4 overflow-y-auto flex-1 min-h-0 max-h-[60vh]">
              <div className="space-y-3">
                {/* Metadata */}
                <div className="text-xs text-zinc-600">
                  <div>Timestamp: {selectedMemory.timestamp ? new Date(selectedMemory.timestamp).toLocaleString() : 'Unknown'}</div>
                  <div>Type: {selectedMemory.type || selectedMemory.collection || 'memory'}</div>
                  {selectedMemory.session_id && <div>Session: {selectedMemory.session_id}</div>}
                  {selectedMemory.uses !== undefined && <div>Uses: {selectedMemory.uses}</div>}
                  {selectedMemory.last_outcome && <div>Last Outcome: <span className={`font-medium ${
                    selectedMemory.last_outcome === 'worked' ? 'text-green-400' :
                    selectedMemory.last_outcome === 'failed' ? 'text-red-400' :
                    selectedMemory.last_outcome === 'partial' ? 'text-yellow-400' :
                    'text-zinc-400'
                  }`}>{selectedMemory.last_outcome}</span></div>}
                  {selectedMemory.persist_session && <div className="text-blue-400">📌 Persistent</div>}
                </div>

                {/* Full Content */}
                <div className="text-sm text-zinc-300 whitespace-pre-wrap">
                  {selectedMemory.text || selectedMemory.content || 'No content available'}
                </div>

                {/* Failure Reasons */}
                {selectedMemory.failure_reasons && (() => {
                  try {
                    const reasons = typeof selectedMemory.failure_reasons === 'string'
                      ? JSON.parse(selectedMemory.failure_reasons)
                      : selectedMemory.failure_reasons;
                    if (reasons && reasons.length > 0) {
                      return (
                        <div className="pt-3 border-t border-zinc-800">
                          <div className="text-xs font-medium text-red-400 mb-1">Failure History:</div>
                          <div className="space-y-1">
                            {reasons.slice(-3).map((r: any, i: number) => (
                              <div key={i} className="text-xs text-zinc-500">
                                • {r.reason} <span className="text-zinc-600">({new Date(r.timestamp).toLocaleDateString()})</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    }
                  } catch (e) {
                    console.error('[MemoryPanel] Failed to parse failure contexts:', e);
                  }
                  return null;
                })()}

                {/* Success Contexts */}
                {selectedMemory.success_contexts && (() => {
                  try {
                    const contexts = typeof selectedMemory.success_contexts === 'string'
                      ? JSON.parse(selectedMemory.success_contexts)
                      : selectedMemory.success_contexts;
                    if (contexts && contexts.length > 0) {
                      return (
                        <div className="pt-3 border-t border-zinc-800">
                          <div className="text-xs font-medium text-green-400 mb-1">Worked In:</div>
                          <div className="space-y-1">
                            {contexts.slice(-3).map((c: any, i: number) => (
                              <div key={i} className="text-xs text-zinc-500">
                                • {Object.entries(c).map(([k, v]) => `${k}: ${v}`).join(', ')}
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    }
                  } catch (e) {
                    console.error('[MemoryPanel] Failed to parse success contexts:', e);
                  }
                  return null;
                })()}

                {/* Scores */}
                {(selectedMemory.score !== undefined || selectedMemory.usefulness_score !== undefined || selectedMemory.sentiment_score !== undefined) && (
                  <div className="pt-3 border-t border-zinc-800 space-y-2">
                    {selectedMemory.score !== undefined && (
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-zinc-600">Learning Score</span>
                        <span className={`font-medium ${
                          selectedMemory.score >= 0.7 ? 'text-green-500' :
                          selectedMemory.score >= 0.5 ? 'text-orange-500' : 'text-red-500'
                        }`}>
                          {Math.round(selectedMemory.score * 100)}%
                        </span>
                      </div>
                    )}
                    {selectedMemory.usefulness_score !== undefined && (
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-zinc-600">Usefulness</span>
                        <span className="text-zinc-400">{selectedMemory.usefulness_score.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Modal Footer */}
            <div className="p-4 border-t border-zinc-800 flex justify-end gap-2">
              <button
                onClick={() => setSelectedMemory(null)}
                className="px-3 py-1.5 text-xs bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Memory Type Info Modal */}
      {showTypeInfo && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          onClick={() => setShowTypeInfo(false)}
        >
          <div
            className="bg-zinc-900 rounded-xl border border-zinc-800 w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col mx-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-zinc-100">Understanding Memory Types</h3>
              <button
                onClick={() => setShowTypeInfo(false)}
                className="p-1 hover:bg-zinc-800 rounded transition-colors"
              >
                <svg className="w-5 h-5 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto flex-1 min-h-0 max-h-[60vh] space-y-5">
              {/* Working Memory */}
              <div className="space-y-2 pb-4 border-b border-zinc-800">
                <div className="flex items-center gap-2 mb-3">
                  <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-blue-500/10 text-blue-400 border border-blue-500/20">
                    working
                  </span>
                  <h4 className="text-base font-semibold text-zinc-100">Working Memory</h4>
                </div>
                <div className="pl-1">
                  <p className="text-sm text-zinc-300 leading-relaxed mb-2">
                    What you're discussing right now
                  </p>
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1.5 text-sm">
                    <span className="text-zinc-500">Lifespan:</span>
                    <span className="text-zinc-400">24 hours</span>
                    <span className="text-zinc-500">What happens:</span>
                    <span className="text-zinc-400">Good stuff gets saved, old stuff gets cleaned up</span>
                    <span className="text-zinc-500">Example:</span>
                    <span className="text-zinc-400">"We're brainstorming presentation ideas"</span>
                  </div>
                </div>
              </div>

              {/* History */}
              <div className="space-y-2 pb-4 border-b border-zinc-800">
                <div className="flex items-center gap-2 mb-3">
                  <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-green-500/10 text-green-400 border border-green-500/20">
                    history
                  </span>
                  <h4 className="text-base font-semibold text-zinc-100">History</h4>
                </div>
                <div className="pl-1">
                  <p className="text-sm text-zinc-300 leading-relaxed mb-2">
                    Past conversations and things you tried
                  </p>
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1.5 text-sm">
                    <span className="text-zinc-500">Lifespan:</span>
                    <span className="text-zinc-400">Stays until proven useful or not</span>
                    <span className="text-zinc-500">What happens:</span>
                    <span className="text-zinc-400">If it works repeatedly, becomes a pattern. If not, gets removed</span>
                    <span className="text-zinc-500">Example:</span>
                    <span className="text-zinc-400">"Three different ways we explained that topic"</span>
                  </div>
                </div>
              </div>

              {/* Patterns */}
              <div className="space-y-2 pb-4 border-b border-zinc-800">
                <div className="flex items-center gap-2 mb-3">
                  <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20">
                    patterns
                  </span>
                  <h4 className="text-base font-semibold text-zinc-100">Patterns</h4>
                </div>
                <div className="pl-1">
                  <p className="text-sm text-zinc-300 leading-relaxed mb-2">
                    Solutions that worked really well
                  </p>
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1.5 text-sm">
                    <span className="text-zinc-500">How they get here:</span>
                    <span className="text-zinc-400">Automatically promoted when they prove valuable</span>
                    <span className="text-zinc-500">What happens:</span>
                    <span className="text-zinc-400">Used first when similar problems appear</span>
                    <span className="text-zinc-500">Example:</span>
                    <span className="text-zinc-400">"Using STAR format always helps your interview answers"</span>
                  </div>
                </div>
              </div>

              {/* Books */}
              <div className="space-y-2 pb-4 border-b border-zinc-800">
                <div className="flex items-center gap-2 mb-3">
                  <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20">
                    books
                  </span>
                  <h4 className="text-base font-semibold text-zinc-100">Books</h4>
                </div>
                <div className="pl-1">
                  <p className="text-sm text-zinc-300 leading-relaxed mb-2">
                    Documents you uploaded for reference
                  </p>
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1.5 text-sm">
                    <span className="text-zinc-500">Lifespan:</span>
                    <span className="text-zinc-400">Forever (never removed)</span>
                    <span className="text-zinc-500">What happens:</span>
                    <span className="text-zinc-400">Always available when relevant</span>
                    <span className="text-zinc-500">Example:</span>
                    <span className="text-zinc-400">Your favorite recipes, research papers, recipes, travel guides</span>
                  </div>
                </div>
              </div>

              {/* Memory Bank */}
              <div className="space-y-2 pb-4">
                <div className="flex items-center gap-2 mb-3">
                  <span className="px-2.5 py-1 text-xs font-medium rounded-md bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                    memory_bank
                  </span>
                  <h4 className="text-base font-semibold text-zinc-100">Memory Bank</h4>
                </div>
                <div className="pl-1">
                  <p className="text-sm text-zinc-300 leading-relaxed mb-2">
                    What the AI remembers about you
                  </p>
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1.5 text-sm">
                    <span className="text-zinc-500">Lifespan:</span>
                    <span className="text-zinc-400">Forever (manage in Settings)</span>
                    <span className="text-zinc-500">What happens:</span>
                    <span className="text-zinc-400">AI adds preferences and context as it learns about you</span>
                    <span className="text-zinc-500">Example:</span>
                    <span className="text-zinc-400">"You prefer simple explanations, with real examples"</span>
                  </div>
                </div>
              </div>

              {/* How It Works */}
              <div className="mt-4 p-4 bg-zinc-800/30 rounded-lg border border-zinc-700/50">
                <h4 className="text-base font-semibold text-zinc-100 mb-3">How It All Works</h4>
                <div className="grid gap-2.5 text-sm">
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 items-start">
                    <svg className="w-4 h-4 text-blue-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-zinc-400">The AI evaluates what helps you most</span>
                  </div>
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 items-start">
                    <svg className="w-4 h-4 text-green-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                    </svg>
                    <span className="text-zinc-400">Successful solutions get promoted automatically</span>
                  </div>
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 items-start">
                    <svg className="w-4 h-4 text-red-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    <span className="text-zinc-400">Things that don't work get removed</span>
                  </div>
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 items-start">
                    <svg className="w-4 h-4 text-purple-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <span className="text-zinc-400">Learns which memories answer which questions best</span>
                  </div>
                  </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Graph Info Modal */}
      {showGraphInfo && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          onClick={() => setShowGraphInfo(false)}
        >
          <div
            className="bg-zinc-900 rounded-xl border border-zinc-800 w-full max-w-2xl max-h-[80vh] overflow-hidden flex flex-col mx-auto"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-zinc-100">Understanding the Concept Routing Map</h3>
              <button
                onClick={() => setShowGraphInfo(false)}
                className="p-1 hover:bg-zinc-800 rounded transition-colors"
              >
                <svg className="w-5 h-5 text-zinc-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* Modal Content */}
            <div className="p-6 overflow-y-auto flex-1 min-h-0 max-h-[60vh] space-y-6">
              {/* What is the Routing Map */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-zinc-100">What is this?</h4>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  Shows your top 20 most valuable learned patterns. When you search memory or chat with Roampal,
                  the system tracks which concepts lead to successful outcomes (you say "thanks", "worked", etc.).
                </p>
                <p className="text-sm text-zinc-400 leading-relaxed">
                  Each node represents a concept the system has learned to route effectively — it knows which
                  memory sources (books, patterns, history, working, memory_bank) tend to have the best information
                  for that topic.
                </p>
              </div>

              {/* Triple Knowledge Graph System (v0.2.1) */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-zinc-100">Three Types of Learning</h4>
                <div className="space-y-2">
                  <div className="p-3 bg-zinc-800/50 rounded-lg">
                    <p className="text-xs font-medium text-blue-400 mb-1">Routing Graph (Query-based)</p>
                    <p className="text-xs text-zinc-400">
                      Learns which memory collections successfully answer queries containing specific concepts.
                      When you search for "Python debugging", it learns that "patterns" collection has the best answers.
                    </p>
                  </div>
                  <div className="p-3 bg-zinc-800/50 rounded-lg">
                    <p className="text-xs font-medium text-green-400 mb-1">Content Graph (Memory-based)</p>
                    <p className="text-xs text-zinc-400">
                      Learns relationships between concepts in your memory_bank content. Tracks which ideas
                      appear together in your saved knowledge.
                    </p>
                  </div>
                  <div className="p-3 bg-zinc-800/50 rounded-lg">
                    <p className="text-xs font-medium text-orange-400 mb-1">Action Effectiveness Graph</p>
                    <p className="text-xs text-zinc-400">
                      Tracks which actions (search, add, update) work best in different contexts (recall, learning, analysis).
                      Helps the system choose smarter actions based on what's worked before.
                    </p>
                  </div>
                </div>
              </div>

              {/* Visual Elements */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-zinc-100">Node Colors (by Source)</h4>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 rounded-full bg-blue-500 mt-1"></div>
                    <div>
                      <p className="text-sm text-zinc-300 font-medium">Blue (Query Routing)</p>
                      <p className="text-sm text-zinc-500">Concepts learned from your search queries — knows which collections answer best</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 rounded-full bg-green-500 mt-1"></div>
                    <div>
                      <p className="text-sm text-zinc-300 font-medium">Green (Memory Content)</p>
                      <p className="text-sm text-zinc-500">Entities extracted from your memory_bank — tracks which ideas appear together</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 rounded-full bg-purple-500 mt-1"></div>
                    <div>
                      <p className="text-sm text-zinc-300 font-medium">Purple (Both Graphs)</p>
                      <p className="text-sm text-zinc-500">Concepts found in both routing and content graphs — strong signals</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-3 h-3 rounded-full bg-orange-500 mt-1"></div>
                    <div>
                      <p className="text-sm text-zinc-300 font-medium">Orange (Action Patterns)</p>
                      <p className="text-sm text-zinc-500">Action effectiveness patterns — which operations work in which contexts</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-8 h-0.5 bg-zinc-600 mt-2"></div>
                    <div>
                      <p className="text-sm text-zinc-300 font-medium">Lines (Connections)</p>
                      <p className="text-sm text-zinc-500">Concepts that appear together in successful query-answer pairs</p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <div className="w-4 h-4 rounded-full bg-zinc-600 mt-1 flex items-center justify-center text-xs font-bold text-zinc-300">
                      S
                    </div>
                    <div>
                      <p className="text-sm text-zinc-300 font-medium">Node Size</p>
                      <p className="text-sm text-zinc-500">Larger = √usage × √quality (balances frequency and success rate)</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Node Information */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-zinc-100">When You Click a Concept</h4>
                <div className="p-4 bg-zinc-800/50 rounded-lg space-y-2">
                  <div className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-2 text-sm">
                    <span className="text-zinc-500">Quality Score:</span>
                    <span className="text-zinc-400">Percentage of queries where this concept found helpful answers</span>
                    <span className="text-zinc-500">Usage Count:</span>
                    <span className="text-zinc-400">How many memory searches included this concept</span>
                    <span className="text-zinc-500">Hybrid Score:</span>
                    <span className="text-zinc-400">√usage × √quality (determines node size)</span>
                    <span className="text-zinc-500">Best Memory Type:</span>
                    <span className="text-zinc-400">Which collection (books/patterns/history/working/memory_bank) performs best for this concept</span>
                    <span className="text-zinc-500">Related Concepts:</span>
                    <span className="text-zinc-400">Other concepts that appear together in successful searches</span>
                  </div>
                </div>
              </div>

              {/* How It Works Example */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-zinc-100">How It Works (Example)</h4>
                <div className="p-3 bg-zinc-800/30 rounded-lg">
                  <ol className="text-xs text-zinc-400 space-y-1.5 list-decimal list-inside">
                    <li>You search for "Python debugging"</li>
                    <li>System identifies key concepts: "Python", "debugging"</li>
                    <li>Checks which collections answered similar queries successfully</li>
                    <li>Searches those collections first (e.g., "patterns" if you've solved Python bugs before)</li>
                    <li>Based on your feedback ("that worked!"), updates routing strategy</li>
                  </ol>
                </div>
              </div>

              {/* How It Learns */}
              <div className="space-y-2">
                <h4 className="text-sm font-semibold text-zinc-100">How It Improves</h4>
                <div className="grid grid-cols-[auto,1fr] gap-x-3 gap-y-1.5 text-sm">
                  <span className="text-zinc-500">Learning:</span>
                  <span className="text-zinc-400">Tracks which memory collections successfully answer queries</span>
                  <span className="text-zinc-500">Connections:</span>
                  <span className="text-zinc-400">Links concepts that appear together in successful searches</span>
                  <span className="text-zinc-500">Cleanup:</span>
                  <span className="text-zinc-400">Removes outdated routing patterns automatically</span>
                </div>
              </div>

              {/* Important Note */}
              <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <p className="text-sm text-blue-400">
                  <strong>Note:</strong> This is a routing optimizer, not a content store. It helps the AI find information faster by learning which memory sources work best for different topics. The actual content lives in your memory collections.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MemoryPanelV2; 
