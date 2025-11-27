import React, { useEffect, useState } from 'react';
import { ArrowUpIcon, ArrowDownIcon, ClockIcon, SparklesIcon } from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface Fragment {
  id: string;
  content: string;
  score: number;
  timestamp: Date;
  type: 'global' | 'private';
  tags?: string[];
}

interface FragmentBadgesProps {
  onFragmentClick?: (id: string) => void;
}

type SortType = 'recent' | 'score';

export const FragmentBadges: React.FC<FragmentBadgesProps> = ({
  onFragmentClick,
}) => {
  const [fragments, setFragments] = useState<Fragment[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState<SortType>('recent');

  useEffect(() => {
    fetchFragments();
  }, []);

  const fetchFragments = async () => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/memory/fragments`);
      if (response.ok) {
        const data = await response.json();
        setFragments(data);
      }
    } catch (error) {
      console.error('Failed to fetch fragments:', error);
      // Mock data for development
      setFragments([
        {
          id: 'f1',
          content: 'Use DataLoader pattern for batch loading to reduce N+1 queries',
          score: 0.95,
          timestamp: new Date(Date.now() - 5 * 60000),
          type: 'global',
          tags: ['optimization', 'database'],
        },
        {
          id: 'f2',
          content: 'Redis caching with 5-minute TTL for user session data',
          score: 0.88,
          timestamp: new Date(Date.now() - 30 * 60000),
          type: 'private',
          tags: ['caching', 'redis'],
        },
        {
          id: 'f3',
          content: 'Always index foreign keys in PostgreSQL for better join performance',
          score: 0.92,
          timestamp: new Date(Date.now() - 2 * 60 * 60000),
          type: 'global',
          tags: ['database', 'postgresql'],
        },
        {
          id: 'f4',
          content: 'Component memoization reduces re-renders by 60% in React',
          score: 0.85,
          timestamp: new Date(Date.now() - 4 * 60 * 60000),
          type: 'private',
          tags: ['react', 'performance'],
        },
        {
          id: 'f5',
          content: 'Implement rate limiting: 100 requests per minute per API key',
          score: 0.78,
          timestamp: new Date(Date.now() - 24 * 60 * 60000),
          type: 'global',
          tags: ['api', 'security'],
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sortedFragments = [...fragments].sort((a, b) => {
    if (sortBy === 'score') {
      return b.score - a.score;
    } else {
      return b.timestamp.getTime() - a.timestamp.getTime();
    }
  });

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString();
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'text-green-400 bg-green-400/10 border-green-400/30';
    if (score >= 0.7) return 'text-blue-400 bg-blue-400/10 border-blue-400/30';
    if (score >= 0.5) return 'text-yellow-400 bg-yellow-400/10 border-yellow-400/30';
    return 'text-zinc-400 bg-zinc-400/10 border-zinc-400/30';
  };

  return (
    <div className="w-80 h-full flex flex-col bg-zinc-950 border-l border-zinc-800">
      {/* Header */}
      <div className="h-14 px-4 flex items-center justify-between border-b border-zinc-800">
        <div className="flex items-center gap-2">
          <SparklesIcon className="w-4 h-4 text-purple-400" />
          <span className="text-sm font-medium text-zinc-100">Memory Fragments</span>
        </div>

        {/* Sort Controls */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setSortBy('recent')}
            className={`px-2 py-1 text-xs rounded transition-all ${
              sortBy === 'recent'
                ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30'
                : 'text-zinc-400 hover:text-zinc-300'
            }`}
          >
            <ClockIcon className="w-3.5 h-3.5 inline mr-1" />
            Recent
          </button>
          <button
            onClick={() => setSortBy('score')}
            className={`px-2 py-1 text-xs rounded transition-all ${
              sortBy === 'score'
                ? 'bg-blue-600/20 text-blue-400 border border-blue-600/30'
                : 'text-zinc-400 hover:text-zinc-300'
            }`}
          >
            <ArrowDownIcon className="w-3.5 h-3.5 inline mr-1" />
            Score
          </button>
        </div>
      </div>

      {/* Fragment Badges */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="w-6 h-6 border-2 border-purple-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : sortedFragments.length === 0 ? (
          <div className="text-center py-8">
            <SparklesIcon className="w-8 h-8 mx-auto text-zinc-700 mb-2" />
            <p className="text-sm text-zinc-500">No fragments yet</p>
          </div>
        ) : (
          sortedFragments.map((fragment) => (
            <div
              key={fragment.id}
              onClick={() => onFragmentClick?.(fragment.id)}
              className="group relative p-3 rounded-xl cursor-pointer transition-all transform hover:scale-[1.02] bg-zinc-900 hover:bg-zinc-800 border border-zinc-800"
            >
              {/* Type Badge */}
              <div className="flex items-start justify-between mb-2">
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  fragment.type === 'global'
                    ? 'bg-purple-600/20 text-purple-400 border border-purple-600/30'
                    : 'bg-zinc-700 text-zinc-400 border border-zinc-600'
                }`}>
                  {fragment.type}
                </span>

                {/* Score Badge */}
                <span className={`text-xs px-2 py-0.5 rounded-full border ${getScoreColor(fragment.score)}`}>
                  {Math.round(fragment.score * 100)}%
                </span>
              </div>

              {/* Content */}
              <p className="text-xs text-zinc-300 line-clamp-2 mb-2">
                {fragment.content}
              </p>

              {/* Tags */}
              {fragment.tags && fragment.tags.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {fragment.tags.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="text-xs px-1.5 py-0.5 bg-zinc-800 text-zinc-500 rounded"
                    >
                      #{tag}
                    </span>
                  ))}
                  {fragment.tags.length > 3 && (
                    <span className="text-xs text-zinc-600">
                      +{fragment.tags.length - 3}
                    </span>
                  )}
                </div>
              )}

              {/* Timestamp */}
              <div className="flex items-center gap-1">
                <ClockIcon className="w-3 h-3 text-zinc-600" />
                <span className="text-xs text-zinc-500">
                  {formatTime(fragment.timestamp)}
                </span>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer Stats */}
      <div className="h-12 px-4 flex items-center justify-center border-t border-zinc-800">
        <div className="flex items-center gap-4 text-xs text-zinc-500">
          <span>{sortedFragments.filter(f => f.type === 'global').length} global</span>
          <span>•</span>
          <span>{sortedFragments.filter(f => f.type === 'private').length} private</span>
          <span>•</span>
          <span>{sortedFragments.length} total</span>
        </div>
      </div>
    </div>
  );
};