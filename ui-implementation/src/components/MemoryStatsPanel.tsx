import React, { useState, useEffect } from 'react';
import {
  ChartBarIcon,
  CircleStackIcon,
  SparklesIcon,
  ClockIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface MemoryStats {
  conversation_id: string;
  collections: {
    books: number;
    working: number;
    history: number;
    patterns: number;
  };
  kg_patterns: number;
  knowledge_graph: {
    routing_patterns: number;
    failure_patterns: number;
    problem_categories: number;
    problem_solutions: number;
    solution_patterns: number;
  };
  relationships: {
    related: number;
    evolution: number;
    conflicts: number;
  };
  learning: {
    outcome_detection_enabled: boolean;
    knowledge_graph_active: boolean;
  };
}

interface MemoryStatsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

const MemoryStatsPanel: React.FC<MemoryStatsPanelProps> = ({ isOpen, onClose }) => {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Poll memory stats every 5 seconds
  useEffect(() => {
    if (!isOpen) return;

    const fetchStats = async () => {
      try {
        setLoading(true);
        const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/chat/stats`);
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        setStats(data);
        setError(null);
      } catch (err) {
        console.error('Failed to fetch memory stats:', err);
        setError(err instanceof Error ? err.message : 'Failed to load stats');
      } finally {
        setLoading(false);
      }
    };

    // Fetch immediately
    fetchStats();

    // Then poll every 5 seconds
    const interval = setInterval(fetchStats, 5000);

    return () => clearInterval(interval);
  }, [isOpen]);

  if (!isOpen) return null;

  const totalMemories = stats
    ? stats.collections.books + stats.collections.working + stats.collections.history + stats.collections.patterns
    : 0;

  const totalKG = stats
    ? stats.knowledge_graph.routing_patterns + stats.knowledge_graph.problem_solutions + stats.knowledge_graph.solution_patterns
    : 0;

  return (
    <div className="fixed inset-y-0 right-0 w-80 bg-zinc-900 border-l border-zinc-700 shadow-xl z-50 overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 bg-zinc-900 border-b border-zinc-700 p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ChartBarIcon className="w-5 h-5 text-blue-400" />
          <h2 className="font-semibold text-white">Memory System</h2>
        </div>
        <button
          onClick={onClose}
          className="p-1 hover:bg-zinc-800 rounded transition-colors"
        >
          <XMarkIcon className="w-5 h-5 text-zinc-400" />
        </button>
      </div>

      {/* Content */}
      <div className="p-4 space-y-6">
        {loading && !stats && (
          <div className="text-center text-zinc-400 py-8">
            Loading memory stats...
          </div>
        )}

        {error && (
          <div className="bg-red-900/20 border border-red-700 rounded p-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {stats && (
          <>
            {/* Overview Card */}
            <div className="bg-zinc-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <CircleStackIcon className="w-4 h-4 text-zinc-400" />
                <h3 className="text-sm font-medium text-zinc-300">Memory Collections</h3>
              </div>

              <div className="space-y-2">
                <StatRow
                  label="Books"
                  value={stats.collections.books}
                  color="purple"
                  description="Reference documents"
                />
                <StatRow
                  label="Working"
                  value={stats.collections.working}
                  color="blue"
                  description="Current session (24h)"
                />
                <StatRow
                  label="History"
                  value={stats.collections.history}
                  color="amber"
                  description="Past conversations (30d)"
                />
                <StatRow
                  label="Patterns"
                  value={stats.collections.patterns}
                  color="green"
                  description="Proven solutions"
                />
              </div>

              <div className="mt-3 pt-3 border-t border-zinc-700">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-zinc-400">Total Memories</span>
                  <span className="font-semibold text-white">{totalMemories}</span>
                </div>
              </div>
            </div>

            {/* Knowledge Graph Card */}
            <div className="bg-zinc-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <SparklesIcon className="w-4 h-4 text-zinc-400" />
                <h3 className="text-sm font-medium text-zinc-300">Knowledge Graph</h3>
              </div>

              <div className="space-y-2">
                <KGStatRow
                  label="Routing Patterns"
                  value={stats.knowledge_graph.routing_patterns}
                  description="Learned collection preferences"
                />
                <KGStatRow
                  label="Problem → Solutions"
                  value={stats.knowledge_graph.problem_solutions}
                  description="Mapped problem-solution pairs"
                />
                <KGStatRow
                  label="Solution Patterns"
                  value={stats.knowledge_graph.solution_patterns}
                  description="Reusable solution templates"
                />
              </div>

              <div className="mt-3 pt-3 border-t border-zinc-700">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-zinc-400">Total Patterns</span>
                  <span className="font-semibold text-white">{totalKG}</span>
                </div>
              </div>
            </div>

            {/* Learning Status */}
            <div className="bg-zinc-800 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-3">
                <ClockIcon className="w-4 h-4 text-zinc-400" />
                <h3 className="text-sm font-medium text-zinc-300">Learning Status</h3>
              </div>

              <div className="space-y-2">
                <StatusRow
                  label="Outcome Detection"
                  enabled={stats.learning.outcome_detection_enabled}
                />
                <StatusRow
                  label="Knowledge Graph"
                  enabled={stats.learning.knowledge_graph_active}
                />
              </div>
            </div>

            {/* Relationships */}
            {(stats.relationships.related > 0 || stats.relationships.evolution > 0) && (
              <div className="bg-zinc-800 rounded-lg p-4">
                <h3 className="text-sm font-medium text-zinc-300 mb-2">Memory Relationships</h3>
                <div className="space-y-1 text-xs text-zinc-400">
                  <div className="flex justify-between">
                    <span>Related:</span>
                    <span>{stats.relationships.related}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Evolution:</span>
                    <span>{stats.relationships.evolution}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Last Updated */}
            <div className="text-xs text-zinc-500 text-center">
              Updates every 5 seconds
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// Helper Components

interface StatRowProps {
  label: string;
  value: number;
  color: 'blue' | 'green' | 'amber' | 'purple';
  description: string;
}

const StatRow: React.FC<StatRowProps> = ({ label, value, color, description }) => {
  const colorClasses = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
    amber: 'bg-amber-500',
    purple: 'bg-purple-500'
  };

  return (
    <div className="flex items-center justify-between group">
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${colorClasses[color]}`} />
        <span className="text-sm text-zinc-300">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-sm font-mono text-white">{value}</span>
      </div>
      <div className="absolute left-0 top-full mt-1 hidden group-hover:block bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-400 whitespace-nowrap z-10">
        {description}
      </div>
    </div>
  );
};

interface KGStatRowProps {
  label: string;
  value: number;
  description: string;
}

const KGStatRow: React.FC<KGStatRowProps> = ({ label, value, description }) => {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-zinc-400">{label}</span>
      <span className="font-mono text-white">{value}</span>
      <div className="absolute left-0 top-full mt-1 hidden group-hover:block bg-zinc-950 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-400 whitespace-nowrap z-10">
        {description}
      </div>
    </div>
  );
};

interface StatusRowProps {
  label: string;
  enabled: boolean;
}

const StatusRow: React.FC<StatusRowProps> = ({ label, enabled }) => {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-zinc-400">{label}</span>
      <div className="flex items-center gap-2">
        <div className={`w-2 h-2 rounded-full ${enabled ? 'bg-green-500' : 'bg-zinc-600'}`} />
        <span className="text-xs text-zinc-500">{enabled ? 'Active' : 'Inactive'}</span>
      </div>
    </div>
  );
};

export default MemoryStatsPanel;