import React, { useState, useEffect } from 'react';
import { DeleteConfirmationModal } from './DeleteConfirmationModal';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface DataManagementModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialTab?: 'export' | 'delete' | 'summarize';
}

interface ExportOptions {
  sessions: boolean;
  memory: boolean;
  books: boolean;
  knowledge: boolean;
}

interface SizeEstimate {
  total_mb: number;
  breakdown: {
    sessions_mb?: number;
    memory_mb?: number;
    books_mb?: number;
    knowledge_mb?: number;
  };
  file_counts: {
    sessions?: number;
    memory?: number;
    books?: number;
    knowledge?: number;
  };
}

interface DataStats {
  memory_bank: { count: number; active: number; archived: number };
  working: { count: number };
  history: { count: number };
  patterns: { count: number };
  books: { count: number };
  sessions: { count: number };
  knowledge_graph: { nodes: number; edges: number };
  // v0.3.0: New data types
  outcomes?: { exists: boolean };
}

type ActiveTab = 'export' | 'delete' | 'summarize';

interface ScanResult {
  candidates: Record<string, number>;
  total: number;
  sidecar_configured: boolean;
}

interface SummarizeResult {
  summarized: number;
  facts: number;
  tags: number;
}
type DeleteTarget = 'memory_bank' | 'working' | 'history' | 'patterns' | 'books' | 'sessions' | 'knowledge-graph' | 'outcomes' | null;

export const DataManagementModal: React.FC<DataManagementModalProps> = ({ isOpen, onClose, initialTab }) => {
  const [activeTab, setActiveTab] = useState<ActiveTab>(initialTab || 'export');

  // Export tab state
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    sessions: true,
    memory: true,
    books: true,
    knowledge: true,
  });
  const [sizeEstimate, setSizeEstimate] = useState<SizeEstimate | null>(null);
  const [isEstimating, setIsEstimating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  // Delete tab state
  const [dataStats, setDataStats] = useState<DataStats | null>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(false);
  const [statsError, setStatsError] = useState<string | null>(null);  // v0.3.0: Show errors to user
  const [deleteTarget, setDeleteTarget] = useState<DeleteTarget>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [isCompacting, setIsCompacting] = useState(false);

  // Summarize tab state
  const [scanResult, setScanResult] = useState<ScanResult | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summarizeProgress, setSummarizeProgress] = useState<{ current: number; total: number; message: string } | null>(null);
  const [summarizeResult, setSummarizeResult] = useState<SummarizeResult | null>(null);
  const [summarizeError, setSummarizeError] = useState<string | null>(null);
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  // Fetch export size estimate
  useEffect(() => {
    if (!isOpen || activeTab !== 'export') return;

    const fetchEstimate = async () => {
      setIsEstimating(true);
      try {
        const selectedTypes = Object.keys(exportOptions)
          .filter((key) => exportOptions[key as keyof ExportOptions])
          .join(',');

        if (!selectedTypes) {
          setSizeEstimate(null);
          return;
        }

        const response = await apiFetch(
          `${ROAMPAL_CONFIG.apiUrl}/api/backup/estimate?include=${selectedTypes}`
        );

        if (response.ok) {
          const data = await response.json();
          setSizeEstimate(data);
        }
      } catch (error) {
        console.error('Failed to estimate size:', error);
      } finally {
        setIsEstimating(false);
      }
    };

    fetchEstimate();
  }, [exportOptions, isOpen, activeTab]);

  // Fetch stats for both tabs (needed for accurate memory counts in export)
  useEffect(() => {
    if (!isOpen) return;

    fetchDataStats();
  }, [isOpen, activeTab]);

  // Honor initialTab prop
  useEffect(() => {
    if (isOpen && initialTab) {
      setActiveTab(initialTab);
    }
  }, [isOpen, initialTab]);

  // Scan for summarize candidates when tab opens
  useEffect(() => {
    if (!isOpen || activeTab !== 'summarize') return;
    const scanForCandidates = async () => {
      setIsScanning(true);
      try {
        const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/data/summarize/scan`);
        if (response.ok) {
          setScanResult(await response.json());
        }
      } catch (error) {
        console.error('Summarize scan failed:', error);
      } finally {
        setIsScanning(false);
      }
    };
    scanForCandidates();
  }, [isOpen, activeTab]);

  const handleStartSummarize = async () => {
    setIsSummarizing(true);
    setSummarizeResult(null);
    setSummarizeError(null);
    setSummarizeProgress(null);
    const controller = new AbortController();
    setAbortController(controller);

    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/data/summarize/run`, {
        method: 'POST',
        signal: controller.signal,
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'progress') {
              setSummarizeProgress({ current: data.current, total: data.total, message: data.message });
            } else if (data.type === 'complete') {
              setSummarizeResult({ summarized: data.summarized, facts: data.facts, tags: data.tags });
            } else if (data.type === 'cancelled') {
              setSummarizeResult({ summarized: data.summarized, facts: data.facts, tags: data.tags });
              setSummarizeError('Cancelled by user');
            }
          } catch {
            // Skip malformed SSE lines
          }
        }
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        setSummarizeError(error.message || 'Summarization failed');
      }
    } finally {
      setIsSummarizing(false);
      setAbortController(null);
      // Refresh scan results and trigger memory panel update
      try {
        const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/data/summarize/scan`);
        if (response.ok) setScanResult(await response.json());
      } catch { /* ignore */ }
      window.dispatchEvent(new CustomEvent('memoryUpdated', { detail: { source: 'summarize' } }));
    }
  };

  const handleCancelSummarize = async () => {
    abortController?.abort();
    try {
      await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/data/summarize/cancel`, { method: 'POST' });
    } catch { /* best effort */ }
  };

  const fetchDataStats = async () => {
    setIsLoadingStats(true);
    setStatsError(null);
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/data/stats`);
      if (response.ok) {
        const data = await response.json();
        setDataStats(data);
      } else {
        // v0.3.0: Show error when stats fail to load (delete buttons would be disabled otherwise)
        const errorText = await response.text();
        console.error('Stats API error:', response.status, errorText);
        setStatsError(`Failed to load data stats (${response.status})`);
      }
    } catch (error: any) {
      console.error('Failed to fetch data stats:', error);
      setStatsError(error.message || 'Failed to connect to backend');
    } finally {
      setIsLoadingStats(false);
    }
  };

  const toggleExportOption = (key: keyof ExportOptions) => {
    setExportOptions((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const handleExportData = async () => {
    try {
      setIsExporting(true);

      const selectedTypes = Object.keys(exportOptions)
        .filter((key) => exportOptions[key as keyof ExportOptions])
        .join(',');

      if (!selectedTypes) {
        alert('⚠️ Please select at least one data type to export');
        return;
      }

      const url = selectedTypes === 'sessions,memory,books,knowledge'
        ? `${ROAMPAL_CONFIG.apiUrl}/api/backup/create`
        : `${ROAMPAL_CONFIG.apiUrl}/api/backup/create?include=${selectedTypes}`;

      const response = await apiFetch(url, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`Backup failed: ${response.statusText}`);
      }

      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;

      const contentDisposition = response.headers.get('Content-Disposition');
      let filename = `roampal_backup_${new Date().toISOString().split('T')[0]}.zip`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="?(.+)"?/);
        if (filenameMatch) {
          filename = filenameMatch[1];
        }
      }

      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(downloadUrl);

      const includedTypes = Object.keys(exportOptions)
        .filter((key) => exportOptions[key as keyof ExportOptions])
        .map(type => `• ${type.charAt(0).toUpperCase() + type.slice(1)}`)
        .join('\n');

      alert(`✅ Export created successfully!\n\nIncluded:\n${includedTypes}\n\nSize: ${sizeEstimate?.total_mb.toFixed(1)} MB\n\nKeep this file safe!`);

      onClose();
    } catch (error) {
      console.error('Export failed:', error);
      alert('❌ Failed to create export. Please check console for details.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleDeleteClick = (target: DeleteTarget) => {
    setDeleteTarget(target);
    setShowDeleteConfirm(true);
  };

  const handleDeleteConfirm = async () => {
    if (!deleteTarget) return;

    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/data/clear/${deleteTarget}`, {
        method: 'POST'
      });

      if (!response.ok) {
        // v0.3.0: Handle non-JSON error responses gracefully
        let errorMsg = `Delete failed (${response.status})`;
        try {
          const error = await response.json();
          errorMsg = error.detail || errorMsg;
        } catch {
          // Response body wasn't JSON, use status-based message
        }
        throw new Error(errorMsg);
      }

      const result = await response.json();

      // Show success message
      alert(`✅ Deleted successfully!\n\n${result.deleted_count || 0} items removed`);

      // Refresh stats
      await fetchDataStats();

      // Notify memory panel to refresh (memories deleted)
      window.dispatchEvent(new CustomEvent('memoryUpdated', {
        detail: { source: 'data_delete', target: deleteTarget, timestamp: new Date().toISOString() }
      }));

      setShowDeleteConfirm(false);
      setDeleteTarget(null);
    } catch (error: any) {
      console.error('Delete failed:', error);
      alert(`❌ Delete failed: ${error.message}`);
    }
  };

  const handleCompactDatabase = async () => {
    try {
      setIsCompacting(true);

      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/data/compact-database`, {
        method: 'POST'
      });

      if (!response.ok) {
        // v0.3.0: Handle non-JSON error responses gracefully
        let errorMsg = `Compaction failed (${response.status})`;
        try {
          const error = await response.json();
          errorMsg = error.detail || errorMsg;
        } catch {
          // Response body wasn't JSON, use status-based message
        }
        throw new Error(errorMsg);
      }

      const result = await response.json();

      if (result.space_reclaimed_mb > 0.1) {
        alert(`✅ Database compacted!\n\nReclaimed ${result.space_reclaimed_mb.toFixed(1)} MB of disk space`);
      } else {
        alert(`ℹ️ Database already optimized\n\nNo significant space to reclaim (${result.space_reclaimed_mb.toFixed(2)} MB)`);
      }
    } catch (error: any) {
      console.error('Compaction failed:', error);
      alert(`❌ Compaction failed: ${error.message}`);
    } finally {
      setIsCompacting(false);
    }
  };

  const allExportSelected = Object.values(exportOptions).every((v) => v);
  const noneExportSelected = Object.values(exportOptions).every((v) => !v);

  if (!isOpen) return null;

  const getDeleteItemCount = (collection: string): number => {
    if (!dataStats) return 0;
    switch (collection) {
      case 'memory_bank':
        return dataStats.memory_bank.count;
      case 'working':
        return dataStats.working.count;
      case 'history':
        return dataStats.history.count;
      case 'patterns':
        return dataStats.patterns.count;
      case 'books':
        return dataStats.books.count;
      case 'sessions':
        return dataStats.sessions.count;
      case 'knowledge-graph':
        return (dataStats.knowledge_graph?.nodes || 0) + (dataStats.knowledge_graph?.edges || 0);
      case 'outcomes':
        return dataStats.outcomes?.exists ? 1 : 0;
      default:
        return 0;
    }
  };

  const getDeleteMessage = (collection: string): string => {
    switch (collection) {
      case 'memory_bank':
        return 'This will permanently delete all memories AI has stored about you. This includes identity, preferences, goals, and context.';
      case 'working':
        return 'This will clear current conversation context from the last 24 hours.';
      case 'history':
        return 'This will delete past conversation history (30-day retention).';
      case 'patterns':
        return 'This will remove all proven solution patterns AI has learned.';
      case 'books':
        return 'This will delete all uploaded books and reference documents.';
      case 'sessions':
        return 'This will delete all conversation session files. Your active conversation will be preserved.';
      case 'knowledge-graph':
        return 'This will clear all concept relationships and connections AI has built.';
      case 'outcomes':
        return 'This will clear all outcome tracking data used for memory scoring and ranking.';
      default:
        return 'This action cannot be undone.';
    }
  };

  return (
    <>
      <div
        className="fixed inset-0 bg-black/50 z-[60] flex items-center justify-center p-4"
        onClick={onClose}
      >
        <div
          className="bg-zinc-900 rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-hidden border border-zinc-800 flex flex-col"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex justify-between items-center p-6 border-b border-zinc-800">
            <h2 className="text-xl font-bold">Data Management</h2>
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
          <div className="flex border-b border-zinc-800">
            <button
              onClick={() => setActiveTab('export')}
              className={`flex-1 px-6 py-3 font-medium transition-colors ${
                activeTab === 'export'
                  ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-600/5'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export
              </div>
            </button>
            <button
              onClick={() => setActiveTab('summarize')}
              className={`flex-1 px-6 py-3 font-medium transition-colors ${
                activeTab === 'summarize'
                  ? 'text-amber-400 border-b-2 border-amber-400 bg-amber-600/5'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Summarize
              </div>
            </button>
            <button
              onClick={() => setActiveTab('delete')}
              className={`flex-1 px-6 py-3 font-medium transition-colors ${
                activeTab === 'delete'
                  ? 'text-red-400 border-b-2 border-red-400 bg-red-600/5'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50'
              }`}
            >
              <div className="flex items-center justify-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                Delete
              </div>
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto">
            {activeTab === 'summarize' ? (
              <div className="p-6 space-y-4">
                {/* Sidecar warning */}
                {scanResult && !scanResult.sidecar_configured && (
                  <div className="p-4 bg-amber-600/10 border border-amber-600/20 rounded-lg">
                    <p className="text-sm text-amber-400 font-medium">Sidecar model required</p>
                    <p className="text-xs text-zinc-400 mt-1">
                      Configure a sidecar model in the header dropdown to enable summarization.
                    </p>
                  </div>
                )}

                {/* Description */}
                <div className="p-4 bg-amber-600/10 border border-amber-600/20 rounded-lg">
                  <p className="text-sm text-amber-400 font-medium">Optimize Legacy Memories</p>
                  <p className="text-xs text-zinc-400 mt-1">
                    Summarize old raw exchanges into concise notes, extract searchable tags and atomic facts.
                    Improves retrieval quality for memories created before the sidecar was active.
                  </p>
                </div>

                {/* Scan results */}
                {isScanning ? (
                  <div className="flex items-center justify-center py-8">
                    <svg className="animate-spin w-6 h-6 text-zinc-500" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    <span className="ml-3 text-sm text-zinc-400">Scanning memories...</span>
                  </div>
                ) : scanResult ? (
                  <>
                    {scanResult.total === 0 ? (
                      <div className="p-4 bg-green-600/10 border border-green-600/20 rounded-lg">
                        <p className="text-sm text-green-400 font-medium">All memories are optimized</p>
                        <p className="text-xs text-zinc-400 mt-1">No unsummarized memories found.</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <p className="text-sm text-zinc-300 font-medium">
                          {scanResult.total} {scanResult.total === 1 ? 'memory' : 'memories'} can be summarized
                        </p>
                        {Object.entries(scanResult.candidates).map(([coll, count]) => (
                          count > 0 && (
                            <div key={coll} className="flex items-center justify-between px-3 py-2 bg-zinc-800/50 rounded-lg">
                              <span className="text-sm text-zinc-400 capitalize">{coll}</span>
                              <span className="text-sm text-zinc-300">{count}</span>
                            </div>
                          )
                        ))}
                      </div>
                    )}
                  </>
                ) : null}

                {/* Progress */}
                {isSummarizing && summarizeProgress && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-zinc-300">{summarizeProgress.message}</span>
                      <span className="text-xs text-zinc-500">
                        {summarizeProgress.current}/{summarizeProgress.total}
                      </span>
                    </div>
                    <div className="w-full bg-zinc-800 rounded-full h-2">
                      <div
                        className="bg-amber-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${(summarizeProgress.current / summarizeProgress.total) * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Results */}
                {summarizeResult && (
                  <div className={`p-4 rounded-lg border ${summarizeError ? 'bg-yellow-600/10 border-yellow-600/20' : 'bg-green-600/10 border-green-600/20'}`}>
                    <p className={`text-sm font-medium ${summarizeError ? 'text-yellow-400' : 'text-green-400'}`}>
                      {summarizeError ? 'Partially completed' : 'Migration complete'}
                    </p>
                    <div className="mt-2 space-y-1">
                      <p className="text-xs text-zinc-400">{summarizeResult.summarized} memories summarized</p>
                      <p className="text-xs text-zinc-400">{summarizeResult.facts} facts extracted</p>
                      <p className="text-xs text-zinc-400">{summarizeResult.tags} tags added</p>
                    </div>
                  </div>
                )}

                {/* Error */}
                {summarizeError && !summarizeResult && (
                  <div className="p-4 bg-red-600/10 border border-red-600/20 rounded-lg">
                    <p className="text-sm text-red-400">{summarizeError}</p>
                  </div>
                )}

                {/* Action buttons */}
                <div className="pt-2">
                  {isSummarizing ? (
                    <button
                      onClick={handleCancelSummarize}
                      className="w-full h-10 px-3 py-2 flex items-center justify-center gap-2 rounded-lg font-medium bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 text-zinc-300 transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                      <span className="text-sm">Cancel</span>
                    </button>
                  ) : (
                    <button
                      onClick={handleStartSummarize}
                      disabled={!scanResult || scanResult.total === 0 || !scanResult.sidecar_configured}
                      className={`w-full h-10 px-3 py-2 flex items-center justify-center gap-2 rounded-lg font-medium transition-colors ${
                        !scanResult || scanResult.total === 0 || !scanResult.sidecar_configured
                          ? 'bg-zinc-800 text-zinc-600 cursor-not-allowed border border-zinc-700'
                          : 'bg-amber-600/10 hover:bg-amber-600/20 border border-amber-600/30 text-amber-500'
                      }`}
                    >
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      <span className="text-sm">Summarize All</span>
                    </button>
                  )}
                </div>
              </div>
            ) : activeTab === 'export' ? (
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-zinc-400">Select data to include:</p>
                  <button
                    onClick={() => {
                      if (allExportSelected) {
                        setExportOptions({ sessions: false, memory: false, books: false, knowledge: false });
                      } else {
                        setExportOptions({ sessions: true, memory: true, books: true, knowledge: true });
                      }
                    }}
                    className="text-xs text-blue-500 hover:text-blue-400 transition-colors font-medium"
                  >
                    {allExportSelected ? 'Deselect All' : 'Select All'}
                  </button>
                </div>

                {/* Export Options */}
                <div className="space-y-2">
                  <label className="flex items-start gap-2 p-2.5 bg-zinc-800/50 hover:bg-zinc-800 rounded-lg cursor-pointer transition-colors">
                    <input
                      type="checkbox"
                      checked={exportOptions.sessions}
                      onChange={() => toggleExportOption('sessions')}
                      className="mt-0.5 w-4 h-4 rounded border-zinc-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium">Conversations</div>
                      <div className="text-xs text-zinc-400">
                        {sizeEstimate?.file_counts.sessions || 0} sessions
                        {sizeEstimate?.breakdown.sessions_mb !== undefined &&
                          ` • ${sizeEstimate.breakdown.sessions_mb.toFixed(1)} MB`
                        }
                      </div>
                    </div>
                  </label>

                  <label className="flex items-start gap-2 p-2.5 bg-zinc-800/50 hover:bg-zinc-800 rounded-lg cursor-pointer transition-colors">
                    <input
                      type="checkbox"
                      checked={exportOptions.memory}
                      onChange={() => toggleExportOption('memory')}
                      className="mt-0.5 w-4 h-4 rounded border-zinc-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium">Memory (ChromaDB)</div>
                      <div className="text-xs text-zinc-400">
                        {dataStats ? (
                          <>
                            {(dataStats.memory_bank.count + dataStats.working.count + dataStats.history.count + dataStats.patterns.count)} memories
                            {sizeEstimate?.breakdown.memory_mb !== undefined &&
                              ` • ${sizeEstimate.breakdown.memory_mb.toFixed(1)} MB`
                            }
                          </>
                        ) : (
                          <>
                            {sizeEstimate?.breakdown.memory_mb !== undefined &&
                              `${sizeEstimate.breakdown.memory_mb.toFixed(1)} MB`
                            }
                          </>
                        )}
                      </div>
                    </div>
                  </label>

                  <label className="flex items-start gap-2 p-2.5 bg-zinc-800/50 hover:bg-zinc-800 rounded-lg cursor-pointer transition-colors">
                    <input
                      type="checkbox"
                      checked={exportOptions.books}
                      onChange={() => toggleExportOption('books')}
                      className="mt-0.5 w-4 h-4 rounded border-zinc-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium">Books & Documents</div>
                      <div className="text-xs text-zinc-400">
                        {dataStats ? (
                          <>
                            {dataStats.books.count} books
                            {sizeEstimate?.breakdown.books_mb !== undefined &&
                              ` • ${sizeEstimate.breakdown.books_mb.toFixed(1)} MB`
                            }
                            {dataStats.books.count === 0 && sizeEstimate?.breakdown.books_mb && sizeEstimate.breakdown.books_mb > 0.1 && (
                              <span className="text-zinc-500"> (empty database)</span>
                            )}
                          </>
                        ) : (
                          <>
                            {sizeEstimate?.breakdown.books_mb !== undefined &&
                              `${sizeEstimate.breakdown.books_mb.toFixed(1)} MB`
                            }
                          </>
                        )}
                      </div>
                    </div>
                  </label>

                  <label className="flex items-start gap-2 p-2.5 bg-zinc-800/50 hover:bg-zinc-800 rounded-lg cursor-pointer transition-colors">
                    <input
                      type="checkbox"
                      checked={exportOptions.knowledge}
                      onChange={() => toggleExportOption('knowledge')}
                      className="mt-0.5 w-4 h-4 rounded border-zinc-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium">Knowledge & Learning</div>
                      <div className="text-xs text-zinc-400">
                        {dataStats ? (
                          <>
                            {(dataStats.knowledge_graph?.nodes || 0) + (dataStats.knowledge_graph?.edges || 0)} items
                            {sizeEstimate?.breakdown.knowledge_mb !== undefined &&
                              ` • ${(sizeEstimate.breakdown.knowledge_mb < 0.1 ? '< 0.1' : sizeEstimate.breakdown.knowledge_mb.toFixed(1))} MB`
                            }
                            {((dataStats.knowledge_graph?.nodes || 0) + (dataStats.knowledge_graph?.edges || 0)) === 0 && sizeEstimate?.breakdown.knowledge_mb && sizeEstimate.breakdown.knowledge_mb > 0 && (
                              <span className="text-zinc-500"> (empty files)</span>
                            )}
                          </>
                        ) : (
                          <>
                            {sizeEstimate?.breakdown.knowledge_mb !== undefined &&
                              `${(sizeEstimate.breakdown.knowledge_mb < 0.1 ? '< 0.1' : sizeEstimate.breakdown.knowledge_mb.toFixed(1))} MB`
                            }
                          </>
                        )}
                      </div>
                    </div>
                  </label>
                </div>

                {/* Size Estimate */}
                {sizeEstimate && !noneExportSelected && (
                  <div className="p-3 bg-blue-600/10 border border-blue-600/20 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-blue-400">Backup size:</span>
                      <span className="text-sm font-bold text-blue-300">
                        {isEstimating ? '...' : `${sizeEstimate.total_mb.toFixed(1)} MB`}
                      </span>
                    </div>
                    {dataStats &&
                     (dataStats.memory_bank.count + dataStats.working.count + dataStats.history.count + dataStats.patterns.count) === 0 &&
                     sizeEstimate.breakdown.memory_mb && sizeEstimate.breakdown.memory_mb > 1 && (
                      <div className="mt-1.5 text-xs text-blue-400/70">
                        Includes {sizeEstimate.breakdown.memory_mb.toFixed(1)} MB database infrastructure (0 memories stored)
                      </div>
                    )}
                  </div>
                )}

                {/* Export Button */}
                <button
                  onClick={handleExportData}
                  disabled={noneExportSelected || isExporting}
                  className={`w-full h-10 px-3 py-2 flex items-center justify-center gap-2 rounded-lg font-medium transition-colors ${
                    noneExportSelected || isExporting
                      ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed border border-zinc-700'
                      : 'bg-blue-600/10 hover:bg-blue-600/20 border border-blue-600/30 text-blue-500'
                  }`}
                >
                  {isExporting ? (
                    <>
                      <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      <span className="text-sm">Creating Export...</span>
                    </>
                  ) : (
                    <>
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      <span className="text-sm">Export Selected Data</span>
                    </>
                  )}
                </button>
              </div>
            ) : (
              <div className="p-6 space-y-4">
                <div className="p-4 bg-red-600/10 border border-red-600/20 rounded-lg">
                  <p className="text-sm text-red-400 font-medium">⚠️ Danger Zone - These actions are permanent</p>
                </div>

                {isLoadingStats ? (
                  <div className="flex items-center justify-center py-12">
                    <svg className="animate-spin w-8 h-8 text-zinc-500" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                  </div>
                ) : statsError ? (
                  // v0.3.0: Show error when stats fail to load
                  <div className="p-4 bg-yellow-600/10 border border-yellow-600/30 rounded-lg">
                    <p className="text-sm text-yellow-400 font-medium">⚠️ {statsError}</p>
                    <p className="text-xs text-zinc-400 mt-2">Unable to show data counts. The backend may not be running.</p>
                    <button
                      onClick={fetchDataStats}
                      className="mt-3 px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded text-sm text-zinc-300 transition-colors"
                    >
                      Retry
                    </button>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {[
                      { key: 'memory_bank', label: 'Memory Bank', desc: 'User facts & preferences', count: dataStats?.memory_bank.count || 0 },
                      { key: 'working', label: 'Working Memory', desc: 'Current context (24h)', count: dataStats?.working.count || 0 },
                      { key: 'history', label: 'History', desc: 'Past conversations (30d)', count: dataStats?.history.count || 0 },
                      { key: 'patterns', label: 'Patterns', desc: 'Proven solutions', count: dataStats?.patterns.count || 0 },
                      { key: 'books', label: 'Books', desc: 'Reference documents', count: dataStats?.books.count || 0 },
                      { key: 'sessions', label: 'Sessions', desc: 'Conversation files', count: dataStats?.sessions.count || 0 },
                      { key: 'knowledge-graph', label: 'Knowledge Graph', desc: 'Concept relationships', count: (dataStats?.knowledge_graph?.nodes || 0) + (dataStats?.knowledge_graph?.edges || 0) },
                      { key: 'outcomes', label: 'Outcomes', desc: 'Scoring history', count: dataStats?.outcomes?.exists ? 1 : 0 }
                    ].map((item) => (
                      <div key={item.key} className="flex items-center justify-between p-3 bg-zinc-800/50 rounded-lg border border-zinc-700">
                        <div className="flex-1">
                          <div className="text-sm font-medium text-zinc-200">{item.label}</div>
                          <div className="text-xs text-zinc-500">{item.desc}</div>
                        </div>
                        <div className="flex items-center gap-3">
                          <span className="text-sm text-zinc-400">{item.count} items</span>
                          <button
                            onClick={() => handleDeleteClick(item.key as DeleteTarget)}
                            disabled={item.count === 0}
                            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                              item.count === 0
                                ? 'bg-zinc-800 text-zinc-600 cursor-not-allowed'
                                : 'bg-red-600/10 hover:bg-red-600/20 border border-red-600/30 text-red-500'
                            }`}
                          >
                            Delete
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Compact Database Button */}
                {!isLoadingStats && (
                  <div className="pt-4 border-t border-zinc-800">
                    <button
                      onClick={handleCompactDatabase}
                      disabled={isCompacting}
                      className={`w-full h-10 px-3 py-2 flex items-center justify-center gap-2 rounded-lg font-medium transition-colors ${
                        isCompacting
                          ? 'bg-zinc-800 text-zinc-500 cursor-not-allowed border border-zinc-700'
                          : 'bg-blue-600/10 hover:bg-blue-600/20 border border-blue-600/30 text-blue-500'
                      }`}
                    >
                      {isCompacting ? (
                        <>
                          <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                          <span className="text-sm">Compacting Database...</span>
                        </>
                      ) : (
                        <>
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                          </svg>
                          <span className="text-sm">Compact Database (Reclaim Disk Space)</span>
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      <DeleteConfirmationModal
        isOpen={showDeleteConfirm}
        onClose={() => {
          setShowDeleteConfirm(false);
          setDeleteTarget(null);
        }}
        onConfirm={handleDeleteConfirm}
        title={`Delete ${deleteTarget?.replace('-', ' ').replace(/_/g, ' ')}?`}
        message={deleteTarget ? getDeleteMessage(deleteTarget) : ''}
        itemCount={deleteTarget ? getDeleteItemCount(deleteTarget) : 0}
        collectionName={deleteTarget || undefined}
      />
    </>
  );
};
