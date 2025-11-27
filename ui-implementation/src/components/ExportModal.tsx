import React, { useState, useEffect } from 'react';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
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

export const ExportModal: React.FC<ExportModalProps> = ({ isOpen, onClose }) => {
  const [exportOptions, setExportOptions] = useState<ExportOptions>({
    sessions: true,
    memory: true,
    books: true,
    knowledge: true,
  });
  const [sizeEstimate, setSizeEstimate] = useState<SizeEstimate | null>(null);
  const [isEstimating, setIsEstimating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);

  // Fetch size estimate when options change
  useEffect(() => {
    if (!isOpen) return;

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

        const response = await fetch(
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
  }, [exportOptions, isOpen]);

  const toggleOption = (key: keyof ExportOptions) => {
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

      // Create backup with selected types
      const url = selectedTypes === 'sessions,memory,books,knowledge'
        ? `${ROAMPAL_CONFIG.apiUrl}/api/backup/create`
        : `${ROAMPAL_CONFIG.apiUrl}/api/backup/create?include=${selectedTypes}`;

      const response = await fetch(url, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error(`Backup failed: ${response.statusText}`);
      }

      // Download the backup zip file
      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;

      // Extract filename from Content-Disposition header or use default
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

      // Build success message
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

  const allSelected = Object.values(exportOptions).every((v) => v);
  const noneSelected = Object.values(exportOptions).every((v) => !v);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 z-[60] flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-zinc-900 rounded-xl shadow-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto border border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b border-zinc-800">
          <h2 className="text-xl font-bold">Export Data</h2>
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

        {/* Content */}
        <div className="p-6 space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-sm text-zinc-400">Select data to include:</p>
            <button
              onClick={() => {
                if (allSelected) {
                  setExportOptions({ sessions: false, memory: false, books: false, knowledge: false });
                } else {
                  setExportOptions({ sessions: true, memory: true, books: true, knowledge: true });
                }
              }}
              className="text-xs text-blue-500 hover:text-blue-400 transition-colors font-medium"
            >
              {allSelected ? 'Deselect All' : 'Select All'}
            </button>
          </div>

          {/* Export Options */}
          <div className="space-y-2">
            {/* Sessions */}
            <label className="flex items-start gap-2 p-2.5 bg-zinc-800/50 hover:bg-zinc-800 rounded-lg cursor-pointer transition-colors">
              <input
                type="checkbox"
                checked={exportOptions.sessions}
                onChange={() => toggleOption('sessions')}
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

            {/* Memory */}
            <label className="flex items-start gap-2 p-2.5 bg-zinc-800/50 hover:bg-zinc-800 rounded-lg cursor-pointer transition-colors">
              <input
                type="checkbox"
                checked={exportOptions.memory}
                onChange={() => toggleOption('memory')}
                className="mt-0.5 w-4 h-4 rounded border-zinc-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
              />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium">Memory (ChromaDB)</div>
                <div className="text-xs text-zinc-400">
                  {sizeEstimate?.file_counts.memory || 0} files
                  {sizeEstimate?.breakdown.memory_mb !== undefined &&
                    ` • ${sizeEstimate.breakdown.memory_mb.toFixed(1)} MB`
                  }
                </div>
              </div>
            </label>

            {/* Books */}
            <label className="flex items-start gap-2 p-2.5 bg-zinc-800/50 hover:bg-zinc-800 rounded-lg cursor-pointer transition-colors">
              <input
                type="checkbox"
                checked={exportOptions.books}
                onChange={() => toggleOption('books')}
                className="mt-0.5 w-4 h-4 rounded border-zinc-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
              />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium">Books & Documents</div>
                <div className="text-xs text-zinc-400">
                  {sizeEstimate?.file_counts.books || 0} files
                  {sizeEstimate?.breakdown.books_mb !== undefined &&
                    ` • ${sizeEstimate.breakdown.books_mb.toFixed(1)} MB`
                  }
                </div>
              </div>
            </label>

            {/* Knowledge */}
            <label className="flex items-start gap-2 p-2.5 bg-zinc-800/50 hover:bg-zinc-800 rounded-lg cursor-pointer transition-colors">
              <input
                type="checkbox"
                checked={exportOptions.knowledge}
                onChange={() => toggleOption('knowledge')}
                className="mt-0.5 w-4 h-4 rounded border-zinc-600 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
              />
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium">Knowledge & Learning</div>
                <div className="text-xs text-zinc-400">
                  {sizeEstimate?.file_counts.knowledge || 0} files
                  {sizeEstimate?.breakdown.knowledge_mb !== undefined &&
                    ` • ${(sizeEstimate.breakdown.knowledge_mb < 0.1 ? '< 0.1' : sizeEstimate.breakdown.knowledge_mb.toFixed(1))} MB`
                  }
                </div>
              </div>
            </label>
          </div>

          {/* Size Estimate Summary */}
          {sizeEstimate && !noneSelected && (
            <div className="p-3 bg-blue-600/10 border border-blue-600/20 rounded-lg">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-blue-400">
                  Total size:
                </span>
                <span className="text-sm font-bold text-blue-300">
                  {isEstimating ? '...' : `${sizeEstimate.total_mb.toFixed(1)} MB`}
                </span>
              </div>
            </div>
          )}

          {/* Export Button */}
          <button
            onClick={handleExportData}
            disabled={noneSelected || isExporting}
            className={`w-full h-10 px-3 py-2 flex items-center justify-center gap-2 rounded-lg font-medium transition-colors ${
              noneSelected || isExporting
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
      </div>
    </div>
  );
};
