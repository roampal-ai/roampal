import React, { useState } from 'react';
import { FileText, Check, X, ChevronDown, ChevronUp, GitBranch, AlertCircle } from 'lucide-react';

interface CodeChange {
  change_id: string;
  file_path: string;
  diff: string;
  description: string;
  status: 'pending' | 'applied' | 'skipped';
  lines_added?: number;
  lines_removed?: number;
}

interface CodeChangePreviewProps {
  changes: CodeChange[];
  onApply: (changeId: string) => void;
  onSkip: (changeId: string) => void;
  onApplyAll: () => void;
}

export const CodeChangePreview: React.FC<CodeChangePreviewProps> = ({
  changes,
  onApply,
  onSkip,
  onApplyAll
}) => {
  const [expandedChanges, setExpandedChanges] = useState<Set<string>>(new Set());
  const [appliedChanges, setAppliedChanges] = useState<Set<string>>(new Set());
  const [skippedChanges, setSkippedChanges] = useState<Set<string>>(new Set());

  const toggleExpanded = (changeId: string) => {
    const newExpanded = new Set(expandedChanges);
    if (newExpanded.has(changeId)) {
      newExpanded.delete(changeId);
    } else {
      newExpanded.add(changeId);
    }
    setExpandedChanges(newExpanded);
  };

  const handleApply = (changeId: string) => {
    setAppliedChanges(new Set([...appliedChanges, changeId]));
    onApply(changeId);
  };

  const handleSkip = (changeId: string) => {
    setSkippedChanges(new Set([...skippedChanges, changeId]));
    onSkip(changeId);
  };

  const handleApplyAll = () => {
    const allChangeIds = changes.map(c => c.change_id);
    setAppliedChanges(new Set(allChangeIds));
    onApplyAll();
  };

  const pendingChanges = changes.filter(
    c => !appliedChanges.has(c.change_id) && !skippedChanges.has(c.change_id)
  );

  if (changes.length === 0) return null;

  return (
    <div className="my-4 border border-zinc-800 rounded-lg bg-zinc-900/50 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 bg-zinc-800/30 border-b border-zinc-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <GitBranch className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-zinc-300">
              Code Changes Preview
            </span>
            <span className="text-xs text-zinc-500">
              ({pendingChanges.length} pending)
            </span>
          </div>
          {pendingChanges.length > 0 && (
            <button
              onClick={handleApplyAll}
              className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors flex items-center space-x-1"
            >
              <Check className="w-3 h-3" />
              <span>Apply All</span>
            </button>
          )}
        </div>
      </div>

      {/* Changes List */}
      <div className="divide-y divide-zinc-800">
        {changes.map((change) => {
          const isExpanded = expandedChanges.has(change.change_id);
          const isApplied = appliedChanges.has(change.change_id);
          const isSkipped = skippedChanges.has(change.change_id);
          const isPending = !isApplied && !isSkipped;

          return (
            <div key={change.change_id} className="relative">
              {/* Change Header */}
              <div className="px-4 py-3 hover:bg-zinc-800/20 transition-colors">
                <div className="flex items-start justify-between">
                  <button
                    onClick={() => toggleExpanded(change.change_id)}
                    className="flex-1 flex items-start space-x-3 text-left"
                  >
                    <FileText className="w-4 h-4 text-zinc-400 mt-0.5" />
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-mono text-zinc-300">
                          {change.file_path}
                        </span>
                        {isApplied && (
                          <span className="text-xs text-green-500 flex items-center space-x-1">
                            <Check className="w-3 h-3" />
                            <span>Applied</span>
                          </span>
                        )}
                        {isSkipped && (
                          <span className="text-xs text-zinc-500 flex items-center space-x-1">
                            <X className="w-3 h-3" />
                            <span>Skipped</span>
                          </span>
                        )}
                      </div>
                      <div className="text-xs text-zinc-500 mt-1">
                        {change.description}
                      </div>
                      {(change.lines_added || change.lines_removed) && (
                        <div className="flex items-center space-x-3 text-xs mt-1">
                          {change.lines_added && change.lines_added > 0 && (
                            <span className="text-green-400">
                              +{change.lines_added}
                            </span>
                          )}
                          {change.lines_removed && change.lines_removed > 0 && (
                            <span className="text-red-400">
                              -{change.lines_removed}
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                    {isExpanded ? (
                      <ChevronUp className="w-4 h-4 text-zinc-500" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-zinc-500" />
                    )}
                  </button>

                  {/* Action Buttons */}
                  {isPending && (
                    <div className="flex items-center space-x-2 ml-4">
                      <button
                        onClick={() => handleApply(change.change_id)}
                        className="px-2 py-1 text-xs bg-green-600/20 hover:bg-green-600/30 text-green-400 rounded transition-colors flex items-center space-x-1"
                      >
                        <Check className="w-3 h-3" />
                        <span>Apply</span>
                      </button>
                      <button
                        onClick={() => handleSkip(change.change_id)}
                        className="px-2 py-1 text-xs bg-zinc-700/50 hover:bg-zinc-700 text-zinc-400 rounded transition-colors flex items-center space-x-1"
                      >
                        <X className="w-3 h-3" />
                        <span>Skip</span>
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Diff View */}
              {isExpanded && (
                <div className="px-4 pb-3 bg-black/30">
                  <div className="bg-zinc-950 rounded border border-zinc-800 overflow-hidden">
                    <pre className="p-3 text-xs font-mono overflow-x-auto">
                      {change.diff.split('\n').map((line, idx) => {
                        let className = 'text-zinc-400';
                        if (line.startsWith('+') && !line.startsWith('+++')) {
                          className = 'text-green-400 bg-green-900/20';
                        } else if (line.startsWith('-') && !line.startsWith('---')) {
                          className = 'text-red-400 bg-red-900/20';
                        } else if (line.startsWith('@@')) {
                          className = 'text-blue-400 bg-blue-900/20';
                        } else if (line.startsWith('diff') || line.startsWith('index')) {
                          className = 'text-zinc-500';
                        }

                        return (
                          <div key={idx} className={`${className} leading-5`}>
                            {line || ' '}
                          </div>
                        );
                      })}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Footer Summary */}
      {(appliedChanges.size > 0 || skippedChanges.size > 0) && (
        <div className="px-4 py-2 bg-zinc-800/20 border-t border-zinc-800">
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-3 text-zinc-500">
              {appliedChanges.size > 0 && (
                <span className="flex items-center space-x-1">
                  <Check className="w-3 h-3 text-green-500" />
                  <span>{appliedChanges.size} applied</span>
                </span>
              )}
              {skippedChanges.size > 0 && (
                <span className="flex items-center space-x-1">
                  <X className="w-3 h-3 text-zinc-500" />
                  <span>{skippedChanges.size} skipped</span>
                </span>
              )}
            </div>
            {pendingChanges.length === 0 && (
              <span className="text-zinc-400">All changes processed</span>
            )}
          </div>
        </div>
      )}

      {/* Warning for risky changes */}
      {changes.some(c => c.file_path.includes('.env') || c.file_path.includes('config')) && (
        <div className="px-4 py-2 bg-yellow-900/20 border-t border-yellow-800/30 flex items-center space-x-2">
          <AlertCircle className="w-4 h-4 text-yellow-500" />
          <span className="text-xs text-yellow-400">
            Review configuration changes carefully
          </span>
        </div>
      )}
    </div>
  );
};