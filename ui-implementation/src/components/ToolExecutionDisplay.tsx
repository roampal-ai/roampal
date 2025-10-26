import React from 'react';
import { CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/solid';

interface ToolExecution {
  tool: string;
  status: 'running' | 'completed' | 'failed';
  description: string;
  detail?: string;
  metadata?: Record<string, any>;
}

interface ToolExecutionDisplayProps {
  executions: ToolExecution[];
}

export const ToolExecutionDisplay: React.FC<ToolExecutionDisplayProps> = ({ executions }) => {
  console.log('[ToolExecutionDisplay] Received executions:', executions);

  if (!executions || executions.length === 0) {
    console.log('[ToolExecutionDisplay] No executions, returning null');
    return null;
  }

  console.log('[ToolExecutionDisplay] Rendering', executions.length, 'executions');

  return (
    <div className="space-y-2">
      {executions.map((execution, index) => (
        <div
          key={index}
          className="flex items-start gap-2 px-3 py-2 bg-zinc-900/40 border border-zinc-700/50 rounded-lg"
        >
          {/* Status Icon */}
          <div className="flex-shrink-0 mt-0.5">
            {execution.status === 'running' && (
              <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            )}
            {execution.status === 'completed' && (
              <CheckCircleIcon className="w-4 h-4 text-green-500" />
            )}
            {execution.status === 'failed' && (
              <XCircleIcon className="w-4 h-4 text-red-500" />
            )}
          </div>

          {/* Content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-zinc-300">
                {execution.description}
              </span>
              {execution.status === 'running' && (
                <span className="text-xs text-blue-400">Running...</span>
              )}
            </div>
            {execution.detail && (
              <div className="text-xs text-zinc-500 mt-1">
                {execution.detail}
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};
