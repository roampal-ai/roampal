import React from 'react';
import { FileIcon, AlertCircleIcon, CodeIcon, DatabaseIcon } from 'lucide-react';

interface ContextInfo {
  current_file?: string | null;
  last_error?: string | null;
  has_code?: boolean;
  session_id?: string;
  memories_retrieved?: number;
}

interface ContextIndicatorProps {
  context?: ContextInfo;
  className?: string;
}

export const ContextIndicator: React.FC<ContextIndicatorProps> = ({
  context,
  className = ""
}) => {
  if (!context || Object.keys(context).length === 0) {
    return null;
  }

  return (
    <div className={`flex items-center gap-3 px-3 py-2 bg-gray-800 rounded-lg text-xs ${className}`}>
      {context.current_file && (
        <div className="flex items-center gap-1 text-blue-400">
          <FileIcon className="w-3 h-3" />
          <span>{context.current_file}</span>
        </div>
      )}

      {context.last_error && (
        <div className="flex items-center gap-1 text-red-400">
          <AlertCircleIcon className="w-3 h-3" />
          <span>Error present</span>
        </div>
      )}

      {context.has_code && (
        <div className="flex items-center gap-1 text-green-400">
          <CodeIcon className="w-3 h-3" />
          <span>Code ready</span>
        </div>
      )}

      {context.memories_retrieved !== undefined && context.memories_retrieved > 0 && (
        <div className="flex items-center gap-1 text-purple-400">
          <DatabaseIcon className="w-3 h-3" />
          <span>{context.memories_retrieved} memories</span>
        </div>
      )}
    </div>
  );
};