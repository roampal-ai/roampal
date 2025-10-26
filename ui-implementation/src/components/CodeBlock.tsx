import React, { useState } from 'react';
import { CopyIcon, CheckIcon, PlayIcon, FileIcon } from 'lucide-react';

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  onApplyToFile?: (filename: string, code: string) => void;
  onRun?: (code: string, language: string) => void;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code,
  language = 'plaintext',
  filename,
  onApplyToFile,
  onRun
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleRun = () => {
    if (onRun) {
      onRun(code, language);
    }
  };

  const handleApply = () => {
    if (onApplyToFile && filename) {
      onApplyToFile(filename, code);
    }
  };

  return (
    <div className="relative group bg-gray-900 rounded-lg overflow-hidden my-2">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">{language}</span>
          {filename && (
            <>
              <span className="text-gray-600">•</span>
              <span className="text-xs text-gray-400">{filename}</span>
            </>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex gap-1">
          {onRun && ['python', 'javascript', 'js', 'py'].includes(language.toLowerCase()) && (
            <button
              onClick={handleRun}
              className="p-1.5 rounded hover:bg-gray-700 transition-colors"
              title="Run code"
            >
              <PlayIcon className="w-4 h-4 text-green-400" />
            </button>
          )}

          {onApplyToFile && filename && (
            <button
              onClick={handleApply}
              className="p-1.5 rounded hover:bg-gray-700 transition-colors"
              title={`Apply to ${filename}`}
            >
              <FileIcon className="w-4 h-4 text-blue-400" />
            </button>
          )}

          <button
            onClick={handleCopy}
            className="p-1.5 rounded hover:bg-gray-700 transition-colors"
            title="Copy code"
          >
            {copied ? (
              <CheckIcon className="w-4 h-4 text-green-400" />
            ) : (
              <CopyIcon className="w-4 h-4 text-gray-400" />
            )}
          </button>
        </div>
      </div>

      {/* Code content */}
      <pre className="p-4 overflow-x-auto">
        <code className={`language-${language} text-sm text-gray-300`}>
          {code}
        </code>
      </pre>
    </div>
  );
};