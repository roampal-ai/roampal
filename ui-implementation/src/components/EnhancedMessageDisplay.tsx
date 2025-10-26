import React, { useState } from 'react';
import { Copy, Check, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CodeBlock {
  language: string;
  code: string;
}

interface Citation {
  citation_id: number;
  source: string;
  confidence: number;
  collection: string;
  text?: string;
}

interface EnhancedMessageDisplayProps {
  content: string;
  codeBlocks?: CodeBlock[];
  citations?: Citation[];
}

export const EnhancedMessageDisplay: React.FC<EnhancedMessageDisplayProps> = ({
  content,
  codeBlocks,
  citations
}) => {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [showCitations, setShowCitations] = useState(false);

  const handleCopy = (code: string, index: number) => {
    navigator.clipboard.writeText(code);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // Custom renderer for code blocks
  const components = {
    code({ inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '');
      const language = match ? match[1] : '';
      const codeString = String(children).replace(/\n$/, '');

      if (!inline && language) {
        const codeIndex = codeBlocks?.findIndex(
          block => block.code.trim() === codeString.trim()
        );

        return (
          <div className="relative group my-3">
            <div className="absolute top-2 right-2 flex items-center gap-2">
              <span className="text-xs text-zinc-500 bg-zinc-800 px-2 py-1 rounded">
                {language}
              </span>
              <button
                onClick={() => handleCopy(codeString, codeIndex || 0)}
                className="p-1.5 bg-zinc-800 hover:bg-zinc-700 rounded transition-colors"
                title="Copy code"
              >
                {copiedIndex === codeIndex ? (
                  <Check className="w-4 h-4 text-green-500" />
                ) : (
                  <Copy className="w-4 h-4 text-zinc-400" />
                )}
              </button>
            </div>
            <SyntaxHighlighter
              style={oneDark}
              language={language}
              PreTag="div"
              className="rounded-lg !mt-0"
              {...props}
            >
              {codeString}
            </SyntaxHighlighter>
          </div>
        );
      }

      // Inline code
      return (
        <code className="px-1 py-0.5 bg-zinc-800 text-zinc-300 rounded text-sm" {...props}>
          {children}
        </code>
      );
    }
  };

  return (
    <div className="space-y-3">
      {/* Main content with markdown rendering */}
      <div className="prose prose-invert max-w-none break-words overflow-wrap-anywhere">
        <ReactMarkdown components={components}>
          {content}
        </ReactMarkdown>
      </div>

      {/* Citations Section (collapsible) */}
      {citations && citations.length > 0 && (
        <div className="border-t border-zinc-800 pt-3">
          <button
            onClick={() => setShowCitations(!showCitations)}
            className="flex items-center gap-2 text-sm text-zinc-400 hover:text-zinc-300 transition-colors"
          >
            {showCitations ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            {citations.length} Citation{citations.length !== 1 ? 's' : ''}
          </button>

          {showCitations && (
            <div className="mt-2 space-y-2">
              {citations.map((citation) => (
                <div
                  key={citation.citation_id}
                  className="px-3 py-2 bg-zinc-900/50 border border-zinc-800 rounded text-sm"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-zinc-300">[{citation.citation_id}] {citation.source}</span>
                    <span className="text-xs text-zinc-500">
                      {(citation.confidence * 100).toFixed(0)}% • {citation.collection}
                    </span>
                  </div>
                  {citation.text && (
                    <div className="text-xs text-zinc-500 mt-1">{citation.text}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

    </div>
  );
};