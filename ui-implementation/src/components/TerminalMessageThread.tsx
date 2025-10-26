import React, { useRef, useEffect, useMemo, useState } from 'react';
import { Terminal, CheckCircle, XCircle, AlertCircle, Loader2, Brain, Search, Wrench, MessageSquare, CheckCircle2, ChevronDown, ChevronRight, BookOpen, Sparkles, Database } from 'lucide-react';
import { MemoryCitation } from './MemoryCitation';
import { CodeChangePreview } from './CodeChangePreview';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

// Thinking Block Component - Terminal-style (inline, matches CitationsBlock)
const ThinkingBlock: React.FC<{ thinking: string }> = ({ thinking }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const lineCount = thinking.split('\n').length;

  return (
    <div className="mb-3 font-mono">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
      >
        <span className="text-zinc-600">{isExpanded ? '▼' : '▶'}</span>
        <span className="text-zinc-500">reasoning ({lineCount} {lineCount === 1 ? 'line' : 'lines'})</span>
      </button>

      {isExpanded && (
        <div className="mt-2 pl-4 border-l border-zinc-800/50 text-xs text-zinc-500 leading-relaxed whitespace-pre-wrap">
          {thinking}
        </div>
      )}
    </div>
  );
};

// Citations Block Component - Terminal-style
const CitationsBlock: React.FC<{ citations: any[] }> = ({ citations }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const collectionColors: Record<string, string> = {
    'books': 'text-purple-400',
    'working': 'text-blue-400',
    'history': 'text-green-400',
    'patterns': 'text-yellow-400',
    'memory_bank': 'text-pink-400'
  };

  return (
    <div className="mt-2 pt-2 border-t border-zinc-800/30 font-mono flex justify-end">
      <div className="text-right">
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
        >
          <span className="text-zinc-600">{isExpanded ? '▼' : '▶'}</span>
          <span className="text-zinc-500">
            [{citations.length}] {citations.length === 1 ? 'reference' : 'references'}
          </span>
        </button>

        {isExpanded && (
          <div className="mt-2 space-y-1.5 pl-4 border-l border-zinc-800/50 text-left">
            {citations.map((citation: any, idx: number) => (
              <div key={idx} className="text-xs text-zinc-500 leading-relaxed">
                <div className="flex items-start gap-2">
                  <span className="text-zinc-600 select-none">[{citation.citation_id || idx + 1}]</span>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className={collectionColors[citation.collection] || 'text-zinc-400'}>
                        {citation.collection}
                      </span>
                    </div>
                    {citation.text && (
                      <div className="text-zinc-600 text-[11px] leading-relaxed pl-4 border-l border-zinc-800/30">
                        {citation.text}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

interface Message {
  id: string;
  sender: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  status?: 'sent' | 'sending' | 'error';
  streaming?: boolean;
  shard?: string;
  attachments?: any[];
  memories?: any[];
  citations?: any[];
  thinking?: string; // LLM reasoning from <think> tags
  toolExecutions?: Array<{
    tool: string;
    status: 'running' | 'completed' | 'failed';
    description: string;
    detail?: string;
    metadata?: Record<string, any>;
  }>;
  code_changes?: any[]; // For future code change previews
  events?: Array<{
    type: 'thinking' | 'tool_execution' | 'text';
    timestamp: number;
    data: any;
  }>; // Chronological event timeline
}

interface TerminalMessageThreadProps {
  messages: Message[];
  activeShard: string;
  onMemoryClick?: (memoryId: string) => void;
  onCommandClick?: (command: string) => void;
  isProcessing?: boolean;
  processingStage?: string;
  processingStatus?: string | null;
}

const TerminalMessageThreadComponent: React.FC<TerminalMessageThreadProps> = ({
  messages,
  activeShard,
  onMemoryClick,
  onCommandClick,
  isProcessing,
  processingStage,
  processingStatus
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Memoize processing message to prevent recalculation
  const getProcessingMessage = useMemo(() => () => {
    // Always return a message, even if processingStatus is null or empty string
    if (processingStatus && processingStatus.trim()) {
      return processingStatus;
    }

    // Find the last user message
    const lastUserMessage = messages.filter(m => m.sender === 'user').pop();
    if (!lastUserMessage) return 'Processing...';

    const text = lastUserMessage.content.toLowerCase();

    // Analyze intent and return appropriate message
    if (text.includes('find') || text.includes('search') || text.includes('look for') || text.includes('where')) {
      if (text.includes('file')) return 'Finding files...';
      if (text.includes('function') || text.includes('method')) return 'Searching for functions...';
      if (text.includes('error') || text.includes('bug')) return 'Searching for errors...';
      return 'Searching...';
    }

    if (text.includes('edit') || text.includes('change') || text.includes('modify') || text.includes('update')) {
      if (text.includes('code')) return 'Editing code...';
      if (text.includes('file')) return 'Modifying file...';
      return 'Making changes...';
    }

    if (text.includes('create') || text.includes('make') || text.includes('new') || text.includes('add')) {
      if (text.includes('file')) return 'Creating file...';
      if (text.includes('function')) return 'Creating function...';
      if (text.includes('component')) return 'Creating component...';
      return 'Creating...';
    }

    if (text.includes('fix') || text.includes('repair') || text.includes('solve')) {
      return 'Fixing issue...';
    }

    if (text.includes('delete') || text.includes('remove')) {
      return 'Removing...';
    }

    if (text.includes('run') || text.includes('execute')) {
      if (text.includes('test')) return 'Running tests...';
      if (text.includes('build')) return 'Building...';
      return 'Executing...';
    }

    if (text.includes('explain') || text.includes('what') || text.includes('how')) {
      return 'Analyzing...';
    }

    if (text.includes('list') || text.includes('show')) {
      return 'Gathering information...';
    }

    if (text.includes('install')) {
      return 'Installing...';
    }

    if (text.includes('debug')) {
      return 'Debugging...';
    }

    if (text.includes('refactor')) {
      return 'Refactoring code...';
    }

    if (text.includes('optimize') || text.includes('improve')) {
      return 'Optimizing...';
    }

    if (text.includes('test') && !text.includes('run')) {
      if (text.includes('write')) return 'Writing tests...';
      return 'Testing...';
    }

    if (text.includes('review') || text.includes('check')) {
      return 'Reviewing...';
    }

    if (text.includes('analyze') || text.includes('understand')) {
      return 'Analyzing code...';
    }

    if (text.includes('document')) {
      return 'Documenting...';
    }

    if (text.includes('connect') || text.includes('integrate')) {
      return 'Connecting...';
    }

    if (text.includes('deploy') || text.includes('publish')) {
      return 'Deploying...';
    }

    if (text.includes('configure') || text.includes('setup')) {
      return 'Configuring...';
    }

    if (text.includes('import') || text.includes('export')) {
      return 'Processing data...';
    }

    if (text.includes('compile') || text.includes('transpile')) {
      return 'Compiling...';
    }

    if (text.includes('validate')) {
      return 'Validating...';
    }

    if (text.includes('migrate')) {
      return 'Migrating...';
    }

    // Default fallback
    return 'Processing...';
  }, [messages, processingStatus, processingStage]);

  // Only scroll when messages length changes (new message added)
  const prevMessageCount = useRef(messages.length);
  useEffect(() => {
    if (messages.length > prevMessageCount.current) {
      scrollToBottom();
    }
    prevMessageCount.current = messages.length;
  }, [messages.length]);

  const formatTime = (date: Date) => {
    if (!(date instanceof Date)) {
      date = new Date(date);
    }
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'sending':
        return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />;
      case 'sent':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />;
      default:
        return null;
    }
  };

  const renderContent = (content: string) => {
    // Custom callout syntax parser (:::success, :::warning, :::info)
    const processCallouts = (text: string) => {
      return text.replace(/:::(\w+)\n([\s\S]*?):::/g, (_, type, content) => {
        return `<div class="callout callout-${type}">${content.trim()}</div>`;
      });
    };

    const processedContent = processCallouts(content);

    return (
      <div className="markdown-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeRaw]}
          components={{
          // Headings
          h1: ({node, ...props}) => <h1 className="text-lg font-bold text-zinc-200 mt-4 mb-2" {...props} />,
          h2: ({node, ...props}) => <h2 className="text-base font-bold text-zinc-300 mt-3 mb-2" {...props} />,
          h3: ({node, ...props}) => <h3 className="text-sm font-semibold text-zinc-300 mt-2 mb-1" {...props} />,

          // Text formatting
          strong: ({node, ...props}) => <strong className="font-bold text-zinc-100" {...props} />,
          em: ({node, ...props}) => <em className="italic text-zinc-300" {...props} />,

          // Code
          code: ({node, inline, className, children, ...props}: any) => {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';

            if (!inline && language) {
              return (
                <div className="my-3">
                  <div className="bg-zinc-900/50 border border-zinc-800 rounded-md overflow-hidden">
                    <div className="flex justify-between items-center px-3 py-2 border-b border-zinc-800 bg-zinc-900/30">
                      <span className="text-xs text-zinc-600">{language}</span>
                      <button
                        onClick={() => navigator.clipboard.writeText(String(children))}
                        className="text-xs text-zinc-600 hover:text-zinc-400 transition-colors"
                      >
                        Copy
                      </button>
                    </div>
                    <pre className="text-xs overflow-x-auto p-3">
                      <code className="text-zinc-300">{children}</code>
                    </pre>
                  </div>
                </div>
              );
            }
            return <code className="px-1 py-0.5 bg-zinc-800/50 text-zinc-300 rounded text-xs font-mono" {...props}>{children}</code>;
          },

          // Lists
          ul: ({node, ...props}) => <ul className="list-disc list-inside text-zinc-300 space-y-1 my-2" {...props} />,
          ol: ({node, ...props}) => <ol className="list-decimal list-inside text-zinc-300 space-y-1 my-2" {...props} />,
          li: ({node, ...props}) => <li className="text-sm" {...props} />,

          // Blockquotes (styled as callouts)
          blockquote: ({node, ...props}) => (
            <div className="my-2 pl-3 py-2 border-l-2 border-blue-500/50 bg-blue-900/10 text-zinc-300 text-sm italic">
              {props.children}
            </div>
          ),

          // Custom callout divs
          div: ({node, className, ...props}: any) => {
            if (className?.includes('callout-success')) {
              return <div className="my-2 px-3 py-2 bg-green-900/20 border border-green-700/50 rounded text-green-300 text-sm" {...props} />;
            }
            if (className?.includes('callout-warning')) {
              return <div className="my-2 px-3 py-2 bg-yellow-900/20 border border-yellow-700/50 rounded text-yellow-300 text-sm" {...props} />;
            }
            if (className?.includes('callout-info')) {
              return <div className="my-2 px-3 py-2 bg-blue-900/20 border border-blue-700/50 rounded text-blue-300 text-sm" {...props} />;
            }
            return <div {...props} />;
          },

          // Paragraphs
          p: ({node, ...props}) => <p className="text-zinc-300 text-sm my-1 whitespace-pre-wrap" {...props} />,
        }}
      >
        {processedContent}
      </ReactMarkdown>
      </div>
    );
  };

  return (
    <div className="bg-zinc-950 p-6" style={{ fontFamily: 'SF Mono, Monaco, Inconsolata, Fira Code, monospace' }}>
      {/* Welcome message when no messages */}
      {messages.length === 0 && (
        <div className="mb-8 text-zinc-400 text-sm">
          <div className="flex items-center gap-2 mb-1">
            <Terminal className="w-4 h-4" />
            <span>Welcome to the Roampal Terminal</span>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="space-y-4">
        {messages.map((message) => (
          <div key={message.id} className="group">
            {message.sender === 'user' && (
              <div className="flex items-start gap-3">
                <span className="text-zinc-600 text-sm mt-0.5">&gt;</span>
                <div className="flex-1">
                  <div className="text-zinc-400 text-sm leading-relaxed">{message.content}</div>
                  {message.attachments && message.attachments.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {message.attachments.map((att: any, idx: number) => (
                        <span key={idx} className="inline-flex items-center gap-1 px-2 py-1 bg-zinc-800 rounded text-xs text-zinc-400">
                          📎 {att.name} {att.size ? `(${Math.ceil(att.size / 1024)}KB)` : ''}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}

            {message.sender === 'assistant' && (
              <div className="flex items-start gap-3">
                <div className="flex-1 min-w-0">
                  {/* Render events in chronological order if timeline exists */}
                  {message.events && message.events.length > 0 ? (
                    <>
                      {message.events.map((event, idx) => {
                        if (event.type === 'tool_execution') {
                          const tool = event.data;
                          const isRunning = tool.status === 'running';
                          const resultCount = tool.metadata?.result_count || tool.metadata?.fragmentCount;
                          return (
                            <div key={idx} className="text-xs font-mono mb-3">
                              <div className={`text-zinc-500 ${isRunning ? 'animate-pulse-subtle' : ''}`}>
                                {isRunning ? '⋯ ' : '✓ '}
                                {tool.description || (tool.tool === 'search_memory' ? 'Searching memory' : tool.tool)}
                                {!isRunning && resultCount !== undefined ? ` · ${resultCount} results` : ''}
                              </div>
                            </div>
                          );
                        } else if (event.type === 'text' && event.data.firstChunk && message.content) {
                          // Text started - render the actual content here in timeline
                          return (
                            <div key={idx} className="text-zinc-100 text-sm leading-relaxed mb-6">
                              {renderContent(message.content)}
                            </div>
                          );
                        }
                        return null;
                      })}
                    </>
                  ) : (
                    /* Fallback to old static order if no timeline */
                    <>
                      {/* 1. TOOL EXECUTION */}
                      {message.toolExecutions && message.toolExecutions.length > 0 && (
                        <div className="text-xs font-mono mb-3">
                          {message.toolExecutions.map((tool, idx) => {
                            const isRunning = tool.status === 'running';
                            const resultCount = tool.metadata?.result_count || tool.metadata?.fragmentCount;

                            return (
                              <div key={idx} className={`text-zinc-500 ${isRunning ? 'animate-pulse-subtle' : ''}`}>
                                {isRunning ? '⋯ ' : '✓ '}
                                {tool.description || (tool.tool === 'search_memory' ? 'Searching memory' : tool.tool)}
                                {!isRunning && resultCount !== undefined ? ` · ${resultCount} results` : ''}
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {/* 1.5. THINKING BLOCK - LLM reasoning */}
                      {message.thinking && <ThinkingBlock thinking={message.thinking} />}

                      {/* 2. RESPONSE CONTENT */}
                      {message.content && (
                        <div className={`text-zinc-100 text-sm leading-relaxed ${message.citations && message.citations.length > 0 ? 'mb-0' : 'mb-2'}`}>
                          {renderContent(message.content)}
                        </div>
                      )}
                    </>
                  )}

                  {/* 4. CITATIONS - Source references (only when complete) */}
                  {!message.streaming && message.citations && message.citations.length > 0 && (
                    <CitationsBlock citations={message.citations} />
                  )}

                  {/* Show code changes if present - keeping for future use */}
                  {message.code_changes && message.code_changes.length > 0 && (
                    <CodeChangePreview
                      changes={message.code_changes}
                      onApply={(changeId) => {
                        // TODO: Implement code change application
                      }}
                      onSkip={(changeId) => {
                        // TODO: Implement code change skipping
                      }}
                      onApplyAll={() => {
                        // TODO: Implement apply all
                      }}
                    />
                  )}

                  {/* Show memories if present (legacy) */}
                  {message.memories && message.memories.length > 0 && !message.citations && (
                    <div className="mt-2 text-xs text-zinc-600">
                      Using {message.memories.length} relevant memories
                    </div>
                  )}
                </div>
              </div>
            )}

            {message.sender === 'system' && (
              <div className="flex items-start gap-3">
                <span className="text-amber-600 text-sm mt-0.5">!</span>
                <span className="text-amber-500 text-sm">{message.content}</span>
              </div>
            )}
          </div>
        ))}

        {/* Global processing indicator - only show when no assistant message is currently streaming */}
        {((isProcessing || processingStatus) && !messages.some(m => m.sender === 'assistant' && m.streaming)) && (
            <div className="flex items-start gap-3">
              <div className="flex-1">
                <div className="text-sm text-zinc-500 flex items-center gap-2">
                  {(() => {
                    const msg = getProcessingMessage();
                    if (msg.includes('Searching') || msg.includes('Finding')) {
                      return <Search className="w-4 h-4 animate-pulse" />;
                    } else if (msg.includes('Analyzing') || msg.includes('Thinking')) {
                      return <Sparkles className="w-4 h-4 animate-pulse" />;
                    } else if (msg.includes('documentation')) {
                      return <BookOpen className="w-4 h-4 animate-pulse" />;
                    } else {
                      return <Loader2 className="w-4 h-4 animate-spin" />;
                    }
                  })()}
                  <span>{getProcessingMessage()}</span>
                </div>
              </div>
            </div>
        )}
      </div>

      <div ref={messagesEndRef} />
    </div>
  );
};

// Export memoized version to prevent unnecessary re-renders
export const TerminalMessageThread = React.memo(TerminalMessageThreadComponent);
