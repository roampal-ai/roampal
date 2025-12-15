import React, { useRef, useEffect, useMemo, useState, memo, useCallback } from 'react';
import { Terminal, CheckCircle, XCircle, AlertCircle, Loader2, Search, Wrench, MessageSquare, CheckCircle2, ChevronDown, ChevronRight, BookOpen, Database } from 'lucide-react';
import { MemoryCitation } from './MemoryCitation';
import { CodeChangePreview } from './CodeChangePreview';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';

// v0.2.8: Memoized markdown rendering component to prevent expensive re-parses
// ReactMarkdown builds AST and creates React elements on every render - memoizing prevents this
const MemoizedMarkdown = memo(({ content }: { content: string }) => {
  // Strip thinking tags as safety net
  const stripThinkingTags = (text: string) => {
    let cleaned = text.replace(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/gi, '');
    cleaned = cleaned.replace(/<think(?:ing)?>([\s\S]*?)(?=\n\n|$)/gi, '');
    return cleaned.trim();
  };

  const processCallouts = (text: string) => {
    return text.replace(/:::(\w+)\n([\s\S]*?):::/g, (_, type, calloutContent) => {
      return `<div class="callout callout-${type}">${calloutContent.trim()}</div>`;
    });
  };

  const processedContent = processCallouts(stripThinkingTags(content));

  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{
          h1: ({node, ...props}) => <h1 className="text-lg font-bold text-zinc-200 mt-4 mb-2" {...props} />,
          h2: ({node, ...props}) => <h2 className="text-base font-bold text-zinc-300 mt-3 mb-2" {...props} />,
          h3: ({node, ...props}) => <h3 className="text-sm font-semibold text-zinc-300 mt-2 mb-1" {...props} />,
          strong: ({node, ...props}) => <strong className="font-bold text-zinc-100" {...props} />,
          em: ({node, ...props}) => <em className="italic text-zinc-300" {...props} />,
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
          ul: ({node, ...props}) => <ul className="list-disc list-inside text-zinc-300 space-y-1 my-2" {...props} />,
          ol: ({node, ...props}) => <ol className="list-decimal list-inside text-zinc-300 space-y-1 my-2" {...props} />,
          li: ({node, ...props}) => <li className="text-sm" {...props} />,
          blockquote: ({node, ...props}) => (
            <div className="my-2 pl-3 py-2 border-l-2 border-blue-500/50 bg-blue-900/10 text-zinc-300 text-sm italic">
              {props.children}
            </div>
          ),
          a: ({node, ...props}) => (
            <a className="text-blue-400 hover:text-blue-300 hover:underline" {...props} />
          ),
          p: ({node, ...props}) => <p className="text-sm leading-relaxed mb-2" {...props} />,
          table: ({node, ...props}) => (
            <div className="overflow-x-auto my-2">
              <table className="min-w-full border border-zinc-700" {...props} />
            </div>
          ),
          th: ({node, ...props}) => <th className="px-3 py-2 bg-zinc-800 border border-zinc-700 text-xs font-semibold text-left" {...props} />,
          td: ({node, ...props}) => <td className="px-3 py-2 border border-zinc-700 text-xs" {...props} />,
          hr: () => <hr className="my-4 border-zinc-700" />,
          div: ({node, className, ...props}: any) => {
            if (className?.startsWith('callout')) {
              const type = className.replace('callout callout-', '');
              const colors: Record<string, string> = {
                success: 'border-green-500 bg-green-900/20',
                warning: 'border-yellow-500 bg-yellow-900/20',
                info: 'border-blue-500 bg-blue-900/20',
                error: 'border-red-500 bg-red-900/20'
              };
              return (
                <div className={`my-2 p-3 border-l-2 rounded-r ${colors[type] || colors.info}`} {...props} />
              );
            }
            return <div {...props} />;
          }
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  );
}, (prevProps, nextProps) => prevProps.content === nextProps.content);

// Animated Thinking Dots Component - cycles through "Thinking." -> "Thinking.." -> "Thinking..."
const ThinkingDots: React.FC = () => {
  const [dots, setDots] = useState(1);

  useEffect(() => {
    const interval = setInterval(() => {
      setDots(d => d >= 3 ? 1 : d + 1);
    }, 400);
    return () => clearInterval(interval);
  }, []);

  return (
    <span className="text-blue-400 font-mono">
      Thinking{'.'.repeat(dots)}
    </span>
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
    <div className="mt-2 pt-2 border-t border-zinc-800/30 font-mono">
      <div>
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
    resultCount?: number;
    arguments?: Record<string, any>;
    resultPreview?: string;
  }>;
  code_changes?: any[]; // For future code change previews
  events?: Array<{
    type: 'thinking' | 'tool_execution' | 'text' | 'text_segment';
    timestamp: number;
    data: any;
  }>; // Chronological event timeline
  _lastTextEndIndex?: number; // v0.2.5: Internal tracking for text segment boundaries
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

  // v0.2.5: Only show real status from backend, no fallback "Thinking..." text
  const getProcessingMessage = useMemo(() => () => {
    return processingStatus?.trim() || '';
  }, [processingStatus]);

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

  // v0.2.8: Use memoized markdown component for performance
  // The actual rendering logic is now in MemoizedMarkdown at the top of the file
  const renderContent = useCallback((content: string) => {
    return <MemoizedMarkdown content={content} />;
  }, []);

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
                      {/* v0.2.5: Check if we have text_segment events for true interleaving */}
                      {(() => {
                        const hasTextSegments = message.events.some((e: any) => e.type === 'text_segment');
                        const hasTools = message.events.some((e: any) => e.type === 'tool_execution');

                        // Calculate live streaming text (after last segment boundary)
                        let liveStreamingText = '';
                        if (message.streaming && hasTools && message.content) {
                          const lastEndIndex = message._lastTextEndIndex || 0;
                          liveStreamingText = message.content.slice(lastEndIndex);
                        }

                        return (
                          <>
                            {message.events.map((event: any, idx: number) => {
                              if (event.type === 'tool_execution') {
                                const tool = event.data;
                                // v0.2.5: If message is no longer streaming, tool must be completed
                                const isRunning = message.streaming && tool.status === 'running';
                                const resultCount = tool.metadata?.result_count || tool.metadata?.fragmentCount;
                                return (
                                  <div key={idx} className="text-xs font-mono mb-3">
                                    <div className={`text-zinc-500 ${isRunning ? 'animate-pulse-subtle' : ''}`}>
                                      {isRunning ? '⋯ ' : '✓ '}
                                      {tool.description || (tool.tool === 'search_memory' ? 'Searching memory' : tool.tool)}
                                      {!isRunning && resultCount !== undefined && (
                                        <>
                                          {` · ${resultCount} results`}
                                          {tool.resultPreview && (
                                            <span className="text-zinc-600"> · {tool.resultPreview}</span>
                                          )}
                                        </>
                                      )}
                                    </div>
                                  </div>
                                );
                              } else if (event.type === 'text_segment') {
                                // v0.2.5: Render this specific text segment (true chronological interleaving)
                                return (
                                  <div key={idx} className="text-zinc-100 text-sm leading-relaxed mb-2">
                                    {renderContent(event.data.content)}
                                  </div>
                                );
                              } else if (event.type === 'text' && event.data.firstChunk && message.content && !hasTextSegments && !hasTools) {
                                // DEPRECATED: Fallback for old messages without text_segment events AND no tools
                                // Only render full content if no segments exist (backwards compat)
                                return (
                                  <div key={idx} className="text-zinc-100 text-sm leading-relaxed mb-6">
                                    {renderContent(message.content)}
                                  </div>
                                );
                              }
                              return null;
                            })}
                            {/* v0.2.5: Show live streaming text (accumulating after last tool) */}
                            {liveStreamingText && (
                              <div className="text-zinc-100 text-sm leading-relaxed mb-2">
                                {renderContent(liveStreamingText)}
                              </div>
                            )}
                          </>
                        );
                      })()}
                    </>
                  ) : (
                    /* Fallback to old static order if no timeline */
                    <>
                      {/* 1. TOOL EXECUTION */}
                      {message.toolExecutions && message.toolExecutions.length > 0 && (
                        <div className="text-xs font-mono mb-3">
                          {message.toolExecutions.map((tool, idx) => {
                            // v0.2.5: If message is no longer streaming, tool must be completed
                            const isRunning = message.streaming && tool.status === 'running';
                            // v0.2.4: Fix - store saves resultCount directly, not in metadata
                            const resultCount = tool.resultCount ?? tool.metadata?.result_count ?? tool.metadata?.fragmentCount;
                            const query = tool.arguments?.query || tool.metadata?.query;

                            // Build description with query if available (v0.2.4)
                            let description = tool.description;
                            if (!description) {
                              if (tool.tool === 'search_memory' && query) {
                                description = `searching "${query.length > 40 ? query.substring(0, 40) + '...' : query}"`;
                              } else {
                                description = tool.tool;
                              }
                            }

                            return (
                              <div key={idx} className={`text-zinc-500 ${isRunning ? 'animate-pulse-subtle' : ''}`}>
                                {isRunning ? '⋯ ' : '✓ '}
                                {description}
                                {!isRunning && resultCount !== undefined && (
                                  <>
                                    {` · ${resultCount} results`}
                                    {tool.resultPreview && (
                                      <span className="text-zinc-600"> · {tool.resultPreview}</span>
                                    )}
                                  </>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}

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

        {/* Global processing indicator - "Thinking..." while processing */}
        {/* Hide when tools are actively running (those show inline ⋯ indicators) */}
        {(() => {
          // Check if any tool is currently running in any message
          const hasRunningTool = messages.some(msg =>
            msg.toolExecutions?.some(tool => tool.status === 'running') ||
            msg.events?.some(e => e.type === 'tool_execution' && e.data?.status === 'running')
          );

          // Only show "Thinking..." if processing AND no tools are running
          return isProcessing && !hasRunningTool && (
            <div className="flex items-start gap-3">
              <div className="flex-1">
                <div className="text-sm text-zinc-500 flex items-center gap-2">
                  <ThinkingDots />
                </div>
              </div>
            </div>
          );
        })()}
      </div>

      <div ref={messagesEndRef} />
    </div>
  );
};

// Export memoized version to prevent unnecessary re-renders
export const TerminalMessageThread = React.memo(TerminalMessageThreadComponent);
