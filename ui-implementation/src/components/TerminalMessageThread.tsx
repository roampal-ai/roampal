import React, { useRef, useEffect, useState, memo, useCallback } from 'react';
import { Terminal } from 'lucide-react';
import { useVirtualizer } from '@tanstack/react-virtual';
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
          h1: ({ node, ...props }) => <h1 className="text-lg font-bold text-zinc-200 mt-4 mb-2" {...props} />,
          h2: ({ node, ...props }) => <h2 className="text-base font-bold text-zinc-300 mt-3 mb-2" {...props} />,
          h3: ({ node, ...props }) => <h3 className="text-sm font-semibold text-zinc-300 mt-2 mb-1" {...props} />,
          strong: ({ node, ...props }) => <strong className="font-bold text-zinc-100" {...props} />,
          em: ({ node, ...props }) => <em className="italic text-zinc-300" {...props} />,
          code: ({ node, inline, className, children, ...props }: any) => {
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
          ul: ({ node, ...props }) => <ul className="list-disc list-inside text-zinc-300 space-y-1 my-2" {...props} />,
          ol: ({ node, ...props }) => <ol className="list-decimal list-inside text-zinc-300 space-y-1 my-2" {...props} />,
          li: ({ node, ...props }) => <li className="text-sm" {...props} />,
          blockquote: ({ node, ...props }) => (
            <div className="my-2 pl-3 py-2 border-l-2 border-blue-500/50 bg-blue-900/10 text-zinc-300 text-sm italic">
              {props.children}
            </div>
          ),
          a: ({ node, ...props }) => (
            <a className="text-blue-400 hover:text-blue-300 hover:underline" {...props} />
          ),
          p: ({ node, ...props }) => <p className="text-sm leading-relaxed mb-2" {...props} />,
          table: ({ node, ...props }) => (
            <div className="overflow-x-auto my-2">
              <table className="min-w-full border border-zinc-700" {...props} />
            </div>
          ),
          th: ({ node, ...props }) => <th className="px-3 py-2 bg-zinc-800 border border-zinc-700 text-xs font-semibold text-left" {...props} />,
          td: ({ node, ...props }) => <td className="px-3 py-2 border border-zinc-700 text-xs" {...props} />,
          hr: () => <hr className="my-4 border-zinc-700" />,
          div: ({ node, className, ...props }: any) => {
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

// v0.3.0: Surfaced Memories Block - shows what context was injected
interface SurfacedMemoriesBlockProps {
  memories: Array<{ id: string; collection: string; text: string; score?: number }>;
  isExpanded: boolean;
  onToggle: () => void;
}
const SurfacedMemoriesBlock: React.FC<SurfacedMemoriesBlockProps> = ({ memories, isExpanded, onToggle }) => {
  const [expandedItems, setExpandedItems] = React.useState<Set<number>>(new Set());
  const collectionColors: Record<string, string> = {
    'books': 'text-purple-400',
    'working': 'text-blue-400',
    'history': 'text-green-400',
    'patterns': 'text-yellow-400',
    'memory_bank': 'text-pink-400'
  };

  const PREVIEW_LENGTH = 80;

  const toggleItem = (idx: number) => {
    setExpandedItems(prev => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  return (
    <div className="mt-2 pt-2 border-t border-zinc-800/30 font-mono">
      <button
        onClick={onToggle}
        className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
      >
        <span className="text-zinc-600">{isExpanded ? '▼' : '▶'}</span>
        <span className="text-zinc-500">
          context: {memories.length} {memories.length === 1 ? 'memory' : 'memories'}
        </span>
      </button>

      {isExpanded && (
        <div className="mt-2 space-y-1.5 pl-4 border-l border-zinc-800/50 text-left">
          {memories.map((mem, idx) => {
            const isItemExpanded = expandedItems.has(idx);
            const needsTruncation = mem.text && mem.text.length > PREVIEW_LENGTH;
            const displayText = isItemExpanded || !needsTruncation
              ? mem.text
              : mem.text.slice(0, PREVIEW_LENGTH) + '...';

            return (
              <div key={idx} className="text-xs text-zinc-500 leading-relaxed">
                <div className="flex items-start gap-2">
                  <span className="text-zinc-600 select-none">[{idx + 1}]</span>
                  <div className="flex-1">
                    <span className={collectionColors[mem.collection] || 'text-zinc-400'}>
                      {mem.collection}
                    </span>
                    {mem.text && (
                      <div
                        className={`text-zinc-600 text-[11px] leading-relaxed mt-0.5 ${needsTruncation ? 'cursor-pointer hover:text-zinc-500' : ''}`}
                        onClick={needsTruncation ? () => toggleItem(idx) : undefined}
                        title={needsTruncation ? (isItemExpanded ? 'Click to collapse' : 'Click to expand') : undefined}
                      >
                        {displayText}
                        {needsTruncation && !isItemExpanded && (
                          <span className="text-zinc-700 ml-1">[+]</span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// Citations Block Component - Terminal-style
interface CitationsBlockProps {
  citations: any[];
  isExpanded: boolean;
  onToggle: () => void;
}
const CitationsBlock: React.FC<CitationsBlockProps> = ({ citations, isExpanded, onToggle }) => {
  const [expandedItems, setExpandedItems] = React.useState<Set<number>>(new Set());
  const collectionColors: Record<string, string> = {
    'books': 'text-purple-400',
    'working': 'text-blue-400',
    'history': 'text-green-400',
    'patterns': 'text-yellow-400',
    'memory_bank': 'text-pink-400'
  };

  const PREVIEW_LENGTH = 80;

  const toggleItem = (idx: number) => {
    setExpandedItems(prev => {
      const next = new Set(prev);
      if (next.has(idx)) {
        next.delete(idx);
      } else {
        next.add(idx);
      }
      return next;
    });
  };

  return (
    <div className="mt-2 pt-2 border-t border-zinc-800/30 font-mono">
      <div>
        <button
          onClick={onToggle}
          className="flex items-center gap-2 text-xs text-zinc-500 hover:text-zinc-400 transition-colors"
        >
          <span className="text-zinc-600">{isExpanded ? '▼' : '▶'}</span>
          <span className="text-zinc-500">
            [{citations.length}] {citations.length === 1 ? 'reference' : 'references'}
          </span>
        </button>

        {isExpanded && (
          <div className="mt-2 space-y-1.5 pl-4 border-l border-zinc-800/50 text-left">
            {citations.map((citation: any, idx: number) => {
              const isItemExpanded = expandedItems.has(idx);
              const text = citation.text || '';
              const needsTruncation = text.length > PREVIEW_LENGTH;
              const displayText = isItemExpanded || !needsTruncation
                ? text
                : text.slice(0, PREVIEW_LENGTH) + '...';

              return (
                <div key={idx} className="text-xs text-zinc-500 leading-relaxed">
                  <div className="flex items-start gap-2">
                    <span className="text-zinc-600 select-none">[{citation.citation_id || idx + 1}]</span>
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className={collectionColors[citation.collection] || 'text-zinc-400'}>
                          {citation.collection}
                        </span>
                      </div>
                      {text && (
                        <div
                          className={`text-zinc-600 text-[11px] leading-relaxed pl-4 border-l border-zinc-800/30 ${needsTruncation ? 'cursor-pointer hover:text-zinc-500' : ''}`}
                          onClick={needsTruncation ? () => toggleItem(idx) : undefined}
                          title={needsTruncation ? (isItemExpanded ? 'Click to collapse' : 'Click to expand') : undefined}
                        >
                          {displayText}
                          {needsTruncation && !isItemExpanded && (
                            <span className="text-zinc-700 ml-1">[+]</span>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
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
  surfacedMemories?: Array<{
    id: string;
    collection: string;
    text: string;
    score?: number;
  }>;
  thinking?: string;
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
  code_changes?: any[];
  events?: Array<{
    type: 'thinking' | 'tool_execution' | 'text' | 'text_segment';
    timestamp: number;
    data: any;
  }>;
  _lastTextEndIndex?: number;
}

// v0.3.0: Simplified MessageRow - no longer needs virtualization-specific props
// TanStack Virtual handles measurement automatically via measureElement ref
interface MessageRowProps {
  message: Message | 'loading-indicator';
  renderContent: (content: string) => React.ReactNode;
  isProcessing: boolean;
  isMemoryExpanded: boolean;
  isCitationExpanded: boolean;
  onToggleMemory: () => void;
  onToggleCitation: () => void;
}

const MessageRow = memo(({
  message,
  renderContent,
  isProcessing,
  isMemoryExpanded,
  isCitationExpanded,
  onToggleMemory,
  onToggleCitation,
}: MessageRowProps) => {
  // v0.3.0: Don't render loading-indicator at all when not processing
  if (message === 'loading-indicator') {
    if (!isProcessing) {
      return null;
    }
    return (
      <div className="pb-4">
        <div className="flex items-start gap-3">
          <div className="flex-1">
            <div className="text-sm text-zinc-500 flex items-center gap-2">
              <ThinkingDots />
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="pb-4">
      <div className="group">
        {message.sender === 'user' && (
          <div className="flex items-start gap-3 border-l-2 border-blue-500/40 pl-3">
            <span className="text-blue-400/70 text-sm mt-0.5">&gt;</span>
            <div className="flex-1">
              <div className="text-zinc-300 text-sm leading-relaxed">{message.content}</div>
              {message.attachments && message.attachments.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-2">
                  {message.attachments.map((att: any, idx: number) => (
                    <span key={idx} className="inline-flex items-center gap-1 px-2 py-1 bg-zinc-800 rounded text-xs text-zinc-400">
                      {att.name} {att.size ? `(${Math.ceil(att.size / 1024)}KB)` : ''}
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
              {/* v0.3.0: Show thinking dots when streaming but no content yet */}
              {message.streaming && !message.content && (!message.events || message.events.length === 0) && (!message.toolExecutions || message.toolExecutions.length === 0) && (
                <div className="text-sm text-zinc-500 flex items-center gap-2">
                  <ThinkingDots />
                </div>
              )}
              {message.events && message.events.length > 0 ? (
                <>
                  {(() => {
                    const hasTools = message.events.some((e: any) => e.type === 'tool_execution');
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
                          } else if (event.type === 'text_segment' || event.type === 'text') {
                            // Handle both text_segment (captured before/after tools) and text (initial token)
                            const content = event.data.content || event.data.chunk || '';
                            if (!content) return null;
                            return (
                              <div key={idx} className="text-zinc-100 text-sm leading-relaxed mb-2">
                                {renderContent(content)}
                              </div>
                            );
                          }
                          return null;
                        })}
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
                <>
                  {/* Legacy Rendering */}
                  {message.toolExecutions && message.toolExecutions.length > 0 && (
                    <div className="text-xs font-mono mb-3">
                      {message.toolExecutions.map((tool, idx) => {
                        const isRunning = message.streaming && tool.status === 'running';
                        const resultCount = tool.resultCount ?? tool.metadata?.result_count ?? tool.metadata?.fragmentCount;
                        const query = tool.arguments?.query || tool.metadata?.query;
                        let description = tool.description;
                        if (!description && tool.tool === 'search_memory' && query) {
                          description = `searching "${query.length > 40 ? query.substring(0, 40) + '...' : query}"`;
                        } else if (!description) {
                          description = tool.tool;
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
                  {message.content && (
                    <div className={`text-zinc-100 text-sm leading-relaxed ${message.citations && message.citations.length > 0 ? 'mb-0' : 'mb-2'}`}>
                      {renderContent(message.content)}
                    </div>
                  )}
                </>
              )}

              {/* v0.3.0: Show surfaced memories (context injection) */}
              {!message.streaming && message.surfacedMemories && message.surfacedMemories.length > 0 && (
                <SurfacedMemoriesBlock
                  memories={message.surfacedMemories}
                  isExpanded={isMemoryExpanded}
                  onToggle={onToggleMemory}
                />
              )}

              {/* Show citations from search_memory tool calls */}
              {!message.streaming && message.citations && message.citations.length > 0 && (
                <CitationsBlock
                  citations={message.citations}
                  isExpanded={isCitationExpanded}
                  onToggle={onToggleCitation}
                />
              )}

              {message.code_changes && message.code_changes.length > 0 && (
                <CodeChangePreview changes={message.code_changes} onApply={() => { }} onSkip={() => { }} onApplyAll={() => { }} />
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
    </div>
  );
}, (prev, next) => {
  if (prev.message === 'loading-indicator' || next.message === 'loading-indicator') {
    return prev.message === next.message && prev.isProcessing === next.isProcessing;
  }
  return (
    prev.message.content === next.message.content &&
    prev.message.streaming === next.message.streaming &&
    prev.message.events?.length === next.message.events?.length &&
    prev.message.citations?.length === next.message.citations?.length &&
    prev.message.surfacedMemories?.length === next.message.surfacedMemories?.length &&
    prev.isProcessing === next.isProcessing &&
    prev.isMemoryExpanded === next.isMemoryExpanded &&
    prev.isCitationExpanded === next.isCitationExpanded
  );
});

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
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevWidthRef = useRef<number>(0);

  // v0.3.0: Lift expanded state out of child components
  const [expandedMemories, setExpandedMemories] = useState<Set<string>>(new Set());
  const [expandedCitations, setExpandedCitations] = useState<Set<string>>(new Set());

  const toggleMemoryExpanded = useCallback((messageId: string) => {
    setExpandedMemories(prev => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }, []);

  const toggleCitationExpanded = useCallback((messageId: string) => {
    setExpandedCitations(prev => {
      const next = new Set(prev);
      if (next.has(messageId)) {
        next.delete(messageId);
      } else {
        next.add(messageId);
      }
      return next;
    });
  }, []);

  // v0.3.0: Only show loading indicator when processing AND no streaming message exists
  const lastMessage = messages[messages.length - 1];
  const hasStreamingMessage = lastMessage && lastMessage.streaming;
  const showLoadingIndicator = isProcessing && !hasStreamingMessage;
  const itemCount = showLoadingIndicator ? messages.length + 1 : messages.length;

  // v0.3.0: TanStack Virtual - the fix for width-change jank
  const virtualizer = useVirtualizer({
    count: itemCount,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 80,
    overscan: 5,
  });

  // v0.3.0: On width change, preserve scroll position
  // Don't call measure() - it nukes all cached heights causing 1-frame overlap
  // Let measureElement's ResizeObserver handle height updates naturally
  useEffect(() => {
    if (!scrollRef.current) return;
    const resizeObserver = new ResizeObserver(entries => {
      for (let entry of entries) {
        const newWidth = entry.contentRect.width;
        const widthDelta = Math.abs(newWidth - prevWidthRef.current);

        if (prevWidthRef.current > 0 && widthDelta > 10) {
          // Just preserve scroll position - don't call measure()
          const visibleItems = virtualizer.getVirtualItems();
          const firstVisibleIndex = visibleItems[0]?.index ?? 0;
          requestAnimationFrame(() => {
            virtualizer.scrollToIndex(firstVisibleIndex, { align: 'start' });
          });
        }
        prevWidthRef.current = newWidth;
      }
    });
    resizeObserver.observe(scrollRef.current);
    return () => {
      resizeObserver.disconnect();
    };
  }, [virtualizer]);

  // Auto-scroll to bottom when new messages arrive
  const prevMessagesLength = useRef(messages.length);
  useEffect(() => {
    const len = messages.length;
    if (len > prevMessagesLength.current) {
      // New message added - scroll to bottom
      virtualizer.scrollToIndex(itemCount - 1, { align: 'end' });
    }
    prevMessagesLength.current = len;
  }, [messages.length, itemCount, virtualizer]);

  // Also scroll when processing starts (to show loading indicator)
  useEffect(() => {
    if (showLoadingIndicator) {
      virtualizer.scrollToIndex(itemCount - 1, { align: 'end' });
    }
  }, [showLoadingIndicator, itemCount, virtualizer]);

  const renderContent = useCallback((content: string) => {
    return <MemoizedMarkdown content={content} />;
  }, []);

  return (
    <div className="h-full w-full bg-zinc-950 pl-6 pt-6 pb-6 overflow-hidden" style={{ fontFamily: 'SF Mono, Monaco, Inconsolata, Fira Code, monospace' }}>
      {messages.length === 0 && !isProcessing && (
        <div className="text-zinc-400 text-sm">
          <div className="flex items-center gap-2 mb-1">
            <Terminal className="w-4 h-4" />
            <span>Welcome to the Roampal Terminal</span>
          </div>
        </div>
      )}

      {/* v0.3.0: TanStack Virtual scroll container */}
      <div
        ref={scrollRef}
        className="h-full overflow-y-auto pr-2"
      >
        <div
          style={{
            height: virtualizer.getTotalSize(),
            width: '100%',
            position: 'relative',
          }}
        >
          {virtualizer.getVirtualItems().map(virtualRow => {
            const index = virtualRow.index;
            const isLoader = index === messages.length;
            const message = isLoader ? 'loading-indicator' : messages[index];
            const messageId = isLoader ? '' : (message as Message).id;

            return (
              <div
                key={virtualRow.key}
                data-index={index}
                ref={virtualizer.measureElement}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  transform: `translateY(${virtualRow.start}px)`,
                }}
              >
                <MessageRow
                  message={message}
                  renderContent={renderContent}
                  isProcessing={!!isProcessing}
                  isMemoryExpanded={expandedMemories.has(messageId)}
                  isCitationExpanded={expandedCitations.has(messageId)}
                  onToggleMemory={() => toggleMemoryExpanded(messageId)}
                  onToggleCitation={() => toggleCitationExpanded(messageId)}
                />
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export const TerminalMessageThread = React.memo(TerminalMessageThreadComponent);