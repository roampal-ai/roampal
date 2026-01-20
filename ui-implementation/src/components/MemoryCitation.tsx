import React, { useState, useRef, useEffect } from 'react';
import {
  BookOpenIcon,
  CpuChipIcon,
  ClockIcon,
  BoltIcon,
  InformationCircleIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  CircleStackIcon
} from '@heroicons/react/24/outline';

const Database = CircleStackIcon;

const InfoIcon = InformationCircleIcon;

interface Citation {
  citation_id: number;
  source: string;
  confidence: number;
  collection: 'books' | 'working' | 'history' | 'patterns';
  text?: string;
  timestamp?: string;
}

interface MemoryCitationProps {
  message: string;
  citations: Citation[];
}

const collectionIcons: Record<string, React.ReactNode> = {
  'books': <BookOpenIcon className="w-3 h-3" />,
  'working': <CpuChipIcon className="w-3 h-3" />,
  'history': <ClockIcon className="w-3 h-3" />,
  'patterns': <BoltIcon className="w-3 h-3" />
};

const collectionColors: Record<string, string> = {
  'books': 'text-purple-400',
  'working': 'text-blue-400',
  'history': 'text-green-400',
  'patterns': 'text-yellow-400'
};

const confidenceColors = (confidence: number): string => {
  if (confidence >= 0.9) return 'text-green-500';
  if (confidence >= 0.7) return 'text-yellow-500';
  return 'text-orange-500';
};

export const MemoryCitation: React.FC<MemoryCitationProps> = ({ message, citations }) => {
  const [expandedCitations, setExpandedCitations] = useState(false);
  const [hoveredCitation, setHoveredCitation] = useState<number | null>(null);
  const citationRefs = useRef<{ [key: number]: HTMLElement | null }>({});

  // SECURITY: Parse message to add citation superscripts using React elements (not dangerouslySetInnerHTML)
  const parsedMessageElements = React.useMemo(() => {
    if (citations.length === 0) return [message];

    // Build a regex to match all citation patterns [n]
    const citationIds = citations.map(c => c.citation_id);
    const pattern = new RegExp(`(\\[(?:${citationIds.join('|')})\\])`, 'g');

    // Split message by citation patterns, keeping delimiters
    const parts = message.split(pattern);

    return parts.map((part, index) => {
      // Check if this part is a citation reference like [1], [2], etc.
      const match = part.match(/^\[(\d+)\]$/);
      if (match) {
        const citationId = parseInt(match[1]);
        if (citationIds.includes(citationId)) {
          return (
            <sup
              key={`citation-${index}`}
              className="citation-link text-blue-400 cursor-pointer hover:text-blue-300"
              data-citation-id={citationId}
              onMouseEnter={() => setHoveredCitation(citationId)}
              onMouseLeave={() => setHoveredCitation(null)}
            >
              [{citationId}]
            </sup>
          );
        }
      }
      // Regular text - render safely as text node
      return <span key={`text-${index}`}>{part}</span>;
    });
  }, [message, citations]);

  // SECURITY: Removed useEffect document event listeners - hover now handled directly on React elements

  if (citations.length === 0) {
    return <div className="text-zinc-300 whitespace-pre-wrap">{message}</div>;
  }

  return (
    <div className="space-y-2">
      {/* Message with inline citations - SECURITY: Using React elements instead of dangerouslySetInnerHTML */}
      <div className="text-zinc-300 whitespace-pre-wrap">
        {parsedMessageElements}
      </div>

      {/* Hover tooltip */}
      {hoveredCitation !== null && (
        <div className="fixed z-50 max-w-sm p-3 bg-zinc-800 border border-zinc-700 rounded-lg shadow-xl">
          {citations
            .filter(c => c.citation_id === hoveredCitation)
            .map(citation => (
              <div key={citation.citation_id} className="space-y-2">
                <div className="flex items-center space-x-2">
                  <span className={collectionColors[citation.collection]}>
                    {collectionIcons[citation.collection]}
                  </span>
                  <span className="text-sm font-medium text-zinc-300">
                    {citation.source}
                  </span>
                </div>
                <div className="flex items-center space-x-3 text-xs">
                  <span className={confidenceColors(citation.confidence)}>
                    {(citation.confidence * 100).toFixed(0)}% confidence
                  </span>
                  <span className="text-zinc-500">
                    {citation.collection}
                  </span>
                </div>
                {citation.text && (
                  <div className="text-xs text-zinc-400 italic">
                    "{citation.text}"
                  </div>
                )}
              </div>
            ))}
        </div>
      )}

      {/* Citations list */}
      <div className="mt-3 border-t border-zinc-800 pt-2">
        <button
          onClick={() => setExpandedCitations(!expandedCitations)}
          className="flex items-center space-x-2 text-sm text-zinc-500 hover:text-zinc-400 transition-colors"
        >
          <Database className="w-4 h-4" />
          <span>Used {citations.length} memor{citations.length > 1 ? 'ies' : 'y'}</span>
          {expandedCitations ? (
            <ChevronUpIcon className="w-4 h-4" />
          ) : (
            <ChevronDownIcon className="w-4 h-4" />
          )}
        </button>

        {expandedCitations && (
          <div className="mt-2 space-y-2 pl-6">
            {citations.map((citation) => (
              <div
                key={citation.citation_id}
                ref={el => citationRefs.current[citation.citation_id] = el}
                className="flex items-start space-x-2 p-2 bg-zinc-900/50 rounded"
              >
                <span className="text-xs font-mono text-zinc-600">
                  [{citation.citation_id}]
                </span>
                <div className="flex-1 space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className={collectionColors[citation.collection]}>
                      {collectionIcons[citation.collection]}
                    </span>
                    <span className="text-sm text-zinc-300">
                      {citation.source}
                    </span>
                  </div>
                  <div className="flex items-center space-x-3 text-xs">
                    <span className={confidenceColors(citation.confidence)}>
                      {(citation.confidence * 100).toFixed(0)}% confidence
                    </span>
                    <span className="text-zinc-500">
                      Collection: {citation.collection}
                    </span>
                  </div>
                  {citation.text && (
                    <div className="text-xs text-zinc-400 italic mt-1">
                      "{citation.text.substring(0, 200)}
                      {citation.text.length > 200 && '...'}
                      "
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

    </div>
  );
};