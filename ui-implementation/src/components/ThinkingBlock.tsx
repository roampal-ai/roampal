import React from 'react';

interface ThinkingBlockProps {
  thinking?: string;
  content?: string;
  isStreaming?: boolean;
}

export const ThinkingBlock: React.FC<ThinkingBlockProps> = ({ thinking, content, isStreaming }) => {
  const displayContent = thinking || content;
  if (!displayContent) return null;

  return (
    <div className="thinking-block" style={{
      padding: '8px 12px',
      margin: '4px 0',
      backgroundColor: '#f0f9ff',
      borderLeft: '3px solid #3b82f6',
      borderRadius: '4px',
      fontSize: '0.9em',
      color: '#1e40af'
    }}>
      <div style={{ fontWeight: 600, marginBottom: '4px' }}>
        💭 Thinking{isStreaming ? '...' : ''}
      </div>
      <div style={{ whiteSpace: 'pre-wrap' }}>{displayContent}</div>
    </div>
  );
};

export default ThinkingBlock;
