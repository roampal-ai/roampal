import React, { useState, useRef, useEffect } from 'react';
import { PaperClipIcon, MicrophoneIcon, PaperAirplaneIcon } from '@heroicons/react/24/outline';

interface CommandInputProps {
  onSend: (message: string, attachments?: File[]) => void;
  onCommand: (command: string, args: string[]) => void;
  onVoiceStart: () => void;
  onVoiceEnd: () => void;
  isProcessing: boolean;
  placeholder?: string;
}

interface Command {
  trigger: string;
  name: string;
  description: string;
}

const COMMANDS: Command[] = [
  { trigger: '/switch', name: 'switch [shard]', description: 'Switch active shard' },
  { trigger: '/memory', name: 'memory [query]', description: 'Search memories' },
  { trigger: '/clear', name: 'clear', description: 'Clear conversation' },
  { trigger: '/export', name: 'export', description: 'Export chat history' },
  { trigger: '/help', name: 'help', description: 'Show all commands' },
];

const MENTIONS = ['@roampal', '@dev', '@creative', '@analyst', '@coach'];

export const CommandInput: React.FC<CommandInputProps> = ({
  onSend,
  onCommand,
  onVoiceStart,
  onVoiceEnd,
  isProcessing,
  placeholder = 'Type a message or / for commands...',
}) => {
  const [message, setMessage] = useState('');
  const [attachments, setAttachments] = useState<File[]>([]);
  const [showCommands, setShowCommands] = useState(false);
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);
  
  // Command detection
  useEffect(() => {
    const firstChar = message[0];
    if (firstChar === '/' && message.length === 1) {
      setShowCommands(true);
      setSelectedCommandIndex(0);
    } else if (firstChar === '/') {
      const query = message.slice(1).toLowerCase();
      const filtered = COMMANDS.filter(cmd => 
        cmd.trigger.slice(1).startsWith(query)
      );
      setShowCommands(filtered.length > 0);
    } else {
      setShowCommands(false);
    }
  }, [message]);
  
  const handleSend = () => {
    if (!message.trim() || isProcessing) return;
    
    // Check for commands
    if (message.startsWith('/')) {
      const [cmd, ...args] = message.slice(1).split(' ');
      onCommand(cmd, args);
      setMessage('');
      return;
    }
    
    onSend(message, attachments);
    setMessage('');
    setAttachments([]);
  };
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (showCommands) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedCommandIndex(prev => 
          Math.min(prev + 1, COMMANDS.length - 1)
        );
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedCommandIndex(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'Tab' || e.key === 'Enter') {
        e.preventDefault();
        const selected = COMMANDS[selectedCommandIndex];
        setMessage(selected.trigger + ' ');
        setShowCommands(false);
      } else if (e.key === 'Escape') {
        setShowCommands(false);
      }
    } else if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };
  
  const handleFileSelect = (files: FileList | null) => {
    if (!files) return;
    setAttachments(prev => [...prev, ...Array.from(files)]);
  };
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files);
  };
  
  const removeAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };
  
  return (
    <div className="relative">
      {/* Command palette */}
      {showCommands && (
        <div className="absolute bottom-full left-0 right-0 mb-2 p-2 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl">
          <div className="space-y-1">
            {COMMANDS.map((cmd, index) => (
              <button
                key={cmd.trigger}
                onClick={() => {
                  setMessage(cmd.trigger + ' ');
                  setShowCommands(false);
                }}
                className={`w-full px-3 py-2 flex items-center justify-between rounded-md transition-colors ${
                  index === selectedCommandIndex
                    ? 'bg-zinc-800 text-zinc-100'
                    : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100'
                }`}
              >
                <span className="font-mono text-sm">{cmd.name}</span>
                <span className="text-xs text-zinc-500">{cmd.description}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      
      {/* Input container */}
      <div
        className={`min-h-[80px] p-4 bg-zinc-950 border rounded-2xl transition-all ${
          isDragging
            ? 'border-blue-500 bg-blue-500/5'
            : 'border-zinc-800 focus-within:border-blue-500/50'
        }`}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        {/* Attachments */}
        {attachments.length > 0 && (
          <div className="mb-2 flex flex-wrap gap-2">
            {attachments.map((file, index) => (
              <div
                key={index}
                className="px-2 py-1 bg-zinc-800 rounded-md flex items-center gap-1 group"
              >
                <span className="text-xs">📎</span>
                <span className="text-xs text-zinc-300">{file.name}</span>
                <button
                  onClick={() => removeAttachment(index)}
                  className="ml-1 text-zinc-500 hover:text-red-400 opacity-0 group-hover:opacity-100"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
        
        {/* Textarea and actions */}
        <div className="flex items-end gap-2">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isProcessing}
            className="flex-1 bg-transparent text-sm text-zinc-100 placeholder-zinc-600 resize-none focus:outline-none min-h-[24px] max-h-[200px]"
            rows={1}
          />
          
          {/* Action buttons */}
          <div className="flex items-center gap-1">
            <input
              ref={fileInputRef}
              type="file"
              multiple
              onChange={(e) => handleFileSelect(e.target.files)}
              className="hidden"
            />
            
            <button
              onClick={() => fileInputRef.current?.click()}
              className="p-1.5 text-zinc-500 hover:text-zinc-300 transition-colors"
              title="Attach files"
            >
              <PaperClipIcon className="w-5 h-5" />
            </button>
            
            <button
              onMouseDown={onVoiceStart}
              onMouseUp={onVoiceEnd}
              onMouseLeave={onVoiceEnd}
              className="p-1.5 text-zinc-500 hover:text-zinc-300 transition-colors"
              title="Hold for voice"
            >
              <MicrophoneIcon className="w-5 h-5" />
            </button>
            
            <button
              onClick={handleSend}
              disabled={!message.trim() || isProcessing}
              className={`p-1.5 transition-colors ${
                message.trim() && !isProcessing
                  ? 'text-blue-500 hover:text-blue-400'
                  : 'text-zinc-600 cursor-not-allowed'
              }`}
              title="Send message"
            >
              <PaperAirplaneIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
        
        {/* Helper text */}
        <div className="mt-2 flex items-center justify-between text-xs text-zinc-600">
          <span>
            {isProcessing ? 'Processing...' : 'Shift+Enter for new line'}
          </span>
          <span>⌘+Enter to send</span>
        </div>
      </div>
    </div>
  );
};