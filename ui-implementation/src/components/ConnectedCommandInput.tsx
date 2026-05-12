import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { PaperAirplaneIcon, PhotoIcon } from '@heroicons/react/24/outline';
import { useChatStore } from '../stores/useChatStore';
import logger from '../utils/logger';

interface ConnectedCommandInputProps {
  hasChatModel?: boolean;
  modelHasVision?: boolean;
}

/**
 * Connected version of CommandInput that uses the chat store
 */
const ConnectedCommandInputComponent: React.FC<ConnectedCommandInputProps> = ({ hasChatModel = true, modelHasVision = false }) => {
  const [message, setMessage] = useState('');
  const [showCommands, setShowCommands] = useState(false);
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);
  const [attachments, setAttachments] = useState<File[]>([]);  // v0.3.3 Section 4: image attachments
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const {
    sendMessage,
    searchMemory,
    isProcessing,
    processingStatus,
    cancelProcessing,
  } = useChatStore();
  
  const COMMANDS = [
    { trigger: '/memory', name: 'memory search [query]', description: 'Search memories', handler: handleMemorySearch },
    { trigger: '/memory', name: 'memory save [text]', description: 'Save memory', handler: handleMemorySave },
    { trigger: '/clear', name: 'clear', description: 'Clear conversation', handler: handleClear },
    { trigger: '/help', name: 'help', description: 'Show all commands', handler: handleHelp },
  ];
  
  // Auto-resize textarea (also on width change from sidebar toggle)
  const resizeTextarea = useCallback(() => {
    if (textareaRef.current) {
      // Reset height to auto to get the correct scrollHeight
      textareaRef.current.style.height = 'auto';
      // Set to scrollHeight for all cases (handles both newlines and text wrapping)
      const newHeight = Math.max(24, Math.min(textareaRef.current.scrollHeight, 208));
      textareaRef.current.style.height = `${newHeight}px`;
    }
  }, []);

  // Resize on message change
  useEffect(() => {
    resizeTextarea();
  }, [message, resizeTextarea]);

  // Resize on container width change (sidebar toggle)
  useEffect(() => {
    if (!textareaRef.current) return;
    const observer = new ResizeObserver(() => {
      resizeTextarea();
    });
    observer.observe(textareaRef.current);
    return () => observer.disconnect();
  }, [resizeTextarea]);

  // Set initial height on mount
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '24px';
    }
  }, []);
  
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
  
  // Command handlers
  async function handleMemorySearch(args: string[]) {
    const query = args.slice(1).join(' '); // Skip 'search' keyword
    if (query) {
      await searchMemory(query);
    }
  }
  
  async function handleMemorySave(args: string[]) {
    const text = args.slice(1).join(' '); // Skip 'save' keyword
    if (text) {
      // TODO: Implement memory save
      logger.debug('Memory save not yet implemented:', text);
    }
  }
  
  function handleClear() {
    useChatStore.getState().clearSession();
  }
  
  function handleHelp() {
    const helpText = COMMANDS.map(cmd => `${cmd.name} - ${cmd.description}`).join('\n');
    logger.info('Available commands:\n' + helpText);
  }

  // v0.3.3 Section 4: Image handling helpers
  const addImageFiles = useCallback((files: FileList | File[]) => {
    const imageFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
    if (imageFiles.length) {
      setAttachments(prev => [...prev, ...imageFiles]);
    }
  }, []);

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    if (!modelHasVision) return;

    const items = e.clipboardData.items;
    const imageFiles: File[] = [];
    for (const item of items) {
      if (item.kind === 'file' && item.type.startsWith('image/')) {
        const file = item.getAsFile();
        if (file) imageFiles.push(file);
      }
    }
    if (imageFiles.length) {
      e.preventDefault();
      setAttachments(prev => [...prev, ...imageFiles]);
    }
  }, [modelHasVision]);

  const removeAttachment = useCallback((index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  }, []);

    const handleSend = async () => {
    if ((!message.trim() && attachments.length === 0) || isProcessing) return;

    // Store message before clearing
    const messageToSend = message;

    // Clear input immediately
    setMessage('');
    const filesToSend = attachments;
    setAttachments([]);

    // Check for commands
    if (messageToSend.startsWith('/')) {
      const [cmd, ...args] = messageToSend.slice(1).split(' ');
      const command = COMMANDS.find(c => c.trigger === `/${cmd}`);
      if (command) {
        await command.handler(args);
        return;
      }
    }

    // Check for !memory commands
    if (messageToSend.includes('!memory')) {
      // Handle !memory commands
      const memoryMatch = messageToSend.match(/!memory\s+(\w+)\s*(.*)/);
      if (memoryMatch) {
        const [, action, content] = memoryMatch;
        if (action === 'recent') {
          await searchMemory('?recent=true');
        } else if (action === 'save' && content) {
          // TODO: Implement memory save
          logger.debug('Memory save not yet implemented:', content);
        } else if (action === 'search' && content) {
          await searchMemory(content);
        }
      }
      // Still send the message
      await sendMessage(messageToSend, filesToSend.length ? filesToSend : undefined);
    } else {
      // Regular message
      await sendMessage(messageToSend, filesToSend.length ? filesToSend : undefined);
    }
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
    } else if (e.key === 'Enter') {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        handleSend();
      } else if (!e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    }
  };
  
  return (
    <div className="relative">
      {/* Command palette */}
      {showCommands && (
        <div className="absolute bottom-full left-0 right-0 mb-2 p-2 bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl">
          <div className="space-y-1">
            {COMMANDS.map((cmd, index) => (
              <button
                key={`${cmd.trigger}-${index}`}
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
      <div className="p-3 bg-zinc-950 border border-zinc-800 focus-within:border-blue-500/50 rounded-2xl transition-all">
        {/* v0.3.3 Section 4: Image attachment previews */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2 pb-2 border-b border-zinc-800">
            {attachments.map((file, index) => (
              <div key={index} className="relative group">
                <img
                  src={URL.createObjectURL(file)}
                  alt="attachment"
                  className="w-16 h-16 rounded-lg object-cover ring-1 ring-zinc-700"
                />
                <button
                  onClick={() => removeAttachment(index)}
                  className="absolute -top-1.5 -right-1.5 w-5 h-5 bg-red-500 text-white rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity text-xs"
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
            onPaste={handlePaste}  // v0.3.3 Section 4: paste images
            onKeyDown={handleKeyDown}
            placeholder={!hasChatModel ? "Install a chat model to start" : isProcessing ? "Processing... (type your next message)" : "Ready when you are."}
            disabled={!hasChatModel}
            className="flex-1 bg-transparent text-sm text-zinc-100 placeholder-zinc-600 resize-none focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
            style={{ minHeight: '24px', maxHeight: '208px', overflowY: 'auto' }}
            rows={1}
          />

          {/* v0.3.3 Section 4: Photo icon button - gated by model vision capability */}
          {modelHasVision && (
            <>
              <input
                type="file"
                ref={fileInputRef}
                accept="image/*"
                multiple
                onChange={(e) => e.target.files && addImageFiles(e.target.files)}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                type="button"
                className="p-1.5 text-zinc-500 hover:text-zinc-300 transition-colors"
                title="Attach image (paste also supported)"
              >
                <PhotoIcon className="w-5 h-5" />
              </button>
            </>
          )}

          {/* Action buttons */}
          <div className="flex items-center gap-1">
            <button
              onClick={isProcessing ? cancelProcessing : handleSend}
              disabled={!isProcessing && !message.trim() && attachments.length === 0}
              className={`p-1.5 transition-all duration-200 ${
                isProcessing
                  ? 'text-red-500 hover:text-red-400 hover:scale-110'
                  : (message.trim() || attachments.length > 0)
                  ? 'text-blue-500 hover:text-blue-400 hover:scale-110 hover:drop-shadow-[0_0_8px_rgba(59,130,246,0.5)]'
                  : 'text-zinc-600 cursor-not-allowed'
              }`}
            >
              {isProcessing ? (
                <svg className="w-5 h-5 animate-spin" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" viewBox="0 0 24 24" stroke="currentColor">
                  <path d="M6 6L18 18M6 18L18 6" />
                </svg>
              ) : (
                <PaperAirplaneIcon className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
        
        {/* Helper text and transparency status */}
        <div className="mt-2">
          {/* Simplified - no tool or thinking displays */}

          <div className="flex items-center justify-between text-xs text-zinc-600">
            <span>
              {isProcessing ? 'New messages will be sent after current response' : 'Shift+Enter for new line'}
            </span>
            <span>{!isProcessing ? '⌘+Enter to send' : ''}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// Export memoized version to prevent unnecessary re-renders
export const ConnectedCommandInput = React.memo(ConnectedCommandInputComponent);