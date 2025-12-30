import React, { useState, useCallback, useEffect, useRef } from 'react';
import { X, Upload, FileText, BookOpen, AlertCircle, CheckCircle, Loader2, Trash2, XCircle } from 'lucide-react';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface BookProcessorModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ProcessingFile {
  id: string;
  file: File;
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  message?: string;
  error?: string;
  taskId?: string;
  book_id?: string;
  total_chunks?: number;
  customTitle?: string;  // User-provided title (overrides filename)
  customAuthor?: string; // User-provided author (defaults to "Unknown")
}

interface WebSocketMessage {
  task_id: string;
  status: string;
  stage?: string;
  progress: number;
  message?: string;
  error?: string;
  total_chunks?: number;
  current_chunk?: number;
}

// Get the base URL from environment or default to localhost
const API_BASE_URL = ROAMPAL_CONFIG.apiUrl;
const WS_BASE_URL = API_BASE_URL.replace('http://', 'ws://').replace('https://', 'wss://');

interface ExistingBook {
  book_id: string;
  title: string;
  author?: string;
  upload_timestamp?: string;
  processing_stats?: {
    total_chunks?: number;
    chunks_processed?: number;
  };
}

export const BookProcessorModal: React.FC<BookProcessorModalProps> = ({
  isOpen,
  onClose
}) => {
  const [files, setFiles] = useState<ProcessingFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [existingBooks, setExistingBooks] = useState<ExistingBook[]>([]);
  const [showExisting, setShowExisting] = useState(false);
  const [loadingBooks, setLoadingBooks] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<{bookId: string, bookTitle: string} | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const wsConnections = useRef<Map<string, WebSocket>>(new Map());
  const processingTimeouts = useRef<Map<string, number>>(new Map());

  // Cleanup WebSocket connections and timeouts on unmount
  useEffect(() => {
    return () => {
      wsConnections.current.forEach(ws => ws.close());
      wsConnections.current.clear();
      processingTimeouts.current.forEach(timeout => clearTimeout(timeout));
      processingTimeouts.current.clear();
    };
  }, []);

  // Load existing books when modal opens
  useEffect(() => {
    if (isOpen && showExisting) {
      loadExistingBooks();
    }
  }, [isOpen, showExisting]);

  const loadExistingBooks = async () => {
    setLoadingBooks(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/book-upload/books`);
      if (response.ok) {
        const data = await response.json();
        setExistingBooks(data.books || []);
      } else {
        console.error('Failed to load books');
      }
    } catch (error) {
      console.error('Error loading books:', error);
    } finally {
      setLoadingBooks(false);
    }
  };

  const deleteExistingBook = async () => {
    if (!deleteConfirm || isDeleting) return;
    const { bookId, bookTitle } = deleteConfirm;

    setIsDeleting(true);
    setDeleteError(null);

    // 30 second timeout to prevent infinite hang
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
      const response = await fetch(`${API_BASE_URL}/api/book-upload/books/${bookId}`, {
        method: 'DELETE',
        signal: controller.signal
      });
      clearTimeout(timeoutId);

      if (response.ok) {
        // Remove from list
        setExistingBooks(prev => prev.filter(b => b.book_id !== bookId));
        setDeleteConfirm(null);
        setDeleteError(null);
        console.log(`Successfully deleted book ${bookTitle}`);

        // Notify memory panel to refresh (book removed from books collection)
        window.dispatchEvent(new CustomEvent('memoryUpdated', {
          detail: { source: 'book_delete', timestamp: new Date().toISOString() }
        }));
      } else {
        const errorText = await response.text();
        let errorMessage = `Delete failed (${response.status})`;
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.detail || errorJson.message || errorMessage;
        } catch {
          if (errorText) errorMessage = errorText;
        }
        setDeleteError(errorMessage);
      }
    } catch (error) {
      clearTimeout(timeoutId);
      console.error('Error deleting book:', error);
      if (error instanceof Error && error.name === 'AbortError') {
        // Timeout - book likely deleted but response never arrived
        setDeleteError('Request timed out. The book may have been deleted - try refreshing.');
      } else {
        setDeleteError(error instanceof Error ? error.message : 'Network error - could not reach server');
      }
    } finally {
      setIsDeleting(false);
    }
  };

  const closeDeleteModal = () => {
    setDeleteConfirm(null);
    setDeleteError(null);
    setIsDeleting(false);
  };

  const connectWebSocket = (taskId: string, fileId: string, retryCount = 0) => {
    const wsUrl = `${WS_BASE_URL}/api/book-upload/ws/progress/${taskId}`;
    const ws = new WebSocket(wsUrl);
    let reconnectTimeout: number;

    ws.onopen = () => {
      console.log(`WebSocket connected for task ${taskId}`);

      // Start processing timeout (5 minutes max)
      const timeout = window.setTimeout(() => {
        console.warn(`Processing timeout for task ${taskId}`);
        ws.close();
        setFiles(prev => prev.map(f =>
          f.id === fileId
            ? {
                ...f,
                status: 'error',
                error: 'Processing timeout (5 minutes)',
                message: 'Processing took too long'
              }
            : f
        ));
      }, 5 * 60 * 1000); // 5 minutes

      processingTimeouts.current.set(fileId, timeout);
    };

    ws.onmessage = (event) => {
      const data: WebSocketMessage = JSON.parse(event.data);

      setFiles(prev => prev.map(f => {
        if (f.id === fileId) {
          let status: ProcessingFile['status'] = 'processing';
          let message = data.message || 'Processing...';

          if (data.status === 'completed' || data.status === 'complete') {
            status = 'completed';
            message = 'Processing complete!';

            // Clear processing timeout
            const timeout = processingTimeouts.current.get(fileId);
            if (timeout) {
              clearTimeout(timeout);
              processingTimeouts.current.delete(fileId);
            }

            // Notify memory panel to refresh (new book added to books collection)
            window.dispatchEvent(new CustomEvent('memoryUpdated', {
              detail: { source: 'book_upload', timestamp: new Date().toISOString() }
            }));

            // Auto-clear completed files after a short delay and switch to library
            setTimeout(() => {
              setFiles(current => current.filter(file => file.id !== fileId));
              setShowExisting(true); // Switch to library view
              loadExistingBooks(); // Refresh library
            }, 2000);
          } else if (data.status === 'failed' || data.status === 'error') {
            status = 'error';
            message = data.error || 'Processing failed';
          } else if (data.status === 'cancelled') {
            status = 'error';
            message = 'Processing cancelled';
          }

          return {
            ...f,
            status,
            progress: data.progress || f.progress,
            message,
            error: data.error,
            book_id: f.book_id, // Preserve the book_id
            total_chunks: data.total_chunks || f.total_chunks
          };
        }
        return f;
      }));
    };

    ws.onerror = (error) => {
      console.error(`WebSocket error for task ${taskId}:`, error);
      // Don't immediately set to error - let onclose handle reconnection
      // Only log the error here
    };

    ws.onclose = (event) => {
      console.log(`WebSocket closed for task ${taskId}`);
      wsConnections.current.delete(taskId);

      // Attempt reconnection if not a normal closure and we haven't exceeded retry limit
      if (event.code !== 1000 && retryCount < 3) {
        const delay = Math.min(1000 * Math.pow(2, retryCount), 5000); // Exponential backoff
        console.log(`Attempting to reconnect WebSocket in ${delay}ms (attempt ${retryCount + 1}/3)`);

        reconnectTimeout = window.setTimeout(() => {
          // Check if the file is still in processing state
          setFiles(prev => {
            const file = prev.find(f => f.id === fileId);
            if (file && (file.status === 'processing' || file.status === 'uploading')) {
              connectWebSocket(taskId, fileId, retryCount + 1);
            }
            return prev;
          });
        }, delay);
      }
    };

    // Store cleanup function with the WebSocket
    // (ws as any).reconnectTimeout = reconnectTimeout;

    wsConnections.current.set(taskId, ws);
  };

  const handleFileSelect = (selectedFiles: FileList | null) => {
    if (!selectedFiles || selectedFiles.length === 0) {
      console.log('No files selected');
      return;
    }

    console.log(`Processing ${selectedFiles.length} file(s)`);

    const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

    const newFiles: ProcessingFile[] = Array.from(selectedFiles).map(file => {
      console.log(`Adding file: ${file.name} (${file.size} bytes)`);

      // Check file size
      if (file.size > MAX_FILE_SIZE) {
        return {
          id: Math.random().toString(36).substr(2, 9),
          file,
          status: 'error' as const,
          progress: 0,
          message: 'File too large',
          error: `File exceeds 10MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB)`,
          customTitle: file.name.replace(/\.[^/.]+$/, ''),
          customAuthor: ''
        };
      }

      // Check file type - v0.2.3: expanded to support PDF, DOCX, Excel, CSV, HTML, RTF
      const allowedExtensions = ['.txt', '.md', '.pdf', '.docx', '.xlsx', '.xls', '.csv', '.tsv', '.html', '.htm', '.rtf'];
      const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
      if (!allowedExtensions.includes(extension)) {
        return {
          id: Math.random().toString(36).substr(2, 9),
          file,
          status: 'error' as const,
          progress: 0,
          message: 'Invalid file type',
          error: `Unsupported file type. Allowed: ${allowedExtensions.join(', ')}`,
          customTitle: file.name.replace(/\.[^/.]+$/, ''),
          customAuthor: ''
        };
      }

      return {
        id: Math.random().toString(36).substr(2, 9),
        file,
        status: 'pending' as const,
        progress: 0,
        message: 'Ready to upload',
        customTitle: file.name.replace(/\.[^/.]+$/, ''),  // Default to filename without extension
        customAuthor: ''  // Default empty (will use "Unknown" if not filled)
      };
    });

    setFiles(prev => [...prev, ...newFiles]);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const dt = e.dataTransfer;
    const files = dt.files;

    console.log('Files dropped:', files.length);
    handleFileSelect(files);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();

    // Only set dragging to false if we're leaving the drop zone entirely
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX;
    const y = e.clientY;

    if (x < rect.left || x >= rect.right || y < rect.top || y >= rect.bottom) {
      setIsDragging(false);
    }
  }, []);

  const processFile = async (file: ProcessingFile) => {
    console.log(`Starting to process file: ${file.file.name}`);

    // Update status to uploading
    setFiles(prev => prev.map(f =>
      f.id === file.id
        ? { ...f, status: 'uploading', message: 'Uploading file...', progress: 5 }
        : f
    ));

    try {
      // Validate title is not empty
      const title = file.customTitle?.trim() || file.file.name.replace(/\.[^/.]+$/, '');
      if (!title) {
        setFiles(prev => prev.map(f =>
          f.id === file.id
            ? { ...f, status: 'error', error: 'Title cannot be empty', message: 'Please provide a title' }
            : f
        ));
        return;
      }

      const author = file.customAuthor?.trim() || 'Unknown';

      const formData = new FormData();
      formData.append('file', file.file);
      formData.append('title', title);
      formData.append('author', author);

      console.log(`Uploading to: ${API_BASE_URL}/api/book-upload/upload with title="${title}", author="${author}"`);

      // Upload file
      const uploadResponse = await fetch(`${API_BASE_URL}/api/book-upload/upload`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - let browser set it with boundary for multipart
      });

      console.log(`Upload response status: ${uploadResponse.status}`);

      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text();
        console.error(`Upload failed: ${errorText}`);

        let errorMessage = `Upload failed: ${uploadResponse.statusText}`;
        try {
          const errorJson = JSON.parse(errorText);
          errorMessage = errorJson.detail || errorJson.message || errorMessage;
        } catch {
          // If not JSON, use the text
          if (errorText) {
            errorMessage = errorText;
          }
        }
        throw new Error(errorMessage);
      }

      const uploadData = await uploadResponse.json();
      console.log('Upload response data:', uploadData);

      // Check if upload was rejected as duplicate
      if (!uploadData.success && uploadData.processing_status === 'duplicate') {
        console.log('Book rejected as duplicate:', uploadData.message);
        setFiles(prev => prev.map(f =>
          f.id === file.id
            ? {
                ...f,
                status: 'error',
                error: uploadData.message || 'This book already exists in the library',
                message: 'Duplicate book detected',
                progress: 0
              }
            : f
        ));
        return; // Skip to next file
      }

      if (uploadData.task_id) {
        console.log(`Got task ID: ${uploadData.task_id}, connecting WebSocket...`);

        // Connect WebSocket for progress tracking
        connectWebSocket(uploadData.task_id, file.id);

        // Update file with task info and book_id
        setFiles(prev => prev.map(f =>
          f.id === file.id
            ? {
                ...f,
                status: 'processing',
                taskId: uploadData.task_id,
                book_id: uploadData.book_id,
                message: 'Processing book...',
                progress: 10
              }
            : f
        ));
      } else {
        console.error('No task ID in response:', uploadData);
        throw new Error(uploadData.error || uploadData.message || 'No task ID received from server');
      }
    } catch (error) {
      console.error('Error processing file:', error);
      setFiles(prev => prev.map(f =>
        f.id === file.id
          ? {
              ...f,
              status: 'error',
              progress: 0,
              error: error instanceof Error ? error.message : 'Unknown error',
              message: error instanceof Error ? error.message : 'Upload failed'
            }
          : f
      ));
    }
  };

  const startProcessing = async () => {
    setIsProcessing(true);
    const pendingFiles = files.filter(f => f.status === 'pending');

    // Process files sequentially
    for (const file of pendingFiles) {
      await processFile(file);
      // Small delay between files
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    setIsProcessing(false);
  };

  const cancelProcessing = async (fileId: string) => {
    const file = files.find(f => f.id === fileId);
    if (!file || !file.taskId) return;

    try {
      console.log(`Cancelling task ${file.taskId}`);
      const response = await fetch(`${API_BASE_URL}/api/book-upload/cancel/${file.taskId}`, {
        method: 'POST'
      });

      if (response.ok) {
        console.log('Task cancelled successfully');

        // Close WebSocket connection
        const ws = wsConnections.current.get(file.taskId);
        if (ws) {
          ws.close();
          wsConnections.current.delete(file.taskId);
        }

        // Clear timeout
        const timeout = processingTimeouts.current.get(fileId);
        if (timeout) {
          clearTimeout(timeout);
          processingTimeouts.current.delete(fileId);
        }

        // Update UI
        setFiles(prev => prev.map(f =>
          f.id === fileId
            ? { ...f, status: 'error', error: 'Cancelled by user', message: 'Processing cancelled', progress: 0 }
            : f
        ));
      } else {
        console.error('Failed to cancel task');
      }
    } catch (error) {
      console.error('Error cancelling task:', error);
    }
  };

  const removeFile = async (fileId: string) => {
    const file = files.find(f => f.id === fileId);

    if (file && file.status === 'completed' && file.book_id) {
      // Confirm deletion for processed books
      const confirmDelete = window.confirm(
        `Are you sure you want to delete "${file.file.name}" from your library?\n\nThis will remove all processed data including chunks, quotes, models, and summaries.`
      );

      if (!confirmDelete) {
        return;
      }

      // If the file was processed and has a book_id, delete from backend
      try {
        console.log(`Deleting book ${file.book_id}`);
        const response = await fetch(`${API_BASE_URL}/api/book-upload/books/${file.book_id}`, {
          method: 'DELETE'
        });

        if (!response.ok) {
          console.error('Failed to delete book from backend');
        } else {
          console.log('Book deleted successfully from backend');
        }
      } catch (error) {
        console.error('Error deleting book:', error);
        // Show error to user
        alert(`Failed to delete book: ${error instanceof Error ? error.message : 'Unknown error'}`);
        return; // Don't remove from UI if deletion failed
      }
    } else if (file && file.status === 'completed') {
      console.log('Cannot delete: Book ID not available (file was processed before delete feature was added)');
    } else if (file && file.status === 'error') {
      // For failed uploads, just remove from UI
      console.log('Removing failed upload from list');
    }

    // Remove from UI regardless
    setFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const clearCompleted = () => {
    setFiles(prev => prev.filter(f => f.status !== 'completed'));
  };

  const getStatusIcon = (status: ProcessingFile['status']) => {
    switch (status) {
      case 'pending':
        return <FileText className="w-5 h-5 text-zinc-400" />;
      case 'uploading':
      case 'processing':
        return <Loader2 className="w-5 h-5 text-blue-400 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-400" />;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-zinc-950 border border-zinc-800 rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-zinc-800">
          <div className="flex items-center gap-3">
            <BookOpen className="w-6 h-6 text-blue-400" />
            <div>
              <h2 className="text-xl font-semibold text-zinc-100">Document Processor</h2>
              <p className="text-xs text-zinc-500 mt-1">Processing with current selected model</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-zinc-800 rounded-lg transition-colors"
          >
            <X className="w-5 h-5 text-zinc-400" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="border-b border-zinc-800 px-6">
          <div className="flex gap-4">
            <button
              onClick={() => setShowExisting(false)}
              className={`py-3 px-1 border-b-2 transition-colors ${
                !showExisting
                  ? 'border-blue-400 text-zinc-100'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Upload New
            </button>
            <button
              onClick={() => setShowExisting(true)}
              className={`py-3 px-1 border-b-2 transition-colors ${
                showExisting
                  ? 'border-blue-400 text-zinc-100'
                  : 'border-transparent text-zinc-400 hover:text-zinc-200'
              }`}
            >
              Manage Library
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {!showExisting ? (
            <>
          {/* Drop Zone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragEnter={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`
              border-2 border-dashed rounded-lg p-8 text-center transition-all
              ${isDragging
                ? 'border-blue-400 bg-blue-400/10'
                : 'border-zinc-700 hover:border-zinc-600 bg-zinc-900/50'
              }
            `}
          >
            <Upload className="w-12 h-12 mx-auto mb-4 text-zinc-500" />
            <p className="text-zinc-300 mb-2">
              Drag and drop your documents here, or{' '}
              <button
                onClick={() => fileInputRef.current?.click()}
                className="text-blue-400 hover:text-blue-300 underline"
              >
                browse files
              </button>
            </p>
            <p className="text-sm text-zinc-500">
              Supported formats: TXT, MD
            </p>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".txt,.md,.pdf,.docx,.xlsx,.xls,.csv,.tsv,.html,.htm,.rtf"
              onChange={(e) => handleFileSelect(e.target.files)}
              className="hidden"
            />
          </div>

          {/* File List */}
          {files.length > 0 && (
            <div className="mt-6 space-y-2">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-zinc-400">
                  Files ({files.length})
                </h3>
                {files.some(f => f.status === 'error') && (
                  <button
                    onClick={() => setFiles(prev => prev.filter(f => f.status !== 'error'))}
                    className="text-xs text-zinc-500 hover:text-zinc-300"
                  >
                    Clear errors
                  </button>
                )}
              </div>

              {files.map(file => (
                <div
                  key={file.id}
                  className="bg-zinc-900 border border-zinc-800 rounded-lg p-4"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start gap-3 flex-1">
                      {getStatusIcon(file.status)}
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <p className="text-sm font-medium text-zinc-200">
                            {file.file.name}
                          </p>
                          <span className="text-xs text-zinc-500">
                            {formatFileSize(file.file.size)}
                          </span>
                        </div>

                        {/* Title/Author fields - only show for pending files */}
                        {file.status === 'pending' && (
                          <div className="mt-3 space-y-2">
                            <div>
                              <label className="text-xs text-zinc-500 block mb-1">
                                Title
                              </label>
                              <input
                                type="text"
                                value={file.customTitle || ''}
                                onChange={(e) => {
                                  setFiles(prev => prev.map(f =>
                                    f.id === file.id
                                      ? { ...f, customTitle: e.target.value }
                                      : f
                                  ));
                                }}
                                placeholder="Auto-extracted from filename"
                                className="w-full px-2 py-1.5 text-sm bg-zinc-800 border border-zinc-700 rounded text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-blue-500 transition-colors"
                                maxLength={200}
                              />
                            </div>
                            <div>
                              <label className="text-xs text-zinc-500 block mb-1">
                                Author (optional)
                              </label>
                              <input
                                type="text"
                                value={file.customAuthor || ''}
                                onChange={(e) => {
                                  setFiles(prev => prev.map(f =>
                                    f.id === file.id
                                      ? { ...f, customAuthor: e.target.value }
                                      : f
                                  ));
                                }}
                                placeholder="Unknown"
                                className="w-full px-2 py-1.5 text-sm bg-zinc-800 border border-zinc-700 rounded text-zinc-200 placeholder-zinc-600 focus:outline-none focus:border-blue-500 transition-colors"
                                maxLength={100}
                              />
                            </div>
                          </div>
                        )}

                        {/* Status message */}
                        <p className="text-xs text-zinc-500 mt-1">
                            {file.status === 'error' && file.error?.includes('Duplicate book') ? (
                            <span className="text-yellow-500">
                              ⚠️ {file.error}
                            </span>
                          ) : (
                            file.error || file.message
                          )}
                          {file.status === 'processing' && file.total_chunks && (
                            <span className="ml-2 text-blue-400">
                              ({Math.round(file.progress)}% - {file.total_chunks} chunks)
                            </span>
                          )}
                        </p>

                        {/* Progress bar */}
                        {(file.status === 'uploading' || file.status === 'processing') && (
                          <div className="mt-2">
                            <div className="h-1 bg-zinc-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-blue-500 transition-all duration-300"
                                style={{ width: `${file.progress}%` }}
                              />
                            </div>
                          </div>
                        )}

                        {/* Show success message briefly before auto-clear */}
                        {file.status === 'completed' && (
                          <div className="flex gap-4 mt-2">
                            <span className="text-xs text-green-400">
                              ✓ Successfully processed - Moving to library...
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {file.status === 'pending' || file.status === 'error' ? (
                      <button
                        onClick={() => removeFile(file.id)}
                        className="p-1 hover:bg-zinc-800 rounded transition-colors"
                        title={
                          file.status === 'error'
                            ? 'Remove from list'
                            : 'Cancel'
                        }
                      >
                        <Trash2 className="w-4 h-4 text-zinc-500 hover:text-red-400 transition-colors" />
                      </button>
                    ) : (file.status === 'uploading' || file.status === 'processing') ? (
                      <button
                        onClick={() => cancelProcessing(file.id)}
                        className="p-1 hover:bg-zinc-800 rounded transition-colors"
                        title="Cancel processing"
                      >
                        <XCircle className="w-4 h-4 text-zinc-500 hover:text-red-400 transition-colors" />
                      </button>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
          )}
          </>
          ) : (
            /* Existing Books Library View */
            <div>
              {loadingBooks ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="w-8 h-8 animate-spin text-zinc-400" />
                </div>
              ) : existingBooks.length === 0 ? (
                <div className="text-center py-12">
                  <BookOpen className="w-12 h-12 mx-auto mb-4 text-zinc-500" />
                  <p className="text-zinc-400">No books in your library</p>
                  <p className="text-sm text-zinc-500 mt-2">Upload documents to get started</p>
                </div>
              ) : (
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-zinc-400 mb-3">
                    Library ({existingBooks.length} books)
                  </h3>
                  {existingBooks.map(book => (
                    <div
                      key={book.book_id}
                      className="bg-zinc-900 border border-zinc-800 rounded-lg p-4"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <BookOpen className="w-4 h-4 text-blue-400" />
                            <p className="text-sm font-medium text-zinc-200">
                              {book.title}
                            </p>
                          </div>
                          {book.author && (
                            <p className="text-xs text-zinc-500 mb-2">
                              By {book.author}
                            </p>
                          )}
                          <div className="flex gap-4 text-xs text-zinc-400">
                            {book.processing_stats?.total_chunks !== undefined && (
                              <span>📦 {book.processing_stats.total_chunks} chunks processed</span>
                            )}
                            {book.upload_timestamp && (
                              <span className="text-zinc-500">
                                {new Date(book.upload_timestamp).toLocaleDateString()}
                              </span>
                            )}
                          </div>
                        </div>
                        <button
                          onClick={() => setDeleteConfirm({ bookId: book.book_id, bookTitle: book.title })}
                          className="p-2 hover:bg-zinc-800 rounded transition-colors"
                          title="Delete from library"
                        >
                          <Trash2 className="w-4 h-4 text-zinc-500 hover:text-red-400 transition-colors" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-zinc-800 p-6">
          <div className="flex items-center justify-between">
            <div className="text-sm text-zinc-500">
              {!showExisting ? (
                `${files.filter(f => f.status === 'completed').length} of ${files.length} processed`
              ) : (
                `${existingBooks.length} books in library`
              )}
            </div>
            <div className="flex gap-3">
              <button
                onClick={onClose}
                className="px-4 py-2 text-zinc-300 hover:text-zinc-100 transition-colors"
              >
                Close
              </button>
              {!showExisting && (
                <button
                  onClick={startProcessing}
                  disabled={!files.some(f => f.status === 'pending') || isProcessing}
                  className={`
                    px-6 py-2 rounded-lg font-medium transition-all
                    ${files.some(f => f.status === 'pending') && !isProcessing
                      ? 'bg-blue-500 text-white hover:bg-blue-600'
                      : 'bg-zinc-800 text-zinc-500 cursor-not-allowed'
                    }
                  `}
                >
                  {isProcessing ? (
                    <span className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Processing...
                    </span>
                  ) : (
                    'Process Documents'
                  )}
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      {deleteConfirm && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-[60]">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl p-6 max-w-md w-full mx-4">
            <div className="flex items-start gap-3 mb-4">
              <div className="p-2 bg-red-500/10 rounded-lg">
                <AlertCircle className="w-5 h-5 text-red-500" />
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-medium text-zinc-100 mb-2">Delete Book</h3>
                <p className="text-sm text-zinc-400">
                  Are you sure you want to delete <span className="text-zinc-200 font-medium">"{deleteConfirm.bookTitle}"</span>?
                </p>
                <p className="text-sm text-zinc-500 mt-2">
                  This will permanently remove all processed data including chunks and embeddings.
                </p>
              </div>
            </div>
            {/* Error message */}
            {deleteError && (
              <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                <p className="text-sm text-red-400">{deleteError}</p>
              </div>
            )}
            <div className="flex gap-3 justify-end">
              <button
                onClick={closeDeleteModal}
                disabled={isDeleting}
                className="px-4 py-2 text-zinc-300 hover:text-zinc-100 transition-colors disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={deleteExistingBook}
                disabled={isDeleting}
                className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 ${
                  isDeleting
                    ? 'bg-red-500/50 text-white/70 cursor-not-allowed'
                    : 'bg-red-500 text-white hover:bg-red-600'
                }`}
              >
                {isDeleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  'Delete Book'
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};