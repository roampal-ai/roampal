import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useChatStore } from '../stores/useChatStore';
import { useSplitPane } from '../hooks/useSplitPane';
// import { useAutoLoadSession } from '../hooks/useAutoLoadSession';
// import { useFeatureFlag } from '../hooks/useFeatureFlags';
import { ROAMPAL_CONFIG } from '../config/roampal';
import { ChatBubbleLeftRightIcon, CircleStackIcon } from '@heroicons/react/24/outline';
import { Sidebar } from './Sidebar';
import { ContextBar } from './ContextBar';
import MemoryPanelV2 from './MemoryPanelV2';
import DevPanel from './DevPanel';
import { TerminalMessageThread } from './TerminalMessageThread';
import { ConnectedCommandInput } from './ConnectedCommandInput';
import { ConnectionStatus } from './ConnectionStatus';
// Removed: ActionStatus, ProcessingIndicator (tool-related components)
import { ImageAnalysis } from './stubs';
// Image Gallery removed - images display inline in chat
import { ShardCreationModal } from './stubs';
import { ShardBooksModal } from './stubs';
import { ShardManagementModal } from './stubs';
import { ProcessingBubble } from './stubs';
import { VoiceConversationModal } from './stubs';
import { BookProcessorModal } from './BookProcessorModal';
import { VoiceSettingsModal } from './stubs';
import type { Message } from './MessageThread';
// Clean imports - no defensive bloat
import { isTauri, getPlatformBadge } from '../utils/tauri';
import { setupTauriEventListeners, showNotification } from '../utils/tauriEvents';
import { apiFetch } from '../utils/fetch';
// Removed: InlinePermissionDialog (tool-related component)
import { SettingsModal } from './SettingsModal';
import { OllamaRequiredModal } from './OllamaRequiredModal';
import { PersonalityCustomizer } from './PersonalityCustomizer';
import MemoryStatsPanel from './MemoryStatsPanel';

/**
 * Neural UI with all features properly integrated
 */
export const ConnectedChat: React.FC = () => {
  // Using MemoryPanelV2 - other panels removed for code cleanup

  // Refs for scroll management
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);

  const {
    conversationId,
    connectionStatus,
    messages,
    searchMemory,
    clearSession,
    initialize,
    getCurrentState,
    getCurrentMessages,
    getCurrentProcessingState,
    switchConversation,
    createConversation,
    isProcessing,
    loadSessions,
    processingStatus,
  } = useChatStore();

  // Removed: Permission dialog state (tool-related)

  // Removed: Action status tracking (tool-related)
  
  // LLM Model state - with localStorage persistence
  const [selectedModel, setSelectedModel] = useState(() => {
    // Load saved model from localStorage, no default fallback
    const savedModel = localStorage.getItem('selectedLLMModel');
    return savedModel || '';
  });
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isSwitchingModel, setIsSwitchingModel] = useState(false);
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [showModelInstallModal, setShowModelInstallModal] = useState(false);
  const [installingModelName, setInstallingModelName] = useState<string | null>(null);
  const [installProgress, setInstallProgress] = useState('');
  const [showInstallPopup, setShowInstallPopup] = useState(false);
  const [showBookProcessor, setShowBookProcessor] = useState(false);
  const [uninstallConfirmModel, setUninstallConfirmModel] = useState<string | null>(null);
  const [showCancelDownloadConfirm, setShowCancelDownloadConfirm] = useState(false);
  const [modelSwitchPending, setModelSwitchPending] = useState<string | null>(null);
  const hasLoadedModels = useRef(false);
  const [showOllamaRequired, setShowOllamaRequired] = useState(false);

  // Check if Ollama is available
  const checkOllamaStatus = async () => {
    try {
      const response = await apiFetch('http://localhost:8000/api/model/ollama/status');
      if (response.ok) {
        const data = await response.json();
        if (!data.available) {
          // Only show modal once per session to avoid spam
          const hasShownThisSession = sessionStorage.getItem('hasShownOllamaPopup');
          if (!hasShownThisSession) {
            setShowOllamaRequired(true);
            sessionStorage.setItem('hasShownOllamaPopup', 'true');
          }
          return false;
        }
        return true;
      }
    } catch (error) {
      console.error('[Ollama] Error checking status:', error);
      const hasShownThisSession = sessionStorage.getItem('hasShownOllamaPopup');
      if (!hasShownThisSession) {
        setShowOllamaRequired(true);
        sessionStorage.setItem('hasShownOllamaPopup', 'true');
      }
      return false;
    }
    return false;
  };

  // Fetch available models on mount
  const fetchModels = async () => {
    try {
      const response = await apiFetch('http://localhost:8000/api/model/available');
      if (response.ok) {
        const data = await response.json();
        // Extract just the model names from the objects
        const modelNames = data.models ? data.models.map((m: any) => m.name) : [];
        setAvailableModels(modelNames);
        hasLoadedModels.current = true;
      }
    } catch (error) {
      console.error('[Models] Error fetching models:', error);
      // Don't check Ollama status on every error - it's already checked at startup
      hasLoadedModels.current = true;
    }
  };

  const handleOllamaRetry = async () => {
    setShowOllamaRequired(false);
    // Recheck Ollama status
    await checkOllamaStatus();
    // Reload models
    await fetchModels();
  };
  
  useEffect(() => {
    fetchModels();
    // Also fetch current model from backend
    fetchCurrentModel();
  }, []);

  // Auto-open Model Manager only on true first run (after models are loaded and still empty)
  useEffect(() => {
    if (hasLoadedModels.current && availableModels.length === 0) {
      setShowModelInstallModal(true);
    }
  }, [availableModels]);

  // Function to get current model from backend
  const fetchCurrentModel = async () => {
    try {
      const response = await apiFetch('http://localhost:8000/api/model/current');
      if (response.ok) {
        const data = await response.json();
        if (data.current_model) {
          // Don't select embedding models
          const isEmbeddingModel = data.current_model.includes('nomic-embed') ||
                                   data.current_model.includes('bge-') ||
                                   data.current_model.includes('all-minilm') ||
                                   data.current_model.includes('llava');

          if (!isEmbeddingModel) {
            setSelectedModel(data.current_model);
            localStorage.setItem('selectedLLMModel', data.current_model);
          }
        }
        return data;
      }
    } catch (error) {
      console.error('[Models] Error fetching current model:', error);
    }
    return null;
  };

  // Function to switch model on backend
  const switchModel = async (modelName: string) => {
    // Warn if switching mid-conversation
    if (messages.length > 0) {
      setModelSwitchPending(modelName);
      return;
    }

    // No conversation, proceed immediately
    performModelSwitch(modelName);
  };

  // Actually perform the model switch
  const performModelSwitch = async (modelName: string) => {
    setIsSwitchingModel(true);
    try {
      const response = await apiFetch('http://localhost:8000/api/model/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_name: modelName })
      });

      if (response.ok) {
        const data = await response.json();
        setSelectedModel(modelName);
        localStorage.setItem('selectedLLMModel', modelName);
        console.log('[Models] Successfully switched to:', modelName);

        // Dispatch custom event for model change
        window.dispatchEvent(new CustomEvent('modelChanged', { detail: { model: modelName } }));

        // Show success message
        const successMsg = document.createElement('div');
        successMsg.className = 'fixed top-4 right-4 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg z-50';
        successMsg.textContent = `Switched to ${modelName}`;
        document.body.appendChild(successMsg);
        setTimeout(() => successMsg.remove(), 3000);
      } else {
        throw new Error('Failed to switch model');
      }
    } catch (error) {
      console.error('[Models] Error switching model:', error);
      // Show error message
      const errorMsg = document.createElement('div');
      errorMsg.className = 'fixed top-4 right-4 bg-red-600 text-white px-4 py-2 rounded-lg shadow-lg z-50';
      errorMsg.textContent = 'Failed to switch model';
      document.body.appendChild(errorMsg);
      setTimeout(() => errorMsg.remove(), 3000);
    } finally {
      setIsSwitchingModel(false);
    }
  };

  // Save selected model to localStorage when it changes
  useEffect(() => {
    localStorage.setItem('selectedLLMModel', selectedModel);
    console.log('[ConnectedChat] Saved selected model:', selectedModel);
  }, [selectedModel]);

//   // Poll for permission requests - ONLY when actively processing
//   useEffect(() => {
//     // Only poll if we're currently processing something
//     if (!processingStatus) {
//       return;
//     }
// 
//     const pollPermissions = async () => {
//       try {
//         const response = await fetch('http://localhost:8000/api/permissions/pending');
//         if (response.ok) {
//           const data = await response.json();
//           // Handle both formats - router returns {pending: []} or main.py returns {request: ...}
//           if (data.pending && data.pending.length > 0) {
//             setPendingPermission(data.pending[0]);
//             console.log('[Permissions] New permission request from pending:', data.pending[0]);
//           } else if (data.request) {
//             setPendingPermission(data.request);
//             console.log('[Permissions] New permission request:', data.request);
//           }
//         }
//       } catch (error: any) {
//         // Silently ignore permission polling errors
//         if (!error?.message?.includes('429')) {
//           console.error('[Permissions] Error polling:', error?.message || error);
//         }
//       }
//     };
// 
//     // Initial poll
//     pollPermissions();
// 
//     // Only poll when actively processing, with longer interval
//     const interval = setInterval(pollPermissions, 5000); // 5 seconds between polls
// 
//     return () => {
//       clearInterval(interval);
//     };
//   }, [processingStatus]); // Only poll when processingStatus changes

//   // Listen for real-time action status updates from SSE
//   useEffect(() => {
//     const handleActionStatusUpdate = (event: CustomEvent) => {
//       const status = event.detail;
//       if (status) {
//         console.log('[ActionStatus] Received real-time update:', status);
// 
//         // Deduplicate actions by ID
//         setActionSteps(prev => {
//           const exists = prev.some(a => a.id === status.id);
//           if (exists) {
//             // Update existing action
//             return prev.map(a => a.id === status.id ? status : a);
//           }
//           return [...prev, status];
//         });
// 
//         setIsExecutingActions(true);
// 
//         // Auto-clear completed/failed actions after 10 seconds (more time to read)
//         if (status.type === 'completed' || status.type === 'failed') {
//           setTimeout(() => {
//             setActionSteps(prev => {
//               const filtered = prev.filter(a => a.id !== status.id);
//               // Check if any actions are still executing
//               const stillExecuting = filtered.some(a =>
//                 a.type === 'executing' || a.type === 'permission'
//               );
//               setIsExecutingActions(stillExecuting);
//               return filtered;
//             });
//           }, 10000);
//         }
//       }
//     };
// 
//     window.addEventListener('action-status-update', handleActionStatusUpdate as EventListener);
//     return () => {
//       window.removeEventListener('action-status-update', handleActionStatusUpdate as EventListener);
//     };
//   }, [actionSteps]);

  // Removed duplicate polling - using consolidated polling below

  // Handle permission response
  // Handle permission dialog responses
//   const handlePermissionResponse = async (allowed: boolean, remember?: 'session' | 'always') => {
//     if (!pendingPermission) return;
// 
//     try {
//       const response = await fetch('http://localhost:8000/api/permissions/respond', {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({
//           request_id: pendingPermission.id,  // Changed from 'id' to 'request_id'
//           allowed,
//           remember
//         })
//       });
// 
//       if (response.ok) {
//         console.log('[Permissions] Response sent:', { allowed, remember });
//         setPendingPermission(null); // Clear the dialog
//       }
//     } catch (error) {
//       console.error('[Permissions] Error sending response:', error);
//     }
//   };

  // Consolidated memory refresh on conversation change
  // (Removed duplicate - handled by debounced effect below)

//   // Auto-clear action steps when response completes and no actions are executing
//   useEffect(() => {
//     // Check if any assistant message is currently streaming
//     const isAnyMessageStreaming = messages.some(m =>
//       m.sender === 'assistant' && m.streaming
//     );
// 
//     // If nothing is streaming and we have action steps, clear them after a delay
//     if (!isAnyMessageStreaming && !isExecutingActions && actionSteps.length > 0) {
//       const timeout = setTimeout(() => {
//         console.log('[ActionStatus] Auto-clearing completed actions');
//         setActionSteps([]);
//       }, 5000); // Clear after 5 seconds
// 
//       return () => clearTimeout(timeout);
//     }
//   }, [messages, isExecutingActions, actionSteps.length]);

  // Removed duplicate initialization - handled below

  // TEMPORARILY DISABLED POLLING - Relying on SSE/WebSocket for real-time updates
  // The polling was causing excessive requests even when WebSocket was connected
  useEffect(() => {
    console.log('[Polling] Polling disabled - using SSE/WebSocket for real-time updates');
    // Will re-enable with proper dependency tracking when needed
  }, []);

  // Handle model installation with real progress streaming
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadDetails, setDownloadDetails] = useState({ downloaded: '', total: '', speed: '' });
  const [downloadAbortController, setDownloadAbortController] = useState<AbortController | null>(null);

  const handleInstallModel = async (modelName: string) => {
    setInstallingModelName(modelName);
    setShowInstallPopup(true);
    setInstallProgress(`Initializing download for ${modelName}...`);
    setDownloadProgress(0);
    setDownloadDetails({ downloaded: '', total: '', speed: '' });

    // Create abort controller for cancellation
    const abortController = new AbortController();
    setDownloadAbortController(abortController);

    try {
      // Always use SSE for model downloads (WebSocket has CSP/CORS issues in Tauri)
      const isTauriProduction = false; // Force SSE - works better than WebSocket in Tauri webview

      if (isTauriProduction) {
        // Use WebSocket in Tauri production (SSE doesn't work due to webview limitations)
        console.log('[Model Install] Using WebSocket for Tauri production');

        const wsUrl = 'ws://127.0.0.1:8000/api/model/pull-ws';
        const ws = new WebSocket(wsUrl);

        // Store WebSocket for potential cleanup
        const wsCleanup = () => {
          if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
            ws.close();
          }
        };

        // Handle abort
        abortController.signal.addEventListener('abort', wsCleanup);

        ws.onopen = () => {
          console.log('[Model Install] WebSocket connected successfully');
          // Send model name to start download
          ws.send(JSON.stringify({ model: modelName }));
          console.log('[Model Install] Sent model name:', modelName);
        };

        ws.onmessage = (event) => {
          clearTimeout(connectionTimeout);
          try {
            console.log('[Model Install] Received WebSocket message:', event.data.substring(0, 100));
            const data = JSON.parse(event.data);

            if (data.type === 'progress') {
              setDownloadProgress(data.percent || 0);
              if (data.downloaded && data.total) {
                setDownloadDetails({
                  downloaded: data.downloaded,
                  total: data.total,
                  speed: data.speed || ''
                });
                setInstallProgress(
                  `Downloading ${modelName}: ${data.percent}% (${data.downloaded}/${data.total}${data.speed ? ' @ ' + data.speed : ''})`
                );
              } else {
                setInstallProgress(data.message || `Downloading ${modelName}...`);
              }
            } else if (data.type === 'complete') {
              setDownloadProgress(100);
              setInstallProgress(`✓ ${modelName} installed successfully`);
              fetchModels().then(() => {
                // Auto-switch to first chat model only
                const nonLLMs = ['llava', 'nomic-embed', 'bge-', 'all-minilm', 'mxbai-embed'];
                const isChatModel = !nonLLMs.some(excluded => modelName.toLowerCase().includes(excluded));

                if (isChatModel) {
                  // Check backend's current model to determine if this is first chat model
                  fetchCurrentModel().then(currentModelData => {
                    const backendHasNoChatModel = !currentModelData?.can_chat || currentModelData?.is_embedding_model;

                    if (backendHasNoChatModel) {
                      // First chat model - auto-switch
                      performModelSwitch(modelName);
                    }
                    // Else: subsequent install - toast already shows success, user switches manually
                  });
                }

                setTimeout(() => {
                  setShowInstallPopup(false);
                  setInstallingModelName(null);
                  setInstallProgress('');
                  setDownloadProgress(0);
                  setDownloadDetails({ downloaded: '', total: '', speed: '' });
                  setDownloadAbortController(null);
                  wsCleanup();
                }, 2000);
              });
            } else if (data.type === 'error') {
              setInstallProgress(`❌ ${data.message}`);
              setDownloadProgress(0);
              setInstallingModelName(null);
              setDownloadAbortController(null);
              wsCleanup();
              // Show error for 5 seconds then clear
              setTimeout(() => {
                setInstallProgress('');
              }, 5000);
            }
          } catch (e) {
            console.error('Failed to parse WebSocket data:', e);
          }
        };

        // Add connection timeout
        const connectionTimeout = setTimeout(() => {
          if (ws.readyState !== WebSocket.OPEN) {
            console.error('[Model Install] WebSocket connection timeout');
            ws.close();
            setInstallProgress(`❌ Connection timeout. Please ensure backend is running on port 8000.`);
            setDownloadProgress(0);
            // Keep modal open but show error
            // User can manually close if needed
            setDownloadProgress(0);
          }
        }, 5000);

        ws.onerror = (error) => {
          clearTimeout(connectionTimeout);
          console.error('WebSocket error:', error);
          console.error('WebSocket URL was:', wsUrl);
          console.error('WebSocket state:', ws.readyState);
          setInstallProgress(`❌ Connection error. Please check if backend is running on port 8000.`);
          setDownloadProgress(0);
          setTimeout(() => {
            setShowInstallPopup(false);
            setInstallingModelName(null);
            setInstallProgress('');
            setDownloadAbortController(null);
          }, 3000);
        };

        ws.onclose = (event) => {
          clearTimeout(connectionTimeout);
          console.log('[Model Install] WebSocket closed. Code:', event.code, 'Reason:', event.reason);
          if (!installingModelName) {
            // Already cleaned up, don't show error
            return;
          }
          if (event.code !== 1000 && event.code !== 1001) {
            // Abnormal close
            setInstallProgress(`❌ Connection lost. Please try again.`);
            // Keep modal open but show error
            // User can manually close if needed
            setDownloadProgress(0);
          }
        };

        return; // Exit after setting up WebSocket
      }

      // Use SSE for dev mode (original implementation)
      console.log('[Model Install] Using SSE for development mode');
      const response = await apiFetch('http://localhost:8000/api/model/pull-stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model: modelName }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      if (!reader) {
        // Fallback for when streaming isn't available
        console.log('[Model Install] Streaming not available, using fallback');
        setInstallProgress(`Downloading ${modelName}... (progress tracking not available)`);

        // Just wait for the response to complete
        const text = await response.text();
        if (text.includes('complete') || text.includes('success')) {
          setInstallProgress(`✓ ${modelName} installed successfully`);
          await fetchModels();
          setTimeout(() => {
            setShowInstallPopup(false);
            setInstallingModelName(null);
            setInstallProgress('');
          }, 2000);
        }
        return;
      }

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));

                if (data.type === 'progress') {
                  setDownloadProgress(data.percent || 0);
                  if (data.downloaded && data.total) {
                    setDownloadDetails({
                      downloaded: data.downloaded,
                      total: data.total,
                      speed: data.speed || ''
                    });
                    setInstallProgress(
                      `Downloading ${modelName}: ${data.percent}% (${data.downloaded}/${data.total}${data.speed ? ' @ ' + data.speed : ''})`
                    );
                  } else {
                    setInstallProgress(data.message || `Downloading ${modelName}...`);
                  }
                } else if (data.type === 'complete') {
                  setDownloadProgress(100);
                  setInstallProgress(`✓ ${modelName} installed successfully`);
                  await fetchModels(); // Refresh model list

                  // Auto-switch to first chat model only
                  const nonLLMs = ['llava', 'nomic-embed', 'bge-', 'all-minilm', 'mxbai-embed'];
                  const isChatModel = !nonLLMs.some(excluded => modelName.toLowerCase().includes(excluded));

                  if (isChatModel) {
                    // Check backend's current model to determine if this is first chat model
                    const currentModelData = await fetchCurrentModel();
                    const backendHasNoChatModel = !currentModelData?.can_chat || currentModelData?.is_embedding_model;

                    if (backendHasNoChatModel) {
                      // First chat model - auto-switch
                      await performModelSwitch(modelName);
                    }
                    // Else: subsequent install - toast already shows success, user switches manually
                  }

                  setTimeout(() => {
                    setShowInstallPopup(false);
                    setInstallingModelName(null);
                    setInstallProgress('');
                    setDownloadProgress(0);
                    setDownloadDetails({ downloaded: '', total: '', speed: '' });
                    setDownloadAbortController(null);
                  }, 2000);
                } else if (data.type === 'error') {
                  setInstallProgress(`❌ ${data.message}`);
                  setDownloadProgress(0);
                  setInstallingModelName(null);
                  setDownloadAbortController(null);
                  // Show error for 5 seconds then clear
                  setTimeout(() => {
                    setInstallProgress('');
                  }, 5000);
                  return; // Exit early, don't throw
                }
              } catch (e) {
                console.error('Failed to parse SSE data:', e);
              }
            }
          }
        }
      }
    } catch (error: any) {
      // Check if error was due to abort
      if (error.name === 'AbortError') {
        setInstallProgress(`Download cancelled for ${modelName}`);
      } else {
        setInstallProgress(`Error installing ${modelName}: ${error.message || error}`);
      }
      setDownloadProgress(0);
      setTimeout(() => {
        setInstallingModelName(null);
        setShowInstallPopup(false);
        setInstallProgress('');
        setDownloadDetails({ downloaded: '', total: '', speed: '' });
        setDownloadAbortController(null);
      }, 3000);
    }
  };
  
  // Handle model uninstallation - show confirmation modal
  const handleUninstallModel = (modelName: string) => {
    setUninstallConfirmModel(modelName);
  };

  // Confirm and execute uninstall
  const confirmUninstall = async () => {
    if (!uninstallConfirmModel) return;

    const modelName = uninstallConfirmModel;
    setUninstallConfirmModel(null); // Close modal

    try {
      const response = await apiFetch(`http://localhost:8000/api/model/uninstall/${encodeURIComponent(modelName)}`, {
        method: 'DELETE',
      });

      const data = await response.json();

      if (data.success) {
        await fetchModels(); // Refresh model list
        if (selectedModel === modelName) {
          // If we're uninstalling the selected model, switch to default
          // Don't default to anything, let backend decide
          fetchCurrentModel();
        }
      } else {
        alert(`Failed to uninstall ${modelName}: ${data.error}`);
      }
    } catch (error) {
      alert(`Error uninstalling ${modelName}: ${error}`);
    }
  };

  // Confirm and execute download cancellation
  const confirmCancelDownload = () => {
    setShowCancelDownloadConfirm(false);
    if (downloadAbortController) {
      downloadAbortController.abort();
      setDownloadAbortController(null);
      setShowInstallPopup(false);
      setInstallingModelName(null);
      setInstallProgress('');
    }
  };

  // Confirm and execute model switch
  const confirmModelSwitch = () => {
    if (modelSwitchPending) {
      performModelSwitch(modelSwitchPending);
      setModelSwitchPending(null);
    }
  };
  
  // Model token limits for agent mode capability (October 2025)
  const MODEL_TOKEN_LIMITS: Record<string, number> = {
    // AGENT-CAPABLE (12K+ context)
    'gpt-oss:120b': 128000,
    'gpt-oss:20b': 128000,
    'llama3.2:8b': 131072,
    'llama3.2:3b': 131072,
    'llama3.2:1b': 131072,
    'llama3.3:70b': 131072,
    'llama3.1:70b': 131072,
    'llama3.1:8b': 131072,
    'gemma3:27b': 128000,
    'gemma3:12b': 128000,
    'gemma3:4b': 128000,
    'qwen3:235b': 262144,  // MoE with 256K context
    'qwen3:32b': 40960,
    'qwen3:14b': 40960,
    'qwen3:8b': 40960,
    'qwen3:4b': 262144,  // 256K context
    'qwen2.5:32b': 32768,
    'qwen2.5:14b': 32768,
    'qwen2.5:7b': 32768,
    'mistral:7b': 32768,
  };

  // Models categorized by capability (Updated October 2025 - Tool Capable Only)
  const curatedModels = [
    // Recommended for Chat + Memory - Models with verified tool calling support
    {
      category: 'Recommended for Chat + Memory',
      description: 'Models with native tool calling support for Roampal\'s memory system',
      icon: 'sparkles',
      models: [
        // OpenAI Open Source (NEW)
        { name: 'gpt-oss:120b', description: 'OpenAI\'s flagship open model - Native tools', size: '80GB', tokens: 128000, agentCapable: true, license: 'Apache 2.0', badge: 'top' },
        { name: 'gpt-oss:20b', description: 'OpenAI\'s efficient model - Excellent tools', size: '16GB', tokens: 128000, agentCapable: true, license: 'Apache 2.0', badge: 'recommended' },

        // Meta Llama Series (Tool Support)
        { name: 'llama3.3:70b', description: 'Meta\'s latest 70B - Native tools, 128K context', size: '43GB', tokens: 131072, agentCapable: true, license: 'Meta License' },

        // Qwen Series (Best Tool Support)
        { name: 'qwen2.5:72b', description: 'Massive Qwen - Superior tool calling', size: '41GB', tokens: 32768, agentCapable: true, license: 'Qwen License' },
        { name: 'qwen2.5:32b', description: 'Powerful Qwen - Excellent tools', size: '20GB', tokens: 32768, agentCapable: true, license: 'Apache 2.0', badge: 'recommended' },
        { name: 'qwen2.5:14b', description: 'Larger Qwen - Great tool performance', size: '9.0GB', tokens: 32768, agentCapable: true, license: 'Apache 2.0' },
        { name: 'qwen2.5:7b', description: 'Best-in-class tool calling', size: '4.7GB', tokens: 32768, agentCapable: true, license: 'Apache 2.0', badge: 'recommended' },

        // Mistral Family
        { name: 'mixtral:8x7b', description: 'MoE architecture with native tools', size: '26GB', tokens: 32768, agentCapable: true, license: 'Apache 2.0' },
      ]
    },


    // Lightweight & Tool-Capable - Smaller models that still support tools
    {
      category: 'Lightweight & Tool-Capable',
      description: 'Smaller models (<7B) may struggle with complex reasoning, memory scoring, and reliable tool calling. May output tool JSON as text instead of invoking tools. ⚠️ Use qwen2.5:7b or larger for production.',
      icon: 'chat-bubble-left-right',
      models: [
        // Small but capable - with warnings
        { name: 'llama3.1:8b', description: '⚠️ Unreliable tool calling - Use qwen2.5:7b instead', size: '4.7GB', tokens: 131072, agentCapable: true, license: 'Meta License' },
        { name: 'qwen2.5:3b', description: '⚠️ May have inconsistent tool calling', size: '1.9GB', tokens: 32768, agentCapable: true, license: 'Apache 2.0' },
        { name: 'llama3.2:3b', description: '⚠️ May output tool JSON as text', size: '2.0GB', tokens: 131072, agentCapable: true, license: 'Meta License' },

      ]
    }
  ];
  
  // Close dropdown on click outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest('.model-dropdown-container')) {
        setShowModelDropdown(false);
      }
    };
    
    if (showModelDropdown) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [showModelDropdown]);
  
  // Get model options with descriptions
  const getModelOptions = () => {
    const modelDescriptions: Record<string, string> = {
      // OpenAI Open Source Models
      'gpt-oss:120b': '80GB • OpenAI flagship open model • Native tools',
      'gpt-oss:20b': '16GB • OpenAI efficient model • Excellent tools',

      // Llama 3 Series (Tool Support)
      'llama3.3:70b': '43GB • Meta latest 70B • Native tools',
      'llama3.1:8b': '4.7GB ⚠️ Unreliable tools • Use qwen2.5:7b',
      'llama3.2:3b': '2.0GB ⚠️ May hallucinate tool calls',

      // Qwen 2.5 Series (Best Tool Support)
      'qwen2.5:72b': '41GB • Massive Qwen • Superior tools',
      'qwen2.5:32b': '20GB • Powerful Qwen • Excellent tools',
      'qwen2.5:14b': '9.0GB • Larger Qwen • Great tools',
      'qwen2.5:7b': '4.7GB • Best-in-class tool calling',
      'qwen2.5:3b': '1.9GB • Efficient with tool support',


      // Mistral
      'mistral:7b': '4GB • Fast and efficient',
      'mixtral:8x7b': '26GB • Mixture of Experts',

      // Gemma 3 Series
      'gemma3:12b': '8.1GB • Multimodal (text + image) • 128K',
      'gemma3:4b': '3.3GB • Compact multimodal',
      'gemma3:1b': '815MB • Ultra-lightweight',

      // Gemma 2
      'gemma2:9b': '5.4GB • Google balanced',

      // Phi
      'phi3:3.8b': '2.3GB • Compact • 128K context',
    };
    
    // Filter out non-LLM models (embedding, vision, etc.)
    const nonLLMs = ['llava', 'nomic-embed', 'bge-', 'all-minilm', 'mxbai-embed'];
    const filteredModels = availableModels.filter(model =>
      !nonLLMs.some(excluded => model.toLowerCase().includes(excluded))
    );

    return filteredModels.map(model => {
      const isAgentCapable = MODEL_TOKEN_LIMITS[model] && MODEL_TOKEN_LIMITS[model] >= 12000;
      const baseDescription = modelDescriptions[model] || 'Custom model';

      return {
        value: model,
        label: model.split(':')[0].replace(/-/g, ' '),
        description: baseDescription,
        agentCapable: isAgentCapable
      };
    });
  };

  // Check if chat model is available
  const hasChatModel = React.useMemo(() => {
    const options = getModelOptions();
    return options.length > 0;
  }, [availableModels]);

  // Get processing state
  const userId = 'default';  // Roampal is single-user
  
  // Get actual processing state from store
  const currentState = getCurrentState();
  const { processingStage } = getCurrentProcessingState();
  const sessionId = currentState?.conversationId || conversationId || 'default';
  
  // Local state for memories since V2 store doesn't have global activeMemories
  const [localMemories, setLocalMemories] = useState<any[]>([]);
  const [isRefreshingMemories, setIsRefreshingMemories] = useState(false);
  const [lastMemoryRefresh, setLastMemoryRefresh] = useState<Date | null>(null);
  const [knowledgeGraphData, setKnowledgeGraphData] = useState<any>({ concepts: 0, relationships: 0 });
  const activeMemories = localMemories.length > 0 ? localMemories : (currentState?.memories || []);
  
  // Ref to track if title regeneration has been triggered (to prevent duplicates)
  const titleRegenerationTriggered = useRef(false);

  // Function to scroll to bottom
  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  // Auto-scroll ONLY if user is near bottom (smart scroll)
  useEffect(() => {
    if (!messagesContainerRef.current) return;

    const container = messagesContainerRef.current;
    const { scrollTop, scrollHeight, clientHeight } = container;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

    // Only auto-scroll if user is within 150px of bottom
    if (distanceFromBottom < 150) {
      const timer = setTimeout(() => {
        scrollToBottom();
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [messages]);

  // Handle scroll to show/hide scroll-to-bottom button
  const handleScroll = () => {
    if (messagesContainerRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = messagesContainerRef.current;
      const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
      setShowScrollButton(!isNearBottom);
    }
  };

  
  // Helper function to capitalize names for display
  const capitalizeForDisplay = (name: string | null | undefined): string => {
    if (!name) return '';
    return name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
  };
  
  
  // Sidebar collapse state management
  const [leftPaneCollapsed, setLeftPaneCollapsed] = useState(false);
  const [rightPaneCollapsed, setRightPaneCollapsed] = useState(false);

  // Use the split pane hook for draggable sidebars - SAME SIZES
  const leftPaneHook = useSplitPane({
    initialSize: 300,
    minSize: 0,  // Allow dragging to fully collapsed
    maxSize: 500,
    direction: 'horizontal',
    storageKey: 'loopsmith-left-sidebar-width'
    // Normal behavior: drag right = sidebar grows, drag left = sidebar shrinks
  });

  const rightPaneHook = useSplitPane({
    initialSize: 300,
    minSize: 0,  // Allow dragging to fully collapsed
    maxSize: 500,
    direction: 'horizontal',
    storageKey: 'loopsmith-right-sidebar-width',
    inverted: true  // Right sidebar grows when dragging left
  });

  const leftPane = {
    width: leftPaneHook.size,
    isCollapsed: leftPaneCollapsed || leftPaneHook.size <= 50,
    isResizing: leftPaneHook.isDragging,
    toggle: () => setLeftPaneCollapsed(!leftPaneCollapsed),
    toggleCollapsed: () => {
      if (leftPaneCollapsed || leftPaneHook.size <= 50) {
        setLeftPaneCollapsed(false);
        leftPaneHook.setSize(300); // Reset to default size when reopening
      } else {
        setLeftPaneCollapsed(true);
      }
    },
    onMouseDown: leftPaneHook.handleMouseDown,
    resizerProps: {
      onMouseDown: leftPaneHook.handleMouseDown,
      style: { cursor: 'col-resize' }
    }
  };

  const rightPane = {
    width: rightPaneHook.size,
    isCollapsed: rightPaneCollapsed || rightPaneHook.size <= 50,
    isResizing: rightPaneHook.isDragging,
    toggle: () => setRightPaneCollapsed(!rightPaneCollapsed),
    toggleCollapsed: () => {
      if (rightPaneCollapsed || rightPaneHook.size <= 50) {
        setRightPaneCollapsed(false);
        rightPaneHook.setSize(300); // Reset to default size when reopening
      } else {
        setRightPaneCollapsed(true);
      }
    },
    onMouseDown: rightPaneHook.handleMouseDown,
    resizerProps: {
      onMouseDown: rightPaneHook.handleMouseDown,
      style: { cursor: 'col-resize' }
    }
  };
  
  
  // Essential UI states only
  const [showSettings, setShowSettings] = useState(false);
  const [showDevPanel, setShowDevPanel] = useState(false);
  const [showPersonalityCustomizer, setShowPersonalityCustomizer] = useState(false);
  const [showMemoryStats, setShowMemoryStats] = useState(false);

  // Keyboard shortcut for dev panel (Ctrl+Shift+D)
  useEffect(() => {
    const handleKeydown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'D') {
        e.preventDefault();
        setShowDevPanel(true);
      }
    };
    
    document.addEventListener('keydown', handleKeydown);
    return () => document.removeEventListener('keydown', handleKeydown);
  }, []);
  
  
  // Setup Tauri event listeners if in Tauri mode
  useEffect(() => {
    if (isTauri()) {
      setupTauriEventListeners();
      
      // Listen for custom events from system tray
      const handleOpenSettings = () => {
        setShowSettings(true);
      };

      // Note: handleNewChat is defined later in the component
      // We'll add its event listener there
      window.addEventListener('tauri-open-settings', handleOpenSettings);

      return () => {
        window.removeEventListener('tauri-open-settings', handleOpenSettings);
      };
    }
  }, []);
  

  // Initialize on mount
  useEffect(() => {
    const initializeComponent = async () => {
      console.log('[ConnectedChat] Component mounting, initializing...');

      // Initialize the store once (this will set connection status)
      await initialize();

      // Check Ollama status ONCE on app startup
      checkOllamaStatus();

      // Roampal is single-user - no user management needed
      
      // Wait a bit for persisted state to load before checking for user
      setTimeout(() => {
        const currentState = useChatStore.getState();
        console.log('[ConnectedChat] Current user after persistence load:', {
          // No users in single-user system
        });
        
        // Wait for authManager to check auth status first
        const waitForAuth = async () => {
          // Give authManager time to restore session from localStorage
          let attempts = 0;
          while (attempts < 10) {
            if (window.authManager && !window.authManager.isLoading) {
              break;
            }
            await new Promise(resolve => setTimeout(resolve, 100));
            attempts++;
          }
          
          // Now check if authenticated user exists
          if (window.authManager && window.authManager.user) {
            const authUser = window.authManager.user;
            const userId = authUser.user_id || authUser.id;
            const displayName = authUser.display_name || authUser.displayName;
            console.log('[ConnectedChat] Using authenticated user:', displayName, userId);
            // No user switching needed
            
            // Load sessions
            console.log('[ConnectedChat] Loading sessions...');
            currentState.loadSessions().catch(err => {
              console.error('[ConnectedChat] Failed to load sessions:', err);
            });
          } else {
            // No authenticated user - clear any invalid persisted user
            console.log('[ConnectedChat] No authenticated user, clearing invalid stored user');
            // No action needed for single-user system
          }
        };
        
        waitForAuth();
      }, 100); // Small delay to ensure persistence has loaded
    };
    
    initializeComponent();

    // Load backend sessions (memory fetch handled by consolidated effect)
    setTimeout(() => {
      console.log('[ConnectedChat] Loading sessions...');
      useChatStore.getState().loadSessions().catch(err => {
        console.error('[ConnectedChat] Failed to load sessions:', err);
      });
    }, 100);
  }, [initialize]);  // Add initialize as dependency

  

  // Fetch users when Settings modal opens
  useEffect(() => {
    if (showSettings) {
      console.log('[ConnectedChat] Settings opened, fetching users from backend...');
      fetchBackendUsers();
    }
  }, [showSettings]);

  // WebSocket connection is managed by useChatStore to prevent duplicates
  
  // Fetch users from backend (removed - single user system)
  const fetchBackendUsers = async () => {
    console.log('[fetchBackendUsers] Single-user system - no user management needed');
  };
  
  // Consolidated memory fetch: on mount and conversation change (debounced)
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      console.log('[ConnectedChat] Fetching memories for:', {
        conversation: conversationId,
        mounted: !conversationId
      });
      fetchMemories();
      fetchKnowledgeGraph();
    }, 300); // Debounce for 300ms

    return () => clearTimeout(timeoutId);
  }, [conversationId]); // Triggers on mount (conversationId=null) and on change
  
  // Disabled auto-refresh - was causing UI instability
  // Manual refresh is sufficient
  // useEffect(() => {
  //   if (conversationId) {
  //     const interval = setInterval(() => {
  //       fetchMemories();
  //       fetchKnowledgeGraph();
  //     }, 60000);
  //     return () => clearInterval(interval);
  //   }
  // }, [conversationId]);
  
  // Debug knowledge graph state changes
  useEffect(() => {
    console.log('[ConnectedChat] Knowledge graph state updated:', {
      hasData: !!knowledgeGraphData,
      concepts: knowledgeGraphData?.concepts,
      relationships: knowledgeGraphData?.relationships,
      rawNodes: knowledgeGraphData?.raw?.nodes?.length,
      rawEdges: knowledgeGraphData?.raw?.edges?.length
    });
  }, [knowledgeGraphData]);

  // Auto-generate title for new conversations
  const generateTitle = async (conversationId: string, messages: any[]) => {
    try {
      const response = await fetch(`${ROAMPAL_CONFIG.API_BASE}/api/chat/generate-title`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          conversation_id: conversationId,
          messages: messages.slice(0, 4) // First 4 messages
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.title && !data.fallback) {
          // Update session title in store
          useChatStore.getState().updateSessionTitle(conversationId, data.title);
          console.log(`[ConnectedChat] Generated title for ${conversationId}: ${data.title}`);
        }
      }
    } catch (error) {
      console.error('[ConnectedChat] Title generation failed:', error);
    }
  };

  // Auto-generate titles after messages are sent
  useEffect(() => {
    const unsubscribe = useChatStore.subscribe(
      (state) => {
        const userShardKey = 'all';
        const currentSession = Array.isArray(state.sessions)
          ? state.sessions.find((s: any) => s.id === state.conversationId)
          : (state.sessions as any)?.[userShardKey]?.find((s: any) => s.id === state.conversationId);

        // Auto-generate title after 2 messages if still "Untitled Session"
        if (currentSession &&
            currentSession.name === 'Untitled Session' &&
            currentSession.messages &&
            currentSession.messages.length >= 2 &&
            state.conversationId) {
          console.log(`[ConnectedChat] Triggering title generation for ${state.conversationId}`);
          generateTitle(state.conversationId, currentSession.messages);
        }
      }
    );

    return unsubscribe;
  }, []);
  
  // Fetch memories from Roampal backend
  const fetchMemories = useCallback(async (isManualRefresh = false) => {
    try {
      if (isManualRefresh) {
        setIsRefreshingMemories(true);
      }
      console.log('[Fetching memories] Fetching fragments from Roampal backend');

      // Fetch from working knowledge collections (exclude books - they're reference material, not memories)
      const collections = ['working', 'history', 'patterns'];
      const allFragments: any[] = [];

      for (const collectionType of collections) {
        try {
          const response = await apiFetch(`http://localhost:8000/api/memory/enhanced/collections/${collectionType}`);
          if (response.ok) {
            const data = await response.json();
            const items = data.memories || [];  // API returns 'memories' not 'items'
            // Add collection type to each item
            items.forEach((item: any) => {
              item.collection_type = collectionType;
            });
            allFragments.push(...items);
          }
        } catch (err) {
          console.error(`[Fetching memories] Error fetching ${collectionType}:`, err);
        }
      }

      console.log(`[API Response] Retrieved ${allFragments.length} fragments from ${collections.length} collections`);
      const fragments = allFragments;

      console.log(`[API Response] Retrieved ${fragments.length} fragments`);

      // Convert fragments to the format expected by MemoryPanelV2
      const processedMemories = fragments.map((f: any, idx: number) => {
        // Parse timestamp - check both top level and metadata
        let timestamp = f.timestamp ? new Date(f.timestamp) :
                       f.metadata?.timestamp ? new Date(f.metadata.timestamp) :
                       new Date();

        // Get type from collection_type or metadata
        const memoryType = f.collection_type || f.collection || f.type || 'memory';

        // Get score
        const score = f.score || f.metadata?.composite_score || 0.5;

        return {
          id: f.id || `mem-${Date.now()}-${idx}`,
          text: f.content || '',
          content: f.content || '',
          type: memoryType,
          timestamp,
          score,
          relevance: score,
          session_id: f.session_id || f.metadata?.session_id,
          tags: f.tags || f.metadata?.tags || [],
          usefulness_score: f.usefulness_score || f.metadata?.usefulness_score,
          sentiment_score: f.sentiment_score || f.metadata?.sentiment_score,
          uses: f.uses || f.metadata?.uses,
          last_outcome: f.last_outcome || f.metadata?.last_outcome,
          collection: f.collection || memoryType
        };
      });

      // Update memories in the local state
      setLocalMemories(processedMemories);
      setLastMemoryRefresh(new Date());
      console.log(`[Memory Update] Successfully loaded ${processedMemories.length} memories`);
      return processedMemories;

    } catch (error: any) {
      console.error('[Fetching memories] Error:', error);
      console.error('[Fetching memories] Error details:', {
        message: error.message,
        stack: error.stack,
        type: error.name
      });
    } finally {
      if (isManualRefresh) {
        setIsRefreshingMemories(false);
      }
    }
  }, []);
  
  // Fetch knowledge graph from Roampal backend
  const fetchKnowledgeGraph = useCallback(async () => {
    try {
      console.log('[Knowledge Graph] Fetching from Roampal backend');

      // Roampal has a unified knowledge graph
      const response = await apiFetch('http://localhost:8000/api/memory/knowledge-graph');

      if (!response.ok) {
        console.error('[Knowledge Graph] API error:', response.status);
        setKnowledgeGraphData({
          concepts: 0,
          relationships: 0,
          activeTopics: [],
          nodes: []
        });
        return;
      }

      const data = await response.json();

      console.log('[Knowledge Graph] API response:', data);

      // Transform the data to match what ContextBar expects
      const transformedData = {
        concepts: data.total_concepts || 0,
        relationships: data.total_relationships || 0,
        activeTopics: data.activeTopics || [],
        nodes: data.nodes || [],
        edges: data.edges || []
      };

      setKnowledgeGraphData(transformedData);
    } catch (error: any) {
      console.error('[Knowledge Graph] Error details:', error);
      console.error('[Knowledge Graph] Error stack:', error.stack);
    }
  }, []);

  // Listen for real-time memory updates from SSE complete event
  useEffect(() => {
    const handleMemoryUpdate = (event: CustomEvent) => {
      console.log('[ConnectedChat] Memory updated event received, refreshing...', event.detail);
      fetchMemories();
      fetchKnowledgeGraph();
      // Also refresh sessions list (in case sessions were deleted in data management)
      loadSessions();
    };

    window.addEventListener('memoryUpdated', handleMemoryUpdate as EventListener);
    return () => {
      window.removeEventListener('memoryUpdated', handleMemoryUpdate as EventListener);
    };
  }, [fetchMemories, fetchKnowledgeGraph, loadSessions]);

  // Removed shard books - not part of Roampal architecture
  
  // Convert store messages to component format with proper timestamp handling
  console.log('[ConnectedChat] Store messages count:', messages.length);
  const componentMessages: Message[] = messages.map((msg: any) => {
    // Convert images to attachments if present
    let attachments = msg.attachments;
    // Ensure images is an array before trying to use it
    const images = msg.images ? (
      Array.isArray(msg.images) ? msg.images : 
      typeof msg.images === 'string' ? [msg.images] : []
    ) : [];
    
    if (!attachments && images.length > 0) {
      attachments = images.map((imgUrl: string, idx: number) => ({
        id: `img-${msg.id}-${idx}`,
        name: `Image ${idx + 1}`,
        size: 0,
        type: 'image/jpeg',
        url: imgUrl,
      }));
    }
    
    return {
      id: msg.id,
      sender: msg.sender as 'user' | 'assistant' | 'system',
      shard: msg.shard,
      content: msg.content || msg.text || '',
      timestamp: msg.timestamp instanceof Date ? msg.timestamp : new Date(msg.timestamp),
      status: 'sent' as const,
      attachments, // Use converted attachments
      // Pass memory citations with confidence scores directly
      citations: msg.citations || [],
      memories: msg.citations?.map((c: any) => ({
        id: c.id,
        count: 1,
        preview: [c.snippet || c.title || c.text || ''],
      })),
      streaming: msg.streaming || false,
      toolExecutions: msg.toolExecutions || undefined
    };
  });
  // Sidebar now reads sessions directly from store - no transformation needed here

  // Mock knowledge graph data if not loaded
  const knowledgeGraph = knowledgeGraphData || {
    concepts: activeMemories.length * 3,
    relationships: activeMemories.length * 7,
    activeTopics: ['loopsmith'],
  };
  
  // Mock references (from citations with URLs)
  const references = messages
    .flatMap(m => m.citations || [])
    .filter(c => c.url)
    .map(c => ({
      id: c.id,
      title: c.title || 'Reference',
      url: c.url,
      snippet: c.snippet || '',
      timestamp: new Date(),
    }));
  
  const handleShardChange = async (shard: string) => {
    // TODO: Implement shard switching
    console.log('Shard change:', shard);
    // Refresh knowledge graph for the new shard
    await fetchKnowledgeGraph();
  };
  
  const handleMemoryClick = async (memoryId: string) => {
    // Fetch full memory details
    console.log('Memory clicked:', memoryId);
  };
  
  const handleCommandClick = (command: string) => {
    console.log('Command clicked:', command);
  };
  
  const handleSessionSelect = async (sessionId: string) => {
    console.log('Session selected:', sessionId);
    // Save current session before loading new one
    // TODO: Save current session
    // const { saveCurrentSession, loadSession } = useChatStore.getState();
    // saveCurrentSession();
    // Load session from the V2 store
    await useChatStore.getState().loadSession(sessionId);
  };
  
  const handleSessionDelete = async (sessionId: string) => {
    console.log('Deleting session:', sessionId);

    const wasActive = conversationId === sessionId;

    // Delete session from backend first
    await useChatStore.getState().deleteSession(sessionId);

    // If it was the active conversation, create a new one
    if (wasActive) {
      await handleNewChat();
    }
  };

  const handleNewChat = async () => {
    // Clear session and prepare for new conversation
    // Backend automatically persists sessions, no manual saving needed
    // Conversation will be created lazily when user sends first message
    await clearSession();
  };

  // Add tauri event listener for new chat
  useEffect(() => {
    if (isTauri()) {
      window.addEventListener('tauri-new-chat', handleNewChat);
      return () => {
        window.removeEventListener('tauri-new-chat', handleNewChat);
      };
    }
  }, []);

  return (
    <div className="h-screen flex flex-col bg-black text-zinc-100 overflow-hidden">
      {/* Header */}
      <header className="h-14 px-4 flex items-center justify-between bg-black/50 backdrop-blur-lg border-b border-zinc-800 flex-shrink-0">
        <div className="flex items-center gap-3">
          {/* Roampal Logo - Compass with memory trail */}
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            {/* Compass circle */}
            <circle cx="16" cy="16" r="13" stroke="#4A90E2" strokeWidth="1.5" opacity="0.3"/>

            {/* Compass cardinal points */}
            <line x1="16" y1="4" x2="16" y2="8" stroke="#4A90E2" strokeWidth="1.5" strokeLinecap="round"/>
            <line x1="16" y1="24" x2="16" y2="28" stroke="#4A90E2" strokeWidth="1.5" strokeLinecap="round"/>
            <line x1="4" y1="16" x2="8" y2="16" stroke="#4A90E2" strokeWidth="1.5" strokeLinecap="round"/>
            <line x1="24" y1="16" x2="28" y2="16" stroke="#4A90E2" strokeWidth="1.5" strokeLinecap="round"/>

            {/* Compass needle - north (blue) */}
            <path
              d="M 16 16 L 20 10 L 16 14 Z"
              fill="#4A90E2"
            />

            {/* Compass needle - south (green for memory) */}
            <path
              d="M 16 16 L 12 22 L 16 18 Z"
              fill="#10B981"
              opacity="0.7"
            />

            {/* Center pivot */}
            <circle cx="16" cy="16" r="2" fill="#F59E0B"/>
            <circle cx="16" cy="16" r="1" fill="#FBBF24"/>
          </svg>
          <span className="text-xl font-light tracking-wide text-zinc-100">
            Roampal
          </span>
          {/* Removed ProcessingIndicator from header - should be in chat area */}
        </div>
        <div className="flex items-center gap-3">
          <ConnectionStatus status={connectionStatus} />
        </div>
      </header>
      
      {/* Main content with draggable panels */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar - draggable with collapse support */}
        {!leftPane.isCollapsed && (
          <div
            className="flex-shrink-0 overflow-hidden bg-zinc-950"
            style={{
              width: `${leftPane.width}px`,
              transition: 'none' // Remove transition for smoother dragging
            }}
          >
            <div className="h-full">
              <Sidebar
                activeShard={'loopsmith'}
                activeSessionId={sessionId}
                availableShards={[]}
                onShardChange={handleShardChange}
                onNewChat={hasChatModel ? handleNewChat : undefined}
                hasChatModel={hasChatModel}
                onSelectChat={handleSessionSelect}
                onDeleteChat={handleSessionDelete}
                onMemoryPanel={() => {
                  setShowMemoryStats(!showMemoryStats);
                }}
                onManageShards={() => {
                  setShowBookProcessor(true);
                }}
                onSettings={() => setShowSettings(true)}
                onCollapse={leftPane.toggleCollapsed}
                onPersonalityCustomizer={() => setShowPersonalityCustomizer(true)}
                // Image Gallery removed
              />
            </div>
          </div>
        )}

        {/* Resize handle for LEFT sidebar - always visible for consistency */}
        <div
          onMouseDown={leftPane.onMouseDown}
          onDoubleClick={() => leftPane.toggleCollapsed()}
          className={`relative flex-shrink-0 h-full cursor-col-resize group ${
            leftPane.isResizing ? 'bg-blue-500' : 'hover:bg-zinc-600'
          } transition-colors`}
          style={{ width: '5px', backgroundColor: leftPane.isResizing ? '#3B82F6' : '#27272A' }}
          title="Drag to resize • Double-click to toggle sidebar"
        >
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col gap-1 pointer-events-none">
            <div className="w-1 h-1 bg-zinc-500 rounded-full group-hover:bg-zinc-300" />
            <div className="w-1 h-1 bg-zinc-500 rounded-full group-hover:bg-zinc-300" />
            <div className="w-1 h-1 bg-zinc-500 rounded-full group-hover:bg-zinc-300" />
          </div>
        </div>
          {/* Main chat column */}
          <main
            className="relative flex-1 flex flex-col h-full min-w-0 bg-black"
          >
          {/* Conversation header with toggle buttons */}
          <div className="h-12 px-4 flex items-center justify-between border-b border-zinc-800 flex-shrink-0 relative">
            <div className="flex items-center gap-2">
              {leftPane.isCollapsed && (
                <button
                  onClick={leftPane.toggleCollapsed}
                  className="p-2 bg-zinc-900 hover:bg-zinc-800 rounded-lg transition-colors"
                  title="Open Conversations"
                >
                  <ChatBubbleLeftRightIcon className="w-4 h-4 text-zinc-400" />
                </button>
              )}
            </div>
            
            {/* Model Selector - Compact Combo */}
            <div className="flex items-center gap-2">
              <div className="relative model-dropdown-container">
                <button
                    onClick={() => setShowModelDropdown(!showModelDropdown)}
                  className="flex items-center gap-2 px-2 sm:px-3 py-1.5 text-xs sm:text-sm bg-zinc-800/50 hover:bg-zinc-700/50 border border-zinc-700/50 hover:border-zinc-600 rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-zinc-500/30 min-w-[140px] max-w-[200px]"
                    disabled={isSwitchingModel}
                    title="Switch or install models"
                  >
                    {isSwitchingModel ? (
                      <div className="flex items-center gap-2 w-full">
                        <div className="w-3 h-3 border-2 border-blue-500 border-t-transparent rounded-full animate-spin flex-shrink-0" />
                        <span className="text-zinc-400 truncate">Switching...</span>
                      </div>
                    ) : (
                      <div className="flex items-center gap-1.5 w-full">
                        <svg className="w-3.5 h-3.5 text-zinc-400 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M10.394 2.08a1 1 0 00-.788 0l-7 3a1 1 0 000 1.84L5.25 8.051a.999.999 0 01.356-.257l4-1.714a1 1 0 11.788 1.838L7.667 9.088l1.94.831a1 1 0 00.787 0l7-3a1 1 0 000-1.838l-7-3zM3.31 9.397L5 10.12v4.102a8.969 8.969 0 00-1.05-.174 1 1 0 01-.89-.89 11.115 11.115 0 01.25-3.762zM9.3 16.573A9.026 9.026 0 007 14.935v-3.957l1.818.78a3 3 0 002.364 0l5.508-2.361a11.026 11.026 0 01.25 3.762 1 1 0 01-.89.89 8.968 8.968 0 00-5.35 2.524 1 1 0 01-1.4 0zM6 18a1 1 0 001-1v-2.065a8.935 8.935 0 00-2-.712V17a1 1 0 001 1z"/>
                        </svg>
                        <span className="text-zinc-300 truncate flex-1 text-left" title={selectedModel}>
                          {getModelOptions().length === 0 ? 'Download Your First Model' : selectedModel.split(':')[0].replace('-', ' ')}
                        </span>
                        <svg className={`w-3.5 h-3.5 text-zinc-500 transition-transform flex-shrink-0 ${showModelDropdown ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                        </svg>
                      </div>
                    )}
                </button>
              
                {showModelDropdown && (
                  <div className="absolute top-full mt-1 right-0 w-[280px] sm:w-[320px] bg-gradient-to-b from-zinc-900 to-zinc-950 border border-zinc-700 rounded-xl shadow-2xl z-[100] overflow-hidden">
                    {/* Installed Models Section */}
                    <div className="px-3 py-2 bg-zinc-800/50 border-b border-zinc-700">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-medium text-zinc-400">Installed Models</span>
                        <span className="text-xs text-zinc-500">{getModelOptions().length}</span>
                      </div>
                    </div>
                    <div className="max-h-[280px] overflow-y-auto custom-scrollbar p-1">
                      {getModelOptions().map(option => (
                        <div
                          key={option.value}
                          className={`group flex items-center justify-between px-3 py-2.5 mx-1 my-0.5 rounded-lg hover:bg-zinc-800/70 transition-all cursor-pointer ${
                            selectedModel === option.value ? 'bg-zinc-800 border border-blue-500/30' : ''
                          }`}
                        >
                          <button
                            onClick={() => {
                              switchModel(option.value);
                              setShowModelDropdown(false);
                            }}
                            className="flex-1 text-left min-w-0"
                            disabled={isSwitchingModel}
                          >
                            <div className="flex items-center justify-between">
                              <div className="min-w-0 flex-1">
                                <div className="text-sm font-medium text-zinc-200 group-hover:text-white truncate">{option.label}</div>
                                <div className="text-xs text-zinc-500 mt-0.5 truncate">{option.description}</div>
                              </div>
                              {selectedModel === option.value && (
                                <div className="ml-2 flex-shrink-0">
                                  <div className="w-5 h-5 bg-blue-500/20 rounded-full flex items-center justify-center">
                                    <svg className="w-3 h-3 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                    </svg>
                                  </div>
                                </div>
                              )}
                            </div>
                          </button>
                          {/* Uninstall button - don't show for default model */}
                          {(
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                handleUninstallModel(option.value);
                              }}
                              className="ml-2 p-1.5 text-zinc-600 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all opacity-0 group-hover:opacity-100"
                              title={`Uninstall ${option.label}`}
                            >
                              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                            </button>
                          )}
                        </div>
                      ))}
                    </div>

                    {/* Install New Models Section */}
                    <div className="border-t border-zinc-700">
                      <button
                        onClick={() => {
                          setShowModelDropdown(false);
                          fetchModels();
                          setShowModelInstallModal(true);
                        }}
                        className="w-full px-3 py-2.5 text-left flex items-center gap-2 hover:bg-zinc-800/50 transition-all group"
                      >
                        <div className="w-7 h-7 rounded-lg bg-zinc-800 group-hover:bg-zinc-700 flex items-center justify-center transition-colors">
                          <svg className="w-4 h-4 text-zinc-400 group-hover:text-zinc-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                          </svg>
                        </div>
                        <div className="flex-1">
                          <div className="text-sm font-medium text-zinc-300 group-hover:text-white">Install New Model</div>
                          <div className="text-xs text-zinc-500">Browse available models</div>
                        </div>
                        <svg className="w-4 h-4 text-zinc-600 group-hover:text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              {rightPane.isCollapsed && (
                <button
                  onClick={rightPane.toggleCollapsed}
                  className="p-2 bg-zinc-900 hover:bg-zinc-800 rounded-lg transition-colors"
                  title="Open Memory Panel"
                >
                  <CircleStackIcon className="w-4 h-4 text-zinc-400" />
                </button>
              )}
            </div>
          </div>
          
          {/* Messages - terminal style in scrollable region with fixed scroll button */}
          <div className="relative flex-1 min-h-0">
            <div
              ref={messagesContainerRef}
              className="h-full overflow-y-auto overflow-x-hidden"
              onScroll={handleScroll}
            >
              {getModelOptions().length === 0 ? (
                <div className="h-full flex items-center justify-center px-6">
                  <div className="max-w-md text-center space-y-6">
                    <div className="w-20 h-20 mx-auto bg-zinc-900 rounded-2xl flex items-center justify-center">
                      <svg className="w-10 h-10 text-zinc-600" fill="currentColor" viewBox="0 0 20 20">
                        <path d="M10.394 2.08a1 1 0 00-.788 0l-7 3a1 1 0 000 1.84L5.25 8.051a.999.999 0 01.356-.257l4-1.714a1 1 0 11.788 1.838L7.667 9.088l1.94.831a1 1 0 00.787 0l7-3a1 1 0 000-1.838l-7-3zM3.31 9.397L5 10.12v4.102a8.969 8.969 0 00-1.05-.174 1 1 0 01-.89-.89 11.115 11.115 0 01.25-3.762zM9.3 16.573A9.026 9.026 0 007 14.935v-3.957l1.818.78a3 3 0 002.364 0l5.508-2.361a11.026 11.026 0 01.25 3.762 1 1 0 01-.89.89 8.968 8.968 0 00-5.35 2.524 1 1 0 01-1.4 0zM6 18a1 1 0 001-1v-2.065a8.935 8.935 0 00-2-.712V17a1 1 0 001 1z"/>
                      </svg>
                    </div>
                    <div className="space-y-2">
                      <h3 className="text-xl font-semibold text-zinc-200">No Model Installed</h3>
                      <p className="text-sm text-zinc-400 leading-relaxed">
                        You need to download an AI model to start chatting. Choose from powerful models like Llama or Qwen.
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        fetchModels();
                        setShowModelInstallModal(true);
                      }}
                      className="inline-flex items-center gap-2 px-6 py-2.5 bg-blue-600/10 hover:bg-blue-600/20 border border-blue-600/30 text-blue-400 rounded-lg transition-all font-medium"
                    >
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                      </svg>
                      Install Your First Model
                    </button>
                  </div>
                </div>
              ) : (
                <TerminalMessageThread
                  messages={componentMessages}
                  activeShard={'loopsmith'}
                  onMemoryClick={handleMemoryClick}
                  onCommandClick={handleCommandClick}
                  isProcessing={isProcessing}
                  processingStage={processingStage}
                  processingStatus={processingStatus}
                />
              )}
            </div>

            {/* Scroll to bottom button - positioned relative to parent container */}
            {showScrollButton && (
              <button
                onClick={scrollToBottom}
                className="absolute bottom-4 right-4 z-40 bg-gray-900 hover:bg-black text-white rounded-full p-2 shadow-lg transition-all duration-200 hover:scale-110"
                title="Scroll to bottom"
              >
                <svg
                  className="w-4 h-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 14l-7 7m0 0l-7-7m7 7V3"
                  />
                </svg>
              </button>
            )}
          </div>

          {/* Processing steps are now shown inline in the chat */}

          {/* Input composer with voice and attachments */}
          <div className="px-6 pb-4 flex-shrink-0">
            <ConnectedCommandInput hasChatModel={hasChatModel} />
          </div>
        </main>
        
        {/* Resize handle between chat and right memory panel - made wider for easier grabbing */}
        <div
          onMouseDown={rightPane.onMouseDown}
          onDoubleClick={() => rightPane.toggleCollapsed()}
          className={`relative flex-shrink-0 h-full cursor-col-resize group ${
            rightPane.isResizing ? 'bg-blue-500' : 'hover:bg-zinc-600'
          } transition-colors`}
          style={{ width: '5px', backgroundColor: rightPane.isResizing ? '#3B82F6' : '#27272A' }}
          title="Drag to resize • Double-click to collapse"
        >
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col gap-1 pointer-events-none">
            <div className="w-1 h-1 bg-zinc-500 rounded-full group-hover:bg-zinc-300" />
            <div className="w-1 h-1 bg-zinc-500 rounded-full group-hover:bg-zinc-300" />
            <div className="w-1 h-1 bg-zinc-500 rounded-full group-hover:bg-zinc-300" />
          </div>
        </div>
        
        {/* Right Memory Panel - separate and independent */}
        {!rightPane.isCollapsed ? (
          <div
            className="relative flex-shrink-0 h-full overflow-hidden shadow-xl border-l border-zinc-800 z-30"
            style={{
              width: `${rightPane.width}px`,
              transition: 'none' // Remove transition for smoother dragging
            }}
          >
            <div className="h-full max-h-full flex flex-col overflow-hidden">
              <MemoryPanelV2
                memories={activeMemories}
                knowledgeGraph={knowledgeGraphData}
                onMemoryClick={handleMemoryClick}
                onClose={() => rightPane.toggleCollapsed()}
                onRefresh={() => {
                  fetchMemories(true);
                  fetchKnowledgeGraph();
                }}
                isRefreshing={isRefreshingMemories}
                lastRefresh={lastMemoryRefresh}
                currentUserId={'default'}
                activeShard={'loopsmith'}
              />
            </div>
          </div>
        ) : null}
      </div>
      
      
      
      {/* Settings Modal */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />

      {/* Book Processor Modal */}
      <BookProcessorModal
        isOpen={showBookProcessor}
        onClose={() => setShowBookProcessor(false)}
      />

      {/* Personality Customizer Modal */}
      {showPersonalityCustomizer && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
          <div className="relative w-full max-w-3xl h-[90vh] bg-zinc-950 rounded-lg shadow-2xl overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-4 h-12 bg-zinc-900 border-b border-zinc-800 flex-shrink-0">
              <h2 className="text-sm font-semibold text-zinc-100">Personality & Identity</h2>
              <button
                onClick={() => setShowPersonalityCustomizer(false)}
                className="p-1.5 text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 rounded transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <PersonalityCustomizer apiBase={ROAMPAL_CONFIG.apiBase} />
            </div>
          </div>
        </div>
      )}

      {/* Memory Stats Panel */}
      <MemoryStatsPanel
        isOpen={showMemoryStats}
        onClose={() => setShowMemoryStats(false)}
      />

      {/* Action Status Display */}
      {/* Removed duplicate ActionStatus popup - already shown inline in chat */}

      {/* Developer Panel - Neural AI Feature Control */}
      <DevPanel
        isOpen={showDevPanel}
        onClose={() => setShowDevPanel(false)}
      />


      {/* Model Installation Modal */}
      {showModelInstallModal && (
        <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center z-50 animate-fadeIn">
          <div className="bg-gradient-to-b from-zinc-900 to-zinc-950 border border-zinc-800 rounded-xl w-full max-w-3xl mx-4 max-h-[85vh] flex flex-col shadow-2xl">
            {/* Fixed Header */}
            <div className="flex items-center justify-between p-6 border-b border-zinc-800">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                  <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 19l3 3m0 0l3-3m-3 3V10" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-white">Model Library</h2>
                  <p className="text-xs text-zinc-400 mt-0.5">Install and manage AI models</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                {/* Refresh button */}
                <button
                  onClick={() => {
                    fetchModels();
                    setInstallProgress('Refreshing model list...');
                    setTimeout(() => setInstallProgress(''), 1000);
                  }}
                  className="p-2 hover:bg-zinc-800/50 rounded-lg transition-all hover:scale-105"
                  title="Refresh model list"
                  disabled={!!installingModelName}
                >
                  <svg className="w-4 h-4 text-zinc-400 hover:text-zinc-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
                {/* Close button */}
                <button
                  onClick={() => {
                    setShowModelInstallModal(false);
                  }}
                  className="p-2 hover:bg-red-500/10 rounded-lg transition-all hover:scale-105 group"
                  title={installingModelName ? "Close (download continues in background)" : "Close"}
                >
                  <svg className="w-5 h-5 text-zinc-400 group-hover:text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Scrollable Content */}
            <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
              {installProgress && (
                <div className="mb-6 p-4 bg-gradient-to-r from-blue-500/10 to-blue-600/10 border border-blue-500/20 rounded-xl backdrop-blur-sm">
                  <div className="text-sm text-zinc-300">{installProgress}</div>
                  {installingModelName && (
                    <div className="mt-2">
                      <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-blue-600 h-2 rounded-full transition-all duration-500 ease-out animate-pulse"
                          style={{ width: `${downloadProgress}%` }}
                        ></div>
                      </div>
                      {downloadProgress < 100 && downloadAbortController && (
                        <button
                          onClick={() => setShowCancelDownloadConfirm(true)}
                          className="mt-3 px-4 py-1.5 bg-red-600/10 hover:bg-red-600/20 border border-red-600/30 text-red-400 rounded-lg transition-colors text-xs font-medium"
                        >
                          Cancel Download
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="space-y-3">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-medium text-white">Available Models</h3>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                    <span className="text-xs text-zinc-400">Connected to Ollama</span>
                  </div>
                </div>
                
                <div className="space-y-3 mb-6">
                  {/* Size & VRAM Guide */}
                  <div className="text-sm text-zinc-300 p-4 bg-zinc-800/50 rounded-xl border border-zinc-700/50 backdrop-blur-sm">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-6 h-6 bg-yellow-500/20 rounded-full flex items-center justify-center">
                        <span className="text-yellow-500 text-xs">!</span>
                      </div>
                      <span className="font-medium text-white">Size & VRAM Guide</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 ml-8 text-xs">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                        <span><strong className="text-zinc-200">&lt;5GB:</strong> 6-8GB VRAM</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                        <span><strong className="text-zinc-200">5-10GB:</strong> 12-16GB VRAM</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-purple-500 rounded-full"></span>
                        <span><strong className="text-zinc-200">10-30GB:</strong> 24GB VRAM</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                        <span><strong className="text-zinc-200">30GB+:</strong> 48GB+ VRAM</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                {curatedModels.map(category => (
                  <div key={category.category}>
                    {/* Category Header */}
                    <div className="mb-3 p-3 bg-zinc-800/40 rounded-lg border border-zinc-700/50">
                      <div className="flex items-center gap-2 mb-1">
                        {category.icon === 'sparkles' ? (
                          <svg className="w-5 h-5 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
                          </svg>
                        ) : (
                          <svg className="w-5 h-5 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                          </svg>
                        )}
                        <h3 className="font-semibold text-white">{category.category}</h3>
                        {category.category === 'Premium Models' && (
                          <span className="ml-auto px-2 py-0.5 bg-green-500/20 text-green-400 rounded-full text-xs font-medium">
                            Recommended
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-zinc-400 ml-7">{category.description}</p>
                    </div>

                    {/* Models in Category */}
                    <div className="space-y-2 ml-3">
                      {category.models.map(model => {
                        const isAlreadyInstalled = availableModels.includes(model.name);
                        return (
                          <div
                            key={model.name}
                            className={`group flex items-center justify-between p-4 bg-zinc-800/30 border rounded-xl hover:bg-zinc-800/50 transition-all duration-200 ${
                              model.agentCapable
                                ? 'border-green-700/30 hover:border-green-600/50'
                                : 'border-zinc-700/50 hover:border-zinc-600/50'
                            }`}
                          >
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center gap-2 mb-1 flex-wrap">
                                <span className="font-medium text-white text-sm truncate group-hover:text-blue-400 transition-colors">
                                  {model.name}
                                </span>
                                <span className="text-xs px-2 py-0.5 bg-zinc-700/50 text-zinc-300 rounded-full whitespace-nowrap">
                                  {model.size}
                                </span>
                                {model.agentCapable && (
                                  <span className="text-xs px-2 py-0.5 bg-green-600/20 text-green-400 rounded-full whitespace-nowrap flex items-center gap-1">
                                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                      <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                    </svg>
                                    Premium
                                  </span>
                                )}
                                {model.tokens && (
                                  <span className="text-xs px-2 py-0.5 bg-blue-600/20 text-blue-400 rounded-full whitespace-nowrap">
                                    {model.tokens >= 1000 ? `${Math.floor(model.tokens/1000)}K` : model.tokens} tokens
                                  </span>
                                )}
                              </div>
                              <p className="text-xs text-zinc-500 line-clamp-2 mt-1">{model.description}</p>
                            </div>
                            <div className="flex items-center gap-2 ml-3">
                              {isAlreadyInstalled ? (
                                <>
                                  <span className="px-3 py-1.5 bg-green-500/20 text-green-400 rounded-lg text-xs font-medium flex items-center gap-1">
                                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                    </svg>
                                    Installed
                                  </span>
                                  <button
                                    onClick={() => handleUninstallModel(model.name)}
                                    disabled={!!installingModelName}
                                    className="p-1.5 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-all hover:scale-110"
                                    title={`Uninstall ${model.name}`}
                                  >
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                    </svg>
                                  </button>
                                </>
                              ) : (
                                <button
                                  onClick={() => handleInstallModel(model.name)}
                                  disabled={!!installingModelName || installingModelName === model.name}
                                  className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                                    installingModelName === model.name
                                      ? 'bg-zinc-700/50 text-zinc-500 cursor-not-allowed'
                                      : model.agentCapable
                                        ? 'bg-green-600 hover:bg-green-700 text-white shadow-sm hover:shadow-md'
                                        : 'bg-blue-600 hover:bg-blue-700 text-white shadow-sm hover:shadow-md'
                                  }`}
                                >
                                  {installingModelName === model.name ? 'Installing...' : 'Install'}
                                </button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
                </div>

                <div className="mt-6 pt-4 border-t border-zinc-800/50">
                  <div className="flex items-start gap-2">
                    <svg className="w-4 h-4 text-zinc-500 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-xs text-zinc-500">
                      Models are downloaded from Ollama Hub. Larger models provide better quality but require more VRAM.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Installation Progress Popup - Shows only for the model being installed */}
      {showInstallPopup && installingModelName && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-6 w-96 shadow-2xl">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-semibold text-zinc-200">Installing Model</h3>
            <button
              onClick={() => setShowInstallPopup(false)}
              className="p-1 hover:bg-zinc-800 rounded-lg transition-colors"
              title="Close (download continues in background)"
            >
              <svg className="w-4 h-4 text-zinc-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-zinc-400">Model:</span>
              <span className="text-sm text-zinc-200 font-medium">{installingModelName}</span>
            </div>

            <div className="text-sm text-zinc-300">{installProgress}</div>

            {downloadProgress > 0 && (
              <div>
                <div className="flex justify-between text-xs text-zinc-400 mb-1">
                  <span>Progress</span>
                  <span>{Math.floor(downloadProgress)}%</span>
                </div>
                <div className="w-full bg-zinc-700 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-500 ease-out"
                    style={{ width: `${downloadProgress}%` }}
                  />
                </div>
              </div>
            )}

            {downloadProgress < 100 && downloadAbortController && (
              <button
                onClick={() => setShowCancelDownloadConfirm(true)}
                className="w-full mt-3 px-4 py-2 bg-red-600/10 hover:bg-red-600/20 border border-red-600/30 text-red-400 rounded-lg transition-colors text-sm font-medium"
              >
                Cancel Download
              </button>
            )}

            <div className="text-xs text-zinc-500 mt-2">
              Installation continues in background if you close this window
            </div>
          </div>
          </div>
        </div>
      )}

      {/* Uninstall Confirmation Modal */}
      {uninstallConfirmModel && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-red-600/10 flex items-center justify-center">
                <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-lg font-semibold text-zinc-100 mb-2">Uninstall Model?</h3>
                <p className="text-sm text-zinc-400 mb-1">
                  Are you sure you want to uninstall <span className="font-mono text-zinc-300">{uninstallConfirmModel}</span>?
                </p>
                <p className="text-sm text-zinc-500">
                  This will permanently delete the model from your system.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 mt-6">
              <button
                onClick={() => setUninstallConfirmModel(null)}
                className="flex-1 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg transition-colors font-medium text-sm"
              >
                Cancel
              </button>
              <button
                onClick={confirmUninstall}
                className="flex-1 px-4 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors font-medium text-sm"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Cancel Download Confirmation Modal */}
      {showCancelDownloadConfirm && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-yellow-600/10 flex items-center justify-center">
                <svg className="w-6 h-6 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-lg font-semibold text-zinc-100 mb-2">Cancel Download?</h3>
                <p className="text-sm text-zinc-400 mb-1">
                  Are you sure you want to cancel this download?
                </p>
                <p className="text-sm text-zinc-500">
                  The model will not be installed and any partial download will be discarded.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 mt-6">
              <button
                onClick={() => setShowCancelDownloadConfirm(false)}
                className="flex-1 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg transition-colors font-medium text-sm"
              >
                Continue Download
              </button>
              <button
                onClick={confirmCancelDownload}
                className="flex-1 px-4 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors font-medium text-sm"
              >
                Cancel Download
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Model Switch Confirmation Modal */}
      {modelSwitchPending && (
        <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 backdrop-blur-sm">
          <div className="bg-zinc-900 border border-zinc-800 rounded-xl shadow-2xl p-6 max-w-md w-full mx-4">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-600/10 flex items-center justify-center">
                <svg className="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-lg font-semibold text-zinc-100 mb-2">Switch Model Mid-Conversation?</h3>
                <p className="text-sm text-zinc-400 mb-3">
                  Switching models mid-conversation may result in different response styles or behavior.
                </p>
                <div className="space-y-1.5 text-sm">
                  <div className="flex items-center gap-2">
                    <span className="text-zinc-500">Current:</span>
                    <span className="font-mono text-zinc-300">{selectedModel}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-zinc-500">New:</span>
                    <span className="font-mono text-zinc-300">{modelSwitchPending}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3 mt-6">
              <button
                onClick={() => setModelSwitchPending(null)}
                className="flex-1 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-200 rounded-lg transition-colors font-medium text-sm"
              >
                Cancel
              </button>
              <button
                onClick={confirmModelSwitch}
                className="flex-1 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors font-medium text-sm"
              >
                Switch Model
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Ollama Required Modal */}
      <OllamaRequiredModal
        isOpen={showOllamaRequired}
        onRetry={handleOllamaRetry}
      />

    </div>
  );
};
