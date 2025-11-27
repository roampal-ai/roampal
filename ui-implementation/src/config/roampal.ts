// Roampal Configuration
// Port configured via VITE_API_PORT env var (default: 8765 for prod, 8766 for dev)
const API_PORT = import.meta.env.VITE_API_PORT || '8765';

export const ROAMPAL_CONFIG = {
  name: 'Roampal',
  apiUrl: `http://localhost:${API_PORT}`,
  API_BASE: `http://localhost:${API_PORT}`,
  apiBase: `http://localhost:${API_PORT}`,
  WS_URL: `ws://localhost:${API_PORT}`,
  ENABLE_DEBUG_LOGGING: false,
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000,
  REQUEST_TIMEOUT: 30000,
  WS_MAX_RECONNECT_ATTEMPTS: 5,
  TRANSPORT: 'http',
  HEALTH: '/health',
  CHAT: {
    SEND: '/api/chat',
    STREAM: '/api/chat/stream',
    toString: () => '/api/chat'
  },
  MEMORY: {
    SEARCH: '/api/memory/search',
    ADD: '/api/memory/add',
    LIST: '/api/memory/list',
    AVAILABLE_SHARDS: '/api/memory/shards',
    toString: () => '/api/memory'
  },
  SHARDS: {
    SWITCH: '/api/shards/switch',
    LIST: '/api/shards/list',
    toString: () => '/api/shards'
  },
  endpoints: {
    chat: '/api/chat',
    execute: '/api/execute',
    file: '/api/file',
    project: '/api/analyze-project',
    test: '/api/test',
    git: '/api/git'
  },
  theme: {
    primary: '#4A90E2', // Blue - memory/intelligence
    secondary: '#10B981', // Green - learning/growth
    accent: '#F59E0B', // Amber - active/current
    dark: '#1a1a1a'
  },
  personality: 'Your AI that actually remembers'
};

// Tauri API integration
export const useTauri = () => {
  const isTauri = window.__TAURI__ !== undefined;

  const openInVSCode = async (path: string) => {
    if (isTauri) {
      const { invoke } = window.__TAURI__.tauri;
      return await invoke('open_in_vscode', { path });
    } else {
      console.log('Would open in VS Code:', path);
    }
  };

  const openFolderInVSCode = async (path: string) => {
    if (isTauri) {
      const { invoke } = window.__TAURI__.tauri;
      return await invoke('open_folder_in_vscode', { path });
    } else {
      console.log('Would open folder in VS Code:', path);
    }
  };

  const readFile = async (path: string) => {
    if (isTauri) {
      const { invoke } = window.__TAURI__.tauri;
      return await invoke('read_file', { path });
    } else {
      // Fallback to API
      const response = await fetch(`${ROAMPAL_CONFIG.apiUrl}/api/file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operation: 'read', path })
      });
      return await response.json();
    }
  };

  const writeFile = async (path: string, content: string) => {
    if (isTauri) {
      const { invoke } = window.__TAURI__.tauri;
      return await invoke('write_file', { path, content });
    } else {
      // Fallback to API
      const response = await fetch(`${ROAMPAL_CONFIG.apiUrl}/api/file`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ operation: 'write', path, content })
      });
      return await response.json();
    }
  };

  const listFiles = async (path: string) => {
    if (isTauri) {
      const { invoke } = window.__TAURI__.tauri;
      return await invoke('list_files', { path });
    } else {
      console.log('Would list files in:', path);
      return [];
    }
  };

  const runGitCommand = async (args: string[]) => {
    if (isTauri) {
      const { invoke } = window.__TAURI__.tauri;
      return await invoke('run_git_command', { args });
    } else {
      // Fallback to API
      const response = await fetch(`${ROAMPAL_CONFIG.apiUrl}/api/git`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: args[0], args: args.slice(1) })
      });
      return await response.json();
    }
  };

  return {
    isTauri,
    openInVSCode,
    openFolderInVSCode,
    readFile,
    writeFile,
    listFiles,
    runGitCommand
  };
};

// Window type augmentation for TypeScript
declare global {
  interface Window {
    __TAURI__?: any;
  }
}