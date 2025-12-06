import React, { useState, useEffect } from 'react';
import { apiFetch } from '../utils/fetch';
import { Toast } from './Toast';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface MCPServer {
  name: string;
  command: string;
  args: string[];
  enabled: boolean;
  status: 'connected' | 'disconnected' | 'error';
  toolCount: number;
  tools: string[];
  lastError?: string;
}

interface PopularServer {
  name: string;
  command: string;
  args: string[];
  description: string;
}

interface MCPServersPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ToastState {
  message: string;
  type: 'success' | 'error' | 'info';
}

export const MCPServersPanel: React.FC<MCPServersPanelProps> = ({ isOpen, onClose }) => {
  const [servers, setServers] = useState<MCPServer[]>([]);
  const [popularServers, setPopularServers] = useState<PopularServer[]>([]);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastState | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newServer, setNewServer] = useState({ name: '', command: '', args: '' });

  // Fetch servers when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchServers();
      fetchPopularServers();
    }
  }, [isOpen]);

  const fetchServers = async () => {
    setLoading(true);
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/servers`);
      if (response.ok) {
        const data = await response.json();
        setServers(data.servers || []);
      }
    } catch (error) {
      console.error('Failed to fetch MCP servers:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPopularServers = async () => {
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/popular`);
      if (response.ok) {
        const data = await response.json();
        setPopularServers(data.servers || []);
      }
    } catch (error) {
      console.error('Failed to fetch popular servers:', error);
    }
  };

  const handleAddServer = async () => {
    if (!newServer.name || !newServer.command) {
      setToast({ message: 'Name and command are required', type: 'error' });
      return;
    }

    setActionLoading('add');
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/servers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newServer.name,
          command: newServer.command,
          args: newServer.args.split(' ').filter(a => a.trim()),
        }),
      });

      if (response.ok) {
        setToast({ message: `Server "${newServer.name}" added`, type: 'success' });
        setShowAddModal(false);
        setNewServer({ name: '', command: '', args: '' });
        await fetchServers();
      } else {
        const error = await response.json();
        setToast({ message: error.detail || 'Failed to add server', type: 'error' });
      }
    } catch (error) {
      setToast({ message: 'Failed to add server', type: 'error' });
    } finally {
      setActionLoading(null);
    }
  };

  const handleAddPopular = async (server: PopularServer) => {
    // Check if already added
    if (servers.some(s => s.name === server.name)) {
      setToast({ message: `"${server.name}" is already configured`, type: 'info' });
      return;
    }

    setActionLoading(server.name);
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/servers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: server.name,
          command: server.command,
          args: server.args,
        }),
      });

      if (response.ok) {
        setToast({ message: `Server "${server.name}" added`, type: 'success' });
        await fetchServers();
      } else {
        const error = await response.json();
        setToast({ message: error.detail || 'Failed to add server', type: 'error' });
      }
    } catch (error) {
      setToast({ message: 'Failed to add server', type: 'error' });
    } finally {
      setActionLoading(null);
    }
  };

  const handleRemoveServer = async (name: string) => {
    setActionLoading(name);
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/servers/${name}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        setToast({ message: `Server "${name}" removed`, type: 'success' });
        await fetchServers();
      } else {
        const error = await response.json();
        setToast({ message: error.detail || 'Failed to remove server', type: 'error' });
      }
    } catch (error) {
      setToast({ message: 'Failed to remove server', type: 'error' });
    } finally {
      setActionLoading(null);
    }
  };

  const handleTestConnection = async (name: string) => {
    setActionLoading(name);
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/servers/${name}/test`, {
        method: 'POST',
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setToast({ message: data.message, type: 'success' });
        } else {
          setToast({ message: data.message, type: 'error' });
        }
        await fetchServers();
      }
    } catch (error) {
      setToast({ message: 'Connection test failed', type: 'error' });
    } finally {
      setActionLoading(null);
    }
  };

  const handleReconnect = async (name: string) => {
    setActionLoading(name);
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/servers/${name}/reconnect`, {
        method: 'POST',
      });

      if (response.ok) {
        setToast({ message: `Reconnected to "${name}"`, type: 'success' });
        await fetchServers();
      }
    } catch (error) {
      setToast({ message: 'Reconnection failed', type: 'error' });
    } finally {
      setActionLoading(null);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'bg-green-500';
      case 'disconnected':
        return 'bg-yellow-500';
      default:
        return 'bg-red-500';
    }
  };

  if (!isOpen) return null;

  return (
    <>
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
      <div
        className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        <div
          className="bg-zinc-900 rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto border border-zinc-800"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex justify-between items-center p-6 border-b border-zinc-800">
            <div>
              <h2 className="text-xl font-bold">MCP Tool Servers</h2>
              <p className="text-sm text-zinc-400 mt-1">Connect external tools (Blender, filesystem, git, etc.)</p>
            </div>
            <button
              onClick={onClose}
              className="p-2 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 rounded-lg transition-colors"
              title="Close"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            {/* Connected Servers */}
            <div>
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-sm font-medium text-zinc-400">Configured Servers</h3>
                <button
                  onClick={fetchServers}
                  disabled={loading}
                  className="text-xs px-2 py-1 bg-zinc-800 hover:bg-zinc-700 rounded transition-colors disabled:opacity-50"
                >
                  {loading ? 'Loading...' : 'Refresh'}
                </button>
              </div>

              {servers.length === 0 ? (
                <div className="text-center py-8 text-zinc-500 bg-zinc-800/30 rounded-lg">
                  <svg className="w-8 h-8 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                  </svg>
                  <p className="text-sm">No MCP servers configured</p>
                  <p className="text-xs text-zinc-600 mt-1">Use Add Custom Server below</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {servers.map((server) => (
                    <div
                      key={server.name}
                      className="flex items-center justify-between p-4 bg-zinc-800/50 rounded-lg border border-zinc-700/50"
                    >
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <div className={`w-3 h-3 rounded-full flex-shrink-0 ${getStatusColor(server.status)}`} />
                        <div className="min-w-0">
                          <p className="font-medium truncate">{server.name}</p>
                          <p className="text-xs text-zinc-500">
                            {server.status === 'connected'
                              ? `${server.toolCount} tools available`
                              : server.lastError || server.status}
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-2 flex-shrink-0">
                        {server.status !== 'connected' && (
                          <button
                            onClick={() => handleReconnect(server.name)}
                            disabled={actionLoading === server.name}
                            className="px-3 py-1.5 text-xs bg-blue-600 hover:bg-blue-500 rounded transition-colors disabled:opacity-50"
                          >
                            {actionLoading === server.name ? '...' : 'Reconnect'}
                          </button>
                        )}
                        {server.status === 'connected' && (
                          <button
                            onClick={() => handleTestConnection(server.name)}
                            disabled={actionLoading === server.name}
                            className="px-3 py-1.5 text-xs bg-zinc-700 hover:bg-zinc-600 rounded transition-colors disabled:opacity-50"
                          >
                            {actionLoading === server.name ? '...' : 'Test'}
                          </button>
                        )}
                        <button
                          onClick={() => handleRemoveServer(server.name)}
                          disabled={actionLoading === server.name}
                          className="p-1.5 text-zinc-400 hover:text-red-400 hover:bg-zinc-700 rounded transition-colors disabled:opacity-50"
                          title="Remove server"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Popular Servers - only show if any exist */}
            {popularServers.length > 0 && (
              <div>
                <h3 className="text-sm font-medium text-zinc-400 mb-3">Popular Servers</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {popularServers.map((server) => {
                    const isAdded = servers.some(s => s.name === server.name);
                    return (
                      <div
                        key={server.name}
                        className="flex items-center justify-between p-3 bg-zinc-800/30 rounded-lg border border-zinc-700/30"
                      >
                        <div className="min-w-0 flex-1">
                          <p className="font-medium text-sm truncate">{server.name}</p>
                          <p className="text-xs text-zinc-500 truncate">{server.description}</p>
                        </div>
                        <button
                          onClick={() => handleAddPopular(server)}
                          disabled={isAdded || actionLoading === server.name}
                          className={`px-3 py-1.5 text-xs rounded transition-colors flex-shrink-0 ml-2 ${
                            isAdded
                              ? 'bg-zinc-700 text-zinc-500 cursor-not-allowed'
                              : 'bg-blue-600 hover:bg-blue-500 disabled:opacity-50'
                          }`}
                        >
                          {isAdded ? 'Added' : actionLoading === server.name ? '...' : 'Add'}
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Add Custom Server */}
            <div className="pt-4 border-t border-zinc-800">
              <button
                onClick={() => setShowAddModal(true)}
                className="w-full px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm text-zinc-300 transition-colors flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                Add Custom Server
              </button>
            </div>

            {/* Info box */}
            <div className="p-4 bg-purple-900/20 border border-purple-800/30 rounded-lg">
              <div className="flex items-start gap-3">
                <svg className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="text-sm text-purple-200/90">
                  <p className="font-medium mb-1">How MCP Tool Servers Work</p>
                  <ul className="space-y-1 text-xs text-purple-200/70">
                    <li>External tools run as separate processes</li>
                    <li>Roampal discovers and uses their capabilities</li>
                    <li>Requires npx/uvx and the server package installed</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Add Custom Server Modal */}
      {showAddModal && (
        <div
          className="fixed inset-0 bg-black/70 z-[60] flex items-center justify-center p-4"
          onClick={() => setShowAddModal(false)}
        >
          <div
            className="bg-zinc-900 rounded-lg p-6 max-w-md w-full border border-zinc-800"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold mb-4">Add Custom MCP Server</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-zinc-400 mb-1">Name</label>
                <input
                  type="text"
                  value={newServer.name}
                  onChange={(e) => setNewServer({ ...newServer, name: e.target.value })}
                  placeholder="e.g., my-tool"
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm text-zinc-400 mb-1">Command</label>
                <input
                  type="text"
                  value={newServer.command}
                  onChange={(e) => setNewServer({ ...newServer, command: e.target.value })}
                  placeholder="e.g., npx, uvx, python"
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm text-zinc-400 mb-1">Arguments (space-separated)</label>
                <input
                  type="text"
                  value={newServer.args}
                  onChange={(e) => setNewServer({ ...newServer, args: e.target.value })}
                  placeholder="e.g., -y @modelcontextprotocol/server-filesystem /"
                  className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>

            <div className="flex gap-2 mt-6">
              <button
                onClick={handleAddServer}
                disabled={actionLoading === 'add'}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm transition-colors disabled:opacity-50"
              >
                {actionLoading === 'add' ? 'Adding...' : 'Add Server'}
              </button>
              <button
                onClick={() => {
                  setShowAddModal(false);
                  setNewServer({ name: '', command: '', args: '' });
                }}
                className="flex-1 px-4 py-2 bg-zinc-700 hover:bg-zinc-600 rounded-lg text-sm transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
