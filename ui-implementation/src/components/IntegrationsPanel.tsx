import React, { useState, useEffect } from 'react';
import { apiFetch } from '../utils/fetch';
import { Toast } from './Toast';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface MCPTool {
  name: string;
  status: 'connected' | 'available' | 'not_installed';
  config_path?: string;
}

interface IntegrationsPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ToastState {
  message: string;
  type: 'success' | 'error' | 'info';
}

export const IntegrationsPanel: React.FC<IntegrationsPanelProps> = ({ isOpen, onClose }) => {
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastState | null>(null);
  const [showCustomPath, setShowCustomPath] = useState(false);
  const [customPath, setCustomPath] = useState('');
  const [hiddenTools, setHiddenTools] = useState<string[]>([]);
  const [showDiscoverMore, setShowDiscoverMore] = useState(false);

  // Scan for tools when modal opens
  useEffect(() => {
    if (isOpen) {
      scanForTools();
    }
  }, [isOpen]);

  // Load hidden tools from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('hiddenMCPTools');
    if (stored) {
      try {
        setHiddenTools(JSON.parse(stored));
      } catch (e) {
        console.error('Failed to load hidden tools:', e);
      }
    }
  }, []);

  // Save hidden tools to localStorage when changed
  useEffect(() => {
    localStorage.setItem('hiddenMCPTools', JSON.stringify(hiddenTools));
  }, [hiddenTools]);

  const scanForTools = async () => {
    setLoading(true);
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/scan`);
      if (response.ok) {
        const data = await response.json();
        setTools(data.tools || []);
      }
    } catch (error) {
      console.error('Failed to scan for MCP tools:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleConnect = async (configPath: string) => {
    setActionLoading(configPath);
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_path: configPath }),
      });

      if (response.ok) {
        const data = await response.json();
        setToast({ message: data.message, type: 'success' });
        setShowCustomPath(false);
        setCustomPath('');
        await scanForTools(); // Refresh list
      } else {
        const error = await response.json();
        setToast({ message: `Connection failed: ${error.detail}`, type: 'error' });
      }
    } catch (error) {
      console.error('Failed to connect:', error);
      setToast({ message: 'Connection failed. Please try again.', type: 'error' });
    } finally {
      setActionLoading(null);
    }
  };

  const handleDisconnect = async (configPath: string) => {
    setActionLoading(configPath);
    try {
      const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/mcp/disconnect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config_path: configPath }),
      });

      if (response.ok) {
        const data = await response.json();
        setToast({ message: data.message, type: 'success' });
        await scanForTools(); // Refresh list
      } else {
        const error = await response.json();
        setToast({ message: `Disconnection failed: ${error.detail}`, type: 'error' });
      }
    } catch (error) {
      console.error('Failed to disconnect:', error);
      setToast({ message: 'Disconnection failed. Please try again.', type: 'error' });
    } finally {
      setActionLoading(null);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'bg-green-500';
      case 'available':
        return 'bg-yellow-500';
      default:
        return 'bg-zinc-600';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'connected':
        return 'Connected';
      case 'available':
        return 'Available';
      default:
        return 'Not Installed';
    }
  };

  const handleHideTool = (toolName: string) => {
    setHiddenTools([...hiddenTools, toolName]);
    setToast({
      message: `${toolName} hidden from list. Find it in "Discover More Tools"`,
      type: 'info'
    });
  };

  const handleUnhideTool = (toolName: string) => {
    setHiddenTools(hiddenTools.filter(name => name !== toolName));
    setToast({
      message: `${toolName} added back to list`,
      type: 'success'
    });
  };

  // Filter tools into visible and hidden
  const visibleTools = tools.filter(tool => !hiddenTools.includes(tool.name));
  const hiddenToolsList = tools.filter(tool => hiddenTools.includes(tool.name));

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
            <h2 className="text-xl font-bold">Integrations</h2>
            <p className="text-sm text-zinc-400 mt-1">Connect Roampal memory to AI tools</p>
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
        <div className="p-6 space-y-4">
          {/* Scan button */}
          <div className="flex justify-between items-center">
            <p className="text-sm text-zinc-400">Available AI tools</p>
            <button
              onClick={scanForTools}
              disabled={loading}
              className="text-sm px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors disabled:opacity-50"
            >
              {loading ? 'Scanning...' : 'Refresh'}
            </button>
          </div>

          {/* Tools list */}
          {loading && visibleTools.length === 0 ? (
            <div className="text-center py-8 text-zinc-500">
              <svg className="w-8 h-8 mx-auto mb-2 animate-spin" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              <p className="text-sm">Scanning for tools...</p>
            </div>
          ) : visibleTools.length === 0 && hiddenToolsList.length === 0 ? (
            <div className="text-center py-8 text-zinc-500">
              <svg className="w-8 h-8 mx-auto mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-sm">No MCP-compatible tools detected</p>
              <p className="text-xs text-zinc-600 mt-1">Install an MCP-compatible tool to connect</p>
            </div>
          ) : (
            <>
              {visibleTools.length === 0 ? (
                <div className="text-center py-6 text-zinc-500">
                  <p className="text-sm">All tools hidden</p>
                  <p className="text-xs text-zinc-600 mt-1">Check "Discover More Tools" below</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {visibleTools.map((tool) => (
                    <div
                      key={tool.config_path}
                      className="flex items-center justify-between p-4 bg-zinc-800/50 rounded-lg border border-zinc-700/50"
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(tool.status)}`} />
                        <div>
                          <p className="font-medium">{tool.name}</p>
                          <p className="text-xs text-zinc-500">{getStatusText(tool.status)}</p>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        {tool.status === 'available' && (
                          <button
                            onClick={() => handleConnect(tool.config_path!)}
                            disabled={actionLoading === tool.config_path}
                            className="px-4 py-2 text-sm bg-blue-600 hover:bg-blue-500 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {actionLoading === tool.config_path ? 'Connecting...' : 'Connect'}
                          </button>
                        )}

                        {tool.status === 'connected' && (
                          <button
                            onClick={() => handleDisconnect(tool.config_path!)}
                            disabled={actionLoading === tool.config_path}
                            className="px-4 py-2 text-sm bg-zinc-700 hover:bg-zinc-600 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {actionLoading === tool.config_path ? 'Disconnecting...' : 'Disconnect'}
                          </button>
                        )}

                        <button
                          onClick={() => handleHideTool(tool.name)}
                          className="p-2 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 rounded-lg transition-colors"
                          title="Hide from list"
                        >
                          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                          </svg>
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Discover More Tools section */}
              {hiddenToolsList.length > 0 && (
                <div className="mt-4 pt-4 border-t border-zinc-800">
                  <button
                    onClick={() => setShowDiscoverMore(!showDiscoverMore)}
                    className="w-full flex items-center justify-between p-3 bg-zinc-800/30 hover:bg-zinc-800/50 rounded-lg transition-colors"
                  >
                    <div className="flex items-center gap-2 text-sm text-zinc-400">
                      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      Discover More Tools ({hiddenToolsList.length})
                    </div>
                    <svg
                      className={`w-4 h-4 text-zinc-400 transition-transform ${showDiscoverMore ? 'rotate-180' : ''}`}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {showDiscoverMore && (
                    <div className="mt-3 space-y-2">
                      {hiddenToolsList.map((tool) => (
                        <div
                          key={tool.config_path}
                          className="flex items-center justify-between p-3 bg-zinc-800/30 rounded-lg border border-zinc-700/30"
                        >
                          <div className="flex items-center gap-3">
                            <div className={`w-3 h-3 rounded-full ${getStatusColor(tool.status)}`} />
                            <div>
                              <p className="font-medium text-sm">{tool.name}</p>
                              <p className="text-xs text-zinc-500">{getStatusText(tool.status)}</p>
                            </div>
                          </div>

                          <button
                            onClick={() => handleUnhideTool(tool.name)}
                            className="px-3 py-1.5 text-xs bg-zinc-700 hover:bg-zinc-600 rounded-lg transition-colors"
                          >
                            Show in List
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {/* Add custom client button */}
          <div className="mt-4 pt-4 border-t border-zinc-800">
            <button
              onClick={() => setShowCustomPath(true)}
              className="w-full px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm text-zinc-300 transition-colors flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              Add Custom MCP Client
            </button>
          </div>

          {/* Info box */}
          <div className="mt-6 p-4 bg-blue-900/20 border border-blue-800/30 rounded-lg">
            <div className="flex items-start gap-3">
              <svg className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div className="text-sm text-blue-200/90">
                <p className="font-medium mb-1">How it works</p>
                <ul className="space-y-1 text-xs text-blue-200/70">
                  <li>• Roampal memory stays 100% local on your machine</li>
                  <li>• Connected tools can search and add to your memory</li>
                  <li>• Restart the tool after connecting to activate</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
      </div>

      {/* Custom path modal */}
      {showCustomPath && (
        <div
          className="fixed inset-0 bg-black/70 z-[60] flex items-center justify-center p-4"
          onClick={() => setShowCustomPath(false)}
        >
          <div
            className="bg-zinc-900 rounded-lg p-6 max-w-md w-full border border-zinc-800"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="text-lg font-bold mb-4">Add Custom MCP Client</h3>
            <p className="text-sm text-zinc-400 mb-4">
              Enter the path to your MCP client's config file
            </p>
            <input
              type="text"
              value={customPath}
              onChange={(e) => setCustomPath(e.target.value)}
              placeholder="e.g., ~/.config/my-tool/mcp.json"
              className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm mb-4 focus:outline-none focus:border-blue-500"
              autoFocus
            />
            <div className="flex gap-2">
              <button
                onClick={() => customPath && handleConnect(customPath)}
                disabled={!customPath || actionLoading === customPath}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {actionLoading === customPath ? 'Connecting...' : 'Connect'}
              </button>
              <button
                onClick={() => {
                  setShowCustomPath(false);
                  setCustomPath('');
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
