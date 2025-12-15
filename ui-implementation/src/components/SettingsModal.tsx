import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { DataManagementModal } from './DataManagementModal';
import { MemoryBankModal } from './MemoryBankModal';
import { ModelContextSettings } from './ModelContextSettings';
import { IntegrationsPanel } from './IntegrationsPanel';
import { MCPServersPanel } from './MCPServersPanel';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialTab?: 'integrations';
}

export const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose, initialTab }) => {
  const [showDataManagementModal, setShowDataManagementModal] = useState(false);
  const [showMemoryBankModal, setShowMemoryBankModal] = useState(false);
  const [showModelContextSettings, setShowModelContextSettings] = useState(false);
  const [showIntegrationsPanel, setShowIntegrationsPanel] = useState(false);
  const [showMCPServersPanel, setShowMCPServersPanel] = useState(false);
  const [currentModel, setCurrentModel] = useState<string>('');
  const [currentProvider, setCurrentProvider] = useState<string>('');
  const [providers, setProviders] = useState<any[]>([]);

  // Open Integrations tab if specified by initialTab
  useEffect(() => {
    if (isOpen && initialTab === 'integrations') {
      setShowIntegrationsPanel(true);
    }
  }, [isOpen, initialTab]);

  // Fetch current model and providers when modal opens
  useEffect(() => {
    const fetchCurrentModel = async () => {
      try {
        const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/model/current`);
        if (response.ok) {
          const data = await response.json();
          setCurrentModel(data.current_model || data.model || '');
          setCurrentProvider(data.provider || '');
        }
      } catch (error) {
        console.error('Failed to fetch current model:', error);
      }
    };

    const fetchProviders = async () => {
      try {
        const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/model/providers/detect`);
        if (response.ok) {
          const data = await response.json();
          setProviders(data.providers || []);
        }
      } catch (error) {
        console.error('Failed to fetch providers:', error);
      }
    };

    if (isOpen) {
      fetchCurrentModel();
      fetchProviders();
    }
  }, [isOpen]);

  const handleDataManagementClick = () => {
    setShowDataManagementModal(true);
  };

  const handleMemoryBankClick = () => {
    setShowMemoryBankModal(true);
  };

  const handleModelContextClick = () => {
    setShowModelContextSettings(true);
  };

  const handleIntegrationsClick = () => {
    setShowIntegrationsPanel(true);
  };

  const handleMCPServersClick = () => {
    setShowMCPServersPanel(true);
  };

  if (!isOpen) return null;

  return (
    <>
      <div
        className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-2 sm:p-4"
        onClick={onClose}
      >
        <div
          className="bg-zinc-900 rounded-xl shadow-2xl w-full max-w-[95vw] sm:max-w-md border border-zinc-800 max-h-[90vh] overflow-y-auto"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex justify-between items-center p-4 sm:p-6 border-b border-zinc-800">
            <h2 className="text-lg sm:text-xl font-bold">Settings</h2>
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
          <div className="p-4 sm:p-6 space-y-3 pb-8">
            {/* LLM Providers Status */}
            {providers.length > 0 && (
              <div className="space-y-2 pb-3 border-b border-zinc-800">
                <div className="flex items-center gap-2 px-2">
                  <svg className="w-4 h-4 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-sm font-medium text-zinc-400">LLM Providers</span>
                </div>
                <div className="space-y-1">
                  {providers.map((provider) => (
                    <div
                      key={provider.name}
                      className="flex items-center justify-between px-4 py-2 bg-zinc-800/50 rounded-lg"
                    >
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-500" />
                        <span className="text-sm text-zinc-300 capitalize">{provider.name}</span>
                      </div>
                      <span className="text-xs text-zinc-500">{provider.model_count || 0} models</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Memory Bank */}
            <div className="space-y-2">
              <button
                onClick={handleMemoryBankClick}
                className="w-full flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2.5 sm:py-3 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors text-left"
              >
                <svg className="w-5 h-5 flex-shrink-0 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <span className="flex-1 truncate">Memory Bank</span>
              </button>
            </div>

            {/* Model Context Settings */}
            <div className="space-y-2">
              <button
                onClick={handleModelContextClick}
                className="w-full flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2.5 sm:py-3 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors text-left"
              >
                <svg className="w-5 h-5 flex-shrink-0 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
                <span className="flex-1 truncate">Model Context Settings</span>
              </button>
            </div>

            {/* Integrations */}
            <div className="space-y-2">
              <button
                onClick={handleIntegrationsClick}
                className="w-full flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2.5 sm:py-3 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors text-left"
              >
                <svg className="w-5 h-5 flex-shrink-0 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                </svg>
                <span className="flex-1 truncate">Integrations</span>
              </button>
            </div>

            {/* MCP Tool Servers */}
            <div className="space-y-2">
              <button
                onClick={handleMCPServersClick}
                className="w-full flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2.5 sm:py-3 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors text-left"
              >
                <svg className="w-5 h-5 flex-shrink-0 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
                </svg>
                <span className="flex-1 truncate">MCP Tool Servers</span>
              </button>
            </div>

            {/* Other Settings */}
            <div className="space-y-2">
              <button
                onClick={() => alert('Voice settings coming soon!')}
                className="w-full flex items-center gap-2 sm:gap-3 px-3 sm:px-4 py-2.5 sm:py-3 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors text-left"
              >
                <svg className="w-5 h-5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                <span className="flex-1 truncate">Voice Settings</span>
                <span className="text-xs text-zinc-500 flex-shrink-0">Coming Soon</span>
              </button>
            </div>

            {/* Data Management Button */}
            <div className="pt-4 border-t border-zinc-800">
              <button
                onClick={handleDataManagementClick}
                className="w-full h-10 px-3 py-2 flex items-center justify-center gap-2 rounded-lg bg-blue-600/10 hover:bg-blue-600/20 border border-blue-600/30 transition-colors"
              >
                <svg className="w-4 h-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
                <span className="text-sm font-medium text-blue-500">Data Management</span>
              </button>
            </div>

            {/* v0.2.8: Exit Roampal Button - clean shutdown */}
            <div className="pt-2">
              <button
                onClick={async () => {
                  try {
                    await invoke('exit_app');
                  } catch (error) {
                    console.error('Failed to exit:', error);
                  }
                }}
                className="w-full h-10 px-3 py-2 flex items-center justify-center gap-2 rounded-lg bg-red-600/10 hover:bg-red-600/20 border border-red-600/30 transition-colors"
              >
                <svg className="w-4 h-4 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                <span className="text-sm font-medium text-red-500">Exit Roampal</span>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Data Management Modal */}
      <DataManagementModal
        isOpen={showDataManagementModal}
        onClose={() => setShowDataManagementModal(false)}
      />

      {/* Memory Bank Modal */}
      <MemoryBankModal
        isOpen={showMemoryBankModal}
        onClose={() => setShowMemoryBankModal(false)}
      />

      {/* Model Context Settings Modal */}
      <ModelContextSettings
        isOpen={showModelContextSettings}
        onClose={() => setShowModelContextSettings(false)}
        currentModel={currentModel}
        currentProvider={currentProvider}
      />

      {/* Integrations Panel */}
      <IntegrationsPanel
        isOpen={showIntegrationsPanel}
        onClose={() => setShowIntegrationsPanel(false)}
      />

      {/* MCP Servers Panel */}
      <MCPServersPanel
        isOpen={showMCPServersPanel}
        onClose={() => setShowMCPServersPanel(false)}
      />
    </>
  );
};
