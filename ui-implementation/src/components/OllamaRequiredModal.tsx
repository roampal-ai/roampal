import React, { useState } from 'react';
import { open as openUrl } from '@tauri-apps/api/shell';

interface OllamaRequiredModalProps {
  isOpen: boolean;
  onClose: () => void;
  onOpenIntegrations?: () => void;
}

export const OllamaRequiredModal: React.FC<OllamaRequiredModalProps> = ({ isOpen, onClose, onOpenIntegrations }) => {
  const [activeTab, setActiveTab] = useState<'llm' | 'mcp'>('llm');

  if (!isOpen) return null;

  const handleDownload = async (url: string) => {
    try {
      await openUrl(url);
    } catch (error) {
      // Fallback to window.open if Tauri shell fails
      window.open(url, '_blank');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <h2 className="text-xl font-semibold text-white mb-4">Welcome to Roampal</h2>

        {/* Tabs */}
        <div className="flex gap-2 mb-4 border-b border-zinc-700">
          <button
            onClick={() => setActiveTab('llm')}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === 'llm'
                ? 'text-white border-b-2 border-blue-500'
                : 'text-zinc-400 hover:text-zinc-300'
            }`}
          >
            Setup LLM Provider
          </button>
          <button
            onClick={() => setActiveTab('mcp')}
            className={`px-4 py-2 font-medium transition-colors ${
              activeTab === 'mcp'
                ? 'text-white border-b-2 border-blue-500'
                : 'text-zinc-400 hover:text-zinc-300'
            }`}
          >
            MCP Integration (Optional)
          </button>
        </div>

        {/* LLM Provider Tab */}
        {activeTab === 'llm' && (
          <div className="space-y-4 text-zinc-300 text-sm">
            <p>
              Roampal requires a local LLM provider to run AI models on your computer.
            </p>

            <div className="bg-zinc-800 border border-zinc-700 rounded p-3">
              <p className="font-semibold text-white mb-2">Recommended Provider:</p>
              <ul className="space-y-2 text-xs">
                <li>
                  <span className="font-medium text-green-400">Ollama</span>
                  <span className="text-zinc-400"> - Zero-config, runs automatically</span>
                </li>
                <li>
                  <span className="font-medium text-amber-400">LM Studio</span>
                  <span className="text-zinc-400"> - Advanced users only</span>
                  <div className="mt-1 pl-4 text-[11px] text-zinc-500">
                    Requires manual server setup in app
                  </div>
                </li>
              </ul>
            </div>

            <div className="bg-zinc-800/50 border border-zinc-700/50 rounded p-3">
              <p className="font-semibold text-white mb-2">Why local providers?</p>
              <ul className="space-y-1 text-xs text-zinc-400">
                <li>• Runs AI models completely offline</li>
                <li>• No data leaves your computer</li>
                <li>• Free and open source</li>
              </ul>
            </div>

            <p className="text-xs text-zinc-500">
              After installing a provider, make sure it's running and click "Get Started" below.
            </p>

            <div className="flex gap-3">
              <button
                onClick={() => handleDownload('https://ollama.com/download')}
                className="flex-1 flex items-center justify-center p-4 bg-zinc-700 hover:bg-zinc-600 text-white rounded-lg transition-colors"
              >
                <span className="font-medium">Download Ollama</span>
              </button>
              <button
                onClick={() => handleDownload('https://lmstudio.ai')}
                className="flex-1 flex items-center justify-center p-4 bg-zinc-700 hover:bg-zinc-600 text-white rounded-lg transition-colors"
              >
                <span className="font-medium">Download LM Studio</span>
              </button>
            </div>
          </div>
        )}

        {/* MCP Integration Tab */}
        {activeTab === 'mcp' && (
          <div className="space-y-4 text-zinc-300 text-sm">
            <p className="text-zinc-200 font-medium">
              Connect any MCP-compatible AI tool to Roampal's memory system.
            </p>

            <div className="bg-blue-900/20 border border-blue-700/50 rounded p-3">
              <p className="font-semibold text-blue-300 mb-2">✨ What is MCP?</p>
              <p className="text-xs text-zinc-400">
                Model Context Protocol is a universal standard that allows any AI tool to read and write to Roampal's 5-tier memory system.
                Your conversations across all connected tools can learn from and contribute to the same knowledge base.
              </p>
            </div>

            <div className="bg-zinc-800 border border-zinc-700 rounded p-4">
              <p className="font-semibold text-white mb-3">Quick Setup:</p>
              <ol className="space-y-3 text-xs list-decimal list-inside">
                <li>
                  <span className="text-zinc-300">Click the <span className="font-semibold">Settings</span> button in the left sidebar</span>
                </li>
                <li>
                  <span className="text-zinc-300">Go to the <span className="font-mono bg-zinc-700 px-1 rounded">Integrations</span> tab</span>
                </li>
                <li>
                  <span className="text-zinc-300">Click <span className="font-semibold text-blue-400">"Scan for Tools"</span></span>
                </li>
                <li>
                  <span className="text-zinc-300">Select your MCP-compatible tool from the detected list</span>
                </li>
                <li>
                  <span className="text-zinc-300">Click <span className="font-semibold text-green-400">"Connect"</span></span>
                </li>
                <li>
                  <span className="text-zinc-300">Restart your tool to complete setup</span>
                </li>
              </ol>
            </div>

            <div className="bg-zinc-800/50 border border-zinc-700/50 rounded p-3">
              <p className="font-semibold text-white mb-2">Works with any MCP client:</p>
              <ul className="space-y-1 text-xs text-zinc-400">
                <li>• <span className="text-zinc-300 font-medium">Claude Desktop</span> - Full memory access with outcome scoring</li>
                <li>• <span className="text-zinc-300 font-medium">Cursor</span> - Code-aware memory integration</li>
                <li>• <span className="text-zinc-300 font-medium">Continue.dev</span> - VS Code memory bridge</li>
                <li>• <span className="text-zinc-300 font-medium">Cline</span> - Autonomous coding agent</li>
                <li>• <span className="text-zinc-300 font-medium">Any MCP-compatible tool</span> - 6 core tools via standard protocol</li>
              </ul>
            </div>

            <div className="bg-amber-900/20 border border-amber-700/50 rounded p-3">
              <p className="font-semibold text-amber-300 mb-2">⚡ Pro Tip:</p>
              <p className="text-xs text-zinc-400">
                After connecting, your tool can search Roampal's memory, add to your memory bank,
                and learn what works through outcome scoring. All memories sync in real-time across all connected tools.
              </p>
            </div>

            <button
              onClick={() => {
                onClose();
                onOpenIntegrations?.();
              }}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2.5 px-4 rounded-lg transition-colors text-sm"
            >
              Open Integrations Settings
            </button>
          </div>
        )}

        {/* Bottom Button */}
        <button
          onClick={onClose}
          className="w-full mt-6 bg-green-600 hover:bg-green-700 text-white font-medium py-2.5 px-4 rounded-lg transition-colors"
        >
          Get Started
        </button>
      </div>
    </div>
  );
};
