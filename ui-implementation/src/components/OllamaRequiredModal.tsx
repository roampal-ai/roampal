import React from 'react';
import { open as openUrl } from '@tauri-apps/api/shell';

interface OllamaRequiredModalProps {
  isOpen: boolean;
  onRetry: () => void;
}

export const OllamaRequiredModal: React.FC<OllamaRequiredModalProps> = ({ isOpen, onRetry }) => {
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
      <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-6 max-w-md w-full mx-4">
        <h2 className="text-xl font-semibold text-white mb-4">LLM Provider Required</h2>

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
            After installing a provider, make sure it's running and click "I Installed a Provider" below.
          </p>
        </div>

        <div className="space-y-3 mt-6">
          <div className="flex gap-3">
            <button
              onClick={() => handleDownload('https://ollama.com/download')}
              className="flex-1 flex items-center justify-center p-4 bg-zinc-700 hover:bg-zinc-600 text-white rounded-lg transition-colors"
            >
              <span className="font-medium">Ollama</span>
            </button>
            <button
              onClick={() => handleDownload('https://lmstudio.ai')}
              className="flex-1 flex items-center justify-center p-4 bg-zinc-700 hover:bg-zinc-600 text-white rounded-lg transition-colors"
            >
              <span className="font-medium">LM Studio</span>
            </button>
          </div>
          <button
            onClick={onRetry}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-2.5 px-4 rounded-lg transition-colors"
          >
            I Installed a Provider
          </button>
        </div>
      </div>
    </div>
  );
};
