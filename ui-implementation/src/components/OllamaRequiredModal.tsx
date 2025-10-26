import React from 'react';
import { open as openUrl } from '@tauri-apps/api/shell';

interface OllamaRequiredModalProps {
  isOpen: boolean;
  onRetry: () => void;
}

export const OllamaRequiredModal: React.FC<OllamaRequiredModalProps> = ({ isOpen, onRetry }) => {
  if (!isOpen) return null;

  const handleDownload = async () => {
    try {
      await openUrl('https://ollama.com/download');
    } catch (error) {
      // Fallback to window.open if Tauri shell fails
      window.open('https://ollama.com/download', '_blank');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-zinc-900 border border-zinc-700 rounded-lg p-6 max-w-md w-full mx-4">
        <h2 className="text-xl font-semibold text-white mb-4">Ollama Required</h2>

        <div className="space-y-4 text-zinc-300 text-sm">
          <p>
            Roampal requires <span className="font-semibold text-white">Ollama</span> to run AI models locally on your computer.
          </p>

          <div className="bg-zinc-800 border border-zinc-700 rounded p-3">
            <p className="font-semibold text-white mb-2">Why Ollama?</p>
            <ul className="space-y-1 text-xs text-zinc-400">
              <li>• Runs AI models completely offline</li>
              <li>• No data leaves your computer</li>
              <li>• Free and open source</li>
            </ul>
          </div>

          <p className="text-xs text-zinc-500">
            After installing Ollama, make sure it's running and click "I Installed It" below.
          </p>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={handleDownload}
            className="flex-1 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded transition-colors"
          >
            Download Ollama
          </button>
          <button
            onClick={onRetry}
            className="flex-1 bg-zinc-700 hover:bg-zinc-600 text-white font-medium py-2 px-4 rounded transition-colors"
          >
            I Installed It
          </button>
        </div>
      </div>
    </div>
  );
};
