import React from 'react';

interface SidecarReloadWarningModalProps {
  isOpen: boolean;
  chatModel: string;
  sidecarModel: string;
  onCancel: () => void;
  onConfirm: () => void;
}

export const SidecarReloadWarningModal: React.FC<SidecarReloadWarningModalProps> = ({
  isOpen,
  chatModel,
  sidecarModel,
  onCancel,
  onConfirm,
}) => {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/60 z-[60] flex items-center justify-center p-4"
      onClick={onCancel}
      data-testid="sidecar-reload-warning"
    >
      <div
        className="bg-zinc-900 rounded-xl shadow-2xl w-full max-w-md border border-zinc-800"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-5 border-b border-zinc-800">
          <h3 className="text-lg font-semibold text-amber-300">
            Running two models on one GPU
          </h3>
        </div>

        <div className="p-5 space-y-3 text-sm text-zinc-300">
          <p>
            Your sidecar will use{' '}
            <span className="font-mono text-zinc-100">{sidecarModel}</span>,
            which is different from your chat model{' '}
            <span className="font-mono text-zinc-100">{chatModel}</span>.
          </p>
          <p>
            On a single GPU, each message will briefly unload one model and
            load the other &mdash; typically <strong>10&ndash;30 seconds</strong>{' '}
            the first time, faster while both stay warm.
          </p>
          <p className="text-zinc-500 italic text-xs">
            This is a limitation of local model hosts (Ollama / LM Studio), not
            Roampal.
          </p>
        </div>

        <div className="flex gap-2 p-4 border-t border-zinc-800">
          <button
            onClick={onCancel}
            className="flex-1 h-10 px-3 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-sm font-medium text-zinc-200 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="flex-1 h-10 px-3 rounded-lg bg-amber-600 hover:bg-amber-500 text-sm font-medium text-white transition-colors"
          >
            Use different sidecar model
          </button>
        </div>
      </div>
    </div>
  );
};
