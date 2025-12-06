import React, { useState, useEffect } from 'react';
import { modelContextService } from '../services/modelContextService';
import { apiFetch } from '../utils/fetch';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface ModelContextSettingsProps {
  isOpen: boolean;
  onClose: () => void;
  currentModel?: string;
  currentProvider?: string;
}

interface ModelContextInfo {
  model: string;
  default: number;
  max: number;
  current: number;
  is_override: boolean;
  provider?: string;
}

export const ModelContextSettings: React.FC<ModelContextSettingsProps> = ({
  isOpen,
  onClose,
  currentModel,
  currentProvider,
}) => {
  const [modelContexts, setModelContexts] = useState<ModelContextInfo[]>([]);
  const [loading, setLoading] = useState(true);

  // Fetch installed models and their contexts
  useEffect(() => {
    const fetchModelContexts = async () => {
      if (!isOpen) return;

      setLoading(true);
      try {
        // Fetch installed models
        const response = await apiFetch(`${ROAMPAL_CONFIG.apiUrl}/api/model/available`);
        if (!response.ok) {
          console.error('Failed to fetch installed models');
          return;
        }

        const data = await response.json();
        const installedModels = (data.models || [])
          .filter((m: any) => {
            const name = typeof m === 'string' ? m : m.name;
            return name && name.length > 0 && !name.includes('embed');
          });

        // Fetch context info for each installed model
        const contexts: ModelContextInfo[] = [];
        for (const modelData of installedModels) {
          const modelName = typeof modelData === 'string' ? modelData : modelData.name;
          const modelProvider = typeof modelData === 'object' ? modelData.provider : 'ollama';
          try {
            const info = await modelContextService.getModelContext(modelName);
            contexts.push({
              model: modelName,
              default: info.default,
              max: info.max,
              current: info.current,
              is_override: info.is_override,
              provider: modelProvider,
            });
          } catch (error) {
            console.error(`Failed to fetch context for ${modelName}:`, error);
          }
        }

        // Sort: current model first, then by name
        contexts.sort((a, b) => {
          if (a.model === currentModel) return -1;
          if (b.model === currentModel) return 1;
          return a.model.localeCompare(b.model);
        });

        setModelContexts(contexts);
      } catch (error) {
        console.error('Failed to fetch model contexts:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchModelContexts();
  }, [isOpen, currentModel]);

  const handleContextChange = async (model: string, newValue: number) => {
    // Update local state immediately for responsiveness
    setModelContexts(prev =>
      prev.map(m =>
        m.model === model
          ? { ...m, current: newValue, is_override: newValue !== m.default }
          : m
      )
    );

    // Save to backend
    try {
      await modelContextService.setContextSize(model, newValue);
    } catch (error) {
      console.error(`Failed to update context for ${model}:`, error);
    }
  };

  const handleReset = async (model: string) => {
    try {
      await modelContextService.resetContextSize(model);

      // Update local state
      setModelContexts(prev =>
        prev.map(m =>
          m.model === model
            ? { ...m, current: m.default, is_override: false }
            : m
        )
      );
    } catch (error) {
      console.error(`Failed to reset context for ${model}:`, error);
    }
  };

  const formatTokens = (tokens: number): string => {
    if (tokens >= 1000) {
      return `${(tokens / 1000).toFixed(0)}K`;
    }
    return tokens.toString();
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-zinc-900 rounded-xl shadow-2xl max-w-2xl w-full border border-zinc-800 max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex justify-between items-center p-6 border-b border-zinc-800">
          <div>
            <h2 className="text-xl font-bold">Context Window Settings</h2>
            <p className="text-sm text-zinc-400 mt-1">
              Adjust context window sizes for your installed models
            </p>
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
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500" />
            </div>
          ) : modelContexts.length === 0 ? (
            <div className="text-center py-12 text-zinc-500">
              <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
              </svg>
              <p>No models installed</p>
              <p className="text-sm mt-2">Install models from the Model Manager to configure their context windows</p>
            </div>
          ) : (
            <div className="space-y-4">
              {modelContexts.map((context) => {
                const isCurrentModel = context.model === currentModel;
                const isModelLMStudio = context.provider === 'lmstudio';

                return (
                  <div
                    key={context.model}
                    className={`bg-zinc-800 rounded-lg p-4 transition-all ${
                      isCurrentModel ? 'ring-2 ring-blue-500/50 bg-blue-500/5' : ''
                    }`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <h3 className="font-medium">{context.model}</h3>
                          {isCurrentModel && (
                            <span className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-400 rounded flex items-center gap-1">
                              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                              </svg>
                              Active
                            </span>
                          )}
                          {context.is_override && (
                            <span className="text-xs px-2 py-0.5 bg-yellow-500/20 text-yellow-400 rounded">
                              Custom
                            </span>
                          )}
                        </div>
                        <div className="text-sm text-zinc-400 mt-1">
                          Default: {formatTokens(context.default)} • Max: {formatTokens(context.max)}
                        </div>
                      </div>
                      {context.is_override && !isModelLMStudio && (
                        <button
                          onClick={() => handleReset(context.model)}
                          className="text-xs px-2 py-1 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-700 rounded transition-colors"
                        >
                          Reset
                        </button>
                      )}
                    </div>

                    <div className="flex items-center gap-4">
                      <input
                        type="range"
                        min={Math.min(512, context.default)}
                        max={context.max}
                        step={512}
                        value={context.current}
                        onChange={(e) => handleContextChange(context.model, parseInt(e.target.value))}
                        disabled={isModelLMStudio}
                        className={`flex-1 h-2 bg-zinc-700 rounded-lg appearance-none slider ${
                          isModelLMStudio ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'
                        }`}
                      />
                      <span className={`text-sm font-medium w-16 text-right ${
                        isModelLMStudio ? 'text-zinc-500' : 'text-blue-400'
                      }`}>
                        {formatTokens(context.current)}
                      </span>
                    </div>
                    <div className="flex justify-between text-xs text-zinc-500 mt-1">
                      <span>{formatTokens(Math.min(512, context.default))}</span>
                      <span>{formatTokens(context.max)}</span>
                    </div>
                    {isModelLMStudio && (
                      <p className="text-xs text-yellow-500/70 mt-2">
                        LM Studio manages context internally
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-zinc-800">
          <div className="text-xs text-zinc-500 mb-4">
            <p>
              Context window size affects how much text the model can process at once.
              Larger contexts use more memory.
            </p>
            {modelContexts.some(m => m.provider === 'lmstudio') && (
              <p className="text-yellow-500/80 mt-2">
                LM Studio models: context is managed by LM Studio, not Roampal.
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-blue-600/10 hover:bg-blue-600/20 border border-blue-600/30 text-blue-500 rounded-lg transition-colors font-medium"
          >
            Done
          </button>
        </div>
      </div>

      <style>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          width: 16px;
          height: 16px;
          background: #3b82f6;
          border-radius: 50%;
          cursor: pointer;
        }
        .slider::-moz-range-thumb {
          width: 16px;
          height: 16px;
          background: #3b82f6;
          border-radius: 50%;
          cursor: pointer;
          border: none;
        }
      `}</style>
    </div>
  );
};
