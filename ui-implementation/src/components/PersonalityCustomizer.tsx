import React, { useState, useEffect } from 'react';
import {
  ArrowDownTrayIcon,
  CheckIcon,
  XMarkIcon,
  InformationCircleIcon,
  ArrowPathIcon,
  DocumentTextIcon,
  ChatBubbleBottomCenterTextIcon,
  BriefcaseIcon,
  BoltIcon,
  SparklesIcon,
  DocumentIcon,
  BookOpenIcon,
  ClockIcon
} from '@heroicons/react/24/outline';
import yaml from 'js-yaml';
import { ROAMPAL_CONFIG } from '../config/roampal';

interface PersonalityTemplate {
  template_id: string;
  name: string;
  content: string;
  is_preset: boolean;
}

interface PersonalityCustomizerProps {
  apiBase?: string;
}

const EXAMPLE_TEMPLATE = `# Roampal Personality Configuration
# Edit these values to customize how Roampal responds

identity:
  name: "Roampal"          # Change this to customize the assistant name (appears in UI)
  role: "Memory-Enhanced Assistant"
  background: "I'm an intelligent assistant with persistent memory."

communication:
  tone: "warm"              # Options: warm | direct | enthusiastic | professional
  verbosity: "balanced"      # Options: concise | balanced | detailed
  formality: "professional"  # Options: casual | professional | formal
  use_analogies: true
  use_examples: true
  use_humor: false

memory_usage:
  priority: "when_relevant"      # Options: always_reference | when_relevant
  pattern_trust: "balanced"      # Options: heavily_favor | balanced

personality_traits:
  - "Helpful and reliable"
  - "Clear and organized"
  - "Learns from past interactions"

custom_instructions: |
  I prioritize accuracy and helpfulness. When I find a proven solution
  in memory, I'll let you know it worked before.`;

// Simplified guided fields - only the most important ones
const QUICK_SETTINGS = [
  {
    key: 'identity.name',
    label: 'Assistant Name',
    type: 'text',
    description: 'What should your assistant be called?',
    placeholder: 'e.g., Roampal, Alex, Helper'
  },
  {
    key: 'communication.tone',
    label: 'Conversation Style',
    type: 'select',
    description: 'How your assistant communicates with you',
    options: [
      { value: 'warm', label: 'Warm & Friendly', icon: ChatBubbleBottomCenterTextIcon },
      { value: 'professional', label: 'Professional', icon: BriefcaseIcon },
      { value: 'direct', label: 'Direct & Efficient', icon: BoltIcon },
      { value: 'enthusiastic', label: 'Playful & Engaging', icon: SparklesIcon }
    ]
  },
  {
    key: 'communication.verbosity',
    label: 'Response Length',
    type: 'select',
    description: 'How detailed responses should be',
    options: [
      { value: 'concise', label: 'Brief & Clear', icon: DocumentIcon },
      { value: 'balanced', label: 'Balanced', icon: DocumentTextIcon },
      { value: 'detailed', label: 'Detailed & Thorough', icon: BookOpenIcon }
    ]
  },
  {
    key: 'memory_usage.priority',
    label: 'Memory References',
    type: 'select',
    description: 'How often to mention past conversations',
    options: [
      { value: 'when_relevant', label: 'Only when relevant', icon: ClockIcon },
      { value: 'always_reference', label: 'Reference frequently', icon: ClockIcon }
    ]
  },
  {
    key: 'identity.background',
    label: 'Assistant Identity',
    type: 'textarea',
    description: 'Core description of who your assistant is',
    placeholder: 'e.g., "I\'m an intelligent assistant with persistent memory..."'
  },
  {
    key: 'identity.role',
    label: 'Primary Role',
    type: 'text',
    description: 'What your assistant specializes in',
    placeholder: 'e.g., Memory-Enhanced Assistant, Research Helper'
  },
  {
    key: 'communication.formality',
    label: 'Formality Level',
    type: 'select',
    description: 'How formal responses should be',
    options: [
      { value: 'casual', label: 'Casual & Relaxed' },
      { value: 'professional', label: 'Professional' },
      { value: 'formal', label: 'Formal & Structured' }
    ]
  },
  {
    key: 'communication.use_analogies',
    label: 'Use Analogies',
    type: 'toggle',
    description: 'Explain concepts using comparisons'
  },
  {
    key: 'communication.use_examples',
    label: 'Use Examples',
    type: 'toggle',
    description: 'Provide concrete examples'
  },
  {
    key: 'communication.use_humor',
    label: 'Use Humor',
    type: 'toggle',
    description: 'Add lighthearted moments'
  },
  {
    key: 'custom_instructions',
    label: 'Custom Instructions',
    type: 'textarea',
    description: 'Additional guidelines for Roampal (optional)',
    placeholder: 'e.g., "Always explain technical concepts with examples" or "Keep responses concise"'
  }
];

export const PersonalityCustomizer: React.FC<PersonalityCustomizerProps> = ({
  apiBase = ROAMPAL_CONFIG.apiUrl
}) => {
  const [presets, setPresets] = useState<string[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('default');
  const [yamlContent, setYamlContent] = useState<string>('');
  const [originalContent, setOriginalContent] = useState<string>('');
  const [parsedData, setParsedData] = useState<any>({});
  const [viewMode, setViewMode] = useState<'quick' | 'advanced'>('quick');
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [validationError, setValidationError] = useState<string>('');
  const [hasChanges, setHasChanges] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

  useEffect(() => {
    loadPresets();
    loadCurrentTemplate();
  }, []);

  // Validate and parse YAML whenever content changes
  useEffect(() => {
    if (yamlContent) {
      try {
        const parsed = yaml.load(yamlContent);
        setParsedData(parsed || {});
        setValidationError('');
        setHasChanges(yamlContent !== originalContent);
      } catch (error: any) {
        const friendlyError = error.message
          .replace(/bad indentation/i, 'Indentation error')
          .replace(/at line (\d+)/i, 'on line $1')
          .replace(/column (\d+)/i, '')
          .trim();
        setValidationError(friendlyError);
        setParsedData({});
      }
    }
  }, [yamlContent, originalContent]);

  const loadPresets = async () => {
    try {
      const response = await fetch(`${apiBase}/api/personality/presets`);
      if (response.ok) {
        const data = await response.json();
        setPresets(data.presets || []);
      }
    } catch (error) {
      console.error('Failed to load presets:', error);
    }
  };

  const loadCurrentTemplate = async () => {
    try {
      const response = await fetch(`${apiBase}/api/personality/current`);
      if (response.ok) {
        const template: PersonalityTemplate = await response.json();
        setYamlContent(template.content);
        setOriginalContent(template.content);
        setSelectedPreset(template.template_id);
      }
    } catch (error) {
      console.error('Failed to load current template:', error);
    }
  };

  const handlePresetChange = async (templateId: string) => {
    if (hasChanges) {
      if (!confirm('Switch presets? Your unsaved changes will be lost.')) {
        return;
      }
    }

    try {
      const response = await fetch(`${apiBase}/api/personality/template/${templateId}`);
      if (response.ok) {
        const template: PersonalityTemplate = await response.json();
        setYamlContent(template.content);
        setOriginalContent(template.content);
        setSelectedPreset(templateId);
        setHasChanges(false);

        // Activate the template
        await fetch(`${apiBase}/api/personality/activate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ template_id: templateId })
        });

        // Notify UI components that personality was updated
        window.dispatchEvent(new CustomEvent('personalityUpdated'));
      }
    } catch (error) {
      console.error('Failed to load template:', error);
      setErrorMessage('Failed to load template');
    }
  };

  const handleSaveAndApply = async () => {
    if (validationError) {
      setErrorMessage('Cannot save: Fix the error in your configuration first');
      return;
    }

    setSaveStatus('saving');
    setErrorMessage('');

    try {
      const saveResponse = await fetch(`${apiBase}/api/personality/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: 'custom',
          content: yamlContent
        })
      });

      if (saveResponse.ok) {
        const activateResponse = await fetch(`${apiBase}/api/personality/activate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ template_id: 'custom' })
        });

        if (activateResponse.ok) {
          setOriginalContent(yamlContent);
          setHasChanges(false);
          setSaveStatus('success');
          setTimeout(() => setSaveStatus('idle'), 2000);

          // Notify UI components that personality was updated
          window.dispatchEvent(new CustomEvent('personalityUpdated'));
        } else {
          throw new Error('Failed to activate template');
        }
      } else {
        const error = await saveResponse.json();
        throw new Error(error.detail || 'Failed to save template');
      }
    } catch (error: any) {
      console.error('Failed to save and apply:', error);
      setErrorMessage(error.message || 'Failed to save template');
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 3000);
    }
  };

  const handleReset = () => {
    if (confirm('Reset to last saved version? Your changes will be lost.')) {
      setYamlContent(originalContent);
      setHasChanges(false);
      setErrorMessage('');
    }
  };

  const getNestedValue = (obj: any, path: string): any => {
    return path.split('.').reduce((acc, part) => acc?.[part], obj);
  };

  const setNestedValue = (obj: any, path: string, value: any): any => {
    const parts = path.split('.');
    const last = parts.pop()!;
    const target = parts.reduce((acc, part) => {
      if (!acc[part]) acc[part] = {};
      return acc[part];
    }, obj);
    target[last] = value;
    return obj;
  };

  const updateQuickSetting = (key: string, value: any) => {
    try {
      const newData = JSON.parse(JSON.stringify(parsedData));
      setNestedValue(newData, key, value);
      const newYaml = yaml.dump(newData, { indent: 2, lineWidth: -1 });
      setYamlContent(newYaml);
    } catch (error) {
      console.error('Failed to update setting:', error);
    }
  };

  const handleDownload = () => {
    const blob = new Blob([yamlContent], { type: 'text/yaml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `roampal_personality.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const loadExample = () => {
    if (hasChanges) {
      if (!confirm('Load example template? Your current changes will be lost.')) {
        return;
      }
    }
    setYamlContent(EXAMPLE_TEMPLATE);
  };

  return (
    <div className="h-full flex flex-col bg-zinc-950 text-zinc-100">
      {/* Content - scrollable */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {/* Status indicator */}
        <div className="flex items-center justify-between">
          {hasChanges && (
            <span className="text-xs text-amber-400 font-medium">• Unsaved changes</span>
          )}
          <button
            onClick={() => setShowInfo(!showInfo)}
            className={`ml-auto p-1.5 rounded-lg transition-colors ${
              showInfo ? 'bg-blue-600/20 text-blue-400' : 'hover:bg-zinc-800 text-zinc-400'
            }`}
            title="Show information"
          >
            <InformationCircleIcon className="w-4 h-4" />
          </button>
        </div>
        {/* Info Panel */}
        {showInfo && (
          <div className="p-4 bg-blue-600/10 border border-blue-600/20 rounded-lg space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-blue-400">How it Works</h3>
              <button
                onClick={() => setShowInfo(false)}
                className="text-zinc-400 hover:text-zinc-200"
              >
                <XMarkIcon className="w-4 h-4" />
              </button>
            </div>
            <div className="text-xs text-zinc-300 space-y-2">
              <p>
                <strong className="text-zinc-200">Personality:</strong> Control how Roampal talks and thinks -
                from casual and friendly to professional and direct.
              </p>
              <p>
                <strong className="text-zinc-200">Quick Settings:</strong> Adjust the most common options with simple dropdowns.
              </p>
              <p>
                <strong className="text-zinc-200">Advanced Mode:</strong> Edit the full configuration file for complete control.
              </p>
              <p className="pt-2 border-t border-blue-500/20 text-blue-300">
                Changes take effect immediately in new conversations. Your memory system stays the same.
              </p>
            </div>
          </div>
        )}

        {/* Preset Selector */}
        <div>
          <label className="block text-xs font-medium text-zinc-400 mb-2">
            Choose a Starting Point
          </label>
          <select
            value={selectedPreset}
            onChange={(e) => handlePresetChange(e.target.value)}
            className="w-full h-9 px-3 bg-zinc-900 border border-zinc-800 rounded-md text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {presets.map((preset) => (
              <option key={preset} value={preset}>
                {preset === 'default' ? 'Default (Recommended)' :
                 preset.charAt(0).toUpperCase() + preset.slice(1)}
              </option>
            ))}
          </select>
          <p className="mt-1.5 text-xs text-zinc-500">
            Pick a preset, then customize it to your preference
          </p>
        </div>

        {/* Mode Toggle */}
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode('quick')}
            className={`flex-1 h-9 px-4 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'quick'
                ? 'bg-blue-600/10 border border-blue-600/30 text-blue-500'
                : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
            }`}
          >
            Quick Settings
          </button>
          <button
            onClick={() => setViewMode('advanced')}
            className={`flex-1 h-9 px-4 rounded-lg text-sm font-medium transition-colors ${
              viewMode === 'advanced'
                ? 'bg-blue-600/10 border border-blue-600/30 text-blue-500'
                : 'bg-zinc-900 border border-zinc-800 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
            }`}
          >
            Advanced
          </button>
        </div>

        {viewMode === 'quick' ? (
          /* Quick Settings */
          <div className="space-y-4">
            {QUICK_SETTINGS.map((setting, idx) => (
              <div key={idx}>
                <label className="block text-sm font-medium text-zinc-200 mb-1">
                  {setting.label}
                </label>
                <p className="text-xs text-zinc-500 mb-2">{setting.description}</p>

                {setting.type === 'text' && (
                  <input
                    type="text"
                    value={getNestedValue(parsedData, setting.key) || ''}
                    onChange={(e) => updateQuickSetting(setting.key, e.target.value)}
                    className="w-full h-9 px-3 bg-zinc-900 border border-zinc-800 rounded-md text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder={setting.placeholder}
                  />
                )}

                {setting.type === 'select' && setting.options && (
                  <select
                    value={getNestedValue(parsedData, setting.key) || ''}
                    onChange={(e) => updateQuickSetting(setting.key, e.target.value)}
                    className="w-full h-9 px-3 bg-zinc-900 border border-zinc-800 rounded-md text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {setting.options.map((opt, optIdx) => (
                      <option key={optIdx} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                )}

                {setting.type === 'textarea' && (
                  <textarea
                    value={getNestedValue(parsedData, setting.key) || ''}
                    onChange={(e) => updateQuickSetting(setting.key, e.target.value)}
                    className="w-full h-24 px-3 py-2 bg-zinc-900 border border-zinc-800 rounded-md text-sm text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                    placeholder={setting.placeholder}
                  />
                )}

                {setting.type === 'toggle' && (
                  <button
                    onClick={() => updateQuickSetting(setting.key, !getNestedValue(parsedData, setting.key))}
                    className={`w-full h-9 px-3 rounded-md text-sm font-medium transition-colors ${
                      getNestedValue(parsedData, setting.key)
                        ? 'bg-blue-600/20 border border-blue-600/40 text-blue-400'
                        : 'bg-zinc-900 border border-zinc-800 text-zinc-400'
                    }`}
                  >
                    {getNestedValue(parsedData, setting.key) ? 'Enabled' : 'Disabled'}
                  </button>
                )}
              </div>
            ))}

            <div className="p-3 bg-zinc-900/50 border border-zinc-800 rounded-md">
              <p className="text-xs text-zinc-400">
                <strong className="text-zinc-300">Need more control?</strong> Switch to Advanced mode to edit
                personality traits, expertise areas, and other detailed options.
              </p>
            </div>
          </div>
        ) : (
          /* Advanced Editor */
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <label className="block text-sm font-medium text-zinc-200">Configuration File</label>
                <p className="text-xs text-zinc-500 mt-0.5">
                  Edit YAML directly for complete control
                </p>
              </div>
              <button
                onClick={loadExample}
                className="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1.5 transition-colors"
                title="Load example template"
              >
                <DocumentTextIcon className="w-3.5 h-3.5" />
                <span>Load Example</span>
              </button>
            </div>

            {validationError && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-md flex items-start gap-2">
                <XMarkIcon className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-xs font-medium text-red-400">Configuration Error</p>
                  <p className="text-xs text-red-300 mt-0.5">{validationError}</p>
                </div>
              </div>
            )}

            <textarea
              value={yamlContent}
              onChange={(e) => setYamlContent(e.target.value)}
              className={`w-full h-[450px] px-3 py-2 bg-zinc-900 border rounded-lg text-zinc-100 font-mono text-xs leading-relaxed focus:outline-none focus:ring-2 resize-none ${
                validationError
                  ? 'border-red-500/50 focus:ring-red-500'
                  : 'border-zinc-800 focus:ring-blue-600/30'
              }`}
              spellCheck={false}
            />
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-2 pt-2 border-t border-zinc-800">
          <button
            onClick={handleSaveAndApply}
            disabled={!hasChanges || saveStatus === 'saving' || !!validationError}
            className="h-9 px-4 flex items-center gap-2 bg-blue-600/10 hover:bg-blue-600/20 border border-blue-600/30 disabled:bg-zinc-800 disabled:border-zinc-700 disabled:text-zinc-600 disabled:cursor-not-allowed text-sm font-medium text-blue-500 disabled:text-zinc-600 rounded-lg transition-colors"
          >
            {saveStatus === 'saving' ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                <span>Saving...</span>
              </>
            ) : saveStatus === 'success' ? (
              <>
                <CheckIcon className="w-4 h-4" />
                <span>Saved!</span>
              </>
            ) : (
              <>
                <CheckIcon className="w-4 h-4" />
                <span>Save Changes</span>
              </>
            )}
          </button>

          {hasChanges && (
            <button
              onClick={handleReset}
              className="h-9 px-4 flex items-center gap-2 bg-zinc-800 hover:bg-zinc-700 text-sm text-zinc-300 rounded-md transition-colors"
            >
              <ArrowPathIcon className="w-4 h-4" />
              <span>Reset</span>
            </button>
          )}

          <button
            onClick={handleDownload}
            className="h-9 px-4 flex items-center gap-2 bg-zinc-800 hover:bg-zinc-700 text-sm text-zinc-300 rounded-md transition-colors ml-auto"
            title="Download configuration file"
          >
            <ArrowDownTrayIcon className="w-4 h-4" />
            <span>Export</span>
          </button>
        </div>

        {/* Error Message */}
        {errorMessage && (
          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-md">
            <p className="text-xs text-red-400">{errorMessage}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PersonalityCustomizer;