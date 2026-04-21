import { describe, it, expect } from 'vitest';
import { buildModelDisplay, providerLabel } from '../../utils/modelDisplay';

describe('buildModelDisplay (v0.3.2 0h)', () => {
  const curatedDescriptions: Record<string, string> = {
    'qwen3:8b': '5.2GB • Strong reasoning',
  };

  it('shows the full Ollama tag in the label', () => {
    const a = buildModelDisplay({ name: 'gemma4:26b', provider: 'ollama' }, {});
    const b = buildModelDisplay({ name: 'gemma4:e2b', provider: 'ollama' }, {});
    expect(a.label).toBe('gemma4:26b');
    expect(b.label).toBe('gemma4:e2b');
    expect(a.label).not.toBe(b.label);
  });

  it('never writes "Custom model" in the subtitle', () => {
    const out = buildModelDisplay({ name: 'some-org/foo:latest', provider: 'ollama' }, {});
    expect(out.description).not.toContain('Custom');
  });

  it('uses param format even when a curated blurb exists (uniform subtitles)', () => {
    // v0.3.2 fix: param estimate always wins when parsable, so every row
    // in the dropdown reads the same way instead of mixing marketing copy
    // with parameter counts.
    const out = buildModelDisplay({ name: 'qwen3:8b', provider: 'ollama' }, curatedDescriptions);
    expect(out.description).toBe('≈ 8B parameters • Ollama');
    expect(out.description).not.toContain('Strong reasoning');
  });

  it('uses param format when tag parses (canonical path)', () => {
    const out = buildModelDisplay({ name: 'qwen3:72b', provider: 'ollama' }, {});
    expect(out.description).toBe('≈ 72B parameters • Ollama');
  });

  it('falls back to curated when tag has NO param count', () => {
    const curated: Record<string, string> = {
      'some-org/foo:latest': '16GB runtime',
    };
    const out = buildModelDisplay({ name: 'some-org/foo:latest', provider: 'ollama' }, curated);
    expect(out.description).toBe('16GB runtime • Ollama');
  });

  it('falls back to provider-only when tag has no param marker', () => {
    const out = buildModelDisplay({ name: 'some-org/foo:latest', provider: 'lmstudio' }, {});
    expect(out.description).toBe('LM Studio');
  });

  it('decimal parameters parse correctly', () => {
    const out = buildModelDisplay({ name: 'llama3.2:3.2b', provider: 'ollama' }, {});
    expect(out.description).toBe('≈ 3.2B parameters • Ollama');
  });

  it('providerLabel maps lmstudio to "LM Studio"', () => {
    expect(providerLabel('lmstudio')).toBe('LM Studio');
    expect(providerLabel('ollama')).toBe('Ollama');
  });
});
