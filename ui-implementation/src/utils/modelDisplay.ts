// v0.3.2 (0h): Dropdown display helpers.
//
// - Labels show the full Ollama tag (`gemma4:26b`), never the collapsed
//   family name — otherwise two installed variants become identical rows.
// - Subtitles uniformly use "≈ NB parameters • Provider" when the tag
//   parses. This replaces the old curated-marketing-copy dict
//   ("OpenAI efficient model", "Strong reasoning", etc.) that produced
//   inconsistent subtitle formats across rows. Curated strings still
//   accepted as last-resort fallback ONLY when the tag can't parse a
//   param count, because the alternative is showing nothing useful.
// - We never write "Custom model" at the user.

export type ProviderName = 'ollama' | 'lmstudio';

export interface ModelDisplayInput {
  name: string;
  provider: ProviderName;
}

export interface ModelDisplayOption {
  label: string;
  description: string;
}

export function providerLabel(provider: ProviderName): string {
  return provider === 'lmstudio' ? 'LM Studio' : 'Ollama';
}

export function buildModelDisplay(
  model: ModelDisplayInput,
  curatedDescriptions: Record<string, string>,
): ModelDisplayOption {
  const provider = providerLabel(model.provider);
  const paramMatch = model.name.match(/:(\d+(?:\.\d+)?)b/i);

  let description: string;
  if (paramMatch) {
    // Canonical: "≈ 20B parameters • Ollama". Uniform across all rows.
    description = `≈ ${paramMatch[1]}B parameters • ${provider}`;
  } else {
    // No parseable param count — try curated blurb, else just provider.
    const curated = curatedDescriptions[model.name];
    description = curated ? `${curated} • ${provider}` : provider;
  }

  return { label: model.name, description };
}
