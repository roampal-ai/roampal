# Roampal Desktop v0.3.2.1 — 2026-04-22

**Release Date:** 2026-04-22
**Type:** Hotfix — remove hardcoded `num_gpu=99` Ollama override

## Summary

A customer found that Roampal could not load certain models post-v0.3.2 install,
surfacing this Ollama error in the chat:

```
Model error: {"error":"memory layout cannot be allocated with num_gpu = 99"}
```

The `= 99` in the error is the exact value Roampal was forcing in every Ollama
request payload. This override (present since an earlier release) was intended
to ensure full GPU offload on systems where Ollama's auto-detection was too
conservative. Modern Ollama versions handle layer offload appropriately on
their own, and on hardware where the preferred backend can't fit all layers
the forced value produced a hard load failure rather than a graceful fallback.

v0.3.2.1 removes the override at all three Ollama-client call sites. Ollama's
own layer auto-detection now applies, and users can still opt back in to
forced max-offload via the `OLLAMA_NUM_GPU` environment variable.

## Why this matters

**v0.3.2 Section 0a (universal provider error handling) is what surfaced this
visibly.** Before v0.3.2, the same failure terminated the response stream
silently. The customer report was only possible because 0a made the raw
Ollama error reach the chat window. The hotfix closes the loop by removing
the config that triggered the error in the first place.

## Scope

### 1. Remove `num_gpu=99` override from `ollama_client.py`

**File:** `ui-implementation/src-tauri/backend/modules/llm/ollama_client.py`

Three sites previously set `options["num_gpu"] = 99`:

| Line | Path | Before | After |
|---|---|---|---|
| ~173 | `/api/generate` payload builder | `options["num_gpu"] = 99` | Removed |
| ~445 | `/api/chat` with tools | `options = {"num_gpu": 99}` | `options = {}` + guard for empty dict |
| ~684 | `stream_with_tools` | `{"num_ctx": ..., "num_gpu": 99}` | `{"num_ctx": ...}` |

The corresponding log line at `:686` was updated to drop the `(num_gpu=99)`
suffix. `num_ctx` handling is unchanged — context-window sizing is still
managed per-model via `config/model_contexts.py`.

**No new config, no new env-var reads, no schema changes.** If a user
explicitly sets `OLLAMA_NUM_GPU` in their environment, Ollama itself honors
it (we just stop *overriding* that preference).

### 2. Version bump

- `ui-implementation/package.json`: `0.3.2` → `0.3.2.1`
- `ui-implementation/src-tauri/Cargo.toml`: `0.3.2` → `0.3.2.1`
- `ui-implementation/src-tauri/tauri.conf.json`: `0.3.2` → `0.3.2.1`

### 3. Auto-updater endpoint

`website/updates/latest.json` → version bumped to `0.3.2.1` so existing
v0.3.x installs see the update notification on next poll.

## Non-goals

- No new user-facing "VRAM-aware offload" feature. The removal is a pure
  revert to Ollama defaults — no Roampal-side GPU detection logic added.
- No BIOS/driver/backend advice baked into the UI. Users on unusual
  hardware (iGPUs, Vulkan-only GPUs, dynamic VRAM) should configure their
  Ollama install per their vendor's guidance.

## Files touched

- `ui-implementation/src-tauri/backend/modules/llm/ollama_client.py` — 3 override sites removed
- `ui-implementation/package.json` — version bump
- `ui-implementation/src-tauri/Cargo.toml` — version bump
- `ui-implementation/src-tauri/tauri.conf.json` — version bump
- `dev/docs/releases/v0.3.2.1/RELEASE_NOTES.md` — this doc
- `website/updates/latest.json` — version bump (separate repo)

## Tests

- Existing `tests/unit/test_ollama_client.py` — 22 passed, unchanged
- No new test needed. The override was a single hardcoded value with no
  conditional branching; removing it does not introduce new code paths to
  cover. Unit tests continue to verify payload shape, tool retries, stale-
  model handling, and universal error handling from v0.3.2.

## Coordination with core

Not applicable. `num_gpu` is an Ollama-specific payload option; roampal-core
(MCP server) does not directly issue chat/generate calls against Ollama —
it forwards prompts through whatever client the host AI tool (Claude Code,
OpenCode, Cursor) is wired to. No core-side change required.
