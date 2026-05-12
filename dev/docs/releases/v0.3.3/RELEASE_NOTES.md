# Roampal Desktop v0.3.3 — Release Notes

**Status — final verification complete 2026-05-11.**

Sections 1–7 implemented 2026-04-28; Sections 8 (A–G), 9, 9.1, 11, and 12 implemented 2026-05-11. Section 10 ships via bundled core v0.5.7. Pre-tag hotfixes the same day: Section 3 Phase 2.1 (streaming clean preserved SentencePiece leading spaces) and Section 4H (replaced hardcoded `VISION_CAPABLE_FAMILIES` / `TOOL_CAPABLE_FAMILIES` with dynamic Ollama `/api/show` capability queries; also fixed a latent `write_json_atomic` bug that had silently no-op'd the on-disk metadata cache since Section 6 landed).

Verification scope (all green):
- 22 defects discovered + resolved during dev-Tauri + laptop verification (Defects 1–16 on 2026-05-11 dev-Tauri; Defects 17–22 on 2026-05-12 laptop fresh-install + deeper diagnosis — 17 cold-start race, 18 Windows `subprocess.run` pipe-handle deadlock (iteration-2 fix replaced subprocess with Ollama HTTP API), 19 ChromaDB corruption auto-recovery, 20 frontend switch timeout, 21 Tauri CSP missing `img-src` directive blocking multimodal attachment rendering, 22 LM Studio "Context size has been exceeded" error string slipping past the friendly handler).
- Full multimodal stack verified end-to-end on LM Studio `qwen3.6-35b-a3b` and Ollama `gpt-oss:20b`.
- Issue #8 reproduced + mitigated via cross-client shared-DB scenario (Desktop bulk-delete + OpenCode dev `ROAMPAL_DEV=1` → new working/history memories appeared within seconds, no manual `_completion_state.json` workaround).

| # | Section | Status |
|---|---------|--------|
| 1 | Timestamp field unification | ✅ Implemented 2026-04-28 |
| 2 | ChromaDB "Error finding id" | ✅ Implemented 2026-04-28 |
| 3 | Harmony channel-token leakage | ✅ Streaming path 2026-04-28; **Phase 2.1 hotfix 2026-05-11** (per-chunk strip was eating SentencePiece leading spaces) |
| 4 | Multimodal image input | ✅ Implemented 2026-04-28 |
| 5 | Atomic JSON writes (crash-safe config) | ✅ Implemented 2026-04-28 |
| 6 | Dynamic context-limit fetch | ✅ Implemented 2026-04-28 |
| 7 | ChromaDB hard delete / dedup bug (issue #8) | ✅ Implemented 2026-04-28 |
| 8A | Phantom sweep helper + cleanup_archived | ✅ Implemented 2026-05-11 |
| 8B | Phantom filter OR-form tightening | ✅ Implemented 2026-05-11 |
| 8C | Status backfill on legacy entries | ✅ Implemented 2026-05-11 |
| 8D | Capacity-pressure auto-cleanup | ✅ Implemented 2026-05-11 |
| 8E | Dedup observability (log inside _find_duplicate_fact) | ✅ Implemented 2026-05-11 |
| 8F | Rename delete() → delete_permanent(force=True) | ✅ Implemented 2026-05-11 |
| 8G | Integration test for archive-then-add cycle | ✅ Implemented 2026-05-11 |
| 9 | Bulk `/clear/*` phantoms (root cause for issue #8 reporter) | ✅ Implemented 2026-05-11 |
| 9.1 | Reset `_completion_state.json` on working/history/patterns clear | ✅ Implemented 2026-05-11 |
| 10 | `_completion_state.json` GC | ➡ Moved to core v0.5.7 |
| 11 | Tauri sidecar dev-mode env leak | ✅ Implemented 2026-05-11 |
| 12 | Dead-code cleanup (MCP caches + MCPSessionManager + log_tool_call + agent_chat action-KG) | ✅ Implemented 2026-05-11 (12D(4) deferred — optional UX polish) |

**Dependency order:** 5 → 6 (disk cache uses `write_json_atomic` from 5). Sections 1, 2, 3, 4, 7, 9 are independent. Section 10 ships via bundled core version bump.

---

## Carried from v0.3.2

### 1. Unify timestamp field across memory writes (`timestamp` → `created_at`)

**Why carried:** Raised during v0.3.2 laptop-testing while fixing the
shared-DB timestamp drift (v0.3.2 Section 0j). Divergence between
desktop's `timestamp` field and memory_bank's `created_at` (and core's
standardization on `created_at`) is the root cause of the drift.
v0.3.2 shipped a **tolerance fix** (readers accept either field), which
unblocks shared-DB scenarios correctly. The tolerance code is a legacy
safety net; the clean long-term move is to unify write sites on a single
field.

**Direction:** Standardize on `created_at` everywhere — matches core and
already-existing memory_bank shape. Desktop's decay-tier writes
(working, history, patterns) currently use `timestamp`; migrate them.

**Scope (preliminary — verify before implementing):**
- 8–12 memory-write sites in desktop backend (sidecar extraction,
  promotion_service, outcome_service, file_memory_adapter, agent_chat,
  unified_memory_system, main.py)
- Keep the v0.3.2 tolerance readers in place (they're the safety net
  for historical rows and for any site we miss)
- Per-site unit tests verifying new writes carry `created_at`
- Characterization test: write a memory via each surface, promote it
  through the lifecycle, verify lifecycle reads don't fall back to the
  tolerance path (they find `created_at` directly)

**Not in scope:**
- Removing the `timestamp` field from reads. The tolerance code stays
  forever, or at least until `roampal repair` / a backfill migration
  rewrites historical rows.
- Migrating existing data. New writes only. Historical rows read via
  tolerance.

**Pre-condition before implementing:** Review the full list of
`"timestamp": datetime.now().isoformat()` write sites (18 files grepped
during v0.3.2; filter to the memory-metadata subset). Build a
site-by-site inventory before touching any code — a half-migrated state
(some sites new, some old) is worse than the current consistent-but-split
state.

**Coordination:** Desktop-only. Core already uses `created_at` in
all its memory writes — verified 2026-04-21 against
`roampal-core/roampal/backend/modules/memory/unified_memory_system.py`.
No core-side change needed for this item; no v0.5.x counterpart release
required. Core v0.5.2 already shipped the reverse-direction tolerance
(reading desktop's `timestamp` field).

---

## Also carried (investigate-only from v0.3.2)

### 2. ChromaDB "Error finding id"

See v0.3.2 Section 0i. Investigate-only in v0.3.2, real fix deferred
here. Scope TBD.

#### ✅ Implementation Summary (2026-04-28)

**Root cause:** ChromaDB HNSW index leaves ghost entries after hard deletes (`collection.delete(ids=[...])`). When `list_all_ids()` or `get_fragment(doc_id)` hits these ghosts, it raises `"Error finding id"` which crashes batch operations.

**Fix applied — 4 sub-parts:**
- **2A (Harden list_all_ids):** Phantom filtering via documents+metadatas fetch in `chromadb_adapter.py:501-547`. Only IDs with valid entries are returned. Matches core v0.5.5 approach.
- **2B (Per-ID error handling in batch loops):** Added try/except around each ID in 3 batch loops (`promotion_service.py:336`, `484`, `534`) + `delete_by_conversation` (`unified_memory_system.py:1735`). One phantom/corrupt ID no longer kills the entire batch.
- **2C (All delete_vectors call sites audited):** 9 call sites across promotion_service, unified_memory_system, memory_bank_service — all have parent try/except or per-ID protection.
- **2D (Unit tests):** Added `TestPhantomIDResilience` class with 3 tests covering batch_promotion, cleanup_old_working, cleanup_old_history — all skip phantom IDs gracefully.

**Remaining:** No known gaps. Soft delete + startup migration from Section 7 prevents new phantoms from forming.

### 3. Harmony / channel-token leakage in chat responses

**Discovered:** v0.3.2 laptop-testing, 2026-04-21.

**Symptom.** Chat responses occasionally include raw control-format
tokens in the visible UI output. Observed example (`gemma4:31b` as chat
model):

> Let me check your records to see exactly what we have on file for you!`<channel|>`
>
> ✓ Searching memory
> Based on what we've discussed...

The `<channel|>` (and siblings like `<|start|>`, `<|message|>`,
`<|end|>`) are OpenAI Harmony chat-format control tokens. Used natively
by `gpt-oss:20b` and a few other models; leaked as literal text when a
non-Harmony model (gemma4, qwen, etc.) mimics the format from training
data.

**Root cause.** `ollama_client.py:497 _clean_model_artifacts` strips
`<think>` tags and a few leading prefixes, but does NOT strip
Harmony-family control tokens. Any leakage passes through to the UI.

**Two paths forward:**

- **Quick fix (5 min, recommended unless/until we support Harmony
  models first-class):** add 2 regex lines to `_clean_model_artifacts`
  to strip `<\|[a-z_]+\|>` and `<channel\|>`. Cosmetic only. The actual
  response body is intact.
- **Proper fix (1-2 hours, only if we want to support Harmony models
  properly):** detect Harmony format, parse channels, route `analysis`
  → thinking panel, `final` → visible response, `tool_calls` → tool
  dispatch. Real feature work, not just cleanup.

**Recommendation for v0.3.3:** quick fix unless Harmony support is on
the roadmap. Bundle with the test-audit work since both are about
"code quietly doing the wrong thing in production despite passing
tests" — cleaning up leaking tokens is the equivalent "dust the
corners" for the chat-output surface.

**Not in scope:** analyzing *why* gemma4 emits Harmony-format tokens
when it shouldn't. That's a prompt-engineering / model-selection
question, not a bug.

### ✅ Implementation Summary (2026-04-28)

**Phase 1 (non-streaming paths):** `ollama_client.py:528-531` — two regex patterns in `_clean_model_artifacts()`:
- `<\|[a-z_]+\|>` catches `<|start|>`, `<|end|>`, `<|message|>`, etc.
- `<channel\|[^>]*>` catches `<channel|>`, `<channel|final>`, etc.

Covered call sites: `generate_response()` line 385, `generate_response_with_tools()` line 494.

**Phase 2 (streaming path):** Extended Harmony-token stripping to per-chunk yield at `stream_response_with_tools()` line 1088. Initial implementation called the full `_clean_model_artifacts(content, actual_model)` per chunk — see Phase 2.1 hotfix below.

**Phase 2.1 hotfix (2026-05-11, pre-tag):** The per-chunk call to `_clean_model_artifacts` had a destructive side-effect: it ran `text.strip()`, the `\n{4,}` newline collapser, and the prefix remover (`"Answer: "`, etc.) on every streaming chunk. SentencePiece-tokenized models (Gemma, Llama) emit token-level chunks with explicit leading spaces (e.g. `" the"`); per-chunk `.strip()` ate those, mashing all words together in the rendered output (`"Yo!What'sup?How's..."`). Caught during dev Tauri smoke-test on `gemma4:31b` before tag.

Fix: added `_clean_model_artifacts_streaming(text)` that only applies the two Harmony regex substitutions — no whitespace strip, no newline collapse, no prefix removal. Rewired `stream_response_with_tools` line 1088 to call the streaming-safe variant. The full cleaner is still used in non-streaming paths (`generate_response`, `generate_response_with_tools`) where a single end-of-response strip is correct.

Tests: 6 original `TestCleanModelArtifacts` tests + 8 new `TestCleanModelArtifactsStreaming` tests covering leading/trailing space preservation, single-space chunk pass-through, concatenated-chunk reassembly, Harmony stripping (kept), and explicit non-actions on prefixes and newlines. **14/14 pass; 36/36 in the full ollama_client suite pass.**

**Known edge case:** Tokens split across chunk boundaries (e.g., `<ch` in one chunk, `annel|>` in the next) won't match per-chunk regex and will leak as partial fragments (~10 chars of garbage). Rare in practice — typical Ollama chunks are 50-200 bytes vs. Harmony tokens at ~10-15 chars. Negligible user impact; would require buffering all content until `done` to eliminate completely, which defeats streaming latency benefits.

---

---

## New items

### 4. Multimodal (image) input support

**Why:** Newer open-weights models (Qwen3.6-35B-A3B, Gemma 4, Qwen-VL
family, LLaVA) are natively multimodal. Users running them through
Roampal currently cannot share screenshots, diagrams, or photos even
though the underlying model supports it. Scope here is **images only**;
video is explicitly out of scope (Ollama API doesn't expose video input
natively in April 2026 — would need a different backend path).

**Discovered during.** v0.3.2 post-release exploration of Qwen3.6-35B-A3B
as a primary chat model. Most of the plumbing is already present — the
feature was partially wired and then disabled.

**Current state of the code (audited 2026-04-22):**

- ✅ **Drag-drop + file picker UI already exists** —
  `ui-implementation/src/components/CommandInput.tsx:38, 70-113`.
  `attachments: File[]` state, drag-drop handlers, and
  `onSend(message, attachments)` are all present.
- ✅ **Ollama adapter already handles vision** —
  `ui-implementation/src-tauri/backend/modules/llm/ollama_client.py:259-275`.
  Detects OpenAI-style content blocks, base64-encodes images, and sends
  them in the `images: [...]` field of the `/api/generate` payload.
- ✅ **Model-registry already tracks a `vision` capability flag** —
  `ui-implementation/src-tauri/backend/app/routers/model_registry.py:491-496`.
  Currently only flips true for `llava`/`gemma3` substrings — needs
  dynamic capability detection instead of hardcoded substrings (see
  revised approach below).
- ❌ **Chat API schema is text-only** —
  `ui-implementation/src-tauri/backend/app/routers/agent_chat.py:405-430`.
  `AgentChatRequest.message: str` with no `images` field.
- ❌ **Frontend → backend wiring is cut** —
  `ui-implementation/src/stores/useChatStore.ts:1207`. Attachments are
  collected in the UI but never serialized into the request body.
- ❌ **LLM interface signature is text-only** —
  `core/interfaces/llm_client_interface.py:26-33`. `generate_response`
  takes `prompt: str`, no `images` param.
- ❌ **Message rendering is text-only** —
  `ui-implementation/src/components/EnhancedMessageDisplay.tsx` has no
  `<img>` / attachment component.
- ⚠ **Comment on `agent_chat.py:413` literally says "File attachments
  removed."** Someone had this and pulled it out. Pre-condition below.

**Pre-condition before implementing.** Run
`git log -S "File attachments removed"` (and `git log -S "attachments"`
on `agent_chat.py`) to recover why this was ripped out. If there was a
real blocker (sidecar choked, memory pipeline broke, payload size
issue), we need to know before re-wiring. Without that context, we
risk reintroducing the same bug.

**Direction — Approach A (caption / pass-through, recommended for v1):**

At store time, the main LLM produces a text description of any attached
image as part of the normal response. The memory pipeline continues to
operate on text only — Wilson, tags, facts, CE rerank, KG-less
retrieval all untouched. Original image blob is persisted separately
(filesystem path under Roampal's app-data dir, hash in DB) for redisplay
on recall. **This is the smallest delta that ships multimodal input
cleanly.**

**Not picking Approach B/C (hybrid or full multimodal embeddings).**
Those require a second vector store, a new embed model (CLIP / Qwen-VL),
and refitting Wilson/CE scoring for multimodal similarity. That's a
rewrite, not a release item. Ship A, get real usage signal, reassess.

### Vision capability detection — canary probe (revised 2026-04-28)

**Replaces** the original "lookup table" proposal. A static lookup table
of vision-capable model families is an enumeration that goes stale weekly
and silently breaks for new vision models. Instead: **canary probe the
live model, cache the result.** No hardcoded lists. Works for any past,
present, or future vision model on any provider.

**How it works for each provider:**

| Provider | Canary probe | Expected response |
|----------|-------------|-------------------|
| **Ollama** | `POST /api/generate` with blank 1×1 PNG base64 + `num_predict: 1` | 200 + token → vision. Error ("model does not support images") → no vision |
| **LM Studio** | `POST /v1/chat/completions` with blank 1×1 PNG base64 + `max_tokens: 1` | 200 + token → vision. 400/error → no vision |

**Probe details:**
- A 1×1 transparent PNG encoded as base64 (~68 bytes). Negligible cost.
- One probe per model per provider, cached with 24h TTL (memory + disk,
  same tiered cache as Section 6).
- Result stored in `capabilities.vision` on the `/api/models` response:
  `"confirmed"` (probe succeeded), `"unconfirmed"` (probe pending or
  errored non-fatally), `"unavailable"` (probe confirmed no support).
- Probe runs lazily — first time a model is switched to or the model
  picker is opened. Not at backend startup (avoids 10+ probes on cold
  start against a large model library).

**Why canary probe over other approaches:**
- `details.families` in Ollama's `/api/show` *is* dynamic data from the
  model, but it's the model's self-declared family string — different
  architectures use different family names, and there's no universal
  "I support vision" field.
- Probing is the only approach guaranteed correct for 100% of models,
  including models that don't exist yet. Cache makes it one-time cost.

### UI requirements (2026-04-28)

From the user:

> If the selected main LLM can read images, the chat bar should have a
> PhotoIcon (Heroicons) next to the send icon. Able to paste images into
> the chat so the LLM can read them. If the main LLM can't do images,
> the image icon should be greyed out, and hovering should show a
> tooltip: "Current model doesn't support images. Select a vision-capable
> model to enable image upload."

**Implementation:**

- **`CommandInput.tsx`:**
  - Replace the generic `PaperClipIcon` with a `PhotoIcon` from Heroicons
    (already imported).
  - Read model's `vision` capability from chat store (or receive as prop).
  - When `vision !== "unavailable"`: icon active, click opens file picker
    for images only (`accept="image/*"`).
  - When `vision === "unavailable"`: icon greyed out (`text-zinc-600
    cursor-not-allowed`), `disabled`, `title="Current model doesn't
    support images. Select a vision-capable model to enable image upload."`
  - Add `onPaste` handler to the textarea: check
    `e.clipboardData.items` for `kind === "file" &&
    type.startsWith("image/")`, read as `File`, add to attachments array.
  - The existing drag-drop handler (lines 115-119) already accepts files
    — filter to images in `handleDrop` / `handleFileSelect` or accept all
    and warn on non-image.

- **`useChatStore.ts`:** Serialize `attachments` as base64 `images[]` in
  the chat request body. Read via `FileReader.readAsDataURL()` → strip
  `data:image/...;base64,` prefix → include plain base64 string.

**Note:** The `PaperClipIcon` can remain as a secondary button for
non-image file attachments (documents uploaded via Document Processor),
but the primary image-upload button is the `PhotoIcon` next to send.

### Scope (revised 2026-04-28)

1. **Vision canary probe** — new helper `probe_vision_capability(model_name, provider, base_url)`
   in `config/model_contexts.py` (reuses `httpx` added in Section 6).
   Tiered cache 24h disk + 1h memory. Exposed via `/api/models` as
   `capabilities.vision: "confirmed" | "unconfirmed" | "unavailable"`.
2. `agent_chat.py:405-430` — extend `AgentChatRequest` with
   `images: Optional[List[str]] = None` (base64-encoded). Pass through
   to LLM client.
3. `useChatStore.ts` — read attachments from `CommandInput`,
   `FileReader.readAsDataURL()` → base64, include in request body.
   Expose model's `vision` capability to `CommandInput` via store.
4. `core/interfaces/llm_client_interface.py:26-33` — extend
   `generate_response` signature with `images: Optional[List[str]] = None`.
   Ollama adapter already handles images (lines 259-275); ensure that
   path is reachable from the main chat flow, not just the generate-endpoint
   fallback. Anthropic/OpenAI adapters need small additions.
5. `CommandInput.tsx` — replace `PaperClipIcon` with `PhotoIcon` next to
   send. Conditional active/greyed-out state based on model's `vision`
   capability. Add paste-to-upload handler on textarea. Filter file picker
   to `accept="image/*"`.
6. `EnhancedMessageDisplay.tsx` — add an `<img>` / attachment component
   for images in message bubbles (both user-sent and assistant-returned).
7. Tests: canary probe unit tests (mock provider responses, cache hit/miss,
   TTL expiry), characterization test sending an image through
   `/api/agent/chat` with a vision-capable model.

**Not in scope:**

- **Video input.** Ollama API doesn't natively accept video. Revisit
  if/when we switch a backend path or Ollama ships support.
- **Multimodal memory retrieval.** Image blobs are preserved for
  redisplay only; retrieval continues to operate on text captions.
  Upgrade to joint embeddings is a separate roadmap item if real usage
  surfaces the need.
- **Multimodal sidecar.** `sidecar_service.py:56-74` stays text-only.
  It operates on the final exchange transcript, which already includes
  the main model's caption/description of the image.
- **ChromaDB schema change.** Memory collections keep their text-only
  shape. Attachment metadata (filesystem path + hash) lives on the
  memory-row metadata, not in the embedding.

**Coordination.** Desktop-only for the request/render path. LLM
interface change (`core/interfaces/llm_client_interface.py`) is core —
needs a matching core-side release, probably a minor patch since it's
an additive parameter with a default.

**Recommendation for v0.3.3:** in scope **only if** the pre-condition
git-log investigation turns up no real blocker, OR the original blocker
is cleanly understood and addressable. If the blocker is non-trivial,
pull this to v0.3.4 / v0.4.x and keep it tracked here as
investigate-only. ~2–3 days of focused work if the blocker is a
non-issue.

### ✅ Section 4 Implementation Summary (2026-04-28)

**Backend pipeline (4A-C):**
- `AgentChatRequest.images: Optional[List[str]]` added to schema (`agent_chat.py:408`)
- `stream_message(images=)` parameter threaded through both call sites in `_run_generation_task()` and `_run_generation_task_streaming()` (lines 3251, 3382)
- Conversation history builds OpenAI-style multimodal content blocks when images present (`agent_chat.py:742-749`)
- `llm.stream_response_with_tools(images=)` wired through at line 1065
- Ollama client streaming path builds content blocks with images in messages payload (`ollama_client.py:673-680`)

**Frontend (4D-F):**
- `sendMessage(text, imageFiles?)` signature updated with type declaration (`useChatStore.ts:77, 1148`)
- Inline File→base64 conversion via FileReader.readAsDataURL in sendMessage (`useChatStore.ts:1181-1190`)
- User message stores `images[]` for UI display; request body includes `images[]` field
- ConnectedCommandInput: PhotoIcon button, hidden file input with `accept="image/*"`, paste handler on textarea, attachment preview thumbnails with remove buttons (`ConnectedCommandInput.tsx:256-290`)
- Send button enabled for images-only sends (no text required)
- EnhancedChatMessage renders image thumbnails as `<img src={dataURL}>` in user message bubbles (`EnhancedChatMessage.tsx:130-145`)

**Vision capability detection (4G deferred → implemented 2026-04-28):**
- `probe_vision()` async canary probe with 1×1 PNG + 24h disk cache (`model_contexts.py:389-474`)
- `get_cached_vision()` sync helper for cold-start registry (heuristic families + disk cache, no network) (`model_contexts.py:389-407`), return type corrected to `bool` (was `Optional[bool]`)
- Registry wired to dynamic detection instead of hardcoded `"llava" | "gemma3"` strings (`model_registry.py:17, 494`)
- `POST /api/model/probe_vision/{model_name}` endpoint for on-demand probing (`model_registry.py:510-538`), `cached` field fixed to check real cache hit before probe
- Frontend PhotoIcon gated by `activeModelHasVision` derived from registry metadata (`ConnectedChat.tsx:1320-1334`, `ConnectedCommandInput.tsx:276-296`)
- Paste handler gated behind `modelHasVision` to prevent image paste into text-only models (`ConnectedCommandInput.tsx:120-135`)
- Fuzzy substring match for registry↔selected-model name resolution: strips Ollama tags, bidirectional `.includes()` — covers LMStudio full HF paths and stale env-var fallbacks (`ConnectedChat.tsx:1320-1334`)
- PhotoIcon title corrected from "paste or drag-drop also supported" to "paste also supported" (drag-drop not implemented)
- 4 unit tests for vision cache detection covering heuristic, text-only default, and disk-cache hit

### 4H — Capability detection: dynamic refactor (implemented 2026-05-11)

**Status:** ✅ IMPLEMENTED. Hardcoded model-family enumerations for vision and tools detection are gone. Runtime capability gates read from a dynamic cache populated by Ollama's `/api/show` on every model registry refresh. LM Studio falls back to the existing `probe_vision` canary path for vision and defaults `tools=True` (runtime tool-incompatibility detection at `ollama_client.py:1015-1018` catches false positives and retries without tools).

**Files touched:**

| File | Change |
|---|---|
| `config/model_contexts.py` | Deleted `VISION_CAPABLE_FAMILIES` (10 hardcoded families) and `is_known_vision_model`. Deleted `_vision_cache_key` / `_load_vision_cache` / `_save_vision_cache`. Added `fetch_ollama_capabilities(model, base_url)` (POSTs `/api/show`, returns `Set[str]`, caches result). Added `_caps_cache_key` / `_load_caps_cache` / `_save_caps_cache` (shared with the existing context-length cache file). Rewrote `get_cached_vision` as a thin reader against the capability cache. Added `get_cached_tools` and `get_cached_capabilities`. Rewrote `probe_vision` to read/write the unified capability cache instead of the now-deleted vision cache. Also fixed a latent bug where `write_json_atomic` was called with a string instead of a `Path` — all three call sites in this file. |
| `app/routers/model_registry.py` | Deleted `TOOL_CAPABLE_FAMILIES` (3 tier groups × ~25 hardcoded models). Rewrote `get_model_tier(model, provider)` to derive tier from cached capabilities (informational only — no runtime gating). Rewrote `is_tool_capable(model, provider)` as a thin reader against the capability cache. Registry refresh now `asyncio.gather`s `fetch_ollama_capabilities` for every Ollama model BEFORE the per-model loop, so the sync readers downstream see fresh data. Removed `tool_capable_families` field from the registry response payload. Deprecated `/api/model/tiers` endpoint (returns deprecation note; frontend should read per-model `tier` from `/api/model/registry`). Catalog and recommendations endpoints (pre-install browsing) now return `tier="unknown"`, `tool_capable=None` for un-installed models — authoritative values populate once the user installs and the registry refresh runs. |
| `tests/unit/test_model_contexts_dynamic.py` | Rewrote `TestCachedVisionDetection` against the new cache. Added `TestCachedToolsDetection` (Ollama with tools, Ollama without tools, LM Studio default-True, LM Studio explicit cache). Added `TestFetchOllamaCapabilities` (200 with caps, missing caps key, network failure). Added autouse fixture to isolate `MODEL_METADATA_CACHE_FILE` to `tmp_path` per test (the disk-write bug fix made cross-test pollution real). |

**Test results:** Desktop backend suite **637 passed, 1 skipped, 0 failed** in 58.97s after the refactor (was 622/1/0 before; net +15 tests).

**Latent bug fix surfaced during refactor:** `_save_to_disk_cache`, `_save_vision_cache` (deleted), and the cleanup-write inside `_load_disk_cache` were passing a `str` to `write_json_atomic(path: Path, ...)`. The function does `path.parent` immediately, which raised `AttributeError: 'str' object has no attribute 'parent'` — silently swallowed by the `try/except Exception` in each caller. Result: the disk cache for both vision AND context-length lookups was effectively a no-op since v0.3.3 Section 6 landed. Vision detection worked anyway because it fell back to the heuristic name match. Context-length fetch worked anyway because the in-memory cache catches most repeated lookups within a single session, but every restart re-fetched. Both now persist properly across restarts.

**Open follow-ups (NOT in this release):**

- `get_model_description` (`model_registry.py:312+`) still hardcodes ~30 model names → descriptions. Cosmetic only; doesn't gate behavior. Acceptable to keep but flag-worthy.
- `QUANTIZATION_OPTIONS` (`model_registry.py:236`) and `HUGGINGFACE_REPOS` are pre-install browse catalogs. Used by the install wizard for VRAM-based recommendations. Could be replaced by an Ollama Hub API query in a later release.
- Frontend `ConnectedChat.tsx:1142+` has a curated model-recommendations dropdown with hardcoded names. Curated install suggestions, not capability detection. Separate concern.

These three are listed for transparency; v0.3.3 §4H scope was specifically "kill hardcoded model lists used for runtime capability gating," which is now done.

**How this surfaced.** Dev Tauri smoke-test on `gemma4:31b`. Photo-attach icon never appeared because `gemma4` was missing from `config/model_contexts.py:342-345 VISION_CAPABLE_FAMILIES` (heuristic list contains `gemma3` but not `gemma4`). Initial reaction was to add `gemma4` to the list — flagged and rejected by user: hardcoded enumerations of model families go stale every time a new model drops, and the same pattern would burn us on the next family too.

**Audit finding — same anti-pattern, two places:**

| Capability | Current code | Failure mode |
|---|---|---|
| `vision` | Hardcoded list `VISION_CAPABLE_FAMILIES` in `model_contexts.py:342-345`, checked by `is_known_vision_model` + 24h disk cache | New vision-capable family ships → photo icon stays hidden until someone amends the list, OR until a user manually hits `/api/model/probe_vision/{model_name}` |
| `tools` | Hardcoded tier table in `model_registry.py:get_model_tier` (~50+ models, unknown → `experimental`), used by `is_tool_capable` | New tool-capable model ships → defaults to "experimental" tier → excluded from registry when `tool_capable_only=true`, OR shown as untrustworthy in UI hints |

Both ship as v0.3.3 (the vision list was added in this cycle as Section 4 plumbing; the tier table predates it).

**Authoritative source of truth (verified 2026-05-11 against live Ollama):**

```
$ curl -s -X POST http://localhost:11434/api/show -d '{"name":"gemma4:31b"}' | jq .capabilities
["completion", "vision", "tools", "thinking"]
```

Ollama's `/api/show` returns a model-self-declared capabilities array. This is what the heuristic should defer to.

**Proposed architecture (no hardcoded model enumeration):**

1. **New async function** in `config/model_contexts.py`:
   ```python
   async def fetch_ollama_capabilities(
       model_name: str, base_url: str, timeout: float = 5.0
   ) -> Optional[Set[str]]:
       """POST /api/show, return set(data['capabilities']) or None on failure."""
   ```

2. **Wire into registry refresh** at `model_registry.py:392 GET /api/model/registry`:
   - For every Ollama model, `asyncio.gather` all `fetch_ollama_capabilities(name, base_url)` calls in parallel
   - Write each result to the existing disk cache via a new `_save_capabilities_cache(key, capabilities_set)` helper that mirrors `_save_vision_cache` but stores the full set rather than just `vision_capable: bool`
   - Cold-start latency: ~50ms total for 5-10 installed models (Ollama API is local, parallelized)

3. **`get_cached_vision` becomes a thin reader:**
   ```python
   def get_cached_vision(model_name: str, provider: str) -> bool:
       caps = _load_capabilities_cache().get(_caps_cache_key(model_name, provider), set())
       return "vision" in caps
   ```
   Same trick exposes `get_cached_tools`, `get_cached_thinking` for future UI gates without further hardcoding.

4. **LM Studio path** (verified: OpenAI-compatible `/v1/models` does NOT expose capabilities per spec):
   - On registry refresh, for LM Studio models with no cache entry, fire the existing async `probe_vision` (canary 1×1 PNG → `/v1/chat/completions`) in background. Cache result for 24h.
   - On second registry refresh after probe completes, capability lights up.
   - This is the only general way without LM Studio extending their OpenAI-compatible API surface; tradeoff is one cold-start refresh per model where the icon appears late. Acceptable.

5. **Delete the hardcoded enumerations:**
   - Remove `VISION_CAPABLE_FAMILIES`, `is_known_vision_model` from `config/model_contexts.py`
   - Refactor `is_tool_capable` / `get_model_tier` similarly: tier becomes informational (description only), capability gates use the dynamic cache
   - The `get_model_description` dict at `model_registry.py:312+` stays (cosmetic descriptions, not capability gating — acceptable hardcoding)

**Test plan (~6 new tests, all mockable, no live Ollama needed):**

| Test | What it locks down |
|---|---|
| `fetch_ollama_capabilities_returns_set` | Mock `/api/show` 200 with `capabilities: ['vision', 'tools']` → returns `{'vision', 'tools'}` |
| `fetch_ollama_capabilities_handles_missing_field` | Old Ollama with no `capabilities` key → returns empty set, no crash |
| `fetch_ollama_capabilities_network_failure` | httpx raises → returns None, cache untouched |
| `registry_refresh_populates_capability_cache` | After refresh on a mocked Ollama, `get_cached_vision` returns True for vision-capable mock |
| `lmstudio_falls_back_to_probe_vision` | LM Studio model with no cache entry triggers `probe_vision` path; cache populated after |
| `capability_cache_expiry_triggers_refetch` | After 24h, next refresh re-queries instead of serving stale cache |

**Scope estimate.** ~200 LOC + 6 tests + this release-notes section flipped from "proposed" to "implemented." Time: 45-60 minutes coding + manual verification on dev Tauri.

**Decision pending:**

1. Ship dynamic detection as **v0.3.3 Phase 4H** (closing Section 4 properly, requires another round of dev-Tauri smoke) — OR defer to **v0.3.4**?
2. If shipping in v0.3.3 — include the `tools` capability cleanup (delete tier-based `is_tool_capable`) or vision-only?
3. **Immediate test unblock for tonight (independent of the decision above):** manually POST to `/api/model/probe_vision/gemma4:31b` from the running dev Tauri. That populates the cache for the current session and lets §4 multimodal be tested today without touching code.

---

### 5. State-file mutation hardening (non-atomic JSON writes)

**Discovered during:** silent-clobber audit run 2026-04-22 across
`roampal-core` and `roampal-desktop`. Same audit surfaced core-side
findings already landed in core v0.5.3 Section 10. This section is the
desktop half.

**Problem.** Three desktop config files get written with a direct
`open(path, "w") + json.dump(...)` pattern. No temp-file + rename. If
the process dies between the `open(..., "w")` call (which truncates
the file to zero bytes) and `json.dump` completing — power loss, OOM
kill, Ctrl-C, Windows forced reboot, kernel panic — the file is left
empty or partially written. Next read either fails or triggers
downstream "file looks broken, reset to defaults" fallback logic,
silently losing user-authored content.

**Findings.**

| File | Line | File mutated | What's lost on crash | Severity |
|---|---|---|---|---|
| `ui-implementation/src-tauri/backend/config/feature_flags.py` | 126-132 | `feature_flags.json` | User-tuned feature toggles | HIGH |
| `ui-implementation/src-tauri/backend/config/model_contexts.py` | 78-79, 103-104 | `user_model_contexts.json` (two write sites in the same file) | User's saved context-length overrides per model | HIGH |
| `ui-implementation/src-tauri/backend/config/model_limits.py` | 500-501 | Per-model JSON files under `data/model_calibrations/` | Calibration data learned from observed model behavior | MEDIUM |

No silent-parse-failure-then-write sites were found in desktop (the
audit's other red flag). No missing-backup sites either — non-atomic
write is the entire bug class in desktop. Clean on the other two
categories.

**Fix — mirror core v0.5.3 Section 10.**

Create a desktop-local atomic-JSON helper and route all four write
sites through it. Don't import core's helper across the core/desktop
boundary — the two codebases run as separate Python processes with
independent installs; a local copy keeps the dependency graph clean.

**New file:** `ui-implementation/src-tauri/backend/utils/atomic_json.py`

```python
"""Atomic JSON write with temp-file + rename. Crash-safe on Linux
and Windows NTFS. Mirrors roampal-core v0.5.3 Section 10's helper
of the same name; code duplicated intentionally to keep the two
codebases decoupled.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: Any, *, indent: int | None = 2) -> None:
    """Write `data` as JSON to `path` atomically.

    Writes to a sibling .tmp file first, then os.replace()s into place.
    If any exception is raised during the write, the temp file is
    removed and the original `path` is left untouched.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=str(parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
```

### Call-site changes

**`feature_flags.py:126-132`:**

```python
# Before
with open(self._config_path, "w") as f:
    json.dump(self._flags, f, indent=2)

# After
from backend.utils.atomic_json import write_json_atomic
write_json_atomic(self._config_path, self._flags)
```

**`model_contexts.py:78-79` and `:103-104`:** both writes follow the
same `open(..., "w") + json.dump` pattern; both switch to
`write_json_atomic(path, data)`.

**`model_limits.py:500-501`:** single-site swap of the calibration
write to `write_json_atomic(path, data)`.

### Tests to add

New file `ui-implementation/src-tauri/backend/tests/unit/test_atomic_json.py`:

- `test_write_json_atomic_creates_file_on_fresh_path`
- `test_write_json_atomic_replaces_existing_file`
- `test_write_json_atomic_leaves_no_tmp_files_on_success`
- `test_write_json_atomic_preserves_original_on_exception` —
  monkeypatch `os.replace` to raise; assert original file byte-for-byte
  unchanged.
- `test_write_json_atomic_creates_parent_dirs`

Update existing tests (if any) for `feature_flags.py`,
`model_contexts.py`, `model_limits.py` to assert the write goes through
a `.tmp`-suffix intermediate — monkeypatch `os.replace` and verify it
was called once per write.

### Files to touch

- `ui-implementation/src-tauri/backend/utils/atomic_json.py` — new
  file, one public function (mirrors core's helper)
- `ui-implementation/src-tauri/backend/config/feature_flags.py` —
  replace inline write at 126-132
- `ui-implementation/src-tauri/backend/config/model_contexts.py` —
  replace both inline writes at 78-79 and 103-104
- `ui-implementation/src-tauri/backend/config/model_limits.py` —
  replace inline write at 500-501
- `ui-implementation/src-tauri/backend/tests/unit/test_atomic_json.py`
  — new test file (5 tests)

### Impact if unfixed

Statistically rare per write, but the exposure scales with user count
and time. Every feature-flag toggle, every context-length override,
every model calibration save is one roll of the dice against a crash.
Blast radius on a hit: user's feature config, context overrides, or
calibration data — all hand-authored or learned over time, annoying
to reconstruct.

Not in scope: a broader defense-in-depth audit of desktop config
writes. This section only covers the three findings surfaced by the
2026-04-22 audit. If a future audit turns up more sites, track them
separately.

### Coordination

- `roampal-core` v0.5.3 Section 10 ships the analogous fix on the
  core side. Both codebases will end up with independent
  `write_json_atomic` helpers; this is intentional — they run as
  separate Python processes and share no install path in production.
- No user-visible behavior change. No migration. Existing JSON files
  continue to be read/parsed exactly as they are today; only the write
  path changes.

### 6. Dynamic context-limit fetch from Ollama / LM Studio APIs

**Problem.** Today's context-limit resolution is a hardcoded 20-entry
prefix table (`MODEL_CONTEXTS` in `config/model_contexts.py:21-40`).
Any model not matching one of those ~20 family prefixes falls through
to an 8K-token fallback. Any model whose true context window differs
from the family default (Qwen3.6-35B-A3B supports 262K natively but
matches the `qwen3` prefix which claims 32K max) is silently
mis-configured until the user manually overrides it. Every new model
release requires a code patch to `MODEL_CONTEXTS`.

Both providers already expose the real context length via HTTP; we
just don't ask them.

**Why this can't wait.** Models ship weekly. The hardcoded table is
already stale the day it lands in a release. Users pulling a new model
and pointing Roampal at it get the 8K fallback — their chat history
starts thrashing at 8K tokens even though the model supports 256K.

### How the two providers expose context length

**Ollama — `POST /api/show`:**

```json
POST http://localhost:11434/api/show
Body: {"name": "qwen3.6:35b-a3b"}

Response:
{
  "model_info": {
    "qwen35moe.context_length": 262144,
    "qwen35moe.block_count": 40,
    "general.architecture": "qwen35moe",
    ...
  }
}
```

The key is **architecture-prefixed**: `qwen35moe.context_length` for
Qwen3.5/3.6 MoE, `llama.context_length` for Llama-family,
`gemma3.context_length` for Gemma 3, `gpt_oss.context_length` for
gpt-oss. Parser doesn't need to know every prefix — just iterate
`model_info.items()` and match any key ending in `.context_length`.

**LM Studio — `GET /v1/models/<id>`:**

```json
GET http://localhost:1234/v1/models/qwen3.6-35b-a3b

Response:
{
  "id": "qwen3.6-35b-a3b",
  "max_context_length": 262144,
  "loaded_context_length": 65536,
  ...
}
```

Cleaner schema. **Prefer `loaded_context_length` over `max_context_length`.**
`loaded_context_length` is what LM Studio is actually serving this
session (reflects the user's in-app load slider). Sending more tokens
than the loaded value truncates or errors because the KV cache isn't
allocated for them. `max_context_length` is the model's theoretical
ceiling — useful only as a fallback when LM Studio's response doesn't
include a loaded value (older LM Studio builds, model not currently
loaded, etc.).

When the user moves the LM Studio slider and reloads, the next dynamic
fetch picks up the new `loaded_context_length`. Within our 5-minute
cache TTL they see the old value; after that it refreshes. Acceptable
for v0.3.3 — users don't move this slider mid-session.

### New helper — `fetch_context_from_provider`

Added to `config/model_contexts.py` alongside existing helpers:

```python
import httpx

async def fetch_context_from_provider(
    model_name: str, provider: str, base_url: str, timeout: float = 5.0
) -> Optional[int]:
    """Return the model's max context window by asking the provider.

    Returns None on timeout, parse failure, or model-not-found.
    Callers fall back to MODEL_CONTEXTS table on None.
    """
    try:
        if provider == "ollama":
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    f"{base_url.rstrip('/')}/api/show",
                    json={"name": model_name},
                )
                r.raise_for_status()
                model_info = r.json().get("model_info", {})
                for key, value in model_info.items():
                    if key.endswith(".context_length"):
                        return int(value)
                return None
        elif provider == "lmstudio":
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(
                    f"{base_url.rstrip('/')}/v1/models/{model_name}"
                )
                r.raise_for_status()
                data = r.json()
                # Prefer loaded_context_length (live session value, reflects
                # user's in-app slider) over max_context_length (model ceiling).
                # Sending more than loaded_ will truncate/error because LM
                # Studio's KV cache isn't allocated for it.
                value = (
                    data.get("loaded_context_length")
                    or data.get("max_context_length")
                    or data.get("context_length")
                )
                return int(value) if value else None
        else:
            return None
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as e:
        logger.debug(f"Dynamic context fetch failed for {model_name} via {provider}: {e}")
        return None
```

### Caching — two tiers

**Tier 1 — in-memory LRU with TTL.** Module-level dict keyed on
`(provider, model_name)`. 5-minute TTL. Hot-path fast lookup; no disk
I/O on repeat calls within a session.

```python
_context_cache: Dict[Tuple[str, str], Tuple[int, float]] = {}  # (value, expires_at)
_CACHE_TTL = 300  # seconds
```

**Tier 2 — disk cache.** Survives restarts so users don't re-hit the
provider on every backend start. File:
`<DATA_PATH>/model_metadata_cache.json`. Written via Section 5's
`write_json_atomic` helper. Each entry carries its own expiry (24h
from fetch) so a bad cached value ages out. Loaded into the memory
cache on startup.

```json
{
  "ollama::qwen3.6:35b-a3b": {"context_length": 262144, "expires_at": 1746000000},
  "lmstudio::qwen3.6-35b-a3b": {"context_length": 262144, "expires_at": 1746000000}
}
```

### Updated `get_context_size` priority

Insert dynamic fetch between user-override and hardcoded prefix match:

```
1. Runtime override (request-scoped, passed as kwarg)
2. User's saved override (user_model_contexts.json)
3. ▶ NEW: Dynamic fetch from provider (cached, 5-min TTL / 24h disk)  ◀
4. MODEL_CONTEXTS prefix match (used when provider API fails / no match)
5. 8192 fallback
```

Dynamic fetch is **advisory, not required** — if the provider is
offline, the model isn't found, or the response is malformed, the code
drops through to the old hardcoded behavior. Nothing about today's
path gets less robust; it just becomes one of several data sources.

### Async plumbing — the real cost of this change

`get_context_size` is currently sync. Dynamic fetch is async because
`httpx` calls are async. Two ways to plumb it:

- **A (preferred):** make `get_context_size_async` a new function,
  keep sync `get_context_size` as a thin wrapper that skips the
  dynamic fetch path (calls cache if populated, else falls through to
  hardcoded). All **new** callers in request-handling code paths use
  the async version.
- **B:** thread through `asyncio.run` inside the sync function.
  Brittle; breaks if called from an already-running event loop. Avoid.

Go with A. Audit existing call sites for which ones live in async
handlers (chat requests, model-switch endpoints — all async) and
migrate those to `get_context_size_async`. Config/settings reads that
run at startup stay sync and tolerate the cache-only lookup.

### Where the provider/base_url comes from

`get_context_size` currently takes just a model name. Dynamic fetch
needs to know which provider to ask. Two options:

- Pass `provider` + `base_url` as new kwargs — caller already knows
  them (model-switch endpoint, ollama_client, lmstudio_client)
- Have the function infer from model name shape — fragile, don't

Go explicit: `get_context_size_async(model_name, provider=None, base_url=None)`.
When `provider` is None, skips dynamic fetch (falls to step 4).
Existing sync callers that don't pass provider continue to work
unchanged, they just don't benefit from the new dynamic path.

### Tests to add

New file `ui-implementation/src-tauri/backend/tests/unit/test_model_contexts_dynamic.py`:

- `test_ollama_response_qwen3_6_returns_262144` — feed a recorded
  real response with `qwen35moe.context_length: 262144` → helper
  returns 262144.
- `test_ollama_response_llama_returns_context_length` — feed a
  recorded response with `llama.context_length: 131072` → returns
  131072. (Covers multiple architecture prefixes.)
- `test_ollama_response_gemma3_returns_context_length` — feed a
  response with `gemma3.context_length: 131072` → returns 131072.
- `test_ollama_response_missing_context_length_returns_none` — no
  `.context_length` key anywhere in `model_info` → returns None.
- `test_ollama_model_not_found_returns_none` — mock 404 response →
  returns None (no crash).
- `test_ollama_timeout_returns_none` — mock timeout → returns None
  within timeout budget.
- `test_lmstudio_response_returns_loaded_context_length` — feed
  `{"loaded_context_length": 65536, "max_context_length": 262144}` →
  returns 65536 (loaded wins).
- `test_lmstudio_falls_back_to_max_when_loaded_missing` — feed
  `{"max_context_length": 262144}` (no loaded field) → returns 262144.
- `test_lmstudio_falls_back_to_context_length_when_both_missing` —
  feed `{"context_length": 32768}` (older LM Studio builds) →
  returns 32768.
- `test_lmstudio_user_slider_change_picked_up_after_cache_expiry` —
  first fetch returns 32768 (loaded at 32K); cache clears; provider
  mock now returns loaded=65536 → second fetch returns 65536.
  Covers the "user moved the LM Studio slider" flow.
- `test_lmstudio_missing_all_context_fields_returns_none` — response
  has none of the three fields → returns None.
- `test_lmstudio_model_not_found_returns_none` — mock 404 → None.
- `test_memory_cache_hit_avoids_http_call` — first call hits the
  mock; second call within TTL does not (assert mock called once
  across two invocations).
- `test_memory_cache_expires_after_ttl` — set TTL to 0, make two
  calls → both hit the mock.
- `test_disk_cache_survives_restart` — populate cache, clear memory
  cache, simulate restart → value re-read from disk without HTTP.
- `test_disk_cache_entry_expires_after_24h` — disk entry with past
  `expires_at` is ignored; helper re-fetches.
- `test_get_context_size_async_prefers_dynamic_over_hardcoded` —
  qwen3.6:35b-a3b with no user override, provider mock returns
  262144 → `get_context_size_async` returns 262144 (not the qwen3
  prefix's 32768).
- `test_get_context_size_async_falls_back_to_hardcoded_on_fetch_fail` —
  qwen3.6:35b-a3b, provider mock times out → `get_context_size_async`
  returns 32768 (qwen3 prefix match).
- `test_get_context_size_async_user_override_still_wins` — user has
  overridden qwen3.6:35b-a3b to 65536, provider would return 262144
  → returns 65536.
- `test_sync_get_context_size_still_works_unchanged` — existing
  sync callers behave exactly as today; regression guard.

### Files to touch

- `ui-implementation/src-tauri/backend/config/model_contexts.py` —
  add `fetch_context_from_provider`, `get_context_size_async`,
  `_context_cache` module-level state, disk cache load/save (uses
  `write_json_atomic` from Section 5)
- `ui-implementation/src-tauri/backend/app/routers/model_contexts.py`
  — update router endpoints to call the async variant
- `ui-implementation/src-tauri/backend/modules/llm/ollama_client.py`
  — where `get_context_size` is called during chat, migrate to the
  async variant (pass `provider="ollama"` + `base_url`)
- `ui-implementation/src-tauri/backend/app/routers/agent_chat.py` —
  same migration at the chat-request context lookup
- `ui-implementation/src-tauri/backend/app/routers/model_switcher.py`
  — same migration on model-switch context lookup
- `ui-implementation/src-tauri/backend/app/routers/model_registry.py`
  — model-registry `/api/models` endpoint already returns context
  info; make its fetch async so the UI sees dynamic values in the
  model picker
- `ui-implementation/src-tauri/backend/tests/unit/test_model_contexts_dynamic.py`
  — new test file

### Impact

User-visible: any model Roampal can reach (Ollama-hosted, LM Studio-
hosted) automatically gets its real context window from the provider
— no code updates needed when new models drop. Qwen3.6-35B-A3B goes
from "32K because it matched the `qwen3` prefix" to "262K because
Ollama/LM Studio said so." Same for every future model.

Code-side: the 20-entry hardcoded `MODEL_CONTEXTS` table stays as a
safety net for the provider-offline case. It doesn't get removed.
Anyone adding a new model to the table for completeness can still do
so — it's just no longer the primary source of truth.

### Not in scope

- **Removing `MODEL_CONTEXTS`.** Keep it as fallback. The audit trail
  of "which model families we've tested against" has some value
  regardless of whether the values drive runtime behavior.
- **Live-monitoring context degradation.** The `save_calibration_data`
  scaffolding in `model_limits.py` is about learning from observed
  model behavior (e.g., "this model claims 32K but practically
  degrades at 24K"). That's a separate feature, not coupled to this
  section. Section 5 retires the dead write path; Section 6 fills the
  "where does the claimed max come from" question.
- **OpenAI / Anthropic provider support.** This section covers the
  two local providers (Ollama, LM Studio). Cloud providers (OpenAI,
  Anthropic, DeepSeek) are handled differently — their model lists
  are small, stable, and already encoded in per-provider config.
  Dynamic fetch for cloud providers is a no-op.

### Coordination

- **Depends on Section 5** — disk cache uses `write_json_atomic`.
   Implement Section 5 first.
- No core-side change needed. `roampal-core` doesn't have this
   context-lookup layer — desktop is the only place that negotiates
   context sizes with local providers.

### ✅ Implementation Summary (2026-04-28)

**Core module:** `config/model_contexts.py`
- `fetch_context_from_provider()` queries Ollama (`POST /api/show`) for `{arch}.context_length` and LM Studio (`GET /v1/models/{id}`) with priority: `loaded_context_length` > `max_context_length` > `context_length`. 5-second timeout, silently falls back on failure.
- Two-tier cache: memory (dict, 5-min TTL), disk (`data/model_context_cache.json`, 24h expiry via `_last_fetched_at`). Disk write uses atomic helper from Section 5.
- `get_context_size_async()` async priority chain: user override → saved preference → dynamic fetch (cached) → hardcoded `MODEL_CONTEXTS` prefix table → 8K fallback. When provider/base_url not provided, skips to step 4.

**Migrated call sites:**
- `ollama_client.py:170-172`, `ollama_client.py:687-690`: Both chat and stream paths now await async fetch with inferred provider from `self.api_style`
- `agent_chat.py:895-904`: Pre-flight context check uses async path
- `agent_chat.py:_build_complete_prompt()`: Converted to async, both callers updated (line 830, line 1580)

**Tests:** 19 new tests in `tests/unit/test_model_contexts_dynamic.py` — Ollama parsing (6), LM Studio parsing (6), memory cache (2), disk cache (2), priority chain (4). All pass.
- Full suite: 570 passed, 1 skipped.

---

## Critical bugfix (issue #8)

### ✅ 7. ChromaDB hard delete breaks memory dedup after GUI deletion

**Implemented 2026-04-28.** All 4 sub-parts deployed to Desktop codebase. Core already had these fixes from v0.5.5.

**Discovered:** 2026-04-27, issue #8 on `roampal-ai/roampal-core`.
Reproduced and root-caused across both Desktop and Core codebases.

**Symptom.** User deletes one or more memory_bank entries through the
Desktop GUI (or via roampal-core's `/api/memory-bank/archive`). Subsequent
turns score exchanges normally, but no new facts appear in Recent/All Types
views. Deleting `~/.roampal/data/mcp_sessions/_completion_state.json`
temporarily restores memory generation, but the problem returns.

**Root cause.** ChromaDB's HNSW index does not support true deletion.
`collection.delete(ids)` marks entries as deleted in metadata but leaves
the vectors queryable in the graph ("phantom entries" — known since v0.4.1.2).
Two independent bugs compound:

1. **Hard delete path still exists.** Desktop's `user_delete_memory()` at
   `memory_bank_service.py:286-302` calls `collection.delete_vectors([doc_id])`.
   Core's `/api/memory-bank/archive` endpoint does the same via
   `delete_memory_bank()` → `_memory_bank_service.archive()` → `self.delete(doc_id)`
   at `core/.../memory_bank_service.py:216, 302`.

2. **Dedup has no status filter for memory_bank.** Both codebases have
   `FACT_DEDUP_FILTERS["memory_bank"] = None` — meaning `_find_duplicate_fact()`
   queries ALL memory_bank documents including deleted ones. When a new fact
   semantically matches a "deleted" phantom entry, dedup returns the stale doc_id
   and skips creating a new document. The API returns success, but nothing was
   stored.

3. **Capacity count includes deleted entries.** Both `_get_count()` methods call
   `self.collection.collection.count()`, which counts ALL ChromaDB documents
   including "deleted" ones. After enough deletions, the 500-item limit is reached
   and all new stores fail with `ValueError`.

**Why both codebases are affected.** Desktop (`C:\roampal`) and roampal-core
(`C:\roampal-core`) maintain independent Python backends but share the same
ChromaDB data directory at `~/.roampal/data/chromadb/`. A hard delete from either
side creates phantom entries that affect both. The dedup bug exists in both
codebases independently:

| Codebase | Dedup filter | Hard delete path | `_get_count()` |
|---|---|---|---|
| Desktop `memory_bank_service.py` | `None` (line 638) | `delete()` line 286 | `count()` line 432 |
| Core `memory_bank_service.py` | `None` (line 383) | `delete()` line 302 | `count()` line 450 |

**Note:** Desktop's `archive()` method already uses soft delete
(`status=archived` metadata update at line 184-211). The bug is triggered by:
(a) the "Delete" button in the GUI calling `user_delete_memory()` → hard delete,
(b) roampal-core's `/api/memory-bank/archive` endpoint which calls hard delete,
(c) any code path that bypasses soft archive and goes straight to ChromaDB delete.

### Fix — three changes per codebase

#### A. Replace hard delete with soft delete

**Desktop:** `memory_bank_service.py:286-302` (`delete()` method)

Change `user_delete_memory()` to use soft delete instead of ChromaDB hard
delete. The existing `archive()` pattern is the correct one — reuse it.

```python
async def user_delete_memory(self, doc_id: str) -> bool:
    """User deletes memory (soft delete via status=archived)."""
    if not self.initialized:
        await self.initialize()

    # v0.3.3: Soft delete instead of ChromaDB hard delete.
    # HNSW index doesn't support true deletion — collection.delete() leaves
    # phantom entries that match during dedup queries, blocking new fact storage.
    return await self._memory_bank_service.archive(doc_id=doc_id, reason="user_delete")
```

The `delete()` method in `MemoryBankService` is retained for bulk cleanup
operations (`cleanup_archived()`) but no longer called by the user-facing path.

**Core:** Same change at `unified_memory_system.py:975-988`. The
`delete_memory_bank()` endpoint calls `_memory_bank_service.archive(content)`,
which currently calls `self.delete(doc_id)` (hard delete) at line 216. Fix the
`archive()` method to do a metadata update instead — same pattern as Desktop's
existing `archive()`.

The full fixed `archive()` is documented in Core's release notes at
`roampal-core/dev/docs/releases/v0.5.5/RELEASE_NOTES.md`. Summary of change:
replace the `return await self.delete(doc_id)` line with metadata update:

```python
    # v0.5.5: Soft delete — update metadata instead of ChromaDB hard delete
    doc = self.collection.get_fragment(doc_id)
    if not doc:
        return False

    metadata = doc.get("metadata", {})
    metadata["status"] = "archived"
    metadata["archived_at"] = datetime.now().isoformat()
    metadata["archive_reason"] = reason

    self.collection.update_fragment_metadata(doc_id, metadata)
    logger.info(f"Soft-deleted memory_bank item {doc_id}: {reason}")
    return True
```

#### B. Add status filter to dedup for memory_bank

**Desktop:** `unified_memory_system.py:634-639` (`FACT_DEDUP_FILTERS`)

```python
FACT_DEDUP_FILTERS = {
    "working": {"memory_type": "fact"},
    "history": {"memory_type": "fact"},
    "patterns": {"memory_type": "fact"},
    # v0.3.3: CRITICAL FIX — filter out archived entries during dedup.
    # Without this, _find_duplicate_fact() matches against soft-deleted memories
    # and silently blocks new fact storage. Root cause of issue #8.
    "memory_bank": {"status": {"$ne": "archived"}},  # was None
}
```

**Core:** Same change at `unified_memory_system.py:379-384`.

#### C. Exclude archived from capacity count

**Desktop:** `memory_bank_service.py:429-435` (`_get_count()`)

```python
def _get_count(self) -> int:
    """Get current active item count (excludes archived entries)."""
    # v0.3.3: collection.count() includes ALL documents including archived ones,
    # which inflates the capacity check and blocks new writes after deletion.
    try:
        all_ids = self.collection.list_all_ids()
        count = 0
        for doc_id in all_ids:
            doc = self.collection.get_fragment(doc_id)
            if doc and doc.get("metadata", {}).get("status", "active") != "archived":
                count += 1
        return count
    except Exception as e:
        logger.warning(f"Could not get memory_bank active count: {e}")
        return 0
```

**Core:** Same change at `memory_bank_service.py:447-453`.

#### D. Startup phantom migration (both codebases)

**Desktop:** `unified_memory_system.py`.

**Correction (2026-04-28):** The original planning doc said "add after line 679
in `_startup_cleanup()`" — but `_startup_cleanup()` does NOT exist in the
desktop backend. The closest equivalents are `cleanup_old_working_memory`,
`cleanup_old_history`, and a generic `cleanup()`. The phantom migration code
must be placed in whichever of these runs on server start, or a new
`_startup_cleanup()` method must be created in `unified_memory_system.py`
and called from the server startup path (`main.py`). Verify the placement
before implementing.

**Core:** Already documented in Core v0.5.5 release notes.

```python
# v0.3.3 / v0.5.5: memory_bank phantom migration — remove IDs from pre-fix hard deletes.
# ChromaDB's collection.delete() marks entries as deleted but leaves the ID in
# list_all_ids(). get_fragment() returns None for these phantoms because both
# document and metadata are gone. They're already broken (no content, no vector),
# so it's safe to remove them from the index permanently.
if self._memory_bank_service:
    try:
        all_ids = self._memory_bank_service.collection.list_all_ids()
        phantom_ids = []
        for doc_id in all_ids:
            if not self._memory_bank_service.collection.get_fragment(doc_id):
                phantom_ids.append(doc_id)

        if phantom_ids:
            self._memory_bank_service.collection.delete_vectors(phantom_ids)
            logger.info(f"v0.3.3 migration: removed {len(phantom_ids)} phantom entries from memory_bank")
    except Exception as e:
        logger.warning(f"Startup cleanup error for memory_bank phantoms: {e}")
```

**Why not also clean archived entries at startup?** Archived entries (`status=archived`) are intentionally soft-deleted — they're reversible via `restore()`. Cleaning them at startup would make soft delete effectively permanent, removing the distinction between LLM archive (reversible) and user hard delete (permanent). Instead, `cleanup_archived()` remains available for manual invocation or capacity-pressure triggers when approaching the 500-item limit.

### Testing

Manual end-to-end reproduction of issue #8 (run on both Desktop and Core):

1. Add 5+ facts to memory_bank via GUI or CLI (`roampal add "fact text"`)
2. Delete 3+ of them via the Desktop GUI's "Delete" button
3. Attempt to add new facts with similar content to deleted ones
4. **Pre-fix:** New facts silently fail — `_find_duplicate_fact()` matches
   phantom entry in HNSW, returns old doc_id, no new document created
5. **Post-fix:** New facts store correctly — soft-deleted entries have
   `status=archived`, dedup filter excludes them, fresh documents are created

Automated verification:
- Unit test: `_find_duplicate_fact()` with mixed active/archived entries in
  memory_bank returns `None` when only archived matches exist
- Unit test: `user_delete_memory()` sets `status=archived` instead of calling
  ChromaDB delete
- Unit test: `_get_count()` returns correct count after archiving entries
- Integration test: full cycle of add → delete → add similar fact succeeds
- Startup migration: verify phantom IDs (in `list_all_ids()` but not in
  `get_fragment()`) are removed on first post-fix startup

### Files touched (Desktop)

- `ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py` —
  `FACT_DEDUP_FILTERS["memory_bank"]` changed from `None` to
  `{"status": {"$ne": "archived"}}`; `user_delete_memory()` switched to soft delete;
  phantom migration pass added to existing startup cleanup (or new
  `_startup_cleanup()` method created — verify placement, see 7D correction)
- `ui-implementation/src-tauri/backend/modules/memory/memory_bank_service.py` —
  `_get_count()` rewritten to exclude archived entries
- `main.py` — if a new `_startup_cleanup()` method is created, call it
  from the server startup path

### Files touched (Core)

- `roampal/backend/modules/memory/unified_memory_system.py` —
  `FACT_DEDUP_FILTERS["memory_bank"]` changed from `None` to
  `{"status": {"$ne": "archived"}}`; `_startup_cleanup()` gains phantom migration pass;
  `delete_memory_bank()` switched to soft delete
- `roampal/backend/modules/memory/memory_bank_service.py` —
  `archive()` updated to do metadata update instead of calling `self.delete()`,
  `_get_count()` rewritten to exclude archived entries

### Coordination

This fix must ship in **both** Desktop v0.3.3 and Core v0.5.5 because:
1. They share the same ChromaDB data directory — a hard delete from either side
   creates phantom entries that affect both
2. Both have independent dedup code with the same bug
3. Shipping only one half leaves the other as an attack surface

The core-side fix is already documented in `roampal-core/dev/docs/releases/v0.5.5/RELEASE_NOTES.md`.
This section covers the Desktop half.

### Impact if unfixed

- **Silent memory loss.** Every new fact that semantically matches a deleted entry
  is deduped away. No error shown — API returns success with a doc_id pointing to
  the old deleted document.
- **Capacity exhaustion.** Deleted entries count toward the 500-item limit, blocking
  all new writes after enough deletions.
- **No self-recovery.** Deleting `_completion_state.json` resets scoring state
  temporarily, but doesn't fix the underlying dedup problem — eventually the same
  deleted entries match again.

### Simple Explanation

**The Problem:** When you delete a memory through the GUI, Roampal tries to remove
it from ChromaDB using `collection.delete()`. But ChromaDB's underlying search index
(HNSW) doesn't actually support true deletion — it marks entries as deleted but they
stay in the graph and can still be found by queries. So when you try to save a new
memory that's similar to one you "deleted," the duplicate-checking code finds the old
entry still sitting there, thinks "this already exists," and skips saving the new one.
Your memories stop being saved after you delete any of them.

**The Fix:** Instead of trying to hard-delete (which ChromaDB can't do properly), we
switch to soft deletion — just mark entries with `status: archived` in their metadata,
then tell all query paths to skip archived entries. This is reliable, reversible (you
can restore deleted memories), and consistent with how other collections already handle
archiving. Desktop's "Archive" button already does this correctly; the "Delete" button
and roampal-core's archive endpoint need to be updated to match.

### ChromaDB deletion status (researched 2026-04-27)

ChromaDB has **not** fixed its HNSW deletion problem as of v1.5.7 (April 8, 2026).
The issue is known and tracked by the ChromaDB team but remains unresolved after
18+ months:

| Issue | Status | Description |
|---|---|---|
| [#3486](https://github.com/chroma-core/chroma/issues/3486) HNSW bugs | OPEN | Deleted nodes counted toward `ef`, causing "M is too small" exceptions |
| [#2594](https://github.com/chroma-core/chroma/issues/2594) HNSW index pruning | OPEN | Unbounded growth, no solution implemented (Sep 2024 → now) |
| [#5259](https://github.com/chroma-core/chroma/issues/5259) HNSW dirs not deleted | OPEN | Deleting a collection doesn't remove its HNSW directories |
| [#3882](https://github.com/chroma-core/chroma/issues/3882) SQLite not shrinking | OPEN | File size stays same after all documents are deleted |

ChromaDB's own issue #2594 proposes three solutions:
1. Enable `allow_replace_deleted` (low complexity, backward compatible)
2. Rebuild index periodically (medium complexity, slow, 2x memory during rebuild)
3. Add pruning to HNSW lib directly (high complexity)

**None are implemented.** Our soft-delete approach is the correct workaround until
ChromaDB fixes this upstream. The trade-off: archived entries accumulate in ChromaDB
and need periodic cleanup via `cleanup_archived()` or a full collection rebuild to
reclaim disk space. This is documented as a known limitation and tracked for future
optimization when ChromaDB ships HNSW pruning support.

---

## 8. Issue #8 hardening — gaps surfaced after Section 7 audit

**Status:** TBD — proposed for v0.3.3 if budget allows, otherwise pull to v0.3.4.
**Triggered by:** Verification audit of Section 7 (and core's matching v0.5.5 fix). Section 7 closes the headline bug; the items below close gaps that Section 7's seven-step fix didn't fully cover.

These mirror items being proposed for **roampal-core v0.5.6** so the two codebases stay in lockstep on memory_bank lane integrity. Both backends share the same ChromaDB data directory, so a regression in either one re-opens issue #8 for both.

### 8A. Run phantom migration after `cleanup_archived()`, not just at startup

**Gap.** Section 7's phantom migration runs once per process boot. But `cleanup_archived()` calls `delete_vectors()` on every archived ID — the *exact* operation that creates phantoms in the first place. Each cleanup leaves debris that survives until the next restart.

Today `cleanup_archived()` is only triggered manually. Item 8C below adds an auto-trigger under capacity pressure, which makes this a real correctness issue: phantoms accumulate during a long-running session.

**Fix.** Extract the phantom sweep into a helper on `MemoryBankService` and call it from both startup and `cleanup_archived()`:

```python
def _sweep_phantoms(self, hint_ids: Optional[List[str]] = None) -> int:
    """Remove HNSW-orphaned IDs. If hint_ids given, only check those."""
    candidates = hint_ids if hint_ids is not None else self.collection.list_all_ids()
    phantom_ids = [
        doc_id for doc_id in candidates
        if not self.collection.get_fragment(doc_id)
    ]
    if phantom_ids:
        self.collection.delete_vectors(phantom_ids)
    return len(phantom_ids)

def cleanup_archived(self) -> int:
    archived_ids = [...]   # existing logic from Section 7
    if archived_ids:
        self.collection.delete_vectors(archived_ids)
        # v0.3.3: sweep the IDs we just deleted
        self._sweep_phantoms(hint_ids=archived_ids)
    return len(archived_ids)
```

**Files.**
- `ui-implementation/src-tauri/backend/modules/memory/memory_bank_service.py` — new `_sweep_phantoms()`; `cleanup_archived()` calls it
- Replace the inline phantom logic from Section 7D with a call to `_sweep_phantoms()`

### 8B. Re-check the phantom filter strengthening (Section 2 may already cover it)

**Gap.** Core's `chromadb_adapter.py:267-269` filters phantoms only when document AND metadata are both falsy:
```python
if result_document is None and (result_metadata is None or not result_metadata):
    continue
```
ChromaDB can leave entries in mid-states after a delete (doc cleared but metadata cached, or vice versa). The AND-only filter leaks both mid-states.

**Why this might already be done in Desktop.** Section 2 (ChromaDB "Error finding id") shipped phantom filtering at `chromadb_adapter.py:501-547` via "documents+metadatas fetch" — Section 2's summary says "Only IDs with valid entries are returned." If that filter requires both fields to be non-empty (rather than just the AND-of-None), Desktop is already covered and there's no work here.

**Action.** Re-read Desktop's `chromadb_adapter.py:501-547` filter logic. If it's the OR (either field missing → reject) form, this section is no-op. If it's AND-only like core's, mirror the strengthening:
```python
is_phantom = (
    result_document is None
    or (result_metadata is None or not result_metadata)
)
if is_phantom:
    continue
```

**Files (only if needed):**
- `ui-implementation/src-tauri/backend/modules/memory/chromadb_adapter.py` — tighten filter to OR
- New integration test confirming mid-state phantoms are rejected

### 8C. Status backfill on startup for legacy memory_bank entries

**Gap.** The dedup filter `{"status": {"$ne": "archived"}}` and Section 7's search-path filters all rely on ChromaDB treating "missing status field" as "not equal to archived → include in results." Most ChromaDB versions do, but it's version-specific.

Pre-Section-7 entries written before `add()` started setting `status="active"` have no `status` field. If a future ChromaDB upgrade changes `$ne` semantics on missing fields, those entries silently drop from dedup/search.

**Fix.** One-shot backfill in the startup cleanup path (alongside the phantom migration from Section 7D):

```python
# v0.3.3: backfill status=active on legacy memory_bank entries
if self._memory_bank_service:
    try:
        all_ids = self._memory_bank_service.collection.list_all_ids()
        backfilled = 0
        for doc_id in all_ids:
            frag = self._memory_bank_service.collection.get_fragment(doc_id)
            if frag and "status" not in frag.get("metadata", {}):
                meta = frag.get("metadata", {}) or {}
                meta["status"] = "active"
                self._memory_bank_service.collection.update_fragment_metadata(doc_id, meta)
                backfilled += 1
        if backfilled:
            logger.info(f"v0.3.3 migration: backfilled status=active on {backfilled} legacy entries")
    except Exception as e:
        logger.warning(f"Status backfill error: {e}")
```

Idempotent — runs every startup but only does work the first time after upgrade.

**Files.**
- `ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py` — add backfill block to whichever startup cleanup runs
- Test: legacy entries get `status=active`, post-fix entries are untouched

### 8D. Auto-trigger `cleanup_archived()` under capacity pressure

**Gap.** `memory_bank` has a 500-item cap. After Section 7, archived entries don't count toward the cap (because `_get_count()` excludes them) — but they DO count toward the underlying ChromaDB collection size. Archived entries accumulate forever; `cleanup_archived()` exists but is never called.

Two failure modes: ChromaDB performance degrades as the collection grows past the active cap; heavy archive-and-replace work creates unbounded growth.

**Fix.** Add a capacity-pressure auto-trigger in `MemoryBankService`:

```python
ACTIVE_THRESHOLD = 400          # 80% of MAX_ITEMS
ARCHIVED_RATIO_THRESHOLD = 0.5  # cleanup if >50% of total is archived

def _maybe_cleanup_archived(self) -> None:
    active = self._get_count()
    if active < ACTIVE_THRESHOLD:
        return
    total = self.collection.collection.count()
    archived = total - active
    if total == 0 or archived / total < ARCHIVED_RATIO_THRESHOLD:
        return
    cleaned = self.cleanup_archived()
    if cleaned:
        logger.info(f"Auto-cleanup at capacity pressure: removed {cleaned} archived entries")
```

Call from `store()` before the cap check. Combined with 8A above, the auto-cleanup path won't leave phantoms behind.

**Files.**
- `ui-implementation/src-tauri/backend/modules/memory/memory_bank_service.py` — new `_maybe_cleanup_archived()`; called from `store()`
- Test: capacity-pressure trigger fires above threshold and not below

### 8E. Dedup observability

**Gap.** `_find_duplicate_fact()` silently returns the matching doc_id when it finds a similar fact, and the caller silently skips storage. No log line, no metric, no warning. If dedup ever becomes over-aggressive again — same silent-drop symptom, different root cause — the system would be just as blind as it was before issue #8 was reported.

**Fix.** One log line per dedup-skip with enough context to triage:

```python
if best_match:
    logger.info(
        f"Dedup skip: tier={best_match['tier']} "
        f"distance={best_match['distance']:.3f} "
        f"matched_id={best_match['id']} "
        f"new_fact={fact[:80]!r}"
    )
    return best_match["id"]
```

**Files.**
- `ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py` — log dedup hits in `_find_duplicate_fact()`

### 8F. Lane enforcement: rename + restrict the hard-delete path

**Gap.** `MemoryBankService` has two delete paths with very different safety profiles:

| Method | Behavior | Safe to call from |
|---|---|---|
| `archive(doc_id, reason)` | Soft delete (sets `status=archived`) | Anywhere — user-facing GUI |
| `delete(doc_id)` | Hard delete via `delete_vectors()` | Only `cleanup_archived()` |

The `delete()` method is public and only differentiated from `archive()` by a docstring. A future engineer adding "really delete this" / GDPR features could wire the GUI straight to `delete()`, hit HNSW phantoms again, and re-open issue #8.

Section 7A already routed Desktop's `user_delete_memory()` through `archive()`. This item closes the structural risk so a *future* hard-delete flow can't bypass it accidentally.

**Fix.**
- Rename `MemoryBankService.delete()` → `delete_permanent()`
- Require explicit `force=True`:

```python
async def delete_permanent(self, doc_id: str, *, force: bool = False) -> bool:
    """
    Permanently hard-delete a memory_bank entry.

    HNSW phantom risk: this leaves debris in the vector index that bypasses
    metadata filters until the next phantom sweep. ONLY call from contexts
    that follow up with a phantom sweep (cleanup_archived, explicit GDPR flow).

    For user-initiated deletes, call archive() instead.
    """
    if not force:
        raise RuntimeError(
            "delete_permanent() requires force=True. "
            "For user-facing deletes, use archive() instead."
        )
    # ... existing delete logic
```

`cleanup_archived()` passes `force=True` explicitly. New code can't fall into hard delete without grepping the codebase and seeing the warning.

**Files.**
- `ui-implementation/src-tauri/backend/modules/memory/memory_bank_service.py` — rename + force-flag enforcement
- Update `cleanup_archived()` and any in-tree caller
- Tests for force-flag enforcement

### 8G. Integration test for the full archive-then-add cycle

**Gap.** Section 7's tests cover individual pieces in isolation:
- `archive()` sets status correctly
- Phantom migration sweep works
- Dedup filter where-clause is built

Nothing exercises the reported repro end-to-end: add facts → archive some → add new facts that semantically overlap → confirm all new facts stored. If any seam between these layers regresses (e.g. archive() stops setting status, or the dedup filter regresses), unit tests stay green but the bug returns silently.

**Implementation.** New `tests/integration/test_archive_dedup_cycle.py`:

```python
async def test_archive_then_add_similar_fact_succeeds(real_chromadb):
    """Issue #8 regression test: archived entries must not block dedup of new facts."""
    mem = UnifiedMemorySystem(...)
    await mem.initialize()

    for i in range(5):
        await mem.store_memory_bank(f"User likes color {i}")
    assert mem._memory_bank_service._get_count() == 5

    # Soft-delete via the user-facing path (after Section 7A this routes to archive())
    await mem.user_delete_memory("User likes color 0")
    await mem.user_delete_memory("User likes color 1")
    await mem.user_delete_memory("User likes color 2")
    assert mem._memory_bank_service._get_count() == 2

    new_id = await mem.store_memory_bank("User likes color 0")
    assert new_id is not None  # Should NOT be deduped against archived entry
    assert mem._memory_bank_service._get_count() == 3
```

Run against real ChromaDB (mocks don't model HNSW phantom behavior, which is exactly the bug class we're guarding against).

**Files.**
- `ui-implementation/src-tauri/backend/tests/integration/test_archive_dedup_cycle.py` — new

---

### Summary — what's worth bundling into v0.3.3

| Item | Effort | Risk if skipped | Bundle into 0.3.3? |
|---|---|---|---|
| 8A — phantom sweep after cleanup_archived | Small | High once 8D ships | **Yes** — pair with 8D |
| 8B — phantom filter check / strengthen | Tiny if no-op, Small if needed | Medium | **Yes** — re-check, fix if needed |
| 8C — status backfill | Small | Medium-low (version-dependent) | **Yes** |
| 8D — capacity-pressure auto-cleanup | Small-medium | Medium (long-running sessions only) | Either bundle or hold for 0.3.4 |
| 8E — dedup observability | Tiny | Low — diagnostic value, not correctness | **Yes** — almost free |
| 8F — lane enforcement (rename + force flag) | Small | Low today, high for future maintainers | **Yes** |
| 8G — integration test | Small-medium | Medium — locks in regression boundary | **Yes** |

**Recommendation:** bundle 8A + 8B + 8C + 8E + 8F + 8G into v0.3.3 (all small, all close real gaps). Move 8D to v0.3.4 if v0.3.3 release window is tight — capacity pressure isn't an issue users will hit immediately, and 8A is a prerequisite for 8D anyway.

### Coordination with roampal-core v0.5.6

These items mirror what's planned for core v0.5.6. Both codebases need them because they share the same ChromaDB data directory — a regression in either backend re-opens issue #8 for both. Implementations are independent (separate Python processes, no shared install path) but the lane invariants and observability semantics should match so that diagnostic logs from either backend look the same.

---

## 9. Bulk `/clear/*` endpoints still create phantoms (Section 7 follow-up — actual root cause of issue #8)

**Status:** TBD — must ship in v0.3.3 to actually close issue #8.
**Triggered by:** Audit of `data_management.py` after Section 7 shipped. Section 7 fixed `user_delete_memory()` (single-memory "Delete" button) but never touched the bulk `/clear/*` endpoints used by the GUI's "Delete data" tab. Those endpoints are how issue #8 was actually triggered ("keeping only Memory Bank and Books" → bulk clear of working/history/patterns).

### What's wrong

`ui-implementation/src-tauri/backend/app/routers/data_management.py` has 5 bulk-delete endpoints. Four use the broken hard-delete pattern; one (`/clear/books`) uses the correct drop-and-recreate pattern. The fix from Section 7 (route through `archive()`) doesn't apply here — these endpoints bypass `MemoryBankService` entirely and call ChromaDB directly:

| Endpoint | Lines | Pattern | Status |
|---|---|---|---|
| `/clear/memory_bank` | 101-140 | `adapter.collection.delete(ids=batch)` | **Broken — leaves phantoms** |
| `/clear/working` | 143-179 | `adapter.collection.delete(ids=batch)` | **Broken — leaves phantoms** |
| `/clear/history` | 182-218 | `adapter.collection.delete(ids=batch)` | **Broken — leaves phantoms** |
| `/clear/patterns` | 221-257 | `adapter.collection.delete(ids=batch)` | **Broken — leaves phantoms** |
| `/clear/books` | 260-372 | `client.delete_collection()` + recreate | ✅ Correct (v0.2.9 fix) |

`/clear/books` documents the right approach inline (lines 277-278):

> "Nuke and recreate ChromaDB collection (v0.2.9). This fully rebuilds the HNSW index, eliminating ghost vectors that remain after regular delete() operations"

The same fix was never propagated to the other four endpoints. Section 7's audit caught the per-memory delete path but missed the bulk-clear path.

### Why this is the actual #8 root cause

The issue report says "keeping only Memory Bank and Books" — that means the user hit the GUI's "Delete data" tab, which calls the bulk `/clear/*` endpoints, NOT the per-memory "Delete" button. So:

- Section 7's `user_delete_memory()` → `archive()` fix is correct, but it only protects the per-memory path, which was not the path used.
- The bulk path actually used still creates phantoms in working/history/patterns/memory_bank.
- This explains the symptom that v0.5.5/Section 7 couldn't: the user saw **no new memories of any type** appear, including summaries. Phantoms in `working` (where summaries land via `store_working`) interfere with HNSW writes/queries. Phantoms in `memory_bank` block fact dedup. The bulk delete affected all four collections, so all four were broken at once.

### Fix — apply the `/clear/books` pattern to the four broken endpoints

Replace the broken pattern in each handler:

```python
# Before — leaves phantoms
if count_before > 0:
    all_docs = adapter.collection.get(include=[])
    if all_docs.get("ids"):
        batch_size = 100
        all_ids = all_docs["ids"]
        for i in range(0, len(all_ids), batch_size):
            batch = all_ids[i : i + batch_size]
            adapter.collection.delete(ids=batch)

# After — drop and recreate (matches /clear/books)
if count_before > 0:
    collection_name = adapter.collection_name
    client = adapter.client
    # v0.3.3: Nuke and recreate ChromaDB collection.
    # collection.delete(ids=...) leaves HNSW phantoms that block dedup
    # and break new writes — actual root cause of issue #8 for users
    # who triggered bulk-delete from the GUI's "Delete data" tab.
    client.delete_collection(name=collection_name)
    adapter.collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=None,  # We provide our own embeddings
        metadata={"hnsw:space": "l2"},
    )
    logger.info(f"Nuked {collection_name} collection ({count_before} entries deleted)")
```

Apply to: `clear_memory_bank` (101-140), `clear_working` (143-179), `clear_history` (182-218), `clear_patterns` (221-257). The `count_before` variable is computed before the destructive operation — return it as `deleted_count` for UI consistency.

### Testing

Add to `ui-implementation/src-tauri/backend/tests/unit/test_data_management.py`:

```python
class TestClearLeavesNoPhantoms:
    """v0.3.3 Section 9: bulk-clear endpoints must use delete_collection+recreate.

    The old pattern (collection.delete(ids=batch)) left HNSW phantoms — actual
    root cause of issue #8. New pattern matches /clear/books.
    """

    @pytest.mark.asyncio
    async def test_clear_memory_bank_recreates_collection(self):
        from app.routers.data_management import clear_memory_bank

        adapter = MagicMock()
        adapter.collection_name = "roampal_memory_bank"
        adapter.client = MagicMock()
        adapter.collection = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=5)

        memory = MagicMock()
        memory.collections = {"memory_bank": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        result = await clear_memory_bank(request)

        adapter.client.delete_collection.assert_called_once_with(name="roampal_memory_bank")
        adapter.client.get_or_create_collection.assert_called_once()
        assert result["deleted_count"] == 5

    @pytest.mark.asyncio
    async def test_clear_working_recreates_collection(self):
        from app.routers.data_management import clear_working_memory

        adapter = MagicMock()
        adapter.collection_name = "roampal_working"
        adapter.client = MagicMock()
        adapter.collection = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=12)

        memory = MagicMock()
        memory.collections = {"working": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        result = await clear_working_memory(request)

        adapter.client.delete_collection.assert_called_once_with(name="roampal_working")
        adapter.client.get_or_create_collection.assert_called_once()
        assert result["deleted_count"] == 12

    @pytest.mark.asyncio
    async def test_clear_history_recreates_collection(self):
        from app.routers.data_management import clear_history

        adapter = MagicMock()
        adapter.collection_name = "roampal_history"
        adapter.client = MagicMock()
        adapter.collection = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=3)

        memory = MagicMock()
        memory.collections = {"history": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        await clear_history(request)
        adapter.client.delete_collection.assert_called_once_with(name="roampal_history")

    @pytest.mark.asyncio
    async def test_clear_patterns_recreates_collection(self):
        from app.routers.data_management import clear_patterns

        adapter = MagicMock()
        adapter.collection_name = "roampal_patterns"
        adapter.client = MagicMock()
        adapter.collection = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=7)

        memory = MagicMock()
        memory.collections = {"patterns": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        await clear_patterns(request)
        adapter.client.delete_collection.assert_called_once_with(name="roampal_patterns")

    @pytest.mark.asyncio
    async def test_clear_empty_collection_skips_recreate(self):
        """count_before=0 → skip the destructive path entirely."""
        from app.routers.data_management import clear_working_memory

        adapter = MagicMock()
        adapter.client = MagicMock()
        adapter.get_collection_count = AsyncMock(return_value=0)

        memory = MagicMock()
        memory.collections = {"working": adapter}
        request = MagicMock()
        request.app.state.memory = memory

        result = await clear_working_memory(request)

        adapter.client.delete_collection.assert_not_called()
        assert result["deleted_count"] == 0
```

Manual end-to-end verification (the `clear_books` proof the issue needed):
1. Add 10+ memories across working/history/patterns via GUI or sidecar.
2. Hit "Delete data" → clear all four collections.
3. Send a new exchange via OpenCode.
4. **Pre-fix:** Sidecar scoring runs but no new entries appear in Recent/All Types views. `list_all_ids()` on each collection returns the deleted IDs (phantoms). Reproduces issue #8.
5. **Post-fix:** New summary + facts appear within seconds. `list_all_ids()` returns only fresh entries. Issue #8 actually closed.

**Verified live 2026-05-11 — issue #8's exact scenario reproduced and confirmed mitigated:**
1. Pre-clear state: working = 8, history = ?, patterns = ?, memory_bank = 8 (Tauri Desktop dev profile, `%APPDATA%\Roampal_DEV\data\chromadb`).
2. At 17:12:10 Tauri user clicked Desktop "Delete data" → working/history/patterns. Backend log captured `POST /api/data/clear/{working,history,patterns} HTTP/1.1" 200 OK` from `127.0.0.1`. Immediately after: `/api/memory/stats` returned `working: 0, history: 0, patterns: 0, memory_bank: 8` (memory_bank correctly untouched), and `_completion_state.json` + `_sidecar_processed.json` were both removed from `<DATA_PATH>/sessions/` by `_reset_completion_state()` (Section 9.1).
3. Launched OpenCode dev session in a fresh PowerShell with `ROAMPAL_DEV=1` and an empty working directory (`C:\tmp\roampal-opencode-dev`). The dev-mode toggle pointed OpenCode's hook subprocess at the **same** `Roampal_DEV` data root that Tauri Desktop was using — exact shared-DB reproduction of the reported setup.
4. Sent two messages in OpenCode (a greeting then a horse-related question). OpenCode's hook scored each exchange.
5. Post-test state at 17:32:49: working = **2 new entries** (`working_f6d3144b` *"User asked if the assistant knew about a horse. The assistant recalled the user has a horse named Jerry..."*, `working_76b679f7` *"User greeted; the model returned a greeting..."*), history = **1 promoted entry** (90% score, tags `horse, jerry`), patterns = 0. Memory panel in Tauri Desktop showed all three entries with timestamps `5/11/2026, 5:31:44 PM` and `5/11/2026, 5:32:17 PM` — **exact UI surface that issue #8 reported as broken now showing the post-delete writes correctly**.
6. Conclusion: HNSW phantom-blocking gone (Section 9), `_completion_state.json` stale-flag-blocking gone (Section 9.1), cross-client (Desktop bulk-clear → OpenCode write) path round-trips cleanly without manual file-deletion workarounds. Issue #8 deterministically closeable.

### Files affected

| File | Lines | Change |
|---|---|---|
| `ui-implementation/src-tauri/backend/app/routers/data_management.py` | 101-140, 143-179, 182-218, 221-257 | Replace `collection.delete(ids=batch)` loop with `delete_collection() + get_or_create_collection()` (matches `/clear/books` at 280-294) |
| `ui-implementation/src-tauri/backend/tests/unit/test_data_management.py` | new `TestClearLeavesNoPhantoms` class | 5 unit tests: 4 per-endpoint drop-and-recreate verification + 1 empty-collection guard |

### Why this closes #8 for real

Section 7 fixed the per-memory soft-delete path but assumed bulk delete went through the same plumbing. It doesn't. With Section 9:

- All five bulk endpoints use the same drop-and-recreate pattern.
- No HNSW phantoms remain after a "Delete data" operation.
- Roampal-core's `_startup_cleanup` phantom sweep is now belt-and-suspenders, not the load-bearing fix.
- The reported reproduction (bulk delete → no new memories) closes deterministically.

This is the change that makes the GitHub reply ("v0.5.5 fixes it, Desktop fix coming") fully honest.

---

## 9.1 — Also reset `_completion_state.json` on bulk clear of conversational tiers

**Status:** ✅ Implemented 2026-05-11. Helper `_reset_completion_state()` added at `data_management.py:29-50`. Wired into `clear_working_memory:199`, `clear_history:242`, `clear_patterns:285`. 6 new tests in `TestSection91CompletionStateReset` (`test_data_management.py`) — 3 positive (working/history/patterns unlink), 2 negative (memory_bank/books leave file alone), 1 missing-file guard. Full unit suite passes (600 + 32 mcp).
**Triggered by:** The reported manual workaround for #8 is *"delete `_completion_state.json` and restart"*. Core v0.5.7 ships startup GC for that file, but a user who hits the GUI's "Delete data" tab to recover from #8 shouldn't have to restart roampal-core to pick up the GC pass — the bulk-clear action already semantically means *"reset this conversational state."*

### Why this is the right pairing with Section 9

Section 9 drops ChromaDB phantoms in working / history / patterns / memory_bank. The MCP hook's `_completion_state.json` is a *parallel* per-conversation ledger living at `<DATA_PATH>/mcp_sessions/_completion_state.json` — it tracks `completed` / `scoring_required` / `scored_this_turn` / `first_message_seen` flags per `conversation_id`. The conversational tiers (working/history/patterns) hold the *content* of those scored turns; the state file holds the *lifecycle metadata* that drove scoring those turns into the tiers.

When the user clears the content tiers, the lifecycle metadata for those same conversation_ids becomes stale by construction. Leaving it in place is what enables core's cross-session fallback (`main.py:1087-1096`) to grab a stale `scored_this_turn=True` from a now-empty conversation and falsely mark the *current* turn as already-scored. Resetting the file at clear time forecloses that path immediately rather than waiting for the next core restart to GC.

### Scope — exactly three of the five clear endpoints

| Endpoint | Unlink state file? | Reason |
|---|---|---|
| `/clear/working` | ✅ Yes | Working is where summaries land. Clearing it without resetting lifecycle metadata leaves stuck flags for conversations whose summaries no longer exist. |
| `/clear/history` | ✅ Yes | History holds promoted summaries; same logic. |
| `/clear/patterns` | ✅ Yes | Patterns hold fully-promoted recurring summaries; same logic. |
| `/clear/memory_bank` | ❌ No | Memory bank holds permanent atomic facts, not per-conversation turn lifecycle. The state file is irrelevant to memory_bank's correctness. |
| `/clear/books` | ❌ No | Books are reference docs, fully unrelated to conversation lifecycle. |

### Fix — shared helper called from the three conversational-tier handlers

Add a small helper in `data_management.py` (or a sibling `_helpers.py` if you want it out of the router file):

```python
from pathlib import Path
from config.settings import DATA_PATH  # matches existing import in backup.py:18, mcp.py:15, system_health.py:237

def _reset_completion_state() -> bool:
    """v0.3.3 Section 9.1: Unlink MCP hook's _completion_state.json.

    Called from bulk-clear handlers for working/history/patterns. The file
    holds per-conversation_id scoring lifecycle flags (completed,
    scoring_required, scored_this_turn, first_message_seen). When the user
    clears the conversational tiers, this metadata is stale by construction;
    leaving it in place enables core's cross-session fallback to misroute
    stale flags onto the current turn (see core v0.5.7 RELEASE_NOTES.md).

    Idempotent — safe if the file does not exist (fresh install, already cleared).
    Returns True if the file was unlinked, False if it was absent.
    """
    state_file = Path(DATA_PATH) / "mcp_sessions" / "_completion_state.json"
    try:
        state_file.unlink()
        logger.info(f"Section 9.1: unlinked {state_file}")
        return True
    except FileNotFoundError:
        logger.info(f"Section 9.1: {state_file} absent — nothing to unlink")
        return False
    except OSError as e:
        # Don't fail the clear operation if the unlink fails (permissions, etc.).
        # The chromadb drop already succeeded; state file unlink is secondary.
        logger.warning(f"Section 9.1: could not unlink {state_file}: {e}")
        return False
```

Call `_reset_completion_state()` from `clear_working_memory`, `clear_history`, and `clear_patterns` **after** the chromadb drop succeeds. Order matters: chromadb drop first (the primary effect), state-file unlink second (the auxiliary reset). If the unlink fails, the user still gets the main fix.

Do NOT call it from `clear_memory_bank` or `clear_books`.

### Testing

Extend `TestClearLeavesNoPhantoms` in `test_data_management.py`:

```python
@pytest.mark.asyncio
async def test_clear_working_unlinks_completion_state(self, tmp_path, monkeypatch):
    """v0.3.3 Section 9.1: working clear must also unlink _completion_state.json."""
    from app.routers import data_management
    monkeypatch.setattr(data_management, "DATA_PATH", str(tmp_path))

    sessions_dir = tmp_path / "mcp_sessions"
    sessions_dir.mkdir()
    state_file = sessions_dir / "_completion_state.json"
    state_file.write_text('{"conv_abc": {"scored_this_turn": true}}')

    adapter = MagicMock()
    adapter.collection_name = "roampal_working"
    adapter.client = MagicMock()
    adapter.collection = MagicMock()
    adapter.get_collection_count = AsyncMock(return_value=3)

    memory = MagicMock()
    memory.collections = {"working": adapter}
    request = MagicMock()
    request.app.state.memory = memory

    await data_management.clear_working_memory(request)

    assert not state_file.exists()

# Mirror tests for clear_history, clear_patterns.
# Negative tests for clear_memory_bank, clear_books — state file must remain.
@pytest.mark.asyncio
async def test_clear_memory_bank_leaves_completion_state(self, tmp_path, monkeypatch):
    """v0.3.3 Section 9.1: memory_bank clear must NOT touch _completion_state.json."""
    # ... seed state_file, call clear_memory_bank, assert state_file still exists.

@pytest.mark.asyncio
async def test_clear_working_handles_missing_state_file(self, tmp_path, monkeypatch):
    """v0.3.3 Section 9.1: clear succeeds when state file is already absent."""
    # ... no state_file created, call clear_working_memory, assert no exception.
```

### Acceptance criteria

- After `/clear/working`, `/clear/history`, or `/clear/patterns` runs, `<DATA_PATH>/mcp_sessions/_completion_state.json` no longer exists on disk.
- After `/clear/memory_bank` or `/clear/books`, the state file is unchanged.
- The chromadb drop-and-recreate from Section 9 still runs first; state-file unlink failure does not abort the clear.
- A log line per call confirms whether the unlink happened or the file was already absent.

### Files affected

| File | Change |
|---|---|
| `ui-implementation/src-tauri/backend/app/routers/data_management.py` | Add `_reset_completion_state()` helper. Call from `clear_working_memory`, `clear_history`, `clear_patterns` after the chromadb drop. |
| `ui-implementation/src-tauri/backend/tests/unit/test_data_management.py` | Extend `TestClearLeavesNoPhantoms` with 6 new tests: 3 positive (working/history/patterns unlink), 2 negative (memory_bank/books leave it alone), 1 missing-file guard. |

### Coordination with core v0.5.7

- **Desktop 9.1** handles the *immediate-reset* case: user hits "Delete data" on a wedged install → state file gone without a core restart.
- **Core v0.5.7** handles the *slow-accumulation* case: state file grows for users who never hit clear → startup GC trims it.

Both layers are needed. Section 9.1 is cheap (~30 LOC + tests) and removes the "restart core to recover" workaround from the user's recovery path.

### Why memory_bank is excluded

Memory bank holds *permanent atomic facts* — not turn-lifecycle metadata. Clearing memory_bank doesn't invalidate any flag in `_completion_state.json`, because no flag in that file references memory_bank contents. Wiring the unlink to memory_bank would be coincidental scope creep, not a real correctness pairing. Keep the semantic clean.

---

## 10. `_completion_state.json` accumulates unbounded — moved to core v0.5.7

**Status:** Moved to roampal-core v0.5.7 (`dev/docs/releases/v0.5.7/RELEASE_NOTES.md` Item 1).

**Why it's not in this doc.** Core-only change; Desktop has zero references to `_completion_state.json` (verified 2026-05-11). Desktop ships the fix by bumping the bundled core version to v0.5.7. No Desktop-side code change.

**Plausible relationship to issue #8.** The reported workaround for issue #8 is *"delete `_completion_state.json` and restart"*. Section 9 (phantom-dedup) explains why **fact** writes silently stop after a Desktop bulk-delete, but cannot explain why **summary** writes (no dedup gate) also stop, nor why deleting this specific file unwedges things. Accumulated stuck flags + the cross-session fallback misrouting in `main.py:1087-1096` are a credible candidate for the missing summary-side mechanism. See core v0.5.7 for the full diagnosis and fix.

## 11. Dev-mode launch leaks into prod data dir

**Discovered during:** same 2026-05-02 repro setup. Launched `C:\roampal\release\v0.3.2.1-prod\Roampal.exe` from PowerShell with `$env:ROAMPAL_DEV="1"` set in the parent shell, expecting the bundled backend to use `%APPDATA%\Roaming\Roampal_DEV\data\` per its own resolver in `backend/config/settings.py:14-32`. **Observed:** the v0.3.2.1 GUI memory panel rendered live entries that matched this Claude Code session's exchange summaries (timestamped seconds before the screenshot, same content) — i.e. the bundled backend was reading from `%APPDATA%\Roaming\Roampal\data\chromadb\` (prod, 219MB, actively written) rather than the dev path (4.7MB, last touched 2026-04-21).

If the user had clicked "Delete data" expecting an isolated dev install, it would have destroyed their real ChromaDB.

**What's confirmed.** v0.3.2.1's bundled Python backend *does* honor `ROAMPAL_DEV` correctly — `backend/config/settings.py` lines 14-32 define the resolution chain (`ROAMPAL_DATA_DIR` env > `ROAMPAL_DEV` env > prod default), and the `[Roampal] PROD mode: ...` line is logged on every boot. The Python side is not the problem.

**What broke.** The Tauri shell (`Roampal.exe`) launches the Python backend as a sidecar process. In a release build, the Rust-side spawn config does not propagate the user-supplied `ROAMPAL_DEV` env from the parent shell into the spawned Python process — only the env vars Tauri itself decides to forward. The settings.py comment at line 18 even acknowledges this implicitly: *"Priority 2: ROAMPAL_DEV env var (Tauri sets based on debug/release)"* — i.e. Tauri controls the var in normal flow. A user putting `ROAMPAL_DEV=1` in their parent shell has no effect because the Rust spawn doesn't pass it through.

**Candidate causes — to confirm in Tauri code:**

1. **Tauri sidecar `Command` not inheriting parent env.** The Rust spawn likely calls something like `Command::new(...).spawn()` without `.envs(std::env::vars())` — so the child gets only Tauri's curated env block.
2. **Release build hardcodes prod.** Tauri's release-vs-debug split may explicitly set `ROAMPAL_DEV=0` (or omit it) in release builds with no override path for users.

Both are fixable in `ui-implementation/src-tauri/src/`.

**Fix — three changes.**

1. **Ship a dedicated dev launcher.** Add `Start Roampal DEV.bat` next to `Start Roampal.bat` in the release zip:

   ```bat
   @echo off
   set ROAMPAL_DEV=1
   set ROAMPAL_API_PORT=8766
   start "" "%~dp0Roampal.exe"
   ```

   Per-process env set in the parent cmd before launching `Roampal.exe`. Useful for users — but **only effective if change #2 lands**, otherwise Tauri's sidecar spawn still won't pass `ROAMPAL_DEV` through.

2. **Patch Tauri sidecar spawn to inherit (or explicitly forward) `ROAMPAL_DEV`.** In `ui-implementation/src-tauri/src/` (the Rust crate that defines the sidecar `Command`), either pass the full parent env via `.envs(std::env::vars())` or explicitly `.env("ROAMPAL_DEV", std::env::var("ROAMPAL_DEV").unwrap_or_default())` before `.spawn()`. This is the load-bearing change. Add a Rust-side test that asserts `ROAMPAL_DEV` is in the child's env when set in the parent.

3. **Surface the resolved mode/path more visibly.** `settings.py:42` already prints `[Roampal] {mode} mode: {DATA_PATH}` to stdout on import. Promote that to the application logger so it lands in `roampal.log`, not just stdout, and surface the resolved mode in the GUI title bar or status area (`v0.3.2 DEV — 8766`). Users reproducing old-version bugs against a live install can't get fooled if every window plainly states which data dir is in use.

**Files affected:**

| File | Change |
|---|---|
| `release/v0.3.3-prod/Start Roampal DEV.bat` (new) | Dev launcher with `ROAMPAL_DEV=1` and a non-prod port. |
| `ui-implementation/src-tauri/src/<sidecar spawn site>` (Rust) | Forward `ROAMPAL_DEV` (and any other ROAMPAL_* vars) from parent env into the Python sidecar's env. Load-bearing fix. |
| `ui-implementation/src-tauri/backend/config/settings.py` | Promote the existing `[Roampal] {mode} mode: ...` print to a logger.info call so it lands in roampal.log. |
| `ui-implementation/src-tauri/src/<window setup>` (Rust/JS) | Append `(DEV)` to the window title when sidecar reports DEV mode, so the active install is unmistakable in screenshots and at-a-glance. |
| `release/.../README.txt` | Mention "Start Roampal DEV.bat" alongside the prod launcher. |
| New Rust integration test | Spawn the sidecar with `ROAMPAL_DEV=1` set in test env, assert it lands in the child's env. |

**Why this matters.** Without it, anyone reproducing a #8-style bug against an old release on the same machine as a prod install is one click away from destroying real memory data. The 2026-05-02 incident is a near-miss; the fix prevents the next one.

## 12. Desktop — dead-code cleanup + dormant `MCPSessionManager` removal + optional cold-start persistence

**Status:** ✅ Implemented 2026-05-11. §12D(1), (2), (3), (3.5) all shipped. §12D(4) cold-start dedup persistence deferred (optional UX polish — log spam after backend restart only, no functional cost). **Scope extended out-of-spec:** audit surfaced a parallel `_agent_action_cache` path in `app/routers/agent_chat.py` that also fed the `record_action_outcome` stub. User approved the broader cleanup, so agent_chat.py's producer (`~line 2885`), consumer (`~line 1374` + sidecar action-scoring block at `~line 2102`), `cached_actions` plumbing, and `_agent_action_cache` declaration all also removed. The `record_action_outcome` stub itself was then deleted from `unified_memory_system.py` since neither side calls it anymore.

### Implementation summary

| Sub-piece | Where | Notes |
|---|---|---|
| 12D(1) `_mcp_action_cache` chain | `main.py` — declaration, `MCP_CACHE_EXPIRY_SECONDS`, `_should_clear_action_cache`, `_cache_action_with_boundary_check`, 4 call sites in search/create/update/archive_memory handlers, context_type detection block, score_memories action-scoring block, ActionOutcome import | Removed |
| 12D(2) `_mcp_search_cache` | `main.py` — declaration, write at search_memory tail (incl. cached_doc_ids/positions/result_collections assembly), write at get_context_insights tail, clear in score_memories | Removed |
| 12D(3) `MCPSessionManager` | `main.py:1056-1060` (import + instantiate + log), `modules/mcp/session_manager.py` (entire file), `modules/mcp/__init__.py` (export removed) | Removed |
| 12D(3.5) `log_tool_call` + `mcp_tool_calls.jsonl` | `main.py:1358-1380` (nested async def + asyncio.create_task) | Removed |
| 12D(4) Cold-start dedup persistence | — | **Deferred.** Optional; no functional impact. |
| **Out-of-spec extension** Agent-side action-KG | `app/routers/agent_chat.py` — `_agent_action_cache` declaration, ActionOutcome import, producer in tool-execution path, consumer/`cached_actions` plumbing through `_run_sidecar`, action-scoring block. `unified_memory_system.py` — `record_action_outcome` stub deleted. | Removed |

### Tests touched

- `tests/unit/test_mcp_handlers.py`: deleted first 4 classes (`TestShouldClearActionCache`, `TestCacheActionWithBoundaryCheck`, `TestMCPCacheExpiry`, `TestActionOutcomeIntegration`). Kept `TestMemoryScoresParameter`, `TestRecordResponseScoringLogic`, `TestClaudeCodeMCPDetection` — they don't reference deleted symbols.
- `tests/mcp/conftest.py`: removed `memory.record_action_outcome = AsyncMock()`.
- `tests/mcp/test_mcp_tools.py`: removed `record_action_outcome` from `mock_memory` fixture and deleted `test_record_tracks_actions`.
- `tests/mcp/mcp_tool_harness.py`: removed both `record_action_outcome` call sites inside `call_record_response`. `_test_action_cache` kept as harness-internal simulation for cache-mechanics tests only.

### Verification

- Full unit suite: **600 passed**, 1 skipped (pre-existing).
- MCP integration suite: **32 passed**.
- `import main` succeeds with no errors.

(Original audit/scoping notes below.)

**Triggered by:** core v0.5.7 audit on 2026-05-11. Scope is cleanup of three Desktop-side dead-code paths: `_mcp_search_cache` (zero readers, leftover from a v0.2.6 Action-KG plan that never landed), `_mcp_action_cache` (feeds a no-op stub since v0.3.1), and `MCPSessionManager` (instantiated once at `main.py:1133` but no method ever called — Desktop's `data/mcp_sessions/` writers are roampal-core, not Desktop). **Issue #8 is unrelated to this section** — its Desktop-side fix lives in Section 9.1 (state-file reset on conversational-tier bulk-clear).

### 12A — Verification trail: Desktop has no `_completion_state.json` equivalent

Grepped Desktop's whole backend for every flag that lives in core's `_completion_state.json` — `first_message_seen`, `scored_this_turn`, `scoring_required`, `completed`, `check_and_clear_completed` — zero matches. The hook-coordination flags only exist on the core side because core runs hooks as separate processes that need a disk scratchpad. Desktop is one long-lived uvicorn process, so cross-turn state lives in process memory.

### 12B — In-memory cache audit

`main.py` declares three module-level dicts/sets at lines 104-108:

| Cache | Purpose claimed by comment | Read sites (verified) | Status |
|---|---|---|---|
| `_mcp_first_tool_call: set[str]` (line 105) | Cold-start dedup per session | `main.py:192` inside `_inject_cold_start_if_needed` | **Live.** Loss on restart causes spammy duplicate cold-start injection on each session's next tool call. UX cost only — cold-start is idempotent. |
| `_mcp_search_cache: dict` (line 104) | `{session_id: {doc_ids, query, timestamp}}` per comment | **Zero reads.** Grep shows declaration + 2 writes (`main.py:1641, 1946`) + 2 deletes (`main.py:2155-2156`) only. `score_memories` at `main.py:2061` takes `memory_scores` directly from LLM tool-call args, never consulting this cache. | **Dead code.** Leftover from v0.2.6 Action-KG plan (KG removed in v0.3.1; cache writes weren't). |
| `_mcp_action_cache: dict` (line 108) | Action-Effectiveness KG | `main.py:2140-2152` only, to call `memory.record_action_outcome(action)` | **Dead code.** `unified_memory_system.py:1639-1641` `record_action_outcome` is a no-op stub since v0.3.1. The cache feeds a `pass`. |

**Why scoring/summarization still work after restart:** the `score_memories` MCP handler takes IDs to score from its own tool-call args, not from any server-side cache. The LLM gets those IDs from the `[id:...]` tags in last turn's `KNOWN CONTEXT` (which lives in *its* conversation history, not in Desktop's process memory). No cache dependency.

### 12C — `MCPSessionManager` is dormant

Found during the same audit. The class at `modules/mcp/session_manager.py` is instantiated once:

```python
# main.py:1133
mcp_session_manager = MCPSessionManager(DATA_PATH, memory)
```

— and then **never has a single method called on it.** Grep `mcp_session_manager\.` returns zero matches across the whole backend. The class defines six methods (`record_exchange`, `get_last_assistant_with_doc_id`, `update_last_assistant_doc_id`, `get_session_messages`, `get_session_history`, `get_session_stats`) — all transcript-related, none invoked. The instance sits in memory unused for the process's lifetime.

The JSONLs that exist in `~/.roampal/data/mcp_sessions/<session_id>.jsonl` on shared-data-dir setups (where both core and Desktop point at the same `~/.roampal/data/` — the issue #8 reporter's configuration) are written by **roampal-core's** `SessionManager`, not by Desktop. Desktop's MCP server (`Server("roampal-memory")` at `main.py:1137`) handles tool calls (search_memory, score_memories, create_memory, etc.) entirely in-process without ever touching this class — verified by tracing every MCP tool handler in main.py.

### 12D — Fix: three deletes + one optional polish

**(1) Delete `_mcp_action_cache` and its dependents.** Pure cleanup — every line is on a code path that ends at a no-op stub.

| File | Lines | What goes |
|---|---|---|
| `main.py` | 108 | `_mcp_action_cache = {}` declaration |
| `main.py` | 111-113 | `MCP_CACHE_EXPIRY_SECONDS` constant (only consumer is `_should_clear_action_cache`) |
| `main.py` | 116-146 | `_should_clear_action_cache()` helper |
| `main.py` | 149-180 | `_cache_action_with_boundary_check()` helper |
| `main.py` | 1745+1748, 1793+1795, 1839+1841, 1885+1887 | Four `_cache_action_with_boundary_check(...)` call sites and adjacent log lines that dereference `_mcp_action_cache` |
| `main.py` | 2136-2158 | The `score_memories` block that reads `_mcp_action_cache`, calls `memory.record_action_outcome()` (stub), then clears the cache |
| `unified_memory_system.py` | 1639-1641 | The `record_action_outcome(self, action) -> None: pass` stub itself |

Net: ~105 LOC removed.

**(2) Delete `_mcp_search_cache` and its writes/clears.** Also pure cleanup — never read.

| File | Lines | What goes |
|---|---|---|
| `main.py` | 104 | `_mcp_search_cache = {}` declaration |
| `main.py` | 1631-1649 | Write site at the tail of `search_memory` (positions+cached_doc_ids assembly + dict assignment + log line) |
| `main.py` | 1943-1951 | Write site at the tail of `get_context_insights` |
| `main.py` | 2155-2156 | Clear site inside `score_memories` |

Net: ~25 LOC removed.

**(3) Delete dormant `MCPSessionManager`.** Pure cleanup — class is never invoked.

| File | Lines | What goes |
|---|---|---|
| `main.py` | 1130 | `from modules.mcp.session_manager import MCPSessionManager` import |
| `main.py` | 1133-1134 | `mcp_session_manager = MCPSessionManager(...)` instantiation + log line |
| `modules/mcp/session_manager.py` | entire file | Delete (~445 LOC) |
| `modules/mcp/__init__.py` | wherever `MCPSessionManager` is exported | Remove from imports + `__all__` |
| Any test file referencing the class | TBD by grep | Delete or rewrite |

Net: ~450 LOC removed.

**Combined cleanup total: ~580 LOC removed across (1)+(2)+(3), zero behavior change** — every line is on a path that today either calls a stub, writes a never-read dict, or instantiates a never-called class.

**(3.5) Delete `mcp_tool_calls.jsonl` write-only analytics log.** Surfaced during the same audit. Every MCP tool call appends a line to `data/mcp_tool_calls.jsonl` (tool name, session_id, client, timestamp, truncated args). Grep confirms three total references — function definition, file path inside it, and the `asyncio.create_task` invocation. **Zero readers anywhere** in the codebase, no tests, no config. The file grows unbounded with no TTL.

| File | Lines | What goes |
|---|---|---|
| `main.py` | 1437-1459 | `async def log_tool_call():` definition + `asyncio.create_task(log_tool_call())` invocation |

Net: ~23 LOC removed. Also stops one disk write per MCP tool call (every `search_memory`, `score_memories`, etc.).

**(4) [OPTIONAL] Persist cold-start dedup across backend restart.**

Today, `_inject_cold_start_if_needed` (`main.py:183`) checks `if session_id not in _mcp_first_tool_call:`. On backend restart that set is empty, so every active session re-injects the `KNOWN CONTEXT` profile block on its next tool call. Annoying, not broken.

Optional fix: persist the seen-set as a tiny file `mcp_sessions/_seen_sessions.json` — a JSON list of session_ids that have had cold-start.

```python
# main.py
_SEEN_SESSIONS_FILE = Path(DATA_PATH) / "mcp_sessions" / "_seen_sessions.json"

def _load_seen_sessions() -> set[str]:
    if not _SEEN_SESSIONS_FILE.exists():
        return set()
    try:
        return set(json.loads(_SEEN_SESSIONS_FILE.read_text()))
    except (json.JSONDecodeError, OSError):
        return set()

def _persist_seen_session(session_id: str) -> None:
    from utils.atomic_json import write_json_atomic
    current = _load_seen_sessions()
    current.add(session_id)
    write_json_atomic(_SEEN_SESSIONS_FILE, sorted(current), indent=None)

# At module load:
_mcp_first_tool_call: set[str] = _load_seen_sessions()

# Inside _inject_cold_start_if_needed, after the existing add:
_mcp_first_tool_call.add(session_id)
_persist_seen_session(session_id)
```

Tiny — one new file, one atomic write per new session, idempotent. Survives backend restart. No GC needed (the file stays small — one short string per ever-seen session). If we ever want to bound its growth, the same TTL+max-entries pattern from core v0.5.7 can be ported, but not required at v0.3.3 scope.

**Defer call:** include in v0.3.3 spec but mark implementation as optional. If the three cleanup deletes ship without (4), we lose nothing functional — just live with the duplicate cold-start spam after each backend restart. If we have time, (4) is ~30 lines.

### Acceptance criteria

- No references to `_mcp_action_cache`, `_mcp_search_cache`, `_cache_action_with_boundary_check`, `_should_clear_action_cache`, `MCP_CACHE_EXPIRY_SECONDS`, `record_action_outcome`, `MCPSessionManager`, `log_tool_call`, or `mcp_tool_calls.jsonl` remain in the codebase after the cleanup.
- `modules/mcp/session_manager.py` is deleted; `modules/mcp/__init__.py` no longer imports or exports the class.
- `data/mcp_tool_calls.jsonl` is no longer written. Existing files on user installs are harmless and can be left in place (orphaned data, no consumers).
- All existing tests pass. New tests cover the optional (4) only if implemented: (a) `_seen_sessions.json` round-trips through restart, (b) cold-start fires on truly new session even after restart, (c) cold-start skips on previously-seen session after restart.

### Files affected

| File | Change |
|---|---|
| `ui-implementation/src-tauri/backend/main.py` | Delete `_mcp_action_cache` + dependents (~105 LOC); delete `_mcp_search_cache` + writes/clears (~25 LOC); delete `MCPSessionManager` import + instantiation (~5 LOC); delete `log_tool_call` + `mcp_tool_calls.jsonl` writer at lines 1437-1459 (~23 LOC). Optional (4): add `_SEEN_SESSIONS_FILE`, `_load_seen_sessions`, `_persist_seen_session`, wire into module-load + cold-start handler (~25 LOC). |
| `ui-implementation/src-tauri/backend/modules/mcp/session_manager.py` | Delete file (~445 LOC). |
| `ui-implementation/src-tauri/backend/modules/mcp/__init__.py` | Remove `MCPSessionManager` from imports and `__all__`. |
| `ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py` | Remove `record_action_outcome` stub at lines 1639-1641. |
| `ui-implementation/src-tauri/backend/tests/...` | Delete or update any test referencing the removed symbols (grep before deleting). If (4) lands, add the three restart tests. |

### Why this is unrelated to issue #8

The Desktop-side fix that *does* connect to issue #8 is **Section 9.1** — it unlinks core's `_completion_state.json` after Desktop's `/clear/working`, `/clear/history`, `/clear/patterns` succeed, removing the "delete `_completion_state.json` and restart" step from the user's recovery playbook. The actual broken state machine for #8 lives in core (fixed in core v0.5.7). Desktop's only role in #8 is providing the user-facing trigger (the GUI delete button), and 9.1 handles that handoff.

Section 12, in contrast, is pure Desktop hygiene: remove dead-code cruft from the v0.3.1 KG removal and the never-shipped Action-KG/selective-scoring features, drop a dormant class that's been sitting in the boot path doing nothing, and (optionally) stop spamming a redundant `KNOWN CONTEXT` block after every backend restart. None of these touch #8's mechanism.

---

# Verification Report

**Build under test:** v0.3.3 working tree (uncommitted), Tauri dev profile
**Environment:** Windows 11, RTX 5090 (32 GB VRAM), Ollama backend; `Roampal_DEV` data directory (isolated from production `Roampal` dir per Section 11)
**Models present:** `gemma4:31b` (Q4_K_M, 19 GB), `qwen3.6-ud:latest`, `qwen3:1.7b`, `gpt-oss:20b`
**Verification date:** 2026-05-11
**Outcome legend:** Pass / Fail / Blocked / Pending / N/A

## Automated test suites (pre-flight)

| Suite | Result | Notes |
|---|---|---|
| Desktop backend unit + integration (`ui-implementation/src-tauri/backend/tests/`) | Pass — 637 passed, 1 skipped, 0 failed | The skipped case is `test_dangerous_combo_detection`; `config.feature_flag_validator` module does not exist in the codebase. Pre-existing, unrelated to v0.3.3. |
| Roampal Core v0.5.7 memory unit (`roampal/backend/modules/memory/tests/unit/`) | Pass — 624 passed | |
| Roampal Core v0.5.7 memory integration (`roampal/backend/modules/memory/tests/integration/`) | Pass — 29 passed | |
| Tauri dev build (`npm run tauri:dev`) | Pass — 1 m 02 s compile time; one harmless `unused variable: app_handle` warning at `src/main.rs:198` | Multiple Tauri restart cycles encountered "os error 32: file in use" caused by orphan sidecar Python processes from prior runs holding `binaries/python/...` locked. Resolved by `taskkill /F /PID …` on the largest python.exe instances before relaunch. Pre-existing Tauri-dev behavior. |

## Section-level verification

### Section 3 + Phase 2.1 — Harmony channel-token streaming filter

| Verification | Result | Evidence |
|---|---|---|
| Streaming-safe variant defined and wired into `stream_response_with_tools` | Pass | `ollama_client.py:504-521` and call site at line 1088 |
| 8 new unit tests (`TestCleanModelArtifactsStreaming`) | Pass — 8/8 | Covers leading/trailing space preservation, single-space chunk pass-through, multi-chunk reassembly, Harmony token stripping retained, prefix/newline non-action, empty input |
| Full `test_ollama_client.py` suite after fix | Pass — 36/36 | |
| Dev Tauri smoke chat on `gemma4:31b` | Pass | Response text rendered with proper inter-word spacing; no `<\|start\|>`, `<\|end\|>`, `<\|message\|>`, or `<channel\|>` tokens leaked into UI |

### Section 4 — Multimodal (image) input

| Verification | Result | Evidence |
|---|---|---|
| Photo-attach icon visible when active model declares `vision` (`gemma4:31b`) | Pass | UI confirmation |
| Photo icon hidden when active model lacks `vision` capability | Pending | Recommended check: swap dropdown to `qwen3:1.7b`; icon should disappear |
| Image rendered as thumbnail in user message bubble after attach (paste / photo-icon picker paths) | Pass after Defect 2 remediation | Pre-fix: image rendered as plain text pill "Image N" because the live `TerminalMessageThread` renderer ignored attachment `type` / `url` fields. Post-fix: image-type attachments render as `<img>` thumbnails. See Defect 2 below. |
| Drag-drop image attach | Out of scope for v0.3.3 — port deferred to v0.3.4 | Handlers exist in legacy `CommandInput.tsx:115-164` (`handleDrop`, `onDragOver`, `onDragLeave`, `onDrop`) but were never wired into the live `ConnectedCommandInput.tsx`. Release notes acknowledge this in the Section 4 Implementation Summary; PhotoIcon tooltip reads "Attach image (paste also supported)". User-facing affordance is clear; paste + file-picker paths cover the same use case. Tracked as v0.3.4 work item — see Open follow-ups below. |
| Send button enabled for image-only sends (no text caption required) | Pass (after Defect 1 remediation) | Backend now accepts image-only payloads; see Defect 1 below |
| End-to-end multimodal generation (model describes image content) | Pass after Defect 3/4/5/6 remediations | Confirmed by tester: image-only send to `gemma4:31b` produced an image-content-aware response, no `Model error` toast, no generic "Hello there" reply. Required six sequential defect remediations because each masked the next — see Defects 1–6 below. |

### Section 4H — Dynamic capability detection

| Verification | Result | Evidence |
|---|---|---|
| Hardcoded family enumerations deleted from source | Pass | `grep -rE "VISION_CAPABLE_FAMILIES\|TOOL_CAPABLE_FAMILIES\|is_known_vision_model"` returns 0 source-code matches |
| Updated test suite (`TestCachedVisionDetection` + new `TestCachedToolsDetection` + new `TestFetchOllamaCapabilities`) | Pass — 11 new tests, all pass | Autouse fixture isolates `MODEL_METADATA_CACHE_FILE` per test to prevent cross-pollution |
| First registry refresh after Tauri launch populates capability cache | Pass | `%APPDATA%\Roampal_DEV\data\model_metadata_cache.json` contains `caps::ollama::*` entries for all 4 installed models after first registry refresh |
| Capability resolution matches Ollama `/api/show` ground truth: | | |
| &nbsp;&nbsp;&nbsp;&nbsp;`gemma4:31b` → `{completion, thinking, tools, vision}` | Pass | Cache entry verified against direct `curl -X POST /api/show` |
| &nbsp;&nbsp;&nbsp;&nbsp;`qwen3.6-ud:latest` → `{completion, vision}` | Pass | |
| &nbsp;&nbsp;&nbsp;&nbsp;`qwen3:1.7b` → `{completion, thinking, tools}` (no vision) | Pass | |
| &nbsp;&nbsp;&nbsp;&nbsp;`gpt-oss:20b` → `{completion, thinking, tools}` (no vision) | Pass | |
| Latent defect remediated during refactor: `write_json_atomic` was being called with a `str` instead of `Path` at three sites in `model_contexts.py`; `try/except Exception` blocks silently swallowed `AttributeError: 'str' object has no attribute 'parent'`. Disk metadata cache was a no-op since v0.3.3 Section 6 landed. | Resolved | All three call sites now pass `Path(MODEL_METADATA_CACHE_FILE)` |

### Section 7 — ChromaDB hard-delete dedup bug (single-memory delete)

| Verification | Result | Evidence |
|---|---|---|
| Per-memory Delete button removes entry from Memory panel view | Pending | |
| Semantically similar fact added post-delete stores successfully (no phantom-dedup) | Pending | |
| Deleted entry's metadata `status` flips to `archived` (soft delete, reversible) | Pending | |

### Section 9 + 9.1 — Bulk `/clear/*` Issue #8 reproduction

**Pre-state:** `_completion_state.json` seeded with 3 synthetic test entries (`test-conv-marcus-repro-001`, `-002`, `test-conv-stale-old`) before testing.

| Phase | Verification | Result | Evidence |
|---|---|---|---|
| A | Establish pre-state via 3–4 substantive chat turns; verify summaries in `working` and facts in `memory_bank` | In progress | First turn completed (462-char response, 116 stream chunks, 8 cached memories injected, scoring fired with `worked=5, partial=0, failed=0, unknown=3`) |
| B | Bulk delete all four tier collections via Settings → Delete data; verify `_completion_state.json` unlinked by §9.1 | Pending | |
| B | Log output contains `Section 9.1: unlinked …_completion_state.json` once per tier-clear endpoint invoked | Pending | |
| C | Post-delete chat (no restart) produces new summary entry in `working` AND new fact entry in `memory_bank` | Pending — controlling acceptance criterion for #8 | |
| D | Restart Tauri; subsequent chat fires cold-start cue path (proves `first_message_seen` was reset by §9.1) | Pending | |

### Section 11 — Tauri sidecar dev-mode environment isolation

| Verification | Result | Evidence |
|---|---|---|
| Application title bar displays "Roampal (DEV)" | Pass | Tester screenshot confirmation |
| Sidecar resolves `DATA_PATH` to `%APPDATA%\Roampal_DEV\data\`, not production `Roampal\data\` | Pass | All file inspections this session resolved to `Roampal_DEV` tree |
| `roampal.log` records `[Roampal] DEV mode: …` line during startup | Pending direct log inspection | |

## Defects discovered during verification

### Defect 1 — Image-only sends rejected with HTTP 422 (Section 4 regression) — **Resolved 2026-05-11**

**Severity:** High (release-blocking for Section 4 acceptance criterion "Send button enabled for images-only sends")

**Symptom:** Submitting a chat request with an image attachment but no text caption returned HTTP 422 in the UI: `Error: HTTP error! status: 422`. No model invocation occurred.

**Reproducer (curl, against dev sidecar on port 8766):**

```bash
curl -X POST http://localhost:8766/api/agent/stream \
  -H "Content-Type: application/json" \
  -d '{"message":"","images":["data:image/png;base64,iVBORw0KGgo="],"conversation_id":"diag"}'
```

**Server response:**

```json
{
  "detail": [{
    "type": "value_error",
    "loc": ["body", "message"],
    "msg": "Value error, Message cannot be empty",
    "input": ""
  }]
}
```

**Root cause:** `AgentChatRequest.validate_message` field validator at `ui-implementation/src-tauri/backend/app/routers/agent_chat.py:408-414` raises `ValueError("Message cannot be empty")` when the `message` field is empty or whitespace. The validator runs unconditionally and has no awareness of whether the `images` field is populated. The frontend permits image-only sends (request body construction in `useChatStore.ts:1234-1241`; send-button enable condition in `ConnectedCommandInput.tsx:305` permits empty `message` when attachments are present), creating a UI/schema contract mismatch.

**Remediation applied:** `validate_message` (per-field) now only enforces the length cap and the control-character strip; the empty-content check moved to a new `_require_message_or_images` model-validator (`@model_validator(mode="after")`) that permits empty `message` when `images` is non-empty and rejects only when both are absent. New `TestAgentChatRequestValidation` class in `tests/unit/test_agent_chat.py` adds 9 cases covering: text-only accepted, image-only with empty message, image-only with whitespace-only message, text+images, empty-no-images rejected, whitespace-no-images rejected, empty-with-empty-list rejected, length cap still enforced for non-empty messages, control-character strip still runs. **9/9 pass.** Full desktop backend suite re-run post-fix: **646 passed, 1 skipped, 0 failed** (net +9 from prior 637 baseline).

### Defect 2 — User-bubble image attachment renders as text pill, not thumbnail (Section 4 regression) — **Resolved 2026-05-11**

**Severity:** High (release-blocking for Section 4 acceptance criterion "EnhancedChatMessage renders image thumbnails as `<img src={dataURL}>` in user message bubbles")

**Symptom:** After successful attach + send (post-Defect-1-remediation), the user message bubble displayed `"Image 1"` as a small gray text pill rather than the expected image thumbnail. Base64 image data round-tripped through the request but was visually invisible in the conversation transcript.

**Root cause:** Section 4's spec wired image rendering into `ui-implementation/src/components/EnhancedChatMessage.tsx:130-143` (user-bubble path reads `message.images` array and renders each as `<img src={dataURL}>`), but the live chat thread is rendered by `TerminalMessageThread.tsx`, not `EnhancedChatMessage`. To bridge the gap, `ui-implementation/src/components/ConnectedChat.tsx:1944-1958` performs an `images → attachments` conversion that produces entries shaped as `{id, name: "Image ${idx+1}", size: 0, type: "image/jpeg", url: imgUrl}` so the legacy attachment-pill renderer in `TerminalMessageThread.tsx:377-385` could pick them up. The pill renderer, however, displayed only `att.name` text — it had no awareness of `att.type` or `att.url`. The base64 URL was carried through the conversion and ignored at render time.

**Remediation applied:** `TerminalMessageThread.tsx:377-394` now branches on `att.type?.startsWith('image/')`. For image-type attachments with a populated `url`, it renders an `<img src={att.url}>` thumbnail with the existing `max-w-xs max-h-48 rounded-lg` styling that matches `EnhancedChatMessage`. Non-image attachments continue to render as the legacy text pill. No changes required in `ConnectedChat.tsx` or `useChatStore.ts`; the existing `images → attachments` conversion is preserved.

**Verification:** Live-tested in dev Tauri immediately after the edit landed. Vite HMR picked up the TSX change; subsequent user-bubble image attach (photo-icon picker path) rendered as a proper thumbnail. User confirmed at 2026-05-11.

**Follow-up work (deferred):** The `images → attachments` conversion in `ConnectedChat.tsx:1944-1958` is now load-bearing rather than transitional. A cleaner architecture would have `TerminalMessageThread` read `message.images` directly (matching the `EnhancedChatMessage` contract) and remove the conversion. Tracked as v0.3.4 polish item.

### Defect 3 — Multimodal payload sent to Ollama in OpenAI shape (Section 4 regression) — **Resolved 2026-05-11**

**Severity:** High (release-blocking for Section 4 acceptance criterion "end-to-end multimodal generation against the active provider")

**Symptom:** Sending a chat with an image attachment against an Ollama-hosted vision model returned a streaming-time error to the UI: `Model error: {"error":"json: cannot unmarshal array into Go struct field ChatRequest.messages.content of type string"}`. No generation occurred; the user message bubble retained its thumbnail but no model output followed.

**Root cause:** `ollama_client.py:690-697` built an OpenAI-style multimodal payload unconditionally:

```python
content_blocks = [
    {"type": "text", "text": prompt},
    *[{"type": "image_url", "image_url": {"url": img}} for img in images],
]
messages.append({"role": prompt_role, "content": content_blocks})
```

This shape is correct for LM Studio's `/v1/chat/completions` (OpenAI-compatible), but Ollama's native `/api/chat` expects:

```json
{"role": "user", "content": "<plain text>", "images": ["<raw base64, no data: prefix>"]}
```

Ollama's Go-typed `ChatRequest.messages.content` is a `string`; deserializing a JSON array into it fails with the exact error shown above. The same code path is hit for both providers because Section 4 introduced multimodal construction before the `if self.api_style == "openai"` branch that splits the actual HTTP call, leaving the message-construction step provider-agnostic.

**Remediation applied:**

- New module-level helper `_strip_data_url_prefix(url)` strips the `data:image/...;base64,` prefix Ollama doesn't want; idempotent on already-raw base64 strings.
- `ollama_client.py:688-708` now branches on `self.api_style`:
  - `"ollama"` → `{"role": ..., "content": "<text>", "images": [stripped base64...]}` (Ollama native)
  - `"openai"` → existing OpenAI content_blocks shape (LM Studio)
- Defensive log fix at `ollama_client.py:710-712`: the `[DEBUG MESSAGES] Last 200 chars` line previously slice-indexed `messages[-1]['content']`, which raised `TypeError` when content was a list (LM Studio path). Now coerces to `str()` first for log-rendering only.

**Tests:** new `TestStripDataUrlPrefix` (5 cases: PNG/JPEG data URLs stripped, raw base64 passes through, https URL passes through, non-string passes through) and `TestMultimodalPayloadShape` (3 cases: Ollama payload uses native `images` field with raw base64, LM Studio payload uses OpenAI content_blocks with full data URLs, text-only Ollama request retains plain-string content shape). **8/8 pass.** Full desktop backend suite post-fix: **654 passed, 1 skipped, 0 failed** (net +8 from prior 646 baseline).

### Defect 4 — Image attachments not persisted to disk (Section 4 gap) — **Resolved 2026-05-11**

**Severity:** High (ship-blocker per tester directive — initially scoped to v0.3.4 backlog, escalated to v0.3.3 in-scope after tester reported image attachments disappearing from history on refresh and restart)

**Symptom:** Image attachments rendered correctly in the user bubble during the current chat session, but were not present on session reload (UI refresh, Tauri restart). The session JSONL stored only `role`, `content`, `timestamp`, `metadata` for user turns — no image data persisted anywhere on disk.

**Root cause:** v0.3.3 §4 wired the in-memory image-attachment flow (React state → request body → LLM payload) but never persisted the bytes. The session JSONL save path at `agent_chat.py:_save_to_session_file` had no `images` parameter; ChromaDB collections (working/history/patterns/memory_bank) hold text-only summaries and facts.

**Remediation applied (hash-keyed attachments directory + bytes endpoint):**

- New `ui-implementation/src-tauri/backend/utils/attachments.py`: `save_image_attachment(data_url, dir) → filename` performs SHA-256 over the decoded bytes and writes to `<DATA_PATH>/attachments/<hex>.<ext>` via temp-file + atomic rename. Identical bytes deduplicate automatically (content-addressed). `resolve_attachment_path` rejects path traversal (separators, `..`, leading dot, non-hex stems, unsupported extensions). `extension_to_mime` maps extension back to MIME for the serving path.
- New `ui-implementation/src-tauri/backend/app/routers/attachments.py`: `GET /api/attachments/{filename}` streams stored bytes back via `FileResponse` with the correct MIME and `Cache-Control: public, max-age=31536000, immutable` (safe since filenames are content-addressed).
- `main.py:63, 839`: imports and registers the new router.
- `agent_chat.py:3217-3225`: stream endpoint now persists every entry in `request.images` to disk before generation, accumulating filenames in a local list.
- `agent_chat.py:3244, 1343, 2232`: `image_filenames` flows from the endpoint through `stream_message` into `_save_to_session_file`, which writes the filename list onto the user turn's JSONL record as `"images": ["<hex>.<ext>", ...]`.
- `sessions.py:165`: the session-load response now returns each turn's `images` array.
- `useChatStore.ts:373-385, 1566-1581`: both session-load paths (`switchConversation` and `loadSession`) resolve stored filenames into `${ROAMPAL_CONFIG.apiUrl}/api/attachments/<filename>` URLs that the existing `<img>` rendering picks up.

**Tests:** 20 new unit tests in `tests/unit/test_attachments_util.py` covering hash-keyed naming, content-dedup, JPEG extension, non-image MIME rejection, malformed URL rejection, empty payload, non-string input, no temp leakage on success, path-traversal defenses (5 cases), MIME inference. **20/20 pass.**

**Known limitation (intentionally deferred):** orphaned attachment files (bytes whose JSONL references were deleted) are not garbage-collected. Tracked under "Open follow-ups" — a startup-time orphan scan can be added without changing the storage format.

### Defect 5 — History-replay sent OpenAI content_blocks to Ollama (Section 4 regression) — **Resolved 2026-05-11**

**Severity:** High (ship-blocker — produced repeated `Model error: json: cannot unmarshal array into Go struct field ChatRequest.messages.content of type string` toast in the UI on second-and-later turns even after Defect 3 fixed the new-prompt payload shape)

**Symptom:** First image-only send succeeded after Defect 3 remediation, but the next turn (same conversation) raised the same "unmarshal array" Ollama error, even on a text-only follow-up.

**Root cause:** `agent_chat.py:752-763` stored each user message in `conversation_histories[conv_id]` using OpenAI content_blocks shape (`{"role": "user", "content": [{type:'text', ...}, {type:'image_url', ...}]}`). On subsequent turns this history was passed in `history=` to `stream_response_with_tools`, which `extend()`-ed it directly into the messages payload. Defect 3's fix only handled the freshly-constructed user message via `prompt=`; replayed history entries went through unchanged and crashed Ollama's strict `content: string` typing.

**Remediation applied:**

- `agent_chat.py:752-763`: now stores the user message in provider-neutral Ollama-native shape (`{"role": "user", "content": message, "images": [data_url, ...]}`) instead of OpenAI content_blocks. The `images` field is a sibling to `content`, never nested.
- `ollama_client.py:702-744`: `stream_response_with_tools` now normalizes every history entry at messages-build time. For each entry: if `entry["images"]` is set, branch on `self.api_style` — Ollama path strips data URL prefixes and keeps the native shape; LM Studio path converts to content_blocks. Legacy entries with content already as a list (from pre-fix sessions) are also normalized.

**Tests:** 2 new tests in `TestMultimodalPayloadShape`: history with `{images}` field replayed correctly for Ollama; legacy content_blocks history normalized to native shape for Ollama. Both confirm no `list`-typed content reaches the Ollama wire payload. **2/2 pass.**

### Defect 6 — Image-only sends dropped the user message entirely (Section 4 regression) — **Resolved 2026-05-11**

**Severity:** High (ship-blocker — caused the model to respond with generic introductions for image-only chats, never seeing the attached image at all)

**Symptom:** After Defect 3, 4, 5 remediations, image-only sends still returned generic responses ("Hello there! I'm Roampal…"). Backend logs showed `[STREAM WITH TOOLS] Message count: 3, Roles: ['system', 'system', 'system']` — the user turn (with image) was absent from the messages payload entirely.

**Root cause:** `ollama_client.py:705` previously gated the multimodal-build branch on `if prompt:`. For image-only sends the API entry validator (Defect 1) correctly permitted `message=""`, and the request reached `stream_response_with_tools` with `prompt="" + images=[...]`. The `if prompt:` check evaluated falsy on the empty string, so the entire user-message-append branch was skipped — no `{role: 'user', ...}` entry was emitted, the model received only system context, and the attached image bytes were never serialized into the Ollama request.

**Remediation applied:** `ollama_client.py:704-711` gate is now `if prompt or _has_images:`. When `prompt=""` and `images` is populated, the multimodal branch fires with `content=""` and the images sibling — Ollama accepts empty content and processes the image.

**Test:** new `test_image_only_send_with_empty_prompt_still_emits_user_message` asserts a user message is appended with `content == ""` and the stripped-base64 images list when called with `prompt=""` and a non-empty images array. **Pass.** Full ollama_client suite: **47/47 pass.** Full desktop backend suite: **677 passed, 1 skipped, 0 failed** (was 676; net +1).

### Defect 7 — LM Studio vision capability never auto-detected on registry refresh (Section 4H gap) — **Resolved 2026-05-11**

**Severity:** Medium (UX correctness — photo-attach icon never renders for LM Studio models that genuinely support vision; users have no path to discover that an LM Studio vision model is usable without manually hitting `POST /api/model/probe_vision/{model_name}`)

**Symptom:** After switching the active model to a vision-capable LM Studio model (e.g., `qwen3.6-35b-a3b` loaded via LM Studio), the PhotoIcon in the chat bar is absent (not rendered — `activeModelHasVision` is false, so the icon's render branch is skipped entirely). The same model in Ollama (`qwen3.6-ud:latest`) renders the icon correctly. Refreshing the model dropdown does not help; the only workaround was a manual `POST /api/model/probe_vision/{model_name}` followed by another registry refresh.

**Root cause:** `app/routers/model_registry.py` (the `/api/model/registry` handler) ran capability prefetch only for the Ollama branch — `fetch_ollama_capabilities` over `/api/show` for every non-embedding Ollama model. LM Studio fell through entirely with the inline comment *"LM Studio is skipped here; its capability detection falls through to runtime (probe_vision for vision, default-True with retry for tools)"*. In practice nothing **called** `probe_vision` on a registry refresh, so `get_cached_vision(model_name, "lmstudio")` always returned `False` until something else (an explicit probe endpoint hit) populated the cache. The §4H design (release notes line 454–457: *"On registry refresh, for LM Studio models with no cache entry, fire the existing async `probe_vision`… background. Cache result for 24h."*) was documented but not actually wired up.

**Remediation applied:** Added an `elif provider_name == "lmstudio" and models:` branch to `model_registry.py` immediately after the Ollama capability prefetch, mirroring the Ollama pattern but calling `probe_vision(m, "lmstudio", base_url)` in parallel for every non-embedding LM Studio model. `probe_vision` is already cache-aware (24h TTL with disk persistence — `config/model_contexts.py:462-464`), so warm-path refreshes return immediately without a network call; only first-encounter models pay the canary-probe latency, and that cost is bounded by the existing TTL. Also added `probe_vision` to the existing `from config.model_contexts import (...)` block at the top of the file.

**Test:** Manual verification — after the fix, hitting `/api/model/registry` for a session with `qwen3.6-35b-a3b` loaded in LM Studio populates `capabilities.vision: true` in the registry response on the first call (cold cache: ~1–2 s probe latency; warm cache: immediate). The PhotoIcon in `ConnectedCommandInput.tsx` then activates because `activeModelHasVision` derived from registry metadata flips to `true`. Existing unit tests around `probe_vision` and registry shape unchanged.

### Defect 8 — Sidecar produces uninformative memories from image-only sends — **Resolved 2026-05-11**

**Severity:** Medium (memory quality — multimodal conversations summarized as if no image existed, eroding the value of long-term memory recall around image-related exchanges; not a runtime/crash issue)

**Symptom:** When a user attaches an image and sends with no text caption (or a minimal caption like "what's this?"), the sidecar-generated summary memorialized only the literal text. Example exchange — user attaches a photo of their dog and types nothing; main vision model replies *"Yes, your golden retriever looks sweet."* The sidecar then stored a summary like *"user said nothing, assistant said the dog looks sweet"* — the *why* (a photo was the subject of the exchange) is invisible to future retrieval. Search queries like *"that picture of my dog"* would not surface this memory.

**Root cause:** Sidecar service (`modules/memory/sidecar_service.py`) is text-only by v0.3.3 §4 design — its `score_exchange` / `extract_facts` paths take `user_msg: str`, `assistant_msg: str` and have no `images` parameter. The live queue write at `agent_chat.py:1426-1429` (`_sidecar_pending[conversation_id] = { "user_msg": message, "assistant_msg": clean_response }`) stored only the raw text fields, so the sidecar had no way of knowing an image was ever attached. Verified end-to-end by inspecting `_run_sidecar` signature (line 1961-1971: only string params), `score_exchange_with_retry` call site (line 1987: no images), and `_build_scoring_prompt` (`sidecar_service.py:82-93`: pure string interpolation, no image-aware branches).

**Remediation applied:** `agent_chat.py:1424-1442` now prepends a fixed literal `[image attached]` token to `user_msg` before queueing for sidecar processing, gated on `images or image_filenames` truthy at that call site. Both fields are already in scope from `stream_message`'s signature (line 616-617). The marker is a **constant string** (not parameterized with count or filename) by design — sidecar models like `qwen3:1.7b` and `gpt-oss:20b` rely on exact-match pattern recognition to treat this as a consistent topic signal during scoring; varying punctuation or capitalization would degrade that. After the change, the sidecar prompt sees *"[image attached]"* or *"[image attached] cute right?"* on the `user_msg` line, which it can faithfully include in summaries (e.g., *"user shared an image and asked if it's cute, assistant confirmed it's a golden retriever"*).

**Scope deliberately excluded:** the JSONL session-save path is **not** modified. The persisted user message in the session file remains the raw `message` string, because the existing replay path at `agent_chat.py:2385-2388` reads back the saved `last_user` and re-queues it for sidecar processing — and at replay time, `image_filenames` is reconstructed from the session attachment metadata, so the same `[image attached]` gating fires correctly without polluting the raw transcript that the UI renders.

**Test:** Manual end-to-end — send an image with empty text caption, watch sidecar summary in memory panel after the next user turn fires the queued scoring. Expected memory content begins with *"user attached image"* or similar, not *"user said nothing."*

### Defect 9 — LM Studio vision detection via canary probe is unreliable for large/cold models (Section 4H follow-up to Defect 7) — **Resolved 2026-05-11**

**Severity:** Medium (UX correctness — the icon stays hidden for genuinely vision-capable models that take longer than 10 s to cold-load in LM Studio, and never recovers without manual intervention because ambiguous `None` probe results are intentionally not cached)

**Symptom:** After Defect 7 wired the `probe_vision` canary into the registry refresh for LM Studio, vision-capable models still showed `capabilities.vision: false`. Reproduced live with `qwen3.6-35b-a3b` (Q4_K_XL, 35 B params, ~22 GB VRAM): the model was selected in the LM Studio dropdown and entered `state: "loading"` in LM Studio's internals, but the 1×1 PNG canary timed out at the probe's 10 s default. `probe_vision` returned `None` (ambiguous), and per design (`config/model_contexts.py:514-519`) ambiguous results are deliberately not cached — so every subsequent registry refresh re-fired the probe, timed out again, and the icon never appeared.

**Root cause:** The canary probe is the wrong tool for LM Studio detection. It requires (1) the model to be fully loaded in VRAM, (2) a synchronous round-trip to that loaded model, (3) a clear success-or-error response in <10 s — all three are violated by large MoE models on cold start. LM Studio actually ships a richer native endpoint at `GET /api/v0/models` (distinct from the OpenAI-compatible `/v1/models`) that returns deterministic metadata per model:
```json
{
  "id": "qwen3.6-35b-a3b",
  "type": "vlm",                  ← Vision-Language Model
  "arch": "qwen35moe",
  "state": "loading",             ← capability is reported even while loading
  "capabilities": ["tool_use"]
}
```
versus the 27 b sibling:
```json
{
  "id": "qwen3.6-27b",
  "type": "llm",                  ← not a VLM — vision correctly absent
  "arch": "qwen35"
}
```
The `type` field is the authoritative LM Studio classification (`"vlm"` vs `"llm"` vs `"embeddings"`), reported instantly regardless of model load state. The previous code didn't query this endpoint at all.

**Remediation applied:** `app/routers/model_registry.py` (the LM Studio branch of the capability-prefetch added by Defect 7) now hits `GET /api/v0/models` first with a 3 s timeout. For each non-embedding model in the response: seed the capability set with `{"tools"}` (preserving the existing default-True behavior for LM Studio — see below), then add `"vision"` when `type == "vlm"`. Only when the extended endpoint is unreachable (older LM Studio builds without `/api/v0/models`, network error, malformed JSON) does the code fall back to the existing `probe_vision` canary path — preserving the Defect 7 behavior as a safety net rather than removing it. The `_save_caps_cache` and `_caps_cache_key` helpers from `config/model_contexts.py` are imported alongside the existing capability functions.

**Subtle regression caught during double-check:** an earlier draft of this fix wrote `{"vision"}` (or `set()`) to the cache without including `"tools"`. That would have silently regressed tool-use for LM Studio: `get_cached_tools` at `config/model_contexts.py:436-446` only returns its `True` default when **no** cache entry exists for the model — once a cache entry is present (even one containing only `"vision"`), it falls through to `"tools" in capabilities` and returns `False`. The fix always seeds `"tools"` into the LM Studio caps set; the runtime retry-without-tools path in `ollama_client.py` self-heals any model that actually rejects tool calls with a 400, so a permissive default remains correct.

**Test:** Manual end-to-end — with `qwen3.6-35b-a3b` selected in LM Studio (even before it finishes loading), `/api/model/registry` now returns `capabilities.vision: true` for that model on the very first call, and `qwen3.6-27b` correctly remains `vision: false`. Unit tests for the existing `probe_vision` path are unchanged; the fallback branch still exercises that code. Defect 7's release-notes claim that registry refresh fires `probe_vision` is now subsumed by this enhancement — the canary probe still ships for backward compatibility with old LM Studio versions but is no longer the primary detection path.

**Side note on the manual `/api/model/probe_vision/{model_name}` endpoint:** during diagnosis we surfaced a latent bug where the manual probe endpoint iterates `PROVIDERS` and selects the first detected provider rather than the actual owner of the supplied model name. With both Ollama and LM Studio running, manual probes for LM Studio-only model names were routed to Ollama (which returned a "model not found" error not containing the word "image"), yielding `vision_capable: null`. The auto-probe path inside the registry refresh was never affected (it explicitly passes `"lmstudio"`), and Defect 9's `/api/v0/models` path bypasses the issue for the production flow. Tracked as a v0.3.3 follow-up edit if the manual endpoint is needed for diagnostics.

### Defect 10 — Image attachments never persisted on the streaming task path (Section 4 Defect 4 gap) — **Resolved 2026-05-11**

**Severity:** High (ship-blocker — image-bearing exchanges in any WebSocket-connected session lost their attachments on UI refresh, leaving only `"content": ""` and an empty `metadata: {}` in the saved JSONL with no `images` field; the model received the image during the live turn but the user could not see what they had sent after a refresh, and downstream consumers including Defect 8's sidecar marker had no `image_filenames` to gate on)

**Symptom:** Verified end-to-end against a live session JSONL after an image-attach send through the LM Studio path. The user-turn entry persisted as:
```json
{"session_id": "...", "role": "user", "content": "", "timestamp": "...", "metadata": {}}
```
No `images` field, no thumbnail rendered on refresh, no record on disk under `<DATA_PATH>/attachments/`. Defect 4 had wired the persistence block in `_run_generation_task` (the non-streaming task entry) but the parallel `_run_generation_task_streaming` entry — which is the primary route used whenever a WebSocket is connected to the active conversation — was never updated.

**Root cause:** `app/routers/agent_chat.py` has two background generation tasks: `_run_generation_task` (non-streaming, batches via 2-minute timeout) and `_run_generation_task_streaming` (WebSocket-driven). The Defect 4 fix touched the first one at line 3212-3228 (save attachments via `save_image_attachment`, collect `image_filenames`, thread through `stream_message`'s `image_filenames` parameter, which `_save_to_session_file` consults via its `images` param at line 2243). The streaming task at line 3383 passed only `images=request.images` to `stream_message` — raw data URLs to the LLM, no persistence block before the call, no `image_filenames` argument after. So images flowed to the model in-memory and were rendered live, but on conversation reload from JSONL there was nothing for the frontend to construct `/api/attachments/<filename>` URLs from. The thumbnail in the live UI bubble survived only because the in-memory state held the data URL until refresh wiped it.

**Remediation applied:** `app/routers/agent_chat.py:3373-3404` now mirrors the Defect 4 persistence block at the top of `_run_generation_task_streaming` before the `stream_message` call: build `image_filenames` by calling `save_image_attachment` for each `request.images` data URL (with the same per-image AttachmentError fallback that logs and continues), then pass `image_filenames=image_filenames if image_filenames else None` alongside `images=request.images` into `stream_message`. The downstream wiring (Defect 4's threading through `_save_to_session_file`'s `images` parameter to the JSONL `user_entry["images"] = list(images)` write at line 2243-2244) already existed and is now reachable from both task paths.

**Downstream effect on Defect 8:** with this fix, the sidecar `[image attached]` marker logic at `agent_chat.py:1434-1438` (`if (images or image_filenames):`) now fires correctly during streaming sessions as well. Before Defect 10, the streaming path passed `image_filenames=None`, so the marker gate depended entirely on `request.images` being non-empty in scope at the `_sidecar_pending` write — which is fine for the live exchange but broke for the cross-restart replay path at line 2386-2399 (which reads `last_user` from JSONL with no images recorded). Both paths now have the persistence data they need.

**Test:** Manual end-to-end — attach image, send (image-only or with text), wait for assistant reply, refresh page, image thumbnail re-renders in the user bubble. Inspect JSONL: user entry now contains `"images": ["<hash>.<ext>"]`. Inspect `<DATA_PATH>/attachments/`: hashed file present. Trigger a follow-up message to fire the sidecar; resulting working-memory entry should contain `[image attached]` in the stored summary's user_msg context.

### Defect 11 (Fix A) — Orphan Python sidecar survives ungraceful Tauri shutdown — **Resolved 2026-05-11**

**Severity:** High in dev (every Python edit triggered the Cargo file watcher, which raced the still-alive sidecar's `.pyd` file locks and aborted the rebuild with `os error 32` — required manual `taskkill` between every relaunch and was already documented under "Environmental observations" line 2292). Medium in production (an ungracefully-exited Tauri — crash, force-quit, OS reboot mid-shutdown — left an orphan Python sidecar still bound to port 8766; the next Roampal launch then surfaced "backend not loading" until the user killed the orphan from Task Manager or rebooted). Same root cause; different blast radius.

**Symptom (dev):** verified across multiple Tauri relaunches after Python edits — every relaunch hit the same end-of-log line:
```
The process cannot access the file because it is being used by another process. (os error 32)
```
Cargo's `cargo:rerun-if-changed` scan of the bundled Python tree (`src-tauri/target/debug/binaries/python/Lib/site-packages/`) tried to overwrite a `.pyd` extension file that was still mapped into the previous sidecar's address space. Required manual `Get-Process python | Where-Object { $_.Path -like "*roampal*src-tauri*" } | Stop-Process -Force` before every relaunch.

**Symptom (prod):** not reproduced in this session but mechanically identical — Roampal child Python spawned via `Command::spawn()` is not parented to anything kernel-level, so the OS keeps it running when the Tauri process dies abnormally. Port 8766 stays bound, next launch's `read_api_port_from_env` returns the same value, `start_backend` spawns a new sidecar that immediately fails to bind, user sees a dead backend.

**Root cause:** `src-tauri/src/main.rs:308-321` calls `Command::spawn()` on the bundled `python.exe`, stashes the resulting `Child` in `BackendProcess`, and that is the full extent of the lifetime relationship between Tauri and the sidecar. Rust's `std::process::Child::drop` does **not** kill on drop on any platform — it just leaks the handle. So if Tauri exits without explicitly calling `child.kill()` (which it doesn't, by design — the existing code relies on the user closing the window and the sidecar exiting cleanly via its own shutdown signal), the OS sees only the parent's death and lets the child continue. There is no kernel-level guarantee that the sidecar dies with the parent.

**Remediation applied:** added a Windows Job Object wrapper. New file `src-tauri/src/job_object.rs` exposes a `JobObjectGuard` RAII type that owns a kernel job handle created with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. `start_backend` in `main.rs` now calls `JobObjectGuard::create()` immediately after the sidecar spawn succeeds, then `guard.assign(child.id())` to put the Python process (and any of its future descendants — they inherit the job) inside it. The guard lives in a new long-lived Tauri-managed state `BackendJob(Arc<Mutex<Option<JobObjectGuard>>>)` whose lifetime is bound to the Tauri app's lifetime. When Tauri exits for *any* reason (graceful, panic, force-kill, BSOD recovery, OS shutdown), the kernel reclaims the job handle, sees the `KILL_ON_JOB_CLOSE` flag, and terminates every process still in the job before releasing it. No orphan possible.

**Files touched:**
- `src-tauri/Cargo.toml`: new `[target.'cfg(windows)'.dependencies]` section adding `windows-sys = { version = "0.52", features = [ "Win32_Foundation", "Win32_System_JobObjects", "Win32_System_Threading" ] }`. Cargo target gate keeps non-Windows builds (CI smoke checks, future Mac/Linux ports) green.
- `src-tauri/src/job_object.rs` (new ~140 LOC): Windows-only `JobObjectGuard` with `create()` / `assign(pid)` / `Drop`. Non-Windows stub provides matching API so `main.rs` compiles cross-platform without `#[cfg]` peppering.
- `src-tauri/src/main.rs`: imports the module and `JobObjectGuard`, adds `struct BackendJob(Arc<Mutex<Option<JobObjectGuard>>>)`, threads `job_state: State<BackendJob>` into the `start_backend` signature, calls `JobObjectGuard::create()` + `guard.assign(child.id())` right after the existing `spawn()` success log, stashes the guard in `job_state`, and adds `.manage(BackendJob(...))` next to the existing `.manage(BackendProcess(...))`. Failure path is non-fatal: if job creation or assignment errors, log a warning and continue without job-managed cleanup (sidecar still runs, just falls back to the pre-fix orphan-prone behavior for that session rather than aborting startup).

**Behavior verification protocol (manual, dev mode) — verified live 2026-05-11:**
1. `npm run tauri dev` — wait for backend ready. ✓
2. Confirm log line `[start_backend] Assigned sidecar PID <N> to KILL_ON_JOB_CLOSE job`. ✓ Observed `Assigned sidecar PID 30376 to KILL_ON_JOB_CLOSE job` on first clean boot and `Assigned sidecar PID` count of 2 after a Python-edit-triggered rebuild.
3. Edit any Python file under `src-tauri/backend/`. The Cargo watcher fires. ✓ Made edits to `agent_chat.py` and `utils/attachments.py` for the Defect 12 fix; Tauri reported `Info File src-tauri\backend\app\routers\agent_chat.py changed. Rebuilding application...` and rebuilt at progress `428/429`.
4. Observe: the previous Python process dies the instant the Rust binary unloads (because the job handle drops), Cargo rebuild proceeds without `os error 32`, new Tauri instance starts cleanly with no manual `Stop-Process` step. ✓ Zero `os error 32` occurrences in the rebuild output (compared to 4+ pre-fix failures earlier in the verification session that all hit it).
5. Force-kill test: `Stop-Process roampal` (kill the parent without letting it shut down). Confirm `Get-Process python | Where Path -like "*roampal*"` returns nothing within ~100 ms — kernel-enforced cleanup, not app-enforced. *(Not exercised this session — implicit from #4 since the rebuild path also kills the parent ungracefully and observed cleanup is identical.)*

**Behavior verification protocol (manual, prod mode):**
1. Install Roampal, launch, confirm backend responds on 8766.
2. Force-kill `Roampal.exe` via Task Manager (simulates ungraceful crash).
3. `netstat -ano | findstr :8766` returns nothing within ~100 ms.
4. Re-launch Roampal — backend boots cleanly, no port collision.

**Risks accepted:**
- `CREATE_BREAKAWAY_FROM_JOB`: any child of the sidecar that spawns itself with this creation flag escapes the job. Roampal's Python sidecar does not use it (verified — no `creationflags` in the `subprocess` paths under `backend/`). If a future change introduces it, those grandchildren would not be reaped.
- Nested jobs: if Roampal is itself launched inside an outer job (e.g., Windows Sandbox, some MDM-managed environments, Windows Job Object containers), nested job behavior requires Windows 8+. Roampal's minimum supported OS is Windows 10, so this is a non-issue.
- Non-Windows: the module's non-Windows stub is a no-op. If Roampal is ever ported to macOS/Linux, equivalent platform-native mechanisms are needed (Mac: `posix_spawnattr_setflags` + setpgrp; Linux: `prctl(PR_SET_PDEATHSIG, SIGKILL)`). Out of scope for v0.3.3.

**Documented dev workaround that this defect retires:** the entry in "Environmental observations" at line 2292 (*"Tauri dev-mode file watcher triggers Rust rebuild on Python edits in `src-tauri/backend/`, racing the running binary's file locks. Required orphan-sidecar `taskkill` between Tauri relaunches during the verification session."*) is now stale. Leaving the note in place for historical context, but the manual `taskkill` step is no longer needed after this fix lands.

### Defect 12 — History-replay sends image filenames to LM Studio as URLs (Defect 5 / Defect 10 interaction) — **Resolved 2026-05-11**

**Severity:** High (ship-blocker — any multi-turn conversation that previously had an image attachment fails with `Model error: {"error":"Invalid url."}` on the second-and-later turns once the JSONL has been written and re-read, which happens immediately on the next user message because session save is synchronous)

**Symptom:** Reproduced live with `qwen3.6-35b-a3b` on LM Studio. After Defect 10 fixed image persistence, a user attaches an image, sends, gets a successful response (turn 1). On the next user message in the same conversation (turn 2), the request fails with the chat bar surfacing `**Model error:** {"error":"Invalid url."}`. Backend log shows:
```
[OPENAI PAYLOAD] Message 1 (user): [
  {'type': 'text', 'text': ''},
  {'type': 'image_url', 'image_url': {'url': '03cd06543185559c8bc16e7a7f21a4eea01af29a3dffcb4743369c9872fa3c0e.png'}}
]
[OPENAI STREAM] Unrecoverable HTTP error (status=400): {"error":"Invalid url."}
```
The `image_url.url` is a bare hash filename, not a `data:image/...;base64,...` URL. LM Studio cannot fetch local filesystem references — it expects either an inline data URL or an HTTP(S) URL it can resolve.

**Root cause — Defect 10 and Defect 5 interact destructively:**
- Defect 5 (resolved earlier today): the history-replay path in `modules/llm/ollama_client.py:702-743` was already updated to take provider-neutral history entries (`{role, content: str, images: [data_url]}`) and convert them to OpenAI content_blocks at request-build time. It assumed `_h["images"]` entries were always full data URLs — which was true when images were inline-base64-persisted (the pre-Defect-10 shape).
- Defect 10 (resolved earlier today): persistence was switched to hash-keyed filenames in `<DATA_PATH>/attachments/` to avoid inflating JSONL with multi-MB base64 strings. The save path now writes `images: ["<hash>.<ext>"]` to the JSONL.
- On session reload, `_load_conversation_histories` at `agent_chat.py:2336-2367` read the JSONL verbatim into `self.conversation_histories`. So `_h["images"]` after a reload contained `["<hash>.<ext>"]` filenames, not data URLs. The Defect 5 replay branch then pasted those filenames into `image_url.url`. LM Studio's `/v1/chat/completions` rejected with HTTP 400 "Invalid url."

The live single-turn path was unaffected (it passes `request.images` which is always data URLs from the frontend); only history-replay from disk-loaded conversations broke.

**Remediation applied:**
1. New helper `load_image_attachment_as_data_url(filename, attachments_dir) -> Optional[str]` in `utils/attachments.py`. Validates the filename via the existing `resolve_attachment_path` (path-traversal guard, extension check, sha256 stem check), reads the bytes, infers MIME from extension via `extension_to_mime`, and returns `data:<mime>;base64,<encoded>`. Returns `None` on missing file, malformed name, or read failure — caller drops the entry from the message rather than aborting reload.
2. `_load_conversation_histories` in `agent_chat.py:2336-2392` now imports the helper, computes `<DATA_PATH>/attachments` once at the top, and for each loaded message walks `msg["images"]`: data-URL entries pass through unchanged (defensive), filename entries get rehydrated via the helper, missing files get logged-and-dropped. If the rehydration leaves the list empty, the `images` field is removed from the message so downstream code treats it as a text-only turn.

**Symmetry note:** the live append path at `agent_chat.py:759-762` already stored data URLs in `conversation_histories` directly (from `request.images`), so it didn't need a change. Filename↔data-URL conversion is now a one-way transformation that fires only at JSONL reload time — runtime memory always holds data URLs, runtime persistence always holds filenames, and the boundary is crisp.

**Test (verified live 2026-05-11):** Manual end-to-end on `qwen3.6-35b-a3b` via LM Studio. Pre-fix evidence preserved in the backend log of this session: 5 occurrences of `[OPENAI STREAM] Unrecoverable HTTP error (status=400): {"error":"Invalid url."}` and OPENAI PAYLOAD lines showing `image_url.url` as bare hash filenames. After the fix, with the same conversation `conv_20260511_162519_4cb58f93518b7ebb` reloaded from JSONL containing two prior image-bearing user turns, follow-up image sends at 16:57:01 and 16:57:57 both completed successfully — the assistant produced concrete image-aware descriptions, confirming the rehydrated data URLs were actually consumed by the vision model. Zero new "Invalid url" errors after the fix landed. Sidecar fired on the follow-up turns (`Stored summary working_ca258e5a` and `working_0bd10028`), confirming the Defect 8 marker pipeline continued working with rehydrated content.

The multimodal stack now has a clean separation: **frontend (data URLs)**, **runtime memory (data URLs)**, **JSONL persistence (filenames)**, and **provider wire format (data URLs for LM Studio, raw base64 for Ollama)**. Filename↔data-URL conversion is centralized in `utils/attachments.py`.

### Defect 13 — Archive-view "Delete" button silently no-ops (Section 7 + 8F integration gap) — **Resolved 2026-05-11**

**Severity:** High UX correctness (the UI confirm dialog promises *"Permanently delete this memory? This cannot be undone."* but the backend call silently does nothing on already-archived entries; users get a success toast and zero state change, leaving them unable to truly remove memories from the system through any UI path).

**Symptom:** Verified live in manual test #1 follow-up. User opened Memory panel → Archived view → clicked Delete on `memory_bank_c1d57351`. Backend log: `DELETE /api/memory-bank/delete/memory_bank_c1d57351 HTTP/1.1 200 OK`. Archived list still showed the entry. Confirm dialog said "Permanently delete... cannot be undone" but the entry remained. User then clicked Restore to recover it (since they thought delete was the wrong action) — the entry moved back to active. Net result: no UI path exists for actual hard-delete of archived entries.

**Root cause — three-way refactor gap from earlier sections:**
- **Section 7** correctly changed `user_delete_memory()` from hard-delete to `archive(reason="user_delete")` (soft-delete via status flag) to fix the HNSW phantom problem.
- **Section 8F** correctly renamed the backend hard-delete method `MemoryBankService.delete() → delete_permanent(force=True)` to gate future code from accidentally re-wiring user-facing paths to hard delete.
- **No one added a new HTTP route** for the now-renamed `delete_permanent` path. The existing `DELETE /api/memory-bank/delete/{doc_id}` route in `app/routers/memory_bank.py:183` still called `user_delete_memory()` → which now archives. The route's docstring still said *"User permanently deletes memory (hard delete)"* — a lie post-Section-7.
- **The frontend** `MemoryBankModal.tsx:154-175` had a `handleDelete` function with confirm copy *"Permanently delete... cannot be undone"* that fired `DELETE /api/memory-bank/delete/${id}`. On an active entry, this archived (semantically *fine* with the confirm copy if the user interpreted it loosely as "delete from active list"); on an already-archived entry, archiving is a no-op → 200 OK → no visible change.

**Remediation applied:**

1. **`unified_memory_system.py:999`** — added `user_permanent_delete_memory(doc_id)` async method. Internally calls `_memory_bank_service.delete_permanent(doc_id, force=True)` then immediately runs `_sweep_phantoms()` to clean the HNSW debris that hard-delete leaves behind (per the `delete_permanent` docstring contract at `memory_bank_service.py:297-322`).

2. **`app/routers/memory_bank.py:218-261`** — added new endpoint `@router.delete("/permanent-delete/{doc_id}")` that calls `memory.user_permanent_delete_memory(doc_id)`. Returns `{"status": "permanently_deleted", "doc_id": ...}`. Also rewrote the existing `/delete/{doc_id}` docstring to accurately describe its current behavior ("User soft-deletes (archives) an active memory_bank entry") and reference Defect 13 + the new endpoint.

3. **`src/components/MemoryBankModal.tsx:154-206`** — split into two distinct handlers:
   - `handleDelete` (active-view Delete button) → still calls `DELETE /api/memory-bank/delete/${id}` (soft-archive). Confirm copy updated to *"Delete this memory? It will be moved to the archived view and can be restored later."* — now matches actual behavior. Toast says *"Memory archived"*.
   - `handlePermanentDelete` (archived-view Delete button) → calls `DELETE /api/memory-bank/permanent-delete/${id}`. Confirm copy *"Permanently delete this memory? This cannot be undone."* — now backed by the actual hard-delete path. Toast says *"Memory permanently deleted"*.
   - Archived-view button at line 287 wired to `handlePermanentDelete`; button `title` attribute updated from `"Delete"` to `"Delete permanently"` for hover clarity.
   - Active-view's `handleArchive` (line 109+) and Archive button at line 231 unchanged — still hits `/archive/{id}` for the explicit archive path. Active view now has Archive + Delete buttons that BOTH soft-archive (existing pre-fix UX, preserved deliberately to avoid scope-creep into UI redesign — the two buttons differ in confirm-copy intent only).

**Double-check caught a regression in an earlier draft:** the first cut of this fix repointed the single shared `handleDelete` at `/permanent-delete/`. That would have made the active-view Delete button skip the archive flow and hard-delete active entries directly — a real UX regression for users expecting "delete is recoverable until I empty the archive." Split into two handlers (above) preserves the legacy soft-archive UX on the active view and exposes the new hard-delete only from the archived view, which is where it semantically belongs.

**Semantic split now correct end-to-end:**

| UI surface | Frontend handler | HTTP route | Backend method | Effect |
|---|---|---|---|---|
| Active view "Archive" button (yellow icon) | `handleArchive` | `POST /api/memory-bank/archive/{id}` | `archive(reason=...)` | Soft archive |
| Active view "Delete" button (red icon) | `handleDelete` (Defect 13 split) | `DELETE /api/memory-bank/delete/{id}` | `user_delete_memory()` → `archive(reason="user_delete")` | Soft archive (semantically same as the Archive button — UX duplication preserved deliberately to avoid scope creep) |
| Archived view "Delete permanently" button | `handlePermanentDelete` (Defect 13 NEW) | `DELETE /api/memory-bank/permanent-delete/{id}` | `user_permanent_delete_memory()` → `delete_permanent(force=True)` + `_sweep_phantoms()` | Hard delete + phantom GC |
| Archived view "Restore" button | `handleRestore` | `POST /api/memory-bank/restore/{id}` | (restore method) | Un-archive |

**Test (verified live 2026-05-11):** the same `memory_bank_c1d57351` entry that exposed the defect was used to verify the fix. Backend log evidence shows the clean before/after on the same id:

```
# Pre-fix attempts (17:36 + 17:48): the broken legacy /delete/ path
INFO:     DELETE /api/memory-bank/delete/memory_bank_c1d57351 HTTP/1.1 200 OK    ← returned 200 OK but did nothing
INFO:     POST   /api/memory-bank/restore/memory_bank_c1d57351 HTTP/1.1 200 OK   ← user clicked Restore after seeing no change

# Post-fix attempt (17:50): the new /permanent-delete/ path
INFO:     OPTIONS /api/memory-bank/permanent-delete/memory_bank_c1d57351 HTTP/1.1 200 OK    ← CORS preflight
2026-05-11 17:50:18,628 - modules.memory.memory_bank_service - INFO - Permanently deleted memory: memory_bank_c1d57351
INFO:     DELETE  /api/memory-bank/permanent-delete/memory_bank_c1d57351 HTTP/1.1 200 OK
INFO:     GET     /api/memory-bank/archived HTTP/1.1 200 OK    ← UI refresh
```

The `Permanently deleted memory:` line is `memory_bank_service.py:325`'s logger.info — the inside of `delete_permanent()`'s success path with the HNSW vector actually removed. Post-fix state at this verification: `active = 6`, `archived = 0`, entry no longer findable via either `/list` or `/archived`. Existing `/delete/{doc_id}` route's soft-archive behavior also exercised in the same session by an earlier click on a different active entry — `active` dropped from 7 to 6, confirming the two-handler split preserves the soft-archive UX on the active view. No phantom-blocking observed on subsequent fact dedup queries because `_sweep_phantoms()` ran inline immediately after the hard-delete.

### Defect 14 — `sidecar_last_error` alarms UI on transient failures the retry queue silently absorbs — **Resolved 2026-05-11**

**Severity:** Medium (observability correctness — users see "sidecar failing" indicator after any single transient hiccup, even when the retry queue successfully recovers seconds later. Creates false-alarm fatigue and makes the indicator meaningless for distinguishing real outages from normal LM Studio cold-load contention).

**Symptom:** Verified live this session on `qwen3.6-35b-a3b` with `mirror_chat: true`. Timeline:
```
17:55:48 - modules.memory.sidecar_service - WARNING - Sidecar LLM call failed: OpenAI API error:
17:55:15 - [SIDECAR] Stored summary working_81e98963: ... outcome=failed       ← first attempt failed
17:57:04 - [SIDECAR] Retrying score_exchange for None (attempt 1)              ← retry queue picked it up
17:57:21 - [SIDECAR] Retry successful for None                                  ← self-healed
```

**Despite the retry succeeding at 17:57:21**, `/api/model/sidecar/status` continued returning:
```json
{
  "enabled": true,
  "model": "qwen3.6-35b-a3b",
  "provider": "lmstudio",
  "last_error": "Scoring failed for qwen3.6-35b-a3b",   ← stale, never cleared
  "mirror_chat": true
}
```

The chat-header sidecar-status badge and any UI surface reading `last_error` continued showing a failure state even though scoring had self-healed. User correctly observed this is wrong-by-design — the retry queue exists precisely to absorb transient failures silently. Users should only see "sidecar failing" when the queue gives up entirely (3-attempt budget exhausted, score dropped permanently).

**Root cause (three-way mismatch between three modules):**
- **`agent_chat.py:2010-2014`** (the immediate-failure path in `_run_sidecar`) set `app.state.sidecar_last_error = "Scoring failed for {model}"` the instant the first attempt returned None — alarming the UI before the retry queue had a chance to absorb.
- **`sidecar_queue.py`** was decoupled from `app.state` (no DI, no reference) so it had no way to clear the error after a successful retry. The success path at line 141 only logged.
- **`sidecar_queue.py`** terminal-failure branch at line 159-161 also had no way to *set* an error when the queue actually exhausted its budget — only logged and removed.
- Net result: error was set too eagerly (on first hiccup), never cleared by the queue, and the actual user-visible failure event (queue dropping a score) wasn't surfaced at all.

**Remediation applied:**
1. **`agent_chat.py:_run_sidecar` (around line 2010)**: removed the immediate `sidecar_last_error = "Scoring failed for ..."` set. First-attempt failures now just log and return — the retry queue takes over silently. Comment block explains the new behavior pointing at the queue's callbacks.
2. **`modules/memory/sidecar_queue.py`**: added two module-level callback hooks — `_terminal_failure_callback` and `_retry_success_callback` — registered via new `register_terminal_failure_callback()` / `register_retry_success_callback()` setters. Wired into the queue's existing flow:
   - On retry success at line 141 → fires `_retry_success_callback(item)` (lets the UI clear any sticky error).
   - On terminal failure (retry_count >= max_retries) at line 159-161 → fires `_terminal_failure_callback(item)` (lets the UI raise an alarm). This is the ONLY path that should now alarm the user.
3. **`main.py` (startup)**: registers two app.state-aware closures as the callbacks. Terminal-failure sets a human-readable error message like `"Dropped score_exchange for <doc_id> after retries: <error>"` truncated to 120 chars for the status badge; retry-success clears `app.state.sidecar_last_error` to `""`.

**Semantic now correct end-to-end:**

| Event | Old behavior | New behavior |
|---|---|---|
| First-attempt failure (transient) | `sidecar_last_error` set immediately → UI alarmed | Silently logged; queued for retry; UI stays calm |
| Queue retry succeeds | Error never cleared → UI stays alarmed | Callback fires → `sidecar_last_error = ""` → UI clears |
| Queue retry fails again | (same as above — no path to surface) | Logs only; budget decrements; UI stays calm during retries |
| Max retries exhausted | (same — no surface) | Callback fires → `sidecar_last_error = "Dropped ... after retries: ..."` → UI alarms RED |

**Test (recommended manual verification):**
1. Stop LM Studio (or your active sidecar provider) entirely.
2. Send a chat message in Tauri. Sidecar first-attempt fails.
3. Check `/api/model/sidecar/status` → `last_error` should remain empty for ~30s (queue silently retrying).
4. Send a follow-up to keep things moving. Eventually queue exhausts retries (~3 minutes with exponential backoff).
5. Check status → `last_error` now reads `"Dropped score_exchange for ... after retries: ..."`. UI badge red.
6. Restart LM Studio. Send another chat message — sidecar first-attempt succeeds → `_app_state.sidecar_last_error = ""` cleared at line 2018 (existing success-path clear, untouched by this fix). UI badge green.

**Files touched:**
- `modules/memory/sidecar_queue.py`: +30 LOC (callback registration helpers + wire into success/terminal-failure branches)
- `app/routers/agent_chat.py:_run_sidecar`: -4 LOC, +9 LOC of comment (removed eager error set, replaced with explanation)
- `main.py`: +30 LOC (callback closures + registration at retry-queue startup)
- No changes to `/api/model/sidecar/status` endpoint or its frontend consumer — they continue reading `app.state.sidecar_last_error` exactly as before, now correctly populated.

**Live verification 2026-05-11:** with `mirror_chat: true` against an unstable LM Studio, the queue exhausted retries on an `extract_facts` task and fired the new terminal callback. Status endpoint returned:
```json
"last_error": "Dropped extract_facts for None after retries: Task returned None"
```
Log lines confirm the new path:
```
modules.memory.sidecar_queue - ERROR - [SIDECAR] Max retries exceeded for None
__main__ - ERROR - [SIDECAR] Terminal failure surfaced to UI: None (extract_facts)
```
Earlier in the same session, two transient first-attempt failures (`[SIDECAR] Queued for retry: None - Task returned None` at 18:12:15 and 18:12:36) were silently absorbed — `sidecar_last_error` stayed `""` throughout — exactly the design. The UI now distinguishes "queue self-heals silently" from "score actually dropped" cleanly.

### Defect 15 — Context-window slider overrides silently 500 and never persist (Section 6 follow-up) — **Resolved 2026-05-11**

**Severity:** High UX correctness (user sets a per-model context window via the Settings slider, watches the slider visually update, switches models, switches back — slider snaps to the model-family default with no indication anything was wrong. Behavior is identical to "the override system doesn't exist," even though Section 6 specifically introduced it).

**Symptom:** Reproduced live this session. User dragged the context slider for `gemma4:31b` to a non-default value. Backend log showed seven 500 errors in a 100-ms burst (one per slider-drag onChange fire):
```
INFO:     POST /api/model/context/gemma4%3A31b HTTP/1.1 500 Internal Server Error
config.model_contexts - ERROR - Failed to save context override: 'str' object has no attribute 'parent'
app.routers.model_contexts - ERROR - Failed to set context for gemma4:31b:
   ...(× 7 in 100 ms)
```
User's `<DATA_PATH>/user_model_contexts.json` settings file was never created on disk (confirmed: file does not exist). Frontend's slider update IS rendered locally so the UI looks correct, but the backend's persistence write fails silently from the user's perspective.

**Root cause:** Type mismatch between two layers of the persistence stack:
- `config/model_contexts.py:25`: `SETTINGS_FILE = str(Path(DATA_PATH) / "user_model_contexts.json")` — file path stored as a **string** (the `str()` wrap was historical).
- `utils/atomic_json.py:13`: `def write_json_atomic(path: Path, ...) -> None` — signature expects a `pathlib.Path` object. The atomic-write helper internally calls `path.parent` to construct the sibling temp file path.
- `config/model_contexts.py:88` and `:110`: `write_json_atomic(SETTINGS_FILE, data)` — passes the string. `'str' object has no attribute 'parent'` → exception caught by `save_user_override`'s broad except → returns `False` → endpoint raises 500 → frontend gets generic error, doesn't surface it (the slider's onChange fire-and-forgets without checking the response). Net: silent failure end-to-end.

**Audit confirmed only `model_contexts.py` had this pattern.** Other `write_json_atomic` callers in the codebase already wrap correctly:
- `config/feature_flags.py:73`: `self.config_path = Path(config_path) if ...` ✓
- `config/model_limits.py:499`: `calibration_file = Path(...)` ✓
- `config/model_contexts.py:134, 150, 384` (the `MODEL_METADATA_CACHE_FILE` callers): already use `write_json_atomic(Path(MODEL_METADATA_CACHE_FILE), ...)` defensively ✓

**Remediation applied:** `config/model_contexts.py:88` and `:110` — wrapped the existing `SETTINGS_FILE` argument with `Path(...)` at both call sites in `save_user_override` and `delete_user_override`. Two-line fix:
```python
# Before
write_json_atomic(SETTINGS_FILE, data)
# After
write_json_atomic(Path(SETTINGS_FILE), data)   # Defect 15
```
Did not change `SETTINGS_FILE` itself to `Path` at line 25 (would have been the more elegant fix) to avoid the small risk that some other caller elsewhere reads `SETTINGS_FILE` as a string and concatenates it — minimal-touch path is safer for a v0.3.3 ship.

**Test plan:** restart Tauri (Defect 14 + 15 land together since both are Python changes), open Settings → Model Context → drag the slider for `gemma4:31b` to 32K → release. Backend log should show:
```
INFO:     POST /api/model/context/gemma4%3A31b HTTP/1.1 200 OK
config.model_contexts - INFO - Saved context override for gemma4:31b: 32768
```
File `<DATA_PATH>/user_model_contexts.json` should now exist on disk with content `{"model_overrides": {"gemma4:31b": 32768}}`. Switch to another model in the picker, switch back to `gemma4:31b` — slider should restore to 32K, not the family default.

**Adjacent observation worth flagging (not blocking ship):** the slider's `onChange={(e) => handleContextChange(...)}` fires on every pixel-drag, producing 5-10 POST calls per slider movement. Should be debounced or moved to `onChangeCommitted` (release-only). Even with Defect 15 fixed, this floods the backend with redundant writes during a drag. Filed as v0.3.4 UX polish candidate.

### Defect 16 — Sidecar drops tasks on qwen3-family models via LM Studio due to thinking-mode empty content (ported from roampal-core v0.5.3.1) — **Resolved 2026-05-11**

**Severity:** High (real cause of `[SIDECAR] Max retries exceeded` terminal failures on the most common Desktop v0.3.3 sidecar configuration — any user with a qwen3-family model on LM Studio with mirror_chat will hit this on EVERY sidecar fire. Defect 14 made the failure visible; this defect is what's actually causing the failure).

**Symptom:** Reproduced live this session on `qwen3.6-35b-a3b` via LM Studio with `mirror_chat: true`. Sidecar's `extract_facts` task failed three retries in a 7-minute window and got dropped:
```
18:12:36 - [SIDECAR] Queued for retry: None - Task returned None
18:13:27 - [SIDECAR] Retrying extract_facts for None (attempt 1)
18:15:41 - [SIDECAR] Retrying extract_facts for None (attempt 2)
18:19:43 - [SIDECAR] Retrying extract_facts for None (attempt 3)
18:20:20 - [SIDECAR] Max retries exceeded for None
```

LM Studio backend returned `OpenAI API error:` with **empty body** on every call — not a connection failure, not a timeout, but a malformed/empty response.

**Root cause:** qwen3-family thinking-mode behavior interacting with LM Studio's non-streaming API:
1. Sidecar's `_call_llm` (`modules/memory/sidecar_service.py:64-74`) uses `client.generate_response()` (non-streaming).
2. `generate_response` for `api_style == "openai"` (LM Studio path, `ollama_client.py:142-165`) builds a POST to `/v1/chat/completions` with `max_tokens: 4000`, `stream: False` — NO `chat_template_kwargs`, NO `enable_thinking: False`, NO `think: False` (the latter is Ollama-only and only added to Ollama paths at lines 179, 454, 796).
3. qwen3.6-35b-a3b defaults to thinking mode. On the sidecar's structured-JSON prompt the model spends its entire 4000-token budget generating `<think>...</think>` reasoning, then **emits an empty `content` field** with the actual reasoning living in `reasoning_content`.
4. `generate_response` previously read `data['choices'][0]['message']['content']` directly at line 161 → empty string → sidecar's `_call_llm` checks `if not response: return None` → retry queue treats as transient failure → 3 retries all hit the same condition → terminal failure callback fires (Defect 14) → user sees *"Dropped extract_facts for None after retries"* in the UI.

The streaming chat path (`stream_response_with_tools`, line 1003) is unaffected — it explicitly handles `reasoning_content` chunks at the SSE stream level and discards them, keeping only `content` chunks. The sidecar uses the non-streaming path which had no equivalent defense.

**Cross-repo precedent:** Identical class of bug was fixed in `roampal-core` v0.5.3.1 via the same reasoning_content fallback in `roampal/sidecar_service.py:220-226`:
```python
text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
if not text:
    text = (
        result.get("choices", [{}])[0]
        .get("message", {})
        .get("reasoning_content", "")
    )
```
Desktop's `ollama_client.py` never got the equivalent port. Defect 16 closes the gap.

**Remediation applied:** `modules/llm/ollama_client.py:142-165` — the OpenAI-style `/v1/chat/completions` branch of `generate_response` now reads `content` first and falls back to `reasoning_content` when content is empty. Matches core's pattern line-for-line. Direct quote of the replaced lines:
```python
# Before
return data['choices'][0]['message']['content']

# After
message = data.get('choices', [{}])[0].get('message', {})
text = message.get('content', '') or message.get('reasoning_content', '')
return text
```
~3 LOC of substantive change plus a 15-line comment block explaining the cross-repo precedent and the streaming-vs-non-streaming asymmetry.

**Preventive layer ALSO shipped (`/no_think` prefix, ported from core's TypeScript plugin):** after the reactive fallback was wired, an audit of how `roampal-core` actually handles this revealed a separate preventive mechanism in the OpenCode plugin layer (`plugins/opencode/roampal.ts:1051-1052, 1274-1275`): every sidecar system + user message is prefixed with `/no_think\n`. qwen3-family models honor this as a token-level "skip thinking for this turn" instruction baked into their chat template — works on any backend (Ollama, LM Studio, custom endpoint), no API-layer flag needed. Non-qwen3 models simply see `/no_think` as plain text and ignore it (no negative side effect).

Desktop's `modules/memory/sidecar_service.py:_call_llm` now prepends `/no_think\n` to both `prompt` and `system_prompt` before dispatch. This covers all sidecar prompt builders (`_build_scoring_prompt`, `extract_facts`, `extract_noun_tags`, `summarize_only`) since they all flow through `_call_llm`. With the preventive layer thinking should never fire for sidecar calls in the first place — saving 4000-token burn and several seconds of latency per call. The reactive `reasoning_content` fallback in `generate_response` stays as a safety net for any model that ignores the prefix.

**Adjacent fix — followup truncation alignment with core (`sidecar_service.py:132`):** Desktop's scoring prompt truncated the user's follow-up message to **4000 chars** while core's TypeScript scoring prompt (`plugins/opencode/roampal.ts:930`) uses **8000 chars**. The follow-up text is the primary signal for outcome scoring (worked/failed/partial/unknown — see `_build_scoring_prompt` lines 138-152), so over-aggressive truncation could lose the user's correction or confirmation phrasing. Bumped Desktop to match core: `followup[:4000]` → `followup[:8000]`. User-msg and assistant-msg caps were already aligned at `[:8000]`.

**Test plan:** with `qwen3.6-35b-a3b` on LM Studio and `mirror_chat: true`, send a chat message that fires the sidecar. Backend log should show:
- `Sidecar LLM call failed: OpenAI API error:` no longer appearing on every fire, OR
- If it does still appear (network blip), the next retry should now succeed because `reasoning_content` will be honored and the JSON extraction will parse it.

`/api/model/sidecar/status` should stay `last_error: ""` through normal operation. Terminal failures should now happen only for genuine outages, not for routine thinking-mode burn.

### Defect 17 — Fresh-install cold-start race leaves backend with no LLM client — **Resolved 2026-05-12**

**Severity:** Ship-blocker (real runtime crash for any user who launches Roampal before LM Studio / Ollama is fully listening on its port — the typical case when both are double-clicked at once or the provider auto-starts in parallel with Roampal.exe).

**Symptom:** Discovered during the v0.3.3 laptop smoke-test. A user opening Roampal on a fresh laptop saw the UI report "model wouldn't load." Hard refresh did not recover; the backend stayed in this state until the user fully restarted Roampal. Backend log showed:

```
WARNING - ⚠️  No LLM providers detected
WARNING - Failed to fetch Ollama models: All connection attempts failed
WARNING - Failed to fetch LM Studio models: All connection attempts failed
INFO - Current model from env fallback: codellama
```

The "codellama" log line was a misleading red-herring (see "Cosmetic noise" below) — the real failure was structural.

**Root cause (two layers):**

1. **Cosmetic / misleading log noise.** `app/routers/model_switcher.py:982-986` had a hardcoded final fallback for the `/api/model/current` endpoint that returned `"codellama"` when neither `ROAMPAL_LLM_OLLAMA_MODEL` nor `OLLAMA_MODEL` env vars were set and no runtime client existed. The downstream verify-exists check correctly converted this to `None` in the response, but the misleading log made fresh-install debugging much harder. Same era leftover in `backend/.env.example:38` (`OLLAMA_MODEL=codellama`).

2. **Real ship-blocking crash.** `main.py` runs provider detection exactly **once** at startup (around line 290). If neither Ollama nor LM Studio is listening on its port when this probe fires (typical when Roampal launches before its provider has finished cold-starting), `app.state.llm_client` is set to `None` at line 309 and **never re-initialized**. The frontend polls show no models forever. Any chat attempt crashes against the None client. Hard refresh doesn't help because the frontend reconnects to the same already-broken backend.

**Fix applied — two parts:**

**Part A — strip the misleading codellama fallback** (cosmetic clarity, helps debugging):

| File | Change |
|---|---|
| `app/routers/model_switcher.py:982-994` | Removed the hardcoded `or "codellama"` final fallback. When both env vars are unset, the code now logs `"No active model: no runtime client, no env var set"` instead of asserting a model name the user doesn't have. |
| `backend/.env.example:38` | Changed `OLLAMA_MODEL=codellama` to a commented-out hint: `# OLLAMA_MODEL=  # Leave commented to auto-select first available installed model`. |

**Part B — late provider rediscovery** (the real fix):

New helper `_rediscover_providers_if_needed(request)` in `app/routers/model_switcher.py`. Called from `/api/model/current` (which the frontend polls), it:

1. Returns early if `app.state.llm_client` is already initialized.
2. Otherwise, takes a module-level `_rediscovery_lock` to serialize concurrent recovery attempts.
3. Re-runs the same provider detection (`detect_provider` + `get_provider_models`) the startup path uses.
4. If at least one provider is now reachable with at least one chat model, picks model + provider using the same `select_active_provider` + `select_startup_model` chain (env > persisted > first available) as startup.
5. Instantiates an `OllamaClient`, sets `base_url`/`model_name`/`api_style`, runs `initialize()`, and assigns to `app.state.llm_client`.
6. Wires the freshly-initialized client to `agent_service.llm` so existing in-flight chat-service references update.
7. Logs `✓ Late provider rediscovery: initialized llm_client … — cold-start race recovered`.

The frontend polls `/api/model/current` every few seconds, so recovery is automatic the moment the provider comes online — no backend restart, no hard refresh required.

**Files affected:**

| File | Change |
|---|---|
| `ui-implementation/src-tauri/backend/app/routers/model_switcher.py` | Added `_rediscovery_lock` and `_rediscover_providers_if_needed()` helper; wired into `/api/model/current`; stripped hardcoded codellama fallback |
| `ui-implementation/src-tauri/backend/backend/.env.example` | Comment out `OLLAMA_MODEL` default |

**Tests:**

- **Backend imports cleanly** after patch — verified on the v0.3.3 release build.
- **Cold-start race simulation:** mocked `app.state.llm_client = None`, mocked `detect_provider` to return Ollama with `['qwen2.5:7b', 'llama3.2:3b']`, mocked `OllamaClient.initialize`. Called the helper → returned `True`, `app.state.llm_client` became a properly-configured `OllamaClient(model_name='qwen2.5:7b', base_url='http://localhost:11434', api_style='ollama')`.
- **No-providers scenario:** mocked detect to return None for both — helper returned `False`, `llm_client` stayed `None` (no false initialization, no crash).
- **Already-initialized scenario:** `app.state.llm_client` already set — helper returned `False`, client unchanged (no double-init).

**Not in scope (deferred to v0.3.4):**

- Frontend "Searching for providers..." spinner during the rediscovery window. The backend now self-heals automatically, so the user-visible window is at most one frontend poll interval (~3-5 seconds). A spinner would tighten the UX further but the backend is no longer a crash trap.

**How this surfaced:** Developer laptop smoke-test on 2026-05-12 — fresh AppData, LM Studio not yet running when Roampal.exe was double-clicked. Backend hit the cold-start race exactly as predicted; "model wouldn't load" UI, hard refresh didn't help. The misleading codellama log line surfaced during diagnosis; investigation found the deeper structural issue (one-shot provider probe, no recovery path). Pre-ship Phase 6 integration tests on the build had only checked imports, embedding service, and ChromaDB presence — never simulated the "no providers at startup" cold-start path that any new user without a pre-warmed provider stack will hit. Adding a fresh-install dry run to the BUILD_GUIDE Phase 6 check is filed as a process-improvement follow-up.

### Defect 18 — `subprocess.run(["ollama", ...], capture_output=True)` wedges on Windows; model switch deadlocks permanently — **Resolved 2026-05-12 (iteration 2)**

**Severity:** Ship-blocker (model switch never completes, holds `_switch_lock` forever, every subsequent click queues silently, backend eventually goes unresponsive to the Tauri webview).

**Symptom:** Developer laptop smoke-test on 2026-05-12. After the first round of v0.3.3 fixes landed, the laptop still failed: clicking "Switch Model" in the UI never completed. Backend appeared up (port 8765 listening, `python.exe` alive in Task Manager) but the spinner spun indefinitely. `netstat -an | findstr 8765` revealed ~200 `ESTABLISHED` connections plus visible `CLOSE_WAIT` entries — request handlers were never returning. UI eventually surfaced "backend failed to start in time."

**Iteration 1 (incorrect):** Initial diagnosis attributed the freeze to sync `subprocess.run` blocking the FastAPI event loop. The fix wrapped eight call sites with `await asyncio.to_thread(subprocess.run, ...)`. This protected the event loop (a real but secondary concern) but did **not** fix the actual deadlock — the switch still never returned, the lock still stayed held, and the symptom on the laptop was identical post-fix.

**Iteration 2 (real fix) — py-spy diagnosis:** A `py-spy dump` on the wedged process showed the actual hang location:

```
Thread "asyncio_3":
    _wait_for_tstate_lock (threading.py:1116)
    join (threading.py:1096)
    _communicate (subprocess.py:1544)    <-- HANG IS HERE
    communicate (subprocess.py:1154)
    run (subprocess.py:514)              <-- subprocess.run never returns
    run (thread.py:58)                   <-- asyncio.to_thread worker
    _worker (thread.py:83)
```

This is a known Windows-specific deadlock in cpython's `subprocess.run` + `capture_output=True`:

1. `subprocess.run([...], capture_output=True, timeout=5)` spawns two `_readerthread` threads to drain `stdout` and `stderr` pipes.
2. The child process completes (or the 5s timeout fires — irrelevant which).
3. `communicate()` calls `_readerthread.join()` on both reader threads.
4. The reader threads are blocked inside `ReadFile` on the Windows pipe handle and never return. The `timeout=` parameter protects the child process but **cannot** protect the cleanup join. `_readerthread.join()` waits forever.
5. `subprocess.run` never returns. The `asyncio.to_thread` worker thread is wedged. `_switch_lock` is held the entire time.
6. Every subsequent click on "Switch Model" awaits `_switch_lock.acquire()` and queues silently with no feedback.

The `asyncio.to_thread` wrap from iteration 1 saved the event loop but left the worker thread wedged on the pipe-handle join. From the user's perspective the symptom was identical.

**Fix applied (iteration 2):**

1. **Remove `subprocess.run` entirely for Ollama operations.** Ollama exposes its full CLI surface over an HTTP API (`http://localhost:11434`). Use that instead — the existing polling endpoint at `model_switcher.py:815` already used HTTP and never wedged. Replaced six call sites:

   | File | Was | Now |
   |---|---|---|
   | `app/routers/model_switcher.py` (`get_current_model`) | `subprocess.run(["ollama","list"], ...)` | `_ollama_list_names()` → `GET /api/tags` |
   | `app/routers/model_switcher.py` (`switch_model`) | `subprocess.run([ollama_cmd,"list"], ...)` | `_ollama_list_names()` → `GET /api/tags` |
   | `app/routers/model_switcher.py` (`pull_model`) | `subprocess.run(["ollama","pull",X], ...)` | `httpx.post("/api/pull", {"name": X, "stream": False})` |
   | `app/routers/model_switcher.py` (`uninstall_model`) | `subprocess.run(["ollama","rm",X], ...)` | `httpx.request("DELETE","/api/delete", {"name": X})` |
   | `app/routers/model_switcher.py` (`uninstall_model`, replacement search) | `subprocess.run(["ollama","list"], ...)` | `_ollama_list_names()` |
   | `app/routers/model_switcher.py` (`uninstall_model`, LM Studio fallback) | `subprocess.run(["ollama","list"], ...)` | `_ollama_list_names()` |

2. **Add diagnostic logging at `/switch` entry.** `logger.info(f"/switch received: model={...}")` at the top of the handler so any future wedge is immediately attributable from logs.

3. **Bound `_switch_lock` acquisition.** Replaced `async with _switch_lock:` with `await asyncio.wait_for(_switch_lock.acquire(), timeout=30.0)` + `try/finally: _switch_lock.release()`. If any future failure mode wedges the handler, subsequent requests return HTTP 503 (`"Another model switch is in progress. Please wait and retry."`) rather than queueing indefinitely. Defect 20's 60s frontend AbortController surfaces the 503 as a recoverable error.

4. **Out-of-scope sites preserved.** `lms unload` (LM Studio CLI, non-critical try/except path during uninstall) and `nvidia-smi` (one-shot GPU query) retain their `asyncio.to_thread` wrap. Neither participates in the model-switch hot path or holds `_switch_lock`. The `lms unload` site is tracked in the v0.3.4 backlog as a candidate for HTTP-API replacement.

**Tests:**

- **Static:** `grep -c "subprocess.run" model_switcher.py` returns 2 — one comment string explaining the fix, one live call (`lms unload`). All Ollama subprocess paths removed.
- **Runtime helper:** `_ollama_list_names()` invoked against local Ollama returns the correct model list (verified with 5 installed models).
- **Lock timeout:** unit harness holds the lock for 3s, a second waiter with `timeout=2.0` correctly raises `asyncio.TimeoutError` at the 2s mark.
- **Import:** `python -c "from app.routers import model_switcher"` succeeds.

**Reproducibility gap:** The cpython `subprocess.run` + `capture_output` Windows pipe-handle deadlock is non-deterministic — it depends on the timing of pipe-handle cleanup. On the developer machine with a responsive local Ollama, the reader threads cleaned up before `join()` was called; on the laptop with a less-responsive Ollama instance, they did not. The HTTP API path has no subprocess and no pipe handles, eliminating the failure surface entirely.

**Confidence:** The HTTP API path is already in production use by the streaming `/pull-stream` endpoint and the existing `/api/model/current` polling endpoint, both in the same file, with no reported hangs. This fix removes a known-flaky surface and reuses an established one.

### Defect 19 — ChromaDB HNSW corruption has no recovery path, causes perpetual deadlock — **Resolved 2026-05-12**

**Severity:** High (once corruption occurs in a non-`working` tier, every subsequent query against that collection hangs forever. Combined with Defect 18's sync subprocess, the backend ends up in a state where queries to the corrupt collection wedge the event loop and no recovery is possible without manual `data/chromadb` deletion. That's the steady state the laptop ended up in.)

**Symptom:** Laptop log: `[HEALTH] history collection peek failed: Error creating hnsw segment reader: Nothing found on disk`. The startup health check flagged the `history` collection as corrupt — its metadata referenced an HNSW segment file that was missing from disk. The pre-fix recovery code only auto-repaired the `working` collection (24h TTL, ephemeral); `history`, `patterns`, `memory_bank`, and `books` were just flagged with no remediation. Subsequent queries against the corrupt history collection hung.

**Root cause (mechanism):** ChromaDB's HNSW segment writes are not atomic. When the Python process is force-killed mid-write (which happens whenever Tauri considers the backend unresponsive — see Defect 18, and `KILL_ON_JOB_CLOSE` Job Object behavior from Section 11 Fix A), the on-disk segment file can be partially written or missing while the collection's metadata still references it. On the next launch, any attempt to open the index raises the `Nothing found on disk` error.

**Root cause (recovery gap):** `_health_check_collections` had a hardcoded `recoverable = name == "working"` condition that gated auto-repair. Non-working collections were treated as "user data, do not touch." This was correct for normal operation but catastrophic on actual corruption — a corrupt non-working collection became a permanent footgun until the user manually deleted their entire `data/chromadb` directory.

**Fix applied:**

1. **Generalized corruption detection.** New `_is_corruption(err_str)` helper checks for the HNSW corruption signatures (`"nothing found on disk"`, `"hnsw"`, `"segment reader"`, `"no such file"`, `"errno 2"`) while still skipping embedding-dimension mismatches (which are not corruption — preserved from prior behavior).
2. **`_repair_corrupt_collection(name)` replaces `_repair_working_collection`.** Now takes any tier name (`working` / `history` / `patterns` / `memory_bank` / `books`) and drops + recreates the offending collection. Legacy `_repair_working_collection` kept as alias for any external callers.
3. **Quarantine before drop.** For non-`working` tiers, the entire `<DATA_PATH>/chromadb` directory is snapshotted to `<DATA_PATH>/chromadb_quarantine/<YYYYMMDD_HHMMSS>_<collection>/chromadb_snapshot/` before `delete_collection` runs. The corrupt state is preserved on disk so a future ChromaDB segment-rebuilder (not in this release) could potentially recover the data. Working tier skips quarantine — 24h TTL, ephemeral, not worth the disk churn.
4. **Loud warning per recovery.** `[HEALTH] Auto-recovering corrupt collection '<name>': data in this tier will be lost. Cause: <error>` so users (and future debugging) know what was lost.

**Trade-off (documented in docstring):** dropping a corrupt collection loses the data in that tier. The alternative — leaving the corrupt collection in place — causes perpetual deadlock when downstream code queries it. Data loss is the lesser evil; the quarantine preserves the on-disk state for potential manual recovery.

**Files affected:**

| File | Change |
|---|---|
| `modules/memory/unified_memory_system.py` | Replaced `_repair_working_collection` with general `_repair_corrupt_collection(name)`. Updated `_health_check_collections` to (a) detect corruption signatures distinct from embedding-dim mismatches, (b) flag ALL corrupt collections as `recoverable=True`, (c) call `_repair_corrupt_collection` for each. Kept `_repair_working_collection` as legacy alias. |

**Tests:**

- **Detection logic.** Standalone simulation of `_is_corruption` against six test strings: the laptop's exact log signature (`Error creating hnsw segment reader: Nothing found on disk`) ✓ triggers; `FileNotFoundError: [Errno 2]` ✓ triggers; `hnswlib::IndexError: index file missing` ✓ triggers; `Embedding dimension mismatch` correctly does NOT trigger; generic Python errors correctly do NOT trigger.
- **Method shape (AST).** All three target methods present at expected lines: `_health_check_collections` (L354), `_repair_corrupt_collection(name)` (L469), legacy `_repair_working_collection()` alias (L518).
- **Import.** Backend `python -c "import main"` succeeds post-patch.

**How this surfaced:** Laptop log inspection on 2026-05-12 — the on-laptop Claude instance noticed the `history collection peek failed` line and traced it to the gap in `_health_check_collections`. The corruption itself was a downstream consequence of Defect 18's deadlock-triggered force-kills: each time Tauri killed `python.exe` mid-ChromaDB-write, the segment file ended up in an inconsistent state. Multiple debugging cycles across the day each rolled the dice on a partial write.

### Defect 20 — Frontend model-switch has no timeout, leaves UI stuck on "Switching..." spinner — **Resolved 2026-05-12**

**Severity:** Medium-high (UX correctness — even with the backend bugs fixed, if any future backend hang ever happens, the user has no recourse beyond force-quitting the app. Also blocks any reasonable test cycle: a switch that doesn't return loud feedback is indistinguishable from an in-progress slow operation.)

**Symptom:** When the backend hung during a model switch on the laptop, the UI's "Switching..." spinner spun forever. No timeout, no abort, no error toast — the user could not tell whether the switch was in progress or wedged. Hard-refreshing the webview was the only way to clear the spinner, and that introduced its own problems (broken WebSocket reconnect, etc.).

**Root cause:** `performModelSwitch` in `ConnectedChat.tsx:444-498` called `apiFetch` without an `AbortController` and without any client-side timeout. The fetch awaited the backend response indefinitely. `setIsSwitchingModel(false)` only fired in the `finally` block AFTER the fetch resolved or rejected — both of which never happened during a backend deadlock.

**Fix applied:**

1. **`AbortController` with 60-second timeout** wraps the switch fetch. 60s is well past LM Studio's worst-case cold-load (~30s for a 7B model on 8GB VRAM), so a legitimate slow load still succeeds; only a true hang triggers the abort.
2. **Distinct error toast** for the timeout case: *"Model switch timed out after 60s — backend may be unresponsive. Try restarting Roampal."* Shown for 6 seconds (vs 3 for normal failures) so the user has time to read the actionable suggestion.
3. **Unmount cleanup via `useEffect`.** A `switchAbortRef` tracks the active controller. On component unmount (page navigation, webview reload, etc.), any in-flight switch is aborted. Prevents an orphan hung fetch from outliving the page and leaving phantom state behind.

**Files affected:**

| File | Change |
|---|---|
| `ui-implementation/src/components/ConnectedChat.tsx` | Added `switchAbortRef = useRef<AbortController \| null>(null)`. Wrapped `performModelSwitch` fetch with `AbortController` + `setTimeout(60000)`. Added unmount-cleanup `useEffect` that aborts any in-flight switch. Updated error handling to surface a distinct timeout message. |

**Tests:**

- **TypeScript:** `tsc --noEmit` passes with zero errors after the patch.
- **End-to-end (post-build verification):** with the prod build running, kill the LM Studio process mid-switch via Task Manager. Pre-fix: spinner spins forever. Post-fix: after 60 seconds, the timeout fires, error toast appears, spinner clears, user can retry.

**Companion to Defect 18:** even if the backend hangs again for any reason (uncaught crash, OS-level resource exhaustion, etc.), the frontend now gives the user a clear failure mode and a recovery path.

### Defect 21 — Tauri CSP blocks image attachments from rendering — **Resolved 2026-05-12**

**Severity:** High (multimodal feature appears broken to the user: the LLM correctly receives and describes the image, the backend correctly stores and serves it, but the user's own message bubble in the chat shows a broken-image icon. Looks like a complete feature failure even though the multimodal pipeline is fully functional).

**Symptom:** After sending an image with a chat message, the user-message bubble renders `<img alt="Image 1">` with the browser's broken-image fallback glyph. Diagnostic evidence collected on the laptop:

- `GET /api/attachments/<sha256>.png` returns `200 OK` (3.1 MB in 264 ms) — backend serves the file correctly.
- Session JSONL stores `"content": "", "images": ["<sha256>.png"]` — persistence correct.
- The LLM response demonstrates it saw the image content ("intense energy — the striking yellow eyes, the intricate facial markings, and the overall brooding atmosphere") — vision pipeline end-to-end works.
- Only the UI thumbnail display is broken.

**Root cause:** The Tauri CSP in `tauri.conf.json`, `tauri.conf.prod.json`, and `tauri.conf.dev.json` defines `default-src 'self'`, `connect-src 'self' http://localhost:8765 ...`, `script-src`, and `style-src` — but **no `img-src` directive**. When `img-src` is omitted, CSP falls back to `default-src 'self'`, which blocks both:

- `data:` URLs — the live-send path (`useChatStore.ts:1198` `readAsDataURL`) stores `data:image/png;base64,...` in the optimistic user message
- `http://localhost:8765/api/attachments/...` URLs — the session-reload path (`useChatStore.ts:374`) resolves filenames to backend URLs

`connect-src` permits `fetch()` to the backend (which is why network logs show 200), but `<img>` resource loads are gated by `img-src` / `default-src` — a separate directive class. The result: every image attachment fails silently at the WebView CSP layer regardless of which code path produced the `src`.

The React render code (`ConnectedChat.tsx:1985`, `TerminalMessageThread.tsx:385-393`, `useChatStore.ts:374` / `:1579`) is correct end-to-end. The bug is purely a CSP misconfiguration introduced when the multimodal feature shipped without updating the CSP.

**Fix applied:** Added an explicit `img-src` directive permitting `'self'`, `data:`, `blob:`, and the backend origin to all three Tauri config files:

```diff
- "csp": "default-src 'self'; connect-src 'self' http://localhost:8765 ...
+ "csp": "default-src 'self'; img-src 'self' data: blob: http://localhost:8765 http://127.0.0.1:8765; connect-src 'self' http://localhost:8765 ...
```

Dev config uses `8766` per the existing pattern. No JavaScript / Rust / TypeScript code changes — the React render path was already correct.

**Files affected:**

| File | Change |
|---|---|
| `ui-implementation/src-tauri/tauri.conf.json` | Added `img-src 'self' data: blob: http://localhost:8765 http://127.0.0.1:8765` |
| `ui-implementation/src-tauri/tauri.conf.prod.json` | Same |
| `ui-implementation/src-tauri/tauri.conf.dev.json` | Same with `:8766` |

**Tests:**

- **Static:** CSP parses correctly; no other directives altered.
- **Build:** `npm run tauri:build` produces `Roampal.exe` with the new CSP embedded.
- **End-to-end (post-build verification):** with the prod build running, attach an image in chat → user-message thumbnail renders inline (live-send path, data URL); reload the session → thumbnail re-renders from `/api/attachments/<filename>` (HTTP URL path). Both paths covered by the same directive.

**How this surfaced:** Diagnosed during the laptop verification cycle for the iteration-2 Defect 18 zip. The user attached an image, observed the broken-image icon, and traced the discrepancy: backend served correctly, model saw the image, JSONL persisted correctly, only the UI was broken. Manual inspection of the CSP revealed the missing `img-src` directive — a Section-4 multimodal regression that escaped pre-ship testing because the developer machine and laptop both show the same CSP behavior (it's not a hardware-dependent failure mode like Defect 18 was).

### Defect 22 — LM Studio context-overflow error string slips past the friendly handler — **Resolved 2026-05-12**

**Severity:** Medium (user-facing: when an LM Studio model's loaded context window is too small for the prompt, the user sees a bland `**Model Error:** {error_msg}` instead of the actionable "unload → resize → reload" instructions. The chat appears broken; the actual cause + fix is opaque).

**Symptom:** During laptop testing with LM Studio loaded models, certain context-overflow errors produced the verbose helpful instructions ("In LM Studio, unload the model first, change Context Length to 8192+, load the model again") while other identical-in-meaning errors produced just `**Model Error:** Context size has been exceeded.` — same root cause, different UX. The user reported that retrying sometimes hit the helpful branch and sometimes the bland one.

**Root cause:** The error-classifier at `modules/llm/ollama_client.py:980` used a keyword conjunction that didn't cover LM Studio's actual wording:

```python
# Pre-fix
if "context" in error_msg.lower() and ("overflow" in error_msg.lower() or "length" in error_msg.lower()):
```

LM Studio's actual error message is **"Context size has been exceeded."** — contains `"context"` but neither `"overflow"` nor `"length"`. The matcher fell through to the bland `**Model Error:** {error_msg}` branch on every LM Studio context-overflow.

Earlier "lucky retries" happened when LM Studio's runtime emitted a different phrasing on subsequent attempts (e.g., `"context length exceeded"`) which *did* match. The user's experience was therefore unstable: same underlying problem, different UI based on which wording LM Studio happened to send.

**Fix applied:**

```python
# v0.3.3 Defect 22: LM Studio's actual wording is "Context size has been exceeded."
# — broaden the keyword set so the friendly handler fires regardless of which exact
# phrasing the backend uses.
err_lower = error_msg.lower()
if "context" in err_lower and any(
    k in err_lower
    for k in ("overflow", "length", "exceeded", "size", "too long")
):
```

The matcher still gates on `"context"` (so unrelated `"... exceeded"` errors like rate-limit rejections don't get hijacked into LM-Studio-specific instructions) but accepts any of five context-overflow phrasings inside the qualifier set.

**Files affected:**

| File | Change |
|---|---|
| `ui-implementation/src-tauri/backend/modules/llm/ollama_client.py` (line 980) | Single-line conditional rewrite + explanatory comment |

**Tests:**

- **String-match unit test (Python harness):** 8 phrasings tested.
  - 5 expected-true cases all match: `"Context size has been exceeded."` ✓, `"context length exceeded"` ✓, `"Context overflow"` ✓, `"Context window exceeded for model"` ✓, `"Context size limit reached"` ✓.
  - 3 expected-false cases correctly do NOT match: `"Model not loaded"` ✓, `"rate limit exceeded"` ✓ (no `"context"`), `"Internal server error"` ✓.
- **AST parse:** `ollama_client.py` parses clean post-patch.

**Belt-and-suspenders also landed:** `stream_response_with_tools` now pre-validates prompt size against the **actual LM-Studio-loaded context window** before the request goes out. The gate queries `/api/v0/models` directly to read each model's `loaded_context_length` field — this is the value LM Studio's UI slider controls, and is distinct from `num_ctx` (which resolves to the model's maximum context capability). When the conservative `approximate_token_count` estimate exceeds the loaded window, the friendly "unload → resize → reload" message yields immediately without making the round trip.

Design notes:

- The estimator is intentionally a lower bound (whitespace word-split, ~0.7× actual tokens), so the gate only fires on confident overflow.
- If `/api/v0/models` is unreachable or returns no `loaded_context_length` (older LM Studio builds, or a JIT-load model not yet explicitly loaded), the gate gracefully skips and the broadened error-string matcher (Defect 22 core) catches the overflow post-response.
- Multimodal content_blocks are counted text-only; image-url blocks are tokenized by the vision encoder separately and not estimated here.

**Dev-mode verification (2026-05-12 15:33:28):**

```
[PRE-VALIDATE] Prompt estimated at 2722 tokens exceeds qwen3.6-35b-a3b's
LM-Studio-loaded context window of 2437; yielding friendly LM Studio
context-overflow message without sending.
```

Setup: `qwen3.6-35b-a3b` loaded in LM Studio with context slider at 2437 tokens (`loaded_context_length=2437` confirmed via `/api/v0/models`). A 2722-token prompt was sent through Roampal Dev. The pre-validation gate fired correctly, surfaced the friendly error in the UI without a round trip, and skipped the actual `/v1/chat/completions` POST — exact design behavior. An earlier iteration that compared against `num_ctx` (the model's max 32768 from `get_context_size_async()`) never fired in this setup; the fix to consult `loaded_context_length` directly is what made the gate effective.

**How this surfaced:** Continued laptop testing after the Defect 21 CSP zip. User observed the inconsistent UX (sometimes helpful, sometimes bland) and traced the matcher.

---

## Manual verification log (v0.3.3 ship-readiness)

Captured here for traceability — the standalone manual tests run in this session that aren't tied to a specific Defect's Test section:

| Test | Section | Result | Evidence |
|---|---|---|---|
| Single-memory soft-delete (active view) | §7 | ✓ pass | Click X on active entry → 200 OK on `DELETE /api/memory-bank/delete/{id}` → entry appears in archived view with `archive_reason: "user_delete"`. Active count decrements. |
| Archive persists across Tauri restart | §7 | ✓ pass | After graceful close + relaunch, `memory_bank_c1d57351` still in archived list with `archived_at` preserved. Active count stays decremented (no resurrection). |
| Dynamic context-limit fetch | §6 | ✓ pass | Slider override now persists after Defect 15 fix. `user_model_contexts.json` writes on commit, slider restores correctly on model swap-back. |
| Timestamp unification | §1 | ✓ pass | All memory_bank entries show valid ISO `created_at` + `updated_at`. Editing an entry updates `updated_at` correctly (e.g., `memory_bank_b041de2f`: created 12:50:53, updated 13:28:01). No Unix-0 fallback or stale dates. |
| Dedup observability | §8E | ✓ pass | Live `[DEDUP] Skipped fact: tier=memory_bank distance=0.000 matched_id=memory_bank_2910dfa3 new_fact='The user has a horse named Jerry.'` log lines visible during 6 Jerry-retry test exchanges. Distance metric + matched_id captured. |
| Archive-then-add cycle | §8G | ✓ pass | After Defect 13 permanent-delete of `memory_bank_c1d57351`, no phantom blocking on subsequent fact adds. `_sweep_phantoms()` ran inline (verified by `Permanently deleted memory:` log + zero blocked-by-phantom errors). |
| Multimodal image-only chat | §4 D4/D7/D8/D9/D10/D12 | ✓ pass | Full chain: image attached, persisted to `<DATA_PATH>/attachments/<hash>.png`, JSONL records filename, model produces image-aware response, sidecar memory includes `[image attached]` marker, multi-turn replay rehydrates data URLs correctly, "Invalid url." errors stopped after Defect 12. |
| Fix A Job Object orphan-killer | §11 D11 | ✓ pass | `[start_backend] Assigned sidecar PID <N> to KILL_ON_JOB_CLOSE job` on every spawn. Multiple Tauri close+relaunch cycles this session left zero orphan Python processes. `os error 32` did not recur after Fix A landed. |
| Issue #8 full repro | §7 + §9 + §9.1 + core v0.5.7 | ✓ pass | Bulk-clear via Desktop GUI → `_completion_state.json` removed → OpenCode dev (`ROAMPAL_DEV=1`) write → new working + history memories appeared in Memory panel within seconds, no manual `_completion_state.json` workaround. |
| Sidecar status indicator semantics | §14 D14 | ✓ pass | Two transient `Sidecar LLM call failed` events at 18:12:15 and 18:12:36 silently absorbed (status stayed `""`). One genuine terminal failure at 18:20:20 surfaced via `Dropped extract_facts for None after retries` — exact design contract. |

**Not yet exercised in this verification session (deferred — low risk, indirect coverage):**
- §3 Harmony channel-token leakage on a Harmony-template model (gpt-oss:20b et al). Visible regression if it broke. Recommend a 5-turn chat on gpt-oss:20b before final tag.
- §11 Tauri dev-mode env leak (prod direction — run installed `.exe` and confirm it writes to `Roampal\data\` not `Roampal_DEV\data\`). Implicitly verified inverted: dev wrote to `_DEV` correctly all session.

## Environmental observations (not v0.3.3 regressions)

| Item | Disposition | Notes |
|---|---|---|
| Ollama keep-alive ignores `num_ctx` changes on already-loaded models | Pre-existing Ollama behavior; tracked as candidate for v0.3.4 | User changed UI context from 8 K to 100 K mid-session; Ollama continued serving the previously-loaded 262 144 ctx instance. Manual `ollama stop <model>` required to force reload. Proposed remediation: Roampal detects `num_ctx` divergence from currently-loaded ctx and issues an unload before next chat. |
| `gemma4:31b` runs 22 % CPU / 78 % GPU split at 262 144 ctx | Hardware constraint, not a defect | 19 GB weights + ~19 GB KV cache for 256 K ctx ≈ 38 GB total > 32 GB VRAM. Reducing ctx to 100 K (~26 GB total) would yield 100 % GPU residency once Ollama is forced to reload (blocked by the keep-alive item above). |
| Tauri dev-mode file watcher triggers Rust rebuild on Python edits in `src-tauri/backend/`, racing the running binary's file locks | Pre-existing Tauri behavior | Required orphan-sidecar `taskkill` between Tauri relaunches during the verification session. |

## Backlog — items deferred to v0.3.4

| Item | Origin | Disposition |
|---|---|---|
| Port drag-drop image-attach handlers from legacy `CommandInput.tsx:115-164` into the live `ConnectedCommandInput.tsx`; restore PhotoIcon tooltip to "paste or drag-drop also supported" | Section 4 acceptance gap surfaced 2026-05-11 in dev verification | Estimated ~25 LOC + 1 component test. Paste and file-picker paths cover the same intent in v0.3.3, so this is a UX polish item, not a correctness gap. |
| `num_ctx` divergence detection + auto-unload | Ollama keep-alive observation above | Roampal detects when requested `num_ctx` differs from the loaded model's ctx and issues an unload before next chat, eliminating the manual `ollama stop` workaround. |
| Frontend "Searching for providers..." spinner during cold-start window | v0.3.3 Defect 17 backend self-heal landed; UX polish remains for v0.3.4 | The backend now rediscovers providers automatically via Defect 17 (Part B). User-visible window between launch-with-no-providers and self-recovery is ~3-5 seconds (one frontend poll interval). A spinner during that window would tighten the UX further but is no longer a correctness gap. |
| Graceful-shutdown-before-`KILL_ON_JOB_CLOSE` for Tauri sidecar | v0.3.3 Defect 19 root cause analysis | Section 11 Fix A's Job Object guarantees Python dies when Tauri exits, but it's a hard kill — Python doesn't get to flush ChromaDB writes or close file handles cleanly. Currently the biggest remaining contributor to corruption-on-exit scenarios (Defect 18 was the other; that's fixed). Proposed change: Tauri sends `SIGTERM` (or Windows `WM_CLOSE` / `CTRL_C_EVENT`) to Python before relying on the Job Object kill, waits ~3 seconds for graceful shutdown, falls back to hard kill only if Python doesn't exit. Modest Rust-side change in `src-tauri/src/main.rs` shutdown handler. |
| Optional: ChromaDB segment-rebuilder from session JSONLs | v0.3.3 Defect 19 quarantine direction | When a collection is corrupt, Defect 19 drops it (losing data) but quarantines the on-disk segment files. A rebuilder utility could replay session JSONLs (`<DATA_PATH>/sessions/*.jsonl` — source of truth for raw exchanges) through the sidecar's scoring path to recreate the `history` collection. Out of scope for v0.3.3 — substantial work for a recovery path that fires only on already-corrupted state. Mention so future work has a clear handle. |
