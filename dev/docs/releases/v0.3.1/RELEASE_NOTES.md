# Roampal Desktop v0.3.1 — TagCascade Retrieval + Sidecar

**Release Date:** 2026-04-15
**Theme:** Benchmark-validated retrieval + background memory processing. Same architecture as roampal-core v0.4.9.x. Desktop inherits CLI's proven memory module — not a separate implementation.

---

## What Changes for Users

1. **Better retrieval** — TagCascade: tags-first cascade fills candidate pool before cross-encoder reranking. Benchmark-validated at 27.3% Hit@1 (p<0.0001 vs Wilson+CE blend).
2. **Sidecar model** — New dropdown in header to select a background LLM (e.g. qwen2.5:3b). Summarizes exchanges, extracts atomic facts and noun tags automatically after each chat turn. Pick a small fast model — it runs in the background.
3. **Multilingual** — CE model supports 14+ languages (was English-only).
4. **Lighter install** — PyTorch + sentence-transformers removed, replaced with ONNX Runtime (already installed via ChromaDB).
5. **Two-lane injection** — 4 summaries + 4 atomic facts = 8 memories per context. Doubles prior 4-slot allocation.
6. **Atomic facts** — Sidecar extracts facts automatically. `record_response` also accepts `facts` parameter for manual extraction.
7. **Cleaner architecture** — Knowledge graph removed from retrieval. Tag routing replaces it.
8. **Tag filtering in Memory Panel** — Substack-style type-to-add tag input with typeahead suggestions. Applied tags shown as pills. AND logic filtering. Memory cards display noun_tags as clickable badges. Search also matches tags.
9. **Model selector redesign** — Model-first UI replaces provider-first. No more Ollama/LM Studio tabs. All installed models from both providers shown together with provider badges. Type any Ollama model name to pull it directly. Sidecar dropdown shows all providers with small models recommended. Curated list of verified tool-capable models.
10. **Improved onboarding** — Welcome modal explains chat model + sidecar architecture. Curated model list with verified tool-capable models (qwen3, gemma4, gpt-oss). LM Studio treated as first-class provider. Sidecar toast after first model install. Tool calling enabled for all models by default (blocklist instead of allowlist).
11. **Memory migration tools** — Settings > Data Management:
    - **Summarize tab**: One-time tool to summarize legacy raw exchanges into concise notes, extract tags (LLM) and atomic facts (LLM). Scans working + history + patterns. Requires sidecar model. SSE progress streaming. Toast notification on first launch if unsummarized memories detected.
    - **Retag tab** (future): Optional tool to retag all memories with LLM-extracted tags. Updates `noun_tags` metadata for memories with regex tags or no tags. Improves TagCascade retrieval quality.

---

## Architecture: TagCascade Retrieval

```
Query
  |
Tag matching (word-boundary regex against known tag index)
  |
Tag-routed search (if tags match) OR cosine fallback
  Per-tag ChromaDB queries with noun_tags $contains filter
  Overlap counting: how many query tags each result matches
  Tier-fill pool: highest overlap first, cosine tiebreak within tier
  Cosine fill remaining slots from unfiltered search
  |
CE reranks top 40 candidates (ONNX, mmarco-mMiniLMv2-L12-H384-v1)
  Raw CE score as final ranking (NO Wilson blend)
  Wilson computed as metadata only (display, promotion thresholds)
  |
Two-lane injection:
  Lane 1: 4 summaries (memory_type != "fact")
  Lane 2: 4 facts (memory_type == "fact")
  Total: 8 memories per context
```

---

## Changes

### 1. TagCascade search pipeline
**File:** `modules/memory/search_service.py`

Replaces Wilson+CE blend with tag-routed retrieval, matching benchmark TagCascade:
- `_tag_routed_search()`: Per-tag ChromaDB queries, overlap counting, tier-fill pool
- `_merge_filters()`: Combines tag filter + metadata filters via ChromaDB `$and`
- `CE_CANDIDATE_POOL = 40` (was 20)
- Raw CE score as `final_rank_score` — no Wilson blend
- KG dependency removed (TagService replaces KnowledgeGraphService)
- **Fix:** Memory_bank search filters (two-lane `memory_type` + `status != archived`) now wrapped in `$and` (ChromaDB requires single operator per where clause)
- **v0.3.1.1 → v0.3.1.2 Update:** Facts now get noun_tags extracted at store time (matching benchmark: `entity_routed.py:248-250` extracts tags for ALL content including facts). Both lanes use TagCascade. Previous v0.3.1.1 incorrectly removed fact tags based on wrong assumption about benchmark behavior.

### 2. Two-lane context injection (matching benchmark)
**File:** `modules/memory/unified_memory_system.py`

- Lane 1: 4 summaries (`memory_type != "fact"`) — tag-routed + CE
- Lane 2: 4 facts (`memory_type == "fact"`) — tag-routed + CE (same as summaries — benchmark tags all content)
- Facts stored with `memory_type: "fact"` AND `noun_tags` (benchmark: `entity_routed.py:store()` extracts tags for all content at store time)
- Nursery slot removed (benchmark: zero benefit, p=1.0)

### 3. Store-time tag extraction (benchmark-aligned LLM tagging)
**File:** `modules/memory/unified_memory_system.py`

All new memories get noun_tags extracted at store time:
- `store()`: extracts tags via LLM when sidecar available, returns `[]` on failure (no regex fallback)
- `store_memory_bank()`: accepts `noun_tags` param, auto-extracts if omitted
- TagService wired with LLM extraction function in `main.py` when sidecar initialized
- Known tag index rebuilt from ChromaDB on init
- Matches benchmark: `entity_routed.py:store()` → `extract_tags()` returns `[]` on LLM failure

### 4. Cross-encoder reranking via ONNX
**File:** `modules/memory/search_service.py`

- Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (118M params, 14+ languages)
- ONNX Runtime, no PyTorch needed. Lazy-loaded on first search.
- Raw CE score as final ranking signal (Wilson is metadata-only)
- **Fix (v0.3.1.3):** `_ce_predict()` now uses `encode_batch()` instead of per-pair `encode()`. Fixes "inhomogeneous shape" error when tokenized pairs had different lengths — padding wasn't applied uniformly across the batch. CE was silently falling back to cosine-only on every query.

### 5. Embedding model: PyTorch -> ONNX
**File:** `modules/embedding/embedding_service.py`

Same model (`paraphrase-multilingual-mpnet-base-v2`, 768d), loaded via ONNX Runtime. Existing ChromaDB embeddings stay compatible.

### 6. MCP tool updates
**File:** `main.py`

- `add_to_memory_bank`: added `noun_tags` parameter (auto-extracted if omitted). Added SIZE GUIDANCE in description (~300 chars, one concept per fact). Added `MAX_MEMORY_CHARS = 2000` hard cap with silent truncation (matching core).
- `update_memory`: added `MAX_MEMORY_CHARS = 2000` hard cap with silent truncation (matching core).
- `score_memories`: NEW tool matching roampal-core. Per-memory scoring with 7-rule guide, exchange_summary, exchange_outcome, noun_tags, facts. Replaces blanket scoring.
- `record_response`: simplified to `key_takeaway` + `noun_tags` only. Scoring, facts, outcome moved to `score_memories`. Blanket scoring legacy path removed. Added `MAX_MEMORY_CHARS = 2000` hard cap (matching core).
- **Fix (v0.3.1.3):** Removed all `extract_tags_regex()` calls from MCP tool handlers. `record_response`, `score_memories` summary, and `score_memories` facts now use `memory._tag_service.extract_tags_async()` for LLM tag extraction when `noun_tags` not provided. Returns `[]` on failure (no regex fallback). Matches sidecar path behavior.
- Tool count: 7 (search_memory, add_to_memory_bank, update_memory, archive_memory, get_context_insights, record_response, score_memories).

### 7. Routing simplified
**File:** `modules/memory/routing_service.py`

- KG routing removed. `route_query()` returns all collections.
- Tag routing in SearchService handles precision.
- Acronym expansion preserved.

### 8. Scoring simplified
**File:** `modules/memory/scoring_service.py`

- Memory_bank special case removed — all collections use same 5-tier dynamic weight system
- Wilson score still computed — used for display and promotion, not ranking

### 8b. Outcome scoring aligned with core v0.4.5
**Files:** `modules/memory/outcome_service.py`, `modules/memory/promotion_service.py`

- KG calls removed from outcome recording (was: `update_kg_routing()`, `_update_kg_with_outcome()`, `_track_problem_solution()`)
- Time-weight removed from score deltas — flat +0.2/-0.3/+0.05 (was: multiplied by `1/(1+age_days/30)`)
- Unknown outcome delta changed from +0.0 to -0.05 (weak negative, matches core v0.2.9+)
- Promotion working→history: only resets `success_count` (was: reset both `success_count` AND `uses`)
- Promotion history→patterns: now resets `success_count` (was: no reset)
- Batch cleanup every 50 scores: working 24h + history 30d (was: no periodic cleanup)
- `cleanup_old_history(max_age_hours=720.0)` added to PromotionService

### 9. KG infrastructure — fully removed
**Files deleted:** `knowledge_graph_service.py`, `content_graph.py`, `intelligent_router.py`, `autonomous_router.py`, `migrate_knowledge_graph.py`, `test_knowledge_graph_service.py`

KG fully removed from codebase — no routing, no search, no display, no backup/restore. All KG imports, init, and method calls removed from unified_memory_system.py, memory_bank_service.py, context_service.py, data_management.py, backup.py, memory_visualization_enhanced.py, settings.py, startup_cleanup.py, main.py (promotion task `_cleanup_kg_dead_references` call removed). No-op stubs retained for `record_action_outcome()` and `_update_kg_routing()` to avoid breaking MCP callers. KG JSON files on disk are ignored (not loaded, not read). Frontend `/api/memory/knowledge-graph` endpoint removed (was returning 404).

### 10. Sidecar service — sole scorer, benchmark-aligned architecture
**Files:** `modules/memory/sidecar_service.py` (NEW), `app/routers/agent_chat.py`, `app/routers/model_switcher.py`, `config/settings.py`, `main.py`, `ConnectedChat.tsx`

Background LLM that processes exchanges — two focused calls matching core v0.4.8:
- **Call 1:** `score_exchange()` — summary + outcome + per-memory scores (3 fields)
- **Call 2:** `extract_facts()` — atomic fact extraction from exchange

**Sidecar input limits (v0.3.1.4):** Bumped all truncation caps to give sidecar models full exchange context. Even 3b models handle 32K+ tokens — previous caps were unnecessarily restrictive:
- `score_exchange`: user_msg 8000 (was 2000), assistant_msg 8000 (was 1000), followup 4000 (was 500)
- `extract_facts`: content 8000 (was 3000)
- `extract_noun_tags`: text 2000 (was 500)
- Summary output safety trim bumped to 2000 (was 400) — prompt still instructs ~300 chars, 2000 is safety net matching MAX_MEMORY_CHARS
- **Scored memory cap: 8** (`MAX_SCORED_MEMORIES`). Takes the last 8 memories searched (most recent priority — later searches more likely found the answer). Matches two-lane injection (4+4). Optimized for small sidecar models (3b+) — 8 JSON score entries is reliable, 16+ caused output failures. Unscored memories keep their current Wilson score. Worst case: ~5,700 tokens total.
- **Recent exchanges on cold start (v0.3.1.4):** On first message of a new conversation, injects last 4 exchange summaries (pure recency, no semantic query) alongside the user profile. Matches core/OpenCode behavior (`roampal.ts:1187-1232`). Gives the chat LLM continuity from past conversations without searching. Summaries filtered to actual sidecar-generated content (has `summarized_at` or <500 chars). Pre-sets `internal_session_message_counter` for loaded conversations on startup so cold start doesn't re-fire on app restart continuations — only genuinely new conversations get the injection.

Tags extracted via LLM when sidecar available (benchmark-aligned), returns `[]` on failure (no regex fallback). Summary gets tags when `_run_sidecar()` calls `_tag_service.extract_tags_async(summary)`. Facts get tags automatically via `store()` → `_tag_service.extract_tags_async()`. Matches benchmark: `entity_routed.py:store()` → `extract_tags()` returns `[]` on LLM failure.
- Fires on the NEXT user message (the followup signal), not immediately after the exchange
- **Fix (v0.3.1.3):** Sidecar pending survives app restart. On startup, `_load_conversation_histories()` reconstructs `_sidecar_pending` from the last user/assistant pair in each session file. Continuing a conversation after relaunch triggers sidecar for the previous exchange. Marker file (`_sidecar_processed.json`) prevents duplicate processing if sidecar already ran before the restart.
- **Fix (v0.3.1.3):** Chat timeout bumped from 30s to 120s (`ollama_request_timeout_seconds`). 30s was too aggressive for large models (31B+) that need GPU reload after sidecar displaces them. Applies to both Ollama and LM Studio providers.
- **Fix (v0.3.1.3):** Chat errors now visible to user. WebSocket `error` events were silently logged to console — now display as assistant message ("Sorry, I encountered an error..."). Also added handler for `type: "done"` messages (validation errors like "No chat model available") which were silently dropped. User no longer sees a blank response on timeout or model validation failure.
- **Fix (v0.3.1.3):** Model auto-switch race condition. `fetchModels()` and `fetchCurrentModel()` fired in parallel on mount — auto-switch tried to pick a model before the model list was populated, resulting in `current_model: null` while UI showed a model from localStorage. Fixed by moving auto-switch to a separate `useEffect` that triggers after `availableModels` is populated. Picks first available chat model from any provider (Ollama or LM Studio).
- **Fix (v0.3.1.3):** Tool calling switched from allowlist to blocklist. Old allowlist silently disabled tools for any model not explicitly listed (e.g. gemma4). Now all models get tools by default, only embedding models and known-broken models are blocked.
- **v0.3.1.3: Sidecar onboarding** — after first chat model install, if no sidecar is configured, shows sidecar setup toast ("Select a sidecar model to start recording memories") with 3s delay. Complements the existing first-chat toast.
- **v0.3.1.3:** No raw exchange in ChromaDB — matches core Claude Code path. Raw transcript only persisted in session JSONL file. Sidecar stores summary as NEW working memory with LLM tags (no `update_fragment_metadata`, no regex).
- Sidecar succeeds → summary + facts stored with LLM tags. Sidecar fails → nothing in ChromaDB (exchange only in session file).
- Stores ~300 char first-person summary as new working memory (no raw to replace)
- Extracts atomic facts → stored as `memory_type: "fact"` WITH noun_tags (feeds TagCascade — both lanes use tags)
- Scores each cached memory using the 7-rule guide (sidecar is sole scorer)
- **No autoSummarize** — removed to eliminate Ollama contention. Sidecar handles all summarization.
- **No raw exchange storage** — if sidecar fails, exchange is lost. No transcript pollution.
- Sequential calls, no concurrency — clean for local models (Ollama/LM Studio process one request at a time)
- **Fix:** Sidecar client now properly initialized on startup via `OllamaClient.initialize()` (was setting fields without creating httpx client)
- **Fix (v0.3.1.3):** Frontend re-initializes sidecar on mount if localStorage has a saved model but backend reports `enabled: false`. Prevents sidecar silently dropping after DB clean or config loss. `ConnectedChat.tsx` calls `/api/model/sidecar/set` to restore.
- **Fix:** `/api/model/sidecar/set` endpoint also calls `initialize()` (same bug)
- UI dropdown in header next to main model selector
- **v0.3.1.3 UX:** Four-level sidecar awareness:
  1. **Toast on first chat without sidecar** — "Select a sidecar model to start recording memories." with "Set Up" action that opens sidecar dropdown. Once per session.
  2. **Memory Panel empty state** — amber text: "No sidecar model selected. Pick one in the header to enable memory."
  3. **Memory Panel with existing memories** — amber banner: "No sidecar model selected — new memories won't be created."
  4. **Red dot on sidecar failure** — pulsing red dot on sidecar button when last scoring call failed. Hover tooltip shows the error. Border turns red. Clears automatically on next successful call or model re-selection. Backend tracks `sidecar_last_error` in app state (stored on `self._app_state` in agent_chat), exposed via `/api/model/sidecar/status`. Frontend polls every 30s.
- Persisted to `data/sidecar_config.json`
- **v0.3.1.3: Model validation on set** — `/api/model/sidecar/set` now validates model exists on provider. If `qwen2.5:7b` (Ollama format) sent to LM Studio, fuzzy-matches to `qwen2.5-7b-instruct`. Prevents 400 Bad Request from model name mismatch between providers.
- API: `GET /api/model/sidecar/status`, `POST /api/model/sidecar/set`, `POST /api/model/sidecar/disable`

The sidecar is the sole memory scorer. The emoji-based scoring system (`detect_conversation_outcome()`, `EMOJI_TO_OUTCOME` mapping, `parse_memory_marks()`) has been removed entirely.

### 11. 7-rule per-memory scoring (via sidecar)

The sidecar's combined prompt includes the full 7-rule scoring guide for per-memory scoring:
1. Memory NOT about the topic discussed → "unknown"
2. Memory IS about topic AND outcome "worked" → "worked"
3. Memory IS about topic AND outcome "failed" + response echoed memory → "failed"
4. Memory IS about topic AND outcome "failed" + unrelated → "unknown"
5. Memory IS about topic AND outcome "partial" → "partial"
6. Memory contains good advice response IGNORED → "unknown" not "failed" ("failed" means content was WRONG)
7. When in doubt → "unknown"

### 12. ChromaDB adapter fix
**File:** `modules/memory/chromadb_adapter.py`

`update_fragment_metadata()` now passes `documents=` parameter when `text` or `content` is in the metadata update. Previously, document text was never updated — only metadata fields. This fix is required for sidecar summarization to actually replace the searchable exchange text.

### 13b. Dependencies
**File:** `requirements.txt`

Remove (v0.3.1: PyTorch replacement):
- `sentence-transformers==2.2.2`
- `torch>=2.1.0,<2.9.0`
- `transformers>=4.36.2`

Remove (v0.3.1: license/cleanup):
- `PyMuPDF==1.23.8` — AGPL-3.0 (blocks commercial use), replaced by `pypdf` (BSD)
- `chardet==5.2.0` — LGPL-2.1, replaced by `charset-normalizer` (MIT)
- `PyPDF2==3.0.1` — unused (PyMuPDF was used instead, now pypdf)
- `playwright==1.40.0` — unused (never imported)
- `rich==13.7.0` — unused (never imported)
- `typer==0.9.0` — unused (never imported)
- `click==8.1.7` — unused (never imported)
- `watchfiles==0.21.0` — unused (dev tool, never imported)
- `orjson>=3.9.14` — unused (never imported)
- `ollama` (git+https) — unused (httpx used for Ollama API directly)

Add:
- `onnxruntime>=1.14.0`
- `tokenizers>=0.15.0,<1.0.0`
- `huggingface-hub>=0.20.0`
- `pypdf>=4.0.0` — PDF extraction (BSD, replaces PyMuPDF AGPL)
- `charset-normalizer>=3.0.0` — encoding detection (MIT, replaces chardet LGPL)

### 13. Tag filtering UI
**Files:** `MemoryPanelV2.tsx`, `ConnectedChat.tsx`

#### Current Implementation (v0.3.1.3):
- **Substack-style tag input** — type a tag name, hit Enter to apply as filter. Backspace to remove last tag. Escape to dismiss suggestions.
- **Typeahead suggestions** — as you type, matching tags appear in a dropdown with frequency counts. Click or Enter to select.
- **Applied tags as pills** — selected tags shown as blue pills with X buttons inside the input field. AND logic — must have ALL selected tags.
- **Clear all** — X button at right end of input clears all tag filters.
- Tags > 25 chars filtered from suggestions (prevents sentence fragment garbage)
- Only tags with count >= 2 shown in suggestions (hides noise from sparse data)
- Memory cards display `noun_tags` as clickable badge chips (max 4 shown, "+N" overflow)
- Search bar also matches against tags
- Detail modal shows all tags + `memory_type` field
- `ConnectedChat.tsx`: Fixed noun_tags parsing — reads `metadata.noun_tags` (JSON string) instead of `metadata.tags`

- **Font consistency (v0.3.1.3):** All tag UI elements use `text-xs` matching search bar and filter dropdowns (was `text-[11px]`/`text-[10px]`). Placeholder left-aligned (no cursor overlap).

**⚠️ Remaining Limitations:**
1. **Tag filtering only in Memory Panel** — not available in main chat interface (ConnectedChat.tsx)
2. **No OR logic option** — can't search for memories with "tag1 OR tag2"
3. **No persistence** — selected tags reset when panel closes
4. **No tag management** — can't rename/merge/delete tags

### 14. Model selector redesign — model-first
**File:** `ConnectedChat.tsx`

Replaces provider-first UI (Ollama | LM Studio tabs) with model-first design:
- `getModelOptions()` no longer filters by `selectedProvider` — shows ALL models from ALL providers
- Provider selector toggle removed from header bar
- Each model in dropdowns shows provider badge in description (Ollama / LM Studio)
- Chat dropdown passes model's actual `provider` to `switchModel()` and `handleUninstallModel()`
- Sidecar dropdown uses `getSidecarModelOptions()` — groups small models (≤7b) as "Recommended", others under "All Models". Passes model's actual provider to `/api/model/sidecar/set` (was hardcoded `selectedProvider || 'ollama'`)
- Model Library modal: provider tabs removed, unified model list with per-provider install buttons
- Curated model list shrunk from ~15 to ~6 picks organized as "Chat Models" and "Sidecar Models"
- "Pull from Ollama" text input — type any Ollama model name (auto-appends `:latest` if no tag)
- "From LM Studio" note — models loaded in LM Studio appear automatically
- `handleInstallModel()` accepts optional `targetProvider` parameter
- No backend changes needed — endpoints already support cross-provider

### 15. Onboarding & welcome modal updates (v0.3.1.3)
**Files:** `OllamaRequiredModal.tsx`, `ConnectedChat.tsx`

Welcome modal ("Welcome to Roampal") updated:
- Tab 1 renamed "Setup LLM Provider" → "Getting Started"
- LM Studio no longer labeled "Advanced users only" — now "GUI-based, load models visually"
- Added "How Roampal works" section explaining chat model vs sidecar model architecture
- MCP tab: "7 memory tools via standard protocol" (was "6 core tools")

"No Model Installed" state (main chat, no models detected):
- Updated text: "Download a chat model to start chatting (e.g. Llama, Qwen, Gemma). Then pick a small sidecar model to enable persistent memory across conversations."

Curated model list in Model Library (sources: ollama.com/search?c=tools, community testing from clawdbook, morphllm, collabnix):
- **Chat Models** (verified tool calling support, 14B+ recommended):
  - qwen3:32b (20GB) — Stable tool calling, top-tier performance
  - gemma4:26b (17GB) — Native function calling, multimodal
  - gpt-oss:20b (14GB) — OpenAI open-weight, clean tool calling
  - qwen2.5:14b (9.0GB) — Reliable tools, moderate VRAM
  - gemma4 8B (9.6GB) — Native tools, lower VRAM
- **Sidecar Models** (no tool calling needed — any model works):
  - gpt-oss:20b (14GB) — Benchmark validated sidecar
  - qwen2.5:7b (4.7GB) — Fast, low VRAM
  - qwen2.5:3b (1.9GB) — Ultra-light

Model Library header: "Models from Ollama and LM Studio appear automatically" — auto-detected models from both providers show alongside curated picks.

Tool calling blocklist: switched from allowlist (`NATIVE_TOOL_MODELS`) to blocklist (`TOOL_BLOCKLIST`). All models get tools by default. Only embedding models and known-broken models blocked (dolphin, nomic-embed, all-minilm, bge-).

### 16. Memory migration tool — summarize legacy exchanges (LLM tags)
**Files:** `sidecar_service.py`, `data_management.py`, `DataManagementModal.tsx`, `Toast.tsx`, `ConnectedChat.tsx`, `SettingsModal.tsx`

One-time migration tool for pre-sidecar raw exchanges. Per-memory pipeline matches sidecar architecture:
1. `summarize_only(content)` → ~300 char first-person summary (LLM)
2. `extract_noun_tags(summary)` → noun_tags (LLM, not regex)
3. `extract_facts(content)` → atomic facts from original text (LLM)
4. Update memory: replace text, set `summarized_at`, `original_length`, `noun_tags`
5. Store facts as `memory_type: "fact"` working memories with LLM tags

Backend:
- `GET /api/data/summarize/scan` — lightweight candidate count (>400 chars, no `summarized_at`)
- `POST /api/data/summarize/run` — SSE streaming summarization with progress events
- `POST /api/data/summarize/cancel` — flag-based cancellation
- Uses `extract_noun_tags()` for LLM tag extraction (matching sidecar)
- Scans working + history + patterns (fixes core CLI bug that skips patterns)

Frontend:
- "Summarize" tab in DataManagementModal (amber theme)
- Scan results with per-collection breakdown
- Progress bar with SSE streaming
- Cancel button, completion summary
- One-time toast notification on app load when unsummarized memories detected + sidecar configured
- Toast action deep-links to Settings > Data Management > Summarize tab
- localStorage flag prevents repeat toast after dismissal

---

## Data Migration

**Handled.** The Summarize tab in Data Management (Settings > Data Management > Summarize) handles legacy memory migration:

- All **new** memories get LLM tags at store time (no regex fallback)
- **Legacy memories** (pre-sidecar raw exchanges) are processed by the Summarize tool: summary + facts + LLM tags
- TagCascade gracefully falls back to cosine when no tags match (untagged old memories still searchable)

---

## Benchmark Evidence

From roampal-labs LoCoMo evaluation (1,537 non-adversarial questions):

| Config | Hit@1 Clean | Hit@1 Poison | p-value |
|--------|-------------|--------------|---------|
| **TagCascade + cosine** | **27.3%** | **29.0%** | **baseline** |
| Overlap + cosine | 25.8% | 28.0% | p=0.0003 |
| Pure CE | 25.4% | 28.4% | — |
| TagCascade + Wilson | 23.0% | 25.0% | p<0.0001 |
| Wilson only | 30.6% | — | p<0.0001 |

Wilson hurts retrieval by 4.3 points in every configuration tested.
Two-lane retrieval adds +6.1 Hit@1 (p<0.0001).
Nursery slot: zero benefit (p=1.0).

---

## Implementation Verification

**✅ All v0.3.1 architectural claims verified in code:**

### TagCascade Retrieval:
- ✓ `search_service.py`: Tag-routed search with overlap counting implemented
- ✓ `CE_CANDIDATE_POOL = 40` (was 20)
- ✓ Raw CE score as `final_rank_score` — no Wilson blend
- ✓ KG dependency removed (TagService replaces KnowledgeGraphService)

### Two-Lane Context Injection:
- ✓ `unified_memory_system.py:1261-1313`: 4 summaries + 4 facts = 8 memories
- ✓ Facts stored with `memory_type: "fact"` AND `noun_tags`
- ✓ Nursery slot removed (benchmark: p=1.0)

### Store-Time Tag Extraction (Benchmark-Aligned):
- ✓ `unified_memory_system.py:665-678`: Extracts tags at store time
- ✓ `tag_service.py:400-429`: Returns `[]` on LLM failure (no regex fallback)
- ✓ `main.py:510-522`: `async_llm_tag_extractor()` returns `[]` not `None` on failure

### Sidecar 2-Call Architecture:
- ✓ `sidecar_service.py`: `score_exchange()` + `extract_facts()` separate calls
- ✓ `agent_chat.py:1898-2038`: Sequential calls, no concurrency
- ✓ Sidecar is sole scorer (emoji scoring removed)

### Memory Migration Tool:
- ✓ `data_management.py:684-879`: Uses `extract_noun_tags()` (LLM, not regex)
- ✓ Matches sidecar architecture for tag extraction
- ✓ SSE streaming with progress events

### Tag Filtering UI:
- ✓ `MemoryPanelV2.tsx:207-266`: Tag cloud with top 12 tags (count ≥ 2)
- ✓ `MemoryPanelV2.tsx:380-406`: Clickable tags on memory cards
- ✓ AND logic filtering implemented
- ✓ **Note**: Tag filtering currently only in Memory Panel, not main chat interface

### Model Selector Redesign:
- ✓ `ConnectedChat.tsx:1156-1234`: Model-first UI (not provider-first)
- ✓ Shows ALL models from ALL providers
- ✓ Sidecar dropdown groups small models (≤7b) as "Recommended"

### Onboarding & Tool Calling:
- ✓ `OllamaRequiredModal.tsx`: Welcome modal explains chat + sidecar architecture, LM Studio first-class
- ✓ `ConnectedChat.tsx`: Curated model list with verified tool-capable models (qwen3, gemma4, gpt-oss)
- ✓ `ConnectedChat.tsx`: Sidecar toast after first model install
- ✓ `ollama_client.py`: Tool calling blocklist (was allowlist) — all models get tools by default

### Dependencies Updated:
- ✓ `requirements.txt`: Removed PyTorch, added ONNX Runtime
- ✓ Same embedding model (`paraphrase-multilingual-mpnet-base-v2`) via ONNX

**📊 Verification Result:** All architectural claims verified in code. Minor UI scope notes: tag filtering in Memory Panel only (not ConnectedChat), curated model list subject to change as new models release.

---

## Context Window Management (v0.3.1.3)

**File:** `app/routers/agent_chat.py`

At 90% of model context window:
1. **Wrap-up instruction injected** — system message tells LLM: "Wrap up your current response concisely. Tell the user what you've completed and what still needs to be done."
2. **Drop context memories** — removes injected memory context to free space
3. **Trim history** — reduces conversation history to last 2 messages if still over
4. **User notification** — visible message: "Context limit reached (X% full)"

---

## Model Loading Improvements (Post-v0.3.1)

**Issue:** Desktop app was automatically loading large models (e.g., `gemma4:31b`) into GPU memory on startup, causing GPU overload.

**Solution:** Implemented lazy loading with lightweight health checks:

### 1. **Lightweight Model Switching**
- **Health checks**: Changed from inference requests (`/api/chat`, `/v1/chat/completions`) to availability checks (`/api/tags`, `/v1/models`)
- **No GPU loading**: Model switching verifies model availability without loading into GPU
- **Faster**: 10s timeout (was 90s for LM Studio)
- **Clear errors**: "Ollama not responding" instead of model load timeouts

### 2. **Sidecar Client Sharing and Queuing**
- **Client sharing**: Sidecar shares client with main LLM when same provider/model
- **Client locking**: `execute_with_client_lock()` prevents concurrent model loading
- **Retry queue**: Failed sidecar tasks queue with exponential backoff (1min→2min→4min)
- **Always runs**: Sidecar never silently fails - queued tasks retry automatically
- **Background processor**: Runs every 30s to process retry queue

### 3. **Lazy Loading Architecture**
- **First message trigger**: Models load into GPU only on first `generate_response()` call
- **No startup loading**: App initialization doesn't trigger model load
- **OllamaClient.initialize()**: Only creates HTTP client, doesn't call Ollama

### 4. **Sidecar Queuing System**
- **Fix (v0.3.1.3):** Retry queue was passing `task_type` and `doc_id` as kwargs to sidecar functions (e.g. `extract_facts(**task)` where task included `task_type`), causing `got an unexpected keyword argument 'task_type'`. Fixed by filtering internal keys before calling.
- **Persistent queue**: `sidecar_queue.py` manages failed tasks
- **Exponential backoff**: 3 retry attempts with increasing delays
- **Task types**: `score_exchange`, `extract_facts`, `extract_noun_tags`, `summarize_only`
- **Background processor**: Started automatically on app startup
- **No user impact**: Runs in background, doesn't block chat

### 5. **Best Practices**
- **Same model allowed**: Main LLM and sidecar can use same model (client sharing)
- **Configure explicit model**: Set `OLLAMA_MODEL` in `.env` to avoid auto-selection of large models
- **Monitor GPU**: Use `nvidia-smi` or Task Manager to track VRAM usage
- **Reliability**: Sidecar always runs eventually via retry queue

**Files updated:**
- `app/routers/model_switcher.py`: Lightweight health checks, client sharing logic
- `modules/memory/sidecar_queue.py`: **New** - retry queue management
- `modules/memory/sidecar_service.py`: Wrapper functions with locking/retry
- `app/routers/agent_chat.py`: Updated to use retry wrappers
- `app/routers/data_management.py`: Updated to use retry wrappers
- `main.py`: Starts background retry processor
- `dev/docs/architecture.md`: Updated documentation

---

## Reference

- **Benchmark:** `C:/roampal-labs/` — LoCoMo dataset, cascade isolation tests
- **Core spec:** `C:/roampal-core/dev/docs/releases/v0.4.9.2/RELEASE_NOTES.md`
- **CE model:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache 2.0, ONNX)
- **Paper:** `C:/roampal-labs/paper.md` — Section 5.2.3 component elimination table
