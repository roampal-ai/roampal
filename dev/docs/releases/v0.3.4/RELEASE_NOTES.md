# Roampal Desktop v0.3.4 — Release Notes (DRAFT)

**Status:** PLANNING — items below are deferred from v0.3.3 verification (2026-05-11) and other observations during the v0.3.3 cycle. Scope to be confirmed.
**Target ship:** TBD
**Type:** Polish / UX hardening (no functional regressions expected from v0.3.3)

---

## Summary

v0.3.3 closed the structural correctness work (Issue #8 phantom-dedup, capability detection, atomic JSON writes). v0.3.4 picks up the rough edges and missing polish that surfaced during v0.3.3 dev verification but were judged out of scope for that release. The items below are the candidate set as of 2026-05-12; firm scoping will follow once v0.3.3 ships.

**Active scope:** Section 1 (drag-drop image attach) and Section 2 (`num_ctx` auto-unload). Section 3 (image attachment persistence) was originally listed here but was actually completed during v0.3.3 verification via Defects 4 / 10 / 12 — retained below as a struck-out resolution record.

---

## 1. Drag-drop image attach in `ConnectedCommandInput`

**Origin:** v0.3.3 Section 4 verification, 2026-05-11. The Section 4 spec called for drag-drop alongside paste and the photo-icon picker, but only paste and the file picker were wired into the live `ConnectedCommandInput.tsx`. The handler code already exists in the legacy `CommandInput.tsx:115-164` (`handleDrop`, `onDragOver`, `onDragLeave`, `onDrop`, plus an `isDragging` state for visual feedback).

**Scope.** Port the four handlers and the dragging-state visual cue from `CommandInput.tsx` into `ConnectedCommandInput.tsx`. Restore the PhotoIcon tooltip text to "Attach image (paste or drag-drop also supported)" — explicitly downgraded in v0.3.3 because the implementation never landed.

**Files affected:**

| File | Change |
|---|---|
| `ui-implementation/src/components/ConnectedCommandInput.tsx` | Add `handleDrop`, `onDragOver`, `onDragLeave`, `onDrop` handlers; add `isDragging` state with a visual drop-zone overlay; reuse the existing `addImageFiles` helper to push dropped files into `attachments` state |
| `ui-implementation/src/components/ConnectedCommandInput.tsx` | Restore PhotoIcon `title` to `"Attach image (paste or drag-drop also supported)"` |
| `ui-implementation/src/test/components/ConnectedCommandInput.test.tsx` (new or existing) | Add one test: simulating `onDrop` with a `File` event populates `attachments` with the dropped image file |

**Estimated effort.** ~25 LOC + 1 test, ~15 minutes.

**Acceptance criteria.**

- Dragging an image file over the chat input shows a drop-zone visual cue
- Releasing the file inside the drop zone adds it to the attachment preview chips
- Releasing the file outside the chat input does nothing (no accidental capture by other regions)
- Non-image files dropped on the input are ignored silently (mirror the paste handler's behavior)

---

## 2. `num_ctx` divergence detection + auto-unload

**Origin:** v0.3.3 verification observation. Ollama's keep-alive caches a loaded model with whatever `num_ctx` it was first loaded at. Subsequent requests that pass a different (smaller) `num_ctx` are silently served from the existing larger-ctx instance — Ollama does not unload and reload to honor the new value. This manifested during v0.3.3 testing: user lowered the UI context-window setting from 8 K to 100 K, but `ollama ps` continued to report 262 144 ctx and the model continued to run with a 22 % CPU / 78 % GPU split until manually unloaded with `ollama stop gemma4:31b`.

**Proposed remediation.**

- Track the last-known loaded `num_ctx` per `(provider, model)` pair in a small in-memory map (keyed by `f"{provider}::{model}"`).
- Before each generation request, compare the request's `num_ctx` to the last-known loaded value.
- If the values diverge AND the requested value is smaller, issue an explicit unload before the generation request — for Ollama: `POST /api/generate {"model": "...", "keep_alive": 0}` (an empty request with `keep_alive: 0` forces eviction).
- Update the in-memory map after each successful generation.

**Acceptance criteria.**

- User lowers ctx in Settings, sends next chat → `ollama ps` reports the new ctx
- No manual `ollama stop` required
- No measurable latency added to chat path when ctx hasn't changed (cache hit fast path)

**Estimated effort.** ~40 LOC in `ollama_client.py` + 3-4 tests (mock Ollama, simulate request-with-diff-ctx scenarios), ~30 minutes.

---

## ~~3. Image attachment persistence across refresh and restart~~ — Completed in v0.3.3

**Status:** ✅ Already shipped in v0.3.3 via Defects 4, 10, and 12 (resolved 2026-05-11). This item was originally drafted before the v0.3.3 defect cycle closed the same gap; documenting the resolution here for traceability.

**What landed in v0.3.3:**

- Defect 4 (`agent_chat.py:3266-3278`): non-streaming task path saves base64 images to `<DATA_PATH>/attachments/<sha256>.<ext>` and writes filename references to the session JSONL `images` field.
- Defect 10 (`agent_chat.py:3420-3436`): streaming task path mirrors the same persistence (was the primary regression — WebSocket-connected sessions hit this path).
- Defect 12 (`agent_chat.py:2349-2394`): on `_load_conversation_histories`, filenames are rehydrated to data URLs via `utils.attachments.load_image_attachment_as_data_url` so multi-turn replay to LM Studio doesn't fail with `Invalid url.`.
- `app/routers/attachments.py`: `GET /api/attachments/{filename}` serves bytes back with `Cache-Control: public, max-age=31536000, immutable`.
- `sessions.py:163-165`: session-load endpoint returns the `images` filename list to the frontend.
- `useChatStore.ts:374, 1579`: frontend maps each filename to `${apiUrl}/api/attachments/${f}` before populating `message.images`.
- `EnhancedChatMessage.tsx:133-144`: existing `<img>` renderer continues to work against the resolved URLs.

**Acceptance criteria — all met in v0.3.3:**

- ✅ Send an image, see thumbnail in the user bubble
- ✅ Refresh window → thumbnail still present
- ✅ Quit + relaunch Tauri → thumbnail still present
- ✅ Session JSONL contains the attachment pointer (filename) under the user turn record

Storage strategy chosen: sidecar `attachments/` dir keyed by SHA-256 hash (the second option flagged in the original risk note). Bytes live on disk once per unique image regardless of how many sessions reference them.

---

## 4. Zero-yield stream fallback (latent since v0.1.0; user-visible since v0.2.5)

**Origin:** v0.3.3 post-ship diagnosis (2026-05-12). When a streaming response completes cleanly but emits no visible content, the UI sees nothing — no error, no message, no indication anything happened. The most common trigger is a thinking-mode model (qwen3 family especially) where the model exhausts its context budget on `reasoning_content` and never reaches `content`. Reasoning is intentionally discarded by `ollama_client.py` (introduced v0.2.5 — `# v0.2.5: Discard reasoning content entirely`), so when reasoning is the *only* output, the stream completes with `yield_count == 0` and the chat shows silence.

**Git-blame confirms the gap is from day one:** the stream-finish logging line has been at the same spot since the initial commit (`a4dbfb1` v0.1.0). `git log -S "yield_count == 0"` returns no matches across the entire history of `ollama_client.py` — a zero-yield fallback path has never existed.

**Why this wasn't caught earlier:**

- Pre-v0.2.5: no reasoning-content path existed, so the only way to get to zero-yield was a hard provider error (which already surfaced through the error-handling branches).
- v0.2.5 → v0.3.3: zero-yield became reachable for any thinking-mode model whose budget was exhausted by reasoning. Specific symptom: open new chat with low-ctx LM Studio + qwen3 thinking model, send a question that needs more than `loaded_context_length` tokens of reasoning to answer — model emits reasoning, never gets to the answer, stream closes, UI is silent.
- v0.3.3 Defect 22 broadened the *pre-stream* error matcher and added a `loaded_context_length` pre-validation gate, which catches a large fraction of these cases before the stream starts. The remaining slice — where the request begins streaming successfully but reasoning consumes the whole budget mid-stream — still falls into the silent-yield bucket and is what this item closes.

**Scope.**

- Helper at top of `modules/llm/ollama_client.py`:

  ```python
  def _zero_yield_fallback_message(reasoning_seen: bool) -> str:
      """Surface SOMETHING when a stream completes cleanly but yielded no content."""
      if reasoning_seen:
          return (
              "**Model produced only reasoning, no visible answer.**\n\n"
              "This usually means the model's context window is too small to both "
              "think and respond. Try a larger context (16384+) or switch to a "
              "non-thinking model."
          )
      return (
          "**Model returned no output.**\n\n"
          "The provider closed the stream without producing visible content. "
          "This can happen on context overflow, OOM, or a provider hiccup. "
          "Check LM Studio / Ollama logs."
      )
  ```

- **OpenAI-style stream finish (LM Studio path, ~line 1171):** before the final `[STREAM DEBUG] OpenAI stream finished` log, check `if yield_count == 0:` and yield the fallback. Pass the existing `in_reasoning_mode` flag (already tracked at the reasoning-content branch, ~line 1104) so the helper picks the more specific "reasoning-only" message variant.

- **Ollama-native stream finish (~line 1313):** mirror the same guard. For Ollama-native streams there's no equivalent `in_reasoning_mode` flag, but `'<think' in accumulated_content or '</think' in accumulated_content` reliably detects reasoning-mode output from Ollama-side thinking models.

**Files affected.**

| File | Change |
|---|---|
| `modules/llm/ollama_client.py` | Add `_zero_yield_fallback_message()` helper; add `yield_count == 0` guards in both OpenAI and Ollama-native stream-finish paths |

**Acceptance criteria.**

- Send a non-trivial prompt to an LM Studio qwen3 thinking model with low `loaded_context_length` (~2048). When reasoning eats the budget without reaching an answer, the user-visible UI shows the "produced only reasoning" fallback rather than silence.
- Backend log shows `[OPENAI STREAM] 0 yields after N lines — emitting fallback` (or the Ollama variant).
- Provider-side errors continue to flow through the existing error-handling branch — the fallback only triggers on clean stream completion with zero yields.

**Note on Defect 22 relationship.** Defect 22 catches *pre-stream* overflows (request never sent). This item catches *mid-stream* exhaustion (request sent, stream opens, reasoning consumes budget, stream closes silent). Both are real failure modes for thinking models on small contexts; the pair covers the full envelope.

**Estimated effort.** ~30 LOC + 2-3 tests (one per stream path, exercising the zero-yield branch), ~20 minutes.

---

## Additional items under consideration

The following items were noted during v0.3.3 verification but have not yet been scoped for v0.3.4. They will be promoted, deferred further, or rejected during v0.3.4 planning.

| Item | Origin | Notes |
|---|---|---|
| Frontend "Searching for providers..." spinner for cold-start window | v0.3.3 Defect 17 backend self-heal shipped; frontend UX polish remains | v0.3.3 backend now self-heals via `_rediscover_providers_if_needed` (called from `/api/model/current`). User-visible window between launch-with-no-providers and self-recovery is roughly one poll interval (~3-5 sec). A spinner during that window would tighten UX further; no longer a correctness gap. |
| Cosmetic model description dict at `model_registry.py:312+` (~30 hardcoded model name→description entries) | v0.3.3 §4H audit | Doesn't gate runtime behavior; auto-generation from `/api/show` `parameter_size` and `family` fields possible. Lower priority than runtime-gating hardcodes (already eliminated in v0.3.3). |
| `QUANTIZATION_OPTIONS` and `HUGGINGFACE_REPOS` pre-install browse catalogs | v0.3.3 §4H audit | Used by the install wizard for VRAM-based recommendations before any model is installed. Could be replaced by an Ollama Hub API query in a later release. |
| Frontend curated model recommendation dropdown at `ConnectedChat.tsx:1142+` | v0.3.3 §4H audit | Hardcoded install suggestions, not capability detection. Separate concern from runtime capability gating. |
