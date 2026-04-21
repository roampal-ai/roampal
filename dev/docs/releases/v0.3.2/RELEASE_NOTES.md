# Roampal Desktop v0.3.2 — 2026-04-21

**Release Date:** 2026-04-21
**Theme:** Chat-path performance + customer-reported bug batch + v0.3.1 correctness repairs + test debt cleanup

This release cuts pre-LLM latency on the chat path, fixes a batch of
customer-reported bugs and UX papercuts, repairs several v0.3.1 systems
that were silently broken in production (TagCascade retrieval, memory_bank
filtering, LLM tag extraction), closes out test debt from the v0.3.1 UI
redesign, and clears deprecation warnings surfaced during CI setup.

**Chat-path performance (Section 0).** A beta tester reported v0.3.1 felt
"painfully slow" vs raw Ollama/LM Studio. The cause is entirely in the
pre-LLM memory pipeline — not the model — and this release addresses it
without sacrificing retrieval quality.

**Customer-reported bugs (Sections 0a–0i).** Eight issues from the same
tester: silently-killed streams on tool-incompatible models, a
`.env`-in-Program-Files permission bug that breaks model persistence, a
dropdown that can't distinguish two installed variants of the same model,
ChromaDB telemetry spam, VRAM-aware model picker, sidecar mirror-chat
defaults, and stale-model 404 handling.

**Laptop-testing pass (Sections 0j–0k).** A silent-data-corruption-class
bug where memories shared via a roampal-core MCP install rendered with
the current time and were treated as immortal by the lifecycle (0j), a
fact-extraction crash on small-model output shape, and a boot-order race
that permanently disabled sidecar mirror-chat for upgrading users (0k).

**v0.3.1 correctness repairs (Sections 0l–0p).** The 0l fix (memory-list
endpoint silently dropping tag data) surfaced a cascade of silent-failure
bugs that had been live since v0.3.1 but were masked by each other:
TagCascade's `$contains` where-clause filter erroring on every query and
falling through to unfiltered cosine (0m); a multi-key top-level `where`
filter on memory_bank, masked by 0m's earlier exception (0n); the
TagService wrapper closure going stale after any sidecar swap because it
captured client/model at boot (0o); and `store_memory_bank` calling the
sync `extract_tags()` against the production async `llm_extract_fn`,
silently dropping every AI-created memory_bank entry's auto-extracted
noun_tags (0p).

**Design rule across all fixes.** Prefer universal / capability-detecting
solutions over model-specific enumerations. Keep all runtime-writable
state under `DATA_PATH` (`%APPDATA%\Roampal\data\` on Windows) rather
than the install directory. Log silent-failure exception paths at WARNING
so future regressions surface instead of rotting for weeks.

---

## Scope

### 0. Chat-path latency fixes (new, highest priority)

**Problem reported (2026-04-17):** On weak hardware Desktop v0.3.1 adds ~1.2–2.2 s of
pre-LLM work on steady-state messages and ~3–6 s on message 1 before the chat model
sees the prompt. Raw Ollama/LM Studio skips all of this, hence the "painfully slow"
comparison.

**Scope note on the sidecar.** The sidecar is NOT part of the pre-LLM path for the
*current* message — it fires asynchronously on the NEXT user message and never
blocks the current response. However, on single-GPU single-model setups (the
v0.3.2 default once 0f ships mirror-chat), sidecar LLM calls from the previous
exchange (summary + fact extraction + tag extraction) can still be completing
when the next main-LLM call starts, competing for GPU compute or Ollama's
request queue. This is a *secondary* contributor to perceived slowness
rather than the dominant one measured here, and Section 0e's `keep_alive=24h`
default plus Section 0f's mirror-chat default remove the worst case (model
swap between main and sidecar) without changing the async-not-blocking model.
The fixes below address the measured pre-LLM latency directly; resource
contention with in-flight sidecar work is a known side-effect of running the
memory pipeline and not something we eliminate in this release.

**Verified bottlenecks (audit 2026-04-17):**

| Location | Issue | Added latency |
|---|---|---|
| `app/routers/agent_chat.py:3164` | `await asyncio.sleep(0.5)` hardcoded in `_run_generation_task` to wait for WS connection | 500 ms every message |
| `modules/memory/unified_memory_system.py:1252, 1261` | Two-lane retrieval calls `search()` twice back-to-back; no `asyncio.gather()` | ~500–1000 ms (duplicated sequentially) |
| `modules/memory/search_service.py:603` | `_rerank_with_ce` is a sync `def` — CE ONNX inference blocks the asyncio event loop | no added math, but blocks WS sends + other async work |
| `modules/embedding/embedding_service.py:160` | `embed_text` is `async` in name only; body calls `self._encode()` (sync ONNX) with no `run_in_executor` | blocks event loop |
| `search_service.py:76` `_load_ce()` + `embedding_service.py:108`+ | ONNX models cold-load on first chat message (not at service startup) | +2–4 s on message 1 |

**Fixes (all robustness-preserving, zero quality loss):**

1. **Parallelize the two retrieval lanes** in `UnifiedMemorySystem.get_context_for_injection`
   (`unified_memory_system.py:1252, 1261`) using `asyncio.gather(summary_call, fact_call)`.
   The lanes share no mutable state — `search()` operates on `self.embed_fn` and read-only
   ChromaDB collections. Expected savings: **500–1000 ms per message**.
2. **Warm ONNX models at startup** — kick `_load_ce()` and embedding `_load_model()` in a
   background task during `UnifiedMemorySystem.initialize()`. Eliminates the message-1
   cold-load penalty. Expected savings: **2–4 s on first message**.
3. **Replace the fixed 500 ms sleep** at `agent_chat.py:3164` with a short retry loop
   (poll `app_state.websockets.get(conversation_id)` up to 500 ms, exit early once
   populated). Same safety, typical savings 300–500 ms.
4. **Offload blocking ONNX work** — wrap `_rerank_with_ce` and `_encode` calls in
   `loop.run_in_executor(None, ...)` so WS `send_json` and other async work continue
   during inference. Doesn't speed the math but restores event-loop responsiveness.
   Prevents stalls on WS heartbeat, typing indicator, and parallel tool calls.

**Explicitly NOT doing:**
- Reducing `CE_CANDIDATE_POOL` below 40. Benchmark quality with pool=40 is our
  ceiling and we're not trading that away for latency.
- Adding a "disable CE rerank" settings toggle. CE is load-bearing for the +23pt
  retrieval quality we publish — shipping a fast mode that silently regresses that
  would undercut the benchmark claims. Keep CE on for every user.

**Combined impact (fixes 1–4 applied, CE stays on):**
- Steady-state pre-LLM overhead: ~1.2–2.2 s → **~400–900 ms** (~800 ms saved, driven
  mostly by parallelizing the two CE lanes via `asyncio.gather()`)
- Message-1 pre-LLM overhead: ~3–6 s → **~400–900 ms** (2–4 s saved by warm-start)
- Retrieval quality unchanged — same TagCascade + CE pipeline, just not serialized

**Files to touch:**
- `ui-implementation/src-tauri/backend/app/routers/agent_chat.py` — fix #3 (sleep → ready poll)
- `ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py` — fix #1 (gather) + fix #2 hook at init
- `ui-implementation/src-tauri/backend/modules/memory/search_service.py` — fix #4 (executor for CE)
- `ui-implementation/src-tauri/backend/modules/embedding/embedding_service.py` — fix #4 (executor for encode)

**Tests to add:**
- `test_get_context_for_injection_parallelizes_lanes` — mock `search()`, assert both
  calls are in flight concurrently (track start-time overlap)
- `test_onnx_models_warm_at_init` — assert `_ce_session is not None` and embedding
  `_session is not None` after `initialize()` without any search call
- `test_websocket_ready_poll_exits_early` — assert `_run_generation_task` returns
  before 500 ms when WS already populated

---

### 0a. Universal handling of Ollama 500s (new, customer-reported 2026-04-20)

**Problem reported:** Switching to a Gemma variant (reported as "Gemma4:E2B"),
the UI shows "Thinking…" briefly, then it disappears with no response. Log shows
`POST /api/chat` returning HTTP 500 from Ollama. Same on restart.

**Root cause:** `modules/llm/ollama_client.py:886-917` — the HTTP error handler
only recovers from OOM (checks for `"terminated"` in the body on status 500).
When the model doesn't support Ollama's native tool API, Ollama returns an
HTTP error with a body containing `"does not support tools"`. That body text
isn't matched, so the error re-raises into the outer `except` at line 978,
which logs a traceback and kills the stream. The frontend sees the stream die
with no user-facing message.

**Verified 2026-04-20:** Ollama's `"does not support tools"` response is
reported as **HTTP 400 for most models** (llama3, codellama, qwen3-coder per
upstream issues) but the customer's log shows **500** from Gemma 4. The fix
must be status-code-agnostic — parse the body on *any* 4xx/5xx when tools were
in the payload.

**Verified 2026-04-20:** `gemma4:e2b` is a real Ollama tag published in early
April 2026 (after knowledge cutoff). The customer's "Gemma4:E2B" is an exact
Ollama library identifier for the 26B MoE / 4B-active edge variant. No typo.

**Design constraint:** The model universe is effectively infinite. We do NOT ship
a hardcoded blocklist (current code at `ollama_client.py:627-632` has one — we're
replacing it with capability detection, not expanding it). Any fix that names
specific models — `gemma`, `gemma3`, `embeddinggemma`, etc. — is obsolete on the
day a new model drops.

**Fix (universal, model-agnostic, status-code-agnostic):**

1. **Parse the error body for capability signals on any HTTP error** in
   `ollama_client.py` right where the OOM check runs today (~line 886).
   Restructure the `except httpx.HTTPStatusError` block so it no longer
   narrows to `status_code == 500` first. Instead: read the body once, then
   check in order — (a) `"terminated"` → OOM retry at 2048 ctx (existing
   path); (b) `"does not support tools"` (case-insensitive) AND `tools` in
   payload → remove tools, retry once, log a warning naming the model; (c)
   neither → surface as user-facing error (fix #2). This covers 400, 500,
   and any future status code Ollama uses for this class of failure.
2. **Surface unrecoverable HTTP errors as a user-facing error.** Today the
   outer `except` at line 978 re-raises into agent_chat.py which silently
   terminates the stream. Instead yield a
   `{"type": "text", "content": "**Model error:** <short body>"}` chunk
   before the stream ends, matching the pattern already used for LM Studio
   context errors at `ollama_client.py:767-783`.
3. **Retire the `TOOL_BLOCKLIST` at `ollama_client.py:627-632`.** Once fix #1 is
   in place, the blocklist is redundant — any tool-incompatible model is handled
   generically on first request, at negligible cost (one retry per session).
   Keep embedding-model detection (nomic-embed, all-minilm, bge-) since those
   aren't chat models at all — that's a separate concern from tool support.

**Files to touch:**
- `ui-implementation/src-tauri/backend/modules/llm/ollama_client.py` — fix #1, #2, #3

**Tests to add (to existing `tests/unit/test_ollama_client.py`):**
- `test_tools_not_supported_400_retries_without_tools` — mock 400 with
  `"does not support tools"` body on first call, 200 on second call with no
  tools in payload; assert retry happened, tools absent, stream completes.
- `test_tools_not_supported_500_retries_without_tools` — same as above but
  status code 500 (Gemma 4 case); same universal retry path should fire.
- `test_tools_not_supported_case_insensitive` — mock body
  `"Does Not Support Tools"`; assert match.
- `test_other_http_error_surfaces_user_message` — mock 503 with unknown body;
  assert a `text` chunk with `**Model error:**` prefix is yielded before
  `done`, and stream does NOT raise into agent_chat.
- `test_no_tool_blocklist_for_formerly_blocked_models` — assert a request
  with a previously-blocklisted name (e.g. `dolphin`) still passes tools
  through (the blocklist is gone; capability detection handles it if it
  really fails).

---

### 0b. Main-model persistence parity with sidecar (new, customer-reported 2026-04-20)

**Problem reported:** On restart, Roampal forgets the user's main-model selection
and defaults back to `qwen3:8b`. Sidecar selection IS remembered across restarts.

**Root cause — the install directory is not writable.**

`app/routers/model_switcher.py:1246-1280` writes to `backend/.env`, which lives
next to the packaged code under the install root (typically `C:\Program Files\
Roampal\` or the user-chosen install dir). On default Windows installs that path
requires admin to write to — `env_path.write_text()` silently throws
`PermissionError` under `_env_file_lock`, the exception is swallowed, and nothing
persists. Sidecar works because it writes to `DATA_PATH`
(`config/settings.py:32-39` resolves to `%APPDATA%\Roampal\data\` on Windows,
`~/Library/Application Support/Roampal/data/` on Mac, `~/.local/share/roampal/data/`
on Linux) — all three are per-user writable.

Secondary issue identified but not the primary cause: the Ollama branch at
`model_switcher.py:1262-1267` only rewrites existing `OLLAMA_MODEL=` /
`ROAMPAL_LLM_OLLAMA_MODEL=` lines; the LM Studio branch at 1269-1278 appends
missing ones. The shipped `.env` has the Ollama line commented out. Even if the
file write permission issue were resolved, first-install Ollama users would still
no-op. Fixing the install-dir write is the real fix; the missing-append is moot
once we stop writing to `.env` at all.

On restart, `main.py:395-403` reads `ROAMPAL_LLM_OLLAMA_MODEL` / `OLLAMA_MODEL`,
finds nothing, and falls through to `available_models[0]` — that's where
`qwen3:8b` comes from (alphabetical / Ollama-list order on the tester's machine).

**Architectural rule (new, enforce in CI):** Nothing under the install
directory is writable. All runtime state must live under `DATA_PATH`. Add a CI
grep check that fails if any new code opens a file for write outside
`DATA_PATH` — this bug class will otherwise keep returning.

**Fix (unified with sidecar pattern):**

Replace `.env`-substitution persistence for main-model selection with the same
JSON-file pattern sidecar uses. Add `data_dir/main_model_config.json`:

```json
{"model": "llama3.2:3b", "provider": "ollama"}
```

1. **Write** the file in the `switch_model` endpoint (`model_switcher.py` ~line
   1245, replacing the existing `.env` write block). Use a write-then-rename
   atomic pattern: write to `main_model_config.json.tmp`, then `os.replace()`
   onto the final path so a crash mid-write can't leave a partial file.
   **Also harden `_save_sidecar_config` (`model_switcher.py:2558`)** — it
   currently does a plain `path.write_text(json.dumps(config))` with no temp
   file, which has the same corruption risk. Fix both in this PR; extract
   a shared `_atomic_write_json(path, data)` helper.
2. **Read** the file in `main.py` startup before falling back to
   `available_models[0]` (insert ahead of line 395, between `available_models`
   read and `configured_model` env lookup). If `model_config.json` exists and
   its model is in the detected provider's model list, use it. Otherwise fall
   through to the current env-var / auto-select behavior.
3. **Retire `.env` writes for model selection.** The `ROAMPAL_LLM_PROVIDER` /
   `OLLAMA_MODEL` / `ROAMPAL_LLM_OLLAMA_MODEL` / `ROAMPAL_LLM_LMSTUDIO_MODEL`
   env writes in `model_switcher.py:1237-1280` become dead code once startup
   reads the JSON file. Keep reading env vars at startup (for power users who
   set them in their shell) — env > json > first-available precedence.
4. **Migration:** one-shot read of the old env vars at first startup after
   upgrade — if `model_config.json` is absent but env vars are set, seed the
   JSON file from them. No user action required.

**Files to touch:**
- `ui-implementation/src-tauri/backend/app/routers/model_switcher.py` — fix #1, #3 (write JSON, stop writing .env)
- `ui-implementation/src-tauri/backend/main.py` — fix #2, #4 (read JSON at startup + migration)

**Tests to add:**
- `test_switch_model_persists_to_json` — POST `/switch`, assert
  `data_dir/main_model_config.json` exists and contains the new model+provider.
- `test_startup_reads_main_model_config` — write a `main_model_config.json`,
  start the app with a mocked provider list containing that model, assert
  `app.state.llm_client.model_name` matches the JSON.
- `test_startup_migrates_env_vars_once` — set `OLLAMA_MODEL` env, no JSON file;
  assert startup creates the JSON from env and subsequent switches update the
  JSON (not env).
- `test_startup_precedence_env_over_json_over_fallback` — explicit precedence
  test with all three sources varying.

---

### 0c. Stale model → 404 surfaced as actionable UI error (new, customer-reported 2026-04-20)

**Problem:** If a user runs `ollama rm <model>` outside Roampal while Roampal
still has that model as the active selection, the next chat message hits a 404
from Ollama. Today this dies silently the same way the tools-unsupported 500
does in 0a — the frontend just sees the stream die.

**Fix:** Pair with 0a in `ollama_client.py` at the same 500/404 branch point.
When Ollama returns 404 on `/api/chat`, yield:

```
{"type": "text",
 "content": "**Model '<actual_model>' is no longer installed.**
            Open Settings → Model and pick a new one."}
```

Then clear `self.model_name` to empty string and set
`app.state.llm_client.model_name = ""` so the UI's "currently selected model"
reads blank until the user picks again. Also unset the persisted
`main_model_config.json` (from 0b) so restart doesn't re-select a dead model.

**Tests to add:**
- `test_ollama_404_yields_user_message_and_clears_model` — mock 404 on
  `/api/chat`, assert text chunk with "no longer installed", assert
  `llm_client.model_name == ""` after the call, assert
  `main_model_config.json` is gone.

---

### 0d. ChromaDB telemetry spam (new, customer-reported 2026-04-20)

**Problem:** Log files bloat (3.8MB+ in a single session per tester) from
repeated `capture() takes N args, M given` tracebacks emitted by Chroma's
PostHog-based anonymized telemetry. Cosmetic but degrades log readability and
wastes disk.

**Fix:** Pass `chromadb.Settings(anonymized_telemetry=False)` wherever the
Chroma client/collection is constructed. Search:

```
grep -rn "chromadb.Client\|PersistentClient\|HttpClient" backend/
```

Update each construction site. One-line change per call site.

**Tests to add:**
- `test_chromadb_client_has_telemetry_disabled` — import whatever factory
  builds the client, assert the `Settings` object has
  `anonymized_telemetry=False`.

---

### 0e. Ollama `keep_alive` per-request (new, customer-reported 2026-04-20)

**Problem:** Ollama's default `keep_alive` is 5 minutes — models unload after
5 min idle. For a desktop user chatting intermittently, next message pays a
full model-reload penalty (~10–30s on 8B+). Users report "slow" when the real
issue is warm-reload, not inference.

**Fix:** Send `keep_alive` in the request payload (not a launcher env var —
payload always wins and avoids launcher-coordination bugs). Add to the payload
builders in `modules/llm/ollama_client.py`:

```python
payload["keep_alive"] = settings.llm.ollama_keep_alive  # default "24h"
```

Rename the setting at `config/settings.py:76` from
`ollama_keep_alive_seconds: int = 120` to `ollama_keep_alive: str = "24h"`
(Ollama accepts duration strings; pydantic rejects a string into an `int`
field, so the rename is load-bearing). Update any reader of the old name
to use the new one. Use `"24h"` not `-1` so VRAM-constrained users who run
other GPU apps recover after a full day idle, instead of holding VRAM forever.

**Files to touch:**
- `ui-implementation/src-tauri/backend/config/settings.py` — default
- `ui-implementation/src-tauri/backend/modules/llm/ollama_client.py` — payload

**Tests to add:**
- `test_payload_includes_keep_alive` — assert every outgoing `/api/chat` and
  `/api/generate` payload has `keep_alive` set to the configured value.
- `test_keep_alive_default_is_24h` — assert settings default is `"24h"`.

---

### 0f. Sidecar defaults to chat model (new, customer-reported 2026-04-20)

**Problem reported:** Tester observed "same model for both sidecar and main does
seem a bit quicker." Root cause: on single-GPU systems, running different
models forces Ollama to swap them on/off GPU = 10–30s swap every time sidecar
fires. Two identical models = no swap. 90% of users don't know they could set
this and don't want to think about it.

**Fix (UX + defaults, not a code change in behavior):**

1. When no `sidecar_config.json` exists (first run or post-0b migration),
   seed sidecar config to mirror the current chat model (`enabled: true`,
   `model: <chat_model>`, `provider: <chat_provider>`).
2. When the user changes the chat model via `/switch`, if the sidecar is in
   "mirror chat" mode, update sidecar to match atomically (same request).
3. Add a "Use chat model for sidecar" toggle **inside a collapsed "Advanced"
   disclosure** in Settings, default ON. When ON, sidecar model picker is
   hidden (reduces UI clutter for the 90%). When OFF, the picker appears
   under the same Advanced section so power users control it independently.
4. Persist the mirror-mode choice in `sidecar_config.json`:
   `{"enabled": true, "model": "...", "provider": "...", "mirror_chat": true}`.
5. **Reload-warning modal** — when the user toggles mirror-mode OFF and
   picks a sidecar model different from the chat model, show a confirmation
   modal before saving:

   > **Running two models on one GPU**
   > Your sidecar will use a different model than chat. On a single GPU each
   > message will briefly unload one model and load the other (typically 10–30s
   > the first time, faster while both stay warm). This is a limitation of
   > local model hosts, not Roampal.
   > [Cancel] [Use different sidecar model]

   Skip the modal when mirror-mode is ON, when both picks are the same model,
   or when the user has a multi-GPU system detected in 0g
   (`app.state.gpu_count > 1`).

   **Verified 2026-04-20** — this reload behavior applies to both providers:
   - **Ollama:** default `keep_alive` 5 min (our 0e sets 24h); LRU eviction
     when VRAM fills on model swap. Source: Ollama FAQ / keep-alive docs.
   - **LM Studio:** JIT loading with 60-min default TTL and **Auto-Evict**
     that unloads the previously JIT-loaded model before loading a new one.
     Source: LM Studio "Idle TTL and Auto-Evict" docs.

   Same failure mode on both — the modal wording is provider-agnostic on
   purpose.

**Non-goal:** Auto-detecting whether the user's VRAM could hold two distinct
models simultaneously. That's 0g — do the simple thing here, let VRAM-aware
logic refine the default later.

6. **Chat-header sidecar picker → read-only badge.** The writable sidecar
   dropdown that lived at the top of the chat in v0.3.1 is demoted to a
   read-only status badge. It still shows what's running (`Mirror: qwen3:8b`
   when mirror-chat is on, the custom model name when it's off), but all
   *configuration* goes through Settings → Advanced — one source of truth,
   no UI that can disagree with itself.
   - Clicking the badge opens Settings with the Advanced disclosure
     pre-expanded (via a new `initialFocus="advanced"` prop on
     `SettingsModal`), so users still reach the picker in one click.
   - `/sidecar/status` now returns `mirror_chat` alongside `model/provider`
     so the badge can label itself correctly without an extra round-trip.
   - The "Set Up" action on the no-sidecar toast points to the same
     Advanced panel instead of the retired dropdown.

7. **Separate-but-equal error toasts for chat + sidecar.** Before v0.3.2
   sidecar failures were only passively indicated on the chat-header badge
   (red pulse + tooltip) — easy to miss if LM Studio went down or an Ollama
   service restarted. Main-LLM errors surfaced inline in the chat, but that
   asymmetry meant users got different visibility depending on which model
   broke. v0.3.2 adds two toasts with the same urgency model:
   - **Chat LLM toast** — fires when the assistant message contains the
     `**Model error:**` or `no longer installed` markers emitted by the
     universal Ollama error handler (0a/0c). Deduped by message id. Action
     button opens Settings.
   - **Sidecar toast** — fires on transition from clean → error so a
     persistent outage doesn't re-fire every 30s poll. Action button opens
     Settings → Advanced so the user can switch sidecar model or toggle
     mirror-chat in one click.
   The inline chat error chunk stays as a secondary surface (doesn't
   vanish when the toast auto-dismisses).

8. **Sidecar badge is a read-only status indicator, not a button.** Earlier
   iteration made the badge clickable and opened Settings → Advanced. Logan
   flagged this as the wrong affordance — the badge should read as *status*,
   not a control. Updated to a muted `<div role="status">` with `cursor-default`,
   no hover state, no click handler; the tooltip still names the sidecar model
   and points users to Settings → Advanced if they want to change it. This
   keeps Advanced Settings as the one configuration surface and stops the
   badge from feeling like a shortcut into the config path.

**Files to touch (0f + items 6–8):**
- `ui-implementation/src-tauri/backend/app/routers/model_switcher.py` — mirror
  logic in `switch_model` when `mirror_chat` is true; `/sidecar/status`
  exposes `mirror_chat`; new `/sidecar/mirror` toggle endpoint
- `ui-implementation/src-tauri/backend/main.py` — seed sidecar on first run
- `ui-implementation/src/components/Settings*.tsx` — Advanced disclosure
  containing the mirror toggle + independent sidecar picker; new
  `initialFocus` prop
- `ui-implementation/src/components/SidecarReloadWarningModal.tsx` (new) —
  the confirmation modal described above; wired to the Advanced sidecar
  picker's save action
- `ui-implementation/src/components/ConnectedChat.tsx` — retire the writable
  sidecar dropdown; replace with the read-only badge that deep-links to
  Settings → Advanced. Removes `showSidecarDropdown` state,
  `getSidecarModelOptions` helper, and the click-outside handler since
  they're dead code without the dropdown.

**Tests to add (0f + items 6–8):**
- `test_sidecar_seeded_to_chat_model_on_first_run` — no config file, start
  app with chat model `qwen3:8b`, assert sidecar config written with same
  model + `mirror_chat: true`.
- `test_switch_chat_model_updates_mirrored_sidecar` — POST `/switch` to a
  new model, assert sidecar config now matches.
- `test_switch_chat_model_does_not_touch_unmirrored_sidecar` —
  `mirror_chat: false`, switch chat, assert sidecar unchanged.
- `test_advanced_settings_hidden_by_default` (frontend) — mount Settings,
  assert the sidecar picker + mirror toggle are not visible until the
  Advanced disclosure is expanded.
- `test_reload_warning_shown_on_divergent_sidecar_pick` (frontend) —
  simulate toggling mirror OFF and selecting a model != chat model,
  assert the reload-warning modal appears with both Cancel and confirm
  buttons; Cancel reverts selection, confirm persists it.
- `test_reload_warning_skipped_when_models_match` (frontend) — same flow
  but sidecar pick == chat model, assert no modal.
- `test_sidecar_status_exposes_mirror_chat` (backend) — `/sidecar/status`
  response must include `mirror_chat` so the chat-header badge can render
  the correct label without extra round-trips.
- `test_advanced_auto_expands_with_initial_focus` (frontend) — render
  `SettingsModal` with `initialFocus="advanced"`, assert the Advanced
  panel is open and the mirror toggle is visible immediately.

9. **Fact dedup on store (parity with roampal-labs benchmarks).** Desktop
   was the only Roampal build storing every extracted atomic fact raw —
   semantically identical facts produced separate ChromaDB entries and
   bloated the memory store over time. v0.3.2 ports the dedup pattern used
   by the roampal-labs retrieval strategies (same threshold across
   Wilson-scored, semantic reranker, Wilson+CE, TagCascade / entity_routed,
   and CE-lifecycle). Two write surfaces are guarded:
   - `UnifiedMemorySystem.store()` (lines 658–708) — the sidecar-driven
     fact-extraction path. Only fires when `metadata.memory_type == "fact"`
     so summaries and other types still store normally.
   - `UnifiedMemorySystem.store_memory_bank()` (lines 863+) — the
     AI-facing identity / preference / learned-fact surface (MCP
     `add_to_memory_bank`, direct API writes). All writes here are
     fact-shaped by construction, so the gate runs unconditionally.
     Embedding is computed once in `store_memory_bank` and threaded
     through to `MemoryBankService.store` via a new optional `embedding`
     param — no double-embed cost.

   Both sites share `_find_duplicate_fact` (line 630), which takes a
   `tiers` kwarg for **asymmetric scope by write persistence**:
   - `store()` fact branch → `tiers=None` (default) → scans all 4 tiers
     (`working`, `history`, `patterns`, `memory_bank`). Ephemeral
     sidecar writes defer to any more-persistent existing copy.
   - `store_memory_bank` → `tiers=("memory_bank",)` → scans memory_bank
     only. A permanent-persistence write must never be blocked by an
     ephemeral working-tier copy — if it were, promoting a
     chat-extracted fact to permanent memory would be absorbed and
     silently lost when working rolls over at 24h (the user's name /
     identity / preferences disappearing is exactly the failure mode
     this asymmetry prevents).

   Shared mechanics:
   - Per-tier filter (`FACT_DEDUP_FILTERS`): `memory_type="fact"` for
     working / history / patterns (those tiers are mixed-content); no
     filter for memory_bank (that collection is all facts by
     construction and doesn't tag rows with `memory_type`).
   - If top cosine distance < 0.32 in L2 space (= `cos_sim > 0.95` for
     unit-normalized 768-d embeddings), return the existing duplicate's
     id and skip the write.
   - Logs every skip at `[DEDUP] Skipped storing fact — duplicate of <id>`
     or `[DEDUP] Skipped storing memory_bank fact — duplicate of <id>`.
   - Best-effort: any exception in the query path swallows and falls
     through to normal store, so dedup can never block a legitimate write.

   Age-out rather than cleanup: when `store_memory_bank` writes despite
   a working-tier dup existing, the ephemeral copy is **not** deleted.
   It ages out naturally at the working-tier TTL. In the interim
   memory_bank's `score: 1.0` row dominates retrieval ranking, and
   TagCascade / CE top-K selection already dedup-handles whichever row
   scores highest, so the overlap window is harmless.

   The worked example that drove the asymmetric scope:
   1. User says "Hi, I'm George." Sidecar extracts the fact into
      `working` (24h TTL, `score: 0.5`, decay-eligible).
   2. AI decides to persist this permanently — calls
      `add_to_memory_bank("User's name is George")`.
   3. **Pre-asymmetry (wrong):** dedup scans all tiers, finds the
      working copy, returns its id, skips the memory_bank write.
      24h later, working rolls over; unless the fact was retrieved
      enough to promote to history, it's gone. Name is lost.
   4. **With asymmetry (current):** `store_memory_bank` scans only
      memory_bank, finds no dup, writes permanently. Working copy
      still ages out, but the permanent copy survives.

---

### 0g. VRAM-aware model picker (new, customer-reported 2026-04-20)

**Problem:** Users download models that don't fit their GPU and then complain
the app is slow. No warning surfaces until the model is pulled (gigabytes of
bandwidth) and loaded (OOM or painful CPU offload). Tester hit this personally.

**Fix (two-tier, universal-first per the feedback rule):**

1. **Detect GPU VRAM at startup** once and cache on `app.state.gpu_vram_gb`:
   - Windows/Linux NVIDIA: `nvidia-smi --query-gpu=memory.total --format=csv,noheader`
     (parse MiB → GB). Already a subprocess pattern used elsewhere in
     `model_switcher.py` for `ollama list`.
   - Windows/Linux AMD: try `rocm-smi --showmeminfo vram --csv` if available;
     silently skip if not present.
   - Mac: treat unified memory as shared — read total RAM, assume ~75% usable
     by Metal. `sysctl hw.memsize` / `ps -A -o rss` are both cheap.
   - Fallback: if none of the above work, set `gpu_vram_gb = None` and
     disable the warning entirely (do not guess).
2. **Check at model-pick / switch time:**
   - For models in `app/routers/model_registry.py` curated list (already has
     `vram_gb` per entry): compare against `gpu_vram_gb`. If the selected
     quant won't fit, show a warning before download/switch. Don't block —
     users may know their setup better than us.
   - For arbitrary models (anything not in the curated list): parse the
     parameter count from the tag (`qwen3:72b` → 72B), estimate
     `vram_gb ≈ params_b × 0.75` (Q4_K_M rule of thumb), warn on mismatch.
     Label the warning as "estimated" so the user knows it's a heuristic.
   - If the tag has no parameter marker (e.g. `some-org/custom-model:latest`):
     no warning. Silent pass — universal rule: don't guess wrong, just don't
     guess.

**Files to touch:**
- `ui-implementation/src-tauri/backend/main.py` — detect VRAM at startup
- `ui-implementation/src-tauri/backend/app/routers/model_switcher.py` —
  warning in `/switch` response and in the model-list endpoint
- `ui-implementation/src/components/ConnectedChat.tsx` — **wire an
  info-type toast that reads `response.vram_warning.message` from the
  `/switch` response** (ship-fix: earlier backend-only build left this
  field unconsumed; Logan caught it during manual QA and flagged "tests
  pass ≠ feature shipped"). Toast action "Open Model Settings" deep-links
  the user into the context settings panel.

**Tests to add:**
- `test_vram_detection_nvidia` — mock subprocess, assert parsed GB value.
- `test_vram_detection_fallback_silent` — all detection paths fail, assert
  `gpu_vram_gb is None` and no warning is ever emitted.
- `test_curated_model_vram_warning` — pick a curated model whose `vram_gb`
  exceeds mocked GPU, assert warning in response.
- `test_arbitrary_model_param_estimate` — pick `qwen3:72b`, assert
  `estimated_vram_gb ≈ 54` and warning fires on 16GB GPU.
- `test_unknown_param_tag_no_warning` — pick `some-org/foo:latest`, assert
  no warning (silent).

---

### 0h. Model dropdown shows full tag, not collapsed name (new, customer-reported 2026-04-20)

**Problem reported:** Customer has two Gemma 4 variants installed (`gemma4:26b`
and `gemma4:e2b`). The dropdown displays both as just `gemma4` with the
subtitle `Custom model · Ollama` — identical entries, impossible to tell apart.
Ollama model IDs are always `name:tag`; collapsing to just `name` is lossy.
Also, the "Custom model" subtitle is misleading — these are standard Ollama
models. "Custom" implies user-defined; used as a fallback when our curated
registry doesn't know the model. It shouldn't be surfaced to the user at all.

**Fix:**

1. **Show the full Ollama model ID in the dropdown row title** — `gemma4:26b`,
   not `gemma4`. If name is `org/model:tag` (HuggingFace-style), show as-is.
   Find the display logic in the model-list component (likely
   `ui-implementation/src/components/ModelPicker.tsx` or similar — grep for
   "Custom model").
2. **Replace the "Custom model" subtitle** with meaningful metadata, in
   priority order:
   a. If the model is in the curated registry: use its `description` /
      `quality_label` (e.g. "Qwen3 8B · 6GB VRAM · Q4_K_M").
   b. If not curated but tag parses into a parameter count:
      show "≈ <N>B parameters · Ollama".
   c. Otherwise: show only the provider (`Ollama` or `LM Studio`). No
      "Custom" label.

**Files to touch:**
- `ui-implementation/src/components/` — whichever file renders the dropdown row
  (grep `"Custom model"`)
- `ui-implementation/src-tauri/backend/app/routers/model_switcher.py` — if the
  display string is built server-side, add full tag + metadata fields to the
  `/api/model/providers/all/models` response.

**Tests to add:**
- `test_dropdown_shows_full_tag` — render with two installed variants
  `gemma4:26b` and `gemma4:e2b`, assert both tags appear in the DOM.
- `test_dropdown_no_custom_model_label` — render with any non-curated model,
  assert the string "Custom model" is not present in the output.
- `test_dropdown_curated_model_shows_metadata` — render with `qwen3:8b`
  (curated), assert the curated description appears in the subtitle.

---

### 0i. ChromaDB "Error finding id" — investigate only (new, customer-reported 2026-04-20)

**Problem reported:** Intermittent errors like `Error executing plan:
Internal error: Error finding id` on `memory_bank` queries. Retries usually
succeed, but some UI fetches may fail-open with empty memory results
(silently showing "no memories" when there are some).

**Hypotheses (ranked by likelihood):**

1. **Concurrent access.** The Roampal desktop app and the `pip install roampal`
   CLI can both open the same embedded ChromaDB directory. ChromaDB's embedded
   mode is single-writer; concurrent writes corrupt the index intermittently.
2. **Index corruption from an earlier crash.** Crash mid-write leaves partial
   state; subsequent reads hit "Error finding id" on the orphaned row.
3. **Upstream Chroma bug.** Less likely given the error text suggests a known
   class of HNSW / duckdb issues, but possible.

**Scope for v0.3.2 — investigate only, ship fix in v0.3.3 unless trivial:**

1. Add structured logging at every ChromaDB call site that catches this
   exception — log the query, collection, and pid so we can tell same-process
   vs cross-process.
2. Ask the tester to repro with `roampal --cli` running alongside the desktop
   app; log pids from both sides. If cross-process access is confirmed, that's
   the cause.
3. Ship a `roampal repair` CLI subcommand that runs `chromadb.utils.index
   .validate()` (or equivalent) and rebuilds from raw documents if corruption
   is detected.

**Deferred to v0.3.3 if investigation points that way:**
- Single-writer lockfile at `DATA_PATH/chromadb.lock`. On acquire-fail, second
  process runs in read-only mode or exits with a clear message.
- Alternatively: move Chroma into client/server mode inside the desktop app,
  so all processes attach via HTTP. Matches what's already scaffolded behind
  `CHROMADB_USE_SERVER=true` in `.env`.

**Files to touch (0.3.2 investigation scope):**
- `ui-implementation/src-tauri/backend/modules/memory/*.py` — add logging at
  ChromaDB call sites.
- `ui-implementation/src-tauri/backend/cli/` (new) — `roampal repair`
  subcommand if feasible in this release; otherwise defer.

---

### 0j. Shared-DB timestamp field drift — display + lifecycle (new, customer-reported 2026-04-21)

**Problem reported (Logan, laptop testing v0.3.2 2026-04-21):** Some working and
history memories in the Memory Panel "constantly show the current time and date
like they're auto-updating." The displayed time ticks forward on every refresh.
Only *some* memories are affected — the rest render the correct creation time.

**Root cause — field drift between roampal-desktop and roampal-core sharing
the same ChromaDB.** Logan runs the `pip install roampal` MCP server pointed at
the same Chroma directory as the desktop app. The two tools write the creation
timestamp under *different field names*:

- Desktop `unified_memory_system.py:686` writes `metadata["timestamp"]`
- Core `unified_memory_system.py:1538` (`store_working`) writes
  `metadata["created_at"]`

The desktop's Memory Panel flatten at `memory_visualization_enhanced.py:137`
only read `timestamp` / `upload_timestamp`, so core-written memories returned
with `timestamp: null`. The frontend at `ConnectedChat.tsx:1790-1792` then
fell back to `new Date()` — literally "right now" — on every `fetchMemories`
call. Panel refresh re-hydrated with a fresh `new Date()`, producing the
ever-incrementing timestamp Logan observed.

**This was not just a UI issue — lifecycle was equally broken.** Five desktop
sites read `metadata["timestamp"]` without a `created_at` fallback:

| Site | Effect on core-written memories |
|---|---|
| `promotion_service.py:278-281` (deletion age) | `age_days = 0` → always gets the lenient new-item deletion threshold |
| `promotion_service.py:343` (working→history promotion) | age ignored for gate (benign — promotion gates on score+uses, not age) |
| `promotion_service.py:479` (working TTL cleanup) | `age_hours = 0` → **never exceeds max_age → never cleaned up** |
| `promotion_service.py:523` (history 30-day expiry) | Same — **core-written history memories live forever** |
| `search_service.py:580` (recency metadata "X min ago") | Silently skipped |

So for however long Logan's been running the shared-DB setup, core-written
working and history memories have been **effectively immortal** on the
desktop's lifecycle, and also sinking to the bottom of every "recent first"
sort (`memory_visualization_enhanced.py:101`, `search_service.py:449`).

**Design choice — tolerance, not unification.** Desktop keeps writing
`timestamp`, core keeps writing `created_at`. Every *read* site in desktop
now tolerates both. Zero data migration, zero write-side churn, backward-
compatible with all existing memories (including any written by previous
versions). Matches the pattern core already uses at most of its own read
sites (`promotion_service.py:368, 519, 569`; `search_service.py:642`;
`unified_memory_system.py:184, 668-669, 746`).

**Fix:**

1. **Backend flatten (display + sort):**
   - `memory_visualization_enhanced.py:137` — flatten reads
     `metadata.get("timestamp") or metadata.get("upload_timestamp") or metadata.get("created_at")`.
   - `memory_visualization_enhanced.py:101` + `search_service.py:449` — sort
     keys use the same fallback chain so core-written memories sort correctly
     instead of landing at the bottom as empty strings.
2. **Backend lifecycle:**
   - `promotion_service.py:278-281` (deletion age gate),
     `promotion_service.py:343` (promotion),
     `promotion_service.py:479` (working cleanup),
     `promotion_service.py:523` (history cleanup) — all read
     `metadata.get("timestamp") or metadata.get("created_at", "")` before
     calling `_calculate_age_hours`.
   - `search_service.py:580` — recency metadata ("X minutes ago") uses the
     same fallback so core-written memories get a readable recency label.
3. **Frontend:**
   - `ConnectedChat.tsx:1790-1792` — stop defaulting to `new Date()`. Fall
     back to `null` when no timestamp field is present in either location;
     existing "Unknown" render at `MemoryPanelV2.tsx:438, 552` handles the
     null case. This is defense-in-depth: even if a future write path misses
     both fields, the panel will never misleadingly render "now."

**Files touched:**
- `ui-implementation/src-tauri/backend/app/routers/memory_visualization_enhanced.py`
- `ui-implementation/src-tauri/backend/modules/memory/promotion_service.py`
- `ui-implementation/src-tauri/backend/modules/memory/search_service.py`
- `ui-implementation/src/components/ConnectedChat.tsx`

**Tests to add:**
- `test_flatten_reads_created_at_when_timestamp_absent` — seed a working item
  with only `metadata.created_at`, GET `/api/memory/enhanced/collections/working`,
  assert response `timestamp` equals the `created_at` value.
- `test_sort_mixes_timestamp_and_created_at_consistently` — seed two items,
  one with `timestamp="2026-04-20T..."`, one with `created_at="2026-04-21T..."`,
  assert the `created_at` item sorts first (most recent).
- `test_cleanup_old_working_reads_created_at` — seed a working item 48h old
  via `created_at` only, run `cleanup_old_working(max_age_hours=24)`, assert
  it was deleted (previously would have been preserved as `age_hours=0`).
- `test_cleanup_old_history_reads_created_at` — same pattern, 40-day-old item
  via `created_at` only, assert deleted by 30-day `cleanup_old_history`.
- `test_memory_panel_renders_unknown_when_no_timestamp` (frontend) — render
  MemoryPanelV2 with a memory that has `timestamp: null`, assert "Unknown"
  appears (not a localized current-time string).

**Coordination:** Core's v0.5.2 ships the mirror one-line fix for
`promotion_service.py:298` (the sole core site that still read `timestamp`
only). All other core read paths were already tolerant.

---

### 0k. Sidecar fact-extraction crash + mirror_chat boot-race (new, laptop-testing 2026-04-21)

Two defects surfaced during Logan's laptop smoke-test of the v0.3.2 build.
Both are narrow, both have one-line-class fixes, both shipped into 0.3.2.

#### Bug 1 — `extract_facts` AND `extract_noun_tags` crash on bare JSON array from small models

**Symptom:** `[SIDECAR] Task failed extract_facts_None: 'list' object has no
attribute 'get'`. Fact/tag extraction silently no-ops after the first crash
for that exchange; sidecar retry queue eventually gives up.

**Root cause:** `modules/memory/sidecar_service.py:421` (facts) and `:326`
(tags):
```python
raw_facts = result.get("facts", [])  # facts lane
tags = result.get("tags")            # tags lane — same bug class
```
The prompts explicitly instruct the model to respond with
`{"facts": [...]}` / `{"tags": [...]}`, but small quantized models
(reported from `qwen2.5:3b`; reproducible on similar 3B-class checkpoints)
sometimes emit the raw array `["fact 1", "fact 2"]` / `["tag1", "tag2"]`
instead. `_extract_json` parses that to a Python list, `.get()` then
raises `AttributeError`. Post-0.3.1 we started firing fact + tag
extraction on every scored exchange, so this class of model silently
kills both lanes on every message.

The tag-extraction version is especially quiet — TagCascade retrieval
keeps working, it just runs with empty tag lists for every memory stored
by that sidecar, silently degrading to cosine + CE only (no tag-prefilter
prefix step).

**Fix:** accept either shape at both call sites:
```python
raw_facts = result if isinstance(result, list) else result.get("facts", [])
tags = result if isinstance(result, list) else result.get("tags")
```
Existing `isinstance(..., list)` guard on the next line in each path
already rejects truly pathological shapes (string, dict, etc.), so the
cleaning pipeline downstream is unchanged.

**Files touched:**
- `ui-implementation/src-tauri/backend/modules/memory/sidecar_service.py` — lines 326, 421

**Tests added:**
- `test_bare_array_from_small_model` in `tests/unit/test_sidecar_service.py`
  — one in `TestExtractFacts`, one in `TestExtractNounTags`. Each asserts
  the function returns the expected list when the LLM returns the bare-array
  shape instead of the schema-wrapped form. Pre-fix: raises
  `AttributeError`. Post-fix: passes.

**Core parity:** roampal-core has the identical pattern in three places
(`roampal/sidecar_service.py:908, 968, 1035` — tags, facts, and the
diagnostic validator). All three are documented in core v0.5.3 release
notes for a follow-up release. Desktop ships the fix in v0.3.2; core
ships in v0.5.3.

#### Bug 2 — Boot-order race permanently disables mirror_chat for upgrading users

**Symptom:** log line after v0.3.1 → v0.3.2 upgrade:
```
✓ Sidecar config migrated: mirror_chat=False (sidecar=qwen2.5:3b, chat=)
```
— note the empty `chat=`. User's Advanced Settings checkbox reads OFF, and
no amount of restarts flips it back on. User must manually toggle the
checkbox to recover the intended state.

**Root cause:** `main.py:556-565`:
```python
if "mirror_chat" not in existing:
    chat_model = getattr(app.state.llm_client, "model_name", "") if app.state.llm_client else ""
    matches = bool(chat_model) and existing.get("model") == chat_model
    existing["mirror_chat"] = matches
    ...
```
If the chat LLM client hasn't finished loading its model yet (e.g.
provider-fallback in progress, Ollama cold-loading a large model,
auto-select still resolving), `chat_model == ""`. `bool("")` is False, so
`matches` is False regardless of truth, and `mirror_chat=False` is
persisted to `sidecar_config.json`.

The boot-realign block at `main.py:581` is gated on `if sc.get("mirror_chat") and ...`,
so once the migration has locked `mirror_chat=False`, the realign path
never runs and can't self-correct on a subsequent boot when the chat model
is populated.

**Fix (defer migration when chat_model is empty):**
```python
if "mirror_chat" not in existing:
    chat_model = getattr(app.state.llm_client, "model_name", "") ...
    if not chat_model:
        logger.info("Sidecar mirror_chat migration deferred: chat model not loaded yet")
    else:
        matches = existing.get("model") == chat_model
        existing["mirror_chat"] = matches
        _atomic_write_json(sidecar_config_path, existing)
        ...
```

Considered and rejected:
- **Default True when chat_model empty** (`matches = not chat_model or ...`):
  would silently flip users who legitimately had distinct sidecar+chat
  models into mirror mode; subsequent realign would then overwrite their
  sidecar to match chat, destroying their preference. Bad tradeoff.
- **Drop the `mirror_chat` gate on the realign block**: would cause realign
  to run regardless, overwriting any manual user toggle to
  `mirror_chat=False` on every boot. Bad tradeoff.

Deferring is safe because:
- Leaving `mirror_chat` absent is equivalent to `mirror_chat=False` for all
  code that reads it via `.get("mirror_chat")` (all readers treat missing
  as falsy).
- The realign block (still gated on `sc.get("mirror_chat")`) no-ops in the
  deferred window, so no stale-sidecar correction happens until the state
  is knowable.
- Next boot where `chat_model` is populated, migration runs correctly and
  `mirror_chat` is persisted with the right value.

**Files touched:**
- `ui-implementation/src-tauri/backend/main.py` — lines ~550-568

**Tests to add (deferred — startup-hook code path, harder to unit test):**
- `test_migration_deferred_when_chat_model_empty` — mock
  `app.state.llm_client = None` or `.model_name = ""`, seed a
  `sidecar_config.json` without `mirror_chat`, run the migration block,
  assert `mirror_chat` is still absent and log line contains "deferred".
- `test_migration_runs_when_chat_model_populated` — regression: same seed,
  but `chat_model="qwen2.5:3b"` matching sidecar model, assert
  `mirror_chat=True` is persisted.

---

### 0l. Memory list endpoint doesn't expose `tags` (new, laptop-testing 2026-04-21)

`ui-implementation/src-tauri/backend/app/routers/memory_visualization_enhanced.py:136-151`

**Symptom.** Memory Panel renders rows with no tag pills even when the
underlying rows actually have tags. Reproduced on laptop for memories
created both in Desktop and through the shared core MCP server.

**Root cause.** The `/api/memory/collections/{collection_type}` endpoint
(also aliased as `/enhanced/collections/{collection_type}`) flattens
`score`, `uses`, `collection`, and `timestamp` to top-level fields on the
response memory dict, but leaves `noun_tags` buried inside `metadata`.
`noun_tags` is written by the sidecar as a **JSON-encoded string** (see
`agent_chat.py` — `summary_meta["noun_tags"] = json.dumps(noun_tags)`),
not a JSON array. The frontend (`MemoryPanelV2.tsx`, 8+ sites including
`memory.tags`, `memory.tags.filter(...)`, `memory.tags.slice(...)`) reads
a top-level `tags` array and does **not** parse a JSON string out of
`metadata.noun_tags`, so it always sees `undefined` → renders nothing.
Tags are in the data; the transport layer is dropping them.

```python
# Before (no tags surface — UI sees metadata.noun_tags as opaque string)
memory = {
    'id': ...,
    'content': ...,
    'metadata': metadata,
    'score': ...,
    'uses': ...,
    'collection': collection_type,
    'timestamp': (
        metadata.get('timestamp')
        or metadata.get('upload_timestamp')
        or metadata.get('created_at')
    ),
}

# After (parse + flatten, tolerating either JSON string or already-parsed list)
raw_tags = metadata.get('noun_tags')
if isinstance(raw_tags, list):
    memory['tags'] = raw_tags
elif isinstance(raw_tags, str) and raw_tags:
    try:
        parsed = json.loads(raw_tags)
        memory['tags'] = parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        memory['tags'] = []
else:
    memory['tags'] = []
```

Also adds `import json` at the top of the file.

**Why backend parse, not frontend parse.** Putting the JSON parse in the
endpoint keeps the shape consistent with the rest of the flattened
fields (`score`, `uses`, `timestamp`) — the frontend already expects a
clean value for those, so expecting one for `tags` matches. It also
means any future consumer of this endpoint (tests, scripts, an eventual
web UI) gets a usable array without having to know the storage format.
No UI code change needed.

**Impact if unfixed.** Memory Panel looks tag-less on new installs even
though TagCascade retrieval is working correctly underneath — users
can't see what tags their memories carry, can't verify tag extraction
ran, and filtering-by-tag from the UI (v0.3.1 Substack-style picker)
has nothing to show in its typeahead.

**Scope note.** Core does not have an equivalent visualization endpoint
(it's an MCP server; no UI), so this is a desktop-only fix. No v0.5.x
coordination entry needed.

**Tests to add:**
- `test_collection_memories_flattens_noun_tags` — seed a working-memory
  row with `metadata={"noun_tags": json.dumps(["calvin", "boston"])}`,
  hit the endpoint, assert response `memories[0]["tags"] == ["calvin",
  "boston"]`.
- `test_collection_memories_handles_list_noun_tags` — seed with
  `noun_tags` already as a list (shouldn't happen in production but is
  the forgiving path); assert the same response shape.
- `test_collection_memories_handles_missing_or_malformed_noun_tags` —
  seed with no `noun_tags` key, empty string, and an invalid JSON
  string (`"not json at all"`); assert `memories[0]["tags"] == []` in
  each case, no 500.

---

### 0m. TagCascade `$contains` where-clause silently broken since v0.3.1 (new, laptop-testing 2026-04-21)

`ui-implementation/src-tauri/backend/modules/memory/search_service.py:315`

**Symptom.** Backend log shows 8 × `[ChromaDB] Query failed: Expected
where operator to be one of $gt, $gte, $lt, $lte, $ne, $eq, $in, $nin,
got $contains in query.` per TagCascade query. The TagCascade retrieval
path claims to tag-filter candidates before cosine ranking, but the
filter has been erroring out silently on every query since the v0.3.1
landing.

**Root cause.** TagCascade's tag-prefilter uses:

```python
tag_filter = {"noun_tags": {"$contains": f'"{tag}"'}}
# passed to ChromaDB's `where` clause
```

ChromaDB's `where` parameter filters metadata and accepts only value
operators: `$gt / $gte / $lt / $lte / $ne / $eq / $in / $nin`. `$contains`
is a valid operator but lives in `where_document` (filter by stored
document body text), not `where` (filter by metadata). Every tag query
threw `Expected where operator to be one of ...`, got caught in the
`try/except`, logged a warning, and moved on with an empty result set.
Control then fell through to the cosine-fill branch at line 380:

```python
if len(pool) < pool_size:
    cosine_results = await self._search_collections(...)
```

…which filled 100% of the pool with unfiltered cosine results. Every
single query since v0.3.1. TagCascade never cascaded.

**Why it slipped.** `test_search_service.py` mocked `hybrid_query` with
a function that *accepted* `{"$contains": ...}` as if it worked and
returned pre-seeded results. Mock drift — production ChromaDB rejects
what the mock accepts. All unit tests passed while production silently
errored.

**Fix.** Drop the where-clause tag filter; over-fetch candidates by
vector + text hybrid query (`top_k=limit * 8` — 4× the previous target,
compensating for lost pre-filter selectivity); filter by tag membership
in Python by parsing `metadata.noun_tags` JSON. Tolerates both
JSON-encoded string (sidecar write path) and already-parsed list
(defensive — shouldn't happen but free to handle).

```python
# Before (broken — ChromaDB rejects $contains in where-clause)
tag_filter = {"noun_tags": {"$contains": f'"{tag}"'}}
merged = self._merge_filters(tag_filter, mb_filters)
results = await adapter.hybrid_query(..., filters=merged)

# After (over-fetch + Python-side tag match)
raw = await adapter.hybrid_query(..., top_k=limit * 8, filters=mb_filters)
results = []
for r in raw:
    meta = r.get("metadata") or {}
    raw_tags = meta.get("noun_tags")
    parsed_tags = None
    if isinstance(raw_tags, list):
        parsed_tags = raw_tags
    elif isinstance(raw_tags, str) and raw_tags:
        try:
            candidate = json.loads(raw_tags)
            if isinstance(candidate, list):
                parsed_tags = candidate
        except (ValueError, TypeError):
            parsed_tags = None
    if parsed_tags is not None and tag in parsed_tags:
        results.append(r)
        if len(results) >= limit * 2:
            break
```

**Impact.** TagCascade's tag-scoping now actually runs in production.
Tag-conditioned retrieval should produce noticeably better top-K
selection for queries that hit matched tags (the whole v0.3.1
motivation), and the 8 lines of error spam per query disappear from
logs. No schema change. No data migration. Backward-compatible with all
existing memory rows regardless of how noun_tags was stored.

**Tests.** Added `TestTagCascadePythonFilter` class in
`test_search_service.py` (7 cases):
- JSON-string with tag matches
- JSON-string without tag does not match
- Already-parsed list matches
- Missing noun_tags does not match (no crash)
- Empty-string noun_tags does not match (no crash)
- Malformed JSON does not match (no crash)
- Non-list JSON (e.g., object) does not match (no crash)

Existing `TestMergeFilters` tests updated to use valid ChromaDB
operators (`$eq`) instead of the broken `$contains` example, with a
docstring noting the historical reason.

**Coordination with core.** Core has the IDENTICAL bug at
`roampal/backend/modules/memory/search_service.py:353` — same broken
filter, same silent fall-through, same mock-drift test (core
`test_search_service.py:134` mocks `{"$contains": ...}` as if it works).
Documented in v0.5.3 release notes as a parallel fix; desktop ships
first, core ports the same pattern.

---

### 0n. TagCascade summary-lane multi-key filter on memory_bank (new, laptop-testing 2026-04-21)

`ui-implementation/src-tauri/backend/modules/memory/search_service.py:319-326`

**Discovered while verifying 0m.** The `$contains` bug at Section 0m
was throwing first and short-circuiting the rest of `_tag_routed_search`.
Once 0m was fixed, this latent bug in the same function surfaced:

```
[ChromaDB] Query failed: Expected where to have exactly one operator,
got {'memory_type': {'$ne': 'fact'}, 'status': {'$ne': 'archived'}} in query.
```

**Root cause.** ChromaDB's `where` clause accepts **either** a
single-key operator dict (`{"key": {"$op": val}}`) **or** an explicit
logical wrapper (`$and` / `$or`). A top-level multi-key dict is
rejected. The TagCascade tag-routed path was producing a multi-key
dict:

```python
# Before (broken — multi-key top-level dict)
mb_filters = metadata_filters                       # {"memory_type": {"$ne": "fact"}}
if coll_name == "memory_bank":
    mb_filters = dict(metadata_filters) if metadata_filters else {}
    mb_filters.setdefault("status", {"$ne": "archived"})
    # mb_filters is now {"memory_type": ..., "status": ...} — rejected
```

The caller (`agent_chat.py:699` cold-start summary lane) passes
`metadata_filters = {"memory_type": {"$ne": "fact"}}` to request
summaries. Adding `status != archived` only for memory_bank produces
the 2-key dict. Fires every time the summary lane queries memory_bank.

**Why it was masked.** Section 0m's `$contains` exception threw before
control reached the `adapter.hybrid_query` call, so the filter shape
was never evaluated against ChromaDB. Fixing 0m exposed this latent
bug next to it.

**Why the parallel cosine-fill path at `:547-553` was already correct.**
That code block (added in v0.3.1) uses explicit `$and` wrapping when
combining filters. The TagCascade tag-routed path at `:319-326` was
written without that pattern. This fix aligns the two paths.

**Fix** — same `$and` wrapping pattern already used elsewhere in the
file:

```python
# After
mb_filters = metadata_filters
if coll_name == "memory_bank":
    status_filter = {"status": {"$ne": "archived"}}
    if metadata_filters:
        mb_filters = {
            "$and": [{k: v} for k, v in metadata_filters.items()]
                    + [status_filter]
        }
    else:
        mb_filters = status_filter
```

**Impact.** The summary lane's query against memory_bank previously
failed silently (caught in `try/except`, 1 log error per query, empty
result list from that tier). In practice memory_bank holds facts
(not summaries), so an unfiltered summary-lane query on memory_bank
would still have returned zero useful results — the retrieval-quality
impact is near-zero. What the fix actually delivers is **log
cleanliness** (no more error spam from this path) and **correct
semantics** (if memory_bank ever does hold summary-shaped rows in a
future feature, they'd now be reachable by the summary lane).

**Tests.** Added `TestMemoryBankFilterWrapping` in
`test_search_service.py` (4 cases):
- `test_no_metadata_filters_returns_bare_status_filter` — single-key,
  no `$and` wrapping when caller passes no filter
- `test_empty_metadata_filters_returns_bare_status_filter` — empty-dict
  treated as None
- `test_single_metadata_key_wraps_in_and` — core regression: summary
  lane's `memory_type != fact` filter combined with status produces
  a valid `$and` shape, not a rejected multi-key dict
- `test_multiple_metadata_keys_each_become_conditions` — N metadata
  keys become N separate `$and` conditions + 1 status condition

**Coordination with core.** Core has the IDENTICAL bug at the same
call site — `roampal/backend/modules/memory/search_service.py:319-326`
uses the same broken `setdefault` pattern. Same fix applies. Documented
in v0.5.3 release notes as a parallel fix alongside Section 5
(`$contains`); desktop ships first, core ports the same pattern.

---

### 0o. TagService wrapper closure goes stale on sidecar swap (new, laptop-testing 2026-04-21)

`ui-implementation/src-tauri/backend/main.py:634-663`

**Symptom.** After switching sidecar model via Settings → Advanced, or
after any mirror_chat-driven sidecar swap following a chat-model
change, facts and memory_bank entries stop receiving LLM-extracted
noun_tags. Summaries keep getting tagged (they use a separate direct
call path). No error lines appear in the log at INFO level.

**Root cause.** The TagService LLM-extraction wrapper installed at
boot captured `app.state.sidecar_client` and `app.state.sidecar_model`
as local variables in its closure:

```python
# Before
sidecar_client = app.state.sidecar_client   # captured ONCE at boot
sidecar_model = app.state.sidecar_model

async def async_llm_tag_extractor(text: str):
    ...
    tags = await extract_noun_tags(
        text=text, client=sidecar_client, model=sidecar_model
    )
```

`model_switcher.py`'s `/sidecar/set`, `/sidecar/mirror`, and
`/switch` (when `mirror_chat=True`) handlers correctly update
`app.state.sidecar_client` / `sidecar_model` — but the closure's local
`sidecar_client` / `sidecar_model` vars still point at the original
boot-time `OllamaClient` object. That object's HTTP session may be
closed or its model may be unloaded; calls fail with connection or
404 errors. Both the wrapper and `extract_tags_async`'s exception
handlers logged at **DEBUG** level, suppressed at the prod default
INFO level, so the silent-empty return was invisible.

Laptop-testing repro: store facts with sidecar=gpt-oss:20b → switch
chat model (mirror_chat=true pulls sidecar to match) → store more
facts → new facts have no `noun_tags`.

**Fix.**

1. **Dynamic lookup in the wrapper.** Replace the boot-captured
   locals with call-time `getattr(app_state_ref, "sidecar_client", None)` /
   `sidecar_model` reads. Any update to `app.state.sidecar_client` by
   the swap handlers is picked up on the next tag-extraction call —
   no re-wire required.
2. **Bump the exception log level from DEBUG → WARNING.** Both in
   the wrapper and in `tag_service.py`'s `extract_tags_async`
   exception paths. Silent failures become visible at default log
   level so the next regression of this class surfaces immediately
   instead of rotting for weeks.

**Why not re-wire on swap.** Re-wiring would require every sidecar-
switch callsite in `model_switcher.py` (three paths: `/sidecar/set`,
`/sidecar/mirror`, `/switch` mirror-follow) to explicitly call back
into `app.state.memory._tag_service.set_llm_extract_fn(new_wrapper)`.
Every *future* swap path added would need to remember the re-wire
step or silently regress. Dynamic lookup makes future swap paths
work automatically.

**Files touched:**
- `ui-implementation/src-tauri/backend/main.py` — lines 634-663

**Tests.** Primary regression coverage is at the service level
(Section 0p below) since main.py's lifespan block is an integration
path. Manual verification: switch sidecar model mid-session, store a
new fact, confirm `noun_tags` populated.

**Coordination with core.** Core has no equivalent path — MCP server
shape doesn't expose a runtime sidecar-swap surface. No v0.5.x
counterpart needed.

---

### 0p. Sync `extract_tags()` can't await async `llm_extract_fn` (new, laptop-testing 2026-04-21)

`ui-implementation/src-tauri/backend/modules/memory/tag_service.py:384-404`
`ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py:920`

**Symptom.** Every AI-created memory_bank entry (via MCP
`add_to_memory_bank`, internal `create_memory` tool, direct
`store_memory_bank` calls without explicit `noun_tags`) lands with
no `noun_tags` in its metadata. Category tags (user-passed `tags` arg
like `["identity", "preference"]`) work fine; **auto-extracted** noun
tags from the body text never populate.

**Root cause — sync caller, async callee.** `store_memory_bank` at
`unified_memory_system.py:920` called the **sync** `TagService.extract_tags()`:

```python
# Before
actual_tags = (
    noun_tags if noun_tags else self._tag_service.extract_tags(text)
)
```

The production `_llm_extract_fn` installed by `main.py:634+` is an
**async** function (`async_llm_tag_extractor`). The sync `extract_tags`
invoked it without `await`, got back a coroutine object, passed it to
`_normalize_llm_tags` which tried `for tag in tags` — coroutines
aren't iterable, raised `TypeError`, the blanket `except Exception`
at `tag_service.py:399` caught it, logged at DEBUG (suppressed at
INFO), returned `[]`. Every invocation silently dropped the tags.

The async counterpart `extract_tags_async` at `tag_service.py:406+`
already had `inspect.iscoroutinefunction()` handling and worked
correctly. The sync method was the broken sibling.

**Why it slipped.** The sidecar-driven fact path (`store()` at line
747) uses `await self._tag_service.extract_tags_async(text)` and
was unaffected. Only `store_memory_bank` hit the sync path.
`test_tag_service.py:184` exercised `set_llm_extract_fn` with a
`MagicMock` that returns a plain list — not a coroutine — so the
sync-with-async-fn combination never came up in tests.

**Fix.**

1. **Caller-side (`unified_memory_system.py:920`)** — route through
   the async version:
```python
# After
actual_tags = (
    noun_tags if noun_tags else await self._tag_service.extract_tags_async(text)
)
```

2. **Service-side guard (`tag_service.py:384+`)** — explicit
   `inspect.iscoroutinefunction` check at the top of sync
   `extract_tags`. If a future caller invokes sync with an async
   `_llm_extract_fn`, return `[]` and log at WARNING so the mistake
   surfaces instead of being swallowed:
```python
if inspect.iscoroutinefunction(self._llm_extract_fn):
    logger.warning(
        "TagService.extract_tags() called with an async llm_extract_fn — "
        "sync path can't await it. Use extract_tags_async() instead. "
        "Returning []."
    )
    return []
```

Defense in depth: the caller is fixed; the service now also fails
loudly if anyone reintroduces a sync call site.

**Files touched:**
- `ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py` — line 920
- `ui-implementation/src-tauri/backend/modules/memory/tag_service.py` — lines 384-404

**Tests added (3 total):**
- `test_sync_extract_tags_with_async_llm_returns_empty` in
  `test_tag_service.py` — sync `extract_tags` with an async
  `_llm_extract_fn` returns `[]` and logs at WARNING.
- `test_memory_bank_auto_extraction_uses_async_path` in
  `test_unified_memory_system.py::TestV032Bug5MemoryBankTagAwait` —
  end-to-end verifies `store_memory_bank` without explicit
  `noun_tags` reaches the async LLM path and writes the returned
  tags via `update_fragment_metadata`.
- `test_memory_bank_explicit_noun_tags_shortcircuits_extraction` —
  regression guard: explicit `noun_tags=[...]` skips extraction
  entirely.

**Coordination with core.** Core has the broken sync method in
`tag_service.py` too, but **no production callers** — v0.4.9.1
already removed the three sites (`store_working`, `store_memory_bank`,
direct store paths) that called sync `extract_tags()`. Core's bug is
latent dead code, not a shipping defect. Documented in v0.5.3
RELEASE_NOTES as a defense-in-depth fix to add the same
`iscoroutinefunction` guard next time anyone touches `tag_service.py`.

---

### 1. Rewrite skipped MemoryPanelV2 tag tests

`ui-implementation/src/test/components/MemoryPanelV2.test.tsx`

Skipped:
- `describe.skip('Tag Filtering', ...)` — 4 tests inside: `filters memories when tag is clicked in cloud`, `shows Clear button when tags selected`, `clears tag filter when Clear clicked`, `applies AND logic for multiple tags`
- `it.skip('shows tag counts in the cloud')`

**Why:** v0.3.1 replaced the old tag-cloud click-to-filter flow with a Substack-style
tag input (`MemoryPanelV2.tsx` ~L290–325): user types into a "Filter by tag..."
textfield, sees typeahead suggestions, clicks a suggestion to add a pill, and the
pill filters the list. "Clear" is now an X icon with `title="Clear all tags"`, not
literal text.

**Action for v0.3.2:** Rewrite against the new flow:
- Type in the tag input to trigger the suggestions dropdown
- Click a suggestion and assert the pill appears
- Assert the memory list filters accordingly
- For the clear-all test, click the X icon (query by `title`)
- For AND logic, add two pills and assert only the intersection remains

### 2. Rewrite skipped OllamaRequiredModal tests

`ui-implementation/src/test/components/OllamaRequiredModal.test.tsx`

Skipped:
- `it.skip('shows both tab options')` — expected `"Setup LLM Provider"` and `"MCP Integration (Optional)"`
- `it.skip('shows LLM provider information')` — expected `"Recommended Provider:"` label
- `it.skip('explains why local providers')` — expected `"Why local providers?"` heading + `"No data leaves your computer"` body

**Why:** The onboarding modal was rewritten in v0.3.1 (per the release notes, "Welcome
modal explains chat model + sidecar architecture. Curated model list..."). The tab
labels and body copy were changed in that pass.

**Action for v0.3.2:** Read the current `OllamaRequiredModal.tsx` and update the three
assertions to match the new tab labels and copy. No behavioral rewrite required,
just string alignment.

### 3. Deprecation warnings to clear

Surfaced during the v0.3.1 test sweep. None block anything today, all become errors in
upcoming Python / Pydantic / GitHub Actions versions.

**Pydantic V1 → V2 migration:**
- `app/routers/agent_chat.py:415, 437` — `@validator` decorators. Migrate to `@field_validator` per the Pydantic V2 guide.
- `config/settings.py:94, 129` — class-based `Config` subclasses. Migrate to `ConfigDict`.
- Pydantic V2 will remove V1 support in V3.0.

**Stdlib datetime:**
- `modules/memory/search_service.py:560` — `datetime.datetime.utcnow()` is deprecated. Replace with `datetime.datetime.now(datetime.UTC)` for timezone-aware UTC.

**GitHub Actions runner:**
- `.github/workflows/tests.yml` — `actions/checkout@v4`, `actions/setup-python@v5`, `actions/setup-node@v4` currently run on Node.js 20 which is deprecated. Runner forces Node.js 24 starting 2026-06-02, removed 2026-09-16. Either bump the action versions when majors are released or set `FORCE_JAVASCRIPT_ACTIONS_TO_NODE24=true` as an env var on the jobs.

---

## Manual QA log (2026-04-20)

Tracked live as Logan exercised each feature in Tauri dev. Success/partial/fix
columns reflect what actually happened in the desktop, not just what unit tests
cover.

| Section | Exercised | Result | Fix applied during QA |
|---|---|---|---|
| 0 — latency | Startup on v0.3.2 | ✅ `ONNX models warm` fires at end of `initialize()`; first-message feels comparable to steady-state | — |
| 0a — Ollama universal errors | Killed Ollama mid-session, sent message | ✅ `**Model error:** All connection attempts failed` appeared inline, stream did not die silently | — |
| 0a — chat error toast | Same as above | ❌ → ✅ initial build used `msg.role` but the frontend type uses `msg.sender`; toast never fired. Fixed + added provider-aware "Is Ollama running? Start the service…" copy. |
| 0b — main-model persistence | Switched chat model → closed app → reopened | ❌ → ✅ Same-provider persist worked. Cross-provider broke: picked `lmstudio/qwen2.5-14b-instruct` → restart loaded `gemma4:31b` (ollama) because `active_provider` was locked to env default before persisted was consulted. Fix: added `select_active_provider()` so a detected+persisted provider drives startup; persisted+env+first-detected precedence. +3 tests. **Round-trip verified live:** LM Studio → Ollama → LM Studio → Ollama picks all survived restart with provider+model intact. |
| 0c — stale model 404 | `ollama rm qwen3:8b` with app live, sent a chat message | ❌ → ✅ Backend path fires, inline user-facing message lands, `main_model_config.json` removed. **Copy fix:** "Settings → Model" doesn't exist in the UI; rewrote to "Click the **model picker (download icon) at the top of the chat** and choose a new one." **Follow-up fixes during QA:** (1) `_clear_stale_model` now also removes `sidecar_config.json` when `mirror_chat=true` and sidecar pointed at the same dead model — otherwise sidecar boots into a dead reference. (2) Migration helper `should_migrate_env_to_json` now validates the env model is actually in `available_models` before seeding the JSON — prevents stale env vars (e.g. `OLLAMA_MODEL=codellama:latest`) from writing a bogus persisted selection. (3) Boot-time mirror enforcement: if `mirror_chat=true` and the persisted sidecar model differs from the current chat model, rewrite sidecar to match at startup (not just on switch). +3 tests. |
| 0c — stale model (unified across providers) | Removed `qwen2.5-7b-instruct` from LM Studio, sent a chat message | ❌ → ❌ → ✅ **3 rounds of fixes.** (1) First added stream-chunk stale detection, but LM Studio returned HTTP 500 before any chunks arrived — httpx's `HTTPStatusError` path didn't check the body. User got the ugly `developer.mozilla.org/.../Status/500` default message. (2) Added HTTPStatusError handler on OpenAI path mirroring Ollama's 0a — caught the 500 but missed the specific "Failed to resolve model metadata" pattern LM Studio uses, so stale-clear didn't fire. (3) **Unified everything:** extracted module-level `_STALE_MODEL_PATTERNS` (8 universal phrases incl. "failed to resolve/load/unable to load"), `_is_stale_model_body()` helper, and `_stale_model_user_message()` function. Both Ollama 404/body branch AND LM Studio HTTPStatusError + stream-chunk branches now call the same helpers with the same pattern list and yield the same user copy. **Also:** `_clear_stale_model` now preserves the user's provider preference — rewrites `main_model_config.json` as `{model: "", provider: <last>}` instead of unlinking. Same for `sidecar_config.json` (preserves provider + mirror_chat flag when model=="" cleared). Fixes the silent provider-kick bug where an LM Studio user whose model got removed would boot onto Ollama. |
| LM-Studio-only machine (no Ollama) | Trace-verified, not live-tested | ✅ `select_active_provider` precedence persisted → env → first-detected. If Ollama not detected, step 3 picks LM Studio. Plus `main.py` has a secondary fallback loop: "if active provider has no models, switch to another provider with non-empty list." Both paths land on LM Studio when it's the only running host. Edge case "no providers at all" → `app.state.llm_client = None` → UI surfaces OllamaRequiredModal (Section 2 onboarding). Worth live-testing by stopping the Ollama service and restarting Roampal. |
| 0d — ChromaDB telemetry | Grep'd 3,686-line backend log for `capture() takes` and `PostHog` patterns | ✅ **Zero matches.** Original customer report was 3.8 MB+ of this spam per session. Clean. | — |
| 0e — keep_alive | Not exercised (passive — needs 10+ min idle) | ⏳ | — |
| 0f — sidecar mirror | Toggled mirror ON/OFF via Advanced; switched chat model | ✅ Mirror update fires on chat switch; mirror_chat persists across restarts | — |
| 0f — first-run migration (v0.3.1 → v0.3.2) | Existing sidecar config with no `mirror_chat` field | ✅ `✓ Sidecar config migrated: mirror_chat=False` — infers False because sidecar model ≠ chat model (preserves user intent) | — |
| 0f — Advanced picker | Opened Advanced → toggled mirror OFF | ❌ → ✅ dropdown was empty; fixed — frontend was treating `/providers/all/models` response as array when it returns a dict keyed by provider |
| 0f — reload warning modal | Picked a sidecar model ≠ chat (single-GPU) | ✅ Modal fires with Cancel / Confirm; skipped when picks match | — |
| 0f — sidecar error toast | Killed LM Studio during a session | ✅ Toast fires on transition clean→error with `Start LM Studio, or change sidecar in Advanced` copy |
| 0g — VRAM warning | Added dev endpoint `POST /_debug/set-vram` (gated on `ROAMPAL_DEV=1`) → overrode to 2 GB → re-selected `gpt-oss:20b` (tag heuristic: ~15 GB required) | ❌ → ✅ **backend returned `vram_warning` but UI never read it**. Added `vramWarningToast` state + Toast render in ConnectedChat; caller flags model-switch response; toast opens Model Settings. **Toast confirmed live.** Restored real VRAM after. |
| 0h — dropdown labels | Opened main-model dropdown | ❌ → ✅ Curated rows showed marketing copy ("OpenAI efficient model"), others showed params — inconsistent. Flipped priority so param format is canonical; curated strings now last-resort only. |
| 0h — "Custom model" label | Same | ✅ Absent from dropdown output |
| 0i — ChromaDB logging | Not exercised (investigate-only, v0.3.3 fix scope) | — | — |
| Badge (follow-up) | Clicked / hovered chat-header badge | ❌ → ✅ initially clickable deep-linking to Advanced; Logan flagged wrong affordance. Changed to `<div role="status">` with `cursor-default`, tooltip still names sidecar + points to Advanced. |
| Fact dedup (new in QA) | Sent multiple "Jerry the horse" messages | ❌ → ✅ dedup never fired because threshold `0.1` is cosine-space but desktop ChromaDB uses `hnsw:space: l2`. Raised threshold to `0.32` (= cos_sim > 0.95 in L2 space). |
| Context-length error copy | Hit LM Studio context overflow | ❌ → ✅ "Or use Ollama instead: `ollama pull <lmstudio-id>`" was broken — LM Studio IDs don't match Ollama's `name:tag` convention. Dropped the Ollama fallback line. |
| Section 1 — tag filter UI | Typed tag in input, picked suggestion, added second tag pill, AND-filtered, cleared via X | ✅ Full flow works end-to-end |
| Section 2 — onboarding modal | Stopped both Ollama and LM Studio → reloaded Tauri | ✅ `⚠️  No LLM providers detected` → `Starting in setup mode` → setup modal/model library surfaces cleanly, no spinner hang, no crash |
| Post-stale restart (boot fallback) | After LM Studio stale-clear, close + reopen Tauri | ❌ → ✅ **Fixed `load_main_model_config`** in `model_switcher.py:2670` — old check required both `model` AND `provider` truthy, so `{"model": "", "provider": "lmstudio"}` (written by `_clear_stale_model`) returned `None` and boot fell through to env-default Ollama, silently switching provider. Relaxed to require only `provider`; empty model is explicit "no persisted model, keep the provider preference." Boot now honors persisted provider even when model field is blank, then picks first-available LM Studio model. Logan verified live: `✓ Using persisted provider: lmstudio (model: )` → `✓ Using first available model: qwen2.5-14b-instruct`. |
| Timeout / empty-exception error copy | Ollama gemma4:31b cold-loaded too slow, hit `httpcore.ReadTimeout` | ❌ → ✅ **Fixed empty `**Model error:**`** in `ollama_client.py:1091` — `str(e)` on `ReadTimeout` is empty, rendering a bare header with no detail. Now falls back to `type(e).__name__` when str is empty, and special-cases `timeout`/`timed out` with actionable copy: *"Model timed out. Large models can be slow to cold-load — try again, or pick a smaller model."* |
| Section 3 — deprecation sweep | Pytest suite | ✅ 0 deprecation warnings; audit also caught `modules/ingestion/models.py` still using `class Config:` — migrated to ConfigDict |
| Prompt audit (3rd pass, fact-check) | Traced every factual claim in prompt back to actual code | ❌ → ✅ **3 more defects fixed:** (6) Cold-start paragraph claimed `always_inject` flag drove selection — actually cold-start uses TAG_PRIORITIES (identity/preference/goal/project/system_mastery/agent_growth) to pick one best fact per category via `_build_cold_start_profile`. `always_inject` is used by **organic recall** every-message, not cold-start. Rewrote to match reality. (7) `LIMITED to 4 tools` claim was false when MCP is connected — `AVAILABLE_TOOLS + external MCP tools` get merged at request build. Rewrote to "built-in tools are X; additional tools may be present via MCP; inspect what you were given." (8) `Use tags ... these drive TagCascade retrieval` conflated two mechanisms — TagCascade routes on **noun_tags** (auto-extracted from query), NOT on human tags. Human tags drive cold-start category selection. Rewrote to separate the two. |
| Prompt audit (2nd pass, stricter) | Full audit with criterion "every prompt line must be actionable or describe observable context" | ❌ → ✅ **5 edits landed, 2 considered-and-rejected:** (3) Rewrote the `[How Memory Automation Works]` section — killed "Past Experience (Content KG)", "failure_patterns graph", "Action-Effectiveness KG", "Recommendations" ghost artifacts the LLM never sees. Replaced with what actually lands in context (4 summaries + 4 facts with wilson/uses/last_outcome/age metadata). (4) Rewrote `[Outcome Scoring]` section — it framed the LLM as "you, in a separate call" which is a lie; the sidecar is the scorer. New version honestly says "background sidecar scores; you'll see updated wilson + last_outcome on future turns." (5) **Deleted** the entire `[Action-Effectiveness Stats]` section — stats were never actually injected, the block was instructional text for data that doesn't exist. |
| Prompt audit (1st pass) | Explore agent reviewed `agent_chat.py:_build_complete_prompt` + tool schemas | ❌ → ✅ **2 edits landed, 2 considered-and-rejected:** (1) Deleted dead `[Memory Attribution - v0.3.0]` block (~14 lines) that instructed the LLM to emit `<!-- MEM: 1👍 2🤷 3👎 4➖ -->` annotations — sidecar is the sole scorer in v0.3.1+, response-stripper silently removes them, LLM was wasting tokens. (2) Added `update_memory` / `archive_memory` when-to-use guidance with a scope note clarifying these only affect `memory_bank` entries (not working/history/patterns, which are sidecar-managed). **Rejected — fact-dedup note:** the LLM has no way to check for duplicates; dedup is silent backend plumbing. Telling the LLM could push it to waste tool calls pre-checking. **Rejected — sidecar persistence caveat:** referenced a UI "error indicator" the LLM can't see (frontend-only state, not injected into context). Would have been pure pollution. Both catches courtesy of Logan's real-time sanity check. Tool schemas (search/create/update/archive) already match reality — no tool schema edits needed. |
| Audit artifacts | Explore agent full audit | ✅ Caught 0g UI gap + ingestion Pydantic V1 residue; both fixed |

**Net test score at end of QA session:** backend 498/499 (1 pre-existing conditional skip for optional `feature_flag_validator` import, 0 failed); frontend 523/523; typecheck clean.

**Test count after subsequent fixes (2026-04-21):** backend **546 passed, 1 skipped, 0 failed** after Sections 0l–0p added regression coverage (6 for 0l memory-panel flatten, 7 for 0m TagCascade Python filter, 4 for 0n memory_bank filter wrapping, 3 for 0p memory_bank tag-extraction await, 5 for 0o dynamic sidecar lookup).

**Manual QA verdict:** full pass for the 2026-04-20 surface. Item not live-exercised is 0e (keep_alive — requires passive 10+ min idle then a message), covered by unit tests. Sections 0o and 0p were discovered during code review of the 0l/0m/0n fixes and ship with unit-test coverage only; their end-to-end behavior (tagged facts survive sidecar swap, memory_bank entries receive auto-extracted noun_tags) is verified on the laptop re-test pass that landed this document.

---

## How to find the TODOs in code

```
grep -r "TODO(v0.3.2)" ui-implementation/src/test/
```

Both files carry explanatory comments above the skipped `describe`/`it` blocks.

---

## Non-goals

- 0i (ChromaDB "Error finding id") — investigate-only in v0.3.2. Real fix ships in v0.3.3.

---

## Session context (v0.3.1 post-ship cleanup, 2026-04-17)

For context — the following items were completed during the v0.3.1 CI+cleanup pass and
do NOT belong in v0.3.2:

- Fixed 3 stale sidecar truncation test assertions to match v0.3.1.4 limits (2000/8000/8000).
- Deleted dead code: `services/unified_image_service.py`, `modules/vision/`, `ContextBar.tsx` + test (zero callers, zero renderers).
- Removed unused deps from `requirements.txt`: Pillow, pytesseract.
- Loosened version pins (`fastapi`, `starlette`, `uvicorn`, `numpy`) to match what was actually being built against. Removed redundant `anyio` pin.
- Added `.github/workflows/tests.yml` CI workflow covering backend (pytest) and frontend (vitest + tsc + eslint + build).
- Added repo topics for Desktop discoverability (20 total).
- Created 15 missing git tags (v0.1.5, v0.2.0, v0.2.1, v0.3.0, v0.3.1) and 15 GitHub Releases.
- Fixed characterization tests to use `tmp_path` fixture instead of hardcoded `C:/ROAMPAL/...` paths.

All of the above are on `master` already.
