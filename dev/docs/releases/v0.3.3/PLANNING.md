# Roampal Desktop v0.3.3 — Planning

**Status:** Not scheduled. Item collection only.

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
