# Roampal Desktop v0.3.0 - Performance & Polish

**Release Date:** January 16, 2025

---

## Performance Improvements

### TanStack Virtual Migration
Migrated from react-window to @tanstack/react-virtual for message virtualization.

**Problem:** Blank space / scroll jank when toggling memory side panel. Content would jump, show gaps, then snap back.

**Root cause:** react-window (6 years old, unmaintained) caches cumulative heights internally. When container width changes, text reflows but cached heights become stale. `resetAfterIndex()` recalculates total height which disrupts scroll position.

**Fix:**
- `measureElement` ref callback - Auto-measures each item via per-item ResizeObservers
- **No `measure()` calls** - Calling measure() nukes all cached heights causing 1-frame overlap; let measureElement handle it naturally
- Scroll position preservation on width change (without cache invalidation)

| react-window | TanStack Virtual |
|--------------|------------------|
| `<VariableSizeList>` | `useVirtualizer()` hook |
| `itemSize={getSize}` | `estimateSize` + `measureElement` |
| `resetAfterIndex(i)` | Not needed (auto-measures) |
| `scrollToItem(i)` | `scrollToIndex(i)` |

### Faster Message Rendering
Optimized React component subscriptions to reduce unnecessary re-renders. The chat interface now responds more smoothly, especially during long conversations.

### Message Virtualization
Long conversations now use windowed rendering - only visible messages are in the DOM. This eliminates lag when scrolling through extensive chat histories.

---

## Streaming Fixes

### Fix Text Overlap During Streaming
**Problem:** Messages overlapped during streaming because height estimation couldn't keep up with rapid token arrival.

**Fix:** Proper measurement via debounced height updates:
1. Consistent height estimation - Use simple `/60` chars per line estimate without streaming-specific buffers
2. Debounce height updates - Batch all height changes with 16ms debounce (one frame) and 5px tolerance threshold
3. Single immediate measurement - useLayoutEffect measures once; debouncing handles rapid updates

### Fix Thinking Icon Position and Extra Space
**Problem:** "Thinking." appeared in wrong position during streaming (middle of response), and "phantom space" appeared between user message and assistant response after streaming ended.

**Fix:**
1. Only render loading indicator row when `isProcessing && !hasStreamingMessage`
2. Return zero-height element when not processing (safety net)

### Fix Loading Indicator Height Cache
**Problem:** Brief text overlap when "Thinking." transitions to streaming response.

**Fix:** Clear height cache when `showLoadingIndicator` transitions from true to false.

---

## New Features

### Surfaced Memories Display
Users can now see what memories the system used to form its response.

**Implementation:**
1. Backend sends `surfaced_memories` in `stream_complete` event
2. Frontend stores them on message object as `surfacedMemories`
3. New `SurfacedMemoriesBlock` component displays them with expandable view

Shows "context: N memories" with expandable list after each response.

### Search by Recency
Added `sort_by` parameter to memory search:
- `"relevance"` (default) - semantic similarity order
- `"recency"` - most recent first (by timestamp)
- `"score"` - highest outcome score first

### Timestamps Visible to LLM
Search results now include age: `[1] (working, 2d): content...` instead of just `[1] (working): content...`

### Tool Chaining Support
LLM can now call multiple tools in sequence (e.g., `create_memory` then `search_memory` in one response).

**Before:** After a tool call, continuation disabled all tools (`tools=None`), preventing chained operations.

**After:** Tools remain available on continuation. `MAX_CHAIN_DEPTH = 3` prevents infinite loops.

Example: "Store a memory and search for related ones" now works in a single response.

### Per-Memory Scoring in record_response
The `record_response` MCP tool now supports `memory_scores` parameter for per-memory scoring, matching roampal-core's `score_response` behavior.

**Before (v0.2.x):**
```python
record_response(
    key_takeaway="User prefers dark mode",
    outcome="worked",
    related=[1, 3]  # Score positions 1 and 3
)
```

**After (v0.3.0):**
```python
record_response(
    key_takeaway="User prefers dark mode",  # Required - forces reflection
    memory_scores={
        "history_abc123": "worked",   # This memory helped
        "patterns_xyz789": "failed",  # This memory was misleading
        "working_def456": "unknown"   # Didn't use this one
    }
)
```

Changes:
- `key_takeaway` remains required (forces LLM to reflect on what happened)
- `memory_scores` parameter: `{doc_id: outcome}` for per-memory scoring
- `related` parameter deprecated (still works for backward compatibility)
- Response format simplified: `"Scored (outcome=worked, 3 memories updated)"`

### "Unknown" Outcome Support
Added `unknown` as a valid outcome for per-memory scoring (MCP and internal).

**What "unknown" means:** Memory was surfaced but not used in the response.

**Scoring effect:**
- `score_delta = 0.0` (raw score unchanged)
- `uses += 1` (increment use count)
- `success_delta = 0.25` (weak negative signal for Wilson score)

**Why 0.25?** Creates natural selection over time. If a memory keeps getting surfaced but never used, its Wilson score drifts down (25% < 50% neutral), causing it to rank lower in future searches.

**Usage:**
- MCP: `memory_scores={"doc_id": "unknown"}` in `record_response`
- Internal: Memories marked ➖ (unused) by main LLM are now scored as "unknown" instead of skipped
- v0.2.12 Fix #5/7 fallback paths also score unused memories as "unknown"

**Before v0.3.0:** Unused memories were skipped (no scoring) - they never drifted down.
**After v0.3.0:** Unused memories score as "unknown" - natural selection enables over time.

### 4-Emoji Memory Attribution System

The internal LLM now uses 4 emojis for direct memory attribution, matching all 4 outcome types.

**Format in LLM response:**
```
<!-- MEM: 1👍 2🤷 3👎 4➖ -->
```

**Emoji → Outcome Mapping:**

| Emoji | Meaning | Outcome | Score Δ | success_δ | Wilson Effect |
|-------|---------|---------|---------|-----------|---------------|
| 👍 | Definitely helped | worked | +0.2 | 1.0 | ↑ Strong up |
| 🤷 | Kinda helped | partial | +0.05 | 0.5 | → Neutral |
| 👎 | Misleading/hurt | failed | -0.3 | 0.0 | ↓ Strong down |
| ➖ | Didn't use | unknown | 0.0 | 0.25 | ↓ Weak down |

**Before (v0.2.x):** 3 emojis (👍/👎/➖), scoring LLM determined outcome, then modulated per-memory scores.
**After (v0.3.0):** 4 emojis, direct emoji→outcome mapping, no modulation needed.

**Collection-Specific Behavior:**

| Collection | Score Δ | Wilson | Lifecycle |
|------------|---------|--------|-----------|
| working | ✓ | ✓ | promote (≥0.7, uses≥2) / delete (<0.2) |
| history | ✓ | ✓ | promote (≥0.9, uses≥3, success≥5) / delete (<0.2) |
| patterns | ✓ | ✓ | demote only (<0.4 → history) |
| books | ✗ skip | ✗ | permanent |
| memory_bank | ✓ (useless) | ✓ (ranking) | permanent |

**Promotion Requirements (ported from roampal-core v0.2.9):**
- working → history: score ≥ 0.7, uses ≥ 2
- history → patterns: score ≥ 0.9, uses ≥ 3, **success_count ≥ 5** (must prove usefulness in history)
- On promotion to history: counters reset (uses=0, success_count=0) - memory must prove itself fresh

Note: Patterns cannot be directly deleted. They demote to history first, then history can be deleted.

---

## UI Refinements

### Modern Terminal Aesthetic
- Component backgrounds use zinc-950 (#0a0a0a) for a softer, more readable dark theme
- Refined color palette following modern design principles

### Message Styling
- User messages now have a subtle blue left border accent (`border-blue-500/40`) for modern terminal feel
- Blue-tinted `>` prompt matches the accent
- Brighter user text (`zinc-300`) for better readability
- Clear visual distinction between user input and assistant responses

### Improved Resize Handles
- Panel dividers are now more visible with grip indicators
- Easier to locate and drag for resizing panels

### Better Loading States
- New "Processing" spinner replaces the old dots animation
- Memory stats panel shows loading indicator while fetching data

### Collapsible Memory References
- Individual memories in "context: N memories" and "[N] references" blocks are now collapsible
- Shows first 80 characters with `[+]` indicator for truncated items
- Click to expand/collapse individual memory text
- Prevents "wall of text" when many memories are surfaced

---

## Bug Fixes

- **Books Collection Search**: Fixed issue where searches could return empty results despite data existing
- **Tooltip Positioning**: Memory stats tooltips now appear above the element, preventing clipping
- **Coming Soon Buttons**: Disabled non-functional buttons to prevent confusion
- **Message Height Race Condition**: Fixed initial chat load overlap where `useLayoutEffect` measured height before markdown finished rendering
- **Chat/Memory Panel Gap**: Fixed visual gap between chat and right memory panel caused by padding with mismatched background colors
- **Wilson Score 10-Use Cap Bug**: Fixed Wilson score calculation breaking for memories with >10 uses. Added `success_count` tracking that accumulates without the 10-entry history cap. Also fixed `failed` outcomes not incrementing `uses`. Ported from roampal-core v0.2.8. (`outcome_service.py`, `scoring_service.py`)
- **References Auto-Collapse**: Fixed expanding "context: N memories" or "[N] references" auto-collapsing after ~5 seconds. Lifted expanded state to parent component, keyed by message ID.
- **Score Bug**: Memories created via `record_response` now start at 0.5 (not boosted). Scoring only happens via `score_response` deltas.
- **_extract_concepts Missing Method**: Fixed `AttributeError` when KG service extracts concepts during context injection. Changed `self._extract_concepts()` to `self._kg_service.extract_concepts()`. (`unified_memory_system.py:1085`)
- **Surfaced Memories Variable Overwrite**: Fixed citations crash when >5 search results. The `surfaced_memories` list was being overwritten by `content_map` dict, causing `'int' object has no attribute 'get'` error. Renamed to `outcome_memories` for selective scoring. (`agent_chat.py:1114`)
- **KeyError 'total'/'failures' in KG Routing**: Fixed crash when scoring memories with old routing patterns missing keys. Old patterns could be missing `total`, `successes`, `failures`, or `partials` keys. Added migration check to ensure all keys exist before incrementing. (`knowledge_graph_service.py:240-244`)
- **Claude Code MCP Integration**: Fixed integrations Connect button not working for Claude Code CLI. Claude Code stores MCP config at `~/.claude.json` (mcpServers at root level), NOT `~/.claude/mcp.json`. Updated scan/connect/disconnect to detect and write to the correct location. All tools use standard `mcpServers` wrapper format. (`mcp.py`)
- **Memory Bank Wilson Scoring**: Fixed memory_bank Wilson tracking being completely blocked by early return. Removed safeguard that prevented `uses` and `success_count` from being tracked. The 20% Wilson + 80% quality ranking formula was already in place but had no data. Now Wilson accumulates properly for memory_bank ranking. (`outcome_service.py:100-104` removed, `scoring_service.py:279` unchanged)
- **Sidebar Collapse Gap/Jank**: Fixed visual gaps, layout jank, and text overlap when toggling sidebars. Root cause: `virtualizer.measure()` nuked all cached heights, causing 1-frame overlap as items fell back to 80px estimates. Fix: removed `measure()` entirely - let `measureElement`'s per-item ResizeObservers handle height updates naturally. (`TerminalMessageThread.tsx:608-633`)
- **KG Node Outline Thickness**: Fixed Knowledge Graph nodes having huge grey outlines. Edge drawing set `ctx.lineWidth` based on edge weight, but node stroke didn't reset it. Added `ctx.lineWidth = 1` before node stroke. (`KnowledgeGraph.tsx:470`)
- **KG Dead References Cleanup**: Fixed `'str' object has no attribute 'get'` crash in `_cleanup_kg_dead_references`. Some `problem_solutions` entries are plain strings (doc_id), not dicts. Now handles both formats. (`unified_memory_system.py:1311`)
- **Textarea Resize on Sidebar Toggle**: Fixed chat input not resizing when sidebars collapse/expand. Text wrapping changes when container width changes, but height only recalculated on typing. Added ResizeObserver to trigger resize on width change. (`ConnectedCommandInput.tsx:50-58`)
- **create_memory Tag Parameter Bug**: Fixed memories being stored with wrong tags. Tool definition specifies `tags` (array) but code read `tag` (singular string), defaulting to "context". Now handles both formats and passes through `importance`/`confidence` parameters. (`agent_chat.py:2594-2600`)
- **Memory Tool No Response Bug**: Fixed LLM not responding after calling `create_memory`, `update_memory`, or `archive_memory`. Only `search_memory` had continuation logic to prompt LLM for a text response after tool execution. Now all memory tools trigger continuation. (`agent_chat.py:2825-2826`)
- **search_memory Null Collections Crash**: Fixed `TypeError: 'NoneType' object is not iterable` when LLM passes `collections: null`. Python's `.get(key, default)` returns `None` when key exists with null value, bypassing the default. Changed to `or ["all"]` pattern. (`agent_chat.py:2491`)
- **Tool Display Order & Interleaving**: Fixed tools appearing above response text and text duplication during live streaming. Now preserves true chronological interleaving for both Ollama and LM Studio backends.

  **(1) Session load/switch** - Load paths now create `events` array with proper ordering. (`useChatStore.ts:294-325`)

  **(2) Text duplication bug** - First token created `firstTextEvent` containing the first chunk, but `tool_start` also captured ALL content since boundary 0. This caused the first chunk to render twice.

  **(3) Boundary interference bug** - `tool_complete` set `_lastTextEndIndex` which interfered with subsequent `tool_start` text capture in multi-tool scenarios like `[tool1] text [tool2]`.

  **Solution:** Simplified event building with clean boundary tracking:
  - `token` handler: Creates message with `events: []` (no `firstTextEvent`)
  - `tool_start`: Captures `content.slice(lastBoundary)` as `text_segment`, adds tool, updates boundary; sets `_toolArrivedFirst` flag if no prior tokens
  - `tool_complete`: Updates tool status only (no boundary changes)
  - `tool_complete`: When `_toolArrivedFirst`, records `_toolCompleteContentIndex` to mark preamble/continuation boundary
  - `stream_complete`: Captures trailing text; if `_toolArrivedFirst`, splits content at `_toolCompleteContentIndex` - inserts preamble BEFORE tool, appends continuation AFTER tool (`useChatStore.ts:789-795,876-908`)

  **(4) Continuation text not streaming** - After tool execution, continuation text was buffered (v0.2.5 design) instead of streamed. This caused all continuation text to appear at the end as a single block. Changed from "buffer only, no token yields" to streaming continuation tokens for true interleaving. (`agent_chat.py:950,2830`)

  **(5) Synthetic preamble for models that skip text** - LM Studio models often go directly to tool_calls without producing preamble text, causing tools to render above all text. Added synthetic preamble ("Let me search my memory...") before tool_start for initial tool calls (depth 0 only). This ensures text→tool→response ordering even when model skips preamble. (`agent_chat.py:2449-2463`)

  **Backend Compatibility:**
  - **Ollama**: Tools can appear mid-stream → native interleaving
  - **LM Studio**: Tools come after text in single call. Synthetic preamble + streaming continuation ensures proper ordering
  - Frontend handles both identically via the same event pattern

  **Result:** True interleaving preserved. `"text1" [tool1] "text2" [tool2] "text3"` renders in correct order.

  **(6) Events not passed to render component** - `ConnectedChat.tsx` message mapping included `toolExecutions` but not `events`. Events were built correctly in store but stripped during component format conversion. Added `events`, `_lastTextEndIndex`, `_toolArrivedFirst` to mapping. (`ConnectedChat.tsx:1749-1752`)

  **(7) Continuation LLM client typo** - Backend used `self.ollama_client` but class only has `self.llm`. Caused `AttributeError` during tool continuation, breaking all chained tool calls. Fixed typo. (`agent_chat.py:943`)

  (`useChatStore.ts:661-671,755-796,895-939`, `TerminalMessageThread.tsx:393-446`, `ConnectedChat.tsx:1749-1752`, `agent_chat.py:2449-2463,2830`)
- **Missing get_routing_patterns Method**: Fixed `AttributeError: 'KnowledgeGraphService' object has no attribute 'get_routing_patterns'` in context analysis. Added the missing getter method. (`knowledge_graph_service.py:952-955`)
- **Session Load Tool Order**: Fixed tools not appearing in their exact original position on session reload/refresh. Previously, session load built events array without position info, causing tools to always appear before text regardless of actual order. Now backend includes `content_position` in tool events, and frontend uses this to split content and interleave tools at their exact original positions. Old sessions without position info fall back to tools-first. (`agent_chat.py:917-920,958-961,2465-2473,2582-2797`, `useChatStore.ts:306-370,1509-1580`)
- **Switch During Processing**: Fixed content loss when switching conversations during streaming. Session clicks are now blocked while processing (visual indicator shows disabled state). Previously, switching mid-stream would discard in-progress content. (`ConnectedChat.tsx:1780-1788`, `Sidebar.tsx:65,187-189`)
- **Graceful Tool Chain Limit**: Increased tool chain depth from 3 to 5 for models like Qwen that need more iterations. On final iteration, uses wrap-up prompt and removes tools to force text-only response, preventing abrupt cutoffs. (`agent_chat.py:803,2835-2853`)
- **Memory Bank Modal Spacing**: Fixed inconsistent spacing between memory cards in the Memory Bank modal. Variable heights based on tags caused uneven gaps. Changed to single fixed height (140px) with `h-full` cards that stretch to fill container, `flex-col` layout with date pinned to bottom via `mt-auto`. All cards now have uniform visual height with consistent 8px gaps. (`MemoryBankModal.tsx:40,183-218,232-267`)
- **Stale Routing Pattern Success Rate**: Fixed routing patterns showing incorrect success rates (e.g., 71% with 0 worked / 0 failed). Two code paths modified routing pattern data: `_track_routing_usage` incremented `total` on every search but never recalculated `success_rate`, while `update_kg_routing` did both but only ran when feedback was recorded. If feedback was never provided (or was always "partial"/"unknown"), `success_rate` retained stale historical values. Added success_rate recalculation to `_track_routing_usage`. (`routing_service.py:382-395`)
- **Session Load Tool Position Wrong Field**: Fixed tools jumping to top on app restart/session reload. Backend saves `content_position` in `toolEvents`, but `switchConversation` was checking `toolResults` (which doesn't have positions). Now correctly reads from `toolEvents`. The other load path at line 1518 was already correct. (`useChatStore.ts:307-313`)
- **Action Effectiveness Popup Data Mismatch**: Fixed KG popup showing 0 worked / 0 failed for action effectiveness nodes while the node itself showed correct stats. The `get_concept_definition` API only checked `routing_patterns`, but action effectiveness patterns (format `action@context->collection`) are stored in `context_action_effectiveness`. Now converts display format back to storage key and checks both data sources. (`memory_visualization_enhanced.py:496-526`)
- **Missing BM25 Dependency**: Fixed hybrid search not working due to missing `rank-bm25` package in requirements.txt. BM25 combines keyword matching with vector search for better recall. Without it, searches fell back to vector-only mode, potentially missing exact keyword matches. (`requirements.txt:60`)
- **Routing KG Patterns Never Persisted**: Fixed race condition where routing patterns were never saved to disk. `get_kg_entities()` called `reload_kg()` to sync with MCP process changes, but this wiped in-memory patterns during the 5-second debounce window. If UI polled KG visualization while patterns were pending save, they were lost forever. Now flushes pending saves before reload. (`knowledge_graph_service.py:753-768`)
- **Missing Dependencies Audit**: Added 6 packages that were imported but missing from requirements.txt: `aiohttp` (embedding_client), `Jinja2` (prompt_engine templates), `tiktoken` (token counting), `Pillow` (image processing), `PyYAML` (personality configs), `mcp` (MCP server SDK). (`requirements.txt`)
- **Dependencies Cleanup**: Removed unused packages from requirements.txt to reduce install size and potential conflicts: `pathlib2` (standard pathlib used), `textblob`, `vaderSentiment` (not imported), `docx2txt` (python-docx used), `lmstudio` SDK (httpx used for API). (`requirements.txt`)

---

## Security Fixes

- **CRITICAL: ZIP Path Traversal**: Fixed directory traversal vulnerability in backup restore. Malicious ZIP files could write outside the extraction directory. Added validation to ensure all ZIP member paths resolve within the target directory. (`backup.py:401-406`)
- **CRITICAL: XSS in Memory Citations**: Fixed Cross-Site Scripting vulnerability where `dangerouslySetInnerHTML` was used to render citation links. Refactored to use React elements with proper event handlers, eliminating HTML injection risk. (`MemoryCitation.tsx:56-89`)
- **HIGH: Query Content in Logs**: Removed user query content from log files to protect privacy. Logs now show only metadata (query_len, result_count) instead of actual query text. (`unified_memory_system.py:1078`, `agent_chat.py:2503,2555`)
- **HIGH: MCP Path Validation**: Added path validation to custom MCP config path storage. Paths outside the user's home directory are now rejected. (`mcp.py:140-145`)

---

## Memory Context Sync

Ported roampal-core's cold start and context injection to Desktop for consistency and improved recall.

### What Changed

| Before | After |
|--------|-------|
| Vector search for "user identity name projects..." | Quality-sorted, one fact per tag category |
| Pattern signature exact match only | Unified search ALL collections (except books) |
| No Wilson scoring on recall | Wilson scoring → top 3 proven memories |

### Implementation

**Cold Start Profile** (`unified_memory_system.py:_build_cold_start_profile()`)
- Fetches ALL memory_bank facts, sorts by quality (importance × confidence)
- Picks ONE highest-quality fact per tag category (identity, preference, goal, project, system_mastery, agent_growth)
- Formats as `<roampal-user-profile>` with one line per category
- Note: No identity prompts - small models don't reliably follow through on storing names after asking

**Context Injection** (`unified_memory_system.py:get_context_for_injection()`)
- Gets always_inject memories (core identity)
- Unified search across working, patterns, history, memory_bank (books excluded)
- Applies Wilson scoring for proper ranking
- Returns top 3 proven memories with doc_ids for scoring
- Formats as `═══ KNOWN CONTEXT ═══` block

---

## Dev/Prod Data Path Isolation

Fixed Desktop dev mode incorrectly sharing production data path.

**Problem:** Desktop dev and prod were both using `%APPDATA%/Roampal/data/`.

**Fix:** (`settings.py`)
- Added `ROAMPAL_DEV` env var support (matches roampal-core)
- Fixed auto-detection: `(PROJECT_ROOT.parent.parent.name == "ui-implementation")`
- Dev mode now uses `Roampal_DEV` data path for isolation

---

## Quality Assurance

### Test Coverage Summary
This release includes **876 passing tests** across **62 test files**:

| Suite | Tests | Files | Status |
|-------|-------|-------|--------|
| Frontend | 509 | 47 | ✅ All passing |
| Backend | 367 | 15 | ✅ All passing |
| **Total** | **876** | **62** | ✅ |

### Frontend Tests (509 tests, 47 files)
- **Components (36 files)**: All UI components from simple displays to complex modals
- **Stores (2 files)**: Zustand state management with tool interleaving, surfaced memories
- **Hooks (3 files)**: Custom React hooks including update checker and backend auto-start
- **Utils (6 files)**: Utility functions including fetch, logger, and file upload

Test infrastructure uses Vitest with React Testing Library.

### Backend Tests (367 tests, 15 files)
- **Unit Tests (15 files)**: All memory services, scoring, routing, KG, promotion
- **Security Tests**: ZIP path traversal, sensitive data filtering
- **MCP Tests**: Claude Code CLI detection, action caching, per-memory scoring

Key test files:
- `test_backup_security.py` - ZIP path traversal attack prevention (10 tests)
- `test_mcp_handlers.py` - Claude Code config detection, boundary detection (50+ tests)
- `test_agent_chat.py` - 4-emoji attribution parsing, humanize_age, caching (28 tests)
- `test_outcome_service.py` - Wilson score, success_count tracking (26 tests)
- `test_scoring_service.py` - Wilson formula, quality ranking (24 tests)
- `test_promotion_service.py` - success_count≥5 requirement, demotion (25 tests)

Test infrastructure uses pytest with asyncio support.

---

## Technical Details

### Files Changed
- `TerminalMessageThread.tsx` - TanStack Virtual migration, surfaced memories, streaming fixes, memo optimization, collapsible individual memories
- `ConnectedChat.tsx` - Store subscription optimization, resize handles
- `MemoryStatsPanel.tsx` - Tooltip fix, loading states
- `useChatStore.ts` - Store surfaced memories on messages
- `index.css` - Scrollbar styling (background colors applied at component level via zinc-950)
- `outcome_service.py` - Wilson score fix (success_count tracking)
- `scoring_service.py` - Wilson score fix (success_count parameter)
- `promotion_service.py` - Ported v0.2.9 promotion requirements (success_count≥5 for patterns, counter reset on history entry)
- `unified_memory_system.py` - Memory Context Sync, get_always_inject, _extract_concepts fix
- `agent_chat.py` - Send surfaced memories, _humanize_age(), simplified organic recall, outcome_memories rename, 4-emoji attribution system (🤷 partial), direct emoji→outcome scoring
- `main.py` - Updated MCP get_context_insights to use unified approach, added memory_scores to record_response, sort_by parameter for search_memory
- `settings.py` - Dev/prod data path isolation fix
- `knowledge_graph_service.py` - Migration fix for old routing patterns missing 'total' key
- `mcp.py` - Claude Code flat JSON format support for integrations Connect/Disconnect
- `outcome_detector.py` - Simplified for 4-emoji system (only detects overall outcome, not per-memory)

### Test Files Added
**Frontend (47 files):**
- Component tests (36 files) covering UI from MemoryCitation to KnowledgeGraph
- Store tests (2 files) with tool interleaving, surfaced memories coverage
- Hook and utility tests (9 files)
- Comprehensive mocking for API calls, Tauri, and config
- ResizeObserver mock for virtualized component testing

**Backend (15 files):**
- `test_backup_security.py` - ZIP path traversal security (NEW)
- `test_mcp_handlers.py` - Claude Code CLI detection, per-memory scoring (EXPANDED)
- `test_agent_chat.py` - 4-emoji parsing, humanize_age, caching
- Service tests for outcome, scoring, routing, KG, promotion, memory_bank, search, context
- `test_sensitive_data_filter.py` - API key, password, SSN redaction

### Dependencies
- @tanstack/react-virtual (replaces react-window for message virtualization)
- vitest, @testing-library/react (dev dependencies for testing)

---

## Late Bug Fixes (Added to v0.3.0)

The following issues were discovered during final testing and fixed before release:

### ✅ FIXED: Duplicate Prompts / Messages

**Symptoms:** Users saw message content repeated or duplicated in the chat.

**Root Cause:** Race condition in task cancellation - new task started before old task fully cancelled.

**Fix:** (`agent_chat.py:3526-3540`)
```python
# v0.3.0: Cancel existing task FIRST, await completion, THEN create new task
existing_task = _active_tasks.get(conversation_id)
if existing_task and not existing_task.done():
    existing_task.cancel()
    await asyncio.wait_for(asyncio.shield(existing_task), timeout=2.0)
```

---

### ✅ FIXED: Duplicate User Message with Context Injection

**Symptoms:** Model received duplicate user messages, causing confusion and repetitive/spam responses. Logs showed:
```
Message 5 (user): what else do you need to know
Message 6 (user): what else do you need to know
```

**Root Cause:** The new v0.3.0 context injection (line 736) adds a system message AFTER the user message. The old `[:-1]` slicing removed the last item (system context), not the user message. Result: user message in both `history` AND `prompt`.

**Fix:** (`agent_chat.py:707, 817`)
```python
# Track history length BEFORE adding user message
history_len_before_user = len(self.conversation_histories[conversation_id])
# ... user message added, context injection may add more ...
# Use tracked length instead of [:-1]
history_without_current = conversation_history[:history_len_before_user]
```

---

### ✅ FIXED: Cancel Button Not Stopping Generation

**Symptoms:** Clicking the X (cancel) button during chat didn't stop the model from generating. Response continued streaming despite clicking cancel.

**Root Cause:** `cancelProcessing()` only aborted the HTTP controller but didn't close the WebSocket. Since streaming happens over WebSocket, the stream continued.

**Fix:** (`useChatStore.ts:cancelProcessing`)
```typescript
// 1b. Close WebSocket to stop streaming (v0.3.0 fix)
if (state.websocket && state.websocket.readyState === WebSocket.OPEN) {
  console.log('[Cancel] Closing WebSocket to stop streaming');
  state.websocket.close();
  set({ websocket: null, connectionStatus: 'disconnected' });
}
```

---

### ✅ FIXED: Data Deletion Not Working

**Symptoms:** Delete button in Memory Bank didn't delete memories. No error shown.

**Root Cause:** Frontend silently failed when delete API returned error.

**Fix:** (`MemoryBankModal.tsx:113-175`)
- Added Toast component for user feedback
- Error handling on all memory operations (archive, restore, delete)
- Shows success/error messages to user

---

### ✅ FIXED: Corrupted Working Memories (v0.2.3 Migration)

**Symptoms:** Old installations from v0.2.3 had corrupted working memories after ChromaDB 1.x migration.

**Root Cause:** Schema migrations weren't wrapped in transactions.

**Fixes:**
1. **Transaction wrapper** (`unified_memory_system.py:256-307`)
   - `BEGIN TRANSACTION` / `COMMIT` / `ROLLBACK` around migrations
   - Validation before commit

2. **Startup health check** (`unified_memory_system.py:309-396`)
   - Tests all collections on startup
   - Auto-repairs corrupted `working` collection (temporary data, acceptable loss)
   - Logs clear diagnostics for other collection issues

---

### ✅ FIXED: Model Switching Race Condition

**Symptoms:** Switching models rapidly caused corrupted state - wrong model name in error logs, runtime model showing incorrect value.

**Root Cause:** No lock on `switch_model()` endpoint - multiple concurrent switches corrupted shared state.

**Fixes:** (`model_switcher.py`)
1. **Switch lock** - Added `_switch_lock` to serialize model switches
2. **Download cleanup** - Error handlers now delete partial `.gguf` files
3. **State validation** - Verify model state after successful switch
4. **Ghost cleanup endpoint** - New `GET/DELETE /ghost-models` to detect and remove orphaned models
5. **Model name mapping** - Fixed `llama3.3:70b` not matching `llama-3.3-70b-instruct` (installed models showed "Install" button)
6. **Health check timeout** - Increased to 90s for both Ollama and LM Studio (larger models need time to load into VRAM)
7. **Better timeout errors** - "Model load timed out (no response within 90s). Try a smaller model or pre-load it."
8. **Data Management stats 500** - Added missing `kg_path` property to `UnifiedMemorySystem` (`unified_memory_system.py:1496-1499`)
9. **CRITICAL: Health check nuking working memory** - Health check used `query_texts` which fails on embedding dimension mismatch, falsely triggering repair that deleted working collection on every boot (`unified_memory_system.py:312-361`)

**UI Updates:** (`ConnectedChat.tsx`)
- Updated `qwen2.5:7b` description: "Good tool calling (may struggle with 20+ tools)" - removed "recommended" badge
- 7B models can output tool JSON as text when too many MCP tools are loaded

---

### ✅ FIXED: Message Flicker (Appear → Disappear → Reappear)

**Symptoms:** When sending a message, the assistant response would appear, then disappear, then reappear - causing a visual flicker. User described: "thinking dots appear, response appears quick then disappears then reappears"

**Root Causes:** Two race conditions in WebSocket event handlers:

1. **`stream_start` stale state overwrite:**
   - When WebSocket is connected, `sendMessage()` doesn't create a placeholder (lets token handler create message lazily)
   - First `token` event arrives and creates assistant message with content
   - `stream_start` event arrives (slightly delayed) with a stale state reference
   - `stream_start` was returning `messages: [...state.messages]` unconditionally - this stale copy overwrote the token-created message
   - Result: message appears → disappears → reappears (next token recreates)

2. **`response` handler falsy empty string bug:**
   - Guard condition was `if (lastMsg.streaming && lastMsg.content)`
   - For tool-only messages, `lastMsg.content` is `''` (empty string)
   - Empty string is **falsy** in JavaScript, so guard failed
   - Handler overwrote the streaming message and set `streaming: false`
   - Result: message content cleared mid-stream

**Fixes:**

1. **`stream_start` handler** (`useChatStore.ts:623-654`):
   - Only returns `messages` array if there's actually a placeholder to modify
   - Added condition: `if (lastMsg?.sender === 'assistant' && !lastMsg.content && !lastMsg.streaming)`
   - If no placeholder exists, only updates processing state without touching messages array

2. **`response` handler** (`useChatStore.ts:697-738`):
   - Changed guard from `lastMsg.streaming && lastMsg.content` to just `lastMsg.streaming`
   - Now properly skips all streaming messages regardless of content

---

### ✅ FIXED: Content Flicker When Toggling Side Panels

**Symptoms:** When expanding or collapsing side panels, the chat content would briefly disappear and reappear, or text from different messages would overlap momentarily.

**Root Cause:** Calling `virtualizer.measure()` cleared ALL cached item heights at once. On the next render frame, items were positioned using the 80px `estimateSize` instead of actual heights, causing overlap for messages taller than 80px.

**Failed approaches:**
1. **Immediate measure()** - Caused 1-frame overlap (heights reset to estimates)
2. **150ms debounce** - Text overlap persisted during debounce window
3. **Dual-threshold** - Still had 1-frame overlap on large changes

**Final Fix:** (`TerminalMessageThread.tsx:608-633`)
- **Removed `measure()` entirely** - Don't nuke cached heights
- Let `measureElement` ref handle height updates naturally via its per-item ResizeObservers
- When container width changes → text reflows → item heights change → `measureElement` auto-detects
- Only preserve scroll position on width change (no cache invalidation)

This eliminates all flicker - the virtualizer smoothly adapts to width changes without any frame where items use estimated heights.

---

### ✅ FIXED: Visible Gap When Side Panels Collapsed

**Symptoms:** When one side panel is expanded and the other is collapsed, a visible gap/line appeared where the collapsed panel's resize handle was.

**Root Cause:** The resize handles were always rendered (5px wide with visible background color) regardless of whether the panel was collapsed.

**Fix:** (`ConnectedChat.tsx:1873-1889, 2209-2225, 1838`)
- Resize handles now only render when their corresponding panel is expanded
- Wrapped resize handle divs in `{!leftPane.isCollapsed && ...}` and `{!rightPane.isCollapsed && ...}`
- Added `bg-black` to the main flex container to hide any transient layout gaps
- This eliminates the visual gap when panels are collapsed

---

### ✅ FIXED: Data Management Delete Buttons Disabled

**Symptoms:** All delete buttons in Settings > Data Management showed "0 items" and were disabled, even when data existed.

**Root Cause:** If the `/api/data/stats` endpoint returned an error (e.g., backend not fully started, memory system not initialized), the error was silently ignored. All counts defaulted to 0, disabling all delete buttons.

**Fix:** (`DataManagementModal.tsx:109-129, 557-575`, `unified_memory_system.py:1496-1499`)
- Added `statsError` state to track and display errors
- Show clear error message with "Retry" button when stats fail to load
- Improved error handling in delete operations (handle non-JSON responses)
- Added missing `kg_path` property to `UnifiedMemorySystem` (backend was returning 500 when accessing knowledge graph stats)

---

### ✅ FIXED: Model Warning Inconsistencies

**Symptoms:** Frontend showed warnings like "May output tool JSON with 20+ tools" but Roampal only has 4 tools. Backend described `qwen2.5:7b` as "Best-in-class tool calling" while frontend flagged it with warnings.

**Root Cause:** Stale warnings from early development when the system was designed for MCP with many tools. Frontend and backend model descriptions were inconsistent.

**Fixes:**

1. **Frontend warnings updated** (`ConnectedChat.tsx`):
   - `qwen2.5:7b`: "⚠️ Tool calling may be unreliable" (was "May output JSON with 20+ tools")
   - `llama3.2:3b`: "⚠️ May output JSON instead of calling tools" (was "May output tool JSON as text")
   - Removed all "20+ tools" references (Roampal has 4 tools)

2. **Backend tier demotions** (`model_registry.py`):
   - `qwen2.5:7b` demoted from "verified" (✅) to "compatible" (⚠️)
   - `qwen2.5:3b`, `llama3.1:8b`, `llama3.2:3b` remain in "compatible" with clearer descriptions

3. **Backend descriptions updated** (`model_registry.py`):
   - `qwen2.5:7b`: "Tool calling may be unreliable" (was "Best-in-class tool calling")
   - `qwen2.5:3b`: "May have inconsistent tool calling" (was "Efficient with tool support")
   - `llama3.1:8b`: "Unreliable tool calling" (was "Compact Llama - Good tools, 128K context")
   - `llama3.2:3b`: "May output JSON instead of calling tools" (was "Ultra-compact Llama - May have inconsistent tools")

**Note:** Small models (≤8B) can struggle with verbose system prompts + context injection (~9K chars / ~2300 tokens) plus tools schema. Users with 7B models should ensure adequate context window size in their inference backend (LM Studio/Ollama).

---

### ✅ FIXED: Health Check Deleting Working Memories on Reboot

**Symptoms:** Working memories disappeared after every app restart, even though they weren't expired.

**Root Cause:** The v0.3.0 health check fix only handled dimension/embedding mismatch errors. Empty collections triggered `peek()` failure with "Nothing found on disk" error, which was incorrectly flagged as corruption.

**Fix:** (`unified_memory_system.py:332-336`)
- Skip `peek()` test if collection `count() == 0`
- Empty collections aren't corrupt, they're just empty

```python
# v0.3.0 FIX #2: Skip peek on empty collections - they're not corrupt, just empty
if count == 0:
    logger.debug(f"[HEALTH] {name} is empty (count=0), skipping peek test")
    continue
```

---

### ✅ FIXED: Context Overflow Causing Model Barfing

**Symptoms:** Small models (7B-14B) would generate spam/garbage when history + context injection exceeded context window. No warning was shown - Ollama/LM Studio silently truncated.

**Root Cause:** No pre-flight check for context overflow. System prompt (~2300 tokens) + context injection + history + tools could exceed model's context window, causing silent truncation from the beginning.

**Fix:** (`agent_chat.py:841-852`)
- Added pre-flight context estimation before LLM call
- Warns at 70% context usage, auto-truncates history at 90%
- **User-facing warning**: Shows inline message when truncation happens
- Logs estimated tokens vs context window

```python
# v0.3.0: Pre-flight context check
if estimated_tokens > model_context_window * 0.9:
    logger.warning(f"[CONTEXT OVERFLOW] Estimated {estimated_tokens} tokens exceeds 90%...")
    if history_to_pass and len(history_to_pass) > 2:
        old_count = len(history_to_pass)
        history_to_pass = history_to_pass[-2:]
        # Re-estimate and log post-truncation
        post_tokens = (len(system_instructions) + len(message) + ...) // 4
        logger.warning(f"[CONTEXT OVERFLOW] Truncated {old_count}→2 messages. Post: ~{post_tokens} tokens")
        # Inline warning to user (type: "token" for WebSocket forwarding)
        yield {"type": "token", "content": "*⚠️ Context limit reached...*\n\n"}
```

**User sees:** `*⚠️ Context limit reached (95% full). Trimmed history from 6 to 2 messages to prevent model barfing.*` followed by the normal response.

---

## Knowledge Graph Architecture

Roampal uses **three separate JSON files** for knowledge graph data:

| File | What it stores | When it grows | Used by |
|------|----------------|---------------|---------|
| `content_graph.json` | Concept nodes + relationships extracted from memories | Every time you add/promote memories | UI visualization (the nodes/edges you see) |
| `knowledge_graph.json` | Tool success patterns (e.g., "search_memory worked for identity questions") | When you score tool outcomes | Routing decisions (which tool for which problem) |
| `memory_relationships.json` | Links between memories (related, evolved-from, conflicts-with) | When promotion/scoring detects connections | Memory clustering |

### How KG Data Accumulates

1. **You chat** → memories created in `working` collection
2. **Memories promoted** → concepts extracted → `content_graph.json` grows
3. **You score responses** → tool patterns recorded → `knowledge_graph.json` grows
4. **Related memories detected** → links stored → `memory_relationships.json` grows

The **UI visualization** shows `content_graph.json` - these are the concept nodes and edges you see in the Knowledge Graph panel.

### ✅ FIXED: KG Delete/Stats Not Counting All Data

**Symptoms:** Knowledge Graph visualization showed nodes, but Data Management showed "0 items" for KG.

**Root Cause:** Stats and delete endpoints only looked at `knowledge_graph.json`, ignoring `content_graph.json` (which is what the UI visualizes).

**Fix:** (`data_management.py:77-121, 455-539`)
- Stats now count nodes/edges from BOTH files
- Delete now clears ALL THREE files:
  - `knowledge_graph.json` (routing patterns)
  - `content_graph.json` (concept visualization)
  - `memory_relationships.json` (memory links)

### ✅ FIXED: Qwen Dumping Raw Tool Results

**Symptoms:** After calling `search_memory`, Qwen would dump the raw memory results as output instead of synthesizing a natural response.

**Root Cause:** After tool execution, the continuation call used an empty prompt (`prompt=""`). Strong models (Claude/GPT-4) naturally synthesize a response from injected tool results. Weaker models like Qwen echo the raw results.

**Fix:** (`agent_chat.py:1036-1039`)
- Added minimal continuation prompt: `"Now respond to the user based on what you found."`
- Strong models ignore it (they'd respond naturally anyway)
- Qwen/smaller models get explicit direction to synthesize instead of echo

---

### ✅ FIXED: Content Graph Not Populating

**Symptoms:** KG visualization showed no nodes from memory_bank. Entity extraction was documented but never wired up.

**Root Cause:** MemoryBankService never called `kg_service.add_entities_from_text()`.

**Fix:** (`memory_bank_service.py`, `unified_memory_system.py:471`)
- Added `kg_service` parameter to MemoryBankService
- `store()` / `update()` / `archive()` / `delete()` now update content graph

---

### ✅ FIXED: Sessions Delete Missing Archive Folder

**Symptoms:** Deleting sessions left archived conversations behind.

**Root Cause:** Used `*.jsonl` instead of `**/*.jsonl`.

**Fix:** (`data_management.py:436`) - Recursive glob now includes `sessions/archive/`

---

### ✅ FIXED: Outcomes Not Deletable

**Symptoms:** No way to clear outcomes.db from UI.

**Fix:** (`data_management.py`, `DataManagementModal.tsx`) - Added Outcomes delete button

---

### ✅ FIXED: Backup Missing Content Graph

**Symptoms:** Export didn't include content_graph.json (KG visualization data).

**Fix:** (`backup.py`) - Added content_graph.json to backup/restore

---

### ✅ FIXED: Routing KG Never Saving (Race Condition)

**Symptoms:** `knowledge_graph.json` always empty. Routing patterns and action effectiveness never persisted.

**Root Cause:** `json.dump()` iterates the dictionary during serialization, but concurrent coroutines modify it simultaneously → "dictionary changed size during iteration" error on every save attempt.

**Fix:** (`knowledge_graph_service.py:158`)
- Deep copy before serialization: `kg_snapshot = copy.deepcopy(self.knowledge_graph)`
- Bug existed since v0.2.7 service extraction but went unnoticed until KG features were actively used

---

### ✅ FIXED: Outcome Detection False "Failed" on Thank You Messages

**Symptoms:** User says "ty" or "thanks" → system marks all memories as "failed" instead of "worked".

**Root Cause:** Outcome detection prompt said "Judge both user feedback AND response quality". Small LLMs focused on "response quality" (seeing "generic response") instead of user satisfaction ("ty" = thanks).

**Fix:** (`outcome_detector.py:101-112`)
- Simplified prompt from ~200 words to ~50 words
- Changed focus: "Grade the USER'S REACTION (not the assistant's quality)"
- Removed `used_positions` complexity (main LLM handles per-memory attribution via 4-emoji system)
- Minimal JSON output: just `{"outcome": "worked|failed|partial|unknown"}`

```python
# v0.3.0 simplified prompt
prompt = f"""Based on how the user responded, grade this exchange.

{conv_text}

Grade the USER'S REACTION (not the assistant's quality):
- worked = user satisfied (thanks, great, perfect, got it)
- failed = user unhappy/correcting (no, wrong, didn't work)
- partial = lukewarm (ok, I guess, sure)
- unknown = no clear signal

Return JSON: {{"outcome": "worked|failed|partial|unknown"}}"""
```

---

## Upgrading

Download from [Gumroad](https://roampal.gumroad.com/l/mjxtfg) or build from source.

No data migration required - your memories, books, and settings will carry over automatically.

**v0.2.3 Users:** The startup health check will automatically detect and repair corrupted working memories.

---

*Built with care. Tested thoroughly. 876 tests passing.*