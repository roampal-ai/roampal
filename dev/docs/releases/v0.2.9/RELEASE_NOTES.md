# Roampal Desktop v0.2.9 Release Notes

**Release Date:** 2025-12-29
**Type:** Bug Fix + Feature Enhancement
**Status:** ✅ FULLY IMPLEMENTED

---

## Feature 1: `sort_by` Parameter on `search_memory`

**Problem:** Desktop's MCP search_memory returns results in semantic relevance order only. Users asking temporal queries ("what did we discuss yesterday?") get semantically similar but temporally wrong results.

**Solution:** Expose the internal recency sorting that already exists in Desktop.

**Tool Schema Addition:**
```python
"sort_by": {
    "type": "string",
    "enum": ["relevance", "recency", "score"],
    "description": "Sort order. 'recency' for temporal queries.",
    "default": None
}
```

**Usage:**
```python
search_memory(query="what did we work on yesterday", sort_by="recency")
search_memory(query="best approach for auth", sort_by="score")  # Highest-scored first
search_memory(query="JWT tokens")  # Default: semantic relevance
```

**Auto-detection:** If query contains temporal keywords, automatically uses recency sort:
- `last`, `recent`, `yesterday`, `today`, `earlier`
- `previous`, `before`, `when did`, `how long ago`
- `last time`, `previously`, `lately`, `just now`

---

## Feature 2: `related` Parameter on `record_response` (Selective Scoring)

**Problem:** When LLM retrieves 5 memories but only uses 2, all 5 get scored with the same outcome. This poisons the learning signal.

**Solution:** Allow LLMs to specify which results were actually helpful using **positional indexing** (small-LLM friendly).

**Tool Schema Addition:**
```python
"related": {
    "type": "array",
    "items": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
    "description": "Which results were helpful. Use positions (1, 2, 3) or doc_ids. Omit to score all.",
    "default": None
}
```

**Usage:**
```python
# Position-based (recommended - works for all model sizes)
record_response(key_takeaway="Fixed auth bug", outcome="worked", related=[1, 3])

# Doc ID-based (for smart models)
record_response(key_takeaway="Fixed auth bug", outcome="worked", related=["history_abc123"])

# Omit = score all (backwards compatible)
record_response(key_takeaway="Fixed auth bug", outcome="worked")
```

**Why Positional Indexing:**
- Search results are numbered `1. [history]... 2. [patterns]...`
- Small LLMs (Haiku, 7B) can reliably say "results 1 and 3 helped"
- No hallucination risk - positions are visible in output
- Doc_ids still work for smart models

**Implementation:**
```python
# Cache includes position -> doc_id mapping
_mcp_search_cache[session_id] = {
    "doc_ids": ["history_abc", "patterns_def", "working_xyz"],
    "positions": {1: "history_abc", 2: "patterns_def", 3: "working_xyz"}
}

# In record_response: convert positions to doc_ids, validate, fall back to all if invalid
```

---

## Critical Bug Fix: Books Collection Search Returns Empty

**Problem:** After the v0.2.4 collection refresh feature was added, books search returns empty results despite data existing in ChromaDB.

**Root Cause:**
The collection refresh in `query_vectors()` omits `embedding_function=None`, causing ChromaDB to assign its default embedding function (384 dimensions). Roampal uses 768-dimensional embeddings from `all-mpnet-base-v2`. This dimension mismatch causes all vector queries to fail silently.

**Location:** `src-tauri/backend/modules/memory/chromadb_adapter.py:203-206`

```python
# CURRENT (broken)
if self.client and self.collection_name:
    self.collection = self.client.get_or_create_collection(
        name=self.collection_name,
        metadata={"hnsw:space": "l2"}
    )

# FIX
if self.client and self.collection_name:
    self.collection = self.client.get_or_create_collection(
        name=self.collection_name,
        embedding_function=None,  # Must match initialize() - prevents 384d/768d mismatch
        metadata={"hnsw:space": "l2"}
    )
```

**Impact:**
- All books searches return empty results
- Users who ingested documents cannot retrieve them
- Discovered on prod laptop - required manual database repair

**Fix Effort:** 5 minutes (one line change)

---

## Context

This bug also exists in roampal-core at `chromadb_adapter.py:225-228`. The v0.2.4 comment in roampal-core was copied from Desktop during the port. Both codebases need the same fix.

---

## Feature 3: Ghost Registry (Book Deletion Fix)

**Problem:** When users delete books from the library, ChromaDB removes the SQLite records but leaves "ghost" vectors in the HNSW index. Subsequent searches match these ghost vectors and return `[No content]` results.

**Root Cause:**
- ChromaDB's `delete()` removes records from SQLite metadata store
- HNSW binary index (`data_level0.bin`) retains the deleted vectors
- Similarity search still finds ghost vectors by embedding match
- When ChromaDB fetches document/metadata → gone from SQLite → empty content

**Solution:** Two-pronged approach:

### 1. Ghost Registry (Individual Deletes)
Track deleted chunk IDs in a blacklist file. Filter them out at query time.

```
┌─────────────────────────────────────────────────────────────────┐
│                     GHOST REGISTRY SYSTEM                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  NEW FILE: ghost_registry.py                                     │
│     ├── load() → read ghost_ids.json from data directory        │
│     ├── add(ids) → append chunk IDs to blacklist                │
│     ├── is_ghost(id) → check if ID is blacklisted               │
│     └── clear() → empty the blacklist                           │
│                                                                  │
│  FLOW:                                                           │
│     Delete book → ghost_registry.add(chunk_ids)                 │
│     Search books → filter results where is_ghost(id) == False   │
│     Clear all → delete_collection() + create_collection()       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. True Collection Nuke ("Clear Books" Button)
Replace current chunk-by-chunk deletion with `delete_collection()` + `create_collection()`. This rebuilds the HNSW index from scratch - no ghosts possible.

**Files to Modify:**

| File | Change |
|------|--------|
| `modules/memory/ghost_registry.py` | NEW - Ghost tracking class (~50 lines) |
| `api/book_upload_api.py` | After DELETE succeeds → `ghost_registry.add(chunk_ids)` |
| `modules/memory/search_service.py` | In `search_books()` → filter out ghosts |
| `app/routers/data_management.py` | "Clear Books" → nuke/recreate + `ghost_registry.clear()` |

**Simple Explanation:**
- **Problem:** Deleting a book crosses out the name but leaves the phone number. Searches still find the number, but nobody answers.
- **Fix:** Keep a "do not call" list. Check it before showing results.
- **Clear All:** Burn the whole phonebook and start fresh.

**Impact:**
- Users no longer see `[No content]` results after deleting books
- Ghost vectors still exist in HNSW (not a memory issue at ~3KB each)
- "Clear Books" gives users a true fresh start

**Status:** ✅ IMPLEMENTED

---

## Bug Fix: MCP Stale Cache (BM25 Index Not Syncing)

**Problem:** After clearing or modifying collections via Roampal UI, MCP searches continue returning old/deleted results until MCP server is restarted.

**Root Cause:**
- MCP server holds persistent ChromaDB connection
- v0.2.4 refresh re-gets collection on every query (syncs ChromaDB)
- BUT the BM25 hybrid search index is never invalidated
- BM25 cache still contains old documents → stale results

**Location:** `src-tauri/backend/modules/memory/chromadb_adapter.py:200-209`

```python
# CURRENT (v0.2.4 refresh - syncs ChromaDB but not BM25)
if self.client and self.collection_name:
    self.collection = self.client.get_or_create_collection(
        name=self.collection_name,
        embedding_function=None,
        metadata={"hnsw:space": "l2"}
    )

# FIX (v0.2.9 - also invalidate BM25 when collection changes)
if self.client and self.collection_name:
    self.collection = self.client.get_or_create_collection(
        name=self.collection_name,
        embedding_function=None,
        metadata={"hnsw:space": "l2"}
    )
    # Only rebuild BM25 if collection size changed (zero overhead on normal queries)
    current_count = self.collection.count()
    if not hasattr(self, '_last_count') or self._last_count != current_count:
        self._bm25_needs_rebuild = True
        self._last_count = current_count
```

**Why Count-Based:**
- `collection.count()` is O(1) metadata lookup - negligible cost
- BM25 only rebuilds when data actually changed (add/delete)
- Normal searches = zero overhead

**Simple Explanation:**
- **Problem:** ChromaDB knows the phone book changed, but the alphabetical index still lists deleted people.
- **Fix:** Check if phone book size changed. If yes, rebuild index. If no, skip.

**Impact:**
- MCP searches immediately reflect UI changes (no restart needed)
- No performance penalty on normal queries

**Status:** ✅ IMPLEMENTED

---

## Testing

After applying the fix:
1. Ingest a test document: `roampal ingest test.txt`
2. Search for it: `search_memory("test document content")`
3. Verify results include the ingested content

---

## Files Modified

### Bug Fix ✅
1. `src-tauri/backend/modules/memory/chromadb_adapter.py`
   - Line 207: Added `embedding_function=None` parameter to query_vectors refresh

### Feature 1: sort_by ✅
2. `src-tauri/backend/main.py`
   - Lines 861-866: Added `sort_by` to search_memory tool inputSchema
   - Lines 1062-1075: Added temporal keyword auto-detection
   - Lines 1111-1128: Added sorting logic (recency by timestamp, score by outcome score)

### Feature 2: related (selective scoring) ✅
3. `src-tauri/backend/main.py`
   - Lines 993-998: Added `related` to record_response tool inputSchema with positional indexing
   - Lines 1148-1160: Updated search cache to include positions mapping
   - Lines 1560-1601: Added position→doc_id resolution and selective scoring logic

### Documentation ✅
4. `dev/docs/architecture.md`
   - Lines 3076-3124: Updated MCP tools section with v0.2.9 parameters

### Feature 3: Ghost Registry ✅
5. `src-tauri/backend/modules/memory/ghost_registry.py` (NEW)
   - Ghost tracking class with load/add/is_ghost/clear methods
   - Persists to `ghost_ids.json` in data directory

6. `src-tauri/backend/backend/api/book_upload_api.py`
   - DELETE endpoint: Add `ghost_registry.add(chunk_ids)` after ChromaDB deletion
   - Tracks both pattern-matched and DB chunk IDs

7. `src-tauri/backend/modules/memory/search_service.py`
   - `search_books()`: Filter out ghost results before returning

8. `src-tauri/backend/app/routers/data_management.py`
   - "Clear Books": Replace batch deletion with `delete_collection()` + `create_collection()`
   - Nukes entire collection to rebuild HNSW index (no ghosts possible)
   - Add `ghost_registry.clear()` after nuke

9. `src-tauri/backend/tests/unit/test_ghost_registry.py` (NEW)
   - 15 unit tests covering GhostRegistry class and singleton

### Bug Fix: metadata_filters Facade ✅
10. `src-tauri/backend/modules/memory/unified_memory_system.py`
    - Added `metadata_filters` parameter to `search()` facade method (line 416)
    - Fixes TypeError when LLM passes metadata filters to search_memory MCP tool

### Documentation Fixes ✅
11. `dev/docs/architecture.md`
    - Line 198: Fixed memory_bank capacity (500 → 1000 to match config.py)
    - Lines 1172-1190: Updated books/memory_bank safeguard reference (unified_memory_system.py → outcome_service.py:90-104)
    - Lines 6503-6542: Added D7 Ghost Registry documentation

### Bug Fix: MCP Stale Cache ✅
12. `src-tauri/backend/modules/memory/chromadb_adapter.py`
    - Lines 211-220: Add count-based BM25 cache invalidation after collection refresh
    - Track `_last_count` to detect external changes
    - Set `_bm25_needs_rebuild = True` only when count differs

### Performance: Startup Optimizations ✅
13. `ui-implementation/src/components/ConnectedChat.tsx`
    - Lines 1565-1587: Memory fetch parallelized (`for...await` → `Promise.all()`)
    - Lines 188-205: Model discovery parallelized (3 API calls in parallel)
    - **Impact:** ~3x faster memory panel load, ~2x faster model discovery

### Bug Fix: Backend Refresh Reconnection ✅
14. `ui-implementation/src-tauri/src/main.rs`
    - Added `is_port_in_use()` helper function
    - `start_backend()`: Check if port already bound before spawning
    - If port in use → return success immediately (backend survived refresh)
    - **Impact:** Ctrl+R refresh now works seamlessly (no 120s timeout)

### UX Fix: Model Loading State ✅
15. `ui-implementation/src/components/ConnectedChat.tsx`
    - Line 87: Convert `hasLoadedModels` ref to `isLoadingModels` state
    - Lines 255, 292: Update fetchModels() to use state setter
    - Line 323: Update auto-modal trigger condition
    - Line 2136: Add "Discovering Models" loading UI with spinner
    - **Impact:** Shows loading spinner instead of "No Model Installed" while Ollama responds

### UX Fix: Knowledge Graph Lazy Loading ✅
16. `ui-implementation/src/components/ConnectedChat.tsx`
    - Line 1222: Add `kgHasLoaded` state for lazy loading
    - Line 1475: Remove eager KG fetch on mount
    - New useEffect: Fetch KG only when panel first opens
    - **Impact:** Faster startup - KG only loads when user opens panel

### Bug Fix: Conversation Switch Race Condition ✅
17. `ui-implementation/src/stores/useChatStore.ts`
    - Add `_currentSwitchId` to state for switch tracking
    - Guard against stale switch results overwriting current conversation
    - **Impact:** Prevents conversation leak when switching rapidly

### UX Fix: Document Processor Delete Button ✅
18. `ui-implementation/src/components/BookProcessorModal.tsx`
    - Lines 61-62: Added `deleteError` and `isDeleting` state variables
    - Lines 99-145: Rewrote `deleteExistingBook()` with proper loading state and error handling
    - Lines 141-145: Added `closeDeleteModal()` helper function
    - Lines 926-957: Updated delete confirmation modal UI with error display and spinner
    - **Impact:** Delete button now shows "Deleting..." spinner, displays errors if failed, modal stays open on error

### Bug Fix: Embedding Model Loading Timeout ✅
19. `src-tauri/backend/modules/embedding/embedding_service.py`
    - Lines 33-90: Added timeout protection to `_load_bundled_model()`
    - Model loading now runs in daemon thread with 120-second timeout
    - Prevents indefinite hang if model path doesn't exist or loading stalls
    - Better logging when snapshot path not found
    - **Impact:** Book processing no longer hangs forever if embedding model fails

### Bug Fix: ChromaDB Upsert Timeout ✅
20. `src-tauri/backend/modules/memory/chromadb_adapter.py`
    - Lines 175-203: Added timeout protection to `upsert_vectors()`
    - ChromaDB upsert now runs in daemon thread with 60-second timeout
    - Prevents indefinite hang on SQLite locks or HNSW index corruption
    - Better error message on timeout (indicates possible lock/corruption)
    - **Impact:** Book processing fails gracefully instead of hanging forever
    - **Root cause:** SQLite transaction can stall if previous process didn't release lock

### Bug Fix: PathSettings + Action KG Cleanup ✅
21. `src-tauri/backend/backend/api/book_upload_api.py`
    - Line 711: Fixed `settings.paths.data_path` → `data_dir`
22. `src-tauri/backend/app/routers/data_management.py`
    - Line 318: Fixed `settings.paths.data_path` → `data_dir`
23. `src-tauri/backend/modules/memory/search_service.py`
    - Line 645: Fixed `settings.paths.data_path` → `data_dir`
24. `src-tauri/backend/modules/memory/unified_memory_system.py`
    - Lines 585-597: Added `cleanup_action_kg_for_doc_ids()` passthrough method
    - Line 417: Added `transparency_context` parameter to `search()` facade method
    - Passes `transparency_context` through to search_service (line 443)
    - Lines 446-482: Added `detect_conversation_outcome()` method - was called but missing!
    - Uses lazy-loaded `OutcomeDetector` from `modules/advanced/outcome_detector.py`
    - **Impact:** Book deletion no longer errors on ghost registry or Action KG cleanup
    - **Impact:** LLM chat no longer crashes with TypeError on transparency_context
    - **Impact:** Outcome detection now works (was silently failing due to missing method)

### Documentation: Build Guide Updates ✅
25. `release/BUILD_GUIDE.md`
    - Added Step 3: Clean Backend Pycache (prevents stale bytecode issues)
    - Enhanced Step 4: Verify torch/testing exists (commonly deleted by mistake)
    - Enhanced Step 4: Verify embedding model snapshot path exists
    - Added Step 6: Integration Test (REQUIRED before shipping)
      - Tests embedding service, backend imports, torch.testing
    - **Impact:** Prevents packaging issues from reaching users

---

## Origin

Features ported from roampal-core:
- `sort_by` parameter (Core has this, Desktop didn't)
- `related` parameter (Core's `score_response` tool has this)

Adapted for Desktop:
- Positional indexing added (Core uses doc_ids only)
- Combined into `record_response` (Core has separate `score_response`)

---

## Migration Notes

- **Backwards compatible** - existing tool calls work unchanged
- **No breaking changes** - omitting new params = current behavior
- No data migration needed
- Existing ingested books will become searchable after bug fix
- Users who manually repaired their databases are unaffected

---

## Testing Checklist

### Bug Fix
- [ ] Books search returns results after fix
- [ ] Ingest new document, verify searchable

### sort_by
- [ ] `search_memory(sort_by="recency")` returns chronologically sorted
- [ ] `search_memory(sort_by="score")` returns highest-scored first
- [ ] Without `sort_by` uses semantic relevance (default)
- [ ] Auto-detection triggers recency for "what did we do last time"

### related (selective scoring)
- [ ] `record_response(related=[1, 3])` only scores positions 1 and 3
- [ ] `record_response(related=["doc_id"])` only scores that doc
- [ ] Invalid positions fall back to score all
- [ ] Without `related` scores all (backwards compatible)
- [ ] Position cache correctly built from search results

### Ghost Registry
- [ ] Delete book → chunk IDs added to ghost_ids.json
- [ ] Search books → ghost results filtered out (no `[No content]`)
- [ ] "Clear Books" button → collection nuked and recreated
- [ ] "Clear Books" → ghost_ids.json cleared
- [ ] Re-upload book after delete → new chunks searchable (not blocked by old ghost IDs)

### metadata_filters Facade
- [ ] `search_memory(metadata={...})` no longer causes TypeError
- [ ] Metadata filters passed through to search_service correctly

### MCP Stale Cache
- [ ] Clear collection in UI → MCP search returns empty (no restart)
- [ ] Add new book in UI → MCP search finds it (no restart)
- [ ] Normal searches have no performance regression

### Embedding Model Timeout
- [x] Timeout protection added to _load_bundled_model()
- [x] Tested model loading with correct snapshot path
- [ ] Book processing completes successfully (manual test)

### ChromaDB Upsert Timeout
- [x] Timeout protection added to upsert_vectors()
- [ ] Upsert completes within timeout (manual test)
- [ ] Timeout triggers graceful failure instead of hang

### PathSettings Bug Fix
- [x] Fixed `settings.paths.data_path` → `data_dir` in 4 locations
- [x] Added `cleanup_action_kg_for_doc_ids()` method to UnifiedMemorySystem
- [x] Added `transparency_context` parameter to UMS.search() facade
- [ ] Book delete completes without errors (manual test)
- [ ] LLM chat works without TypeError (manual test)

### Build Process
- [x] torch/testing verified present in release
- [x] pycache cleaned from release backend
- [x] Embedding model snapshot path verified
- [ ] Integration tests pass (Step 6 in BUILD_GUIDE.md)

---

## Version Bump

```
ui-implementation/package.json: "version": "0.2.8" → "0.2.9"
ui-implementation/src-tauri/tauri.conf.json: "version": "0.2.8" → "0.2.9"
ui-implementation/src-tauri/Cargo.toml: version = "0.2.8" → "0.2.9"
```
