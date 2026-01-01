# Roampal Desktop v0.2.10 Release Notes

**Release Date:** TBD
**Type:** Critical Bug Fixes
**Status:** READY

---

## Overview

v0.2.10 is a critical stability release focused on ChromaDB error handling, schema compatibility, and memory lifecycle fixes. All UI/UX improvements have been moved to v0.3.0.

---

## Bug Fixes

### 1. ChromaDB Ghost Entry Error Handling

**Problem:** When ChromaDB has ghost entries (IDs in the HNSW index but no corresponding document data), calls to `get_fragment()` and `list_all_ids()` would throw "Error executing plan: Internal error: Error finding id" exceptions, causing application crashes.

**Root Cause:** ChromaDB's HNSW index can retain IDs for documents that were deleted or never fully written. When these ghost IDs are queried, ChromaDB throws an exception instead of returning empty results.

**Solution:** Added try/except wrappers around ChromaDB calls in `chromadb_adapter.py`.

**Files Changed:**
- `src-tauri/backend/modules/memory/chromadb_adapter.py` - Added error handling to `get_fragment()` and `list_all_ids()`

**Impact:** Prevents application crashes and allows graceful degradation when ghost entries exist. Users with corrupted ChromaDB state will see warnings in logs but the system continues to function.

**Status:** ✅ FIXED

---

### 2. ChromaDB Schema Migration

**Problem:** The bundled ChromaDB (~0.5.x) requires `topic` columns in the SQLite schema. Users with older data (from v0.2.1 or earlier) that lacked these columns would get crashes (`no such column: segments.topic`).

**Solution:** Added `_migrate_chromadb_schema()` in `unified_memory_system.py`:
- Adds missing `topic` columns to both `collections` and `segments` tables
- Runs automatically on startup before ChromaDB initialization
- Idempotent - safe to run multiple times
- Non-fatal - logs warning but doesn't crash if migration fails

**Files Changed:**
- `src-tauri/backend/modules/memory/unified_memory_system.py` - Added schema migration

**Status:** ✅ FIXED

---

### 3. Memory Promotion Not Running on Startup

**Problem:** The background task that promotes working memory to history and cleans up old entries (>24h) waited 30 minutes before its first run. If users closed the app before 30 minutes, promotion/cleanup never happened, causing old working memories to persist indefinitely.

**Root Causes:**
1. The `memory_promotion_task` had `await asyncio.sleep(1800)` at the START of the loop instead of the end.
2. Method name mismatch: code called `clear_old_working_memory()` but the actual method is `cleanup_old_working_memory()`, causing silent AttributeError.

**Solution:**
1. Moved the sleep to after the promotion work, so promotion runs immediately on startup, then every 30 minutes.
2. Fixed method name to `cleanup_old_working_memory()`.

**Files Changed:**
- `src-tauri/backend/main.py` - Reordered sleep and fixed method name

**Status:** ✅ FIXED

---

### 4. Backend Launcher Path Fix

**Problem:** The `start-backend.bat` in the release bundle had an incorrect path (`..\..\..\..\` - 4 levels up) which caused it to look for `main.py` outside the release folder. This meant the bundled backend code was never actually used.

**Root Cause:** The bat file was navigating to the wrong directory before launching Python.

**Solution:** Fixed path to `..\backend` which correctly points to the backend folder sibling to binaries/.

**Files Changed:**
- `binaries/start-backend.bat` - Fixed working directory path

**Status:** ✅ FIXED

---

### 5. Promotion Method Name Mismatch

**Problem:** The memory promotion task was calling `_promote_valuable_working_memory()` (with underscore prefix) but the actual public method is `promote_valuable_working_memory()` (no underscore), causing AttributeError on every promotion attempt.

**Root Cause:** Inconsistent method naming - the internal method has an underscore prefix but the code was using the wrong name.

**Solution:** Fixed method call to use `promote_valuable_working_memory()` (no underscore).

**Files Changed:**
- `src-tauri/backend/main.py` - Fixed method name in promotion task

**Status:** ✅ FIXED

---

### 6. Duplicate Cleanup Causing Ghost Entries

**Problem:** Old working memories appeared to persist in the UI even after cleanup ran successfully. Logs showed "removed 13 of 13 items" but stats still showed 13 items, and items reappeared on refresh.

**Root Cause:** The promotion task called two functions that both cleaned up old memories:
1. `promote_valuable_working_memory()` - already cleans up items > 24h at lines 348-351
2. `cleanup_old_working_memory()` - redundant call that tried to delete the same items

ChromaDB's `list_all_ids()` returned stale data immediately after deletion (HNSW index eventual consistency), causing the second cleanup to "delete" already-deleted items. This created ghost entries where IDs persisted in queries but documents were gone.

**Solution:** Removed the redundant `cleanup_old_working_memory()` call since `_do_batch_promotion()` already handles cleanup.

**Files Changed:**
- `src-tauri/backend/main.py` - Removed redundant cleanup call

**Status:** ✅ FIXED

---

### 7. ChromaDB Upgraded to 1.x (Aligned with roampal-core)

**Problem:** Working memories appeared to be deleted (logs showed "Cleaned up X items") but the same items reappeared on every app restart. Direct SQLite query confirmed the items were still in the database.

**Root Cause:** ChromaDB 0.4.18 does not auto-persist delete operations in embedded `PersistentClient` mode. The `collection.delete()` call succeeds in memory but changes aren't flushed to SQLite.

**Solution:** Upgraded ChromaDB from 0.4.18 to 1.x to align with roampal-core. ChromaDB 1.x properly auto-persists all operations including deletes.

**Files Changed:**
- `backend/requirements.txt` - Updated `chromadb==0.4.18` to `chromadb>=1.0.0,<2.0.0`

**Status:** ✅ FIXED

---

### 8. ChromaDB Query Timeout Protection

**Problem:** The app would freeze/die when sending messages. UI became completely unresponsive, requiring force quit.

**Root Cause:** `self.collection.query()` in ChromaDB is synchronous and blocks the Python async event loop. When ChromaDB hangs (first-use model loading, HNSW index corruption, or SQLite locks), the entire UI freezes because the async event loop is blocked.

**Solution:** Wrapped ChromaDB queries in `ThreadPoolExecutor` with `asyncio.wait_for()` timeout:
- `chromadb_adapter.py`: 10-second timeout on `query_vectors()`
- `search_service.py`: 15-second timeout on `_search_single_collection()`

The thread executor allows the async timeout to actually cancel the operation, since Python asyncio cannot interrupt blocking synchronous calls.

**Files Changed:**
- `src-tauri/backend/modules/memory/chromadb_adapter.py` - Added 10s timeout with ThreadPoolExecutor
- `src-tauri/backend/modules/memory/search_service.py` - Added 15s timeout wrapper

**Impact:** If ChromaDB hangs, the query returns empty results after timeout instead of freezing forever. Users see an error in logs but the app remains responsive.

**Status:** ✅ FIXED

---

## Files Summary

| File | Changes |
|------|---------|
| `backend/requirements.txt` | Upgraded ChromaDB from 0.4.18 to 1.x |
| `src-tauri/backend/modules/memory/chromadb_adapter.py` | Ghost entry error handling, 10s query timeout |
| `src-tauri/backend/modules/memory/search_service.py` | 15s search timeout protection |
| `src-tauri/backend/modules/memory/unified_memory_system.py` | Schema migration for topic columns |
| `src-tauri/backend/main.py` | Memory promotion runs on startup, fixed method names, removed duplicate cleanup |
| `binaries/start-backend.bat` | Fixed backend path |

---

## Upgrade Notes

### From v0.2.9
- No breaking changes
- Automatic schema migration handles ChromaDB version differences
- Ghost entry errors now handled gracefully

### Risk Assessment
- **Low risk** - Both fixes are defensive and non-breaking
- Ghost entry fix: Returns empty/None instead of crashing
- Schema migration: Adds missing columns, doesn't modify existing data

---

## Testing Checklist

- [ ] Clean install works
- [ ] Upgrade from v0.2.9 works
- [ ] Ghost entries in ChromaDB don't cause crashes
- [ ] Schema migration runs silently on fresh install (no-op)
- [ ] Schema migration adds missing columns on upgrade
- [ ] Memory search works after migration
- [ ] Memory bank operations work after migration
- [ ] Memory promotion runs immediately on startup (check logs)
- [ ] Old working memories (>24h) get cleaned up on startup

---

## What Moved to v0.3.0

The following UI/UX improvements were originally planned for v0.2.10 but have been moved to v0.3.0:

- StatRow tooltip positioning fix
- Voice Settings button disabled state
- Settings category grouping
- Memory stats loading spinner
- Resize handle visibility improvements
- Toast system (React state-based)
- Memory search in main view
- Message bubble styling
- Model selector enhancement
- Enhanced "Thinking" indicator
- Enhanced tool chain display
- Knowledge Graph visual improvements (gradients, hover, legend, node sizing)
- Backend context improvements (identity injection, tags visibility)

See [v0.3.0 Release Notes](../v0.3.0/RELEASE_NOTES.md) for details.