# Roampal Desktop v0.2.11 Release Notes

**Release Date:** 2026-01-10
**Type:** Critical Performance & Bug Fixes
**Status:** COMPLETE

---

## Overview

v0.2.11 is a critical release focused on resolving specific performance bottlenecks in the chat interface and knowledge graph, as well as fixing the Books search functionality.

---

## Critical Fixes

### 1. Chat Interface Lag (Store Subscriptions)

**Problem:** The chat interface suffers from significant input lag during typing and generation, caused by the main component re-rendering on every single state change (even irrelevant ones).

**Root Cause:** `ConnectedChat.tsx` subscribed to the entire application state object instead of selecting only specific data fields.

**Solution:**
Refactored `useChatStore` in `ConnectedChat.tsx` to use granular selectors:
```tsx
// Before: const { messages, ... } = useChatStore()
// After:
const conversationId = useChatStore(state => state.conversationId);
const connectionStatus = useChatStore(state => state.connectionStatus);
const messages = useChatStore(state => state.messages);
```

**Files Changed:**
-   `ui-implementation/src/components/ConnectedChat.tsx:46-50`

**Status:** ✅ COMPLETE

---

### 2. Message History Performance (Virtualization)

**Problem:** Long conversations cause the application to slow down, consume excessive memory, and eventually freeze.

**Root Cause:** The application rendered every single message in the history to the DOM, regardless of whether it is visible on screen.

**Solution:**
Implemented "virtualization" (windowed rendering) to only render messages currently visible in the viewport:
1. Added `react-window` dependency
2. Refactored `TerminalMessageThread.tsx` to use `VariableSizeList`
3. Memoized message components with `MessageRow` wrapper

**Files Changed:**
-   `ui-implementation/src/components/TerminalMessageThread.tsx:3,209-230`
-   `ui-implementation/package.json` (added react-window)

**Status:** ✅ COMPLETE

---

### 3. Knowledge Graph Loading Optimization

**Problem:** The Knowledge Graph visualization takes 20+ seconds to load, freezing the backend during the process.

**Root Cause:** Two separate algorithmic bottlenecks:
1. `get_kg_entities()` had O(n×m) complexity - looping through all relationships for every concept
2. Edge retrieval executed N+1 queries (one per node)

**Solution:**
1. Pre-indexed relationship counts with O(1) lookups in `knowledge_graph_service.py`:
```python
# v0.2.11: Pre-build routing connection counts (O(m) once, not O(n*m))
# Before: O(n*m) - looping through all relationships for every concept (~25s)
# After: O(m) build + O(n) lookups (<1s)
routing_connection_counts: Dict[str, int] = {}
for rel_key in self.knowledge_graph.get("relationships", {}).keys():
    for concept in rel_key.split("|"):
        routing_connection_counts[concept] = routing_connection_counts.get(concept, 0) + 1
```

2. Batch edge retrieval in `memory_visualization_enhanced.py` to avoid N+1 queries

**Files Changed:**
-   `ui-implementation/src-tauri/backend/modules/memory/knowledge_graph_service.py:739-753`
-   `ui-implementation/src-tauri/backend/app/routers/memory_visualization_enhanced.py:224-280`

**Status:** ✅ COMPLETE

---

### 4. Books Search Fix

**Problem:** Searching within the "Books" collection often returns empty results despite data being present.

**Root Cause:** The ChromaDB collection initialization used incorrect defaults for the embedding function in some contexts, leading to dimension mismatches or silent failures.

**Solution:**
Explicitly set `embedding_function=None` when retrieving the collection:
```python
embedding_function=None,  # Manual embeddings from EmbeddingService
```

**Files Changed:**
-   `ui-implementation/src-tauri/backend/modules/memory/chromadb_adapter.py:135,226-231`

**Status:** ✅ COMPLETE

---

### 5. Scrollbar Gap Fix

**Problem:** Gap between chat scrollbar and right memory panel.

**Root Cause:** Container had `p-6` (24px padding all sides) pushing scrollbar away from edge.

**Solution:**
Changed padding to `pl-6 pt-6 pb-6 pr-2` to minimize right padding:
```tsx
// Before: className="h-full w-full bg-zinc-950 p-6 overflow-hidden"
// After:  className="h-full w-full bg-zinc-950 pl-6 pt-6 pb-6 pr-2 overflow-hidden"
```

**Files Changed:**
-   `ui-implementation/src/components/TerminalMessageThread.tsx:472`

**Status:** ✅ COMPLETE

---

## Files Summary

| File | Changes |
|------|---------|
| `ConnectedChat.tsx` | Optimized store subscriptions |
| `TerminalMessageThread.tsx` | Message virtualization, scrollbar gap fix |
| `knowledge_graph_service.py` | O(n×m) → O(n+m) pre-indexed counts |
| `memory_visualization_enhanced.py` | Batch edge retrieval |
| `chromadb_adapter.py` | Fixed Books collection embedding_function |

---

## Testing Checklist

-   [x] Verify typing is responsive
-   [x] Verify scrolling is smooth in long chats
-   [x] Verify Knowledge Graph loads quickly (<1s)
-   [x] Verify Books search returns results
-   [x] Verify no gap between scrollbar and memory panel
