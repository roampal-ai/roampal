# Known Issues

## Book Library Deletion Not Fully Working

**Location**: Document Processor > Manage Library

**Symptom**: Deleting books from the library appears to succeed but ghost vectors remain in ChromaDB's HNSW index. Subsequent searches return `[No content]` for deleted book chunks.

**Root Cause**:
- ChromaDB deletion removes records from SQLite and marks items as deleted
- However, the HNSW binary index files (`data_level0.bin` in segment folders) retain the deleted vectors
- Similarity search still matches these "ghost" vectors
- When ChromaDB tries to fetch document/metadata, they're gone, resulting in empty content

**Evidence from logs**:
```
chromadb.segment.impl.vector.local_persistent_hnsw - WARNING - Delete of nonexisting embedding ID: a3b230db-...
```

**Current Workaround**:
1. Use Data Management to clear the entire books collection
2. Re-upload all books

**Permanent Fix Needed**:
- Force HNSW index rebuild after deletions
- Or implement proper segment compaction that removes deleted vectors from binary index files
- ChromaDB's `compact()` doesn't rebuild HNSW indexes

**Files Involved**:
- `backend/api/book_upload_api.py` - DELETE endpoint (lines 611-742)
- `modules/memory/chromadb_adapter.py` - `upsert_vectors`, `query_vectors`
- ChromaDB segment folders: `data/chromadb/{segment-uuid}/data_level0.bin`

**Discovered**: 2025-12-09

---

## MCP Server Caches Stale ChromaDB State

**Location**: MCP Server (Claude Desktop / Claude Code integration)

**Symptom**: After clearing or modifying collections via the Roampal UI, MCP searches continue returning old/deleted results until MCP is restarted.

**Root Cause**:
- MCP server holds a persistent ChromaDB client connection
- ChromaDB's `PersistentClient` caches collection state in memory
- Changes made by other processes (Roampal UI) aren't visible to the cached connection
- The `get_or_create_collection()` refresh in `query_vectors` doesn't fully sync HNSW index state

**Current Workaround**:
1. After clearing collections in Roampal UI, restart the MCP server
2. In Claude Desktop: disconnect and reconnect to Roampal MCP
3. In Claude Code: restart the CLI session

**Permanent Fix Needed**:
- Add a `refresh_collections` MCP tool that forces ChromaDB client reconnection
- Or implement collection version tracking to detect external changes
- Or use a fresh client per query (performance tradeoff)

**Files Involved**:
- `main.py` - MCP server initialization and tool handlers
- `modules/memory/chromadb_adapter.py` - `_ensure_initialized()`, `query_vectors()`

**Discovered**: 2025-12-09
