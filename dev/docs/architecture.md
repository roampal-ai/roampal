# Roampal Architecture

## Overview

Roampal is an intelligent chatbot with persistent memory and learning capabilities. The system features a **memory-first** architecture that learns from conversations and improves over time.

## Memory module structure (v0.3.3)

The memory subsystem follows a **facade pattern**: a thin `UnifiedMemorySystem` (`modules/memory/unified_memory_system.py`) coordinates a set of single-responsibility services. The original 4,746-line monolith was extracted into 8 services in v0.2.7; v0.3.1 added three more for the sidecar + tag stack (KG was removed in the same release).

| Service | Purpose |
|---|---|
| `unified_memory_system.py` | Facade — public API, service coordination, tag extraction on store |
| `search_service.py` | TagCascade retrieval (tag-routed overlap cascade + CE rerank) |
| `chromadb_adapter.py` | Vector DB operations, BM25 hybrid search |
| `smart_book_processor.py` | Book ingestion, chunking, contextual embedding |
| `promotion_service.py` | Working → History → Patterns promotion pipeline |
| `context_service.py` | Conversation context analysis |
| `routing_service.py` | Query routing, acronym expansion |
| `memory_bank_service.py` | Memory bank CRUD (cap: 500 items) |
| `outcome_service.py` | Outcome recording, score updates |
| `scoring_service.py` | Wilson scoring, score calculations |
| `sidecar_service.py` (v0.3.1) | Background LLM scoring + atomic-fact extraction |
| `sidecar_queue.py` (v0.3.1) | Retry queue for failed sidecar tasks |
| `tag_service.py` (v0.3.1) | LLM-based noun-tag extraction + word-boundary matching |
| `types.py` | Shared types, `ActionOutcome` enum |
| `config.py` | Configuration constants |

The facade keeps a stable public surface (`search()`, `store()`, `record_outcome()`, `get_stats()`, `get_cold_start_context()`, `export_backup()`, `import_backup()`, etc.) so router code does not need to know which service answers each call. ChromaDB collection names use the `roampal_` prefix for compatibility with embedded-mode persistence.

Line numbers throughout the rest of this document may point at the pre-refactor monolith and should be treated as approximate.

## Performance Benchmarks

### Headline Result (v0.3.1 — roampal-labs LoCoMo)

> **TagCascade retrieval: 27.3% Hit@1 on LoCoMo (1,537 questions, p<0.0001 vs Wilson+CE blend).**

See [roampal-labs](https://github.com/roampal-ai/roampal-labs) for full benchmark suite and methodology.

---

### LoCoMo Evaluation (v0.3.1)

From roampal-labs LoCoMo evaluation (1,537 non-adversarial questions):

| Config | Hit@1 Clean | Hit@1 Poison | p-value |
|--------|-------------|--------------|---------|
| **TagCascade + cosine** | **27.3%** | **29.0%** | **baseline** |
| Overlap + cosine | 25.8% | 28.0% | p=0.0003 |
| Pure CE | 25.4% | 28.4% | — |
| TagCascade + Wilson | 23.0% | 25.0% | p<0.0001 |

**Key findings:**
- Wilson scoring hurts retrieval by 4.3 points in every configuration tested
- Two-lane retrieval adds +6.1 Hit@1 (p<0.0001)
- Nursery slot: zero benefit (p=1.0)

### Performance Metrics Summary (v0.3.1 LoCoMo)

| Metric | Measured Performance | Source |
|--------|---------------------|--------|
| TagCascade Hit@1 (clean) | 27.3% on 1,537 questions | roampal-labs LoCoMo |
| TagCascade Hit@1 (poison) | 29.0% on 1,537 questions | roampal-labs LoCoMo |
| Wilson harm vs cosine | −4.3 points (p<0.0001) | roampal-labs LoCoMo |
| Two-lane retrieval boost | +6.1 Hit@1 (p<0.0001) | roampal-labs LoCoMo |
| Nursery slot effect | 0 (p=1.0) | roampal-labs LoCoMo |

Older "learning curve" / "token efficiency" / "test suite count" claims that previously sat in this table referenced the v0.2.5 `benchmarks/comprehensive_test/` suite, not the v0.3.1 LoCoMo run. Those numbers are still cited in `releases/v0.2.5/RELEASE_NOTES.md` against their own methodology — they are not directly comparable to the LoCoMo numbers above and have been removed from this table to keep the citation clean.

### Design Principles

1. **Stable Core** - Memory system always works reliably
2. **Learn from Interaction** - Improves through conversation patterns
3. **Graceful Degradation** - System works even if advanced features fail
4. **Privacy First** - All data stored locally
5. **Conversational Intelligence** - Natural dialogue with context awareness

## System Layers

```
┌──────────────────────────────────────────────────┐
│                   UI Layer                       │
│         (Tauri + React + TypeScript)             │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│                API Layer                         │
│         (FastAPI + WebSocket)                    │
│    /api/agent/stream  /api/memory/*              │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│            Service Layer                         │
│  ┌──────────────────────────────────────┐        │
│  │ agent_chat.py - Main chat handler    │        │
│  │ metrics_service.py - Performance     │        │
│  └──────────────────────────────────────┘        │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│         UnifiedMemorySystem (Core)               │
│                                                  │
│   ┌─────────────────────────────────┐            │
│   │ Core Features:                  │            │
│   │ • 5-tier memory collections     │            │
│   │ • TagCascade retrieval          │            │
│   │ • CE reranking (ONNX)           │            │
│   │ • Sidecar LLM processing        │            │ 
│   │ • Score-based promotion         │            │
│   │ • Two-lane context injection    │            │
│   └─────────────────────────────────┘            │
│                                                  │
│   Collections:                                   │
│   • books: Reference material (permanent)        │
│   • working: Current context (24h retention)     │
│   • history: Past conversations (30d retention)  │
│   • patterns: Proven solutions (permanent)       │
│   • memory_bank: User memories (NEW - 2025-10-01)│
└───────────────────────────────────────────────────┘
```

## Memory System

### 5-Tier Memory Architecture (Updated 2025-10-01)

#### Books Collection
- **Purpose**: Store reference documentation and knowledge bases
- **Retention**: Permanent (never decays)
- **Source**: User-uploaded documents (.txt, .md files)
- **Use Case**: Technical documentation, guides, references
- **Contextual Embedding** (v0.2.6): Chunks are prefixed with `"Book: {title}, Section: {section}."` before embedding for ~49% improved retrieval on ambiguous queries

#### Working Memory
- **Purpose**: Current conversation context
- **Retention**: 24 hours from creation
- **Scope**: Global across all conversations (see Cross-Conversation Memory Search below)
- **Search Behavior**: All collections use consistent 3× search multiplier for fair competition
  - When limit=5: Working fetches 15, Memory_bank fetches 15, Books fetches 15 (equal depth)
  - Ensures memory_bank facts compete fairly with conversation context
- **Promotion**: Valuable items promoted automatically via:
  - Working → History: score ≥0.7 AND uses ≥2 (resets uses=0, success_count=0)
  - Triggers: Every 30 minutes (background task), Every 20 messages (auto-promotion), On conversation switch
  - History → Patterns: score ≥0.9 AND uses ≥3 AND success_count ≥5 (v0.2.9+)
- **Use Case**: Active problem-solving context

#### History Collection
- **Purpose**: Past conversations and interactions
- **Retention**: 30 days (high-value items preserved via `clear_old_history()`)
- **Promotion**: Successful patterns promoted to patterns collection
- **Use Case**: Learning from past interactions

#### Patterns Collection
- **Purpose**: Proven solutions and successful patterns
- **Retention**: Permanent (but can demote)
- **Source**: Promoted from history when consistently successful
- **Demotion (v0.3.0)**: Patterns → History if score < 0.4 (solution no longer reliable)
- **Use Case**: Quick retrieval of known solutions

#### Memory Bank Collection (NEW - 2025-10-01)
- **Purpose**: Persistent context for both user AND LLM (identity, preferences, learned knowledge, shared projects)
- **Retention**: Permanent (never decays)
- **Capacity**: 500 items maximum (MemoryBankService.MAX_ITEMS=500; config.max_memory_bank_items=1000 is unused)
- **Ranking**: Results boosted by `importance × confidence` score
  - High-quality memories (importance=0.9, confidence=0.9 → quality=0.81) rank significantly higher
  - Low-quality memories (importance=0.3, confidence=0.4 → quality=0.12) rank lower
  - Quality score reduces semantic distance by up to 50% for maximum-quality items
  - ~~Content KG Entity Boost~~ (disabled in v0.3.1 — CE handles relevance ranking instead)
- **Management**:
  - LLM has full autonomy to store/update/archive
  - User has override via Settings UI (restore/delete)
  - Auto-archives old versions on updates (versioning without complexity)
- **Always Inject**: Memories with `always_inject: true` metadata are included in EVERY context injection, regardless of query relevance. Used for critical identity facts.
- **Structure**:
  - Category tags: Soft guidelines (identity, preference, project, context, goal). v0.3.1 adds noun tags (separate field for retrieval routing).
  - Status: active | archived
  - Metadata: importance (0-1), confidence (0-1), mentioned_count, always_inject (bool)
- **Use Case**: Persistent identity and knowledge layer that enables continuity, personalization, and agent growth across all sessions
- **Purpose**: Three-layer foundation for evolving from stateless assistant to long-term collaborator:
  1. **User Context** - Who you are, what you want (identity, preferences, goals, projects)
  2. **System Mastery** - How to be effective (tool strategies, search patterns, what works/fails)
  3. **Agent Growth** - Self-improvement & continuity (mistakes learned, relationship dynamics, progress tracking)
- **Scope Guidelines**:
  - ✅ **User identity**: Name, preferences, background, career context, communication style
  - ✅ **Project knowledge**: Current work, tech stack, goals, deadlines, domain expertise
  - ✅ **Discovered patterns**: User preferences learned over time, what works for THIS user
  - ✅ **System navigation**: Effective search strategies, tool usage patterns, workflow optimizations
  - ✅ **Agent self-knowledge**: Mistakes made and lessons learned, strengths/weaknesses discovered
  - ✅ **Relationship dynamics**: Trust patterns, communication effectiveness, how you work together
  - ✅ **Progress tracking**: Goals, checkpoints, what worked, what failed, strategy iterations
  - ❌ **Raw conversation exchanges**: Dialog belongs in working memory (24h) or history (30d)
  - ❌ **Temporary session facts**: Current task details (automatic system handles this)
  - ❌ **Every fact heard**: LLM should be selective - memory_bank is for PERMANENT knowledge
  - **Rule of thumb**: If it helps maintain continuity across sessions OR enables learning/improvement, it belongs here. If it's session-specific, it doesn't.
- **API Endpoints**:
  - `GET /api/memory-bank/list` - List all memories (with filters)
  - `GET /api/memory-bank/archived` - Get archived memories
  - `POST /api/memory-bank/restore/{id}` - User restores archived memory
  - `DELETE /api/memory-bank/delete/{id}` - User permanently deletes memory
  - `GET /api/memory-bank/search?q=...` - Semantic search
  - `GET /api/memory-bank/stats` - Statistics and tag cloud
  <!-- Removed v0.3.1: knowledge-graph endpoints. The KG itself was deleted; route handlers no longer exist. -->
    - Returns `last_used` and `created_at` timestamps when available (v0.2.0)
    - Newly created concepts have `null` timestamps until first scored outcome

**Memory Bank Quality Ranking (v0.2.1):**

Unlike outcome-based collections (working/history/patterns), memory_bank uses **quality-based ranking** to prioritize authoritative facts over semantically similar but lower-quality noise.

**Architecture (3-Stage Quality Enforcement):**

1. **Distance Boost (Pre-Ranking):**
   - Formula: `adjusted_distance = L2_distance × (1.0 - quality × 0.8)`
   - High quality (0.93) → 0.26x multiplier (74% distance reduction)
   - Low quality (0.08) → 0.94x multiplier (6% reduction)
   - Applied in `unified_memory_system.py:1196-1210`

2. **L2→Similarity Conversion:**
   - Formula: `similarity = 1 / (1 + distance)`
   - Maps L2 distance [0, ∞) → similarity (0, 1]
   - Preserves quality boost from step 1
   - Fixed in v0.2.1 (was: `1 - min(d, 1)` which capped all distances >1 to 0 similarity)

3. ~~Cross-Encoder Quality Multiplier~~ (removed in v0.3.1 — TagCascade uses raw CE score as sole ranking signal, no Wilson blend)

**Example:**
```
Query: "Sarah Chen TechCorp engineer"
15:1 noise ratio (3 truth docs vs 47 similar-looking noise)

Truth: "Sarah Chen is 34 years old and works as a software engineer at TechCorp"
- Quality: 0.93 (importance=0.95, confidence=0.98)
- Raw L2 distance: 2.4 → boosted: 0.61 → similarity: 0.62
- After CE blend: 0.62 × 1.93 = 11.4

Noise: "Sarah Chen, 34, engineer at TechCorp Inc (consulting firm)"
- Quality: 0.08 (importance=0.25, confidence=0.30)
- Raw L2 distance: 1.05 → boosted: 0.99 → similarity: 0.50
- After CE blend: 0.50 × 1.08 = 6.4

→ Truth (11.4) beats noise (6.4) despite noise being semantically closer ✅
```

**Deduplication (v0.2.1):**
- Similarity threshold: 0.80 (L2 distance < 0.25)
- Merge strategy: keeps higher quality version's metadata
- Fixed in v0.2.1 (was: threshold 0.95 with broken similarity formula)

**Benchmark Results (Semantic Confusion Test):**
- 15:1 noise ratio (47 confusing facts vs 3 ground truth)
- Quality gap: 12x (0.93 vs 0.08)
- Accuracy: 33% (5/15 truth docs found across 5 queries)
- High-quality docs ranked #1 in 4/5 queries ✅
- BRUTAL query ("the user Sarah") fails as expected (no semantic match to truth)

**Implementation:**
- Distance boost: `unified_memory_system.py:1196-1210`
- L2→Similarity: `unified_memory_system.py:1238-1245`
- CE Quality multiplier: `unified_memory_system.py:653-671`
- Dedup similarity: `unified_memory_system.py:786-791`

<!-- KG Visualization Features (v0.2.0) block removed — the green/purple node graph view, concept modal, routing-breakdown UI, and /api/memory/knowledge-graph endpoint were all retired in v0.3.1. The Memory Panel's tag-cloud view replaced them. -->

### Metadata Schema (v0.2.0 - Searchable Fields)

All memory chunks have metadata stored in ChromaDB that can be filtered during search using the `metadata` parameter in `search_memory` tool or internal API.

#### Books Collection Metadata
Every book chunk contains:
- `title` (string) - Book title from upload
- `author` (string) - Author name (can be "Unknown")
- `book_id` (string) - Unique book identifier
- `chunk_index` (int) - Sequential chunk number
- `type` = "book_chunk" (constant)
- `source_context` (string) - Section/chapter name from document structure
- `doc_position` (float) - Position in document (0.0 to 1.0)
- `has_code` (bool) - True if chunk contains code blocks/snippets
- `token_count` (int) - Number of tokens in chunk
- `upload_timestamp` (ISO datetime) - When book was uploaded
- `content`/`text` (string) - Actual chunk text

**Contextual Embedding (v0.2.6):** Text is embedded with prefix `"Book: {title}, Section: {source_context}. {text}"` for improved semantic matching. This helps ambiguous queries (e.g., "DRY principle") match correct book sections even when the query terms aren't literally in the chunk text.

**Example Queries:**
```python
# Find architecture documentation
{"title": "architecture"}

# Find code examples in books
{"has_code": True}

# Find specific author's work
{"author": "Martin Fowler"}

# Combined: code examples from specific book
{"title": "Clean Code", "has_code": True}
```

#### Working/History/Patterns/Memory_Bank Metadata
Learning and conversation chunks contain:
- `role` (string) - "exchange", "learning", etc.
- `conversation_id` (string) - Session/conversation ID
- `source` (string) - Source of memory (e.g., "mcp_claude", "internal")
- `type` (string) - Type of entry (e.g., "key_takeaway", "exchange")
- `timestamp` (ISO datetime) - When created
- `query` (string) - Original search query (for learnings)
- `collection` (string) - Collection name
- `uses` (int) - Number of times retrieved
- `last_outcome` (string) - "worked" | "failed" | "partial" | "unknown"
- `problem_signature` (string) - Query that generated this learning
- `persist_session` (bool) - Survives session clear
- `score` (float) - Memory quality score (0.0 to 1.0)
- `content`/`text` (string) - Memory content

**Example Queries:**
```python
# Find successful MCP learnings
{"source": "mcp_claude", "last_outcome": "worked"}

# Find recent failures
{"last_outcome": "failed"}

# Find specific conversation
{"conversation_id": "session_12345"}

# Find all learnings (vs exchanges)
{"type": "key_takeaway"}
```

#### ChromaDB Where Filter Syntax
Roampal uses ChromaDB's `where` parameter for metadata filtering:

```python
# Exact match
{"title": "architecture"}

# Multiple conditions (AND logic)
{"title": "architecture", "has_code": True}

# Comparison operators
{"uses": {"$gt": 5}}        # Greater than
{"uses": {"$gte": 5}}       # Greater than or equal
{"uses": {"$lt": 10}}       # Less than
{"uses": {"$lte": 10}}      # Less than or equal
{"uses": {"$ne": 0}}        # Not equal

# In/not in
{"collection": {"$in": ["working", "history"]}}
{"collection": {"$nin": ["books"]}}

# Boolean values
{"has_code": True}
{"persist_session": False}

# Timestamp filtering (ISO datetime strings)
{"timestamp": {"$gte": "2025-01-01T00:00:00"}}  # After Jan 1, 2025
{"timestamp": {"$lt": "2025-12-31T23:59:59"}}   # Before end of 2025
{"upload_timestamp": {"$gte": "2025-11-01T00:00:00"}}  # Books uploaded in Nov 2025
```

**Technical Implementation:**
- `unified_memory_system.py:414` - `metadata_filters` parameter added to `search()` method
- `main.py:710-714` - MCP tool definition includes `metadata` parameter
- `tool_definitions.py:46-50` - Internal LLM tool includes `metadata` parameter
- `agent_chat.py:929` - Internal search supports `metadata_filters` parameter
- `agent_chat.py:2070` - Tool execution extracts and passes metadata filters
- `chromadb_adapter.py:196` - ChromaDB `where` parameter passed to query

### Ranking & Retrieval Algorithm (v2.0)

#### Dynamic Weighted Ranking

The memory system uses adaptive weighting that adjusts based on memory quality and maturity. This ensures high-value memories rank well even with imperfect query formulation.

**Formula:**
```
combined_score = (embedding_weight × embedding_similarity) + (learned_weight × learned_score)
```

**Weight Assignment Logic (v0.2.5):**

| Memory Type | Uses | Score | Embedding Weight | Learned Weight |
|-------------|------|-------|------------------|----------------|
| Proven high-value | ≥5 | ≥0.8 | 20% | **80%** |
| Established | ≥3 | ≥0.7 | 25% | **75%** |
| Emerging (positive) | ≥2 | ≥0.5 | 35% | **65%** |
| Failing pattern | ≥2 | <0.5 | **70%** | 30% |
| Memory_bank (high quality) | any | any¹ | 45% | **55%** |
| Memory_bank (standard) | any | any¹ | 60% | **40%** |
| New/Unknown | <2 | any | 70% | **30%** |

¹ Memory_bank quality determined by importance × confidence ≥ 0.8

> **v0.3.1 change:** Memory_bank special-case rows will be removed. All collections use the same 5-tier system (PROVEN/ESTABLISHED/EMERGING/FAILING/NEW). Memory_bank facts still rank high when semantically relevant but won't dominate irrelevant queries. See `dev/docs/releases/v0.3.1/RELEASE_NOTES.md`.

**Design Rationale:**
- **Adaptive Trust**: System trusts learned scores more as memories prove themselves through usage
- **Query Robustness**: High-value memories rank well even with mediocre query formulation
- **Graceful Degradation**: New memories still rely primarily on semantic matching
- **Memory_bank Boost**: Explicitly stored facts get higher learned weight based on importance × confidence quality score

**Example Impact:**
```
Memory A: Proven Python (uses=10, score=0.9), good match (similarity=1.0)
- Weights: 20% embedding, 80% learned
- Combined: 0.2×1.0 + 0.8×0.84 = 0.872

Memory B: Proven JavaScript (uses=20, score=0.95), poor match (similarity=0.3)
- Weights: 20% embedding, 80% learned
- Combined: 0.2×0.3 + 0.8×0.89 = 0.772

Python memory wins (0.872 vs 0.772) - semantic relevance preserved.
With 90/10 ultra-aggressive, JS would win: 0.1×0.3 + 0.9×0.89 = 0.831
```

**Technical Implementation:**
- `unified_memory_system.py:724-742` - Dynamic weight assignment and score calculation
- Weights stored in result metadata for transparency and debugging
- Falls back to 70/30 static weights for compatibility with old code paths

#### Deduplication Strategy

**Memory_bank and Patterns collections use automatic deduplication:**

1. **Similarity Check**: When storing, searches for existing memories with ≥80% embedding similarity
2. **Quality Comparison**: If duplicate found, compares importance × confidence scores
3. **Smart Merge Strategy**:
   - If new memory has higher quality → archives old version, stores new one
   - If existing has higher quality → increments `mentioned_count`, returns existing ID
   - Result: Single authoritative version instead of duplicates
4. **Configurable**: Can be disabled via `deduplicate=False` parameter

**Benefits:**
- Prevents storage pollution from duplicate facts
- Maintains highest-quality version automatically
- Tracks mention frequency via `mentioned_count` metadata
- Reduces search noise and improves relevance

**Not Applied To:**
- Working/History collections: These need temporal context preserved
- Disable via parameter: `store(text, collection, deduplicate=False)`

**Technical Implementation:**
- `unified_memory_system.py:store()` - Deduplication logic before embedding generation
- `SIMILARITY_THRESHOLD = 0.80` - 80% similarity = likely duplicate
- `chromadb_adapter.py:update_metadata()` - Updates metadata without re-embedding

#### Search Depth Consistency

All collections use equal search depth via **`SEARCH_MULTIPLIER = 3`** (hardcoded as `limit * 3`):
- Memory_bank: fetches 15 results for limit=5
- Working: fetches 15 results for limit=5
- Books: fetches 15 results for limit=5
- History/Patterns: fetches 15 results for limit=5

This ensures fair competition when multiple collections are searched together. The 3× multiplier provides deeper candidate pool for better ranking, then final top-k selection occurs after cross-collection merging and re-ranking.

**Implementation Note**: Currently hardcoded as `limit * 3` in 4 locations (unified_memory_system.py:1359, 1398, 1413, 1421). Should be refactored to class constant for maintainability.

**Why This Matters:**
- Previous system: Memory_bank got limit×1, Working got limit×3
- Result: Working memory systematically crowded out memory_bank facts
- Fix: All collections get equal search depth for fair competition

### No Truncation Policy (v0.2.8)

**Decision:** Return full memory content everywhere. No character limits.

**What Changed:**
- Removed `_smart_truncate()` function and all calls to it
- Removed 300-character limit from MCP `search_memory` tool
- Removed content truncation from `get_context_insights` facts/patterns
- Removed truncation from `get_facts_for_entities()`
- Removed truncation from `analyze_conversation_context()`

**Why:**
- Cold-start pulls ~5 memories = maybe 2-3k tokens max
- Modern LLMs handle 100k+ context easily
- Truncation was premature optimization that made context worse
- "Lost in the middle" research was about massive RAG with hundreds of chunks, not 5-10 memories

**Result:**
- `search_memory` (MCP): Full content
- `get_context_insights`: Full content
- Cold-start injection: Full content (limit by count, not chars)

**Files Modified:**
| File | Change |
|------|--------|
| `main.py:1113` | Removed `[:300]` from MCP search_memory |
| `main.py:1383-1395` | Removed truncation from get_context_insights |
| `unified_memory_system.py:2055` | Removed `_smart_truncate()` from cold-start |
| `unified_memory_system.py:2806` | Removed `[:150]` from `get_facts_for_entities()` |
| `unified_memory_system.py:3107,3127,3182` | Removed 3 truncation locations |
| `agent_chat.py:305,483,641,842,2288` | Removed 5 truncation locations |
| `context_service.py:142,174,275` | Removed 3 truncation locations |

### Enhanced Retrieval Pipeline (v0.2.2 - Nov 2025)

**State-of-the-art retrieval combining 4 proven techniques from 2024-2025 research:**

#### 1. Contextual Retrieval (Anthropic, Sep 2024)

**Problem:** Memory chunks lack context → poor embedding quality
**Solution:** Prepend LLM-generated context to each chunk before embedding

**Example:**
```
Before: "Gemma is 31"
After: "User memory, High importance: Gemma is 31"
```

**Impact:** 49% reduction in retrieval failures (67% with reranking)
**Cost:** ~$1 per million tokens (with LLM prompt caching)
**Implementation:**
- `unified_memory_system.py:261-345` - `_generate_contextual_prefix()` method
- `unified_memory_system.py:383` - Applied during storage before embedding
- **Graceful Fallback**: Uses original text if LLM unavailable or timeout (5s)

#### 2. Hybrid Search (BM25 + Vector + RRF)

**Problem:** Pure vector search misses exact keyword matches
**Solution:** Combine semantic (embeddings) + lexical (BM25) search

**Pipeline:**
1. **Vector search**: Finds semantically similar memories via embeddings
2. **BM25 search**: Finds exact keyword/phrase matches (lexical)
3. **Reciprocal Rank Fusion (RRF)**: Merges results using formula: `score = Σ(1/(rank+60))`

**Why it works:**
- Vector catches "deputy marshal" when query says "law enforcement officer"
- BM25 catches "Gemma Crane" exact name matches
- RRF combines both without manual weight tuning

**Impact:** 23.3pp improvement (CLEF CheckThat! 2025 winner)
**Implementation:**
- `chromadb_adapter.py:315-407` - `hybrid_query()` - BM25 + Vector fusion
- `chromadb_adapter.py:285-313` - `_build_bm25_index()` - BM25 index construction
- **Optional Dependency**: Requires `rank-bm25` and `nltk` packages
- **Graceful Fallback**: Falls back to pure vector search if BM25 unavailable

#### 3. Cross-Encoder Reranking (BERT)

**Problem:** First-stage retrieval has false positives
**Solution:** Score top-30 results with cross-encoder model for precision

**Method (v0.3.1):**
- Uses `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual, 14+ languages, ONNX)
- Cross-encoder jointly encodes query + document pairs
- Provides finer-grained relevance scores than embeddings
- v0.3.1 TagCascade: Raw CE score as final ranking (no Wilson blend). Wilson computed as metadata only (display, promotion). Benchmark: Wilson hurts retrieval by 4.3pts (p<0.0001).

**Why cross-encoder vs bi-encoder:**
- Bi-encoder (what we use for embedding): Encodes query and doc separately, fast but less accurate
- Cross-encoder: Encodes query+doc together, slow but very accurate
- Solution: Use bi-encoder for first-stage retrieval (fast), cross-encoder for reranking top-40 (accurate)

**Implementation:**
- `search_service.py` - `_load_ce()`, `_ce_predict()`, `_rerank_with_ce()` methods
- Lazy-loaded via ONNX Runtime on first search (~471MB download, cached)
- **No extra deps**: Uses ONNX Runtime (already required). No PyTorch/sentence-transformers.
- **Graceful Fallback**: Uses Wilson-only ranking if ONNX model unavailable

**Slot allocation (v0.3.1):**
- Slots 1-3: top 3 by Wilson+CE score (all collections compete on merit)
- Slot 4: nursery (low-use memory, < 3 uses) for exploration
- No reserved working/history slots

#### Technical Details

**Dependencies:**
```bash
pip install rank-bm25 onnxruntime tokenizers nltk
```

#### 4. Query Preprocessing (v0.2.2)

**Problem:** Acronyms in queries don't match full names in stored facts
**Solution:** Expand acronyms before embedding generation

**Example:**
```
Query: "User uses API?"
Preprocessed: "User uses API? application programming interface"
```

**Features:**
- **Acronym Expansion**: 100+ common acronyms (tech, business, locations, organizations)
- **Whitespace Normalization**: Consistent spacing
- **Bidirectional**: Works for both acronym → full name and full name → acronym

**Covered Acronyms:**
- Technology: API, SDK, UI/UX, DB, SQL, HTML, CSS, ML, AI, LLM, etc.
- Locations: NYC, LA, SF, UK, USA, etc.
- Organizations: NASA, FBI, MIT, UCLA, etc.
- Business: CEO, CTO, HR, ROI, KPI, B2B, B2C, etc.

**Implementation:**
- `unified_memory_system.py:972-1144` - `ACRONYM_DICT` and `_preprocess_query()` method
- Applied before embedding generation in search()
- Also passed to BM25 search for lexical matching

<!-- Removed unsourced "75% → 100% acronym expansion" claim — no corresponding benchmark in release notes. Feature itself (ACRONYM_DICT in routing_service.py) is live. -->

**Search Flow:**
```
1. Query received → Preprocess (acronym expansion, normalization)
2. Generate embedding from preprocessed query
3. For each collection:
   a. Vector search (semantic)
   b. BM25 search (lexical with preprocessed query)
   c. RRF fusion
4. Merge all collections
5. Dynamic ranking (v0.2.0)
6. v0.3.1 TagCascade: tag matching → tag-routed search (overlap cascade) → cosine fill
7. Cross-encoder rerank top-40 (ONNX, raw CE score — no Wilson blend)
8. Two-lane injection: 4 summaries + 4 facts = 8 memories
```

**Graceful Degradation:**
- If BM25 unavailable → Falls back to vector-only search
- If cross-encoder unavailable → Uses Wilson scoring only (cosine + Wilson, no CE)
- If contextual prefix fails → Uses original text

**Performance Characteristics:**
- Contextual prefix: +100ms per store (only during storage)
- BM25 index build: ~500ms for 1000 docs (lazy, cached)
- Cross-encoder rerank: +200ms for top-30 (only if >10 results)
- Overall search latency: Still <100ms p95 (BM25 index cached)

**BM25 Cache Invalidation (v0.2.9):**
- Problem: MCP server caches BM25 index, doesn't see UI changes until restart
- Fix: Compare `collection.count()` on each query ([chromadb_adapter.py:211-220](modules/memory/chromadb_adapter.py#L211-L220))
- If count changed → mark `_bm25_needs_rebuild = True`
- If count same → use cached index (zero overhead)
- Result: MCP searches immediately reflect UI changes without restart

#### Why Not Other 2025 Techniques?

❌ **ColBERT Late Interaction**: 6-10× storage cost, complex indexing
❌ **RAPTOR Hierarchical Clustering**: Already have 5-tier collections
❌ **Query Decomposition**: Too slow for conversational use (10× latency)
❌ **Fine-tuned Retrieval Models**: Training overhead not justified

The 4 techniques implemented are **production-proven** (used by Google, Anthropic, Microsoft, Elastic, Weaviate, Pinecone in 2024-2025) and provide maximum impact with minimal complexity.

### Outcome-Based Scoring (Working/History/Patterns Only)

**What It Is:**
- System automatically adjusts `score` based on user feedback about helpfulness
- Tracks `last_outcome` metadata: "worked", "failed", "partial", "unknown"
- Score evolves over time as memory is used and rated

**Score Adjustments (v0.3.1 — flat deltas, matches core v0.4.5):**
- ✅ `worked`: +0.2 (capped at 1.0), success_delta=1.0
- ❌ `failed`: -0.3 (minimum 0.0), success_delta=0.0
- ⚠️ `partial`: +0.05 (small boost), success_delta=0.5
- ❓ `unknown`: -0.05 (weak negative), success_delta=0.25

v0.3.1: Time-weight removed. A score is a score regardless of memory age (matches core v0.3.6+).

**Uses Counter (Wilson Scoring):**
- `uses` is incremented on ALL outcomes (worked, failed, partial, unknown)
- v0.3.0 fix: `failed` now correctly increments uses (was a bug in v0.2.x)
- This provides accurate denominator for Wilson score confidence intervals

**Success Count Tracking (v0.3.0):**
- New `success_count` metadata field tracks cumulative successes
- Replaces parsing `outcome_history` JSON (which was capped at 10 entries)
- Calculation: `success_count += success_delta` for each outcome
- Example: 5 worked + 2 partial + 3 failed + 2 unknown = (5×1.0 + 2×0.5 + 3×0.0 + 2×0.25) = 6.5 successes / 12 uses
- Wilson formula uses `success_count` for accurate long-term confidence intervals

**Wilson Score Lower Bound:**
- Formula: `p̃ = (p + z²/2n - z√(p(1-p)/n + z²/4n²)) / (1 + z²/n)`
- Where: p = success_count/uses, n = uses, z = 1.96 (95% confidence)
- Examples:
  - 1/1 success (100%) → Wilson ≈ 0.20 (low confidence, new memory)
  - 90/100 success (90%) → Wilson ≈ 0.84 (high confidence, proven)
  - 0/0 untested → Wilson = 0.5 (neutral)

**Technical Implementation:**
- [outcome_service.py](modules/memory/outcome_service.py) - `record_outcome()` method
- [scoring_service.py](modules/memory/scoring_service.py) - Wilson score calculation
- v0.3.1: KG calls removed from outcome recording (tags handle routing)
- Score updates logged: `Score update [working]: 0.50 → 0.70 (outcome=worked, delta=+0.20, uses=1)`

**Example Evolution:**
```
Creation: score=0.5, uses=0, success_count=0 (neutral baseline)
After 1st use (worked): score=0.7 (+0.2), uses=1, success_count=1.0
After 2nd use (worked): score=0.9 (+0.2), uses=2, success_count=2.0
After 3rd use (failed): score=0.6 (-0.3), uses=3, success_count=2.0
After 4th use (unknown): score=0.6 (no change), uses=4, success_count=2.25
```

### Learning Mechanisms

#### Organic recall (v0.2.x feature — neutralized in v0.3.1)

The `analyze_conversation_context()` method (`context_service.py`) and the corresponding `get_context_insights` MCP tool surfaced "past experience / past failures / proactive insights" by reading from KG structures (`knowledge_graph["problem_categories"]`, `knowledge_graph["failure_patterns"]`, `knowledge_graph["context_action_effectiveness"]`). With the KG removed in v0.3.1, those data sources no longer exist. The method signatures remain in place for API compatibility but return empty results — see `context_service.py` for the `# KG removed in v0.3.1 — pattern matching no longer available` comment. The MCP `get_context_insights` tool was retired with the v0.3.1 tool-set refresh.

In current operation, "organic" surface area is covered by TagCascade itself — tag-routed retrieval surfaces the same kind of cross-conversation continuity the organic-recall feature was designed to provide, without the KG indirection.

**Cross-Conversation Memory Search (2025-09-30):**
Working memory now searches **globally across all conversations** rather than being filtered to the current conversation only.

**Key Design Decision:**
- **Previous behavior**: Working memory was scoped to `conversation_id == self.conversation_id`
- **Current behavior**: All working memory is searchable regardless of conversation_id
- **Rationale**: The LLM already has current conversation context for disambiguation, so global search enables true organic recall

**Technical Implementation:**
```python
# unified_memory_system.py:453-492
# Working memory searches globally across all conversations
if coll_name == "working":
    # Get ALL working memory across all conversations
    results = await self.collections[coll_name].query_vectors(
        query_vector=query_embedding,
        top_k=limit * 3  # Get more for better ranking
    )

    # Add recency metadata for ALL results (for display purposes only)
    # Sorted by semantic distance only (no recency bias)
```

**Benefits:**
- **True Continuity**: "You asked about Docker 3 weeks ago in a different conversation..."
- **Pattern Recognition**: Can detect recurring issues across conversation boundaries
- **Failure Prevention**: "This exact approach failed last Tuesday in another chat"
- **Context-Aware**: LLM uses current conversation context to filter mentally, not database filter

**Results Ranking:**
Results are sorted by semantic distance only (pure relevance):
- **Relevance**: Lower distance = more semantically similar
- **Note**: Recency metadata is still calculated and displayed, but does not affect ranking

**Combined Fragment Storage (2025-09-30):**
Conversation exchanges are stored as **combined user+assistant fragments** rather than separate messages.

**Storage Format:**
```python
# agent_chat.py:264-274
exchange_doc_id = await memory.store(
    text=f"User: {message}\nAssistant: {clean_response}",
    collection="working",
    metadata={
        "role": "exchange",
        "query": message,
        "response": clean_response[:500],
        "conversation_id": conversation_id
    }
)
```

**Why Combined vs Separate:**
- ❌ **Separate fragments problem**: Question and answer can be promoted/deleted independently during decay, leading to orphaned context (answer without question, or vice versa)
- ✅ **Combined fragments solution**: Question and answer treated as single unit, promoted/deleted together, preserving full context
- ✅ **Search still works**: Semantic search finds exchanges by either question content or answer content
- ✅ **Metadata preserved**: Both `query` and `response` stored separately in metadata for relationship building

**Promotion Impact:**
- Exchanges with `score >= 0.7` and `uses >= 2` promote from working → history (resets uses=0, success_count=0)
- History → patterns requires `score >= 0.9` AND `uses >= 3` AND `success_count >= 5` (must prove itself fresh)
- No conversation filter skipping during promotion (unified_memory_system.py:1359, 1599)
- Full context maintained through entire decay lifecycle: working → history → patterns

**Fast-Track Promotion (REMOVED in v0.2.3):**
~~Exceptional memories can skip history collection and promote directly from working to patterns.~~

**Removed in v0.2.3:** All memories now must go through history first to prove themselves over time. The fast-track bypass was too aggressive - 3 consecutive successes in one session doesn't prove long-term value. Memories need to "season" in history before reaching patterns.

#### Unified Outcome-Based Memory Scoring (Updated 2026-04-10)

**Simple, clean system: Sidecar scores exchanges → mechanical promotion.**

The sidecar LLM is the sole scorer. All scoring, promotion, and deletion decisions follow fixed, predictable rules based on accumulated outcomes. The main LLM does NOT score memories.

**v0.3.1 Sidecar Scoring Architecture:**

The sidecar fires ONE combined LLM call (matching OpenCode's `scoreExchangeViaLLM()`) that returns:
- **summary**: ~300 char first-person note (replaces raw exchange text in ChromaDB)
- **outcome**: worked/failed/partial/unknown (based on user's followup message)
- **noun_tags**: topic nouns extracted from the exchange
- **facts**: atomic facts stored as `memory_type: "fact"` memories
- **memory_scores**: per-memory scoring using the 7-rule guide

**Sidecar API:** `score_exchange()` in [sidecar_service.py](../ui-implementation/src-tauri/backend/modules/memory/sidecar_service.py)

**The Clean Flow:**

1. **Turn N — User:** "What's an IRA?"
2. **Assistant searches memory** → returns doc_id_X, doc_id_Y from patterns/history
   - System caches: `_search_cache[conversation_id]` = {position_map, content_map}
3. **Assistant responds:** "An IRA is a retirement account..."
   - Stores exchange in memory: `"User: What's an IRA?\nAssistant: An IRA is..."`
   - Caches exchange data: `_sidecar_pending[conversation_id]` = {doc_id, user_msg, assistant_msg}

4. **Turn N+1 — User follows up:** "that didn't help"
   - **BEFORE** generating response, system fires sidecar (background task):
     - Pops `_sidecar_pending` (previous exchange) and `_search_cache` (previous cached memories)
     - Calls `score_exchange(user_msg, assistant_msg, followup="that didn't help", memories=[...])`
     - Sidecar returns: summary, outcome="failed", facts, tags, memory_scores={doc_id_X: "failed", doc_id_Y: "unknown"}
     - Updates exchange text with summary, stores facts, updates tags
     - Scores each cached memory individually via `record_outcome()`
   - Then generates new response normally

5. **Assistant responds:** "Let me explain better..."
   - Stores NEW exchange, caches it for next turn's sidecar

**Key Principles:**
- The sidecar fires on the NEXT user message (the followup signal), matching core/OpenCode
- Each cached memory gets individually scored using the 7-rule guide
- The exchange itself also gets scored with its outcome
- No fallback to emoji/detect_outcome — sidecar is the sole scorer

**7-Rule Per-Memory Scoring Guide (in sidecar prompt):**
1. Memory NOT about the topic discussed → "unknown"
2. Memory IS about topic AND outcome "worked" → "worked"
3. Memory IS about topic AND outcome "failed" + response echoed memory → "failed"
4. Memory IS about topic AND outcome "failed" + unrelated → "unknown"
5. Memory IS about topic AND outcome "partial" → "partial"
6. Memory contains good advice response IGNORED → "unknown" not "failed" ("failed" means content was WRONG)
7. When in doubt → "unknown"

**v0.3.1.3 Sidecar Queuing and Reliability Architecture:**

The sidecar now includes robust queuing and retry mechanisms to ensure it always runs, even when models are busy or temporarily unavailable.

**Client Locking and Sharing:**
- **Shared clients**: When sidecar uses same provider/model as main LLM, it shares the client instance to avoid GPU memory duplication
- **Client locking**: `execute_with_client_lock()` prevents concurrent model loading for same client
- **Sequential execution**: Sidecar waits politely if main LLM is using the model, then runs after

**Retry Queue with Exponential Backoff:**
- **Persistent queue**: Failed sidecar tasks are queued for retry (`sidecar_queue.py`)
- **Exponential backoff**: 1min → 2min → 4min → give up (max 3 retries)
- **Background processor**: Runs every 30 seconds to process queued tasks
- **Task types**: `score_exchange`, `extract_facts`, `extract_noun_tags`, `summarize_only`

**Implementation:**
- `score_exchange_with_retry()` - Wrapper with locking and retry
- `extract_facts_with_retry()` - Same for fact extraction
- `extract_noun_tags_with_retry()` - Same for tag extraction  
- `summarize_only_with_retry()` - Same for summarization

**Key Benefits:**
1. **Always runs**: Sidecar never silently fails - queued tasks retry automatically
2. **No GPU duplication**: Shared clients prevent double memory usage
3. **No user slowdown**: Sidecar runs in background after main LLM responds
4. **Model consistency**: Same model can be used for both main LLM and sidecar
5. **Reliability**: Exponential backoff handles temporary unavailability

**Configuration:**
- Same model allowed for main/sidecar (no blocking)
- Automatic client sharing when same provider/model
- Background retry processor started on app startup

**Scoring Rules (v0.3.1 — flat deltas, matches core v0.4.5):**
- `worked` → +0.2, success_delta=1.0
- `failed` → -0.3, success_delta=0.0, uses+=1
- `partial` → +0.05, success_delta=0.5
- `unknown` → -0.05, success_delta=0.25, uses+=1 (weak negative signal)

**Automatic Promotion/Demotion/Deletion** (threshold-based, v0.3.1):
- score ≥ 0.7 AND uses ≥ 2 → working → history (resets success_count=0, preserves uses)
- score ≥ 0.9 AND uses ≥ 3 AND success_count ≥ 5 → history → patterns (resets success_count=0)
- score < 0.4 → patterns → history (demotion)
- score < 0.2 → deleted (or score < 0.1 for items < 7 days old)

**v0.3.1.4 Sidecar Input & Scoring Limits:**

Optimized for small sidecar models (3b+). Worst case: ~5,700 tokens total.

| Parameter | Limit | Reason |
|-----------|-------|--------|
| `score_exchange` user_msg | 8,000 chars | Full exchange context for accurate scoring |
| `score_exchange` assistant_msg | 8,000 chars | Full response visible to sidecar |
| `score_exchange` followup | 4,000 chars | Current user message as outcome signal |
| `extract_facts` content | 8,000 chars | Full exchange for fact extraction |
| `extract_noun_tags` text | 2,000 chars | Summary/fact text for tag extraction |
| Summary output trim | 2,000 chars | Safety net (prompt instructs ~300 chars) |
| **Scored memories per turn** | **8 (most recent)** | Last 8 from search cache — later searches more likely found the answer. Matches two-lane injection (4+4). Prevents JSON output complexity failures on small models. Unscored memories keep their current Wilson score. |

### MCP (External LLM) memory scoring flow (v0.3.1+)

In v0.3.1 the scoring path was split across two tools. `record_response` stores a takeaway summary only; `score_memories` records per-memory outcomes against a `doc_id → outcome` map. The system fires a scoring hook in a system reminder when retrieved memory IDs are in flight; the external LLM is expected to call `score_memories` in response.

**Per-turn flow:**

1. External LLM calls `search_memory("...")` → returns memory IDs (`history_abc`, `patterns_xyz`, etc.). The IDs flow through the next system-reminder hook.
2. External LLM produces its response and judges, per memory, whether it `worked` / `failed` / was `partial` / `unknown`.
3. External LLM calls `score_memories(memory_scores={"history_abc": "worked", ...}, exchange_summary="...", exchange_outcome="worked", noun_tags=[...], facts=[...])`.
   - Each memory's raw score moves per outcome (worked +0.2, partial +0.05, unknown −0.05, failed −0.3 — see `outcome_service._calculate_score_update`).
   - `exchange_summary` lands in the working tier as a new entry. Optional `facts` land as separate atomic entries.
4. If the LLM wants to capture an ad-hoc takeaway *without* a scoring hook, it calls `record_response(key_takeaway=..., noun_tags=[...])` — the takeaway is stored to working with initial score 0.7 and follows the normal lifecycle from there.

**Difference from the v0.3.0 record_response semantics:** the old fat `record_response` took `outcome` / `memory_scores` / `related` / `key_takeaway` all in one call. That schema was retired because per-memory scoring conflates two separate decisions (what to store vs how to weight existing memories) into one optional-parameter mess.

**vs internal (in-product) flow:**

| Aspect | Internal | MCP |
|---|---|---|
| Outcome detection | Sidecar LLM analyzes the exchange | External LLM explicitly classifies each retrieved memory |
| Storage format | Sidecar produces summary + atomic facts | External LLM produces `exchange_summary` + optional `facts` |
| Trust model | System decides outcome | System trusts the external LLM |
| Cache | Session-scoped | Session-scoped |

**Implementation Details:**
- `search_memory` tool:
  - Returns **full content** for all results (no truncation)
  - Caches retrieved `doc_ids` for the subsequent scoring hook (so `score_memories` can attribute outcomes by ID)
  - Caches the search query for the same hook (`last_search_query_cache[session_id]`)
- `record_response` tool:
  - **Parameters:** `key_takeaway` (semantic summary, optional in v0.3.0) + `outcome` (optional, defaults to "unknown") + `memory_scores` (v0.3.0 NEW: per-memory scoring dict)
  - **Storage:** If key_takeaway provided, stores semantic summary to ChromaDB with **initial score calculated from outcome** (worked=0.7, failed=0.2, partial=0.55, unknown=0.5)
  - **Scores CURRENT learning:** Unlike internal system (scores previous), MCP scores the learning being recorded immediately
  - **Per-memory scoring (v0.3.0):** If `memory_scores` dict provided, scores each memory individually with its own outcome
  - **Scores retrieved memories:** If no memory_scores, scores all cached memories from the last search with the same outcome
  - **Metadata includes cached query:** Retrieves last search query from cache and stores as `metadata["query"]`
  - **Why query caching:** Enables KG routing to learn "query X → collection Y worked" even though MCP stores semantic summaries, not verbatim transcripts
  - **Scoring:** Uses explicit outcome directly - no automatic detection, no internal LLM call
  - **Trust model:** System trusts provided outcome completely (caller responsibility to assess accurately)
- key_takeaway: 1-2 sentence summary of what was learned (semantic, not verbatim)
- Outcome parameter: Enum ["worked", "failed", "partial", "unknown"] - **explicitly provided by external LLM or user**, defaults to "unknown"
- Session file: Stores learning with doc_id linking to ChromaDB
- ChromaDB: Stores semantic learning content with metadata (including cached search query)

**What We DON'T Do:**
- ❌ NO propagation to cited fragments (removed for simplicity)
- ❌ NO scoring of books or memory_bank (safeguarded)
- ❌ NO complex ChromaDB queries to find doc_ids (use session files)

**Collections Using Outcome-Based Scoring:**
- ✅ `working` - Current session memories (temporary, outcome-scored)
- ✅ `history` - Past conversations (outcome-scored, promotable)
- ✅ `patterns` - Proven solutions (outcome-scored, permanent)
- ❌ `books` - Reference material (distance-ranked, never scored)
- ❌ `memory_bank` - User facts/Useful information (uses importance×confidence for ranking, NOT outcome-scored)

<!-- Removed "Action-Effectiveness KG Tracking" block: built knowledge_graph["context_action_effectiveness"] which no longer exists (KG removed v0.3.1). Outcome tracking is now per-memory via score_memories / sidecar — see outcome_service.py. -->

**Implementation:** Outcome detection runs in the streaming endpoint via `chat_service.memory.outcome_detector` — see `agent_chat.py` (search `outcome_detector` for current location; line numbers shift across refactors).

**Benefits:**
- ✅ Simple: ONE code path, no duplication
- ✅ Predictable: Fixed rules, no LLM hallucination affecting scores
- ✅ Correct: Scores the exchange user is actually reacting to
- ✅ Clean: No spaghetti code, no cited fragment propagation
- ✅ Clean: ~200 lines of code removed, no competing systems

#### Memory Search & Scoring (Updated 2025-10-03)

**Search Results & Score Handling (Updated 2025-10-07):**

The system **intelligently determines context size** based on query complexity using `_estimate_context_limit()`:

```python
# agent_chat.py:514-535 - Dynamic context sizing
def _estimate_context_limit(query: str) -> int:
    # Broad queries ("show me all...") → 20 results
    # Specific queries ("my name") → 5 results
    # Medium complexity ("how to...") → 12 results
    # Default → 5 results

# All fetched results shown to LLM (no arbitrary slicing)
for memory in relevant_memories:  # Use ALL intelligently-fetched results
    collection = memory['collection']
    score = memory['score']  # 0.0-1.0 confidence/quality indicator
    content = memory['content']
    # LLM sees: "[collection] (score: 0.8) content..."
```

**Score Calculation (Fixed 2025-10-03):**

ChromaDB returns **L2 distance** (Euclidean distance between embeddings):
- Distance 0 = identical embeddings
- Distance 50-200 = moderately similar
- Distance 200+ = not very similar

**Conversion to Score:**
1. **Preferred**: Use stored metadata score (if available)
2. **Fallback**: Convert distance using `score = 1.0 / (1.0 + distance)`

```python
# memory_visualization_enhanced.py:508-517
stored_score = item.get('metadata', {}).get('score', None)
if stored_score is not None:
    relevance_score = stored_score  # Use stored (0.5, 0.7, 0.9, etc.)
else:
    distance = item.get('distance', 1.0)
    relevance_score = 1.0 / (1.0 + distance)  # Distance-based fallback
```

**Collection-Specific Scoring & Authority:**

| Collection | Has Stored Score? | Ranking Method | Authority Level | Purpose |
|------------|-------------------|----------------|-----------------|---------|
| **memory_bank** | **NO** | Pure semantic distance | **AUTHORITATIVE** | User info LLM stored (name, preferences) - trust over conversation |
| **books** | **NO** | Pure semantic distance | **AUTHORITATIVE** | Reference docs - trust for technical info |
| **patterns** | **YES** (0.9-1.0) | Outcome-based score | **HIGH** | Proven solutions - repeatedly successful |
| **history** | **YES** (0.5-0.9) | Outcome-based score | **MODERATE** | Past exchanges - may or may not have worked |
| **working** | **YES** (starts 0.5) | Outcome-based score | **LOW** | Recent conversation - temporary context |

**Key Principle: Authority, Not Score**

**Scoring Strategy (Fixed 2025-10-03):**
- **memory_bank & books**: NO stored score → Ranked by semantic distance (pure relevance)
- **working, history, patterns**: HAS stored score → Ranked by outcome-based quality

**Why This Works:**
- LLM sees `[USER INFO] name is Logan` ranked by how well query matches content
- LLM sees `[CONVERSATION] I don't know your name` ranked by outcome score + distance
- Authority comes from **collection label** (USER INFO vs CONVERSATION), not numeric score
- memory_bank/books float to top when semantically relevant, regardless of working memory's outcome scores
- LLM instructions explicitly state memory_bank/books are AUTHORITATIVE despite lacking scores

**Score Interpretation (for LLM):**
- **0.9-1.0**: Proven reliable, use with high confidence
- **0.7-0.9**: Solid information, generally trustworthy
- **0.5-0.7**: Possibly relevant, evaluate context
- **0.3-0.5**: Low confidence, verify before using
- **0.0-0.3**: Probably noise, ignore unless desperate

**Key Design Principles:**
- **LLM orchestrates search**: LLM decides when to search and which collections
- **Authority over score**: Collection purpose determines trust (USER FACT > CONVERSATION), not just numeric score
- **Scores are advisory**: Hints about reliability, but LLM evaluates context and authority
- **Intelligent context sizing**: Query complexity determines result count (5-20 results, see `_estimate_context_limit()`)
- **No pre-filtering**: System doesn't hide low-scoring results from LLM
- **Distance is fallback**: Stored scores preferred over distance-based calculation
- **Clear labels with metadata**: Results show collection, quality score, outcome, usage count, recency, tags

**Bug Fixed (2025-10-03):**
- **Old formula**: `score = 1.0 - distance` → Created negative scores (e.g., -112.6)
- **New formula**: `score = 1.0 / (1.0 + distance)` → Always positive (e.g., 0.0087)
- **Best practice**: Use stored metadata scores when available

#### Outcome Detection (Updated 2026-01-20)
**Philosophy: Simple Prompt, Single Job**

**v0.3.0 Simplification**: The outcome detector now has ONE job - determine if user is happy. No memory tracking, no complex criteria. LLMs naturally understand human satisfaction.

**Key Design:**
- **Single question** - "Was the user happy?" (not "evaluate response quality")
- **Minimal prompt** - ~50 words total, works on small models
- **Grade USER reaction** - Explicitly stated: "(not the assistant's quality)"
- **Simple JSON output** - Returns only `{"outcome": "worked|failed|partial|unknown"}`

**v0.3.0 Prompt:**
```
Based on how the user responded, grade this exchange.

{conversation}

Grade the USER'S REACTION (not the assistant's quality):
- worked = user satisfied (thanks, great, perfect, got it)
- failed = user unhappy/correcting (no, wrong, didn't work)
- partial = lukewarm (ok, I guess, sure)
- unknown = no clear signal

Return JSON: {"outcome": "worked|failed|partial|unknown"}
```

**Why Simplified (v0.3.0):**
- **Old prompt issue**: "Judge both user feedback AND response quality" confused small models
- **Result**: Local LLMs evaluated assistant quality instead of user reaction → false "failed" on "ty"
- **Fix**: Focus ONLY on user reaction, let LLM use natural language understanding

**Issues Fixed:**
- **2025-10-04**: Heuristic regex matched "unhelpful" → false positive
- **2025-10-05**: Analyzed outcomes before user feedback → false positives
- **2025-10-06**: Was too lenient - any positive word → "worked"
- **2026-01-20**: Prompt confused small LLMs ("ty" → "failed" because "generic response")
**Book & Memory Bank Safeguards** ([outcome_service.py](modules/memory/outcome_service.py))

   Books and `memory_bank` entries are reference material, not scorable memories — they bypass the outcome-driven score-update path entirely. (Pre-v0.3.1 versions of this safeguard also updated a Routing KG before bailing; the Routing KG was removed in v0.3.1, so the safeguard is now just the bail.)
   ```python
   # SAFEGUARD: Books are reference material, not scorable memories
   if doc_id.startswith("books_"):
       return None

   # SAFEGUARD: Memory bank is user identity/facts, not scorable patterns
   if doc_id.startswith("memory_bank_"):
       logger.info("[KG] Learned routing pattern for memory_bank, but skipping score update")
       return None
   ```

2. **Knowledge Graph Exposure** ([unified_memory_system.py:605-621](modules/memory/unified_memory_system.py#L605-L621))
   - KG hints now included in search results
   - LLM sees learned success rates: "This pattern succeeded 90% of the time (8 uses)"
   - Routing hints: "Similar queries (python async) had 85% success rate"
   - Enables LLM to leverage system's learned knowledge
   - **Success Rate Calculation** (v0.1.6, fully implemented in v0.1.7): `successes / (successes + failures)` - excludes partial outcomes from denominator
     - Implementation: Lines 1049-1110 (fragment stats), 1353-1362 (routing patterns), 1639 (solution patterns)
     - Partial results tracked separately as contextual data (still useful but not counted in rate)
     - Provides more accurate confidence metrics for routing decisions
     - Example: 2 successes, 11 failures, 10 partials = 15% success rate (2/13, not 2/23)

3. **Score Update Logging** ([unified_memory_system.py:883-887](modules/memory/unified_memory_system.py#L883-L887))
   - Transparent logging of all score changes
   - Format: `Score update [working]: 0.50 → 0.70 (outcome=worked, delta=+0.20, time_weight=1.00, uses=1)`
   - Improves debugging and system understanding

#### Background Maintenance Tasks (Updated 2025-10-09)
The system runs automated maintenance tasks to keep memory healthy:

**Background Tasks** (runs every 30 minutes):
1. **Memory Promotion**: Promote valuable working memory (score ≥0.7, uses ≥2) to history
2. **Working Memory Cleanup**: Delete items older than 24 hours
3. **History Cleanup**: Delete items older than 30 days (720 hours)

**Batch Cleanup** (v0.3.1, every 50 outcome scores):
1. Working memory cleanup (24h)
2. History cleanup (30 days)
Matches core v0.4.5 `outcome_service.py` pattern.

**Startup Tasks** (runs on system start, non-blocking):
1. Clean stale working memory from previous session
2. Clean history older than 30 days

<!-- Removed v0.2.0 "Automatic Cleanup on Deletion" block: described Content KG / Action KG / Routing KG cleanup hooks (content_graph.remove_entity_mention, cleanup_action_kg_for_doc_ids, _cleanup_kg_dead_references). All KG services were deleted in v0.3.1 (release notes Section 9); none of these methods exist in current code. -->

**Auto-Promotion** (fires every 20 messages, non-blocking):
- Triggered after 20 messages in working memory
- Fire-and-forget (doesn't block chat responses)
- Moved to background task in 2025-10-01 update

#### Knowledge Graph (removed v0.3.1)

The original heavyweight KG components (Routing KG, Content KG, Action KG — separate `KnowledgeGraphService`, `ContentGraph`, `IntelligentRouter` modules with learned edge scores) were removed in v0.3.1. Per the paper (`roampal-labs/paper.md`), the surviving v0.3.3 retrieval architecture is itself a *lightweight knowledge graph* — paper.md:224: *"tags are nodes, memories are edges, and overlap count provides multi-hop convergence without graph database overhead."* So "KG removed" is shorthand for "heavyweight edge-learning KG removed; tags-as-lightweight-KG kept and became the final architecture."

**What was tested and eliminated** (paper.md §5.2, summarized at lines 139-141):

> "Four components were tested and eliminated through retrieval analysis on existing databases: Wilson retrieval blend, Wilson cascade sort, Wilson-only retrieval, and single-pool tag routing. Wilson scoring was tested at every possible retrieval stage (blend, cascade pre-sort, standalone) and hurt or added nothing in all configurations."

**What was kept** (paper.md L18, L781):

> "Tag-scoped retrieval (entity-name routing before cross-encoder reranking) improves retrieval ranking (p<0.0001) but produces no statistically significant exam accuracy difference (p=0.618) — both architectures converge with 8 retrieval slots."
>
> "Final architecture: tags-first cascade retrieval + CE reranking + outcome-based lifecycle management."

**Preserved headline numbers** (paper.md table at L330-360, 20B model, MiniMax-regraded, 1,540 non-adversarial questions):

| Architecture | Non-adversarial accuracy |
|---|---|
| **TagCascade (CE + Tags) — shipped** | **85.8%** |
| CE-Only (no tag routing) | 84.5% |
| Raw baseline (ingestion only) | 56.5% |

TagCascade vs CE-Only is statistically indistinguishable on exam accuracy (p=0.618), but tag routing measurably improves retrieval ranking (p<0.0001). The heavyweight edge-learning KG was abandoned mid-benchmark — `roampal-labs/benchmark/runner.py:1346` records the stopping condition (*"STOPPED: KG+CE edge scores frozen at 0.5 — graph not learning"*), and the KG+CE benchmark group is commented out. The lightweight tag-based replacement was then designed and validated against the preserved data above.

Earlier-stage experiment numbers (full-KG Roampal ~42-50%, Reranker ~73-80%, single-hop 9% vs 41% vs 61%) exist in Roampal memory entries `patterns_fb29c1ed`, `patterns_ddd00e20`, `patterns_aae527ff` but are **not preserved in the paper or in `roampal-labs/` filesystem artifacts** — those were intermediate explorations during the abandonment decision, not the final published numbers. They should not be cited as authoritative; the paper-preserved 85.8% / 84.5% / 56.5% comparison is what survives end-to-end review.

Legacy KG code archived to `dev/archive/kg/`.

<!-- Removed ~430 lines of KG implementation detail, future-stopword-learning speculation, v0.2.6 directive-insights and model-agnostic prompt design. All replaced or obsolete as of v0.3.1. 2026-05-12 cleanup. -->

#### Score-Based Promotion
- Items with high success rates get promoted
- Failed approaches get demoted or removed
- Continuous refinement of knowledge base

## Core Components

### 1. UnifiedMemorySystem (`modules/memory/unified_memory_system.py`)

The **single source of truth** for all memory operations, implemented as a **facade pattern** that delegates to 12 specialized services.

> **Refactored in v0.2.7** (December 2025): The original 4,746-line monolithic file was refactored into 12 smaller, focused modules:
> - `unified_memory_system.py` - Facade (~500 lines) maintaining the public API
> - `scoring_service.py` - Wilson score calculations and quality metrics
> - ~~`knowledge_graph_service.py`~~ - *Removed in v0.3.1 (tag routing replaces KG)*
> - `routing_service.py` - Collection routing and concept extraction
> - `search_service.py` - Vector search with reranking
> - `promotion_service.py` - Memory lifecycle (working -> patterns/history)
> - `outcome_service.py` - Action outcome tracking and learning
> - `memory_bank_service.py` - User fact storage operations
> - `context_service.py` - Cold-start and conversation context
>
> **Note**: Many line number references in this document refer to the pre-refactoring monolithic file and are now approximate or spread across services.

**Key Features:**
- Vector-based semantic search (ChromaDB)
- Automatic memory lifecycle management
- Concept extraction and relationship mapping
- Problem->Solution pattern tracking

**Architecture:**
```
+---------------------------------------------------------------------+
|                    UnifiedMemorySystem (Facade)                      |
|  - Public API unchanged                                              |
|  - Delegates to specialized services                                 |
+----------------------------------+----------------------------------+
                                   |
    +------------------------------+------------------------------+
    |                              |                              |
    v                              v                              v
+-------------+    +-------------+    +-------------+
|SearchService|    |ScoringService|   | TagService  |
| - search()  |    | - wilson()   |   | - extract() |
| - rerank()  |    | - ce_score() |   | - index()   |
+-------------+    +-------------+    +-------------+
    +------------------------------+------------------------------+
    |                              |                              |
    v                              v                              v
+--------------+   +---------------+   +----------------+
|RoutingService|   |PromotionService|  |ContextService  |
| - route()    |   | - promote()    |  | - cold_start() |
| - concepts() |   | - demote()     |  | - analyze()    |
+--------------+   +---------------+   +----------------+
    +------------------------------+------------------------------+
    |                                                             |
    v                                                             v
+---------------+                     +--------------------+
|OutcomeService |                     |MemoryBankService   |
| - record()    |                     | - store/update()   |
| - get_chain() |                     | - archive()        |
+---------------+                     +--------------------+
```

**Key Methods:**
```python
async def store(text, collection="working", metadata=None)
async def search(query, limit=10, collections=None)
async def analyze_conversation_context(current_message, recent_conversation, conversation_id)
async def record_outcome(doc_id, outcome, context=None)
async def promote_working_memory()

# v0.2.6 method signatures retained for API compatibility but now return empty
# (the KG they queried was removed in v0.3.1):
# - get_doc_effectiveness(doc_id) — removed entirely
# - get_tier_recommendations(concepts) — returns all collections (route_query no longer KG-backed)
# - get_facts_for_entities(entities) — falls back to memory_bank semantic search
```

**New Method Details:**
```python
async def analyze_conversation_context() -> Dict[str, Any]:
    """
    Returns organic insights before memory search:
    {
        "relevant_patterns": [...]     # Past successful solutions
        "past_outcomes": [...]          # Previous failures to avoid
        "topic_continuity": [...]       # Topic shift detection
        "proactive_insights": [...]     # Success rate recommendations
    }
    """
```

### 2. Chat Service (`app/routers/agent_chat.py`)

Handles all chat interactions and orchestrates memory operations.

**Features:**
- Message processing with memory context
- WebSocket streaming for real-time status updates
- Integrated outcome detection in process_message()
- Session management
- Prompt building with quality-aware memory labeling
- Recent conversation context (last 4 exchanges = 8 messages, full content)

**WebSocket Streaming (Updated 2025-12-05):**
- Token-by-token streaming via WebSocket connection (migrated from SSE for chat)
- **Note**: SSE (Server-Sent Events) still used for model download progress tracking where one-way updates are appropriate
- WebSocket events:
  - `type: "stream_start"` - Streaming begins (v0.3.0: marks existing placeholder as streaming, or no-op if none exists)
  - `type: "token"` - Text chunks as generated (updates placeholder or creates assistant message on first token)
  - `type: "thinking_start"` - LLM entered thinking mode (v0.2.5: shows "Thinking..." status)
  - `type: "thinking_end"` - LLM exited thinking mode (v0.2.5: resumes "Streaming..." status)
  - `type: "tool_start"` - Tool execution begins
  - `type: "tool_complete"` - Tool execution finished
  - `type: "stream_complete"` - Streaming done with citations, memory_updated flag, timestamp, surfaced_memories (v0.3.0)
  - `type: "validation_error"` - Model validation failed (removes user message, no assistant message created)

**Surfaced Memories Display (v0.3.0):**
- `stream_complete` event includes `surfaced_memories` field showing which memories were injected into LLM context
- Format: `[{doc_id, content, collection, age}]`
- Frontend displays "context: N memories" expandable section below response
- Different from "searched" - surfaced memories are those actually used in context injection
- Stored on message as `surfacedMemories` for later viewing
- Frontend message creation (v0.3.0): `sendMessage()` creates placeholder, `stream_start` marks it as streaming, tokens update it. Falls back to lazy creation on first token if no placeholder exists.
- 2-minute timeout prevents model hangs (DeepSeek-R1/Qwen)
- Clean, single-purpose status flow (no redundant thinking blocks)
- Validation errors never enter conversation history (clean architecture, no transient flags)

**Thinking Tag Streaming Filter (v0.2.5):**
- Problem: Models like DeepSeek-R1 and Qwen QwQ output `<think>...</think>` tags that briefly flash in UI during streaming
- Solution: Backend filters thinking content during streaming, not just at the end
- Implementation: [agent_chat.py:742-781](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py#L742-L781)
  - Buffer accumulates chunks to handle tags split across boundaries
  - `<think` detection triggers `thinking_start` event
  - Content inside tags buffered but not yielded as tokens
  - `</think>` detection triggers `thinking_end` event, discards buffer
  - Final `extract_thinking()` still runs at stream end for cleanup
- Frontend: useChatStore.ts handles `thinking_start`/`thinking_end` to show "Thinking..." status
- **Title Generation Fix**: Title generation now uses `extract_thinking()` to strip thinking content from LLM-generated titles

**Streaming Architecture: WebSocket vs SSE (Updated 2025-10-15):**

Roampal uses **adaptive streaming architecture** that selects the appropriate technology based on runtime environment:

1. **WebSocket** - Bidirectional, real-time streaming
   - **Primary use case**: Chat conversations with LLM
   - **Endpoint**: `/ws/conversation/{conversation_id}`
   - **Direction**: Bidirectional (server can push updates, client can send messages)
   - **Implementation**: [main.py:578](../main.py#L578)
   - **Migrated from SSE**: October 2025 (for chat only)

2. **Dual-Mode Model Downloads** - Environment-aware streaming (Fixed 2025-10-15)
   - **Development mode**: Uses SSE (`/api/model/pull-stream`)
   - **Production mode (Tauri)**: Uses WebSocket (`/api/model/pull-ws`)
   - **Why dual-mode?**: `response.body?.getReader()` returns undefined in Tauri's webview context
   - **Detection**: Frontend checks `window.__TAURI__` to select appropriate endpoint
   - **Backend**: Both endpoints coexist ([model_switcher.py:377-521](../app/routers/model_switcher.py#L377) for SSE, [model_switcher.py:525-673](../app/routers/model_switcher.py#L525) for WebSocket)
   - **Frontend**: [ConnectedChat.tsx:374](../ui-implementation/src/components/ConnectedChat.tsx#L374) handles adaptive selection
   - **Enhanced Error Handling** (v0.2.5): Frontend now extracts detailed error messages from HTTP response body:
     - Parses JSON response for `detail`, `message`, or `error` fields
     - Falls back to text response if JSON parsing fails
     - Displays errors with visual prefix and 5s timeout
     - Example: "Invalid model name format. Expected format: name:tag" instead of "HTTP error! status: 400"

3. **SSE (Server-Sent Events)** - One-way progress streaming
   - **Use cases**:
     - Model downloads (Ollama pull progress)
   - **Media type**: `text/event-stream`
   - **Direction**: Unidirectional (server → client only)
   - **Limitation**: Doesn't work in Tauri production due to webview restrictions

**Key Architectural Decision**: Rather than forcing one streaming method, we maintain both endpoints and let the client choose based on its runtime environment. This ensures model downloads work seamlessly in both development and production without breaking existing functionality.

**Real-Time Memory Sync (Fixed 2025-10-10):**

Prior to this fix, memories were stored in ChromaDB but the UI would not refresh automatically. Users had to manually click the refresh button to see new memories appear.

**The Problem:**
- Two-layer streaming architecture created duplicate `stream_complete` events
- Layer 1 (generator `stream_message()`) stored memory and yielded stream_complete WITHOUT `memory_updated` flag
- Layer 2 (WebSocket wrapper `_run_generation_task_streaming()`) sent its OWN stream_complete WITH `memory_updated: true`
- Tool-calling code path had early `return` statement that skipped memory storage entirely
- Frontend only processed first stream_complete, ignoring the second one with the flag

**The Fix (agent_chat.py):**
- **Line 1603**: Removed early `return` after tool execution, changed to `break` to continue to memory storage
- **Line 1721-1726**: Added `memory_updated: true` and `timestamp` to stream_complete in Layer 1
- **Lines 3586-3607**: Layer 2 now forwards stream_complete from Layer 1 instead of creating duplicate
- **Removed duplicate session file save** in Layer 2 (was saving twice - once in Layer 1, once in Layer 2)

**Result:**
- Single `stream_complete` event with all required fields (citations, memory_updated, timestamp)
- Both tool and non-tool code paths store memory in ChromaDB before completion
- Frontend receives memory_updated flag and automatically refreshes memory panel
- Tech debt eliminated: no duplicate saves, no duplicate events, clean separation of concerns

**Architecture:**
- Layer 1 (`stream_message()`): Generator that handles ALL business logic (memory storage, session save, title generation, outcome detection)
- Layer 2 (`_run_generation_task_streaming()`): Pure transport layer that forwards events from Layer 1 to WebSocket
- Clear responsibility: Layer 1 = what to do, Layer 2 = how to deliver it

**Tool Indicator & Title Generation Fixes (2025-10-10):**

Two critical fixes implemented to improve UX:

1. **Title Event Forwarding** (agent_chat.py:3586-3588)
   - **Problem**: Layer 1 generated titles but Layer 2 didn't forward `title` events to frontend
   - **Impact**: Conversations stuck showing "New Chat" instead of auto-generated titles
   - **Fix**: Added `title` event handler in WebSocket wrapper to forward events from Layer 1
   - **Result**: Titles now appear automatically after first exchange

1a. **Title Generation Data Path Fix** ([file_memory_adapter.py:14](../modules/memory/file_memory_adapter.py), 2025-10-21)
   - **Problem**: `get_loopsmith_path()` hardcoded `Path(f"data/{resource}")` instead of using `DATA_PATH` from settings
   - **Impact**: Session files saved to bundled backend `data/` folder instead of user AppData directory in production builds
   - **Fix**: Changed to `return Path(DATA_PATH) / resource` to use platform-specific user data path
   - **Result**: All session files now correctly persist to AppData/Roaming/Roampal/data/ across app updates

2. **Tool Executions Persistence** (useChatStore.ts:283-292)
   - **Problem**: `toolExecutions` not loaded from session file metadata on page refresh
   - **Impact**: Tool indicators showed "⋯ running" indefinitely after refresh (stuck pulsing dots)
   - **Fix**: Extract `toolExecutions` from `msg.metadata.toolResults` when loading messages
   - **Result**: Tool indicators correctly show ✓ checkmark with result count after page refresh

### 3. Book Processor (`modules/memory/smart_book_processor.py`)

Processes uploaded documents for the knowledge base.

**Features:**
- Multi-language text chunking (1500 chars with 300 char overlap)
  - Supports Latin, CJK (Chinese/Japanese/Korean), Arabic/Urdu punctuation
  - Unicode 6.1 full-text search via SQLite FTS5
  - Intelligent boundary detection (respects paragraphs, sentences, structure)
- Batch embedding generation via EmbeddingService (10 chunks in parallel)
  - Uses bundled paraphrase-multilingual-mpnet-base-v2 (50+ languages, 768 dimensions)
- Dual storage: SQLite (full-text search) and ChromaDB (semantic search)
  - Content stored in both metadata and documents for retrieval
- **Context expansion** via `get_surrounding_chunks(chunk_id, radius=2)`
  - Retrieves sequential chunks around a relevant result for deeper context
  - Default radius of 2 returns 5 chunks total (2 before + center + 2 after)
  - Maintains reading order from the original document
  - Returns book metadata (title, author, chunk range)
- Real-time progress tracking via WebSocket
- Security validations:
  - 10MB size limit
  - UUID format validation for book IDs
  - Metadata length limits (200 chars title, 1000 chars description)
  - Prompt injection pattern detection (logged warnings)

### 4.1 Format Extractor (`modules/memory/format_extractor/`) - NEW v0.2.3

Converts various document formats to plain text before processing by SmartBookProcessor.

**Supported Formats:**

| Format | Extension | Library | Notes |
|--------|-----------|---------|-------|
| Plain Text | .txt | built-in | Direct read with encoding detection |
| Markdown | .md | built-in | Direct read |
| PDF | .pdf | pypdf | Text extraction, metadata (title/author) |
| Word | .docx | python-docx | Preserves headings, extracts tables |
| Excel | .xlsx, .xls | openpyxl + pandas | Row-based chunking with headers |
| CSV | .csv, .tsv | pandas | Auto-detects delimiter and encoding |
| HTML | .html, .htm | beautifulsoup4 | Strips tags, preserves structure |
| RTF | .rtf | striprtf | Basic text extraction |

**Architecture:**
```
upload → FormatDetector.detect() → appropriate Extractor → ExtractedDocument → SmartBookProcessor
```

**ExtractedDocument Structure:**
- `content: str` - Extracted text content
- `format_type: str` - Original format (pdf, docx, excel, etc.)
- `title: Optional[str]` - Extracted from document metadata
- `author: Optional[str]` - Extracted from document metadata
- `is_tabular: bool` - True for Excel/CSV data
- `extraction_warnings: List[str]` - Any issues (e.g., "images skipped")

**Tabular Data Handling:**
- Excel and CSV files use row-based chunking (50 rows per chunk)
- Column headers prepended to each chunk for context
- Enables semantic search over structured data

**Limitations:**
- Scanned PDFs: No OCR support (text-based PDFs only)
- Password-protected files: Not supported
- Images in documents: Skipped (text extraction only)
- Excel formulas: Values extracted, not formulas

## Tool-Based Memory Search (NEW - 2025-10-01)

### Overview
The LLM can now autonomously search the memory system using the `search_memory` tool, rather than relying on backend pre-search. This provides better control, token efficiency, and search precision.

### Architecture Change

**Before (Backend-Controlled)**:
```
User Message → Backend searches all collections → Top 5 results → Injected into prompt → LLM responds
```

**After (LLM-Controlled - Multi-Turn Tool Calling)**:
```
1. User Message → LLM receives search tool
2. LLM generates tool_call → Backend executes search
3. Tool results fed back to LLM in new message
4. LLM generates final response using search results
5. Response streamed to UI with citations
```

**Implementation Details**:
- **First Stream**: LLM analyzes user query, decides to use `search_memory` tool, stream ends with tool_call event
- **Tool Execution**: Backend executes memory search with LLM-provided parameters
- **Second Stream**: Conversation history + tool results sent back to LLM, final response generated and streamed
- **No Pre-Search**: Backend does NOT search memory before LLM request (Phase 3)

### search_memory Tool Definition (Internal System - v0.2.0)

**Note**: This is the internal system's tool definition. For the external-LLM (MCP) tool surface, see the [MCP tool inventory](#mcp-tool-inventory-current--v031) section.

```python
{
    "type": "function",
    "function": {
        "name": "search_memory",
        "description": "Search the 5-tier memory system with semantic search and optional metadata filtering",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic search query describing what to find"
                },
                "collections": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["books", "working", "history", "patterns", "memory_bank", "all"]},
                    "description": "Which memory collections to search. Use 'all' for comprehensive search.",
                    "default": ["all"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return (1-20)",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata filters for exact field matching. Examples: {\"title\": \"architecture\"} for books by title, {\"author\": \"Smith\"} for books by author, {\"has_code\": true} for code chunks, {\"source\": \"mcp_claude\"} for MCP learnings, {\"last_outcome\": \"worked\"} for successful learnings",
                    "additionalProperties": true
                }
            },
            "required": ["query"]
        }
    }
}
```

### Benefits
- **Selective Search**: LLM can search specific collections (e.g., "check memory_bank for user preferences")
- **Metadata Filtering** (v0.2.0): LLM can filter by exact metadata fields (title, author, outcome, source, etc.)
- **Iterative Refinement**: Can refine searches based on initial results
- **Token Efficiency**: Only retrieves relevant memories (30-50% reduction in context size)
- **Better Control**: LLM decides when memory search is needed vs when current context is sufficient
- **Strategic Collection Use**: Can prioritize patterns for proven solutions, books for reference, memory_bank for identity

### Usage Examples

```python
# Search user preferences
search_memory(query="user's preferred programming languages", collections=["memory_bank"], limit=3)

# Find proven solutions
search_memory(query="database optimization techniques", collections=["patterns", "history"], limit=8)

# Research authoritative sources
search_memory(query="React documentation hooks", collections=["books"], limit=10)

# Comprehensive search
search_memory(query="authentication best practices", collections=["all"], limit=5)

# Metadata filtering examples (v0.2.0)
# Find specific book by title
search_memory(query="chromadb vector database", collections=["books"], metadata={"title": "architecture"}, limit=10)

# Find code examples from books
search_memory(query="python async examples", collections=["books"], metadata={"has_code": True}, limit=10)

# Find successful MCP learnings
search_memory(query="user preferences", collections=["working", "history"], metadata={"source": "mcp_claude", "last_outcome": "worked"}, limit=5)

# Find specific author's work
search_memory(query="design patterns", collections=["books"], metadata={"author": "Gang of Four"}, limit=5)

# Find recent memories (last 7 days)
search_memory(query="recent discussions", collections=["working", "history"], metadata={"timestamp": {"$gte": "2025-11-01T00:00:00"}}, limit=10)

# Find books uploaded this month
search_memory(query="new documentation", collections=["books"], metadata={"upload_timestamp": {"$gte": "2025-11-01T00:00:00"}}, limit=10)
```

### Result Format (Enhanced with Metadata - 2025-10-06)

LLM receives enriched context for each memory result:

```
1. [✓ PROVEN pattern]
   Relevance: 0.92
   Age: 2 hours ago
   Used: 5x
   Outcome: worked
   Content: When user asks about X, do Y...

2. [📚 Reference material]
   Relevance: 0.85
   Content: React hooks documentation excerpt...
```

**Metadata fields** ([agent_chat.py:318-324](../app/routers/agent_chat.py)):
- **Age**: Human-readable recency ("5 minutes ago", "2 hours ago")
- **Used**: Usage count (how many times this memory was retrieved)
- **Outcome**: Last outcome (worked/failed/partial) - helps LLM assess reliability

### Implementation Status
- ✅ **Tool Definition**: Added to `utils/tool_definitions.py`
- ✅ **System Prompt**: Updated in `app/routers/agent_chat.py` (lines 546-577)
- ✅ **OllamaClient**: Added `generate_response_with_tools()` method
- ✅ **Tool Execution**: Handler implemented in `agent_chat.py` (lines 224-296)
- ✅ **Result Formatting**: Displays collection labels, scores, content, and **metadata** (age, usage, outcome)

### Migration Strategy
- **Phase 1 (COMPLETE)**: Tool definition and basic execution implemented
- **Phase 2 (COMPLETE)**: Removed backend pre-search, tool-only mode active
- **Phase 3 (COMPLETE - 2025-10-06)**: Multi-turn tool calling with result feedback loop implemented

### Prompt Philosophy: Transparency Over Instruction (2025-11-26)

**Core Principle**: Prompts explain **what the system DOES automatically** instead of instructing the LLM what to do manually.

**Problem Identified** ("Prompt-Reality Gap"):
- Previous prompts described a manual decision-making system
- Roampal actually implements highly automated intelligence (cold-start auto-injection + sidecar scoring after every turn)
- This created duplicate work (LLM searched when context was already injected) and ignored automation

**Solution** (Transparency-Focused Rewrite):
- **Cold Start**: "System AUTO-INJECTS user context on message 1" (not "you should search memory_bank")
- **Organic Recall**: "System analyzes and injects guidance before every response" (not "decide when to search")
- **Action Stats**: "45% = effectiveness in context, NOT quality" (explains what numbers mean)
- **Outcome Detection**: "System detects automatically" (not "you detect outcomes")
- **Tool Motivation**: Added "Why Search is Your Superpower" and "Why Storing Matters" sections

**Files Updated**:
- main.py:774-922 - MCP tool descriptions (53% token reduction)
- agent_chat.py:1245-1450 - Production system prompt (28% token reduction, complete rewrite)
- tool_definitions.py:23-26 - Internal tool definitions (cold start section updated)

**Impact**:
- ~1,150 total tokens saved (28% production, 53% MCP)
- Eliminates duplicate cold start searches
- Better use of pre-provided organic guidance
- Clearer interpretation of memory-outcome stats
- Motivated tool usage (explains value proposition upfront)
- No breaking changes - all functionality preserved


### System Prompt Instructions (Updated 2025-10-06)

The LLM receives prominent tool usage instructions at the TOP of Section 2 ([agent_chat.py:633-643](../app/routers/agent_chat.py)):

```
[YOUR PRIMARY TOOL - USE IT FIRST]
search_memory(query, collections, limit) - Search your memory collections

BEFORE answering about past info, user preferences, or documents:
1. Call search_memory first
2. Use results to inform your answer

Quick reference:
• "what books?" → search_memory("books", ["books"], 20)
• "what do you know about me?" → search_memory("user", ["memory_bank"], 10)
• "how did we solve X?" → search_memory("X", ["patterns", "history"], 5)
```

**Evolution of Tool Instructions:**
- **2025-10-06 (Initial)**: Added basic tool instruction, but buried after collection descriptions → Models missed it
- **2025-10-06 (Improved)**: Moved to TOP with imperative language and concrete examples → Much higher reliability

## Prompt Engineering & Security

The prompting system builds structured, secure prompts with personality, memory context, and user input.

**Implementation**: [app/routers/agent_chat.py](../app/routers/agent_chat.py) - `_build_prompt()` method (lines 606-712)
**Unified Usage**: Both regular chat and streaming endpoints use the same `_build_prompt()` method (refactored 2025-10-05)
**Tool Definitions**: [utils/tool_definitions.py](../utils/tool_definitions.py)

### Prompt Structure (Unified via _build_complete_prompt)

**Single Source of Truth**: All prompts are built by `_build_complete_prompt()` method ([agent_chat.py:1231-1420](../app/routers/agent_chat.py))

**Provider Consistency** (Updated 2025-11-26):
- Both Ollama and LM Studio use identical system prompts (~1,100 tokens)
- Previous "condensed prompt" for LM Studio removed
- LM Studio gets additional `[IMPORTANT - Function Calling]` section (~80 tokens) for OpenAI-style tool calling
- Generic placeholder examples prevent small LLMs from assuming concrete facts (e.g., `[name]` instead of "Alex")
- Token-optimized: removed duplicate sections, condensed internal mechanics, ~40% reduction from previous version

**Prompt Components (in order):**

**1. Current Date & Time** ([agent_chat.py:1195-1199](../app/routers/agent_chat.py))
- Real-time date/time updated per request
- Format: `[Current Date & Time]`
- Explicit instruction: "When asked about the date or time, use this information directly - do not search memory or claim lack of access"
- Prevents models from hallucinating lack of access

**2. Tool Usage Instructions** ([agent_chat.py:1031-1048](../app/routers/agent_chat.py))
- Comprehensive `search_memory` tool documentation
- **WHEN TO USE search_memory:**
  - User asks about past conversations or personal information
  - User references previous discussions ("that Docker issue", "my project")
  - User asks about preferences, context, or uploaded documents
  - Query could benefit from learned patterns or proven solutions
  - Ambiguous questions that might have relevant history
- **COLD START BEHAVIOR (v0.2.8 - Simplified Semantic Search):**
  - **What**: Automatic context injection on message 1 of new conversations
  - **Why**: Enables personalized, context-aware first responses
  - **Change in v0.2.8**: Replaced KG-based ranking (`quality × log(mentions)`) with simple semantic search
  - **Query**: Single search covering user context + what works + what to avoid + agent growth:
    ```python
    query = "user name identity preferences goals what works how to help effectively learned mistakes to avoid proven approaches communication style agent mistakes agent needs to learn agent growth areas"
    ```
  - **When**: **ALWAYS** on message 1 (internal) or first tool call (external) - no conditions
  - **Result**: Conversations start with relevant context - who user is, what worked before, what to avoid
  - **Internal LLM**: System message injection before first user message ([agent_chat.py:576-603](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py#L576-L603))
  - **External LLM (MCP)**: Prepended to first tool response ([main.py:163-194](../ui-implementation/src-tauri/backend/main.py#L163-L194))
  - **Shared Logic**: `memory.get_cold_start_context()` ([unified_memory_system.py:1884-1926](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L1884-L1926))
  - **Protection**: Layer 4 injection filtering built-in
  - **Simplicity**: One search call, no KG ranking, no fallback logic
  - **Recent Exchanges (v0.3.1.4)**: After user profile injection, fetches 4 most recent exchange summaries sorted by recency (no semantic query). Injected as `RECENT EXCHANGES (last N):` block. Matches core/OpenCode behavior. Filtered to sidecar-generated summaries (`summarized_at` metadata or <500 chars). Gives the chat LLM continuity from past conversations on first message.
  - **Output Format** (v0.2.8 - Simplified):
    ```
    ═══ KNOWN CONTEXT (auto-loaded) ═══
    - memory 1
    - memory 2
    - memory 3
    ```
  - **Why Simplified**: Removed verbose section labels (`[User Profile]`, `[Proven Patterns]`, etc.) - the header "KNOWN CONTEXT" is sufficient
- **WHEN NOT TO USE search_memory:**
  - General knowledge questions (use training data)
  - Current conversation continuation (context already present)
  - Simple acknowledgments ("thanks", "ok", "got it")
  - Meta questions about the system
- **Quick reference examples:**
  - `"what books?" → search_memory("books", ["books"], 20)`
  - `"what do you know about me?" → search_memory("user", ["memory_bank"], 10)`
  - `"how did we solve X?" → search_memory("X", ["patterns", "history"], 5)`
- **Collections**: memory_bank, patterns, books, history, working
- **Response Format**: Optional `<think>` tags for reasoning
- **Formatting**: Markdown support (bold, italic, code, headings, etc.)
- **Outcome Detection**: System learns from user reactions (worked/failed/partial/unknown)
- **Memory Notebook**: MEMORY_BANK tags for storing user facts/useful information (REQUIRED protocol with enforcement)

**3. Personality (User-Customizable via UI)** ([agent_chat.py:1274-1279](../app/routers/agent_chat.py))
- Loaded from `backend/templates/personality/active.txt`
- Converted from YAML to natural language by `_template_to_prompt()` ([agent_chat.py:1353-1455](../app/routers/agent_chat.py))
- File caching with mtime check for performance
- **UI Customization**: Users can edit via PersonalityCustomizer component
  - Quick Settings: name, tone, verbosity, identity, role, formality, use_analogies, use_examples, use_humor, custom_instructions
  - Advanced Mode: Full YAML editor with syntax highlighting
- **Fallback**: "You are Roampal, a helpful memory-enhanced assistant." if template fails
- Updates without restart (cache invalidation on file change)

**4. Recent Conversation History** ([agent_chat.py:1281-1287](../app/routers/agent_chat.py))
- Last 8 messages (4 exchanges) from current conversation
- Format: `[Recent Conversation]` followed by `USER: {content}` and `ASSISTANT: {content}`
- Long messages truncated to 500 characters
- No pre-search memory results (LLM controls memory via tools)

**5. Current User Message** ([agent_chat.py:1289-1291](../app/routers/agent_chat.py))
- Format: `USER: {message}` followed by `ASSISTANT:`
- LLM generates response starting after `ASSISTANT:` marker
- **Current question** clearly labeled

### Security Features

**Prompt Injection Protection** (Multi-layer defense - v0.2.0)

**Layer 3: Output Validation** ([agent_chat.py:83-131,905-911](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py))
- `ResponseValidator` class detects hijacked LLM responses
- Checks for suspicious patterns:
  - Short responses with malicious keywords ("HACK", "PWNED")
  - Role change admissions ("I am now a pirate")
  - System tag injection attempts (`<system>`)
  - Hijack payloads in final sentence
- Replaces compromised responses with safe fallback
- Logs all injection attempts with `[INJECTION DETECTED]`
- **Result**: Prevents stored injection attacks from reaching users

**Layer 4: Cold-Start Filtering** ([unified_memory_system.py:954-979](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L954-L979))
- Filters memory_bank results before auto-injection (post-v0.3.1: tag-routed / cosine retrieval; pre-v0.3.1: Content KG or vector search)
- Implemented in `_format_cold_start_results()` helper method
- Blocks content containing:
  - "ignore all previous instructions"
  - "ignore instructions"
  - "hacked", "pwned"
  - Other instruction override patterns
- Limits to top 10 facts after filtering (v0.2.8: full content returned, no character truncation)
- Logs filtered attempts with `[COLD-START] All results filtered by Layer 4 injection protection`
- **Result**: Cold-start auto-trigger won't inject malicious data (both internal + external LLMs)

**Legacy Memory Content Sanitization** (Deprecated):
- Old approach replaced `[MEMORY_BANK:` → `[MEMORY_CONTENT:`
- No longer needed with Layer 3+4 protection

**Prompt Length Validation** ([agent_chat.py:2368-2448](../app/routers/agent_chat.py))
- Estimates tokens: `1 token ≈ 4 characters`
- **Dynamic max limit** (Updated 2025-10-10): Uses model-specific context window from `config/model_contexts.py`
  - Allocates 50% of model's context for prompt, 50% for response
  - Examples:
    - Qwen 2.5 7B: 28,000 tokens (50% of 56k context)
    - Llama 3.1 8B: 65,536 tokens (50% of 131k context)
    - GPT-OSS 20B: 64,000 tokens (50% of 128k context)
  - Respects user-configured overrides via UI
  - Fallback: 8,192 tokens (safe default for unknown models)
- **Truncation strategy** if exceeded:
  1. Remove conversation history first
  2. Keep personality + core instructions + memory context
  3. Always keep current question
- Logs truncation events with model-aware limits

### MEMORY_BANK Tag Extraction

**Implementation** ([agent_chat.py:895-1011](../app/routers/agent_chat.py)):
- **Method**: `_extract_and_store_memory_bank_tags()` (consolidated, DRY)
- **Called from**: Both regular chat endpoint (line 353) and streaming endpoint (line 1420)
- **Fixed 2025-10-03**: Previously only regular endpoint extracted tags, streaming endpoint was missing this logic
- **Enhanced 2025-10-03**: Added UPDATE and ARCHIVE operations for full LLM autonomy

**LLM Capabilities - Full CRUD Operations**:

1. **CREATE** - Store new memory:
```
[MEMORY_BANK: tag="preference|identity|goal|context" content="specific information"]
```
Regex: `\[MEMORY_BANK:\s*tag="([^"]+)"\s*content="((?:[^"\\]|\\.)*)"\]`

2. **UPDATE** - Modify existing memory:
```
[MEMORY_BANK_UPDATE: match="old content" content="new updated content"]
```
Regex: `\[MEMORY_BANK_UPDATE:\s*match="((?:[^"\\]|\\.)*)"\s*content="((?:[^"\\]|\\.)*)"\]`
- Uses semantic search to find matching memory
- Auto-archives old version with timestamp
- Overwrites with new content

3. **ARCHIVE** - Soft delete outdated memory:
```
[MEMORY_BANK_ARCHIVE: match="content to archive"]
```
Regex: `\[MEMORY_BANK_ARCHIVE:\s*match="((?:[^"\\]|\\.)*)"\]`
- Uses semantic search to find matching memory
- Sets status to "archived"
- Memory hidden from active searches but preserved

**Features**:
- Handles escaped quotes: `\"` inside content
- Removes ALL tag types from user-facing response (prevents clutter)
- Hidden during streaming via buffer filtering (lines 1366-1412)
- Semantic matching for UPDATE/ARCHIVE (finds best match via embedding similarity)
- Example: `[MEMORY_BANK: tag="preference" content="User prefers \"clean code\" style"]`

**Storage Metadata (CREATE)**:
```python
{
    "tags": json.dumps([tag]),
    "importance": 0.7,
    "confidence": 0.8,
    "status": "active",
    "created_at": timestamp,
    "updated_at": timestamp,
    "mentioned_count": 1,
    "added_by": "ai",
    "conversation_id": conversation_id
}
```

**Backend Methods** ([unified_memory_system.py](../modules/memory/unified_memory_system.py)):
- `store_memory_bank()` - Line 2250 - CREATE operation
- `update_memory_bank()` - Line 2290 - UPDATE with auto-archiving
- `archive_memory_bank()` - Line 2347 - ARCHIVE (soft delete)

### Tool Calling (Native Ollama API)

**Implementation Approach (Updated 2025-10-11)**:
- **Native Ollama Tool API** - Direct integration with Ollama's function calling
- Tools passed via `tools` parameter in payload to Ollama API
- Models that support tools use native function calling format
- Backend executes tool and feeds results back for multi-turn conversation
- **Multi-Tool Chaining** - LLM can call multiple tools sequentially (e.g., search → create → update)

**Active Tools** ([tool_definitions.py:5-37](../utils/tool_definitions.py)):
- `search_memory` - Search across memory collections
  - Parameters: `query` (required), `collections` (array), `limit` (1-20)
  - Collections: books, working, history, patterns, memory_bank, all
- `create_memory` - Create new memory bank entries
- `update_memory` - Update existing memory bank entries
- `archive_memory` - Archive (soft delete) memory bank entries

**Multi-Tool Chaining Architecture (Added 2025-10-11)** ([agent_chat.py:1944-2244](../app/routers/agent_chat.py#L1944-L2244)):

**Unified Tool Handler**:
- Single `_execute_tool_and_continue()` async generator handles all 4 tool types
- Eliminates ~200 lines of duplicated code per tool
- Supports recursive chaining with depth tracking (max 3 levels)
- Returns tuple `(tool_execution_record, tool_event_for_ui)` for persistence

**Chaining Flow**:
1. LLM calls tool (e.g., `search_memory`)
2. Backend executes via `_execute_tool_and_continue(chain_depth=0)`
3. Tool results returned to LLM with tools still enabled
4. LLM can call follow-up tools (e.g., `create_memory`) → `chain_depth=1`
5. Process repeats up to `MAX_CHAIN_DEPTH=5` to prevent infinite loops (increased from 3 in v0.3.0)
6. Final response generated after all tools complete

**Key Components**:
- `chain_depth` tracking (lines 585-587) - Prevents infinite recursion
- `MAX_TOOLS_PER_BATCH=10` (lines 841, 2436) - Prevents runaway tool expansion
- `tool_events` array - Collects tool execution metadata for UI persistence
- **Tool continuation policy** (line 2220): `tools=None` on continuation - prevents recursive tool calls after results are provided

**Safety Limits (Updated v0.3.0)**:
- `MAX_CHAIN_DEPTH=5` - Maximum recursion depth (increased from 3 in v0.3.0 for models like Qwen)
- `MAX_TOOLS_PER_BATCH=10` - Maximum tools per LLM response (prevents runaway expansion within single response)
- Combined protection: Max 5 chains × 10 tools = 50 total tool executions per user message
- Warning logged when truncation occurs: `[TOOL] Truncating X tool calls to 10`

**Tool Execution Flow (Updated 2025-11-19 - Multi-Tool Chaining with Batch Limits)**:
1. User message sent to LLM with tools parameter
2. LLM calls one or more tools via Ollama's native function calling
3. Backend executes each tool via unified handler, tracking chain depth
4. Tool results formatted and sent back to LLM (role: "tool")
   - **v0.2.5**: Books collection results include source metadata: `[1] (books from "Title" by Author): content...`
   - Enables LLM to properly cite sources when referencing book content
5. LLM can call additional tools if under depth limit (chaining)
6. LLM generates final response after all tools complete
7. Tool events and citations sent with `stream_complete` event
8. Frontend displays tool icons and citations after streaming completes

**UI Persistence** ([agent_chat.py:815, 1774-1778](../app/routers/agent_chat.py)):
- `tool_events` array saved to session file metadata: `assistant_entry["metadata"]["toolEvents"]`
- Backend persists tool execution data to JSONL session files
- Frontend reads `metadata.toolExecutions` when loading conversation history
- Tool icons (✓ search_memory, ✓ create_memory) persist across page refresh

**Implementation Note**:
- Native Ollama tool API used for models that support it
- Automatic multi-turn conversation handling for tool results
- Citations displayed in UI after streaming completes
- Tool icons display during live streaming (persistence pending frontend implementation)

**Citation Hallucination Prevention (2025-10-11)** ([agent_chat.py:1140-1145](../app/routers/agent_chat.py#L1140-L1145)):
- System prompt explicitly distinguishes system data (search results) from training data (pre-training knowledge)
- Models forbidden from fabricating sources or claiming system data contains training data
- Required to state "I don't have any information about that in your data" when search returns 0 results
- Prevents models from mixing real citations with training data knowledge (e.g., claiming user has books they don't)

**Cleanup** ([tool_definitions.py:39-44](../utils/tool_definitions.py)):
- Removed 7 unused tools (web_search, vision, speech, code analysis, etc.)
- Prevents LLM confusion about available capabilities
- `FUTURE_TOOLS` list kept for reference

### Supported Models (Updated 2025-12-03)

**Models with Native Tool Calling Support**:

Roampal's memory system requires models that support Ollama's native tool calling API. The following models are verified to work:

**Essential Models** (Under 10GB):
- `gpt-oss:20b` - OpenAI's open source model (Apache 2.0) ✅ Tool Support
- `llama3.2:3b` - Meta's small model ✅ Tool Support (minimum recommended)
- `qwen2.5:3b` - Alibaba's efficient model ✅ Tool Support

**Note on Tiny Models (2025-10-10):**
- ⚠️ Models under 3B parameters (`llama3.2:1b`, `qwen2.5:0.5b`, etc.) are **too small** for RoamPal
- **Symptoms**: Output tool definition JSON as text, garbage responses, failed outcome detection
- **Minimum**: 3B parameters required for reliable tool calling and memory system features
- **Removed from installer**: 1b and smaller models no longer available in download modal (2025-10-10)

**Professional Models** (10-30GB):
- `qwen2.5:7b` - Best-in-class tool calling ✅
- `llama3.1:8b` - Meta's balanced model ✅
- `qwen3-coder:30b` - MoE 30B (3.3B active), 256K context, tool calling (Unsloth fixed) ✅
- `qwen3:32b` - Alibaba flagship, native Hermes tools ✅

**Enterprise Models** (30GB+):
- `gpt-oss:120b` - OpenAI's flagship open model ✅
- `llama3.1:70b` - Meta's large model ✅
- `qwen2.5:32b`, `qwen2.5:72b` - Powerful Qwen variants ✅
- `mixtral:8x7b` - MoE architecture ✅
- `llama4:scout` - MoE 109B (17B active), **10M context window**, native tools ✅ (NEW Dec 2025)
- `llama4:maverick` - MoE 401B (17B active), 128 experts, **1M context**, native tools ✅ (NEW Dec 2025)

**Models to Avoid** (No Tool Support):
- ❌ All DeepSeek models - Broken/unstable tool calling, produces garbage output
- ❌ DeepSeek-R1 - Reasoning model, no tool support
- ❌ TinyLlama - Too small for tools
- ❌ Gemma models - No native tool support (requires fine-tuned versions)
- ❌ OpenChat - No tool support
- ❌ CodeLlama - Good at code but no tool support
- ❌ Dolphin3:8b - Causes 400 Bad Request errors with tool calling (removed 2025-10-09)

**Technical Implementation (v0.3.3+):** Tool-capability is no longer a hardcoded family list. `fetch_ollama_capabilities(model, base_url)` calls `POST /api/show` against Ollama and reads the returned `capabilities` set (`vision`, `tools`, `thinking`, ...). LM Studio falls back to the existing `probe_vision` canary path (1×1 PNG → `/v1/chat/completions`). Results are cached per `(provider, model)` and consumed via `get_cached_tools(model, provider)` and `get_cached_vision(model, provider)`. See v0.3.3 Section 4H.

**Model Installation** ([model_installer.py](../app/routers/model_installer.py)):
- Three tiers: Essential, Professional, Enterprise
- All models verified for commercial use (Apache 2.0, MIT, or equivalent)
- Tool support metadata added to model definitions
- Automatic tier-based installation via UI

### Unified Prompt Building

**Regular Chat** ([agent_chat.py:620-793](../app/routers/agent_chat.py)):
- Uses `_build_prompt()` method
- Full 3-section structure with all features

**Streaming Chat** ([agent_chat.py:1093-1244](../app/routers/agent_chat.py)):
- **Fixed 2025-10-03**: Now includes all 3 sections (previously missing Section 2)
  - Section 1: Personality via `_load_personality_template()` (line 1131)
  - Section 2: Core Memory Instructions including MEMORY_BANK syntax (lines 1141-1197)
  - Section 3: Memory context and conversation history (lines 1199-1235)
- MEMORY_BANK tag extraction now works in streaming (line 1310)
- Consistent prompt structure across both endpoints (no drift)
- Sends `full_prompt` as user message, not separate `system_prompt`

### Prompt Building Best Practices

✅ **Sanitize all user-controlled content** (attachments, memory results)
✅ **Use clear section labels** (`[Current Question]`, `[Context from Memory System]`)
✅ **Validate prompt length** before sending to LLM
✅ **Truncate oldest context first** (conversation history, then memory)
✅ **Cache personality template** with file watching (performance)
✅ **Escape special characters** in MEMORY_BANK tags
✅ **Unified prompt building** across all endpoints (no duplication)

## Data Flow

### Chat Flow (Updated with Tool-Based Search)
```
User Message → Chat Service
    ↓
TagCascade Retrieval (v0.3.1+)
  • Extract entity nouns from message
  • Tag-routed cosine over ChromaDB
  • Cross-encoder rerank → top 4 summaries + 4 facts
    ↓
Build Prompt
  • Retrieved memories (8 total: 4 summaries + 4 facts)
  • Tool definitions (search_memory, etc.)
  • Sliding window: last 4 exchanges
  • User question
    ↓
LLM Generation (Ollama / LM Studio)
  ↓
  ├─ Decides: Use search_memory tool? (NEW)
  │   ↓
  │   Tool Execution (Backend)
  │   • Parse tool call
  │   • Execute search
  │   • Return results
  │   ↓
  │   LLM Continues with Search Results (NEW)
  │   • Can refine query
  │   • Can search other collections
  │   • Can request more results
  │   ↓
  └─ Final Response Generation
    ↓
Store Content in Memory
    ↓
Response with citations (WebSocket push)
    ↓
Outcome Detection
    ↓
Memory Update & Learning
    ↓
Knowledge Graph Update (feeds future organic recall)
```

### Learning Flow
```
Conversation → Outcome Detection
    ↓
Extract Concepts & Patterns
    ↓
Update Knowledge Graph
    ↓
Score Adjustment
    ↓
Memory Promotion/Demotion
```

### Memory Context Presentation (Updated 2025-10-07)

The system presents memory context to the LLM with **rich quality metadata**:

```
• SESSION [outcome:partial | used:3x | 2 hours ago]: User asked about Docker containers
• PATTERN [quality:0.9 | used:12x]: Use nginx for reverse proxy
• NOTE [tag:preference | confidence:0.8 | importance:0.7]: User prefers clean code
• BOOK: React documentation on hooks
```

**Metadata Fields Shown:**
- `quality:X.X` - Outcome-based score (working/history/patterns, only if ≥0.7)
- `outcome:worked/failed/partial` - Last outcome (working memory only)
- `used:Xx` - Reference count (if >1)
- `recency` - Human-readable time (working/history: "2 hours ago", "just now")
- `tag:X` - Primary tag (memory_bank: preference/identity/goal/context)
- `confidence:X.X` - LLM's confidence in fact (memory_bank)
- `importance:X.X` - Importance rating (memory_bank)

**Implementation:** [agent_chat.py:712-787](../app/routers/agent_chat.py)

**No redundant filtering**: The decay system handles memory retention naturally.
If a memory exists, it's worth considering.

## Dynamic Context Window Management (NEW - 2025-10-09)

### Problem: Ollama Context Limitation

**Issue**: By default, Ollama restricts context windows to 2K-8K tokens despite models supporting much larger contexts (128K+ for GPT-OSS, 131K for Llama 3.x, etc.). This severely limits conversation depth and memory utilization.

**Solution**: Roampal implements centralized, per-model context window configuration with user customization.

### Architecture

#### 1. Centralized Configuration (`config/model_contexts.py`)

Single source of truth for all model context window settings:

```python
MODEL_CONTEXTS = {
    "gpt-oss": {"default": 32768, "max": 128000},
    "llama3.1": {"default": 32768, "max": 131072},
    "llama3.2": {"default": 32768, "max": 131072},
    "llama3.3": {"default": 32768, "max": 131072},
    "qwen2.5": {"default": 32768, "max": 32768},
    "mistral": {"default": 16384, "max": 32768},
    # ... more models
}

async def get_context_size_async(
    model_name: str, provider: Optional[str] = None, base_url: Optional[str] = None
) -> int:
    """
    Priority order (v0.3.3+):
    1. Runtime override (request-scoped kwarg)
    2. User settings (data/user_model_contexts.json)
    3. Dynamic provider fetch — Ollama POST /api/show OR LM Studio GET /v1/models/{id}
       (cached: in-memory 5min, disk 24h via write_json_atomic)
    4. MODEL_CONTEXTS prefix table (fallback when provider offline/unmatched)
    5. Safe fallback (8192)
    """
```

`MODEL_CONTEXTS` is now a safety net rather than the primary source. New models pulled via Ollama / loaded in LM Studio get their real context length from the provider on first lookup — no code patches required.

**Key Features**:
- Model prefix matching (e.g., "llama3.2:1b" matches "llama3.2")
- Safe fallback for unknown models
- User override persistence to JSON file
- Eliminated ~50 lines of duplicate code from `ollama_client.py`

#### 2. Backend Integration

**Ollama Client** (`modules/llm/ollama_client.py`):
```python
from config.model_contexts import get_context_size_async

# In generate_response() and stream_response_with_tools()
num_ctx = await get_context_size_async(actual_model, provider=provider, base_url=base_url)
options["num_ctx"] = num_ctx  # Passed to Ollama API
# Note: num_gpu=99 hardcoded override removed in v0.3.2.1 hotfix — Ollama handles GPU offload natively.
```

**Thinking Mode Disabled (v0.2.5):**
```python
payload = {
    "model": actual_model,
    "messages": messages,
    "stream": True,
    "think": False  # Disable thinking mode (qwen3, deepseek, etc.) - faster responses
}
```
- **Why**: Models like qwen3 and deepseek-r1 have extended thinking modes that add 30+ seconds
- **Impact**: Much faster responses without thinking overhead
- **All 3 paths**: `generate_response()`, `generate_response_with_tools()`, `stream_response_with_tools()`

**REST API** (`app/routers/model_contexts.py`):
```python
GET  /api/model/contexts              # Get all model context configurations
GET  /api/model/context/{model_name}  # Get specific model context info
POST /api/model/context/{model_name}  # Set custom context size (512-200000)
DELETE /api/model/context/{model_name} # Reset to default
```

#### 3. Frontend UI

**Model Context Settings** (`ui-implementation/src/components/ModelContextSettings.tsx`):
- Accessible via Settings → "Model Context Settings"
- Shows models from **both Ollama and LM Studio** (fetched from `/api/model/available`)
- Current model highlighted at top (smart selection)
- **Per-model provider detection** (Updated 2025-12-04):
  - Each model has its own `provider` field from the API response
  - Ollama models: sliders enabled, settings apply via `num_ctx`
  - LM Studio models: sliders **disabled** (grayed out), shows "LM Studio manages context internally"
  - Allows adjusting Ollama models even when LM Studio is selected as active provider
- Visual indicators:
  - Blue ring + "Active" badge for current model
  - "Custom" badge for user-overridden values
  - Default/Max values displayed
  - Reset button when customized (Ollama only)

**Service Layer** (`ui-implementation/src/services/modelContextService.ts`):
- Caching (5-minute expiry)
- Fallback handling for unknown models
- Type-safe TypeScript interfaces

### Benefits

1. **Full Context Utilization**: Models use their actual context windows (32K-131K) instead of Ollama's 2K-8K default
2. **User Control**: Per-model customization via UI
3. **Zero Duplication**: Single source of truth eliminates maintenance burden
4. **Safe Defaults**: Conservative defaults with ability to increase as needed
5. **Persistent Settings**: User preferences saved to `data/user_model_contexts.json`
6. **Integrated Prompt Building** (Added 2025-10-10): Prompt truncation logic uses same model contexts
   - `agent_chat.py` allocates 50% of model context for prompts
   - Llama 3.1 can use 65k token prompts vs old 8k hardcoded limit
   - Respects same user overrides configured via UI

### Provider-Specific Behavior (Updated 2025-12-04)

| Provider | Context Control | Notes |
|----------|----------------|-------|
| **Ollama** | ✅ Full | Roampal passes `num_ctx` per request; settings apply automatically |
| **LM Studio** | ❌ None | Context set at model load time in LM Studio; Roampal cannot override |

**LM Studio Limitation**:
- LM Studio's OpenAI-compatible API does **not** accept context length in requests
- Context window is configured in LM Studio's **left sidebar settings panel** → Context Length slider
- **Critical**: User must **unload and reload** the model in LM Studio after changing the slider
- LM Studio UI may show a large context (e.g., 33K) but actually load with 4096 tokens
- Roampal detects context overflow errors from LM Studio and displays helpful instructions

**Error Handling** ([ollama_client.py](../modules/llm/ollama_client.py)):
```python
# Detects context overflow errors from LM Studio
if "context" in error_msg.lower() and ("overflow" in error_msg.lower() or "length" in error_msg.lower()):
    user_msg = "**Context Length Error:** LM Studio loaded this model with only 4096 context..."
```

**Pre-flight Context Check (v0.3.0)** ([agent_chat.py](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py)):
- Estimates token usage BEFORE sending to LLM (prevents silent truncation)
- At 90% context usage: Auto-truncates history from 8 to 2 messages
- Shows inline warning to user: `*⚠️ Context limit reached (95% full). Trimmed history from 8 to 2 messages to prevent model barfing.*`
- At 70% context usage: Logs info for monitoring
- Small models (7B-14B) benefit most since system prompt + context injection + tools = ~2300 tokens baseline

**UI Behavior** ([ModelContextSettings.tsx](../ui-implementation/src/components/ModelContextSettings.tsx)):
- **Per-model controls**: Each model checks its own `provider` field, not the global provider dropdown
- LM Studio models: slider disabled, grayed out, "LM Studio manages context internally" message
- Ollama models: fully interactive regardless of which provider is active
- Footer shows warning only if any LM Studio models exist in the list

### Tech Debt Eliminated

**Before** (Pre-2025-10-09):
- 2 copies of context_limits dictionary in `ollama_client.py` (~50 lines duplicate)
- Hardcoded values scattered across codebase
- No user control
- No persistence
- Reset function saved `0` value (fragile falsy check)
- No file locking (race condition risk)
- Field naming mismatch (`user_override` vs `is_override`)
- Duplicate elif block in model_switcher.py

**After** (2025-10-09 Robustness Updates):
- Single config file with centralized logic
- REST API for management
- User-friendly UI
- Persistent user preferences
- Clean separation of concerns
- **Thread-safe file operations** (`threading.Lock()`)
- **Proper reset function** (`delete_user_override()` removes JSON entries)
- **Standardized field naming** (`is_override` throughout)
- **Input validation** on all endpoints
- **Explicit timeouts** on health checks
- **JSON API** for model listing (fallback to text parsing)

### Implementation Files

**Backend**:
- `config/model_contexts.py` - Core configuration and logic
- `app/routers/model_contexts.py` - REST API endpoints
- `modules/llm/ollama_client.py` - Ollama integration (lines 99-105, 587-591)
- `main.py` - Router registration (line 455)

**Frontend**:
- `ui-implementation/src/components/ModelContextSettings.tsx` - Settings UI
- `ui-implementation/src/components/SettingsModal.tsx` - Integration into settings
- `ui-implementation/src/services/modelContextService.ts` - API service layer

**Data**:
- `data/user_model_contexts.json` - User overrides (auto-created)

## API Endpoints

### Security & Rate Limiting

**Rate Limiting** ([main.py:532-564](../ui-implementation/src-tauri/backend/main.py#L532-L564))
- **Limit**: 100 requests per minute per session
- **Scope**: Session-based (tracked by `X-Session-Id` header or client IP)
- **Exemptions**: Health checks, metrics, WebSocket endpoints
- **Response**: HTTP 429 (Too Many Requests) when exceeded
- **Implementation**: In-memory sliding window using `defaultdict(deque)`
- **Purpose**: Prevent abuse and ensure fair resource allocation

**CORS Configuration** ([main.py:456-467](../ui-implementation/src-tauri/backend/main.py#L456-L467))
- **Origins**: All origins allowed (`*`) for Tauri WebSocket compatibility
- **Methods**: GET, POST, PUT, DELETE, OPTIONS, PATCH
- **Credentials**: Enabled
- **Headers**: All headers allowed

### Core Chat Operations
```
POST /api/agent/stream            # Main streaming chat endpoint (auto-generates title after first exchange)
POST /api/chat/create-conversation # Create new conversation
POST /api/chat/switch-conversation # Switch conversations
POST /api/chat/generate-title     # Manual title generation (fallback only)
GET  /api/chat/stats              # Memory statistics
GET  /api/chat/feature-mode       # Get current feature mode
```

### Memory Management
```
GET  /api/memory/stats            # Memory system statistics
GET  /api/memory/search           # Search memories
POST /api/memory/feedback         # Record user feedback on memory usefulness

# Knowledge graph endpoints removed v0.3.1 (KG itself removed). Tag visualization endpoints under
# /api/memory/tags replace the graph view in the Memory Panel.

# Memory Bank Operations (NEW - For MCP Bridge Integration)
GET  /api/memory-bank/list        # List memories (with pagination)
GET  /api/memory-bank/search      # Semantic search in memory bank
GET  /api/memory-bank/stats       # Memory bank statistics
POST /api/memory-bank/create      # Create new memory (MCP tool endpoint)
PUT  /api/memory-bank/update/{doc_id}  # Update memory (MCP tool endpoint)
POST /api/memory-bank/archive/{doc_id} # Archive memory (MCP tool endpoint)
POST /api/memory-bank/restore/{doc_id} # User restore archived memory
DELETE /api/memory-bank/delete/{doc_id} # User hard delete memory
GET  /api/memory-bank/archived    # List archived memories

# Update Notifications (NEW - v0.2.8)
GET  /api/check-update            # Check for available updates
  Response: {
    available: boolean,
    version?: string,      # e.g., "0.2.9"
    notes?: string,        # Brief changelog
    download_url?: string, # Gumroad product URL
    is_critical?: boolean  # Force update below min_version
  }
```

### Update Notification System (v0.2.8)

**Architecture:**
```
┌─────────────────┐         ┌──────────────────────┐
│  Roampal App    │ ──GET── │ roampal.ai/updates/  │
│  (on startup)   │         │ latest.json          │
└────────┬────────┘         └──────────────────────┘
         │
         ▼ (if newer version)
┌─────────────────┐         ┌──────────────────────┐
│ Show banner:    │ ─click─ │ Open Gumroad page    │
│ "Update avail"  │         │ in default browser   │
└─────────────────┘         └──────────────────────┘
```

**Implementation:**
- Backend: `utils/update_checker.py` - async update check with 5s timeout
- API: `main.py:639-651` - `/api/check-update` endpoint
- Frontend Hook: `hooks/useUpdateChecker.ts` - checks on mount with 3s delay
- Frontend Component: `components/UpdateBanner.tsx` - dismissible notification
- Entry Point: `main.tsx` - UpdateBanner integrated into App

**Update Manifest** (hosted at `roampal.ai/updates/latest.json`):
```json
{
  "version": "0.2.8",
  "notes": "MCP security hardening, performance improvements",
  "pub_date": "2025-12-15T00:00:00Z",
  "download_url": "https://roampal.gumroad.com/l/roampal",
  "min_version": "0.2.0"
}
```

**User Experience:**
- Non-blocking: Check happens 3s after startup
- Dismissible: "Later" button for non-critical updates
- Forced: Critical security updates (`is_critical: true`) cannot be dismissed
- Maintains sales funnel: Opens Gumroad page, not auto-download

### MCP (Model Context Protocol) Server

**Status**: ✅ Production (v0.2.0 - Full Learning Support)

Roampal functions as a native MCP server, enabling external LLMs (Claude Desktop, Cursor, etc.) to access Roampal's memory system with **full learning capabilities** - external LLM outcome judgment, score-based promotion, and cross-client knowledge sharing.

#### MCP architecture (v0.3.1+)

```
┌──────────────────────────────────────────────────────────┐
│  MCP Client (Claude Desktop, Cursor, etc.)               │
│  - search_memory(...) → memory IDs                       │
│  - record_response(key_takeaway, noun_tags)              │
│  - score_memories(memory_scores, exchange_summary, ...)  │
└────────────────────┬─────────────────────────────────────┘
                     │ stdio (JSON-RPC)
┌────────────────────▼─────────────────────────────────────┐
│  Roampal MCP server (main.py --mcp)                      │
│  - Stateless tool dispatch; no internal LLM at all       │
│  - System reminder carries scoring-hook directive        │
│    when retrieved IDs are in flight                      │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│  UnifiedMemorySystem (shared with in-product chat)       │
│  - Same ChromaDB collections                             │
│  - Score deltas: worked +0.2 / partial +0.05 /           │
│    unknown −0.05 / failed −0.3                           │
│  - Promotion + decay run as background tasks             │
└──────────────────────────────────────────────────────────┘
```

#### Key behaviors

1. **External LLM owns outcome judgment.** The MCP server has no internal LLM. Outcomes are whatever the external LLM passes in `score_memories`. The in-product chat uses a different path (sidecar LLM scores after the turn) — different code, same lifecycle once the score lands.

2. **Two-tool scoring API.** `record_response(key_takeaway, noun_tags)` stores a takeaway; `score_memories(memory_scores, exchange_summary, exchange_outcome?, noun_tags, facts?)` records per-memory outcomes and (optionally) a summary + atomic facts. The earlier v0.3.0 fat `record_response` is gone.

3. **Unified Learning**
   - All exchanges stored in same ChromaDB collections
   - Cross-client knowledge sharing (Claude → Cursor → Roampal)
   - Memories visible in Roampal UI with source metadata
   - Promotion works identically for all LLMs

4. **Zero Dependencies**
   - No Ollama or LM Studio required for MCP functionality
   - Bundled `paraphrase-multilingual-mpnet-base-v2` model (1.1GB)
   - 50+ language support

5. **Cross-Platform Support**
   - Auto-detects platform (Windows .exe, macOS .app, Linux binary)
   - Stdio communication (JSON-RPC protocol)

6. **Auto-Discovery** ([mcp.py:109-210](../ui-implementation/src-tauri/backend/app/routers/mcp.py#L109-L210))
   - Scans home directory root and common config directories for MCP client configs
   - **Windows**: `%USERPROFILE%`, `%APPDATA%`, `%LOCALAPPDATA%`, `.config`
   - **macOS**: `~`, `~/Library/Application Support`, `~/.config`
   - **Linux**: `~`, `~/.config`, `~/.local/share`
   - Discovers MCP clients via pattern matching: `*mcp*.json`, `config.json`, `*_config.json`
   - Pure discovery approach - no hardcoded tool names (finds `.cursor`, `.vscode`, `.claude`, etc.)
   - Only shows tools with valid `mcpServers` configuration key
   - Detects connection status by checking if `roampal` server is configured
   - **Manual Path Support**: Users can add custom MCP client paths via "Add Custom MCP Client" button
   - Custom paths are saved to `mcp_custom_paths.json` and persist across restarts
   - Scanner automatically includes both auto-discovered and manually-added paths
   - Security: Manual paths must be within user's home directory
   - **Claude Desktop Fix** ([mcp.py:399-404, 282-288](../ui-implementation/src-tauri/backend/app/routers/mcp.py)):
     - Claude Desktop uses TWO config files: `config.json` (UI settings) and `claude_desktop_config.json` (MCP servers)
     - Connect endpoint auto-redirects: If connecting to `Claude/config.json`, writes to `claude_desktop_config.json` instead
     - Scanner skips `config.json` if `claude_desktop_config.json` exists (prevents wrong file detection)
     - Fix is Claude-specific (checks `parent.name == "Claude"`), doesn't affect other MCP tools

7. **Cold-Start Auto-Trigger**

   **Problem:** LLMs don't consistently search memory_bank on first message, missing user context.

   **Solution:** `get_cold_start_context(limit=5)` auto-injects a user profile on first message of each session. Internal chat injects as a system message before the user message; MCP prepends it to the first tool response with visual separators (`═══ KNOWN CONTEXT (auto-loaded) ═══ ... ═══ Tool Response ═══`). Pre-v0.3.1 versions of this helper read from a Content KG entity index (top-N entities by mention count); the v0.3.1 implementation is **pure semantic/tag-routed retrieval** against `memory_bank` — same goal, simpler implementation, no KG dependency.

8. **MCP Security Hardening** (v0.2.8)

   **Parameter Allowlisting** - Blocks MCP signature cloaking attacks:
   - Research identified attack where malicious MCP servers hide parameters using `InjectedToolArg`
   - Hidden parameters don't appear in schema but are functional at runtime
   - **Fix**: Filter arguments to only declared parameters before sending to server
   - Dropped parameters logged as potential attack indicator
   - Implementation: `manager.py:397-409`

   **Rate Limiting for MCP Tools**:
   - Wire existing `RateLimiter` (50 req/60s) to MCP tool execution
   - Prevents runaway LLM tool loops from overwhelming external servers
   - Implementation: `manager.py:371-375`

   **Audit Logging** - Append-only JSONL log for all MCP tool executions:
   - Logs: timestamp, tool, server, args_keys (not values for PII safety), dropped_params, success, duration_ms
   - Dropped params = attack detection indicator
   - Location: `{DATA_PATH}/mcp_audit.jsonl`
   - Implementation: `manager.py:448-480`

   **Trust Model**: Real security boundary is trusting the MCP server you install (like npm packages, VS Code extensions). Parameter allowlisting closes one specific deception vector.

#### Tool description philosophy

Tool descriptions are written for external LLMs (Claude Desktop, Cursor) and aim for scannable bullet points over explanatory paragraphs, with detailed guidance pushed to `inputSchema` descriptions (which surface as IDE tooltips). Workflow framing: `search_memory(...)` → respond → `score_memories(...)` (when a scoring hook fires) or `record_response(...)` (for ad-hoc takeaways).

#### LLM Prompt Design (v0.2.5)

MCP and Internal prompts use different approaches based on system differences:

**MCP System (External LLMs):**
- LLM must explicitly call tools - no automatic detection
- Prompts include clear triggers ("REQUIRED: Call after EVERY response")
- Scoring mechanics documented (worked=0.7, failed=0.2, etc.)
- Examples provided for memory_bank usage

**Internal System (Ollama/LM Studio):**
- Automatic outcome detection - LLM doesn't need to call anything for scoring
- Prompt explains what happens automatically, not what LLM must do
- Section 8: "Outcome Scoring - Automatic"
- **v0.2.12→v0.3.0 Memory Attribution:** Main LLM adds `<!-- MEM: 1👍 2🤷 3👎 4➖ -->` annotation for causal scoring
  - 👍 = memory helped me answer well (worked)
  - 🤷 = memory somewhat helpful (partial) - v0.3.0 NEW
  - 👎 = memory was wrong/misleading (failed)
  - ➖ = memory not used (unknown)
  - Annotation stripped before showing response to user
  - Parsed by `parse_memory_marks()` and passed to OutcomeDetector

**Implementation:**
- MCP tools: [main.py:807-948](../ui-implementation/src-tauri/backend/main.py#L807-L948)
- Internal prompt: [agent_chat.py:1370-1380](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py#L1370-L1380)
- v0.2.12 memory attribution: [agent_chat.py](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py) - `parse_memory_marks()` function

#### MCP tool inventory (current — v0.3.1+)

The MCP server exposes six tools to external LLMs (Claude Desktop, Cursor, etc.). Authoritative schemas live in `roampal-core/roampal/backend/mcp/` and are returned to the client at tool-discovery time; the summary below describes intent only.

| Tool | Purpose |
|---|---|
| `search_memory` | Search across collections (books/working/history/patterns/memory_bank) with optional `collections`, `limit`, `sort_by`, `days_back`, `metadata`, `id`, `type` filters. Used to recall past context. |
| `add_to_memory_bank` | Pin a permanent fact (identity / preference / goal / project). Dedup-protected (cosine < 0.32 against existing memory_bank entries). |
| `update_memory` | Replace the content of an existing `memory_bank_<8hex>` entry by exact `id`. Preserves `created_at`; optional `tags`, `noun_tags`, `importance`, `confidence` overrides. |
| `delete_memory` | Remove a memory_bank entry by content match. No restore path via MCP. |
| `record_response` | Store a key-takeaway summary to the working tier. Two required params: `key_takeaway`, `noun_tags`. Outcome scoring is no longer a parameter here — it moved to `score_memories`. |
| `score_memories` | Fired in response to a scoring hook in the system reminder. Scores each retrieved memory by `doc_id → worked/failed/partial/unknown`; stores `exchange_summary` + optional atomic `facts` to the working tier. Required: `memory_scores`. Optional: `exchange_summary`, `exchange_outcome`, `noun_tags`, `facts`. |

**Removed in v0.3.1:** the old fat `record_response` schema (with `outcome`, `memory_scores`, `related` parameters merged in) was split into lean `record_response` + dedicated `score_memories`. `get_context_insights` was removed when the KG it queried was deleted.

**Removed earlier:** `list_memory_bank` (use `search_memory(collections=["memory_bank"])`), `query_kg_entities`, `query_kg_relationships`, `get_kg_path` (KG itself removed).

#### User Interface Integration

**Integrations Panel** ([IntegrationsPanel.tsx](../ui-implementation/src/components/IntegrationsPanel.tsx)):
- **Auto-Scan**: Automatically discovers MCP clients on modal open
- **Tool Cards**: Shows each detected tool with connection status (Connected/Available/Not Installed)
- **Connect/Disconnect**: One-click connection management
- **Manual Path Addition**: "Add Custom MCP Client" button for non-standard locations
- **Hide/Unhide Tools** (v0.2.0+):
  - Hide button on each tool card (eye icon)
  - Hidden tools moved to collapsible "Discover More Tools" section
  - "Show in List" button to unhide tools
  - localStorage persistence across app restarts
  - Stored in `hiddenMCPTools` key as JSON array of tool names
- **Toast Notifications**: Real-time feedback for all actions
- **Status Indicators**: Color-coded dots (green=connected, yellow=available, gray=not installed)
- **Info Box**: Explains how MCP works and restart requirements
#### Embedding Model

**Embedding Model** (v0.2.0 - Bundled):
- **Model**: `paraphrase-multilingual-mpnet-base-v2`
- **License**: Apache 2.0 (commercial-safe, bundled with distribution)
- **Dimensions**: 768 (native, no padding)
- **Quality**: Higher accuracy than previous model (all-MiniLM-L6-v2)
- **Languages**: 50+ (see THIRD_PARTY_LICENSES.md for full list)
- **Deployment**: Bundled in `binaries/models/` - no Ollama or internet required
- **Usage**: All users get consistent embeddings (chat + MCP server)
- **Location**: `binaries/models/paraphrase-multilingual-mpnet-base-v2/`
- **Loading**: [embedding_service.py:35-56](../ui-implementation/src-tauri/backend/modules/embedding/embedding_service.py#L35-L56)

**Why Bundled Embedding Model?**
1. **MCP server usage**: Works without Ollama installation
2. **Consistency**: All users get identical embeddings regardless of setup
3. **Offline-first**: No internet connection required for embeddings
4. **Commercial**: Apache 2.0 license allows bundling and redistribution
5. **Quality**: Better than previous all-MiniLM-L6-v2 (384 dim)
- Previous model (all-MiniLM-L6-v2) was trained on MS MARCO dataset (non-commercial license)
- Bundling creates redistribution liability if model has commercial restrictions
- New model trained on commercial-friendly datasets (no MS MARCO)
- Enables offline MCP usage (no internet downloads)

#### Configuration Example (Claude Desktop)

```json
{
  "mcpServers": {
    "roampal": {
      "command": "C:\\Program Files\\Roampal\\Roampal.exe",
      "args": ["--mcp"]
    }
  }
}
```

**Auto-Detection**: See [mcp.py](../ui-implementation/src-tauri/backend/app/routers/mcp.py) for platform-specific paths

#### Implementation Details

**Main Entry Point**: [main.py:640-759](../ui-implementation/src-tauri/backend/main.py#L640)
```python
async def run_mcp_server():
    # Initialize with embedded ChromaDB (no server needed)
    memory = UnifiedMemorySystem(data_dir=str(DATA_PATH), use_server=False)
    await memory.initialize()

    # v0.3.1+: embedding service is ONNX-only — no provider toggle needed.

    # Create MCP server with 3 tools
    server = Server("roampal-memory")
    # ... tool registration ...
```

**Architecture Independence**:
- Chat (Ollama/LM Studio) and MCP (embedding service) are decoupled
- Users can run MCP without any LLM installed
- Embedding service is ONNX-only (`mxbai-embed-large-v1`, 768-d) — no LLM-provider dependency

#### Use Cases

1. **Claude Desktop Integration**
   - Search Roampal's memory during Claude conversations
   - Store important context in memory_bank
   - Access uploaded books/documents

2. **Cursor IDE**
   - Semantic search across project documentation
   - Multi-language code context retrieval

3. **Offline Workflows**
   - No internet required after initial installation
   - All embeddings generated locally with bundled model

#### Performance

- **First Load**: 2-5 seconds (model initialization)
- **Memory Usage**: +1.5GB RAM (model loaded)
- **Inference**: ~5x slower than all-MiniLM-L6-v2 (quality vs speed tradeoff)
- **Caching**: Embeddings cached (200 entry LRU, significant speedup for repeated queries)

### MCP Client (External Tool Servers) - v0.2.5

**Status**: ✅ Production (v0.2.5)

Roampal can act as an MCP *client*, connecting to external MCP tool servers just like Claude Desktop and Cursor do. This enables local LLMs to use external tools like filesystem, GitHub, Blender, databases, and more.

> **Security Note:** Only add MCP servers from sources you trust. MCP servers run with your user permissions and can execute code on your machine.

#### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Roampal UI Chat                                        │
│  User: "Create a cube in Blender"                       │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Ollama / LM Studio (Local LLM)                         │
│  Receives tools: search_memory, blender_create_cube... │
│  Chooses: blender_create_cube                           │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  Roampal Backend (agent_chat.py)                        │
│  Detects external tool, routes to MCP Client Manager    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  MCP Client Manager (modules/mcp_client/manager.py)     │
│  ├─ Manages connections to external MCP servers         │
│  ├─ Tool discovery (tools/list)                         │
│  ├─ Tool prefixing (servername_toolname)               │
│  └─ Request routing based on prefix                     │
└────────────────────┬────────────────────────────────────┘
                     │ stdio (JSON-RPC)
┌────────────────────▼────────────────────────────────────┐
│  External MCP Server (e.g., Blender, filesystem, etc.)  │
│  Executes tool, returns result                          │
└─────────────────────────────────────────────────────────┘
```

#### Key Components

1. **MCPClientManager** ([modules/mcp_client/manager.py](../ui-implementation/src-tauri/backend/modules/mcp_client/manager.py))
   - Manages stdio connections to external MCP servers
   - Discovers tools via `tools/list` JSON-RPC call
   - Routes tool calls based on server name prefix
   - Handles connection lifecycle (connect, disconnect, reconnect)
   - **Windows Compatibility** (v0.2.5): Uses `shutil.which()` to resolve full command paths for `npx`, `npm`, `node`, `uvx` on Windows. Always uses `shell=False` to prevent command injection vulnerabilities.

2. **MCPServerConfig** ([modules/mcp_client/config.py](../ui-implementation/src-tauri/backend/modules/mcp_client/config.py))
   - Configuration management for MCP servers
   - Persists to `mcp_servers.json` in data directory
   - **Popular server presets**: Only `filesystem` and `sqlite` (work out of box without API keys)
   - Users can manually add servers that need configuration (github, slack, brave-search, etc.)

3. **API Router** ([app/routers/mcp_servers.py](../ui-implementation/src-tauri/backend/app/routers/mcp_servers.py))
   - REST endpoints for server management
   - CRUD operations for server configuration
   - Connection testing and status

4. **Settings UI** ([MCPServersPanel.tsx](../ui-implementation/src/components/MCPServersPanel.tsx))
   - Add/remove MCP servers via custom server form
   - Connection status and tool discovery

#### Tool Integration

External tools are injected into the LLM context alongside internal Roampal tools:

```python
# agent_chat.py integration
from modules.mcp_client.manager import get_mcp_manager

mcp_manager = get_mcp_manager()
if mcp_manager:
    external_tools = mcp_manager.get_all_tools_openai_format()
    if external_tools:
        memory_tools = memory_tools + external_tools
```

Tool names are prefixed with server name to avoid collisions:
- `filesystem` server → `filesystem_read_file`, `filesystem_write_file`
- `github` server → `github_create_issue`, `github_create_pr`
- `blender` server → `blender_create_cube`, `blender_render`

#### API Endpoints

```
GET    /api/mcp/servers              # List configured servers
POST   /api/mcp/servers              # Add new server
DELETE /api/mcp/servers/{name}       # Remove server
POST   /api/mcp/servers/{name}/test  # Test connection
POST   /api/mcp/servers/{name}/reconnect  # Reconnect server
GET    /api/mcp/tools                # List all available tools
GET    /api/mcp/popular              # Get popular server presets
```

#### Configuration Format

```json
{
  "servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-filesystem", "C:/allowed/path"],
      "env": {},
      "enabled": true
    }
  ]
}
```

#### LLM Compatibility

| Provider | Support Level | Notes |
|----------|--------------|-------|
| **Ollama** | Full | All tool-capable models work |
| **LM Studio** | Partial | Model-dependent function calling |

#### Graceful Degradation

If MCP servers are unavailable:
- Internal Roampal tools continue working
- Chat functionality unaffected
- Errors logged but not shown to user

### Session Management
```
GET  /api/sessions/list           # List all conversations
GET  /api/sessions/{id}           # Get conversation history
DELETE /api/sessions/{id}         # Delete conversation
```

### Book Processing
```
POST /api/book-upload/upload        # Upload document (.txt, .md)
GET  /api/book-upload/books         # List all books with metadata
DELETE /api/book-upload/books/{id}  # Delete book (removes from SQLite, ChromaDB, filesystem)
GET  /api/book-upload/search        # Full-text search across all books
POST /api/book-upload/cancel/{task_id} # Cancel processing task
WS   /ws/progress/{task_id}         # WebSocket for real-time progress updates
```

### Model Management

The Model Switcher feature allows runtime hot-swapping of LLM models without restart, with full UI integration.

**Backend (Main)**: [app/routers/model_switcher.py](../app/routers/model_switcher.py) (727 lines, Ollama-only)
**Backend (UI Bundle)**: [ui-implementation/src-tauri/backend/app/routers/model_switcher.py](../ui-implementation/src-tauri/backend/app/routers/model_switcher.py) (1781 lines, full dual-provider)
**Frontend**: [ui-implementation/src/components/ConnectedChat.tsx](../ui-implementation/src/components/ConnectedChat.tsx) (lines 66-445, 1460-1585)

> **Note**: The UI bundle backend has significantly more features than the main codebase backend. LM Studio support, multi-provider detection, and GGUF downloads are only available in the UI bundle. For v0.2.3+, consider backporting these features to the main codebase.

#### Endpoints
```
# Provider Detection (UI Bundle only)
GET  /api/model/providers/detect           # Detect all running providers with model lists
GET  /api/model/providers/all/models       # Get models from ALL detected providers
GET  /api/model/providers/{provider}/models # Get models for specific provider

# Provider Status
GET  /api/model/ollama/status    # Check if Ollama is running
GET  /api/model/lmstudio/status  # Check if LM Studio server is running (UI Bundle only)

# Model Operations
GET  /api/model/available        # List locally installed models (filters embedding models)
GET  /api/model/current          # Get currently active model with provider info
POST /api/model/switch           # Switch active model with auto-provider detection + health check
POST /api/model/pull             # Pull new model (blocking)
POST /api/model/pull-stream      # Pull Ollama model with SSE progress streaming
POST /api/model/download-gguf-stream # Download GGUF from HuggingFace for LM Studio (UI Bundle only)
DELETE /api/model/uninstall/{model_name} # Uninstall model, auto-switch if active

# Context Window Management
GET  /api/model/contexts         # Get all model context configurations
GET  /api/model/context/{model_name} # Get specific model's context info (default, max, current)
POST /api/model/context/{model_name} # Set custom context size (512-200000 tokens)
DELETE /api/model/context/{model_name} # Reset to default context size
```

#### Features

**Multi-Provider Detection** (UI Bundle: [model_switcher.py:34-130](../ui-implementation/src-tauri/backend/app/routers/model_switcher.py#L34)) (Updated 2025-12-02)
- Supports 2 LLM providers: **Ollama** (port 11434) and **LM Studio** (port 1234)
- Provider configuration dictionary with ports, health endpoints, and API styles:
  ```python
  PROVIDERS = {
      "ollama": {"port": 11434, "health": "/api/tags", "api_style": "ollama"},
      "lmstudio": {"port": 1234, "health": "/v1/models", "api_style": "openai"},
  }
  ```
- `/api/model/providers/detect` returns comprehensive provider status:
  ```json
  {
    "providers": [
      {"name": "ollama", "port": 11434, "status": "running", "models": ["qwen2.5:7b", ...], "model_count": 4},
      {"name": "lmstudio", "port": 1234, "status": "running", "models": [], "model_count": 0}
    ],
    "active": "ollama",
    "count": 2
  }
  ```
- Filters embedding models from model lists (nomic-embed, mxbai-embed, all-minilm, bge-, text-embedding)
- Used by frontend to detect missing LLM provider installations
- Shows user-friendly modal with download links for both providers if none are available

**LM Studio Integration** (Added 2025-10-28, Updated 2025-12-02)
- **Architecture**: GUI app + CLI tool (`lms.exe`) + Python SDK (`lmstudio`)
- **Model format**: GGUF files stored in `~/.lmstudio/models/` (NOT `.cache/lm-studio/models/`)
- **API compatibility**: OpenAI-compatible API on `http://localhost:1234/v1/`
- **Model listing**: Returns model IDs as-is (e.g., `qwen2.5-7b-instruct`), no normalization to Ollama format
- **CLI location**: Searches multiple paths:
  - `~/.lmstudio/bin/lms.exe` (primary)
  - `%LOCALAPPDATA%/LM Studio/bin/lms.exe` (alternate Windows location)
- **GGUF Download Mapping** (UI Bundle: [model_switcher.py:41-89](../ui-implementation/src-tauri/backend/app/routers/model_switcher.py#L41)):
  Pre-mapped Q4_K_M quantizations for HuggingFace auto-download:
- **Model Name Resolution** ([model_switcher.py:resolve_model_for_lmstudio](../ui-implementation/src-tauri/backend/app/routers/model_switcher.py)) (Added 2025-12-02):
  Resolves quantized model names (e.g., `qwen2.5:7b-instruct-q8_0`) to HuggingFace download info:
  1. Checks legacy `MODEL_TO_HUGGINGFACE` mapping for direct match
  2. Searches `QUANTIZATION_OPTIONS` for matching `ollama_tag`
  3. Falls back to fuzzy matching on base model name
  This enables LM Studio downloads with specific quantization selections.
  | Model | Repo | Size |
  |-------|------|------|
  | qwen2.5:3b | bartowski/Qwen2.5-3B-Instruct-GGUF | 1.93 GB |
  | qwen2.5:7b | bartowski/Qwen2.5-7B-Instruct-GGUF | 4.68 GB |
  | qwen2.5:14b | bartowski/Qwen2.5-14B-Instruct-GGUF | 8.99 GB |
  | qwen2.5:32b | bartowski/Qwen2.5-32B-Instruct-GGUF | 19.9 GB |
  | qwen2.5:72b | bartowski/Qwen2.5-72B-Instruct-GGUF | 47.4 GB |
  | llama3.2:3b | bartowski/Llama-3.2-3B-Instruct-GGUF | 2.0 GB |
  | llama3.1:8b | bartowski/Meta-Llama-3.1-8B-Instruct-GGUF | 4.9 GB |
  | llama3.3:70b | bartowski/Llama-3.3-70B-Instruct-GGUF | 42.5 GB |
  | mixtral:8x7b | TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF | 26.44 GB |
- **One-step automation** (Updated 2025-10-29):
  1. CLI import: `lms import --yes --copy --user-repo "publisher/model" /path/to/file.gguf`
  2. ~~SDK load: Skipped due to `client.llm.load_new_instance()` hanging indefinitely~~
  3. Models available via API immediately after CLI import completes
- **Model unload**: ~~SDK's `loaded_model.unload()` removed~~ LM Studio API caches models until restart

**Model Switching** (UI Bundle: [model_switcher.py:710-905](../ui-implementation/src-tauri/backend/app/routers/model_switcher.py#L710)) (Updated 2025-12-02)
- **Auto-provider detection**: Automatically detects which provider owns a model:
  1. First checks Ollama via `ollama list` subprocess
  2. Then checks LM Studio via `/v1/models` API
  3. Returns 404 if model not found in either provider
- **Multi-provider support**: Can switch between Ollama and LM Studio models seamlessly
- **Lazy initialization**: Creates `llm_client` if None (happens when app starts with no models)
- Updates `app.state.llm_client` with new provider's base_url, model_name, and api_style
- **Health check** with provider-specific endpoints:
  - Ollama: POST to `/api/chat` with minimal payload
  - LM Studio: POST to `/v1/chat/completions` (OpenAI-compatible)
  - 30-second timeout, 2-second delay for model loading
- **Automatic rollback** on health check failure:
  - Restores previous model_name, base_url, AND api_style
  - Returns HTTP 503 with rollback details
- **Environment persistence**:
  - Updates `ROAMPAL_LLM_PROVIDER`, `OLLAMA_MODEL`, `ROAMPAL_LLM_OLLAMA_MODEL` (Ollama)
  - Updates `ROAMPAL_LLM_PROVIDER`, `ROAMPAL_LLM_LMSTUDIO_MODEL` (LM Studio)
  - Persists to .env file with asyncio file locking

**v0.3.0 Model Switching Fixes** (Added 2026-01-16)
- **Switch lock** (`_switch_lock`): Prevents concurrent model switches from corrupting shared state. Rapid clicking or automated testing no longer causes race conditions.
- **State validation**: After successful switch, validates `llm_client.model_name` matches expected value. Auto-corrects if mismatch detected.
- **Unified model name matching** (`_model_names_match()`): Robust function to match registry names (`llama3.3:70b`) with provider IDs (`llama-3.3-70b-instruct`). Uses 3 strategies:
  1. Direct normalization (`:` → `-`)
  2. GGUF filename stem extraction
  3. Component + size matching (prevents `7b` matching `70b`)
- **Ghost model detection**: New `GET /api/model/ghost-models` and `DELETE /api/model/ghost-models/{id}` endpoints to detect and remove orphaned models that appear in LM Studio but have no matching registry entry.
- **Download cleanup**: On download failure, partial `.gguf` files are automatically deleted via `finally` block with `download_success` flag.
- **Lightweight health checks**: Model switching performs lightweight availability checks (`/api/tags` for Ollama, `/v1/models` for LM Studio) instead of inference requests. Models load into GPU only on first user message, not during health checks.
- **Provider-specific timeouts**: Health check timeout is 10s for both providers (lightweight checks only).
- **Better error messages**: Connection errors show "Ollama not responding. Make sure Ollama is running on port 11434." instead of model load timeouts.

**Model UI Updates** (v0.3.0)
- `qwen2.5:7b` description updated to "Good tool calling (may struggle with 20+ tools)" - 7B models may output tool JSON as text when many MCP tools are loaded.
- Removed "recommended" badge from 7B - use 14B+ for production with many tools.

**Model Installation** (`model_switcher.py` — search for `install_model_stream` / `install_lmstudio_model`)
- **Multi-provider support**: Ollama and LM Studio installation workflows
- **Ollama**: SSE streaming with real-time progress updates, parses download percentage/speed/size
- **LM Studio**: Two-step automation (CLI import + SDK load)
  - Downloads GGUF file to `~/.cache/lm-studio/models/`
  - Uses `lms.exe import --yes --copy --user-repo` for non-interactive import
  - Uses Python SDK `client.llm.load_new_instance()` to load into memory
  - Status flow: `downloading` → `importing` → `loading` → `loaded`
- **Concurrency control**: Download lock + tracking set prevents duplicate downloads
- **Auto-switch**: Automatically switches to newly installed model after successful load
- 10-minute timeout for large models
- WebSocket and SSE endpoints for flexible client support

**Quantization Selection & VRAM Management** ([model_registry.py](../ui-implementation/src-tauri/backend/app/routers/model_registry.py)) (Added 2025-12-02)

Allows users to select specific quantization levels before downloading models, with automatic GPU detection to recommend appropriate quantizations based on available VRAM.

**Architecture:**
- **GPU Detection**: Uses `nvidia-smi` subprocess to detect NVIDIA GPUs and query VRAM (total, used, free)
- **VRAM Headroom**: Reserves ~2GB for system/context, recommends models that fit within available VRAM
- **Quality Ratings**: 1-5 scale (Low → Highest) for each quantization level
- **Auto-recommendation**: Pre-selects the highest quality quantization that fits in detected VRAM

**API Endpoints** ([model_registry.py:470-654](../ui-implementation/src-tauri/backend/app/routers/model_registry.py)):
| Endpoint | Description |
|----------|-------------|
| `GET /api/model/gpu` | Detect GPU info (name, VRAM total/free/used, recommended quant) |
| `GET /api/model/catalog` | Full catalog of models with all quantization options |
| `GET /api/model/recommendations?vram_gb=X` | Filtered models that fit in specified VRAM |
| `GET /api/model/{model_name}/quantizations` | Quantization options for specific model with `fits_in_vram` flag |

**Quantization Options** ([model_registry.py:87-259](../ui-implementation/src-tauri/backend/app/routers/model_registry.py)):
```python
QUANTIZATION_OPTIONS = {
    "qwen2.5:7b": {
        "Q2_K": {"size_gb": 2.8, "vram_gb": 3.5, "quality": 1, "ollama_tag": "qwen2.5:7b-instruct-q2_K"},
        "Q3_K_M": {"size_gb": 3.4, "vram_gb": 4.0, "quality": 2, "ollama_tag": "qwen2.5:7b-instruct-q3_K_M"},
        "Q4_K_M": {"size_gb": 4.68, "vram_gb": 5.5, "quality": 3, "default": True, "ollama_tag": "qwen2.5:7b"},
        "Q5_K_M": {"size_gb": 5.3, "vram_gb": 6.0, "quality": 4, "ollama_tag": "qwen2.5:7b-instruct-q5_K_M"},
        "Q6_K": {"size_gb": 6.1, "vram_gb": 7.0, "quality": 4, "ollama_tag": "qwen2.5:7b-instruct-q6_K"},
        "Q8_0": {"size_gb": 8.1, "vram_gb": 9.0, "quality": 5, "ollama_tag": "qwen2.5:7b-instruct-q8_0"},
    },
    # Similar entries for: qwen2.5:3b, 14b, 32b, 72b, llama3.2:3b, llama3.1:8b, llama3.3:70b, mixtral:8x7b
}
```

**Supported Models** (9 total with 6 quantization levels each):
| Model | Size Range | VRAM Range |
|-------|-----------|------------|
| qwen2.5:3b | 1.0-2.0 GB | 1.5-2.5 GB |
| qwen2.5:7b | 2.8-8.1 GB | 3.5-9.0 GB |
| qwen2.5:14b | 5.3-15.7 GB | 6.5-17.0 GB |
| qwen2.5:32b | 12-37 GB | 14-40 GB |
| qwen2.5:72b | 27-81 GB | 30-85 GB |
| llama3.2:3b | 1.3-3.2 GB | 2.0-4.0 GB |
| llama3.1:8b | 3.0-8.5 GB | 4.0-10.0 GB |
| llama3.3:70b | 26-78 GB | 30-82 GB |
| mixtral:8x7b | 10-30 GB | 12-32 GB |

**UI Integration** ([ConnectedChat.tsx:90-113, 141-184, 2769-2930](../ui-implementation/src/components/ConnectedChat.tsx)):
- **Quantization Selector Modal**: Opens when clicking Install on supported models
- **GPU Info Banner**: Shows detected GPU name and available VRAM
- **Quality Stars**: Visual 1-5 star rating for each quantization
- **VRAM Requirements**: Shows size (GB) and VRAM needed for each option
- **Fits in VRAM Indicator**: Green checkmark for options that fit, red warning for those that don't
- **Pre-selected Option**: Highest quality quant that fits is auto-selected
- **Ollama Tag Mapping**: Each quantization maps to specific Ollama pull tag (e.g., `qwen2.5:7b-instruct-q8_0`)

**Provider Availability Protection** ([ConnectedChat.tsx:2572, 2706, 2944](../ui-implementation/src/components/ConnectedChat.tsx)) (Added 2025-12-02, Updated 2025-12-02)

Install buttons are disabled when the corresponding LLM provider is not available, preventing failed download attempts.

**Implementation:**
- **Model List Install Buttons**: Disabled when provider unavailable
  - Ollama: `!availableProviders.find(p => p.name === 'ollama')?.available`
  - LM Studio: `!availableProviders.find(p => p.name === 'lmstudio')?.available`
- **Quantization Modal Install Button**: Also checks `viewProvider` availability (line 2944)
  - Prevents downloads when modal opened but provider went offline
  - Shows tooltip: "{Provider} is not running"
- **Tooltip on Hover**: Shows explanation when button is disabled:
  - Ollama: "Install Ollama to download models"
  - LM Studio: "Start LM Studio server to install"
- **Visual Feedback**: Disabled buttons show `opacity-50` and `cursor-not-allowed`

**User Experience:**
- Warning banner appears at top of model list when provider unavailable
- Install buttons grayed out with hover tooltip explaining why
- User must install/start provider before downloading models
- Auto-polling (every 10s) re-enables buttons when provider becomes available

**Model Uninstallation** ([model_switcher.py:1298-1443](../app/routers/model_switcher.py)) (Updated 2025-10-31)
- **Multi-provider aware**: Detects provider ownership before uninstalling
- **Ollama**: Calls `ollama rm {model_name}` with 30s timeout
- **LM Studio**: Unloads from memory via SDK, then deletes GGUF file from `~/.cache/lm-studio/models/`
- **Smart auto-switch**: If deleting active model, switches to first available chat model from ANY provider
  - Updates `llm_client.model_name`, `base_url`, and `api_style` to match new provider
  - **HTTP client recycling** ([model_switcher.py:1422-1425](../app/routers/model_switcher.py)): Calls `_recycle_client()` after changing `base_url` to ensure httpx AsyncClient uses correct provider URL (port 1234 for LM Studio, 11434 for Ollama)
  - Updates environment variables for Ollama models
- **Embedding model protection**: Filters out embedding models from fallback list
- If no chat models available, sets `model_name` to `None`
- File locking prevents concurrent .env modifications

**Embedding Model Protection** (Added 2025-10-09)

Chat models and embedding models serve different purposes. Roampal uses bundled `paraphrase-multilingual-mpnet-base-v2` for all embeddings. Ollama embedding models (like `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`) are not used and should not be selected as chat models.

**Protection Layers:**

1. **API Layer** ([model_switcher.py:71-96](../app/routers/model_switcher.py))
   - `/api/model/current` returns `can_chat: bool` flag
   - `can_chat: false` when active model is embedding model or None
   - `is_embedding_model: bool` indicates if current model is for embeddings only
   - Frontend should disable send button when `can_chat: false`

2. **Service Layer** ([agent_chat.py:539-557](../app/routers/agent_chat.py))
   - Single validation method: `_validate_chat_model()` (single source of truth)
   - Returns `(is_valid: bool, error_message: Optional[str])`
   - Checks if model is None or in embedding model list
   - Used by both streaming and non-streaming code paths

3. **Request Handling** ([agent_chat.py:3368-3378](../app/routers/agent_chat.py))
   - `stream_message()` validates before processing, yields `{"type": "done", "content": error_msg}`
   - Backend sends validation errors as dedicated `{"type": "validation_error", "message": error_msg}` WebSocket event
   - Frontend receives validation error and removes user message without creating assistant message
   - Error message: "No chat model available. Please install a model to start chatting."
   - No ERROR logs for expected behavior (validation errors are not exceptions)

4. **Fallback Protection** ([model_switcher.py:426-463](../app/routers/model_switcher.py))
   - During model uninstallation, filters embedding models from available models
   - Only switches to chat models (excludes `nomic-embed-text`, `mxbai-embed-large`, `all-minilm`)
   - If no chat models available after uninstall, warns user via logs

**User Experience:**
- Frontend checks `can_chat` flag from `/api/model/current`
- Send button should be disabled with tooltip: "Install a chat model to start"
- Input placeholder: "Install a chat model to start chatting..."
- Backend validates and returns clean error if frontend check bypassed
- No exception logging for expected states (embedding model active is not an error)

**UI Integration** ([ConnectedChat.tsx](../ui-implementation/src/components/ConnectedChat.tsx)) (Updated 2026-04-10)
- **Model-first design** (v0.3.1): UI shows all models from all providers in a single list — no provider tabs or toggle
- **Provider detection**: Checks for available LLM providers on mount, shows modal if none detected
- **`getModelOptions()`**: Returns ALL installed models from ALL providers with provider badge in description. No longer filters by `selectedProvider`.
- **`getSidecarModelOptions()`**: Groups models into "Recommended" (≤7b) and "All Models" for sidecar selection
- **Chat input availability**: `hasChatModel` checks for chat models across ALL providers, filters out embedding models
- **Setup banners**: Both Ollama and LM Studio banners shown simultaneously when their respective provider is offline
- **Chat model dropdown**: All installed models with provider badge, passes model's actual `provider` to `switchModel()` and `handleUninstallModel()`
- **Sidecar dropdown**: Cross-provider — passes model's actual provider to `/api/model/sidecar/set`. Groups small models as "Recommended".
 - **GPU memory management**: Sidecar shares client with main LLM when same provider/model to avoid GPU duplication. Client locking prevents concurrent model loading. Retry queue with exponential backoff ensures sidecar always runs even if temporarily unavailable.
- **Model Library modal**: Unified layout with ~6 curated "Chat Models" and "Sidecar Models". Per-provider install buttons. "Pull from Ollama" text input for any model by name. LM Studio models detected automatically.
- **Type-to-pull**: Text input in Model Library. Auto-appends `:latest` if no tag. Calls `POST /api/model/pull-stream`.
- **Download progress popup**: Real-time progress with provider-specific status messages
  - Ollama: `downloading` → `complete`
  - LM Studio: `downloading` → `importing` → `loading` → `loaded`
- **Download cancellation**: AbortController cancels in-flight downloads
- **Mid-conversation warning**: Confirms switch if messages exist, non-blocking confirmation
- **Model attribution** ([EnhancedChatMessage.tsx:90-94](../ui-implementation/src/components/EnhancedChatMessage.tsx)): Badge showing which model generated each response
- **Auto-refresh**: Model list refreshes after install/uninstall
- **Persistence**: Selected provider and model saved to localStorage
**Confirmation Modals** (Updated 2025-10-21)
- **Tauri Confirm Bug Fix**: Replaced all native `confirm()` dialogs with custom React modals
  - **Problem 1**: Tauri's native `confirm()` returns `true` when X is clicked (should be `false`)
  - **Problem 2**: `window.confirm()` causes immediate React re-renders before user responds, causing UI elements to disappear while dialog is still open
  - **Solution**: Custom modals with explicit Cancel/Confirm buttons that properly block UI state updates
- **Session Delete Confirmation** ([DeleteSessionModal.tsx](../ui-implementation/src/components/DeleteSessionModal.tsx), [Sidebar.tsx:60,197-208,235-245](../ui-implementation/src/components/Sidebar.tsx)):
  - State: `sessionToDelete` tracks session pending deletion (id + title)
  - Modal shows session title, red warning icon, "Cancel" / "Delete" buttons
  - Delete button executes `onConfirm()` then closes modal via `onCancel()`
  - Only explicit "Delete" button click executes deletion
  - Pattern matches model operation confirmations for consistency
- **Uninstall Confirmation** (lines 79, 436-468, 1993-2029):
  - State: `uninstallConfirmModel` tracks model pending deletion
  - Modal shows model name, red warning icon, "Cancel" / "Delete" buttons
  - Only explicit "Delete" button click executes uninstall
- **Cancel Download Confirmation** (lines 80, 470-480, 1764, 1978, 2031-2075):
  - State: `showCancelDownloadConfirm` triggers modal
  - Yellow warning icon, "Continue Download" / "Cancel Download" buttons
  - Cancellation only on explicit button click, not X close
- **Model Switch Confirmation** (lines 81, 135-145, 148, 482-488, 2077-2120):
  - State: `modelSwitchPending` stores new model name during confirmation
  - Blue info icon, shows current vs. new model comparison
  - "Cancel" / "Switch Model" buttons with clear actions
  - Only shows when switching mid-conversation (messages.length > 0)

**Message Tracking** ([agent_chat.py:761-763](../app/routers/agent_chat.py))
- Every assistant message includes `metadata.model_name`
- Enables audit trail and multi-model conversation analysis

**Concurrency Safety**
- `_download_lock`: Prevents race conditions during download checks
- `_downloading_models`: Set tracking active downloads
- `_env_file_lock`: Prevents concurrent .env file writes
- **Race condition fix** (model_switcher.py:276-283): Check moved INSIDE lock acquisition

**Security**
- Model name validation regex: `^[a-zA-Z0-9_.-]+:[a-zA-Z0-9_.-]+$`
- 100-character max length
- URL encoding/decoding for model names with special chars

**Tauri production fetch wrapper**

Tauri prod builds block native `fetch()` to localhost. All frontend localhost calls go through `apiFetch()` (`ui-implementation/src/utils/fetch.ts`), which falls back to native `fetch` in dev and to Tauri's HTTP client in prod. The original migration (2025-10-14) touched ~14 files; new code should always use `apiFetch` rather than reintroducing raw `fetch` to localhost.
- **Impact**: Model installation, switching, and all API calls now work in compiled MSI

```

### Personality Management

The Personality system allows complete customization of assistant behavior, tone, and identity via YAML templates.

**Backend**: [app/routers/personality_manager.py](../app/routers/personality_manager.py)
**Frontend**: [ui-implementation/src/components/PersonalityCustomizer.tsx](../ui-implementation/src/components/PersonalityCustomizer.tsx)
**Prompt Integration**: [app/routers/agent_chat.py](../app/routers/agent_chat.py) (lines 503-618, 630-638)

#### Endpoints
```
GET  /api/personality/presets         # List available preset and custom templates
GET  /api/personality/current         # Get currently active template
GET  /api/personality/template/{id}   # Get specific template by ID
POST /api/personality/save            # Save/update custom template (with overwrite detection)
POST /api/personality/activate        # Activate template (validates before copying to active.txt)
POST /api/personality/upload          # Upload custom template file
DELETE /api/personality/custom/{id}   # Delete custom template (protects active template)
```

#### Template Structure (YAML)

**Required Fields** (enforced by validation):
- `identity.name` - Assistant name (appears in UI)
- `identity` section - Identity configuration
- `communication` section - Communication style

**Optional Fields**:
- `identity.role` - Role description
- `identity.expertise` - List of expertise areas
- `identity.background` - Background description
- `communication.tone` - warm | professional | direct | enthusiastic
- `communication.verbosity` - concise | balanced | detailed
- `communication.formality` - casual | professional | formal
- `communication.use_analogies` - boolean
- `communication.use_examples` - boolean
- `communication.use_humor` - boolean
- `response_behavior.citation_style` - always_cite | cite_patterns | conversational
- `response_behavior.clarification` - ask_questions | make_assumptions
- `response_behavior.show_reasoning` - boolean
- `memory_usage.priority` - always_reference | when_relevant
- `memory_usage.pattern_trust` - heavily_favor | balanced
- `personality_traits` - List of trait strings
- `custom_instructions` - Freeform text instructions

#### Template Storage

**Presets**: `backend/templates/personality/presets/` (default.txt, professional.txt, teacher.txt)
**Custom**: `backend/templates/personality/custom/`
**Active**: `backend/templates/personality/active.txt` (copied from selected template)

#### Features

**Template Validation** ([personality_manager.py:48-74](../app/routers/personality_manager.py))
- **Strict validation**: Enforces required sections (identity, communication) and identity.name field
- YAML syntax validation with friendly error messages
- Warns on missing recommended sections (response_behavior, memory_usage)
- Validates on save, upload, and activate operations

**Template Conversion** ([agent_chat.py:532-618](../app/routers/agent_chat.py))
- Converts YAML template to natural language prompt
- Generates sections for identity, communication style, response approach, memory usage, traits
- Includes custom instructions verbatim

**Prompt Injection** ([agent_chat.py:630-638](../app/routers/agent_chat.py))
- Personality prompt is **Section 1** of every chat message prompt
- Loaded with file watching and caching (mtime check for performance)
- **Fallback handling**: If template fails to load, uses minimal default personality
- Injected BEFORE core memory instructions (user-customizable personality takes precedence)

**Template Management** ([personality_manager.py](../app/routers/personality_manager.py))
- **Overwrite detection** (line 216): Returns `overwrite: true` when saving over existing template
- **Active template protection** (line 306): Prevents deletion of currently active template
- **Optimized detection** (line 137-149): Uses content hash for faster preset matching
- **Preset protection**: Cannot delete preset templates (only custom)

**UI Integration** ([PersonalityCustomizer.tsx](../ui-implementation/src/components/PersonalityCustomizer.tsx))

**Quick Settings Mode** (lines 60-151, 444-508):
- **Identity Controls**: Assistant Name (text), Assistant Identity/Background (textarea), Primary Role (text)
- **Communication Style**: Tone dropdown, Verbosity dropdown, Formality dropdown
- **Communication Features**: Use Analogies (toggle), Use Examples (toggle), Use Humor (toggle)
- **Memory**: Memory Priority dropdown
- **Custom Instructions**: Freeform textarea
- Real-time YAML generation from form inputs
- Toggle UI for boolean fields (Enabled/Disabled states)
- Beginner-friendly interface with descriptions for each field

**Advanced Mode** (lines 510-547):
- Full YAML editor with syntax highlighting
- Live validation with friendly error messages (line 139-145)
- Example template loader (line 297-304)
- Download/export functionality (line 285-295)

**Name Sync** ([EnhancedChatMessage.tsx:42](../ui-implementation/src/components/EnhancedChatMessage.tsx), [Sidebar.tsx:98](../ui-implementation/src/components/Sidebar.tsx))
- Robust regex parsing: `/name:\s*(?:"([^"]+)"|'([^']+)'|([^\n]+))/`
- Handles quoted and unquoted values, special characters
- Sidebar polls every 5 seconds for live updates

**Modal Access**: Settings → "Personality & Identity" ([ConnectedChat.tsx:1715-1730](../ui-implementation/src/components/ConnectedChat.tsx))

#### Security & Validation

- ✅ Required field enforcement prevents broken prompts
- ✅ YAML syntax validation with error details
- ✅ Filename sanitization (alphanumeric, dash, underscore only)
- ✅ Active template validation before activation
- ✅ Overwrite detection and logging
- ✅ Preset immutability (cannot modify/delete presets)
- ✅ Active template deletion protection

```

## Configuration

### Environment Variables
```bash
# Core Settings
ROAMPAL_API_PORT=8001          # Default for backend; PROD desktop builds use 8765, DEV desktop 8766
ROAMPAL_DATA_DIR=Roampal       # Override the data subfolder ('Roampal_DEV' for dev builds)
ROAMPAL_DEV=                   # Set automatically by Tauri debug builds — never set this manually

# LLM Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=                  # Leave unset to auto-select first installed model; ROAMPAL_LLM_OLLAMA_MODEL takes precedence

# Memory Settings (defaults shown)
ROAMPAL_ENABLE_MEMORY=true
ROAMPAL_ENABLE_OUTCOME_TRACKING=true
# ROAMPAL_ENABLE_KG removed v0.3.1 (KG fully removed)

# Logging
ROAMPAL_LOG_LEVEL=INFO
```

The canonical list lives in `backend/.env.example`; this section documents the variables most likely to come up during ops. Rate limiting is hardcoded at 100 req/min and is not env-configurable.

### Memory Retention Policies
- **Working Memory**: 24 hours
- **History**: 30 days (configurable)
- **Patterns**: Permanent (demoted if score < 0.4, was 0.3 before v0.2.3)
- **Books**: Permanent
- **High-value threshold**: 0.9 score (preserved beyond retention period)
- **Promotion threshold**: 0.7 score (minimum for promotion)
- **Demotion threshold**: 0.4 score (patterns → history, raised from 0.3 in v0.2.3)
- **Deletion threshold**: 0.2 score (history deletion)
- **New item deletion threshold**: 0.1 score (for items < 7 days old, more lenient)

## Storage Layout

**Data Location Strategy (Automatic Detection):**
- **3-Tier Automatic Path Selection** - No manual migration required
- System automatically detects environment and chooses appropriate data location
- Implementation: [settings.py:18-48](../ui-implementation/src-tauri/backend/config/settings.py)

**Priority Order (v0.3.0 - Data Path Isolation):**
1. **Environment Variable Override** (Advanced users)
   - If `ROAMPAL_DATA_DIR` is set → uses custom path
   - Allows explicit control over data location

2. **ROAMPAL_DEV Flag (v0.3.0 NEW - Dev/Prod Isolation)**
   - If `ROAMPAL_DEV=true` → uses `Roampal_DEV` subfolder
   - Tauri sets this automatically based on debug/release build
   - Prevents dev builds from corrupting production data
   - Windows: `%APPDATA%\Roaming\Roampal_DEV\data\`

3. **Development Mode** (Auto-detected)
   - Checks if `ui-implementation/` folder exists
   - If YES → uses `PROJECT_ROOT/data/`
   - For developers running from source

4. **Production Mode** (Auto-created)
   - If no `ui-implementation/` folder → bundled .exe detected
   - Windows: `%APPDATA%\Roaming\Roampal\data\`
   - macOS: `~/Library/Application Support/Roampal/data/`
   - Linux: `~/.local/share/roampal/data/`
   - Directory automatically created if it doesn't exist
   - Data survives app reinstalls and updates

**Directory Structure (Same for all modes):**
```
data/
├── chromadb/                 # Vector embeddings (ChromaDB collections)
│   ├── roampal_books/        # Reference documents
│   ├── roampal_working/      # Current context
│   ├── roampal_history/      # Past conversations
│   ├── roampal_memory_bank/  # LLM's stored notes
│   └── roampal_patterns/     # Proven solutions
├── uploads/                  # Document storage
├── sessions/                 # Conversation logs (JSONL)
│   └── *.jsonl              # Conversation history with citations
└── vector_store/             # Legacy compatibility path
```

## UI Features

### Chat Interface
- Clean, minimalist design with dark theme
- **WebSocket streaming** (Updated 2025-10-08: Real-time status updates)
  - WebSocket connection per conversation (`/ws/conversation/{id}`)
  - Backend pushes status updates, response content, and citations
  - **Status Messages**: Global processing indicator shows real-time status (e.g., "Processing...", "Searching memory...")
  - **Inline title generation** (Updated 2025-10-02): After first exchange, title is auto-generated and streamed as part of the response flow (single LLM call optimization)
- **Chat input enhancements** (Updated 2025-10-11)
  - Auto-expanding textarea (handles both newlines and text wrapping)
  - Inline error messages with auto-dismiss (5s) instead of alerts
  - Visual processing indicator with spinning cancel button
  - Send button with scale/glow animation on hover
  - Keyboard shortcut tooltips on hover (Enter, ⌘+Enter, Esc)
  - Command palette with slash commands
  - **Image attachments (v0.3.3+)**: PhotoIcon button next to send; paste-to-attach also supported. Bytes persist to `<DATA_PATH>/attachments/<sha256>.<ext>` and survive refresh/restart via `/api/attachments/{filename}`. PhotoIcon greys out automatically for models without vision capability. Non-image file uploads (PDFs, docs) still go through the Document Processor tab.
- **Enhanced Markdown rendering** (Updated 2025-10-07)
  - Full GitHub Flavored Markdown support via `react-markdown`
  - Styled headings (H1-H3) with proper hierarchy
  - **Bold**, *italic*, `inline code` formatting
  - Syntax-highlighted code blocks with copy button
  - Ordered and unordered lists
  - Blockquotes styled as blue callout boxes
  - Custom callout syntax: `:::success`, `:::warning`, `:::info`
  - Preserves terminal monospace aesthetic
  - LLM has full creative freedom to format responses
- **Processing Status Indicator** (Updated 2025-10-08)
  - **Global Processing Indicator**: Single unified status display positioned like user messages (with `>` prefix)
  - **Real-time Updates**: WebSocket pushes status from backend ("Processing...", "Searching memory...", etc.)
  - **Intent-Based Messages**: When no explicit status provided, analyzes user message intent to show contextual status
  - **Visual Flow Example**:
    - User sends message
    - `> Processing your request...` (global indicator appears)
    - `⋯ Searching memory` (tool starts)
    - `✓ Searching memory · 10 results` (tool completes)
    - Response content appears, indicator disappears
  - **Implementation**:
    - Backend: `agent_chat.py:2720-2855` (_run_generation_task with WebSocket streaming)
    - Frontend: `useChatStore.ts:477-482` (WebSocket status handler), `TerminalMessageThread.tsx:520-542` (processing indicator)
- **Thinking tags display** (REMOVED - v0.2.5 - models output thinking inconsistently)
  - Collapsible reasoning block removed due to model incompatibilities (API field vs tags vs plain text)
  - Backend now filters `<think>` tags during streaming (no flash in UI)
  - Frontend shows animated "Thinking..." status via `ThinkingDots` component during thinking phase
  - **Single indicator UX (v0.2.5)**: "Thinking..." hides when tools are running (inline `⋯ searching...` indicators take over)
    - Prevents duplicate indicators (was showing both "Thinking..." and tool status)
    - Flow: Thinking... → tool runs (`⋯`) → tool completes (`✓`) → Thinking... (if more processing) → response streams
  - **Implementation**:
    - Backend: `agent_chat.py:742-781` (streaming filter with `thinking_start`/`thinking_end` events)
    - Frontend: `useChatStore.ts:631-640` (thinking event handler), `TerminalMessageThread.tsx:10-25` (ThinkingDots component)
    - Frontend: `TerminalMessageThread.tsx:502-519` (hides ThinkingDots when tools running)

#### Technical Debt: Thinking Tags - History

**Original Removal**: 2025-10-17 - Streaming text and XML tag parsing were fundamentally incompatible
**Re-enabled**: 2025-12-03 (v0.2.4) - Post-stream extraction approach
**Final Removal**: 2025-12-05 (v0.2.5) - Collapsible block removed, replaced with animated "Thinking..." status indicator

**Original Issues**:
1. Complex accumulation buffer (introduced bugs)
2. Character-by-character FSM parser (overkill)
3. Non-streaming mode (defeats purpose of real-time UI)

**v0.2.4 Solution**: Post-stream extraction
- Full response accumulated during streaming (already happening for session save)
- After stream complete, regex extracts `<think>` tags from accumulated response
- Single `thinking` event sent to frontend with extracted content
- Zero mid-stream parsing complexity, works reliably
- **Markdown content overflow protection** (Updated 2025-10-11)
  - **CSS constraints**: `.markdown-content` class with `max-width: 100%`, `overflow-wrap: break-word`, `word-break: break-word`
  - **Flex container fix**: Added `min-w-0` to assistant message flex container (line 418) to enable proper shrinking
  - **Code block handling**: `overflow-x: auto` on `<pre>` tags for horizontal scrolling when needed
  - Prevents markdown content (especially malformed code blocks) from extending beyond screen width
  - **Implementation**: `index.css:55-70` (CSS rules), `TerminalMessageThread.tsx:418` (flex constraint)
- **Memory citations display** (Updated 2025-10-08)
  - **UI Display**: Bottom-right metadata placement (only after streaming complete)
  - Collapsible block showing "X memories" used
  - Expands to show: collection name (color-coded), full text preview
  - **Note**: Confidence scores are NOT displayed in UI (removed 2025-10-07 due to unreliable metrics)
  - **Backend Only**: System calculates confidence using exponential decay `exp(-distance / 100.0)` for internal learning (outcome detection, memory promotion)
  - Color-coded collections: purple (books), blue (working), green (history), yellow (patterns), pink (memory_bank)
  - **Implementation**: [TerminalMessageThread.tsx:10-59](../ui-implementation/src/components/TerminalMessageThread.tsx#L10-L59) - CitationsBlock component

### First-Run Experience

**Welcome modal (v0.3.1+):** `OllamaRequiredModal.tsx` explains the chat-model + sidecar-model architecture, surfaces a curated tool-capable model list (qwen3, gemma4, gpt-oss), and treats LM Studio as a first-class provider. After the first chat-model install a follow-up toast prompts the user to pick a sidecar.

**Zero-Model onboarding behavior:**
- **Graceful startup**: backend starts even with no models installed
- **Empty State UI**: prominent centered message when no models are available
- **Clear CTA**: large "Install Your First Model" button in the chat area
- **Embedding model filter**: dropdown excludes embedding-only models (`nomic-embed`, `llava`, `bge-`, `all-minilm`, `mxbai-embed`) — `ConnectedChat.tsx`, `getModelOptions()`
- **Smart first-model auto-switch**: after downloading the first chat model, polls `/api/model/current`; if backend has no chat model (embedding-only or none), auto-switches to the newly installed model. Subsequent installs show a success toast only; user switches manually.
 - **Lazy model loading**: Models load into GPU memory only on first user message, not during app startup or model switching. Health checks verify model availability without loading. Sidecar operations use client locking to prevent concurrent model loading and retry queue for reliability.
- **Auto-selection behavior**: If no model is configured in `.env`, app selects first available model from detected provider. **Recommendation**: Configure `OLLAMA_MODEL` or `ROAMPAL_LLM_OLLAMA_MODEL` in `.env` to avoid auto-selection of large models.

#### Model Selection Synchronization

**Backend as Source of Truth:**
- **Initial sync on mount**: UI syncs with backend on startup via `/api/model/current` ([ConnectedChat.tsx:246-250](../ui-implementation/src/components/ConnectedChat.tsx#L246))
- **Auto-sync after model list changes** ([ConnectedChat.tsx:187-188](../ui-implementation/src/components/ConnectedChat.tsx#L187)): `fetchModels()` calls `fetchCurrentModel()` after updating available models, ensuring UI syncs when backend auto-switches providers (e.g., after uninstalling active model)
- **fetchCurrentModel()** ([ConnectedChat.tsx:268-308](../ui-implementation/src/components/ConnectedChat.tsx#L268)):
  - Fetches backend's active model via `/api/model/current`
  - Backend returns `current_model`, `provider`, `can_chat`, `is_embedding_model` flags
  - Updates UI `selectedModel`, `selectedProvider`, and localStorage to match backend
  - Handles case where backend has no chat model available (switches to first available)
- Prevents phantom selection where UI shows different model than backend is using

**Installation Auto-Switch Logic:**
- **First chat model**: Auto-switches when `can_chat: false` or `is_embedding_model: true` on backend
- **Subsequent models**: Shows "✓ model installed successfully" toast, requires manual switch
- **Rationale**: Unblocks user on first install, preserves workflow control on subsequent installs

**Flow:**
1. User launches Roampal.exe for first time
2. UI detects no models available
3. Chat area shows empty state with install prompt
4. User clicks button to open Model Manager
5. User browses and downloads preferred model(s)
6. System ready to chat

**Implementation:**
- Backend: [main.py:243-245](../main.py) - Allows startup without models, logs setup mode
- UI Empty State: [ConnectedChat.tsx:1502-1529](../ui-implementation/src/components/ConnectedChat.tsx) - Centered prompt with icon, message, and install button
- Dropdown Filter: [ConnectedChat.tsx:1343](../ui-implementation/src/components/ConnectedChat.tsx) - Uses `getModelOptions()` to exclude embedding models
- Dropdown Text: [ConnectedChat.tsx:1384](../ui-implementation/src/components/ConnectedChat.tsx) - Shows "Download Your First Model" when empty

### Personality Customization
Users can customize the AI's personality and response style while preserving core memory functionality:

**Customizable Elements:**
- **Identity & Name** - Customize assistant name (appears in UI sidebar and chat messages)
- **Communication Style** - Tone, verbosity, formality with visual icons
- **Response Behavior** - Citation style, clarification approach, reasoning visibility
- **Memory Usage** - Reference frequency and pattern trust levels
- **Personality Traits** - Custom traits list (helpful, direct, patient, etc.)
- **Custom Instructions** - Free-form additional guidelines

**Base Templates:**
- `default.txt` - Balanced, memory-enhanced assistant (Recommended)
- `professional.txt` - Concise, direct, business-focused
- `teacher.txt` - Detailed, patient, educational

**File Format:** YAML-based templates stored in `backend/templates/personality/`

**User Interface (Two Modes):**
- **Quick Settings** (Default) - 4 essential fields with dropdowns and clear labels
  - Assistant Name (text field)
  - Conversation Style (warm/professional/direct/enthusiastic)
  - Response Length (concise/balanced/detailed)
  - Memory References (when relevant/frequently)
  - Custom Instructions (optional textarea)
- **Advanced Mode** - Full YAML editor with:
  - Real-time validation (using js-yaml library)
  - Friendly error messages
  - Inline comments showing available options
  - Load example template button
  - Syntax highlighting

**User Capabilities:**
- Select from preset templates
- Edit in Quick Settings (beginner-friendly) OR Advanced mode (full control)
- Export personality configuration to file
- Changes apply immediately (no restart required)
- Unsaved changes warning with confirmation dialogs
- Reset to last saved version

**System-Protected Elements (Hardcoded):**
- Memory context injection logic
- Collection reliability labels (✓ PROVEN SOLUTION, 📚 Reference docs, etc.)
- Core instructions for using memory system
- Prompt structure order
- Memory variable injection points

**Technical Flow:**
```
1. User selects/edits personality template in Quick Settings or Advanced mode
2. Frontend validates YAML in real-time (js-yaml library)
3. On save: Backend validates and saves to backend/templates/personality/active.txt
4. agent_chat.py loads template on each message (cached with mtime checking)
5. Template converted to natural language and prepended to prompt
6. Memory context auto-injected after personality layer
7. LLM receives: [Personality] + [Core Instructions] + [Memory] + [History] + [Question]
8. UI components fetch assistant name from identity.name field:
   - Sidebar displays custom name above conversations (polls every 5s)
   - Chat messages show custom name instead of "RoamPal"
```

**Prompt Structure:**
```
[PERSONALITY LAYER]           ← User-customizable
{{ personality_template }}

[CORE MEMORY INSTRUCTIONS]    ← Hardcoded (system-critical)
{{ memory_usage_guidelines }}

[MEMORY CONTEXT]              ← Auto-injected by Roampal
{{ memory_results with collection labels }}

[CONVERSATION HISTORY]        ← Auto-injected by Roampal
{{ recent_messages }}

[CURRENT QUESTION]            ← Auto-injected by Roampal
{{ user_input }}
```

**Key Implementation Files:**
- **Backend:**
  - `app/routers/personality_manager.py` - API endpoints for CRUD operations
  - `app/routers/agent_chat.py` - Template loading and caching (lines 121-124, 376-491)
  - `backend/templates/personality/active.txt` - Currently active template
  - `backend/templates/personality/presets/` - Built-in presets (default, professional, teacher)
- **Frontend:**
  - `ui-implementation/src/components/PersonalityCustomizer.tsx` - Main UI component
  - `ui-implementation/src/components/Sidebar.tsx` - Displays custom name (line 88-113)
  - `ui-implementation/src/components/EnhancedChatMessage.tsx` - Shows name in messages (line 28-50)

**UX Best Practices Applied:**
- Clear terminology ("Quick Settings" vs "Advanced", not "Simple" vs "Guided")
- Progressive disclosure (info panel toggleable, advanced mode optional)
- Unsaved changes protection with confirmation dialogs
- Real-time validation with friendly error messages
- Visual feedback (amber dot for unsaved, checkmark on success)
- Consistent zinc color palette matching rest of UI
- Heroicons instead of emojis for professional appearance

This architecture ensures users can fully customize personality while preserving Roampal's core memory intelligence.

### Processing Transparency
When the AI is processing queries, it shows real-time status:
- Global processing indicator (no prefix, clean display)
- Real-time status updates via WebSocket ("Processing...", "Searching memory...", etc.)
- Intent-based messages when no explicit status provided
- Clean, single-purpose display (no redundant components)
- Automatically hides when response appears

### Session Management
- Create/switch conversations
- Automatic inline title generation (after first exchange, single LLM call)
- **Delete conversations** with hover-activated trash button
  - Subtle trash icon appears on hover over conversation items
  - Confirmation dialog prevents accidental deletion
  - Backend prevents deletion of active conversation (400 error)
  - Memory cleanup removes all ChromaDB entries for deleted conversation
  - Session list auto-refreshes after deletion
- Conversation history with atomic file writes
- Memory context preservation
- Session files protected with FileLock + fsync for power-loss safety

## Security Features

### Development Mode
- No authentication required
- All localhost origins allowed
- Debug logging enabled
- Rate limiting: 100 req/min (always enabled)

### Production Mode
- API key authentication
- IP whitelisting (localhost only)
- Rate limiting: 100 req/min (always enabled)
- Sanitized logging
- CORS restrictions

### Desktop App Process Lifecycle (v0.2.3)

The Tauri desktop app manages the Python backend process with proper cleanup on window close.

**Implementation:** [main.rs:410-424](../ui-implementation/src-tauri/src/main.rs#L410-L424)

**Process Management:**
```rust
// Backend process is tracked in shared state
struct BackendProcess(Arc<Mutex<Option<Child>>>);

// Window close handler kills the backend
main_window.on_window_event(move |event| {
    match event {
        tauri::WindowEvent::CloseRequested { .. } | tauri::WindowEvent::Destroyed => {
            if let Ok(mut backend) = backend_clone.lock() {
                if let Some(mut child) = backend.take() {
                    let _ = child.kill();
                    let _ = child.wait();  // Wait for full termination
                }
            }
        }
        _ => {}
    }
});
```

**Key Events Handled:**
| Event | Description |
|-------|-------------|
| `CloseRequested` | User clicks X button - kills backend immediately |
| `Destroyed` | Window destroyed - fallback cleanup |

**Why Both Events?**
- `CloseRequested` fires first when user clicks X (before window closes)
- `Destroyed` fires after window is already gone (backup handler)
- Using both ensures backend is killed in all close scenarios

**Previous Issue (Fixed v0.2.3):**
- Only handled `Destroyed` event, not `CloseRequested`
- Backend processes became orphaned when closing via X button
- Users had to manually kill Python processes via Task Manager

## Performance Optimization

### Caching
- Concept extraction cached
- Embeddings reused when possible
- Knowledge graph paths cached

### Bounded Collections
- Working memory: Max 100 items
- Memory bank: Max 500 items (`MemoryBankService.MAX_ITEMS = 500`)
- History per conversation: 20 messages
- LLM context: Last 4 exchanges (8 messages)

### Lazy Loading
- Memory search on-demand
- Metrics collected but not exported by default

## Learning System Details

### Outcome Detection Integration

Outcome detection is integrated into the streaming chat flow:
1. When user sends a message, system checks if previous message was from assistant
2. If yes, analyzes [previous assistant response, current user message] for outcome
3. Scores the previous exchange in memory based on detected outcome
4. Memory items are promoted/deleted based on accumulated scores

### How Roampal Learns

1. **Concept Extraction**: Identifies key terms and patterns from queries
2. **Usage Tracking**: Records which memories were helpful
3. **Outcome Recording**: Detects success/failure from conversation
4. **Adaptive Routing**: Learns best memory collections for concepts
5. **Pattern Recognition**: Identifies recurring problem→solution pairs

### Example Learning Cycle
```
User: "How do I fix authentication errors?"
Roampal: [Searches all collections] "Try checking your API key..."
User: "That worked, thanks!"
System: Records success → Links "authentication" to solution →
        Updates routing to prioritize this pattern
Next query about authentication → Faster, more accurate response
```


## Troubleshooting

### Memory not learning?
1. Check `ROAMPAL_ENABLE_OUTCOME_TRACKING=true`
2. Verify clear success/failure signals in conversation
3. Check `outcomes.db` is being written

### Slow responses?
1. Check Ollama is running (`http://localhost:11434`)
2. Verify ChromaDB performance
3. Review memory collection sizes
4. Check system resources

### High memory usage?
1. Check conversation count in cache
2. Review working memory size
3. Run memory cleanup/promotion
4. Restart to clear caches

## Architecture Decisions

### ADR-001: Single Memory System
**Decision**: Use UnifiedMemorySystem as the single source of truth
**Rationale**: Eliminates complexity, ensures consistency
**Impact**: All features integrate through one system

### ADR-002: Memory-First Design
**Decision**: Build all features on top of memory system
**Rationale**: Memory is the core value proposition
**Impact**: Stable foundation for all enhancements

### ADR-003: Local-First Storage
**Decision**: All data stored locally, no cloud dependencies
**Rationale**: Privacy, performance, offline capability
**Impact**: Full data ownership for users

### ADR-004: Learn from Interaction
**Decision**: No separate training, learn during conversations
**Rationale**: Continuous improvement, personalized learning
**Impact**: System improves with use

### ADR-005: Transparent AI
**Decision**: Show thinking process when helpful
**Rationale**: Build trust through transparency
**Impact**: Users understand AI reasoning

### ADR-006: Security-First Book Processing
**Decision**: Implement comprehensive input validation and security checks
**Rationale**: Single-user local app still needs protection from accidental corruption
**Impact**: Prevents data integrity issues and self-inflicted prompt injection
**Implementation**: File type whitelisting, size limits, encoding validation, UUID validation, metadata length limits

## Security

**⚠️ IMPORTANT: This is a single-user local application. It is NOT designed for multi-user or public internet deployment without additional security measures.**

### Security Measures (Current)

**1. Data Privacy** ✅
- All data stored locally in `/data` directory
- No cloud sync or external data transmission
- API keys stored in `.env` (gitignored)
- User conversations and memories never leave your machine

**2. CORS Protection** ✅
- Restricted to `localhost` origins only (Tauri + dev servers)
- No wildcard (`*`) origins allowed
- Prevents external websites from accessing your local API
- Location: `main.py:232` - `allow_origins` from env var

**3. Input Validation** ✅
- Message length limit: 10,000 characters (configurable via `ROAMPAL_MAX_MESSAGE_LENGTH`)
- Conversation size limit: 1000 messages per session (prevents OOM crashes)
- Control character removal (prevents injection attacks)
- Empty message rejection
- Location: `app/routers/agent_chat.py:58-70, 207-210, 195-197`

**4. File Upload Security** ✅
- File size limit: 10MB maximum for books
- File type restriction: Only `.txt` and `.md` allowed
- UTF-8 encoding validation (rejects binary/malformed files)
- Duplicate detection to prevent storage abuse
- Prompt injection sanitization: Removes malicious patterns ([IGNORE], [SYSTEM], <|im_start|>, etc.)
- Location: `backend/api/book_upload_api.py:44-125`, `app/routers/agent_chat.py:206-218`

**5. Path Traversal Protection** ✅
- UUID validation for book_id and session_id (prevents `../../` attacks)
- Files scoped to `/data/books/` and `/data/sessions/` directories
- Regex validation on all ID parameters
- Path resolution checking with relative_to() validation
- Location: `backend/api/book_upload_api.py:215-220`, `app/routers/sessions.py:116-123`

**6. Concurrency Protection** ✅
- Per-session async locks prevent race conditions in file writes
- Atomic writes to session JSONL files
- Thread-safe conversation history management
- Location: `app/routers/agent_chat.py:110-117`

**7. GitIgnore Protection** ✅
- `.env` files excluded (API keys protected)
- `/data` directory excluded (user data protected)
- Log files excluded (prevents info leakage)
- Session files excluded (conversation privacy)
- Location: `.gitignore`

### Security Limitations (By Design for Single-User)

**Not Implemented** ❌
- No user authentication (not needed for localhost single-user)
- No rate limiting (single user can't DoS themselves)
- No API key rotation (manual management only)
- No HTTPS/TLS (localhost uses HTTP)
- No request/response audit logging (privacy-first design)

### For Open Source Contributors

**If you're forking this project:**

1. **API Keys**: Never commit `.env` files - use `.env.example` as template
2. **User Data**: Never commit `/data` directory - it contains personal information
3. **Security Model**: This is designed for localhost single-user use
4. **Production Deployment**: If deploying publicly, you MUST add:
   - User authentication (JWT/session tokens)
   - Rate limiting (prevent API abuse)
   - HTTPS/TLS encryption
   - Input sanitization review
   - Security audit

### Threat Model

**Protected Against** ✅
- Path traversal attacks (UUID validation, path resolution checking)
- Control character injection (XSS prevention)
- Prompt injection attacks (regex-based sanitization)
- File upload abuse (size/type limits: 10MB books via Document Processor)
- DoS via unbounded growth (1000 message limit per conversation)
- Race conditions (per-session async locks)
- External CORS attacks (localhost-only)
- Accidental credential leaks (gitignore)

**Not Protected Against** ⚠️
- Malicious localhost applications (same-origin access)
- Physical machine access (no encryption at rest)
- OS-level vulnerabilities (depends on system security)
- Social engineering (user must protect API keys)

### Best Practices for Users

1. **API Key Security**
   - Store OpenAI/Anthropic keys in `.env` only
   - Never share your `.env` file
   - Rotate keys if compromised

2. **Data Backups**
   - Export data regularly (Settings → Export Data)
   - Back up `/data` directory before major updates
   - Keep backups offline for privacy

3. **System Security**
   - Run Roampal on trusted local machine only
   - Keep OS and dependencies updated
   - Use firewall to block external access to port 8000

4. **For Developers**
   - Review security settings in `.env.example`
   - Test file uploads with edge cases (large files, weird encodings)
   - Never disable CORS restrictions


## Out-of-Memory graceful degradation

When a model is configured with a context window too large for the host's RAM/VRAM, Ollama terminates the runner with `"llama runner process has terminated: exit status 2"`. The chat path handles this in three layers:

1. **Lightweight health check during model switch.** `model_switcher.py` validates a new model with `num_ctx=2048, num_predict=10`, so the switch itself succeeds even on weak hardware.
2. **Auto-retry on OOM during chat.** `ollama_client.py` detects the `terminated` error in 500 responses, retries the same request with `num_ctx=2048`, and tags the payload `_oom_recovered=True`.
3. **User-facing warning.** The recovered response is prefixed with a "Memory limit reached" banner that names the original context size and points at Settings → Context Window Settings → lower context for the offending model.

Strong-hardware users never see this path. Weak-hardware users get a working response on the first message and a one-click permanent fix.

---

## Per-release implementation notes

Historical per-release implementation breakdowns are maintained in `dev/docs/releases/vX.Y.Z/RELEASE_NOTES.md`:

- v0.3.0 — `releases/v0.3.0/RELEASE_NOTES.md` (duplicate-prompt race, data deletion toasts, ChromaDB 1.x migration safety)
- v0.3.1 — `releases/v0.3.1/RELEASE_NOTES.md` (KG removed, TagCascade landed, ONNX embeddings)
- v0.3.2 — `releases/v0.3.2/RELEASE_NOTES.md` (TagCascade `$contains` repair, chat-path latency, customer bug batch)
- v0.3.3 — `releases/v0.3.3/RELEASE_NOTES.md` (multimodal images, dynamic capability detection, issue-#8 closure, 21 defects)

This document describes the current state of the system, not the history of how it got there.

---

## License

Apache 2.0 — see `LICENSE` for details.
