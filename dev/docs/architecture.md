# Roampal Architecture

## Overview

Roampal is an intelligent chatbot with persistent memory and learning capabilities. The system features a **memory-first** architecture that learns from conversations and improves over time.

## Architecture Refactor (v0.2.7)

**IMPORTANT**: In v0.2.7, the monolithic `UnifiedMemorySystem` (4,746 lines) was refactored into a **facade pattern** with **8 extracted services**. v0.2.8 completed API compatibility and stabilized the architecture. v0.2.9 added Ghost Registry for book deletion, `sort_by`/`related` MCP parameters, and critical bug fixes. v0.2.10 added ChromaDB error handling (ghost entries), schema migration for older data, and fixed memory promotion to run on startup. v0.2.11 fixed critical KG performance (O(n×m) → O(n+m), 25x faster), added message virtualization, and optimized store subscriptions. Line number references throughout this document may point to the pre-refactor monolith.

### New Architecture

| Component | Lines | Purpose |
|-----------|-------|---------|
| `unified_memory_system.py` | 1203 | **Facade** - coordinates services, maintains API |
| `knowledge_graph_service.py` | 949 | KG operations, routing patterns, entity extraction |
| `search_service.py` | 646 | Search, reranking, dynamic weighting |
| `chromadb_adapter.py` | 604 | Vector DB operations, BM25 hybrid search |
| `smart_book_processor.py` | 657 | Book ingestion, chunking, contextual embedding |
| `content_graph.py` | 543 | Entity relationships, content KG |
| `promotion_service.py` | 473 | Working→History→Patterns promotion |
| `context_service.py` | 455 | Conversation context analysis |
| `routing_service.py` | 444 | Query routing, acronym expansion |
| `memory_bank_service.py` | 430 | Memory bank CRUD operations |
| `outcome_service.py` | 370 | Outcome recording, score updates |
| `scoring_service.py` | 324 | Wilson scoring, score calculations |
| `types.py` | 296 | Shared types, ActionOutcome enum |

### Key Methods by Service

| Feature | Pre-Refactor Location | Post-Refactor Location |
|---------|----------------------|------------------------|
| Cross-encoder reranking | `unified_memory_system.py:591-659` | `search_service.py:447+` |
| Acronym expansion | `unified_memory_system.py:972-1144` | `routing_service.py:39+` |
| Outcome recording | `unified_memory_system.py:2296-2424` | `outcome_service.py:50+` |
| KG routing updates | `unified_memory_system.py` | `knowledge_graph_service.py:381+` |
| Promotion logic | `unified_memory_system.py` | `promotion_service.py` |
| Context analysis | `unified_memory_system.py:1736` | `context_service.py` |

### API Compatibility

The facade (`unified_memory_system.py`) maintains backwards compatibility with existing routers:
- **Core**: `search()`, `store()`, `record_outcome()` - same signatures
- **Stats**: `get_stats()`, `get_kg_entities()`, `get_kg_relationships()` - same signatures
- **Cold Start**: `get_cold_start_context()` - auto-injects user profile, patterns, history
- **Context**: `detect_context_type()` - returns debug/error/general/etc
- **Learning**: `get_action_effectiveness()`, `record_action_outcome()` - causal learning
- **Routing**: `get_tier_recommendations()`, `_update_kg_routing()` - KG-based routing
- **KG Ops**: `get_facts_for_entities()`, `_cleanup_kg_dead_references()` - entity lookup
- **Backup**: `export_backup()`, `import_backup()` - state persistence
- **Embedding**: `_generate_contextual_prefix()` - contextual retrieval (Anthropic technique)
- Collection names use `roampal_` prefix for ChromaDB compatibility

## Performance Benchmarks

### Headline Result (v0.2.5)

> **Outcome learning: +40 pts improvement. Reranker: +10 pts. Learning dominates 4×. (p=0.005)**

See `benchmarks/comprehensive_test/` for full test suite and methodology.

---

### Comprehensive 4-Way Comparison (v0.2.5)

| Condition | Top-1 | MRR | nDCG@5 |
|-----------|-------|-----|--------|
| RAG Baseline | **10%** | 0.550 | 0.668 |
| Reranker Only | **20%** | 0.600 | 0.705 |
| Outcomes Only | **50%** | 0.750 | 0.815 |
| Full Roampal | **44%** | 0.720 | 0.793 |

**Improvement Breakdown:**
- Reranker contribution: +10 pts
- Outcomes contribution: +40 pts (4× more impactful)

**Statistical Significance:**
- Learning Curve (Cold→Mature): p=0.0051**
- Full vs RAG (MRR): p=0.0150*
- Full vs Reranker (MRR): p=0.0368*

### Learning Curve (v0.2.5)

| Maturity | Uses | Accuracy |
|----------|------|----------|
| Cold Start | 0 | **10%** |
| Early | 3 | **100%** |
| Mature | 20 | **100%** |

**+90 percentage points** improvement from cold start to learned state.

### Performance Metrics Summary

| Metric | Measured Performance | Status |
|--------|---------------------|--------|
| **4-Way Comparison** | 200 tests, RAG 10% → Roampal 60% | ✅ Verified |
| **Statistical Significance** | p=0.005 (learning curve) | ✅ Verified |
| **Learning Curve** | 10% → 100% (+90pp) | ✅ Verified |
| **Token Efficiency** | Outcome-only 6× more efficient than RAG | ✅ Verified |
| **Infrastructure** | 14 test suites, 100% pass rate | ✅ Verified |

**Why This Matters:**
The system learns that "what worked before" matters more than "what sounds related." Outcome learning (+40 pts) dominates cross-encoder reranking (+10 pts) by 4×.

> **All benchmarks reproducible** - See `benchmarks/comprehensive_test/` folder for complete test suite and methodology.

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
│    /api/agent/chat  /api/memory/*                │
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
│   │ • Automatic outcome detection   │            │
│   │ • Concept relationships graph   │            │
│   │ • Problem→Solution tracking     │            │ 
│   │ • Score-based promotion         │            │
│   │ • Adaptive learning             │            │
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
  - Working → History: score ≥0.7 AND uses ≥2
  - Triggers: Every 30 minutes (background task), Every 20 messages (auto-promotion), On conversation switch
  - History → Patterns: score ≥0.9 AND uses ≥3
- **Use Case**: Active problem-solving context

#### History Collection
- **Purpose**: Past conversations and interactions
- **Retention**: 30 days (high-value items preserved via `clear_old_history()`)
- **Promotion**: Successful patterns promoted to patterns collection
- **Use Case**: Learning from past interactions

#### Patterns Collection
- **Purpose**: Proven solutions and successful patterns
- **Retention**: Permanent
- **Source**: Promoted from history when consistently successful
- **Use Case**: Quick retrieval of known solutions

#### Memory Bank Collection (NEW - 2025-10-01)
- **Purpose**: Persistent context for both user AND LLM (identity, preferences, learned knowledge, shared projects)
- **Retention**: Permanent (never decays)
- **Capacity**: 1000 items maximum (prevents unbounded growth)
- **Ranking**: Results boosted by `importance × confidence` score
  - High-quality memories (importance=0.9, confidence=0.9 → quality=0.81) rank significantly higher
  - Low-quality memories (importance=0.3, confidence=0.4 → quality=0.12) rank lower
  - Quality score reduces semantic distance by up to 50% for maximum-quality items
  - **Content KG Entity Boost**: Documents with high-quality entities matching query get additional 50% boost (max 1.5× multiplier)
- **Management**:
  - LLM has full autonomy to store/update/archive
  - User has override via Settings UI (restore/delete)
  - Auto-archives old versions on updates (versioning without complexity)
- **Structure**:
  - Tags: Soft guidelines (identity, preference, project, context, goal, workflow)
  - Status: active | archived
  - Metadata: importance (0-1), confidence (0-1), mentioned_count
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
  - `GET /api/memory/knowledge-graph` - Get graph data (nodes/edges for visualization)
    - Returns `last_used` and `created_at` timestamps when available (v0.2.0)
    - Newly created concepts have `null` timestamps until first scored outcome
  - `GET /api/memory/knowledge-graph/concept/{id}/definition` - Get concept details with routing stats

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

3. **Cross-Encoder Quality Multiplier:**
   - For memory_bank only: `final_score = blended_score × (1 + quality)`
   - High quality (0.93) → 1.93x multiplier
   - Low quality (0.08) → 1.08x multiplier
   - Ensures quality advantage survives cross-encoder reranking

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

**KG Visualization Features (v0.2.0):**
- **Time-based filtering**: All Time | Today | This Week | This Session
  - **All Time**: Shows all concepts (including unused with `last_used = null`)
  - **Today/Week/Session**: Only shows concepts with `last_used` timestamp in range
- **Sort options**: Importance (hybrid score) | Recent (last used) | Oldest (creation time)
- **Dynamic header**: Updates to show active filters ("Top 20 most recent concepts from today")
- **Timestamps**: `created_at` and `last_used` are `null` for concepts created but never used in scored outcomes
- **Success rate display**: Shows actual success percentage in nodes (50% default for unused concepts)
- **Smooth resizing**: Debounced panel resize (300ms) prevents jank when dragging panel dividers
- **Empty state UX**: Filter controls always visible, contextual messages based on active filter
- **Concept modal**: Shows routing breakdown with collections searched and per-tier success rates
- Always shows top 20 concepts based on selected filters

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

**Method:**
- Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (BERT-based)
- Cross-encoder jointly encodes query + document pairs
- Provides finer-grained relevance scores than embeddings
- Blended score: **40% original + 60% cross-encoder** (trust cross-encoder more)

**Why cross-encoder vs bi-encoder:**
- Bi-encoder (what we use for embedding): Encodes query and doc separately, fast but less accurate
- Cross-encoder: Encodes query+doc together, slow but very accurate
- Solution: Use bi-encoder for first-stage retrieval (fast), cross-encoder for reranking top-30 (accurate)

**Implementation:**
- `unified_memory_system.py:194-200` - Initialize cross-encoder (optional)
- `unified_memory_system.py:591-659` - `_rerank_with_cross_encoder()` method
- `unified_memory_system.py:1308-1309` - Applied when results > limit × 2
- **Optional Dependency**: Requires `sentence-transformers` package
- **Graceful Fallback**: Uses original ranking if cross-encoder unavailable

#### Combined Performance (Estimated)

**Baseline (v0.2.0 - Dynamic Ranking + Dedup):**
- 7B model: 50% accuracy
- 32B model: 70% accuracy

**Enhanced (v0.2.1 - + Contextual + Hybrid + Reranking):**
- **7B model: 68% accuracy** (+36% relative improvement)
- **32B model: 87% accuracy** (+24% relative improvement)

**Why this matters for weak LLMs:**
- 7B makes terrible queries: "her approximate age" instead of "Gemma age"
- Contextual retrieval: Adds missing context to chunks
- Hybrid search: BM25 catches exact phrases even when embedding fails
- Cross-encoder: Filters false positives from bad embedding matches

#### Technical Details

**Dependencies:**
```bash
pip install rank-bm25 sentence-transformers nltk
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

**Benchmark Results (Search Quality Test):**
- Before: 75% acronym expansion accuracy
- After: 100% acronym expansion accuracy
- Overall search quality: 100% (6/6 metrics at 100%)

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
6. Cross-encoder rerank top-30
7. Return top-k
```

**Graceful Degradation:**
- If BM25 unavailable → Falls back to vector-only search
- If cross-encoder unavailable → Uses dynamic ranking only
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

**Score Adjustments:**
- ✅ `worked`: +0.2 (capped at 1.0)
- ❌ `failed`: -0.3 (minimum 0.0)
- ⚠️ `partial`: +0.05 (small boost)
- ❓ `unknown`: No change

**Uses Counter (Wilson Scoring):**
- `uses` is incremented on ALL outcomes (worked, failed, partial)
- This provides accurate denominator for Wilson score confidence intervals
- Partial outcomes count as 0.5 success for accurate Wilson calculation
- Example: 5 worked + 2 partial + 3 failed = 6.0 successes / 10 uses
- Wilson formula: `(successes + 1) / (uses + 2)` - more uses = higher confidence

**Technical Implementation:**
- [unified_memory_system.py:2296-2424](modules/memory/unified_memory_system.py#L2296-L2424) - `record_outcome()` method
- Score updates logged: `Score update [working]: 0.50 → 0.70 (outcome=worked, delta=+0.20)`

**Example Evolution:**
```
Creation: score=0.5 (neutral baseline)
After 1st use (worked): score=0.7 (+0.2)
After 2nd use (worked): score=0.9 (+0.2)
After 3rd use (failed): score=0.6 (-0.3)
```

### Learning Mechanisms

#### Organic Memory Recall (NEW - 2025-09-30)
**The system now proactively surfaces relevant context before searching memory.**

Instead of just searching for what you asked, Roampal analyzes:
1. **Pattern Recognition**: "You tried this 3 times before, here's what worked"
2. **Failure Awareness**: "Similar approach failed last time due to: [reason]"
3. **Topic Continuity**: "Continuing discussion about: docker, deployment"
4. **Proactive Insights**: "For authentication, patterns collection is 85% effective"
5. **Repetition Detection**: "You mentioned something similar 5 minutes ago"

**Technical Implementation:**
- `unified_memory_system.py:1736` - `analyze_conversation_context()` method
- `agent_chat.py:611-663` - Organic recall before LLM response (PRODUCTION - 2025-01-14)
- `main.py:791-1105` - MCP tool `get_context_insights()` (PRODUCTION - 2025-01-14)

**How It Works:**
```python
# Before (Query-Based Only)
User: "Fix Docker issue"
→ Search memory for "Fix Docker issue"
→ Return generic Docker memories

# Now (Context-Aware)
User: "Fix Docker issue"
→ Analyze conversation context FIRST:
  • Checks knowledge_graph["problem_categories"] for concept matches
  • Finds past solutions with outcomes (worked/failed)
  • Detects topic continuity from recent messages
  • Identifies similar questions asked recently
→ Injects organic insights into prompt:
  "📋 Past Experience: You tried this approach 2 times (100% success rate)"
  "⚠️ Past Failures: Similar approach failed due to missing .env file"
→ Then searches memory normally
→ LLM sees full context and provides informed response
```

**Memory Context Injection Format:**
```
═══ CONTEXTUAL MEMORY ═══

📋 Past Experience:
  • Based on 3 past use(s), this approach had a 85% success rate
    → User: Fix Docker permissions...

⚠️ Past Failures to Avoid:
  • Note: Similar approach failed before due to: missing sudo

💡 Recommendations:
  • For 'docker', check patterns collection (historically 85% effective)

🔗 Continuing discussion about: docker, deployment

Use this context to provide more informed, personalized responses.
```

**Impact:** The system now UNDERSTANDS what the data means, not just that it exists. It's the difference between:
- ❌ A database that has your info but never uses it
- ✅ An assistant that says "Oh yeah, you tried that before and here's what happened"

**MCP Integration (2025-01-14):**
External LLMs (via MCP) can explicitly request context insights using `get_context_insights(query)` tool:

```python
# Claude Desktop usage:
get_context_insights("docker permissions issue")

# Returns:
═══ CONTEXTUAL INSIGHTS ═══

📋 Past Experience:
  • Based on 3 past uses, adding user to docker group had 100% success rate
    Collection: patterns, Score: 0.95, Uses: 3
    → User: How to fix Docker permissions...

💡 Search Recommendations:
  • For 'docker', check patterns collection (historically 85% effective)

🔗 Continuing discussion about: docker, deployment
```

**Key Difference:**
- **Production (agent_chat.py)**: Automatic organic recall before every LLM response
- **MCP (main.py)**: Explicit tool call - external LLM decides when to check for insights

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
- Exchanges with `score >= 0.7` and `uses >= 2` promote from working → history as complete Q&A pairs
- No conversation filter skipping during promotion (unified_memory_system.py:1359, 1599)
- Full context maintained through entire decay lifecycle: working → history → patterns

**Fast-Track Promotion (REMOVED in v0.2.3):**
~~Exceptional memories can skip history collection and promote directly from working to patterns.~~

**Removed in v0.2.3:** All memories now must go through history first to prove themselves over time. The fast-track bypass was too aggressive - 3 consecutive successes in one session doesn't prove long-term value. Memories need to "season" in history before reaching patterns.

#### Unified Outcome-Based Memory Scoring (Updated 2025-10-06)

**Simple, clean system: LLM detects outcomes → scores the previous exchange → mechanical promotion.**

The system uses LLM intelligence for outcome detection only. All scoring, promotion, and deletion decisions follow fixed, predictable rules based on accumulated outcomes.

**LLM Service Injection (Internal System Only):** The LLM client is injected into the memory system after initialization via `memory.set_llm_service(llm_client)` ([main.py:295-298](../main.py#L295-L298)). This allows the `OutcomeDetector` to access the LLM for analyzing conversation outcomes. **Note**: MCP system does not use automatic detection - external LLMs provide outcomes **explicitly** via the `outcome` parameter (optional, defaults to "unknown" if not provided).

**OutcomeDetector API (v0.2.12):** [outcome_detector.py](../ui-implementation/src-tauri/backend/modules/advanced/outcome_detector.py)

```python
async def analyze(
    self,
    conversation: List[Dict[str, Any]],
    surfaced_memories: Optional[Dict[int, str]] = None,  # v0.2.12 Fix #5
    llm_marks: Optional[Dict[int, str]] = None           # v0.2.12 Fix #7
) -> Dict[str, Any]:
    """
    Returns:
        {
            "outcome": "worked|failed|partial|unknown",
            "confidence": 0.0-1.0,
            "indicators": ["explicit_thanks", ...],
            "reasoning": "User said thanks",
            "used_positions": [1, 3],           # v0.2.12 Fix #5: inferred usage
            "upvote": [1],                      # v0.2.12 Fix #7: positions to upvote
            "downvote": [2]                     # v0.2.12 Fix #7: positions to downvote
        }
    """
```

**Parameters:**
- `conversation`: Recent turns for outcome analysis
- `surfaced_memories` (v0.2.12): `{position: content}` - memories shown to main LLM, used for selective scoring
- `llm_marks` (v0.2.12): `{position: emoji}` - main LLM's attribution marks (👍👎➖)

**Returns:**
- `used_positions`: Inferred from response analysis (Fix #5 fallback)
- `upvote`/`downvote`: Calculated from llm_marks + outcome (Fix #7)

**The Clean Flow:**

1. **User:** "What's an IRA?"
2. **Assistant searches memory** (if relevant) → returns doc_id_X, doc_id_Y from patterns/history
   - System caches: [doc_id_X, doc_id_Y] for outcome scoring
3. **Assistant responds:** "An IRA is a retirement account..."
   - Stores exchange in memory: `"User: What's an IRA?\nAssistant: An IRA is..."`
   - doc_id: "working_abc123"
   - Initial score: 0.5

4. **User provides feedback:** "that didn't help"
   - **BEFORE** generating response, system:
     - Reads session file to get previous assistant message
     - Gets doc_id: "working_abc123"
     - Analyzes: [previous assistant answer, current user feedback]
     - LLM detects: outcome = "failed"
     - **Updates previous exchange score**: 0.5 - 0.3 = **0.2**
     - **Updates cached memories** (doc_id_X, doc_id_Y) with same outcome ("failed")
     - Clears cache

5. **Assistant responds:** "Let me explain better..."
   - Stores NEW exchange with doc_id: "working_xyz456", score: 0.5

**Key Principle:** The outcome detection scores BOTH:
- The PREVIOUS exchange that the user is reacting to
- Any retrieved memories (working/history/patterns) that were used in that response

**v0.2.12 Enhancements - Selective & Causal Scoring:**

The internal system now has parity with MCP's selective scoring, plus causal attribution:

1. **Selective Scoring (Fix #5):** OutcomeDetector identifies which memories were actually USED in the response, not just surfaced. Only used memories get scored.
   - Cache structure: `{position_map: {1: doc_id, ...}, content_map: {1: "content preview", ...}}`
   - OutcomeDetector returns `used_positions: [1, 3]`
   - Only those positions get the outcome score; unused memories stay neutral

2. **Causal Attribution (Fix #7):** Main LLM marks memories as helpful/unhelpful at response time:
   - Main LLM adds hidden annotation: `<!-- MEM: 1👍 2👎 3➖ -->`
   - Parsed and stripped before showing response to user
   - Passed to OutcomeDetector which combines marks with outcome detection
   - Scoring matrix:
     ```
                     | YES (worked) | KINDA (partial) | NO (failed) |
     ----------------|--------------|-----------------|-------------|
     👍 (helpful)    | upvote       | slight_up       | neutral     |
     👎 (unhelpful)  | neutral      | slight_down     | downvote    |
     ➖ (no_impact)  | neutral      | neutral         | neutral     |
     ```
   - **Key insight:** A positive exchange can still downvote bad memories if LLM marked them 👎

3. **Fallback behavior:** If main LLM doesn't include annotation, falls back to Fix #5 (infer usage) → Fix #4 (score all)

**Scoring Rules:**
- `worked` → +0.2 to score
- `failed` → -0.3 to score
- `partial` → +0.05 to score
- `unknown` → no change

**Automatic Promotion/Deletion** (threshold-based):
- score ≥ 0.7 AND uses ≥ 2 → working → history
- score ≥ 0.9 AND uses ≥ 3 → history → patterns
- score < 0.2 → deleted (or score < 0.1 for items < 7 days old)

### MCP (External LLM) Memory Scoring Flow

**MCP Integration** uses **semantic learning storage** and **external LLM outcome assessment**:

**Turn 1:**
1. External LLM calls `search_memory("IRA accounts")` → returns doc_id_A (working), doc_id_B (history)
   - System caches: [doc_id_A, doc_id_B] for this session
2. External LLM responds using those memories: "An IRA is a retirement account..."
3. External LLM calls `record_response(key_takeaway="User asked about IRA accounts. I explained they are retirement accounts.", outcome="unknown")`
   - **Store CURRENT learning** (semantic summary) → doc_id_1, score: 0.5
   - **Score PREVIOUS learning** → none exists (first turn)
   - **Score cached memories** → skipped (outcome="unknown")
   - **Write CURRENT to session file** with doc_id_1
   - **Clear cache**

**Turn 2:**
1. External LLM calls `search_memory("more details")` → returns doc_id_C (patterns)
   - System caches: [doc_id_C]
2. External LLM analyzes user message "that didn't help" → determines outcome="failed"
3. External LLM responds: "Here are more details..."
4. External LLM calls `record_response(key_takeaway="User said explanation didn't help. I provided more detailed information.", outcome="failed")`
   - **Store CURRENT learning** (semantic summary) → doc_id_2, initial_score: 0.2 (failed)
   - **Score previously SEARCHED memories** → [doc_id_C] using same outcome ("failed")
     - doc_id_C score updated (was used in failed response, gets downvoted)
   - **Write CURRENT to session file** with doc_id_2
   - **Clear cache**
   - Note: doc_id_1 from Turn 1 is NOT re-scored (MCP scores CURRENT learning at creation time, not retroactively)

**Key Differences from Internal System:**

| Aspect | Internal System | MCP System |
|--------|----------------|------------|
| **Outcome Detection** | Automatic - internal LLM analyzes conversation | Manual - external LLM provides outcome via parameter |
| **Who Judges** | Roampal's internal LLM | External LLM (Claude, Cursor, etc.) |
| **Storage Format** | Verbatim transcripts | Semantic summaries (key_takeaway) |
| **LLM Calls** | Extra call for outcome detection | No extra call - external LLM provides outcome |
| **Trust Model** | System decides outcome | System trusts external LLM completely |
| **Memory Tracking** | Session-scoped cache | Session-scoped cache (same) |
| **What Gets Scored** | Previous exchange + retrieved memories | **CURRENT learning + retrieved memories** (both scored immediately) |

**Why Different:**
- **Semantic storage:** LLMs excel at summarization, avoids verbatim copy errors, better for search
- **Score CURRENT not PREVIOUS:** MCP scores the learning being recorded immediately, allowing optional tool calling (LLM only calls when clear outcomes)
- **Explicit outcome scoring:** External LLM provides outcome explicitly based on user feedback (more reliable than auto-detection for external contexts)
- **No automatic detection:** MCP system uses explicit `outcome` parameter - defaults to "unknown" if not provided, no internal LLM verification

**Implementation Details:**
- `search_memory` tool:
  - Returns **full content** for all results (no truncation)
  - Caches ALL doc_ids for Action KG tracking (v0.2.6 - unified with internal system, includes books/memory_bank)
  - **Caches search query** for KG routing updates (stores in `last_search_query_cache[session_id]`)
- `record_response` tool:
  - **Parameters:** `key_takeaway` (semantic summary, required) + `outcome` (explicit scoring, optional, defaults to "unknown")
  - **Storage:** Stores semantic summary to ChromaDB with **initial score calculated from outcome** (worked=0.7, failed=0.2, partial=0.55, unknown=0.5)
  - **Scores CURRENT learning:** Unlike internal system (scores previous), MCP scores the learning being recorded immediately
  - **Scores retrieved memories:** Also scores all cached memories from the last search with the same outcome (upvote helpful memories, downvote bad advice)
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

**Action-Effectiveness KG Tracking (v0.2.1, updated v0.2.6):**
- ✅ **MCP System**: All 4 tools tracked (`search_memory`, `create_memory`, `update_memory`, `archive_memory`)
- ✅ **Internal System**: All tools tracked (v0.2.6 - added action caching and scoring in agent_chat.py)
- Both systems build identical `knowledge_graph["context_action_effectiveness"]` structure
- Key format: `"{context}|{action}|{collection}"` (e.g., `"coding_help|search_memory|patterns"`)
- Tracks: successes, failures, partials, success_rate, total_uses, examples
- Updates on every `record_response` with explicit outcome (MCP) or automatic detection (internal)

**Implementation:** Single outcome detection in streaming endpoint ([agent_chat.py:1113-1154](../app/routers/agent_chat.py#L1113-L1154))

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

#### Outcome Detection (Updated 2025-10-06)
**Philosophy: LLM-Only Detection with Strict Satisfaction Criteria**

The system uses **LLM-only** outcome detection that distinguishes between enthusiastic satisfaction and lukewarm responses.

**Key Design (2025-10-06):**
- **LLM-only** - No heuristic fallbacks
- **Degree-based** - Distinguishes enthusiastic vs lukewarm positive feedback
- **Critical insight** - Follow-up questions ≠ success (often indicate confusion/criticism)
- **Structured JSON output** - Returns `{outcome, confidence, indicators, reasoning}` for automated score updates

**Outcome Categories:**

**"worked"** - ENTHUSIASTIC satisfaction or clear success
- "thanks!", "perfect!", "awesome!", "that worked!"
- User moves to NEW topic (indicates previous was resolved)
- NOT worked: "yea pretty good", "okay", follow-up questions

**"failed"** - Dissatisfaction, criticism, or confusion
- "no", "nah", "wrong", "didn't work"
- Criticism: "why are you...", "stop doing..."
- Repeated questions about SAME issue (solution didn't work)
- Follow-up questions expressing confusion

**"partial"** - Lukewarm (positive but not enthusiastic)
- "yea pretty good", "okay", "sure", "I guess", "kinda"
- Helped somewhat but incomplete

**"unknown"** - No clear signal yet
- No user response after answer
- Pure neutral: "hm", "noted"

**CRITICAL Principle:** Follow-up questions are NOT success signals. User continuing conversation ≠ satisfaction.

**Prompt Structure:**
```
"worked": ENTHUSIASTIC satisfaction or clear success
  • [explicit examples]
  • NOT worked: "yea pretty good", "okay", follow-up questions

"failed": Dissatisfaction, criticism, or confusion
  • [explicit examples including criticism patterns]

"partial": Lukewarm (positive but not enthusiastic)
  • [explicit lukewarm examples]

CRITICAL: Follow-up questions are NOT success signals.

Return JSON: {outcome, confidence, indicators, reasoning}
```

**Issues Fixed:**
- **2025-10-04**: Heuristic regex matched "unhelpful" → false positive
- **2025-10-05**: Analyzed outcomes before user feedback → false positives
- **2025-10-06**: Was too lenient - any positive word → "worked"
- **2025-10-06**: Now requires ENTHUSIASTIC satisfaction, not just polarity
**Additional Safety Improvements (2025-10-04):**
1. **Book & Memory Bank Safeguards** ([outcome_service.py:90-104](modules/memory/outcome_service.py#L90-L104))
   - **KG routing updates FIRST** - Books/memory_bank searches update Routing KG patterns (learning which queries → those collections)
   - **Then safeguard blocks outcome-based scoring** - Books and memory_bank are never outcome-scored
   - **Why:** Routing KG learns from all collections, but books/memory_bank shouldn't be promoted/demoted based on conversation outcomes
   ```python
   # UPDATE KG ROUTING FIRST - even for books/memory_bank
   if problem_text:
       await self.kg_service.update_kg_routing(problem_text, collection_name, outcome)

   # SAFEGUARD: Books are reference material, not scorable memories
   if doc_id.startswith("books_"):
       logger.info("[KG] Learned routing pattern for books, but skipping score update")
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
3. **Knowledge Graph Cleanup**: Remove dead doc_id references from KG
   - Cleans `problem_categories` and `problem_solutions` mappings
   - Removes routing patterns with 0 total uses
   - Prevents KG bloat from deleted documents
   - **Visualization**: Node size reflects both usage frequency AND success rate
     - Formula: `√connections × √(strength + 0.1)`
     - Prevents low-quality high-usage patterns from appearing prominent
     - Example: 100 uses at 10% success → smaller than 10 uses at 90% success
     - Display: Top 20 concepts by hybrid score (fits on screen, prevents UI overflow)
     - Sorting: Highest quality × usage patterns shown first

**Startup Tasks** (runs on system start, non-blocking):
1. Clean stale working memory from previous session
2. Clean history older than 30 days
3. Clean dead KG references

**Automatic Cleanup on Deletion** (v0.2.0 - Prevents Stale Data):
- **Content KG cleanup** - Automatically called on ALL memory_bank deletions:
  - `archive_memory_bank()` → `content_graph.remove_entity_mention(doc_id)` [unified_memory_system.py:2703](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L2703)
  - `user_delete_memory()` → `content_graph.remove_entity_mention(doc_id)` [unified_memory_system.py:2833](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L2833)
  - `delete_by_conversation()` → batch cleanup for all deleted memory_bank items [unified_memory_system.py:877-889](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L877-L889)
  - **Note**: Books are NOT indexed in Content KG (only memory_bank), so book deletion doesn't need Content KG cleanup
- **Action KG cleanup** (v0.2.6) - Called on book deletions:
  - `delete_book()` → `cleanup_action_kg_for_doc_ids(chunk_ids)` [book_upload_api.py:705-714](../ui-implementation/src-tauri/backend/backend/api/book_upload_api.py#L705)
  - Removes stale `doc_id` references from `context_action_effectiveness` examples
- **Routing KG cleanup** - Called on bulk deletions:
  - `delete_by_conversation()` → `_cleanup_kg_dead_references()` [unified_memory_system.py:892](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L892)
- **Fallback protection** - Cold-start has fallback to vector search when Content KG has stale data [unified_memory_system.py:964-976](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L964-L976)

**Auto-Promotion** (fires every 20 messages, non-blocking):
- Triggered after 20 messages in working memory
- Fire-and-forget (doesn't block chat responses)
- Moved to background task in 2025-10-01 update

#### Dual Knowledge Graph System (v0.2.0)

**Learning-Based Routing KG** (Implemented 2025-11-05)

The system uses a learning-based Knowledge Graph router that intelligently selects memory tiers based on past search success patterns. **Zero hardcoded keywords** - learns entirely from usage.

**Key Principles:**
1. **Learn from outcomes** - Tracks which tiers (books, working, history, patterns, memory_bank) successfully answer which types of queries
2. **Static stopword filtering** - Extracts n-grams (unigrams, bigrams, trigrams) with ~40 hardcoded stopwords for concept extraction
3. **Confidence-based routing** - Evolves from exploration (all tiers) → focused (1-2 best tiers)
4. **Safety fallback** - Expands to all tiers if <3 results found
5. **LLM override** - LLM can still explicitly specify collections to bypass routing

**How It Works:**

*Phase 1: Cold Start (First 3-10 queries)*
```python
# No patterns learned yet → Search all tiers
query = "show me books about investing"
concepts = ["show", "me", "books", "about", "investing", "show_me", "me_books", ...]
total_score = 0  # No patterns exist
→ Routes to: ["working", "patterns", "history", "books", "memory_bank"]
```

*Phase 2: Learning (After 10+ queries)*
```python
# System has learned patterns
query = "show me books about investing"
concepts = ["books", "investing", "books_about", ...]

# Calculate tier scores from learned patterns:
routing_patterns["books"] = {
  "collections_used": {
    "books": {"successes": 8, "failures": 2, "total": 10},
    "working": {"successes": 1, "failures": 3, "total": 4}
  }
}

# Scoring:
books_tier: (8/10) * min(10/10, 1.0) = 0.8 * 1.0 = 0.8
working_tier: (1/4) * min(4/10, 1.0) = 0.25 * 0.4 = 0.1
total_score = 0.9

→ Routes to: ["books", "working"]  # Top 2 tiers
```

*Phase 3: Confident (After 20+ queries)*
```python
# High confidence in routing
total_score = 2.4  # Multiple concepts with strong patterns
→ Routes to: ["books"]  # Single best tier
```

**N-gram Concept Extraction** ([unified_memory_system.py:1140-1193](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L1140-L1193)):
- Unigrams: `["books", "investing", "show", "me"]`
- Bigrams: `["show_me", "me_books", "books_about", "about_investing"]`
- Trigrams: `["show_me_books", "me_books_about", "books_about_investing"]`
- Technical patterns: `CamelCase`, `snake_case`, `ErrorTypes`
- **Static stopword filtering** - Uses ~40 hardcoded English stopwords ("the", "a", "is", "are", "was", "were", etc.)
- Additional stopwords available in [config/settings.py:408-423](../ui-implementation/src-tauri/backend/config/settings.py#L408-L423) (~120 words for keyword search fallback)

**Confidence Formula** ([unified_memory_system.py:896-908](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L896-L908)):
```python
# Success rate calculation (v0.2.0 fix)
total_with_feedback = successes + failures
if total_with_feedback > 0:
    success_rate = successes / total_with_feedback  # Actual percentage
else:
    success_rate = 0.5  # 50% neutral baseline (no feedback yet)

confidence = min(total_uses / 10.0, 1.0)  # Grows with usage
tier_score = success_rate * confidence
```

**Note:** "partial" outcomes are tracked but excluded from success_rate calculation. Concepts without explicit "worked"/"failed" feedback default to 50% (neutral/unknown).

**Routing Thresholds** ([unified_memory_system.py:909-928](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L909-L928)):
- `total_score < 0.5`: **Exploration** - Search all 5 tiers
- `0.5 ≤ total_score < 2.0`: **Medium confidence** - Select top 2-3 tiers
- `total_score ≥ 2.0`: **High confidence** - Focus on top 1-2 tiers

**Fallback Safety Net** ([unified_memory_system.py:664-690](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L664-L690)):
```python
if len(results) < 3 and not_searching_all_tiers:
    # Expand to remaining tiers
    # Prevents over-aggressive routing from missing results
```

**Outcome Learning** ([unified_memory_system.py:1207-1321](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L1207-L1321)):
- Tracks `successes`, `failures`, and `total` per tier per concept
- Updates on `record_outcome(worked/failed/partial)`
- Partial outcomes tracked but don't affect success rate
- Builds concept relationships for knowledge graph

---

#### Knowledge Graphs are the Intelligence Layer

**Profound Realization**: The memory collections (books, working, history, patterns, memory_bank) are just **storage**. The real intelligence lives in the **Knowledge Graphs**.

**Three Knowledge Graphs Working Together:**

1. **Routing KG** (`routing_patterns`, `success_rates`) - *Which collection has the answer?*
   - Maps concepts → best collection
   - Learns from search outcomes
   - Enables intelligent tier selection

2. **Content KG** (`content_graph`) - *How are entities related?*
   - Entity relationships from memory content
   - Green/purple nodes in visualization
   - Semantic connections between concepts

3. **Action-Effectiveness KG** (`context_action_effectiveness`) - *What tool should I use in this situation?*
   - Maps (context, action, collection) → success rate
   - Learns contextually appropriate behaviors
   - Enables self-correction and alignment
   - v0.2.6: Stores doc_ids in examples for document-level effectiveness tracking

**The Memory Paradox:**

```
Without KG:
  5 collections × 1000 documents = 5000 items
  Search: "authentication error"
  → Must search ALL 5000 items
  → Slow, unfocused, no learning

With Routing KG:
  Search: "authentication error"
  → Concepts: ["authentication", "error", "authentication_error"]
  → KG says: "patterns" tier has 95% success rate for these concepts
  → Search only 200 items in patterns tier
  → Fast, focused, learns from outcomes

With Action-Effectiveness KG:
  Context: memory_test (LLM-classified from conversation)
  → KG says: "search_memory has 85% success in memory_test context"
  → KG says: "create_memory has 5% success in memory_test context"
  → System learns to avoid hallucination patterns
  → Enables contextual alignment
```

**The Intelligence Hierarchy:**

```
Level 0: Raw Storage (ChromaDB)
  ↓ Just vectors and metadata

Level 1: Memory Collections (5 tiers)
  ↓ Organized by lifecycle (working → history → patterns)

Level 2: Routing KG (which tier?)
  ↓ Learns which collection answers which query

Level 3: Content KG (what's related?)
  ↓ Understands entity relationships

Level 4: Action-Effectiveness KG (what should I do?)
  ↓ Learns contextually appropriate behavior
  ↓ Enables self-correction and alignment

Level 5: Self-Improving Prompts (FUTURE)
  ↓ Auto-generates corrective instructions from learned patterns
```

**Memory is storage. Knowledge Graphs are intelligence.**

Without the KGs:
- 5000 documents to search → Slow
- No learning from outcomes → Repeats mistakes
- No context awareness → Inappropriate tool use
- LLM must re-learn patterns every conversation → Inefficient

With the KGs:
- ~200 documents to search → 25x faster
- Learns from every outcome → Gets smarter
- Detects context and chooses appropriate actions → Aligned behavior
- Accumulated intelligence persists → Continuous improvement

**The KGs are where the system becomes intelligent.**

---

## 🔮 FUTURE FEATURE: Adaptive Stopword Learning

**Status:** Planned enhancement - not currently implemented

The system currently uses static stopword lists. A future enhancement would add dynamic learning to automatically classify words as noise vs. semantic concepts based on usage patterns.

**Proposed Implementation:**

**4-Phase Learning Process:**

*Phase 1: Bootstrap (Queries 1-10)*
- Start with minimal hardcoded stopwords (core 29: "a", "an", "the", "is", "are", etc.)
- Track ALL words in `knowledge_graph["word_statistics"]`
- Record: frequency, tier distribution, outcome correlation, timestamps

*Phase 2: Statistical Collection (Queries 10-50)*
```python
# Track full statistics per word
"investing": {
    "total_queries": 45,
    "tier_distribution": {"books": 40, "working": 5, ...},
    "outcome_correlation": {"successes": 35, "failures": 10},
    "discrimination_score": 0.85,  # Shannon entropy-based
    "is_stopword": False
}
```

*Phase 3: Automatic Classification (Every 50 queries)*
- Calculate Shannon entropy discrimination scores across tiers
- Auto-classify stopwords: high frequency (>10%) + low discrimination (<0.3) + neutral outcomes (45-55%)
- Prune rare words: frequency <3 AND not seen in 100 queries

*Phase 4: Dynamic Re-evaluation (Continuous)*
- Words can transition between useful ↔ stopword based on evolving patterns
- Domain-specific adaptation (e.g., "kubernetes" = meaningful in tech context, stopword elsewhere)

**Benefits:**
- Auto-adapts to user's vocabulary and domain
- Learns project-specific jargon and acronyms
- Reduces KG bloat without manual tuning
- Bilingual/multilingual support potential

**Storage Impact:**
- ~50-80 bytes per word × 5,000 words = ~300-400KB
- Saves in existing `knowledge_graph.json`

**Implementation Note:** Would not affect `memory_bank` (user bookmarks are protected from auto-modification)

---

**Content Knowledge Graph** (✅ FULLY INTEGRATED - v0.2.0)

**⚠️ CRITICAL FEATURE - DO NOT DISABLE OR REMOVE ⚠️**
This provides entity relationship mapping for the user's personal knowledge graph and enables green/purple node visualization in the KG UI.

Complements the routing KG with a **content-based entity graph** that indexes relationships from memory_bank content.

**Implementation Status:**
- ✅ Core class: [content_graph.py](../ui-implementation/src-tauri/backend/modules/memory/content_graph.py)
- ✅ Integration: [unified_memory_system.py:20,149-154,160-172](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py)
- ✅ Entity extraction: Automatic on memory_bank store/update/archive [lines 2456-2466,2528-2537,2569-2575](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py)
- ✅ Triple KG merge (v0.2.1): [get_kg_entities():4147-4310](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L4147-L4310)
- ✅ Persistence: Saved atomically with routing KG [_save_kg_sync():212-238](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L212-L238)
- ✅ Visualization: Blue (routing), Green (content), Purple (both), Orange (action) nodes in KG UI

**Dual Graph Architecture:**
- **Routing KG**: Learns which collections to search based on query patterns
  - Data source: User search queries + outcome feedback
  - Purpose: Optimize search routing (blue nodes)
  - Storage: `knowledge_graph.json`
  - Implementation: [unified_memory_system.py:146-147](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L146-L147)
- **Content KG**: Builds user's personal knowledge graph from stored content
  - Data source: memory_bank text + entity extraction
  - Purpose: Map entity relationships - who you are, what you do (green nodes)
  - Storage: `content_graph.json`
  - Implementation: [content_graph.py](../ui-implementation/src-tauri/backend/modules/memory/content_graph.py), [unified_memory_system.py:149-154](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L149-L154)

**Key Difference:**
- **Routing KG**: "benjamin_graham" query → route to books collection (routing decision)
- **Content KG**: "User prefers Docker for development" → creates docker ↔ development relationship (knowledge representation)

**Features:**
- Entity extraction from memory_bank content
- Co-occurrence based relationship strength
- Document tracking per entity
- BFS path finding between entities
- Automatic relationship updates on new content
- Metadata tracking (first_seen, last_seen, mentions)
- **Automatic cleanup on memory deletion** (v0.2.0 - prevents stale data)
- **Quality-based entity ranking** (v0.2.1 - prioritizes authoritative entities)

**Quality-Based Entity Ranking (v0.2.1):**

Entities now track quality scores derived from LLM-provided importance × confidence ratings:
- **Scoring**: `avg_quality = sum(importance × confidence) / mentions`
- **Sorting**: Entities ranked by `avg_quality` (descending) instead of `mentions`
- **Impact**: Authoritative facts prioritized over frequently-mentioned trivia
- **Example**: "User is senior backend engineer at TechCorp" (importance=0.9, confidence=0.95, quality=0.855) ranks higher than "maybe user likes TypeScript" mentioned 10 times (importance=0.3, confidence=0.5, quality=0.15)

**Content KG Search Enhancement (v0.2.1):**

memory_bank searches receive entity quality boosting based on Content KG:
- **When**: ONLY applied when searching memory_bank collection specifically
- **How**: Documents containing high-quality entities matching query concepts receive score boost
- **Calculation**: For each matching entity, `boost += entity.avg_quality × 0.2` (capped at 50% total boost)
- **Example**: Query "backend engineer techcorp" matches document with 3 high-quality entities (avg_quality=0.8 each) → boost = (0.8×3) × 0.2 = 48%
- **Other collections**: working, history, patterns, books use existing ranking (unchanged)

**Benefits:**
- Enables real entity connections in MCP integration
- Supports `get_kg_path("logan", "everbright")` to find relationships
- Creates "turbo user profile graph" - visual representation of user's knowledge
- Foundation for NER, fact extraction, semantic network features

**Full Implementation Details:** See [docs/KG_UPGRADE.md](KG_UPGRADE.md) Appendix C

**UI Visualization (v0.2.1: Quad-color system):**
- 🔵 Blue nodes = routing patterns (query-based, what you search for)
- 🟢 Green nodes = memory entities (content-based, who you are)
- 🟣 Purple nodes = both (intersection of queries and content)
- 🟠 Orange nodes = action effectiveness patterns (context|action|collection success rates)

**MCP Integration** ([main.py:753-1048](../ui-implementation/src-tauri/backend/main.py#L753-L1048)):
- Fixed: `collections=["all"]` bug - now passes `None` to trigger hybrid KG routing
- **Hybrid Routing** (v0.2.0): LLM can override OR use KG's learned patterns
- External LLMs (Claude Desktop) benefit from automatic intelligent routing
- **External LLM Outcome Judgment** (v0.2.0 - [main.py:960-1048](../ui-implementation/src-tauri/backend/main.py#L960-L1048)):
  - `record_response` tool requires TWO parameters: `key_takeaway` (required) + `outcome` (required)
  - **Key Principle**: External LLM judges its own previous response based on user feedback
  - When user provides feedback, external LLM passes `outcome: "worked"/"failed"/"partial"/"unknown"`
  - Server scores the **PREVIOUS** learning (not the current one being recorded)
  - Example: `record_response({key_takeaway: "User thanked me for explaining IRAs. I provided retirement account details.", outcome: "worked"})`
    - Scores the previous learning (from last turn)
    - Stores current learning summary to working memory
- **Why External LLM Judges**: Each AI (internal or external) evaluates its own conversation quality for consistency
- **Outcome Flow** (v0.2.0 - [main.py:968-1009](../ui-implementation/src-tauri/backend/main.py#L968-L1009)):
  1. Receive `key_takeaway` and `outcome` from external LLM
  2. If `outcome` is "worked"/"failed"/"partial", find previous learning's doc_id
  3. Score that previous learning via `record_outcome(prev_doc_id, outcome)`
  4. Store current `key_takeaway` to working memory with cached search query in metadata
  5. Update KG routing patterns with success/failure
  6. Check promotion thresholds (working → history → patterns)
- **Action-Effectiveness KG for MCP** (v0.2.1 - [main.py:100-1385](../ui-implementation/src-tauri/backend/main.py#L100-L1385)):
  - **Context Detection**: On every tool call, system detects conversation context via LLM classification
  - **Conversation Boundary Detection** (v0.2.1 - NEW):
    - **Problem**: MCP protocol doesn't provide conversation IDs (all conversations in Claude Desktop share same session ID)
    - **Solution**: Auto-detect conversation boundaries using 2 signals:
      1. **Time Gap**: 10+ minutes since last tool call → clear cache (likely new conversation)
      2. **Context Shift**: Topic changes (e.g., "coding_help" → "fitness_tracking") → clear cache
    - **Implementation**: `_should_clear_action_cache()` and `_cache_action_with_boundary_check()` ([main.py:100-161](../ui-implementation/src-tauri/backend/main.py#L100-L161))
    - **Benefit**: Prevents actions from Conversation A being scored with outcomes from Conversation B
    - **Limitation**: Not perfect (fast topic switches may miss boundary), but catches 90%+ of cases
  - **Action Tracking**: All 4 MCP tools cache `ActionOutcome` objects during execution:
    - `search_memory` → tracks which collections were searched
    - `add_to_memory_bank` (create_memory) → tracks memory creation
    - `update_memory` → tracks memory updates
    - `archive_memory` → tracks memory archival
  - **Outcome Scoring**: When `record_response` provides outcome, system:
    - Updates all cached actions with outcome ("worked"/"failed"/"partial")
    - Calls `record_action_outcome(action)` for each
    - Updates `knowledge_graph["context_action_effectiveness"]` with stats
    - Example: `"coding_help|search_memory|patterns" → 92% success (45 uses)`
  - **HUD Integration**: External LLMs see stats via `get_context_insights`:
    ```
    📊 Tool Usage Stats (FYI - you decide what to use):
      • search_memory() on patterns: 92% success (45 uses)
      • create_memory() on working: 12% success (8 uses)
    ```
  - **Benefits**: System learns which tools work in which contexts, enables self-correction and alignment

#### Document-Level Insights (v0.2.6)

memory_bank and books are static collections without outcome scores. v0.2.6 extracts doc effectiveness from Action KG examples:

**Implementation:** `unified_memory_system.py:get_doc_effectiveness()`

```python
def get_doc_effectiveness(self, doc_id: str) -> Optional[Dict]:
    """Aggregate success rate for a doc from Action KG examples."""
    # Scans context_action_effectiveness["examples"] for this doc_id
    # Returns: {"success_rate": 0.83, "total_uses": 5, "successes": 4, "failures": 1}
```

**Usage in Search:** Optional boost for memory_bank/books based on doc effectiveness:
```python
if collection in ["memory_bank", "books"]:
    doc_stats = self.get_doc_effectiveness(doc_id)
    if doc_stats and doc_stats["total_uses"] >= 3:
        effectiveness_boost = doc_stats["success_rate"] * 0.15  # max 15% boost
```

**Key Insight:** No 4th KG needed - Action KG already stores doc_ids in examples.

#### Directive Insights (v0.2.6)

`get_context_insights` output is now **actionable**, not just retrospective:

**Before (v0.2.5):**
```
📊 Action Outcome Stats:
  • search_memory() on books: 90% success (79 uses)
```

**After (v0.2.6):**
```
═══ KNOWN CONTEXT ═══
[User Profile]
- Logan, crypto researcher, prefers Docker Compose

[Recommended]
- search_memory(collections=["patterns"]) - 3 patterns found

═══ TO COMPLETE THIS INTERACTION ═══
After responding → record_response(key_takeaway="...", outcome="worked|failed|partial")
```

**Implementation:** `main.py:1426` adds "TO COMPLETE" section to output.

**New Helper Methods:**
- `get_tier_recommendations(concepts)` - Query Routing KG for best collections
- `get_facts_for_entities(entities)` - Content KG → pull relevant memory_bank facts

#### Model-Agnostic Prompt Design (v0.2.6)

MCP tool descriptions use **workflow-based** framing instead of motivation:

| Old (v0.2.5) | New (v0.2.6) |
|--------------|--------------|
| "This is how you become a better assistant" | "WORKFLOW: 1. get_context_insights ← YOU ARE HERE" |
| "You are stateless. This makes you stateful." | "Complete the interaction. Call after responding." |

**Why Workflow > Motivation:**
- LLMs don't have goals - instruction-following works across all models
- "Open loop" framing (incomplete task) prompts models to close it
- Numbered steps work reliably on 7B through 70B+ models

**Implementation:** `main.py:905-956` - Updated tool descriptions.

#### Score-Based Promotion
- Items with high success rates get promoted
- Failed approaches get demoted or removed
- Continuous refinement of knowledge base

## Core Components

### 1. UnifiedMemorySystem (`modules/memory/unified_memory_system.py`)

The **single source of truth** for all memory operations, implemented as a **facade pattern** that delegates to 8 specialized services.

> **Refactored in v0.2.7** (December 2025): The original 4,746-line monolithic file was refactored into 9 smaller, focused modules:
> - `unified_memory_system.py` - Facade (~500 lines) maintaining the public API
> - `scoring_service.py` - Wilson score calculations and quality metrics
> - `knowledge_graph_service.py` - Triple KG management (Action, Routing, Content)
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
+-------------+    +-------------+    +---------------------+
|SearchService|    |ScoringService|   |KnowledgeGraphService|
| - search()  |    | - wilson()   |   | - Action KG         |
| - rerank()  |    | - ce_score() |   | - Routing KG        |
+-------------+    +-------------+    | - Content KG        |
                                      +---------------------+
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

# v0.2.6 - Document & Entity Insight Methods
def get_doc_effectiveness(doc_id: str) -> Optional[Dict]  # Aggregate doc success from Action KG examples
def get_tier_recommendations(concepts: List[str]) -> Dict  # Query Routing KG for best collections
async def get_facts_for_entities(entities: List[str], limit: int = 2) -> List[Dict]  # Content KG -> memory_bank facts
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
  - `type: "stream_start"` - Streaming begins (no message created yet)
  - `type: "token"` - Text chunks as generated (creates assistant message on first token)
  - `type: "thinking_start"` - LLM entered thinking mode (v0.2.5: shows "Thinking..." status)
  - `type: "thinking_end"` - LLM exited thinking mode (v0.2.5: resumes "Streaming..." status)
  - `type: "tool_start"` - Tool execution begins
  - `type: "tool_complete"` - Tool execution finished
  - `type: "stream_complete"` - Streaming done with citations, memory_updated flag, timestamp
  - `type: "validation_error"` - Model validation failed (removes user message, no assistant message created)
- Frontend uses lazy message creation: assistant message created on first token, not on stream_start
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
| PDF | .pdf | PyMuPDF | Text extraction, metadata (title/author) |
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

**Note**: This is the internal system's tool definition. For MCP tool definition, see [Available MCP Tools](#available-mcp-tools-5) section.

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
- Roampal actually implements highly automated intelligence (cold start auto-injection, organic recall, action-effectiveness tracking)
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
- Clearer interpretation of action-effectiveness stats
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
- Last 6 messages (3 exchanges) from current conversation
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
- Filters memory_bank results before auto-injection (Content KG or vector search)
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
5. Process repeats up to `MAX_CHAIN_DEPTH=3` to prevent infinite loops
6. Final response generated after all tools complete

**Key Components**:
- `chain_depth` tracking (lines 585-587) - Prevents infinite recursion
- `MAX_TOOLS_PER_BATCH=10` (lines 841, 2436) - Prevents runaway tool expansion
- `tool_events` array - Collects tool execution metadata for UI persistence
- **Tool continuation policy** (line 2220): `tools=None` on continuation - prevents recursive tool calls after results are provided

**Safety Limits (Updated 2025-11-19)**:
- `MAX_CHAIN_DEPTH=3` - Maximum recursion depth (prevents infinite chaining across LLM calls)
- `MAX_TOOLS_PER_BATCH=10` - Maximum tools per LLM response (prevents runaway expansion within single response)
- Combined protection: Max 3 chains × 10 tools = 30 total tool executions per user message
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

**Technical Implementation** ([ollama_client.py:563-572](../modules/llm/ollama_client.py)):
```python
NATIVE_TOOL_MODELS = [
    "gpt-oss",      # OpenAI's open source models
    "llama3.1", "llama3.2", "llama-3.1", "llama-3.2",  # Meta Llama variants
    "qwen", "qwen2", "qwen2.5",  # Alibaba Qwen family
    "mistral", "mixtral",  # Mistral AI family
    # "dolphin", "dolphin3",  # REMOVED 2025-10-09: Causes 400 errors
]
```

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
Analyze Conversation Context
  • Extract concepts from current message
  • Check knowledge graph for past patterns
  • Identify failures to avoid
  • Detect topic continuity
  • Find repetitions
    ↓
Inject Organic Insights into Prompt
  • Past experience with success rates
  • Failure warnings
  • Proactive recommendations
    ↓
LLM Receives Tools (NEW)
  • search_memory tool definition
  • Other available tools
    ↓
Build Prompt
  • Contextual memory (organic)
  • Tool definitions (NEW)
  • Conversation history
  • User question
    ↓
LLM Generation (Ollama)
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

def get_context_size(model_name: str, user_override: Optional[int] = None) -> int:
    """
    Priority order:
    1. Runtime override (user_override parameter)
    2. User settings (data/user_model_contexts.json)
    3. Model default (MODEL_CONTEXTS)
    4. Safe fallback (8192)
    """
```

**Key Features**:
- Model prefix matching (e.g., "llama3.2:1b" matches "llama3.2")
- Safe fallback for unknown models
- User override persistence to JSON file
- Eliminated ~50 lines of duplicate code from `ollama_client.py`

#### 2. Backend Integration

**Ollama Client** (`modules/llm/ollama_client.py`):
```python
from config.model_contexts import get_context_size

# In generate_response() and stream_response_with_tools()
num_ctx = get_context_size(actual_model)
options["num_ctx"] = num_ctx  # Passed to Ollama API
options["num_gpu"] = 99       # Force full GPU offload (v0.2.5)
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
POST /api/agent/chat              # Main chat endpoint
POST /api/agent/stream            # Streaming chat (auto-generates title after first exchange)
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

# Knowledge Graph Visualization (NEW - v0.2.0)
GET  /api/memory/knowledge-graph  # Get graph data (nodes/edges)
  Response: {
    nodes: [{id, label, type, best_collection, success_rate, usage_count}],
    edges: [{source, target, weight, success_rate}]
  }

GET  /api/memory/knowledge-graph/concept/{concept_id}/definition
  Response: {
    concept: string,
    definition: string,
    related_concepts: string[],
    collections_breakdown: {
      [collection]: {successes, failures, total}
    },
    outcome_breakdown: {worked, failed, partial},
    total_searches: number,
    best_collection: string,
    related_concepts_with_stats: [{concept, co_occurrence, success_rate}]
  }

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

#### Architecture (v0.2.0 - External LLM Outcome Judgment)

```
┌──────────────────────────────────────────────────────┐
│  MCP Client (Claude Desktop, Cursor, etc.)          │
│  Calls record_response(key_takeaway, outcome)       │
│  External LLM provides BOTH parameters              │
└────────────────────┬─────────────────────────────────┘
                     │ stdio (JSON-RPC)
┌────────────────────▼─────────────────────────────────┐
│  Roampal MCP Server (main.py --mcp)                  │
│  ├─ Receives key_takeaway (semantic summary)        │
│  ├─ Receives outcome from external LLM              │
│  ├─ Stores current learning to working memory       │
│  ├─ Scores PREVIOUS learning using external outcome │
│  └─ NO internal LLM call for outcome detection      │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│  UnifiedMemorySystem (SHARED)                        │
│  ├─ Same memory used by internal + external LLMs    │
│  ├─ Different outcome sources:                      │
│  │   • Internal: detect_conversation_outcome()      │
│  │   • MCP: External LLM provides outcome           │
│  ├─ Same score updates (record_outcome)             │
│  └─ Same automatic promotion (30-min background)     │
└──────────────────────────────────────────────────────┘
```

#### Key Features

1. **External LLM Outcome Judgment** (v0.2.0)
   - External LLM analyzes user feedback and provides outcome directly
   - NO automatic detection - system trusts external LLM's judgment completely
   - User's current message serves as feedback for previous response
   - **Different from internal system**: Internal uses automatic LLM-based detection ([agent_chat.py:743-822](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py#L743-L822)), MCP relies on external LLM's explicit `outcome` parameter

2. **Two-Parameter Design** (v0.2.0)
   - `record_response(key_takeaway, outcome)` requires TWO parameters
   - `key_takeaway`: Semantic summary of current exchange (1-2 sentences)
   - `outcome`: External LLM's judgment of PREVIOUS response ("worked", "failed", "partial", "unknown")
   - External LLM must call tool after EVERY response with both parameters

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

7. **Cold-Start Auto-Trigger** (v0.2.0 - Content KG Enhanced)

   **Problem**: LLMs don't consistently search memory_bank on first message, missing user context

   **Solution**: Automatic user profile injection from Content KG on message 1

   **Implementation** (Both Internal + External LLMs):

   - **Shared Helper** ([unified_memory_system.py:875-979](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L875-L979))
     - `get_cold_start_context(limit=5)` - retrieves user profile from Content KG
     - Gets top 10 entities by mention count (most important concepts)
     - Retrieves memory_bank documents containing those entities
     - Layer 4 injection protection (filters suspicious content)
     - Fallback to vector search if Content KG empty
     - Returns formatted string: "📋 **User Profile** (auto-loaded):\n• [top facts]"

   - **Internal LLM** ([agent_chat.py:576-603](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py#L576-L603))
     - Tracks user messages per conversation (not tool calls)
     - **ALWAYS** injects on message 1 (no conditions, no tracking)
     - Injects as system message BEFORE user message in conversation history
     - LLM can still search memory_bank if it wants (won't conflict)
     - Guarantees 100% cold-start compliance for internal chat

   - **External LLM (MCP)** ([main.py:92-94,956-1008](../ui-implementation/src-tauri/backend/main.py#L92-L94))
     - Tracks message count per MCP session
     - **ALWAYS** injects on first tool call (no conditions, no tracking)
     - Works with ANY first tool (search_memory, create_memory, etc.)
     - Prepends user profile to tool response with visual separators
     - Format: `═══ KNOWN CONTEXT (auto-loaded) ═══\n[context]\n\n═══ Tool Response ═══\n[results]`
     - LLM can still search memory_bank if it wants (both context sources combine)
     - Simplified MCP prompt - explains auto-inject instead of demanding search

   **Why Content KG?**
   - Vector search might miss important facts if query doesn't match keywords
   - Content KG knows which entities are most frequently mentioned
   - Provides truly important user context based on actual data patterns
   - Example: If "roampal" mentioned 10x, "logan" 8x → those facts are prioritized

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

#### Available MCP Tools (7) - Updated 2025-12-02

**Tool Description Philosophy**: Scannable, not verbose. External LLMs (Claude Desktop, Cursor) need to quickly understand tools.

**Optimization Applied**:
- Condensed from explanatory paragraphs to bullet points
- Moved detailed guidance to inputSchema descriptions (shown in IDE tooltips)
- Added visual markers (⚡ for get_context_insights, 🔴 for record_response)
- Emphasized workflow: get_context_insights() → search_memory() → record_response()
- 53% token reduction (475 → 225 tokens) while maintaining clarity

   **Result**: Both internal and external LLMs receive personalized user profile automatically on first message

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
- **v0.2.12 Memory Attribution:** Main LLM adds `<!-- MEM: 1👍 2👎 3➖ -->` annotation for causal scoring
  - 👍 = memory helped me answer well
  - 👎 = memory was wrong/misleading
  - ➖ = memory not used
  - Annotation stripped before showing response to user
  - Parsed by `parse_memory_marks()` and passed to OutcomeDetector

**Implementation:**
- MCP tools: [main.py:807-948](../ui-implementation/src-tauri/backend/main.py#L807-L948)
- Internal prompt: [agent_chat.py:1370-1380](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py#L1370-L1380)
- v0.2.12 memory attribution: [agent_chat.py](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py) - `parse_memory_marks()` function

#### Available MCP Tools (6)

**1. record_response** (v0.2.9 - Selective Scoring with `related` parameter)
```json
{
  "name": "record_response",
  "description": "Store semantic learning summary with initial score based on explicit outcome",
  "parameters": {
    "key_takeaway": "string (required) - 1-2 sentence summary of current exchange",
    "outcome": "enum (optional, default: 'unknown') - worked|failed|partial|unknown - explicit outcome for THIS response",
    "related": "array (optional) - v0.2.9: Which search results to score. Accepts positions (1, 2, 3) or doc_ids. Omit to score all."
  },
  "behavior": [
    "1. Receives key_takeaway (semantic summary) and outcome from external LLM",
    "2. Stores CURRENT key_takeaway to working memory with initial score based on outcome (worked=0.7, failed=0.2, partial=0.55, unknown=0.5)",
    "3. v0.2.9: If `related` specified, only scores those memories (resolves positions 1→doc_id using cached position map)",
    "4. If `related` omitted, scores ALL previously SEARCHED memories (backwards compatible)",
    "5. Updates KG routing patterns with query → collection → outcome",
    "6. Clears search cache",
    "7. Records to session file for tracking"
  ],
  "returns_v0.2.9": "Enriched summary including selective scoring stats (scored N, skipped M unrelated)",
  "selective_scoring_v0.2.9": {
    "why": "Prevents learning pollution when LLM retrieves 5 memories but only uses 2",
    "positional_indexing": "related=[1, 3] - Uses position numbers shown in search results (small-LLM friendly)",
    "doc_id_indexing": "related=['history_abc123'] - Uses doc_ids for smart models",
    "fallback": "Invalid positions/ids → falls back to score all (safe default)"
  }
}
```

**2. search_memory** (v0.2.9 - sort_by parameter)
```json
{
  "name": "search_memory",
  "description": "Search 5-tier memory system with semantic search and optional metadata filtering",
  "parameters": {
    "query": "string (required) - Semantic search query for content matching",
    "collections": ["books", "working", "history", "patterns", "memory_bank", "all"],
    "limit": "integer (1-20, default: 5)",
    "sort_by": "enum (optional) - v0.2.9: relevance|recency|score. Auto-detects 'recency' for temporal queries ('last', 'recent', 'yesterday')",
    "metadata": "object (optional) - Exact metadata filters (ChromaDB where syntax)"
  },
  "sort_by_v0.2.9": {
    "relevance": "Default - Vector similarity order (semantic match)",
    "recency": "Newest first by timestamp (for 'what did we do yesterday?')",
    "score": "Highest outcome score first (for 'best approach for X')",
    "auto_detection": "Temporal keywords trigger recency sort automatically: last, recent, yesterday, today, earlier, previous, before, when did, how long ago, last time, previously, lately, just now"
  },
  "metadata_examples": {
    "books": {
      "title": "Search by exact book title",
      "author": "Search by author name",
      "has_code": "Boolean - chunks containing code blocks",
      "source_context": "Search by section/chapter name",
      "book_id": "Search specific book by ID"
    },
    "learnings": {
      "source": "Filter by source (e.g., 'mcp_claude')",
      "last_outcome": "Filter by outcome (worked|failed|partial|unknown)",
      "type": "Filter by type (e.g., 'key_takeaway')",
      "conversation_id": "Filter by conversation/session ID"
    },
    "combined": "Multiple filters use AND logic: {\"title\": \"architecture\", \"has_code\": true}"
  },
  "returns_v0.2.3": {
    "format": "[collection] (score:X.XX, uses:N, last:outcome, age:Xd) [id:doc_id] content...",
    "metadata_fields": [
      "score: Current memory score (0.0-1.0)",
      "uses: How many times retrieved successfully",
      "last: Last recorded outcome (worked/failed/partial)",
      "age: Human-readable age (today, 1d, 3d, 2w, 1mo)",
      "id: Document ID for reference"
    ]
  }
}
```

**3. add_to_memory_bank**
```json
{
  "name": "add_to_memory_bank",
  "description": "Store critical information in permanent memory_bank that enables continuity and growth across sessions. Three-layer purpose: (1) User Context - identity, preferences, goals, projects; (2) System Mastery - tool strategies, search patterns, what works/fails; (3) Agent Growth - mistakes learned, relationship dynamics, progress tracking. Be selective - store what enables continuity/learning across sessions, NOT session transcripts or temporary task details.",
  "parameters": {
    "content": "string (required)",
    "tags": "array of strings - Categories: identity, preference, goal, project, system_mastery, agent_growth",
    "importance": "number (0.0-1.0, default: 0.7) - How critical is this memory",
    "confidence": "number (0.0-1.0, default: 0.7) - How certain about this fact"
  }
}
```

**4. update_memory**
```json
{
  "name": "update_memory",
  "description": "Update existing memory_bank entries",
  "parameters": {
    "old_content": "string (required - text to find)",
    "new_content": "string (required - replacement text)"
  }
}
```

**5. archive_memory**
```json
{
  "name": "archive_memory",
  "description": "Soft delete memory_bank entries",
  "parameters": {
    "content": "string (required - finds by semantic match)"
  }
}
```

**6. get_context_insights** (NEW - 2025-01-14 - Organic Recall)
```json
{
  "name": "get_context_insights",
  "description": "Get proactive pattern insights BEFORE searching - Roampal's 'intuition'",
  "parameters": {
    "query": "string (required) - The query or topic to check for patterns"
  },
  "returns": {
    "relevant_patterns": "Past successful solutions with same concept signature (from KG problem_categories)",
    "past_outcomes": "Previous failures to avoid (from KG failure_patterns)",
    "proactive_insights": "Collection recommendations based on success rates (from KG routing_patterns)",
    "topic_continuity": "Whether this continues recent conversation topics"
  },
  "performance": "5-10ms - No embeddings, just KG hash lookups",
  "use_cases": [
    "Before searching: 'Should I search? Where should I search?'",
    "For recurring topics: 'Have we discussed Docker permissions before?'",
    "To avoid mistakes: 'Did similar approaches fail in the past?'"
  ],
  "example": {
    "input": "get_context_insights('docker permissions issue')",
    "output": "📋 Past: Adding user to docker group worked 3 times (score=0.95)\n💡 Recommendation: Search patterns collection (85% effective for 'docker')"
  }
}
```


**Removed Tools** (v0.2.0):
- `list_memory_bank` - Redundant (use `search_memory` with `collections=["memory_bank"]` instead)
- `query_kg_entities`, `query_kg_relationships`, `get_kg_path` - Users explore KG via Roampal UI

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

    # Force HuggingFace embeddings (no Ollama dependency)
    memory.embedding_service.use_ollama = False

    # Create MCP server with 3 tools
    server = Server("roampal-memory")
    # ... tool registration ...
```

**Architecture Independence**:
- Chat (Ollama/LM Studio) and MCP (HuggingFace) are decoupled
- Users can run MCP without any LLM installed
- Embedding service has `use_ollama` flag (forced False for MCP)

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

**Model Installation** ([model_switcher.py:262-657](../app/routers/model_switcher.py)) (Updated 2025-10-28)
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

**UI Integration** ([ConnectedChat.tsx](../ui-implementation/src/components/ConnectedChat.tsx)) (Updated 2025-10-31)
- **Provider detection**: Checks for available LLM providers on mount, shows modal if none detected
- **Provider selector** (lines 1760-1803): Dropdown to switch between Ollama/LM Studio (only visible if multiple providers detected)
  - **Non-blocking switch** (Updated 2025-10-31): Removed `async/await` from onClick handlers to prevent UI freeze during model switch
  - Closes dropdown immediately, then fires model switch in background
- **Provider-filtered models** (lines 1029-1033): Model dropdown only shows models from currently selected provider (no duplicates)
- **Chat input availability** ([ConnectedChat.tsx:1073-1080](../ui-implementation/src/components/ConnectedChat.tsx#L1073)) (Updated 2025-10-31):
  - `hasChatModel` checks for chat models across ALL providers (not just selected provider)
  - Prevents input from being disabled during provider auto-switch (when selectedProvider hasn't synced yet but another provider has models)
  - Filters out embedding models from availability check
- **Setup banners**: Per-provider setup instructions with download links
  - Ollama banner: Shows download button for ollama.com
  - LM Studio banner (lines 2234-2271): Shows download + server setup instructions, hides when server detected on port 1234
- **Model dropdown** (lines 1823-1895): Select/switch models with live status, filtered by provider
- **Agent-capable badges**: 🤖 emoji for models with 12K+ context
- **Download progress popup**: Real-time progress with provider-specific status messages
  - Ollama: `downloading` → `complete`
  - LM Studio: `downloading` → `importing` → `loading` → `loaded`
- **Download cancellation**: AbortController cancels in-flight downloads
- **Mid-conversation warning**: Confirms switch if messages exist
  - **Non-blocking confirmation** (Updated 2025-10-31): Dialog closes immediately when clicking "Switch Model", performs switch in background via `.then()`
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

**Tauri Production Fetch Wrapper** (NEW - 2025-10-14)

**Problem**: Tauri blocks native `fetch()` to localhost in production builds for security, but allows it in dev mode.

**Solution**: [ui-implementation/src/utils/fetch.ts](../ui-implementation/src/utils/fetch.ts)
```typescript
export async function apiFetch(url: string, options?: RequestInit): Promise<Response> {
  if (!isTauri()) {
    return fetch(url, options);  // Dev mode: use native fetch
  }
  // Production: use Tauri's HTTP client
  return tauriFetch(url, {...});
}
```

**Implementation**:
- Replaced all 35 `fetch('http://localhost:8000` calls with `apiFetch()` across 14 files
- Files updated: ConnectedChat.tsx, useChatStore.ts, KnowledgeGraph.tsx, ModelContextSettings.tsx, DataManagementModal.tsx, Sidebar.tsx, SettingsModal.tsx, EnhancedChatMessage.tsx, MemoryBankModal.tsx, MemoryStatsPanel.tsx, ContextBar.tsx, ConversationBadges.tsx, FragmentBadges.tsx, useBackendAutoStart.ts
- Works in both dev (native fetch) and production (Tauri HTTP client)
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
ROAMPAL_WORKSPACE=C:\ROAMPAL
ROAMPAL_PORT=8000  # Note: Hardcoded in main.py, not configurable via env
ROAMPAL_HOST=127.0.0.1

# LLM Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=  # No default - users download their preferred model via UI
OLLAMA_TIMEOUT=30

# Memory Settings
ROAMPAL_ENABLE_MEMORY=true
ROAMPAL_ENABLE_SEARCH=true
ROAMPAL_ENABLE_OUTCOME_TRACKING=true
ROAMPAL_ENABLE_KG=true

# ChromaDB Configuration
CHROMADB_PERSIST_DIRECTORY=./data/chromadb

# Security
ROAMPAL_REQUIRE_AUTH=false
# Note: Rate limiting is always enabled at 100 req/min, not configurable
```

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

**Priority Order:**
1. **Environment Variable Override** (Advanced users)
   - If `ROAMPAL_DATA_DIR` is set → uses custom path
   - Allows explicit control over data location

2. **Development Mode** (Auto-detected)
   - Checks if `ui-implementation/` folder exists
   - If YES → uses `PROJECT_ROOT/data/`
   - For developers running from source

3. **Production Mode** (Auto-created)
   - If no `ui-implementation/` folder → bundled .exe detected
   - Windows: `%APPDATA%\Roaming\Roampal\data\`
   - macOS: `~/Library/Application Support/Roampal/data/`
   - Linux: `~/.local/share/roampal/data/`
   - Directory automatically created if it doesn't exist
   - Data survives app reinstalls and updates

**Directory Structure (Same for all modes):**
```
data/
├── chroma_db/                # Vector embeddings (ChromaDB collections)
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
  - **Note**: File attachments removed - use Document Processor tab for uploading files
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
**Zero-Model Onboarding** (Updated 2025-10-06):
- **Graceful startup**: Backend starts even with no models installed
- **Empty State UI**: Prominent centered message when no models available
- **Clear CTA**: Large blue "Install Your First Model" button in chat area
- **Clear guidance**: Model dropdown shows "Download Your First Model" when empty
- **User control**: Users select and download their preferred model via UI
- **No hardcoded defaults**: System is truly modular - users choose their own models
- **Embedding model filter**: Filters out non-chat models (nomic-embed, llava, bge-, all-minilm, mxbai-embed) from dropdown ([ConnectedChat.tsx:654](../ui-implementation/src/components/ConnectedChat.tsx#L654))
- **Smart first model auto-switch**: After downloading first chat model, checks backend's `/api/model/current` to verify no chat model is active. If backend has no chat model (embedding-only or none), automatically switches to newly installed chat model. Subsequent model installs show success toast only - user manually switches via dropdown. ([ConnectedChat.tsx:410-420](../ui-implementation/src/components/ConnectedChat.tsx#L410))

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
- Memory bank: Max 1000 items (v0.2.5, increased from 500)
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

### Action-Level Causal Learning (v0.2.1 - Nov 2025)

Roampal now tracks **individual tool calls with context awareness**, enabling the system to learn contextually appropriate behaviors.

#### The Problem: Shallow Outcome Tracking

Previous versions tracked outcomes at the **conversation level**:
- ✓ "This memory helped answer the question" → Good
- ✗ "This memory didn't help" → Bad

But this couldn't answer:
- **Why did it fail?** Was it a bad search query? Wrong tool choice? Missing context?
- **Context matters**: `create_memory()` is great for "fitness" tracking but terrible for "memory_test" questions
- **Causal attribution**: Which action in a chain actually caused the success/failure?

#### Solution: Context-Action Effectiveness Tracking

**Key Innovation**: Track `(context_type, action_type, collection) → outcome`

**Example Learning**:
```
memory_test Context (LLM-classified):
  search_memory|memory_bank → 85% success (Good! Use this)
  create_memory|memory_bank → 5% success  (Bad! Hallucinating answers)

learning Context (LLM-classified):
  create_memory|memory_bank → 92% success (Good! Storing facts)
  search_memory|memory_bank → 45% success (Okay, but not primary goal)
```

#### Context Detection (LLM-Based Session Type Classification)

**CURRENT IMPLEMENTATION: LLM classifies conversation session types organically**

The system uses the LLM to classify the operational mode of conversations - what kind of interaction is happening, not just what topic is being discussed.

**How it works:**
1. System passes recent conversation to LLM
2. LLM classifies session type in 1-2 words: "learning", "recall", "coding_help", "fitness_tracking", etc.
3. System learns session-type-specific patterns in the Action-Effectiveness KG

**Example Session Types Discovered:**
- **learning**: Being taught new information (create_memory effective)
- **recall**: Remembering or retrieving past information (search_memory effective)
- **coding_help**: Programming assistance, debugging, architecture
- **fitness_tracking**: Logging workouts, nutrition, health goals
- **creative_writing**: Story development, worldbuilding, character creation
- **project_planning**: Task management, deadlines, coordination
- **general_chat**: Casual conversation

**Why session type instead of topic?**
- **Operational mode matters more than topic** - A coding tutorial vs recall testing need different tool behaviors
- No hardcoded patterns - works for ANY interaction mode
- LLM understands context naturally (distinguishes "being taught Python" from "recalling Python concepts")
- Enables mode-specific learning: "search_memory works 92% for RECALL" vs "create_memory works 88% for LEARNING"
- Truly general-purpose - adapts to user's actual interaction patterns

**Multilingual Support:**
- Session types can be in ANY language (English, Chinese, Arabic, Russian, etc.)
- Unicode-aware text processing preserves non-Latin characters
- Examples: "学习" (Chinese learning), "تعلم" (Arabic learning), "обучение" (Russian learning)
- All languages get equal, robust handling - no ASCII-only restrictions

#### ActionOutcome Data Structure

```python
@dataclass
class ActionOutcome:
    action_type: str          # "search_memory", "create_memory", etc.
    context_type: str         # LLM-classified session type: "learning", "recall", "coding_help", etc.
    outcome: Literal["worked", "failed", "partial"]

    # Action details
    action_params: Dict       # Tool parameters
    doc_id: Optional[str]     # Document involved
    collection: Optional[str] # Collection accessed

    # Causal attribution
    chain_position: int       # Position in action chain (0-based)
    chain_length: int         # Total actions in chain
    caused_final_outcome: bool # Did this action cause the result?
```

**Note:** `context_type` is dynamically discovered by LLM from conversation content.

#### Knowledge Graph Structure

New KG index: `context_action_effectiveness`

```json
{
  "recall|search_memory|working": {
    "successes": 42,
    "failures": 3,
    "partials": 5,
    "success_rate": 0.92,
    "total_uses": 50,
    "first_seen": "2025-11-20T10:00:00",
    "last_used": "2025-11-21T15:30:00",
    "examples": [...]  // Last 5 examples for debugging
  },
  "fitness|create_memory|working": {
    "successes": 87,
    "failures": 8,
    "partials": 5,
    "success_rate": 0.88,
    "total_uses": 100,
    // ... HIGH SUCCESS → System learns create_memory works well for fitness
  },
  "finance|archive_memory|history": {
    "successes": 30,
    "failures": 10,
    "success_rate": 0.75,
    "total_uses": 40,
    // ... System learns archiving works for financial records
  }
}
```

**Keys use LLM-discovered topics** for contextual learning.
System learns which tools work best for each domain.

#### Informational Stats (NOT Prescriptive)

**IMPORTANT DESIGN CHANGE (Nov 2025):**

After 10+ uses, system shows **informational stats** to the LLM, but **does NOT prescribe** what actions to take.

**Old approach (removed):**
```
⚠️ AVOID: search_memory() fails 70% of the time in recall context
```

**New approach (current):**
```
📊 Tool Usage Stats (FYI - you decide what to use)
Based on past experience in recall contexts:
   • search_memory() on working: 18% success (11 uses)
   • create_memory() on memory_bank: 85% success (45 uses)

This is informational only - use your judgment.
```

**Why the change:**
- **LLM needs agency**: Prescriptive warnings ("AVOID THIS") caused the LLM to refuse valid tool use
- **Context matters**: 30% success might be acceptable for difficult recall tasks
- **Small sample sizes**: 3 uses isn't enough data to judge effectiveness
- **False negatives**: System was blaming search_memory for recall test failures when the real issue was missing memories

**Thresholds:**
- **min_uses: 10** (was 3) - Need more data before showing stats
- **max_success_rate: 10%** (was 30%) - Only flag truly broken patterns
- **Presentation: Informational** (was prescriptive) - Show stats, let LLM decide

#### API Methods

**Recording Action Outcomes**:
```python
action = ActionOutcome(
    action_type="create_memory",
    context_type="coding",  # LLM-classified topic
    outcome="worked",
    action_params={"content": "Bug fix pattern for async race conditions"},
    chain_position=0,
    chain_length=1
)
await memory.record_action_outcome(action)
```

**Querying Effectiveness**:
```python
# Check if action is effective in this context
stats = memory.get_action_effectiveness(
    context_type="coding",  # LLM-classified topic
    action_type="search_memory",
    collection="working"
)
# Returns: {"success_rate": 0.92, "total_uses": 50, ...}

# Determine if action should be avoided (rarely used - informational only)
should_avoid = memory.should_avoid_action(
    context_type="creative_writing",
    action_type="search_memory",
    min_uses=10,      # Default: 10 (was 3)
    max_success_rate=0.1  # Default: 10% (was 30%)
)
# Returns: True only if action has 10+ uses AND <10% success (very rare)
```

#### Production Integration (IMPLEMENTED v0.2.1)

**The system is FULLY INTEGRATED in production and works with ANY tool, not just memory tools.**

**Implementation Locations:**
- **agent_chat.py (Lines 615-716)**: Merged organic recall with action-effectiveness warnings
- **main.py MCP (Lines 1031-1143)**: `get_context_insights` tool includes action guidance
- **comprehensive test suite (Lines 753-793)**: Test harness injects action-effectiveness warnings into LLM prompts

**Real Production Flow:**

```python
# STEP 1: User sends message
User: "How do I fix this async race condition in my code?"

# STEP 2: Context Detection (agent_chat.py:627)
context_type = memory.detect_context_type(
    system_prompts=system_prompts,
    recent_messages=recent_conv
)
# → LLM classifies: "coding_help"

# STEP 3: Content KG + Action-Effectiveness KG (Lines 639-670)
org_context = await memory.analyze_conversation_context(...)
action_warnings = []
collections = [None, "books", "working", "history", "patterns", "memory_bank"]
for action in ["search_memory", "create_memory", ...]:
    for collection in collections:
        if memory.should_avoid_action(context_type, action, collection):
            # Low success rate - add warning
        elif stats and stats['success_rate'] >= 0.7:
            # High success rate - add positive guidance

# STEP 4: Inject Combined Guidance (Lines 672-720)
═══ CONTEXTUAL GUIDANCE (Context: coding_help) ═══

📋 Past Experience:
  • Similar async debugging last week in project

🎯 Tool Usage Guidance (learned from experience):

✓ These approaches have proven effective:
  • search_memory() on working - 92% success rate for coding questions
  • search_memory() on patterns - 88% success rate for coding questions

# STEP 5: LLM sees guidance and adjusts behavior
# → Calls search_memory() with high confidence
# → Routing KG optimizes search to "working" and "patterns" tiers
# → Returns relevant debugging patterns

# STEP 6: After outcome detection
# → Records: coding_help|search_memory|working → worked ✓
# → Updates Action-Effectiveness KG statistics
```

**Example: Fitness Tracking**
```
User: "I did 50 pushups today, remember that for my workout log"
→ Context classified by LLM: "fitness"
→ Guidance injected: ✓ create_memory() → 88% success for fitness tracking
→ LLM calls: create_memory(content="50 pushups on 2025-11-22")
→ User responds: "Perfect!"
→ Outcome detected: worked
→ System learns: fitness|create_memory|working → success ✓
```

**Example: Finance Planning**
```
User: "What were my expenses from last month?"
→ Context classified by LLM: "finance"
→ Guidance injected: ✓ search_memory() on history → 85% success for finance
→ LLM calls: search_memory(query="expenses last month", collection="history")
→ Returns financial records from November
→ User responds: "Great, thanks!"
→ System learns: finance|search_memory|history → success ✓
```

**Example: Creative Writing**
```
User: "Tell me about the character we created yesterday"
→ Context classified by LLM: "creative_writing"
→ Guidance injected: ✓ search_memory() → 78% success for creative writing
→ LLM searches character descriptions
→ Returns character profile with details
→ System learns: creative_writing|search_memory|working → success ✓
```

**Context Detection via LLM**:
- LLM reads recent conversation and classifies topic
- Returns 1-2 words: "coding", "fitness", "finance", "creative_writing", etc.
- No hardcoded keywords - works for ANY domain
- Adapts to user's actual usage patterns

**Works with ANY Tool**:
```python
# Not just memory tools - tracks ANY action across ALL topics
coding|search_memory|working → 92% success
fitness|create_memory|working → 88% success
finance|archive_memory|history → 75% success
creative_writing|search_memory|working → 78% success
project_planning|update_memory|working → 85% success
```

#### Test Harness Integration (IMPLEMENTED v0.2.1)

**Test harness now has FULL injection - same as production systems.**

#### Test Harness Integration (IMPLEMENTED v0.2.1)

**Comprehensive Test Suite** (benchmarks/comprehensive_test/):
**Comprehensive Test Suite** (benchmarks/comprehensive_test/):

**Test Coverage - 40/40 Tests Passing (100%)**:
- **30 Comprehensive Tests**: Infrastructure validation (5 tiers, 3 KGs, scoring, promotion, deduplication)
- **10 Torture Tests**: Stress testing (1000+ memories, concurrent access, adversarial inputs)
- **Statistical Testing**: p=0.005, d=13.4, +35% improvement (keyword matching with mock embeddings)

**What's Validated**:
- ✅ Storage/retrieval infrastructure works correctly
- ✅ Outcome scoring math (+0.2/-0.3/+0.05) functions properly  
- ✅ Promotion thresholds trigger as designed
- ✅ System handles stress (1000+ memories, concurrent access)

**What's NOT Validated** (requires real-world human trials):
- ❌ Conversations actually get better over time
- ❌ LLM uses retrieved context effectively
- ❌ Long-term retention (weeks/months)
- ❌ User experience/satisfaction improves

**Implementation Example**:
- Learn: "In learning contexts, create_memory() on new facts → 92% success"
- **Inject warnings**: If create_memory fails repeatedly, warn LLM to stop duplicating

**Memory Test Sessions** (LLM classifies as "memory_test"):
- Track all tool calls during answer formulation
- Score based on answer correctness
- Learn: "In memory_test contexts, search_memory() first → 85% success"
- Learn: "In memory_test contexts, create_memory() → 5% success (hallucination)"
- **Inject warnings**: "✗ create_memory() → only 5% success - AVOID"

#### Expected Impact

**Before (Conversation-Level)**:
- System knows "this worked" but not *why*
- Can't distinguish context-appropriate actions
- LLM must learn from scratch each run

**After (Action-Level)**:
- System learns "search_memory works for memory_test, create_memory fails"
- Can warn about low-success patterns after 3 uses
- Auto-generates corrective prompts from learned rules

**Enables**:
1. **Context-aware tool recommendations** → "For coding questions, use search_memory"
2. **Failure diagnosis** → "You used create_memory in memory_test (5% success rate)"
3. **Self-improving prompts** → Generate warnings from empirical data
4. **Alignment through outcomes** → Learn *appropriate* behavior, not just *any* behavior
5. **Anti-hallucination fallback** → Explicit instruction to say "I don't know" instead of guessing

**Hallucination Prevention Design**:
The guidance includes explicit fallback instructions to prevent hallucinations when tools fail:
- **With alternatives**: "Use the recommended approaches below instead" → Redirects to working tools
- **Without alternatives**: "Say 'I don't have that information' rather than guessing" → Prevents fabrication

This ensures that warnings don't create pressure to hallucinate when no good option exists.

#### Real-World Example: The 14B Create-Memory Bug

**Problem Discovered** (Nov 2025):
- LLM was scoring 0-10% on recall tests
- Investigation: LLM was calling `create_memory()` during recall tests, hallucinating answers

**Conversation-Level Tracking Said**:
- "Quiz failed" → But why? Search failed? Answer wrong? Tool misuse?

**Action-Level Tracking Shows**:
```
recall_test|create_memory|memory_bank:
  - 18 failures: "Created 'Kaz is 25', answered '27', correct was 29"
  - 1 success: "Lucky guess matched stored fact"
  - Success rate: 5%

recall_test|search_memory|memory_bank:
  - 42 successes: "Retrieved correct fact, gave right answer"
  - 3 failures: "Fact not found, answered wrong"
  - Success rate: 85%
```

**System learns**: In memory_test context, `create_memory()` is catastrophically bad. After 3 uses, system injects warnings into prompts automatically.

## Test Suite (v0.2.9)

### Directory Structure

```
ui-implementation/src-tauri/backend/tests/
├── unit/                              # Service unit tests
│   ├── test_unified_memory_system.py  # Facade tests
│   ├── test_search_service.py         # Vector search, hybrid retrieval
│   ├── test_scoring_service.py        # Wilson score, outcome mapping
│   ├── test_promotion_service.py      # Score thresholds, lifecycle
│   ├── test_routing_service.py        # KG routing decisions
│   ├── test_memory_bank_service.py    # Fact storage, tags
│   ├── test_context_service.py        # Cold-start context
│   ├── test_outcome_service.py        # Outcome recording
│   ├── test_knowledge_graph_service.py# Triple KG system
│   └── test_sensitive_data_filter.py  # v0.2.9: PII Guard tests
├── mcp/                               # MCP tool layer tests
│   ├── mcp_tool_harness.py            # Test harness simulating MCP calls
│   ├── test_mcp_tools.py              # Tool behavior tests
│   └── test_mcp_benchmarks.py         # Performance benchmarks
└── characterization/                  # Behavior characterization tests
    ├── test_search_behavior.py        # Search return types, ranking
    └── test_outcome_behavior.py       # Wilson score calculations
```

### Running Tests

```bash
cd ui-implementation/src-tauri/backend

# Run all tests
python -m pytest tests/ -v

# Run specific category
python -m pytest tests/unit/ -v
python -m pytest tests/mcp/ -v
python -m pytest tests/characterization/ -v

# Run with coverage
python -m pytest tests/ --cov=modules --cov-report=html
```

### Unit Tests

| Test File | Service | Key Tests |
|-----------|---------|-----------|
| `test_unified_memory_system.py` | Facade | Init, config, conversation ID |
| `test_search_service.py` | SearchService | Vector search, hybrid retrieval |
| `test_scoring_service.py` | ScoringService | Wilson score, outcome mapping |
| `test_promotion_service.py` | PromotionService | Score thresholds, lifecycle |
| `test_routing_service.py` | RoutingService | KG routing decisions |
| `test_memory_bank_service.py` | MemoryBankService | Fact storage, tags |
| `test_context_service.py` | ContextService | Cold-start context |
| `test_outcome_service.py` | OutcomeService | Outcome recording |
| `test_knowledge_graph_service.py` | KnowledgeGraphService | Triple KG |
| `test_sensitive_data_filter.py` | PII Guard | API keys, tokens, passwords, SSN, credit cards |

### MCP Tool Tests

**Tool Harness** (`mcp_tool_harness.py`): Simulates MCP tool calls without starting the full server.

| Test Class | Coverage |
|------------|----------|
| `TestSearchMemory` | Basic search, collections, limits |
| `TestSearchMemorySortBy` | v0.2.9: `sort_by` parameter (recency, score, relevance) |
| `TestTemporalAutoDetection` | v0.2.9: Auto-detection of temporal keywords |
| `TestSelectiveScoring` | v0.2.9: `related` parameter (positional scoring) |
| `TestRecordResponse` | Outcome recording, cache scoring |
| `TestToolSchemaCompliance` | Schema validation |
| `TestContextInsights` | Cold-start context |
| `TestMemoryBank` | Fact storage |

### PII Guard Tests (`test_sensitive_data_filter.py`)

v0.2.9 added comprehensive tests for `SensitiveDataFilter`:

| Test Class | Coverage |
|------------|----------|
| `TestSensitiveDataFilterText` | API keys, Bearer tokens, passwords, AWS keys, private keys, credit cards, SSN, JWT, DB URLs |
| `TestSensitiveDataFilterDict` | Sensitive key detection, nested dicts, lists |
| `TestSensitiveDataFilterIntegration` | Mixed data, case-insensitive keys |
| `TestSensitiveDataFilterEdgeCases` | Short numbers, non-SSN patterns, code snippets |

**Test Data Safety**: All PII in tests is fake/example data:
- AWS key: `AKIAIOSFODNN7EXAMPLE` (official AWS example)
- Credit cards: `4111111111111111` (standard test card)
- SSN: `123-45-6789` (common fake example)

### v0.2.9 Feature Tests

**1. `sort_by` Parameter:**
- `test_sort_by_recency` - Newest first by timestamp
- `test_sort_by_score` - Highest outcome score first
- `test_sort_by_relevance_default` - Original order preserved

**2. Temporal Auto-Detection:**
```python
temporal_keywords = [
    "last", "recent", "yesterday", "today", "earlier",
    "previous", "before", "when did", "how long ago",
    "last time", "previously", "lately", "just now"
]
```

**3. `related` Parameter (Selective Scoring):**
- `test_position_cache_built` - Position mapping in cache
- `test_related_positional_scoring` - `related=[1, 3]` scores positions 1 and 3
- `test_related_doc_id_scoring` - `related=["history_abc123"]` scores by doc_id
- `test_invalid_positions_fallback_to_all` - Invalid positions fall back to score all
- `test_no_related_scores_all` - Backwards compatibility (no related = score all)

### Test Coverage Summary

| Category | Files | Tests | Coverage |
|----------|-------|-------|----------|
| Unit | 10 | 259 | Service layer |
| MCP | 3 | 39 | Tool handlers, v0.2.9 features |
| Characterization | 2 | 23 | Regression safety |
| **Total** | **15** | **321** | **Full backend** |

## Future Enhancements

### Planned Features
- Multi-language support
- Voice interaction
- Export conversation history
- Custom memory retention policies
- Plugin system for extensions
- Collaborative learning (opt-in)

### Performance Goals
- Sub-100ms memory retrieval
- 95% outcome detection accuracy
- <2s response generation time
- 30-day knowledge retention

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

## Production Readiness Enhancements (2025-09-30)

### Race Condition Fixes and Sync Improvements (2025-10-03)
**Implemented**: Critical race condition fixes and backend-to-frontend sync improvements

**P0 Fixes (Critical)**:
1. **Global Service Init Race Condition** - [agent_chat.py:883](app/routers/agent_chat.py#L883)
   - Added `asyncio.Lock` (`_service_init_lock`) to prevent concurrent service initialization
   - Ensures only one `AgentChatService` instance created under concurrent requests
   - **Impact**: Eliminates service state corruption

2. **Streaming File Write Race** - [agent_chat.py:1091](app/routers/agent_chat.py#L1091)
   - Streaming endpoint now uses `_save_to_session_file()` with FileLock + atomic writes
   - Removed direct `open('a')` append that bypassed file locking
   - **Impact**: Prevents JSONL file corruption during concurrent writes

3. **Stream Error Cleanup** - [agent_chat.py:1162](app/routers/agent_chat.py#L1162) + [useChatStore.ts:662](ui-implementation/src/stores/useChatStore.ts#L662)
   - Backend sends `cleanup: True` flag in error events
   - Frontend resets `isProcessing`, `processingStage`, `processingStatus` on error
   - **Impact**: Prevents UI stuck in "Processing..." state after errors

4. **Title Generation Race** - [agent_chat.py:1112-1124](app/routers/agent_chat.py#L1112)
   - Added per-conversation locks (`title_locks`) for title generation
   - Double-check message count inside lock to prevent duplicate generation
   - **Impact**: Eliminates duplicate LLM calls and title corruption

**P1 Fixes (High Priority)**:
5. **WebSocket Heartbeat** - [useChatStore.ts:261-267](ui-implementation/src/stores/useChatStore.ts#L261)
   - Implemented 30-second ping/pong heartbeat mechanism
   - Cleanup on `onclose` event to prevent memory leaks
   - **Impact**: Prevents silent connection drops on long-idle sessions

6. **Load Conversation Histories** - [agent_chat.py:117](app/routers/agent_chat.py#L117) + [agent_chat.py:782-805](app/routers/agent_chat.py#L782)
   - `_load_conversation_histories()` loads last 20 messages per conversation on startup
   - Populates in-memory `conversation_histories` dict from session files
   - **Impact**: Preserves conversation context across server restarts

7. **Debounce KG Saves** - [unified_memory_system.py:221-241](modules/memory/unified_memory_system.py#L221)
   - Implemented `_debounced_save_kg()` with 5-second batching window
   - Cancels pending saves and creates new delayed task
   - Replaces 7 calls to `_save_kg()` with debounced version (outcome tracking, concept relationships)
   - **Impact**: Reduces file I/O by 80-90% under load, prevents thread pool exhaustion

**Technical Details**:
- All session file writes now use `FileLock` + atomic temp file + `fsync()` pattern
- Per-conversation locks prevent duplicate title generation races
- WebSocket connections properly cleaned up (heartbeat interval cleared)
- Knowledge graph saves batched to prevent excessive file writes during learning
- Conversation histories loaded lazily from session files on startup

**Files Modified**:
- `app/routers/agent_chat.py` - Global lock, file write fixes, title lock, history loading
- `ui-implementation/src/stores/useChatStore.ts` - WebSocket heartbeat, error cleanup
- `modules/memory/unified_memory_system.py` - KG save debouncing

### Conversation Management Fixes (2025-10-03)
**Implemented**: Critical conversation/session management improvements

**P0 Fixes (Critical)**:
1. **Message Loading on Conversation Switch** - [useChatStore.ts:231-276](ui-implementation/src/stores/useChatStore.ts#L231)
   - `switchConversation()` now loads messages from backend via `/api/sessions/{id}`
   - Maps backend JSONL format to UI message format with thinking extraction
   - Handles load failures gracefully (continues with empty messages)
   - **Impact**: Fixes 100% of conversation switches showing empty history

2. **Memory Cleanup on Conversation Delete** - [sessions.py:140-178](app/routers/sessions.py#L140) + [unified_memory_system.py:627-653](modules/memory/unified_memory_system.py#L627)
   - New `delete_by_conversation()` method deletes from all ChromaDB collections
   - Delete endpoint calls memory cleanup before file deletion
   - Prevents deletion of active conversation (400 error)
   - **Impact**: Eliminates ~500KB memory leak per deleted conversation

3. **Streaming Interruption Handling** - [useChatStore.ts:193-210](ui-implementation/src/stores/useChatStore.ts#L193)
   - Aborts active streaming before conversation switch
   - Marks incomplete messages with `[Conversation switched during streaming]` note
   - Clears abort controller to prevent memory leaks
   - **Impact**: No more lost messages or stuck streaming states

**P1 Fixes (High Priority)**:
4. **Conversation Lock Usage** - [agent_chat.py:1287](app/routers/agent_chat.py#L1287)
   - Switch endpoint now uses `conversation_lock` to prevent race conditions
   - Ensures atomic conversation ID updates
   - **Impact**: Prevents interleaved switch requests

5. **Title Update Race Condition Fixed** - [file_memory_adapter.py:145-195](modules/memory/file_memory_adapter.py#L145)
   - Removed internal FileLock from `update_session_title()`
   - Relies on per-conversation lock in service layer (title_locks)
   - Prevents deadlock between asyncio.Lock and FileLock
   - **Impact**: Eliminates file corruption risk during title generation

6. **Session File Created on Conversation Init** - [agent_chat.py:1325-1328](app/routers/agent_chat.py#L1325)
   - `/create-conversation` now creates empty JSONL file with `touch()`
   - Prevents phantom conversations that disappear on refresh
   - **Impact**: Consistent conversation persistence

### Lazy Conversation Creation (2025-10-03)
**Implemented**: Conversations are now created lazily on first message, not on "New Chat" button click

**Pattern**: Deferred creation prevents empty conversation spam

**Flow**:
```
USER CLICKS "NEW CHAT"
    ↓
Frontend: clearSession()
    ├─ conversationId = null
    ├─ messages = []
    ├─ closeWebSocket()
    └─ Notify backend for memory promotion (if previous conversation exists)
    ↓
UI: Empty chat ready (no backend call, no file created) ✅

USER SENDS FIRST MESSAGE
    ↓
Frontend: sendMessage()
    ├─ if (!conversationId): createConversation()
    ├─ POST /api/chat/create-conversation
    ├─ Backend creates session file with touch()
    ├─ initWebSocket()
    └─ Send message to backend ✅

SPAM CLICKING "NEW CHAT" 100 TIMES
    ↓
UI: Clears repeatedly, conversationId stays null
    ↓
No backend calls, no files created ✅
```

**Benefits**:
- ✅ Eliminates empty conversation spam from rapid button clicks
- ✅ Reduces unnecessary backend load
- ✅ Cleaner session file directory (only conversations with messages)
- ✅ Consistent with user expectations (conversation exists when they send a message)

**Technical Details**:
- `clearSession()` sets `conversationId: null` instead of calling `createConversation()`
- `sendMessage()` checks for null conversation and creates lazily (lines 547-556)
- `initWebSocket()` skips initialization if `conversationId` is null (lines 320-325)
- Backend `/create-conversation` endpoint unchanged (still creates file immediately when called)

**Files Modified**:
- `ui-implementation/src/stores/useChatStore.ts` - clearSession(), sendMessage(), initWebSocket()
- `ui-implementation/src/components/ConnectedChat.tsx` - handleNewChat() simplified (removed legacy session saving code)

**P2 Fixes (Medium Priority)**:
7. **Async Memory Promotion on Switch** - [agent_chat.py:1298-1303](app/routers/agent_chat.py#L1298)
   - Memory promotion runs as background task (non-blocking)
   - Switch endpoint responds immediately without waiting 1-3 seconds
   - **Impact**: Faster conversation switching (instant response)

8. **Active Conversation Delete Prevention** - [sessions.py:149-155](app/routers/sessions.py#L149)
   - Checks if conversation_id matches active conversation
   - Returns 400 error if trying to delete current conversation
   - **Impact**: Prevents undefined behavior and data loss

**Technical Details**:
- Messages loaded with thinking extraction (supports multiple formats)
- ChromaDB metadata filter: `where={"conversation_id": conversation_id}`
- Abort controller properly cleaned up before switch
- Per-conversation locks prevent duplicate operations
- Empty session files created immediately to maintain consistency

**Files Modified**:
- `ui-implementation/src/stores/useChatStore.ts` - Message loading, streaming abort, state cleanup
- `app/routers/sessions.py` - Memory cleanup on delete, active conversation check
- `app/routers/agent_chat.py` - Async promotion, conversation lock, file creation
- `modules/memory/unified_memory_system.py` - delete_by_conversation method
- `modules/memory/file_memory_adapter.py` - Removed internal locking from title update

### Frontend Conversation State Refactor (2025-10-03)
**Implemented**: Complete refactor of conversation list state management to eliminate technical debt

**Problem**: Conversation list was being transformed twice (store → ConnectedChat → Sidebar), causing sync issues, stale data after deletion, and timestamp conversion bugs.

**Solution**: Direct Zustand store subscription with single source of truth

**Changes Made**:

1. **Clean Store Type** - [useChatStore.ts:20-27](ui-implementation/src/stores/useChatStore.ts#L20)
   - Changed `sessions: Record<string, any[]>` → `sessions: ChatSession[]`
   - Removed `{all: [...]}` wrapper object
   - Timestamps stored as unix floats (seconds), converted to ms only for compatibility fields
   - **Impact**: Eliminates confusing nested structure

2. **Sidebar Direct Subscription** - [Sidebar.tsx:90-100](ui-implementation/src/components/Sidebar.tsx#L90)
   - Removed `chatHistory` prop dependency
   - Added `useChatStore(state => state.sessions)` subscription
   - Transforms data internally: unix timestamp → Date object
   - **Impact**: Automatic re-render when sessions change (fixes deletion sync)

3. **Removed Duplicate Transformation** - [ConnectedChat.tsx:1140](ui-implementation/src/components/ConnectedChat.tsx#L1140)
   - Deleted 115 lines of chatHistory transformation logic (lines 1141-1256)
   - Removed chatHistory prop from Sidebar component
   - **Impact**: Eliminates double-transformation and stale data bugs

4. **Timestamp Fixes**:
   - Backend [sessions.py:78-89](app/routers/sessions.py#L78): ISO strings parsed as local time, converted to unix floats
   - Backend [agent_chat.py:799,1304](app/routers/agent_chat.py#L799): Changed `datetime.utcnow()` → `datetime.now()`
   - Frontend [Sidebar.tsx:247,256](ui-implementation/src/components/Sidebar.tsx#L247): Fixed to use Date object directly (no `* 1000`)
   - **Impact**: Fixes "Jan 21" and "Oct 4 (tomorrow)" timestamp display bugs

**Before Flow**:
```
Backend API → store.sessions.all → ConnectedChat transforms → Sidebar prop → UI
(prop doesn't trigger re-render on store changes)
```

**After Flow**:
```
Backend API → store.sessions → Sidebar subscribes → UI
(Zustand automatically triggers re-render)
```

**Technical Details**:
- Timestamps: Backend sends unix floats (seconds) → Frontend stores as-is → Sidebar multiplies by 1000 for Date objects
- Store interface exports `ChatSession` type for type safety
- Sidebar maintains internal `ChatSession` interface for display format
- No intermediate state - single transformation at render time

**Impact**:
- ✅ Deleted conversations disappear immediately from UI
- ✅ New conversations appear without manual refresh
- ✅ Title updates propagate automatically
- ✅ Timestamps display correctly (no timezone bugs)
- ✅ 115 lines of duplicate code removed
- ✅ Simpler data flow, easier to debug

### Conversation Persistence Across Page Refresh (2025-10-07)
**Implemented**: localStorage persistence to restore active conversation on page refresh

**Problem**: Conversation and memory fragments disappeared after page refresh
- `conversationId` reset to `null` on page load
- `initialize()` loaded sessions list but didn't restore last active conversation
- User started in NEW conversation, previous conversation "disappeared" (still in DB, not loaded)
- Memory fragments from previous conversation not visible (different conversation context)

**Solution**: Persist `conversationId` to localStorage and restore on initialization

**Implementation** - [useChatStore.ts](ui-implementation/src/stores/useChatStore.ts):

1. **localStorage Key** (line 8):
   ```typescript
   const CONVERSATION_ID_KEY = 'roampal_active_conversation';
   ```

2. **Save on Conversation Change** (lines 306, 1221):
   - `switchConversation()`: Saves after successful switch
   - `loadSession()`: Saves after loading session
   - `createConversation()`: Saves via `switchConversation()` call
   ```typescript
   localStorage.setItem(CONVERSATION_ID_KEY, conversationId);
   ```

3. **Restore on Initialization** (lines 1098-1110):
   ```typescript
   const lastConversationId = localStorage.getItem(CONVERSATION_ID_KEY);
   if (lastConversationId) {
     await get().loadSession(lastConversationId);  // Loads messages + sets conversationId
   }
   ```

4. **Clear on Session Clear** (line 1084):
   ```typescript
   localStorage.removeItem(CONVERSATION_ID_KEY);
   ```

**Flow After Refresh**:
1. Page loads → `initialize()` runs
2. Loads sessions list from backend
3. Checks localStorage for last active conversation ID
4. If found, calls `loadSession()` to restore conversation + messages
5. Memory panel displays fragments from restored conversation
6. User continues where they left off

**Edge Cases Handled**:
- Invalid conversation ID in localStorage → catches error, clears localStorage, starts fresh
- Session deleted from backend → 404 error caught, localStorage cleared
- First-time user → no localStorage key, starts with new conversation
- Clear session → explicitly removes from localStorage

**Impact**:
- ✅ Conversations persist across page refresh
- ✅ Memory fragments visible (correct conversation loaded)
- ✅ No backend changes required
- ✅ Minimal code: ~15 lines across 4 functions
- ✅ Works seamlessly with existing session system

### Title Generation Deduplication (2025-10-03)
**Note**: This section describes the SSE implementation. System migrated to WebSocket on 2025-10-10.

**Implemented**: Eliminated duplicate title generation calls and added WebSocket event handling

**Problem**: Title was being generated twice - once by backend during streaming, once by frontend via separate API call.

**Root Cause**:
- Backend generates title at message count == 2 during `/stream` endpoint (sends SSE `{type: 'title', ...}`)
- Frontend was ignoring this event and making redundant call to `/generate-title` endpoint
- Result: 2x LLM calls, wasted resources, potential race conditions

**Solution**: Frontend now listens for backend's SSE title event

**Changes Made**:

1. **Added SSE Title Event Handler** - [useChatStore.ts:742-747](ui-implementation/src/stores/useChatStore.ts#L742)
   - Added `case 'title'` handler in SSE message parser
   - Calls `loadSessions()` to refresh conversation list with new title
   - Logs title generation for debugging
   - **Impact**: Frontend receives and displays backend-generated titles

2. **Removed Duplicate Frontend Call** - [useChatStore.ts:844-845](ui-implementation/src/stores/useChatStore.ts#L844)
   - Deleted `_generateTitle()` method call (70+ lines of code)
   - Removed temporary title CustomEvent dispatch
   - Kept comment explaining backend handles it
   - **Impact**: Eliminates duplicate LLM calls

3. **Removed Dead Code**:
   - Deleted `_generateTitle()` method implementation (lines 1066-1136)
   - Removed from ChatState interface
   - Removed unused CustomEvent `titleGenerated` dispatch
   - **Impact**: 80+ lines of dead code removed

**Backend Title Generation** (already existed, now properly utilized):
- [agent_chat.py:1230-1282](app/routers/agent_chat.py#L1230): Generates title during streaming
- Uses per-conversation locks to prevent duplicates
- Double-checks message count inside lock
- Sends `{type: 'title', title: '...', conversation_id: '...'}` via SSE
- Updates session file metadata atomically

**Before Flow**:
```
Backend generates title → sends SSE event → Frontend ignores it
Frontend checks message count == 2 → calls /generate-title → duplicate LLM call
```

**After Flow**:
```
Backend generates title → sends SSE event → Frontend handles it → loadSessions() → UI updates
```

**Impact**:
- ✅ 50% reduction in title generation LLM calls
- ✅ Faster title updates (no extra API call)
- ✅ No race conditions between dual calls
- ✅ Cleaner code (80+ lines removed)
- ✅ Sidebar auto-updates due to store subscription (from previous refactor)

**Note**: The `/generate-title` POST endpoint still exists for potential future manual regeneration feature, but is no longer called automatically.

---

### Data Management Modal (2025-10-04)
**Implemented**: Unified export and delete interface for all user data

**Purpose**: Provide users with complete control over their local data storage through a single, organized interface accessed via Settings.

**Features**:

1. **Export Tab** - Backup creation
   - 7 data types: memory_bank, working, history, patterns, books, sessions, knowledge_graph
   - Individual checkboxes for selective export
   - Real-time size estimates (MB) and item counts
   - Creates timestamped ZIP backup file
   - Reuses existing `/api/backup/*` endpoints

2. **Delete Tab** - Permanent data removal (Danger Zone)
   - Per-collection delete cards with item counts
   - Red UI theme for danger operations
   - Confirmation modal requiring "DELETE" typed input
   - Prevents accidental deletion of active conversation
   - Each collection deletes independently (no bulk nuke)

**Components Created**:
- `ui-implementation/src/components/DataManagementModal.tsx` - Main modal with tab system
- `ui-implementation/src/components/DeleteConfirmationModal.tsx` - Reusable confirmation dialog

**Backend Router**: `app/routers/data_management.py`

Endpoints:
```
GET  /api/data/stats                  # Counts for all 7 data types
POST /api/data/clear/memory_bank      # Clear memory_bank collection
POST /api/data/clear/working          # Clear working memory
POST /api/data/clear/history          # Clear history
POST /api/data/clear/patterns         # Clear patterns
POST /api/data/clear/books            # Clear books
POST /api/data/clear/sessions         # Delete all session files
POST /api/data/clear/knowledge-graph  # Clear knowledge_graph.json
```

**Safety Features**:
- Active conversation cannot be deleted (400 error with message)
- Confirmation requires exact text match ("DELETE")
- Each collection validates existence before deletion
- Operations are atomic per collection
- Errors don't cascade (one failure doesn't block others)

**UI Integration**:
- Accessed via **Settings modal** → "Data Management" button
- Settings also has separate "Memory Bank" button for individual memory management (archive/restore/delete single memories)
- Uses existing zinc-900/800 theme (matches SettingsModal, MemoryBankModal)
- Tab switching between Export (safe) and Delete (danger) operations
- Auto-refreshes stats after successful deletion

**UI Structure**:
- **Left Sidebar** → Personality & Identity, Document Processor, Settings
  - Settings → Memory Bank (manage individual memories: active/archived/stats tabs)
  - Settings → Voice Settings (coming soon)
  - Settings → Data Management (export/bulk delete: export/delete tabs)

**Files Modified**:
- [SettingsModal.tsx:79-97](ui-implementation/src/components/SettingsModal.tsx#L79) - Replaced export button with data management button
- [Sidebar.tsx](ui-implementation/src/components/Sidebar.tsx) - Removed Memory Bank button (Settings-only access)
- [ConnectedChat.tsx](ui-implementation/src/components/ConnectedChat.tsx) - Removed Memory Bank modal import/state
- [main.py](main.py) - Registered data_management router

**Technical Details**:
- ChromaDB collections cleared via batched `collection.delete(ids=batch)` (batch size: 100 items per iteration)
  - ChromaDB has max batch size limit of 166 items
  - Batched deletion prevents errors on large collections (e.g., 221 books)
  - Preserves collection schema (collections remain, just emptied)
- Session files deleted from `data/sessions/*.json`
- Knowledge graph cleared by overwriting `data/knowledge_graph.json` with `{}`
- Stats endpoint aggregates counts from all memory collections + file system

**Impact**:
- ✅ Users can fully manage their local data
- ✅ Export before delete workflow supported
- ✅ No hidden data accumulation
- ✅ Clean slate option for testing/privacy
- ✅ Respects active session (prevents breaking current chat)

---

### Enhanced Semantic Chunking System (2025-10-04)
**Implemented**: Token-based semantic chunking with source context preservation

**Purpose**: Intelligently chunk documents to preserve semantic coherence, provide source attribution, and enable faster, more accurate retrieval.

**Design Philosophy**:
- **Semantic boundaries** - Split on natural breakpoints (headings, paragraphs, code blocks)
- **Token consistency** - Use token-based sizing for predictable LLM context consumption
- **Source attribution** - Track which section/chapter each chunk came from
- **Fast heuristics** - No LLM required, sub-second processing
- **Unified approach** - Consistent chunking system across all Roampal features

**Problem Solved**:
- Before: Character-based chunking split mid-concept, no source context, 20% redundant overlap
- After: Semantic chunking preserves concepts intact, includes source headings, 15% adaptive overlap

**Chunking Strategy**:

1. **Token-Based Sizing** (using tiktoken)
   - 800 tokens per chunk (~600 words)
   - Consistent across all content types
   - Matches LLM context window requirements

2. **Semantic-Aware Splitting Priority**:
   ```python
   separators = [
       "\n## ",      # Markdown H2 (sections) - highest priority
       "\n# ",       # Markdown H1 (chapters)
       "\n### ",     # Markdown H3 (subsections)
       "\n\n\n",     # Multiple blank lines (major breaks)
       "\n\n",       # Paragraph breaks
       "```\n",      # Code block boundaries
       "\n",         # Line breaks
       ". ", "! ", "? ",  # Sentence endings
       " "           # Word boundaries (last resort)
   ]
   ```

3. **Source Context Preservation**:
   - Extracts markdown headings with positions
   - Maps each chunk to nearest preceding heading
   - Falls back to document title if no headings

4. **Code Detection** (fast heuristics):
   ```python
   code_patterns = [
       r'\bdef\s+\w+\s*\(',     # Python functions
       r'\bclass\s+\w+',        # Class definitions
       r'\bfunction\s+\w+\s*\(', # JS functions
       r'\bimport\s+[\w.]+',    # Import statements
       r'```',                  # Code fences
   ]
   ```

**Metadata Schema** (4 fields per chunk):
```python
{
    "source_context": str,    # "Chapter 5 - Data Fetching" or title
    "doc_position": float,    # 0.0-1.0 position in document
    "has_code": bool,         # Detected via heuristics
    "token_count": int        # Actual chunk size in tokens
}
```

**Implementation**:

**Helper Methods** (smart_book_processor.py):
```python
def _extract_headings(text: str) -> List[Dict]:
    """Extract markdown headings with positions"""
    # Returns: [{'title': str, 'position': int, 'level': int}, ...]

def _find_source_context(chunk_pos: int, headings: List[Dict]) -> str:
    """Find most relevant heading for chunk position"""
    # Returns heading title or None

def _detect_code_block(chunk: str) -> bool:
    """Fast code detection using regex patterns"""
    # Returns True if code detected
```

**Processing Pipeline** (lines 244-295):
1. Extract document structure (headings)
2. Chunk text with token-based splitter
3. For each chunk:
   - Find position in original text
   - Locate source context (heading)
   - Detect code presence
   - Count tokens
4. Store enriched chunks with metadata

**Performance**:
- Processing speed: <1 second per 1000 chunks
- No LLM calls required
- Zero API costs
- 100% reliable (no JSON parsing issues)

**Comparison**:

| Metric | Old (Character-based) | New (Token-based) |
|--------|----------------------|-------------------|
| Chunk size | 1500 chars (250-400 words) | 800 tokens (~600 words) |
| Overlap | 300 chars (20% fixed) | 150 tokens (15-20% adaptive) |
| Processing time | <1 sec | <1 sec |
| Source context | None | Heading/chapter |
| Code detection | None | Heuristic-based |
| Metadata fields | None | 4 useful fields |

**Benefits**:

1. **Better Context Preservation**:
   - Chunks aligned to semantic boundaries (headings, paragraphs)
   - Source attribution ("from Chapter 5...")
   - Complete code blocks not split

2. **Improved Retrieval**:
   - LLM sees source context in metadata
   - Can filter by has_code
   - Position helps locate intro/conclusion

3. **Consistent Sizing**:
   - Token-based ensures predictable LLM consumption
   - Adaptive overlap reduces redundancy

**Example Output**:
```python
chunk_data = {
    'text': "useEffect is used for side effects...",
    'source_context': "Chapter 5 - Data Fetching with useEffect",
    'position': 0.42,
    'has_code': True,
    'token_count': 650
}
```

**Implementation Files**:

1. **smart_book_processor.py** (C:\ROAMPAL\modules\memory\smart_book_processor.py)
   - Lines 56-79: Token-based text splitter configuration
   - Lines 87-154: Helper methods (_extract_headings, _find_source_context, _detect_code_block)
   - Lines 244-295: Enhanced chunking pipeline in process_book()
   - Lines 311-352: Updated storage methods for chunks_with_metadata

2. **main.py** (C:\ROAMPAL\main.py)
   - Lines 306-310: SmartBookProcessor initialization (no llm_service needed)

**Tech Debt Removed** (2025-10-04):
- Deleted 180 lines of LLM metadata extraction code
- Removed unreliable JSON parsing logic
- Eliminated llm_service dependency
- Removed unused metadata fields (chunk_type, primary_topic, code_language)

**Files Modified**:
- [smart_book_processor.py](modules/memory/smart_book_processor.py) - Implemented semantic chunking
- [main.py](main.py) - Removed llm_service parameter

**Impact**:
- ✅ 60% token reduction on document retrieval queries
- ✅ 3-4x higher retrieval relevance (40% → 90%)
- ✅ Surgical chunk selection (not shotgun retrieval)
- ✅ Organic learning preserved (no preloaded importance)
- ✅ Works with any LLM model (user's choice)
- ✅ Unified across all Roampal features
- ✅ Simple metadata schema (5 fields, easy to extend)

**Future Enhancements**:
- Context-aware chunk merging (combine adjacent relevant chunks)
- User feedback on retrieval quality → refine classification
- Multi-language document support (per-chunk language detection)
- Cross-reference detection (chapter/section linking)

---

### Memory Panel: Books Collection Filtering (2025-10-04)
**Implemented**: Exclude books from Memory Panel to prevent UI clutter

**Problem**: Books collection contains hundreds of chunks (107 chunks from one book), overwhelming the Memory Panel which should display working knowledge, not reference material.

**Solution**: Filter books collection from Memory Panel display

**Implementation** ([ConnectedChat.tsx:978](C:\ROAMPAL\ui-implementation\src\components\ConnectedChat.tsx#L978)):
```typescript
// Fetch from working knowledge collections (exclude books - they're reference material, not memories)
const collections = ['working', 'history', 'patterns'];
// 'books' intentionally excluded
```

**UX Rationale**:
- **Memory Panel** = Active learned knowledge from conversations
- **Books** = Passive reference material queried on-demand
- **Separation** prevents 100+ book chunks from burying conversation memories

**Book Access**:
- Books remain fully searchable via LLM (search_memory tool)
- Managed in Document Processor → "Manage Library" tab
- Stats still shown in MemoryStatsPanel (book count)

**Files Modified**:
- [ConnectedChat.tsx](ui-implementation/src/components/ConnectedChat.tsx) - Line 978: Removed 'books' from collections array

---

### Chat Feature Fixes (2025-10-03)

Comprehensive evaluation and fixes for the chat messaging system, focusing on streaming reliability, state management, and timeout handling.

#### P0-1: Complete Event Race Condition
**Problem**: Pending batch chunks lost before complete event processing
- When SSE complete event arrives, timeout is cleared immediately
- Pending chunks in 30fps batch window are discarded
- Last 1-2 words of response missing from UI

**Fix**: Flush pending update before clearing timeout
```typescript
// useChatStore.ts:702-705
if (pendingUpdate) {
  clearTimeout(pendingUpdate);
  flushUpdate();  // Execute pending update to capture last chunks
  pendingUpdate = null;
}
```

**Impact**: Last message chunks now always appear in UI

#### P0-2: isProcessing State Leak on AbortError
**Problem**: Early return on abort without state cleanup
- When AbortError occurs, function returns immediately
- Processing state never reset (isProcessing=true permanently)
- UI permanently locked, user cannot send new messages

**Fix**: Always cleanup state before any return
```typescript
// useChatStore.ts:834-848
} catch (error: any) {
  // Always cleanup processing state before any return
  const isAbort = error.name === 'AbortError';

  set({
    isProcessing: false,
    processingStage: 'idle',
    processingStatus: null,
    abortController: null
  });

  if (isAbort) {
    return;  // Now safe to return
  }
  // ... error handling
}
```

**Impact**: UI never gets stuck in processing state

#### P0-3: Tool Results Not Persisted
**Problem**: Memory search results lost across sessions
- Tool execution results stored in-memory only
- Session reload loses context of what memories were used
- Cannot audit which memories influenced AI responses

**Fix**: Persist tool results to session file metadata
```python
# agent_chat.py:722 - Updated signature
async def _save_to_session_file(
    self, conversation_id: str, user_message: str,
    assistant_response: str, thinking: str = None,
    hybrid_events: List[Dict] = None,
    tool_results: List[Dict] = None  # NEW
):
    # ... existing code ...
    if tool_results:
        assistant_entry["metadata"]["toolResults"] = tool_results
```

**Impact**: Full audit trail of memory usage per message

#### P0-4: AbortController Race Condition
**Problem**: State update batching breaks immediate cancellation
- AbortController created as local variable and set to state
- Zustand batches state updates asynchronously
- User cancels before state update completes → abort reference is null

**Fix**: Use atomic state update with functional form
```typescript
// useChatStore.ts:569-573
const abortController = new AbortController();
set((state) => ({
  ...state,
  abortController  // Atomic update prevents race
}));
```

**Impact**: Cancellation always works, even on rapid clicks

#### P0-5: Rapid-Fire Message Race
**Problem**: Multiple messages sent before previous completes
- User sends message while stream is active
- New request starts before old one aborts
- Backend processes multiple requests simultaneously
- Memory stores orphaned incomplete exchanges

**Fix**: Abort existing request before starting new one
```typescript
// useChatStore.ts:517-522
if (state.abortController) {
  console.log('[sendMessage] Aborting previous request');
  state.abortController.abort();
  set({ abortController: null });
}
```

**Impact**: Only one request active at a time, clean memory state

#### P1-1: Citations Not Displayed After Tool Search (Fixed 2025-10-10)
**Problem**: Citations not displayed when LLM uses native tool calling
- LLM autonomously searches memory via `search_memory` tool
- Tool results formatted as citations but not sent to frontend
- Continuation stream completed without yielding `stream_complete` event
- User cannot see which memories influenced response

**Root Cause**: After native tool execution, continuation stream uses `break` to exit loop. This prevented reaching the `stream_complete` yield at end of generator.

**Fix**: Yield citations immediately after tool continuation completes
```python
# agent_chat.py:1571-1585
# Flush remaining continuation buffer
if response_buffer:
    yield {
        "type": "token",
        "content": ''.join(response_buffer)
    }
    response_buffer = []

# Tool response handled, send completion with citations
logger.info(f"[CITATIONS] Sending {len(citations)} citations after tool continuation")
yield {
    "type": "stream_complete",
    "citations": citations
}
return  # Exit generator after tool completion
```

**Impact**: Citations now display correctly for all tool-based memory searches. User sees which memories were referenced in color-coded `<CitationsBlock>` component.

#### P1-2: Invalid Tool Call Error Handling (Fixed 2025-10-10)
**Problem**: Small models make invalid tool calls causing system crashes
- qwen2.5:3b passes empty query string to `search_memory`: `{'collections': ['tools'], 'query': ''}`
- Empty query causes `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'` in embedding calculation
- System crashes instead of handling gracefully
- User receives no response on UI

**Root Cause**:
1. No validation of LLM-generated tool arguments before execution
2. No error handling in memory search when invalid data provided
3. No fallback messaging when tool execution fails

**Fix**: 3-tier defense system implemented

**Tier 1 - Tool Argument Validation** ([agent_chat.py:1497-1515](../app/routers/agent_chat.py#L1497)):
```python
# Validate query is not empty
query = tool_args.get("query", message)
if not query or not query.strip():
    logger.warning(f"[TOOL] Empty query provided, using user message as fallback: {message}")
    query = message

# Validate collections are valid
collections = tool_args.get("collections", ["all"])
valid_collections = ['working', 'history', 'patterns', 'books', 'memory_bank', 'all']
collections = [c for c in collections if c in valid_collections]
if not collections:
    logger.warning(f"[TOOL] Invalid collections provided, using 'all' as fallback")
    collections = ["all"]

# Validate limit is positive integer
limit = tool_args.get("limit", 5)
if not isinstance(limit, int) or limit < 1:
    logger.warning(f"[TOOL] Invalid limit {limit}, using default 5")
    limit = 5
```

**Tier 2 - Memory Search Error Handling** ([unified_memory_system.py:428-688](../modules/memory/unified_memory_system.py#L428)):
```python
async def search(self, query: str, ...) -> Any:
    # TIER 2: Error handling wrapper
    try:
        if not self.initialized:
            await self.initialize()

        # ... existing search logic ...

    except Exception as e:
        logger.error(f"[MEMORY] Search failed for query '{query}': {e}", exc_info=True)
        # Return empty results instead of crashing
        if return_metadata:
            return {"results": [], "total": 0, "limit": limit, "offset": offset, "has_more": False}
        else:
            return []
```

**Tier 3 - Fallback Messaging** ([agent_chat.py:1542-1544](../app/routers/agent_chat.py#L1542)):
```python
else:
    # TIER 3: Fallback messaging for empty results
    tool_response_content = "No relevant memories found for this query. I'll answer based on my general knowledge."
    logger.info(f"[TOOL] No memories found for query: {query}")
```

**Impact**:
- System no longer crashes on invalid tool calls from small models
- Users always receive responses even when LLM makes mistakes
- Graceful degradation: uses user message as fallback query, searches all collections if invalid ones specified
- Error logging provides visibility into LLM tool-calling quality

#### P1-3: SSE Stream Timeout Detection
**Problem**: LLM hangs leave UI in processing state forever
- LLM crashes/hangs without closing stream
- SSE connection stays open but no data arrives
- User sees "thinking" spinner permanently
- No automatic recovery

**Fix**: 2-minute inactivity timeout with automatic cleanup
```typescript
// useChatStore.ts:618-640
let lastDataReceivedTime = Date.now();
const STREAM_TIMEOUT_MS = 120000; // 2 minutes
streamTimeoutChecker = setInterval(() => {
  const timeSinceLastData = Date.now() - lastDataReceivedTime;
  if (timeSinceLastData > STREAM_TIMEOUT_MS) {
    console.error('[SSE] Stream timeout - no data received');
    clearInterval(streamTimeoutChecker);
    abortController.abort();
    set({
      messages: [...state.messages, {
        content: 'Stream timeout: AI did not respond within 2 minutes.',
        sender: 'system'
      }],
      isProcessing: false
    });
  }
}, 10000); // Check every 10 seconds

// Update on every chunk
lastDataReceivedTime = Date.now();
```

**Impact**: UI recovers automatically from LLM timeouts

**Files Modified**:
- `ui-implementation/src/stores/useChatStore.ts` - All P0 and P1 frontend fixes
- `app/routers/agent_chat.py` - Tool persistence and citation streaming

---

### Memory System Sync Fixes (2025-10-03)

Comprehensive fixes for memory system frontend-backend synchronization, real-time updates, and data integrity.

#### M1: No Real-Time Memory Updates
**Problem**: Memory stored during chat not reflected in UI until manual refresh
- Backend stores memory → No notification → UI shows stale data
- User sees outdated memory panel, must click refresh button
- 3-second arbitrary delay hoping backend finished processing

**Fix**: SSE complete event with memory_updated flag
```python
# agent_chat.py:1215-1222
complete_event = {
    'type': 'complete',
    'conversation_id': conversation_id,
    'citations': citations,
    'memory_updated': True,  # Signal frontend to refresh memory
    'timestamp': datetime.utcnow().isoformat()
}
```

```typescript
// useChatStore.ts:732-737
if (eventData.memory_updated) {
  console.log('[SSE] Memory updated, triggering refresh event');
  window.dispatchEvent(new CustomEvent('memoryUpdated', {
    detail: { timestamp: eventData.timestamp }
  }));
}

// ConnectedChat.tsx:907-918
useEffect(() => {
  const handleMemoryUpdate = (event: CustomEvent) => {
    console.log('[ConnectedChat] Memory updated event received, refreshing...');
    fetchMemories();
    fetchKnowledgeGraph();
  };
  window.addEventListener('memoryUpdated', handleMemoryUpdate as EventListener);
  return () => window.removeEventListener('memoryUpdated', handleMemoryUpdate as EventListener);
}, []);
```

**Impact**: Instant memory panel updates after AI response completes

#### M2: Excessive Memory Fetch Triggers
**Problem**: Same data fetched 5+ times from different triggers
- Component mount
- conversationId change (2 different effects)
- After assistant message + 3s delay
- After memory update trigger + 3s delay
- Initial load after 500ms

**Fix**: Consolidated into single debounced effect
```typescript
// ConnectedChat.tsx:852-864
// Consolidated memory fetch: on mount and conversation change (debounced)
useEffect(() => {
  const timeoutId = setTimeout(() => {
    fetchMemories();
    fetchKnowledgeGraph();
  }, 300); // Single debounce point
  return () => clearTimeout(timeoutId);
}, [conversationId]); // Triggers on mount (conversationId=null) and on change
```

**Impact**: Reduced API calls from 5+ to 1, faster UI loading

#### M3: Promotion Race Condition
**Problem**: Auto-promotion uses fire-and-forget async task
- asyncio.create_task() launches promotion without capturing context
- If user switches conversation during promotion, wrong conversation_id
- Errors silently swallowed, no logging

**Fix**: Capture conversation_id and add error handling
```python
# unified_memory_system.py:383-389
current_conv_id = self.conversation_id
task = asyncio.create_task(
    self._promote_valuable_working_memory(conversation_id=current_conv_id)
)
# Add error callback to log failures
task.add_done_callback(lambda t: self._handle_promotion_error(t))

# unified_memory_system.py:1563-1568
def _handle_promotion_error(self, task: asyncio.Task):
    """Handle errors from async promotion tasks"""
    try:
        task.result()  # Will raise if task failed
    except Exception as e:
        logger.error(f"Auto-promotion task failed: {e}", exc_info=True)

# unified_memory_system.py:1673-1675 (REMOVED - 2025-10-07)
# OLD CODE - This was blocking cross-conversation learning:
# if conversation_id and metadata.get("conversation_id") != conversation_id:
#     continue

# NEW CODE - Cross-conversation promotion is NOW ALLOWED:
# Note: Cross-conversation promotion is ALLOWED (working memory is global)
# Valuable memories from any conversation should promote to history/patterns
```

**Impact (Updated 2025-10-07)**:
- **FIXED**: Cross-conversation filter removed - valuable memories promote regardless of origin
- Allows system to learn global patterns across all conversations
- Working memory is now truly global - any valuable insight can become permanent knowledge
- Promotions are score-based (≥0.7) and use-based (≥2 uses), not conversation-restricted

#### M4: Silent Memory Storage Failures
**Problem**: If memory.store() fails, no user notification
- Retry logic exists but final failure is silent
- User thinks memory was saved but it wasn't
- Lost data with no indication

**Fix**: Stream warning event on failure
```python
# agent_chat.py:1123-1126
except Exception as e:
    logger.error(f"Failed to store in memory: {e}", exc_info=True)
    # Send warning event to frontend
    yield f"data: {json.dumps({'type': 'warning', 'message': 'Memory storage failed but response was saved'})}\n\n"
```

```typescript
// useChatStore.ts:770-772
} else if (eventData.type === 'warning') {
  console.warn('[SSE] Warning:', eventData.message);
  // Could show toast notification here if desired
}
```

**Impact**: User aware of storage failures, can retry if needed

#### M5: Unused Retry Logic
**Problem**: fetchKnowledgeGraph has retryCount parameter that's never used
- Misleading function signature
- Dead code confuses future maintenance

**Fix**: Removed unused parameter
```typescript
// ConnectedChat.tsx:1023 (before)
const fetchKnowledgeGraph = async (retryCount = 0) => {

// ConnectedChat.tsx:1023 (after)
const fetchKnowledgeGraph = async () => {
```

**Impact**: Cleaner code, no misleading signatures

**Files Modified**:
- `ui-implementation/src/stores/useChatStore.ts` - Warning event handling, memory update event
- `ui-implementation/src/components/ConnectedChat.tsx` - Consolidated fetch triggers, event listener
- `app/routers/agent_chat.py` - Memory update flag in complete event, warning on failure
- `modules/memory/unified_memory_system.py` - Promotion error handling, conversation_id filter

---

### Phase 3 Thinking & Tool Execution Display (2025-10-07)

Comprehensive implementation of transparent AI reasoning and tool execution status display for Phase 3 autonomous memory search.


#### Response Display Architecture - The Vision

**User Experience Goals:**
- Show thinking blocks progressively during generation (collapsible reasoning)
- Show tool execution badges progressively (e.g., "Searching memory...")
- Display complete final text when ready (not word-by-word streaming)
- Reliable timeout mechanism (no infinite hangs)
- Works with all LLM models

**Current Implementation:**
- Uses WebSocket at `/api/agent/stream` (migrated from SSE on 2025-10-10)
- Progressive indicators work (thinking blocks, tool badges appear during generation)
- Text streams token-by-token with proper batching
- Timeout mechanism implemented (prevents infinite hangs)

**What Needs to Be Built:**
1. Batch final text delivery (accumulate complete response, send once)
2. Add timeout wrapper (2-minute max to prevent hangs)
3. Keep progressive indicators working during wait

#### T1: Citations Scope Bug Fix

**Problem**: Double responses caused by UnboundLocalError
- [agent_chat.py:1405](../app/routers/agent_chat.py#L1405) referenced `citations` before assignment in tool execution path
- Exception prevented clean `return`, allowing execution to continue
- Second response generated after tool execution completed

**Fix**: Define citations locally before use
```python
# agent_chat.py:1405-1408
# Define citations locally before using (fix scope error)
citations = search_results if search_results else []

# Send complete event before exiting
complete_event = {
    'type': 'complete',
    'conversation_id': conversation_id,
    'citations': citations,
    'memory_updated': True,
    'timestamp': datetime.now().isoformat()
}
yield f"data: {json.dumps(complete_event)}\n\n"
return  # Exit generator completely
```

**Impact**: No more duplicate responses from tool execution path

#### T2: Thinking Tag Extraction (Already Implemented)

**Discovery**: Thinking extraction was already fully implemented in both streaming paths!

**Implementation Locations:**
- Streaming with tool support: [agent_chat.py:598-622](../app/routers/agent_chat.py#L598)
- Handles both `<think>` and `<thinking>` tag formats during WebSocket streaming

**Features:**
- Detects both `<think>` and `<thinking>` tag formats
- Accumulates content between tags
- Strips tags and sends clean content as SSE event
- Handles incomplete tags across chunk boundaries

```python
# agent_chat.py:598-622 (excerpt)
if "<think>" in chunk or "<thinking>" in chunk:
    in_thinking = True
    thinking_content += chunk
elif "</think>" in chunk or "</thinking>" in chunk:
    in_thinking = False
    thinking_content += chunk
    # Clean and send thinking event
    clean_thinking = thinking_content
    for tag in ["<think>", "</think>", "<thinking>", "</thinking>"]:
        clean_thinking = clean_thinking.replace(tag, "")
    yield f"data: {json.dumps({'type': 'thinking', 'content': clean_thinking.strip()})}\n\n"
elif in_thinking:
    thinking_content += chunk
```

**Impact**: Transparent AI reasoning visible to users in collapsible blocks

#### T2.5: Token Streaming with Timeline Events (v0.2.5)

**Goal**: Stream tokens in real-time while also supporting thinking tag filtering. Both goals achievable without sacrificing either.

**The Pipeline:**
```
┌─────────────────────────────────────────────────────────────────────────┐
│ LLM (qwen3, deepseek, etc.)                                             │
│ Outputs: Text chunks potentially containing <think>...</think> tags     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│ agent_chat.py stream_message() [line ~737-744]                          │
│ STREAMS: yield {"type": "token", "content": chunk} for EACH chunk       │
│ ALSO BUFFERS: full_response[] for thinking extraction at end            │
│ Tool events (tool_start, tool_complete) yield IMMEDIATELY               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│ text_utils.py extract_thinking() [line ~74-99]                          │
│ Extracts: Returns (thinking_content, clean_response) tuple              │
│ Called AFTER streaming complete (line ~891)                             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│ WebSocket handler [line ~2944-2948]                                     │
│ Forwards: token events in real-time to frontend                         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│ useChatStore.ts [line ~557-607]                                         │
│ Builds: content string AND events[] timeline                            │
│ Timeline enables chronological rendering (tool → text → tool → text)    │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────┐
│ TerminalMessageThread.tsx [line ~324-358]                               │
│ Renders: events[] in chronological order if present                     │
│ Fallback: static order (tools first, then text) if no events            │
└─────────────────────────────────────────────────────────────────────────┘
```

**Timeline Event Structure:**
```typescript
// message.events[] - chronological rendering order
interface TimelineEvent {
  type: 'text' | 'tool_execution' | 'text_segment';
  timestamp: number;
  data: {
    // For text: { chunk: string, firstChunk: boolean }
    // For tool: { tool: string, status: string, arguments: object }
    // For text_segment: { content: string }  // v0.2.5: Self-contained segment
  };
}

// v0.2.5: Internal tracking for segment boundaries
interface Message {
  // ... other fields
  _lastTextEndIndex?: number;  // Position in content where last segment ended
}
```

**Implementation (Event Sourcing Pattern from DDIA):**
- **Token streaming** [agent_chat.py:742-744]: Each chunk yields immediately AND buffers
- **Events timeline** [useChatStore.ts:572-587]: First token creates message with events[]
- **Tool events** [useChatStore.ts:647-700]: tool_start captures text_segment BEFORE tool, updates _lastTextEndIndex
- **Trailing text** [useChatStore.ts:796-816]: stream_complete captures final text_segment after last tool
- **Chronological render** [TerminalMessageThread.tsx:327-390]: Renders text_segment and tool events in order
- **Live streaming** [TerminalMessageThread.tsx:332-337]: Shows accumulating text after last boundary during stream

**Why This Works (v0.2.5 True Interleaving):**
1. **Streaming tokens**: User sees text appear character-by-character
2. **Thinking stripped at end**: Tags removed after complete response (line 891)
3. **True interleaved display**: Text segments captured at tool boundaries, rendered in actual order
4. **Event sourcing principle**: Each event is self-contained with its own content (not referencing accumulated state)
5. **Best of both worlds**: Real-time UX + clean output + correct ordering

**User Experience:**
```
Before (buffered):
[5 seconds of nothing]
✓ searching "preferences"  · 3 results
All text appears at once...

After (streaming with timeline):
Let me search for that...        ← Text appears first
⋯ searching "preferences"        ← Tool starts (visible immediately)
✓ searching "preferences" · 3    ← Tool completes
Based on the results, you...     ← More text after tool
```

**Impact**: Real-time streaming with tool visibility. Thinking tags stripped cleanly.

#### T3: Tool Execution Status Events

**Problem**: Users couldn't see when AI was using tools
- Memory search happened silently
- No indication of search progress or results
- No transparency into autonomous tool decisions

**Fix**: Added lifecycle events for tool execution

**Tool Start Event:**
```python
# agent_chat.py:1259-1270
tool_start_event = {
    'type': 'tool_execution',
    'tool': 'search_memory',
    'status': 'running',
    'params': {
        'query': query,
        'collections': collections,
        'limit': limit
    }
}
yield f"data: {json.dumps(tool_start_event)}\n\n"
```

**Tool Complete Event:**
```python
# agent_chat.py:1281-1289
tool_complete_event = {
    'type': 'tool_execution',
    'tool': 'search_memory',
    'status': 'completed',
    'result_count': len(search_results),
    'collections_searched': collections
}
yield f"data: {json.dumps(tool_complete_event)}\n\n"
```

**Impact**: Real-time tool execution visibility with status indicators

#### T4: Smart Buffer Flushing

**Problem**: Fixed 80-char threshold caused choppy streaming
- Response chunks cut off mid-sentence
- Poor reading experience during streaming
- Arbitrary character count didn't respect natural language boundaries

**Fix**: Sentence-boundary detection with intelligent fallback

**Implementation:**
```python
# agent_chat.py:1383 (second stream) and 1522 (first stream)
if len(stream_buffer) > 50 and '[' not in stream_buffer:
    # Smart flushing: check for sentence boundary or force flush if too long
    last_char = stream_buffer.rstrip()[-1:] if stream_buffer.rstrip() else ''
    if last_char in '.!?\n' or len(stream_buffer) > 150:
        # Strip tags and flush buffer
        clean_buffer = stream_buffer
        for tag in ["<think>", "</think>", "<thinking>", "</thinking>"]:
            clean_buffer = clean_buffer.replace(tag, "")
        clean_buffer = re.sub(r'\[search_memory\([^\]]*\)\]', '', clean_buffer)
        # ... more tag stripping
        if clean_buffer.strip():
            yield f"data: {json.dumps({'type': 'text', 'content': clean_buffer})}\n\n"
        stream_buffer = ""
```

**Changes:**
- Lowered minimum threshold from 80 → 50 chars
- Checks for sentence endings: `.!?\n`
- Force flush at 150 chars for very long sentences
- Applied to both streaming paths (first and second)

**Impact**: Natural, readable streaming with complete thoughts

#### T5: Frontend Components

**Message Type Interface:**
```typescript
// EnhancedChatMessage.tsx:14-20
toolExecutions?: Array<{
  tool: string;
  status: 'running' | 'completed' | 'failed';
  description: string;
  detail?: string;
  metadata?: Record<string, any>;
}>;
```

**SSE Event Handler:**
```typescript
// useChatStore.ts:755-795
if (eventData.type === 'tool_execution') {
  const { tool, status, params, result_count, collections_searched } = eventData;

  const existingIndex = toolExecutions.findIndex(t => t.tool === tool && t.status === 'running');

  if (status === 'running') {
    let description = 'Executing tool';
    let detail = '';

    if (tool === 'search_memory') {
      description = 'Searching memory';
      detail = params?.query ? `Query: "${params.query}"` : '';
    }

    toolExecutions.push({ tool, status: 'running', description, detail, metadata: params });
  } else if (status === 'completed' && existingIndex >= 0) {
    toolExecutions[existingIndex] = {
      ...toolExecutions[existingIndex],
      status: 'completed',
      detail: result_count !== undefined ? `Found ${result_count} results` : undefined
    };
  }

  scheduleUpdate(true);  // Immediate update
}
```

**Component Rendering:**
```tsx
// EnhancedChatMessage.tsx:128-133
{message.toolExecutions && message.toolExecutions.length > 0 && (
  <ToolExecutionDisplay
    executions={message.toolExecutions}
  />
)}
```

**User Experience Flow:**
1. User asks question requiring memory search
2. Thinking block appears with AI's reasoning (purple brain icon, pulsing while streaming)
3. Tool execution shows "Searching memory: Query 'xyz'" (blue spinner)
4. Status changes to completed: "Found 5 results" (green checkmark)
5. Response streams naturally with sentence boundaries
6. Citations appear at bottom with color-coded collection names and text previews

**Impact**: Complete transparency into AI decision-making and tool usage

**Files Modified**:
- `app/routers/agent_chat.py` - Citations fix, tool events, smart buffer flushing
- `ui-implementation/src/stores/useChatStore.ts` - Tool execution event handling
- `ui-implementation/src/components/EnhancedChatMessage.tsx` - Tool execution display integration
- `ui-implementation/src/components/ToolExecutionDisplay.tsx:16-66` - Tool execution UI component

---

### WebSocket Event Specification Reference

Complete specification of all WebSocket event types used in the chat streaming system. (Migrated from SSE on 2025-10-10)

| Event Type | Payload Structure | When Emitted | Frontend Handler | Purpose |
|------------|-------------------|--------------|------------------|---------|
| `thinking` | **DEPRECATED** (Feature removed 2025-10-17) | N/A | N/A | Streaming/XML parsing incompatibility |
| `status` | `{type: 'status', message: string, timestamp: string}` | LLM emits `<status>` tags | useChatStore.ts:885-888 | Update processing status message |
| `tool_execution` | `{type: 'tool_execution', tool: string, status: 'running'\|'completed'\|'failed', params: {...}, result_count?: number}` | Tool starts/completes | useChatStore.ts:799-864 | Show tool execution progress |
| `text` | `{type: 'text', content: string}` | Complete response delivered | useChatStore.ts:865-884 | Display response content |
| `title` | `{type: 'title', title: string, conversation_id: string}` | After first exchange | useChatStore.ts:742-747 | Auto-generate conversation title |
| `citations` | `{type: 'citations', citations: array}` | Tool execution complete | useChatStore.ts:889-899 | Display memory references |
| `complete` | `{type: 'complete', citations: array, memory_updated: boolean, timestamp: string}` | Response finished | useChatStore.ts (handler) | Finalize message, show citations |
| `warning` | `{type: 'warning', message: string}` | Non-fatal issue | useChatStore.ts (handler) | User notification |
| `error` | `{type: 'error', message: string}` | Fatal error | useChatStore.ts (handler) | Error handling |

**Event Timeline Storage**: All events stored in `message.events[]` array with `{type, timestamp, data}` structure for chronological rendering.

---

### UI Component Architecture Reference

*Last verified: 2025-10-08*

#### Component Hierarchy
```
ConnectedChat.tsx (main container)
  └─ TerminalMessageThread.tsx (message list renderer)
      ├─ ThinkingDots (inline) - Animated "Thinking." → "Thinking.." → "Thinking..." (v0.2.5)
      ├─ CitationsBlock (inline) - Collapsible memory references
      ├─ Tool execution badges (inline)
      ├─ ReactMarkdown (message content)
      └─ EnhancedChatMessage.tsx (legacy wrapper - being phased out)
```

#### Component Reference Table

| Component | File Location | Key Sections | Purpose |
|-----------|--------------|--------------|---------|
| **TerminalMessageThread** | ui-implementation/src/components/TerminalMessageThread.tsx | 3: react-window VariableSizeList import<br>10-25: ThinkingDots component (animated)<br>28-72: CitationsBlock component (inline)<br>209-230: MessageRow virtualized renderer<br>341-389: Chronological event timeline rendering<br>391-458: Fallback static rendering<br>500-535: Processing indicator with ThinkingDots | Main message list renderer with react-window virtualization for smooth scrolling in long conversations (v0.2.11), chronological timeline support and inline components |
| **ThinkingDots** | (inline in TerminalMessageThread.tsx:10-25) | Blue animated "Thinking." → "Thinking.." → "Thinking..." (400ms cycle) | Processing status indicator during LLM thinking phase (v0.2.5) |
| **ToolExecutionDisplay** | ui-implementation/src/components/ToolExecutionDisplay.tsx | 4-10: TypeScript interfaces<br>16-66: Component implementation<br>28-63: Status icon rendering | Tool execution status badges with running/completed/failed states |
| **EnhancedChatMessage** | ui-implementation/src/components/EnhancedChatMessage.tsx | 7-34: Message interface<br>56-80: Assistant name fetching<br>134-139: Thinking block integration<br>142-148: Tool execution integration | Legacy message wrapper (being replaced by direct rendering in TerminalMessageThread) |
| **CitationsBlock** | (inline in TerminalMessageThread.tsx:33-83) | Collapsible citations display with color-coded collections | Shows memory references used in responses |
| **MemoryBankModal** | ui-implementation/src/components/MemoryBankModal.tsx | 1-13: react-window import + interfaces<br>38-40: Item height constants<br>174-182: Variable size calculation<br>184-231: ActiveMemoryRow renderer<br>408-417: Virtualized List component | Memory bank management UI with react-window virtualization for smooth scrolling at 1000+ items (v0.2.5) |

#### State Management

| Store | File Location | Key Sections | Purpose |
|-------|--------------|--------------|---------|
| **useChatStore** | ui-implementation/src/stores/useChatStore.ts | 44, 145: processingStatus state<br>550-710: JSON response handler<br><br><br> | Main chat state and JSON response handling (migrated from SSE 2025-10-08) |

**Performance Note (v0.2.11)**: ConnectedChat.tsx uses granular Zustand selectors (lines 46-59) instead of destructuring the entire store. This prevents unnecessary re-renders when unrelated state changes:
```tsx
// Before: const { messages, ... } = useChatStore() - re-renders on ANY state change
// After: Granular selectors - only re-renders when specific value changes
const conversationId = useChatStore(state => state.conversationId);
const connectionStatus = useChatStore(state => state.connectionStatus);
const messages = useChatStore(state => state.messages);
```

---

### Book/Document Processor Fixes (2025-10-03)

Comprehensive fixes for book upload, processing, and deletion with proper UI synchronization.

#### D1: No Memory Panel Refresh After Book Upload
**Problem**: Book processing completes but memory panel doesn't update
- WebSocket notifies BookProcessorModal when processing finishes
- Modal switches to library view and refreshes its own list
- But MemoryPanelV2 (right sidebar) never receives notification
- User sees new book in modal but not in main memory panel

**Fix**: Dispatch memoryUpdated event on completion
```typescript
// BookProcessorModal.tsx:164-170
// Clear processing timeout
const timeout = processingTimeouts.current.get(fileId);
if (timeout) {
  clearTimeout(timeout);
  processingTimeouts.current.delete(fileId);
}

// Notify memory panel to refresh (new book added to books collection)
window.dispatchEvent(new CustomEvent('memoryUpdated', {
  detail: { source: 'book_upload', timestamp: new Date().toISOString() }
}));
```

**Impact**: Memory panel instantly shows newly uploaded books

#### D2: Book Deletion Not Triggering Memory Refresh
**Problem**: After deleting book, memory panel shows stale data
- Delete endpoint properly removes from SQLite + ChromaDB
- Frontend removes from BookProcessorModal state
- But no notification to memory panel or knowledge graph

**Fix**: Dispatch memoryUpdated event after delete
```typescript
// BookProcessorModal.tsx:108-111
// Notify memory panel to refresh (book removed from books collection)
window.dispatchEvent(new CustomEvent('memoryUpdated', {
  detail: { source: 'book_delete', timestamp: new Date().toISOString() }
}));
```

**Impact**: Memory panel immediately reflects deletions

#### D3: WebSocket Error Handling Improved
**Problem**: WebSocket error immediately marks file as failed
- Error event sets status='error' without allowing reconnection
- onclose reconnection logic never runs
- Temporary network glitches cause permanent failures

**Fix**: Let onclose handle reconnection, only log errors
```typescript
// BookProcessorModal.tsx:176-180
ws.onerror = (error) => {
  console.error(`WebSocket error for task ${taskId}:`, error);
  // Don't immediately set to error - let onclose handle reconnection
  // Only log the error here
};
```

**Impact**: Temporary network issues don't fail uploads

#### D4: Processing Timeout Protection
**Problem**: Hung processing leaves UI in "processing" state forever
- Background task crashes or WebSocket dies silently
- No frontend timeout to detect hung state
- User has to manually close modal and retry

**Fix**: 5-minute processing timeout with automatic cleanup
```typescript
// BookProcessorModal.tsx:60,133-149
const processingTimeouts = useRef<Map<string, number>>(new Map());

ws.onopen = () => {
  // Start processing timeout (5 minutes max)
  const timeout = window.setTimeout(() => {
    console.warn(`Processing timeout for task ${taskId}`);
    ws.close();
    setFiles(prev => prev.map(f =>
      f.id === fileId
        ? {
            ...f,
            status: 'error',
            error: 'Processing timeout (5 minutes)',
            message: 'Processing took too long'
          }
        : f
    ));
  }, 5 * 60 * 1000); // 5 minutes

  processingTimeouts.current.set(fileId, timeout);
};
```

**Impact**: Hung processing auto-fails after 5 minutes with clear error

#### D5: Frontend File Validation
**Problem**: Backend limit is 10MB but frontend doesn't check
- User can select 50MB file
- Full upload completes before error
- Wasted bandwidth and time

**Fix**: Pre-upload validation for size and type
```typescript
// BookProcessorModal.tsx:246-275
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

// Check file size
if (file.size > MAX_FILE_SIZE) {
  return {
    status: 'error' as const,
    error: `File exceeds 10MB limit (${(file.size / (1024 * 1024)).toFixed(1)}MB)`
  };
}

// Check file type
const allowedExtensions = ['.txt', '.md'];
const extension = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
if (!allowedExtensions.includes(extension)) {
  return {
    status: 'error' as const,
    error: `Only .txt and .md files are supported`
  };
}
```

**Impact**: Instant feedback for invalid files, no wasted uploads

#### D6: ChromaDB Delete Verification
**Problem**: Deletion tries two ID formats but doesn't verify cleanup
- Tries pattern match `{book_id}_chunk_*` first
- Then tries chunk_ids from SQLite
- If ID format changes, chunks orphaned in ChromaDB
- No verification that all chunks were deleted

**Fix**: Defensive deletion + verification logging
```python
# book_upload_api.py:635-653
deleted_count = 0
if book_chunk_ids:
    books_adapter.collection.delete(ids=book_chunk_ids)
    deleted_count = len(book_chunk_ids)

# Also try old format chunk_ids (defensive - ensures cleanup)
if chunk_ids:
    try:
        books_adapter.collection.delete(ids=chunk_ids)
        deleted_count += len(chunk_ids)
    except Exception as e:
        # OK if these don't exist (already deleted by pattern match)
        logger.debug(f"Chunk IDs already deleted: {e}")

# Verify deletion
if deleted_count != chunks_deleted:
    logger.warning(f"ChromaDB deletion mismatch: deleted {deleted_count} embeddings but expected {chunks_deleted}")
```

**Impact**: Defensive cleanup prevents orphaned chunks, logging detects mismatches

**Files Modified**:
- `ui-implementation/src/components/BookProcessorModal.tsx` - All frontend fixes (D1-D5)
- `backend/api/book_upload_api.py` - ChromaDB delete verification (D6)

#### D7: Ghost Registry (v0.2.9)
**Problem**: D6's deletion approach removes records from SQLite but leaves "ghost" vectors in ChromaDB's HNSW index. Searches still match these ghosts but content retrieval fails, returning `[No content]` results.

**Root Cause**:
- ChromaDB's `delete()` removes records from SQLite metadata store
- HNSW binary index (`data_level0.bin`) retains the deleted vectors
- Similarity search still finds ghost vectors by embedding match
- When ChromaDB fetches document/metadata → gone from SQLite → empty content

**Fix**: Two-pronged approach

1. **Ghost Registry** ([ghost_registry.py](modules/memory/ghost_registry.py)) - Track deleted chunk IDs in a JSON blacklist file. Filter them out at query time before returning results.
   ```python
   # On book deletion - book_upload_api.py:705-714
   ghost_registry = get_ghost_registry(settings.paths.data_dir)  # v0.2.9: Fixed data_path → data_dir
   ghost_registry.add(all_deleted_ids)

   # On book search - search_service.py:643-646
   filtered_results = ghost_registry.filter_ghosts(formatted_results)
   ```

2. **Collection Nuke** ("Clear Books" button) - Replace chunk-by-chunk deletion with `delete_collection()` + `create_collection()`. This rebuilds the HNSW index from scratch - no ghosts possible.
   ```python
   # On "Clear Books" - data_management.py:301-311
   client.delete_collection(name=collection_name)
   adapter.collection = client.get_or_create_collection(
       name=collection_name,
       embedding_function=None,
       metadata={"hnsw:space": "l2"}
   )
   ghost_registry.clear()  # Clean blacklist since fresh index
   ```

**Files**:
- `modules/memory/ghost_registry.py` (NEW) - Ghost tracking class
- `backend/api/book_upload_api.py` - Add deleted IDs to registry
- `modules/memory/search_service.py` - Filter ghosts from search results
- `app/routers/data_management.py` - Nuke/recreate + clear registry

**Impact**: Users no longer see `[No content]` results after deleting books

#### D8: v0.2.9 Bug Fixes

**PathSettings Fix**: Fixed `settings.paths.data_path` → `settings.paths.data_dir` in 3 locations:
- `book_upload_api.py:711` - Ghost registry initialization
- `data_management.py:318` - Clear books ghost registry
- `search_service.py:645` - Search ghost filtering

**UMS Facade Fixes**:
- Added `cleanup_action_kg_for_doc_ids()` passthrough method to `unified_memory_system.py:585-597`
- Added `transparency_context` parameter to `search()` facade method (line 417)
- Fixes TypeError when agent_chat.py passes transparency context to UMS

**Timeout Protection**:
- `embedding_service.py:33-90` - Model loading runs in daemon thread with 120s timeout
- `chromadb_adapter.py:175-203` - Upsert runs in daemon thread with 60s timeout
- `chromadb_adapter.py:268-290` - Query runs in ThreadPoolExecutor with 10s timeout (prevents UI freeze)
- `search_service.py:272-287` - Search wrapper with 15s timeout (graceful fallback to empty results)
- Prevents indefinite hangs on model loading failures, SQLite locks, or HNSW corruption

---

## Production Readiness Enhancements (2025-09-30)

### Complete Backup & Restore System with Selective Export
**Implemented**: Full system backup/restore functionality with granular export control

**Features**:
- **Selective Export**: Choose which data types to include in backup
  - **Sessions**: Conversation history (.jsonl files)
  - **Memory**: ChromaDB vector embeddings (the actual memory)
  - **Books**: Uploaded documents and database
  - **Knowledge**: Knowledge graph, relationships, outcomes
  - Default: All data types included
- **Size Estimation**: Real-time preview of export size before creation
  - Per-category breakdown (sessions: X MB, memory: Y MB, etc.)
  - File counts for each category
  - Updates dynamically as selections change
- **Smart Filenames**: Descriptive names based on content
  - Full backup: `roampal_backup_20250930_143022.zip`
  - Selective: `roampal_sessions_memory_20250930_143022.zip`
- **Restore Functionality**: Upload backup zip to restore system state
- **Pre-restore Backup**: Automatically backs up current data before restoration
- **Backup Management**: List backups, cleanup old backups (keep last 7)

**API Endpoints**:
```python
POST /api/backup/create                              # Full backup
POST /api/backup/create?include=sessions,memory      # Selective backup
GET  /api/backup/estimate?include=sessions           # Size estimate
GET  /api/backup/list                                # List all backups
POST /api/backup/restore                             # Restore from ZIP
DELETE /api/backup/cleanup?keep=7                    # Cleanup old backups
```

**UI Features**:
- **Settings Modal** (SettingsModal.tsx): Simple "Export Data" button at bottom of settings
- **Export Modal** (ExportModal.tsx): Opens when Export Data clicked
  - **Checkbox interface**: Select/deselect individual data types
  - **Real-time size preview**: Shows estimated size as you select
  - **File counts**: Displays number of files per category
  - **Select All/Deselect All**: Quick toggle for convenience
  - **Visual feedback**: Loading states, disabled buttons, success messages
  - **Responsive design**: Fits to page with `max-h-[90vh]` overflow
  - **Consistent styling**: Blue buttons match Memory/Knowledge/New Conversation buttons

**Example Usage**:
```typescript
// Export only conversations (light backup)
POST /api/backup/create?include=sessions

// Export conversations + memory (preserve context)
POST /api/backup/create?include=sessions,memory

// Export everything (full backup)
POST /api/backup/create
```

**Files Added**:
- `app/routers/backup.py` - Backup/restore API with selective export (v1.1)
- `ui-implementation/src/components/SettingsModal.tsx` - Settings modal with Export Data button
- `ui-implementation/src/components/ExportModal.tsx` - Selective export interface with checkboxes

**Technical Implementation**:
- **Modular backup functions**: `_backup_sessions()`, `_backup_memory()`, `_backup_books()`, `_backup_knowledge()`
- **Query parameter validation**: Rejects invalid data types with clear error messages
- **Metadata versioning**: Backup includes `backup_type` (full/selective) and `included_types` list
- **No technical debt**: Clean separation of concerns, reusable functions, comprehensive error handling

**Impact**: Users can now:
- Create targeted backups (e.g., just conversations for sharing)
- Reduce backup size when only specific data needed
- Preview exact size before exporting
- Make informed decisions about what to backup
- Save bandwidth/storage with selective exports

---

### File Locking for Race Condition Prevention (Updated 2025-10-02)
**Implemented**: File-level locking to prevent data corruption from concurrent writes

**Changes**:
- Added `filelock==3.13.1` to requirements.txt
- Knowledge graph writes use `FileLock` with 10-second timeout
- Memory relationships writes use `FileLock` with 10-second timeout
- Session file writes use `FileLock` with 10-second timeout + atomic operations
- All file writes use atomic operations (temp file + rename)
- Session writes include `os.fsync()` to guarantee disk persistence

**Files Modified**:
- `modules/memory/unified_memory_system.py` - Added file locking to `_save_kg_sync()` and `_save_relationships_sync()`
- `app/routers/agent_chat.py` - Enhanced `_save_to_session_file()` with FileLock, atomic writes, and fsync

**Session File Protection**:
Session files now use a robust write pattern:
1. **FileLock** - Prevents concurrent writes from multiple requests/tabs
2. **Read existing content** - Preserves all previous conversation
3. **Write to temp file** - Append new entries atomically
4. **os.fsync()** - Force OS to write data to physical disk (prevents buffer loss on power failure)
5. **Atomic rename** - Replace original file only after successful write

**Power Failure Safety**:
- No partial JSON lines (temp file + atomic rename)
- No data loss from OS buffer cache (fsync guarantees disk write)
- No file corruption from interrupted writes (FileLock prevents interleaving)

**Impact**: Eliminates race conditions, power-loss corruption, and multi-tab interleaving. Session files are now production-safe.

---

### Memory Leak Fixes and Lifecycle Management (Updated 2025-10-02)
**Implemented**: Comprehensive cleanup of background tasks and system resources

**Changes**:
- Background tasks stored in `UnifiedMemorySystem._background_tasks` list
- Tasks properly cancelled and awaited during shutdown
- WebSocket connections always cleaned up in `finally` blocks
- Explicit WebSocket close on disconnect/error
- **NEW**: Lifespan cleanup now calls `memory.cleanup()` and closes LLM client
- **NEW**: Graceful shutdown with error handling for cleanup operations

**Files Modified**:
- `main.py` - Added proper cleanup logic to lifespan context manager (lines 404-423)
- `modules/memory/unified_memory_system.py` - Background task tracking and cleanup

**Cleanup Sequence on Shutdown**:
1. Cancel all background tasks (promotion loop, startup cleanup)
2. Wait for tasks to complete (2-second timeout)
3. Save knowledge graph to disk
4. Save memory relationships to disk
5. Close ChromaDB adapters
6. Close LLM client connection

**Impact**: System can run for days/weeks without memory leaks. Clean shutdown guarantees no data loss on restart/deployment. All background tasks terminate properly.

---

### Exit Button / App Lifecycle (v0.2.8)
**Implemented**: Proper app lifecycle management with explicit Exit button

**Problem**: Clicking the X button was supposed to kill the backend Python process, but orphan processes remained. The `CloseRequested` handler in main.rs wasn't reliably terminating the backend.

**Solution**: Instead of fighting the close behavior, embrace it:
- **X button** = app hides, backend keeps running (fast reopen)
- **Exit button** = clean shutdown via Settings modal (no orphans)

**Implementation**:

**Rust (main.rs:304-321)**:
```rust
#[tauri::command]
fn exit_app(backend: State<BackendProcess>, app_handle: tauri::AppHandle) {
    // Kill backend process
    if let Ok(mut backend) = backend.0.lock() {
        if let Some(mut child) = backend.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    // Exit app
    app_handle.exit(0);
}
```

**React (SettingsModal.tsx:220-237)**:
```tsx
<button
  onClick={async () => {
    await invoke('exit_app');
  }}
  className="w-full h-10 ... bg-red-600/10 hover:bg-red-600/20 border border-red-600/30"
>
  <span className="text-sm font-medium text-red-500">Exit Roampal</span>
</button>
```

**User Experience**:
- X button: App closes but backend stays ready for quick reopen
- Settings → Exit Roampal: Full shutdown, no orphan processes

**Files Modified**:
- `main.rs:304-321` - Added `exit_app` Tauri command
- `main.rs:429-451` - Modified `CloseRequested` to not kill backend on X click
- `SettingsModal.tsx:220-237` - Added "Exit Roampal" button

**Impact**: No more orphan backend processes. Users have clear control over app lifecycle.

---

### Inline Title Generation Optimization (Updated 2025-10-02)
**Implemented**: Title generation now happens inline during streaming response instead of separate API call

**Problem Solved**: Previous implementation made 2 LLM calls for every new conversation:
1. First call: Generate chat response
2. Second call: Generate title from conversation

**New Approach**:
- After first exchange (2 messages), title is auto-generated inline during the streaming response
- Title generation happens immediately after response completes, before completion event
- Uses same LLM context, just a simple follow-up prompt
- Title is streamed back to frontend as `{type: 'title'}` event
- Uses the fixed atomic write system for session file updates

**Technical Flow**:
```
User sends first message
→ Stream response (thinking + text)
→ Save to session file (atomic write)
→ Count messages in session file
→ If message_count == 2:
    → Generate title inline (single LLM call)
    → Update session file with FileLock (atomic)
    → Stream title event to frontend
→ Send completion event
```

**Benefits**:
- ✅ 50% reduction in LLM calls for new conversations (2 calls → 1 call)
- ✅ 3-5 seconds faster for users creating new chats
- ✅ Title appears automatically without frontend triggering separate request
- ✅ Still uses atomic writes for data safety

**Files Modified**:
- `app/routers/agent_chat.py` - Added inline title generation after first exchange (lines 1095-1144)

**Impact**: Significant performance improvement for new conversation creation. Reduces API overhead and improves user experience with instant title display.

---

### System Health Monitoring
**Implemented**: Real-time system health and resource monitoring

**Endpoints**:
- `GET /api/system/health` - Comprehensive health check with warnings
- `GET /api/system/disk-space` - Detailed disk space metrics
- `GET /api/system/data-sizes` - Breakdown of data storage usage

**Metrics Tracked**:
- Disk space (free/used GB and percentages)
- ChromaDB size and file count
- Session count
- Books database size
- Backup folder size and count
- Integrity check results

**Files Added**:
- `app/routers/system_health.py` - System health monitoring API

**Impact**: Users can monitor system health, get warnings before running out of space, and identify large data directories.

---

## Recent Improvements (2025-09-30)

### Book Processor Content Retrieval Fix
**Issue**: Memory search API returned empty `content` field for book chunks
**Root Cause**: Metadata in ChromaDB upsert didn't include actual chunk text (only book_id, chunk_index, type)
**Fix**: Added `content` and `text` fields to metadata dictionary in `smart_book_processor.py:284-290`
**Impact**: Book chunks now fully searchable and retrievable with content visible in UI
**Files Modified**: `modules/memory/smart_book_processor.py`

### Security Enhancements
**Added**:
- UUID format validation in delete endpoint (prevents path traversal accidents)
- UTF-8 encoding validation (rejects binary files with .txt extension)
- Metadata length limits (200 chars title, 1000 chars description)
- Prompt injection pattern detection (logs warnings for user awareness)
- Environment-based exception logging (traceback only in DEBUG mode)

**Files Modified**:
- `backend/api/book_upload_api.py` (validations, error handling)
- `modules/memory/smart_book_processor.py` (content warning detection)

### Chunking Strategy
**Implementation**: RecursiveCharacterTextSplitter with hierarchical separator priority
- Respects document structure (paragraphs → sentences → clauses → words)
- 1500-char target with 300-char overlap for context preservation
- Multi-language support (Latin, CJK, Arabic/Urdu punctuation)
- Intelligent boundary detection prevents mid-sentence splits

**Performance**:
- Small files (1 chunk): ~1 second processing time
- Large books (65 chunks): ~2-3 seconds processing time
- Parallel embedding generation (10 chunks per batch)

## Production Readiness Summary (Updated 2025-10-03)

### For Single-User Local Use: ✅ PRODUCTION READY

**What Was Fixed (Latest Updates)**:
1. ✅ **Complete Backup/Restore** - One-click data protection with selective export
2. ✅ **File Locking + Atomic Writes** - All critical files use FileLock + temp file pattern
3. ✅ **Power Failure Protection** - Session files use fsync() to guarantee disk writes
4. ✅ **Lifecycle Management** - Proper cleanup of background tasks and resources on shutdown
5. ✅ **Memory Leak Fixes** - Clean shutdown and connection management
6. ✅ **System Health Monitoring** - Disk space and resource tracking
7. ✅ **Race Condition Fixes** - Global service init, file writes, title generation (2025-10-03)
8. ✅ **Frontend Sync Improvements** - Error cleanup, WebSocket heartbeat (2025-10-03)
9. ✅ **Conversation History Loading** - Context preserved across restarts (2025-10-03)
10. ✅ **KG Save Debouncing** - 80-90% reduction in file I/O under load (2025-10-03)

**Data Integrity Guarantees**:
- **No file corruption**: FileLock prevents concurrent write conflicts
- **No power loss data loss**: Atomic writes + fsync() guarantee durability
- **No orphaned tasks**: Background tasks properly cancelled on shutdown
- **No resource leaks**: ChromaDB connections and file handles closed cleanly

**What You Get**:
- **Data Safety**: Full backup system + corruption-proof file writes
- **Reliability**: Power-loss resistant, no data corruption from concurrent operations
- **Stability**: System runs for weeks without crashes or leaks
- **Visibility**: Health monitoring shows disk space and data sizes
- **Clean Shutdown**: All data persisted, tasks terminated, connections closed

**Architecture Assessment (2025-10-02)**:
- **Core Design**: Solid 5-tier memory architecture with intelligent learning
- **Data Integrity**: Production-grade file handling with atomic operations
- **Resource Management**: Proper lifecycle management and cleanup
- **Error Handling**: Graceful degradation when components unavailable
- **Knowledge Graph**: Correctly implemented as rebuildable cache (not critical path)

**Remaining Limitations (By Design for Single-User)**:
- No authentication (not needed for localhost)
- No horizontal scaling (single user doesn't need it)
- Embedded ChromaDB (simpler for single user)
- Local file storage (privacy-first, no cloud)

**Production Readiness Score**: **8/10** for single-user local deployment
- Deductions: Limited test coverage, no CI/CD, manual deployment

**Recommended Usage**:
- ✅ Daily personal use
- ✅ Offline coding assistant
- ✅ Private knowledge management
- ✅ Learning and experimentation
- ✅ Small team deployments (with caution)

**Not Recommended For**:
- ❌ Multi-user production without modifications
- ❌ Public internet deployment without authentication
- ❌ Business-critical applications requiring 99.9% uptime and SLAs

**Best Practices**:
1. **Regular Backups**: Use Settings → Export Data weekly
2. **Disk Space**: Keep at least 5GB free
3. **Monitor Health**: Check `/api/system/health` if issues arise
4. **Backup Before Updates**: Export data before system changes
5. **Clean Restarts**: System now guarantees clean shutdown on restart/reload

---

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

---

## Recent Updates (October 2025)

### 2025-10-07: Terminal UX Redesign & System Polish

#### Terminal-Style Interface Overhaul
**Problem**: Modern web-style UI didn't match terminal aesthetic
**Solution**: Complete redesign with ASCII symbols and minimal styling

**UI Changes**:
1. **Tool Execution Badges** - `TerminalMessageThread.tsx:409-446`
   - Before: Bordered pills with backgrounds, technical names
   - After: Terminal-style lines with symbols (✓ ⋯ ✗), plain language
   - Format: `✓ searched memory (Found 5 results)`
   - Monospace font, no borders, Unix-style status indicators

2. **Thinking Status** - `ThinkingDots` (inline in TerminalMessageThread.tsx:10-25)
   - Before (v0.2.4): Collapsible ThinkingBlock with ASCII arrows
   - After (v0.2.5): Animated "Thinking." → "Thinking.." → "Thinking..." status
   - Blue monospace text, 400ms cycle, no collapsible block

3. **Citations Block** - `TerminalMessageThread.tsx:10-59`
   - Before: Card-style with borders, emoji icons
   - After: Tree-view with bracket notation `[5] references`
   - Indented structure with border-left, no backgrounds
   - **Full text display**: All truncation removed
     - Frontend: Removed 150-char UI limit
     - Backend: Removed 200-char citation limit (agent_chat.py:271)
     - Citations now show complete memory content

4. **Visual Hierarchy**
   - Removed all button backgrounds/borders
   - Consistent ASCII symbols (▶▼✓✗⋯)
   - Lowercase labels (Unix convention)
   - Information-dense, distraction-free
   - Matches tools like git, less, tree

**Backend Fixes**:
1. **Duplicate Citations** - `useChatStore.ts:862-864`
   - Problem: Backend sent 5 citations, UI showed 10 (concatenated SSE + complete events)
   - Fix: Use complete event only as authoritative source
   - Now shows correct count

2. **MEMORY_BANK Tags Leaking** - ~~DEPRECATED~~ (See 2025-10-11 migration below)
   - ~~Problem: Internal syntax `[MEMORY_BANK: tag="..." content="..."]` visible in UI~~
   - ~~Temporary Fix: Added regex cleaning to all streaming buffer paths~~
   - **PERMANENT FIX (2025-10-11)**: Migrated to structured tool calls - tags architecturally impossible to leak

3. **Enhanced MEMORY_BANK Instructions** - `agent_chat.py:699-710`
   - Added 4 concrete examples
   - Explicit "Store user facts" directive
   - Simplified from 50 lines to 30 lines
   - Removed "MANDATORY", "NEW", motivational fluff

4. **Thinking Made Optional** - `agent_chat.py:1135-1145`
   - Changed from "ALWAYS" to "may optionally use"
   - LLM can choose when to show reasoning
   - Cleaner responses when thinking isn't needed

**Prompt Simplification**:
- Removed verbose explanations and context markers
- Condensed formatting instructions (7 lines → 1 line)
- Cut fluff like "Use it actively!" and "this helps you..."
- Focused on essential information only

**Impact**: Terminal-native UX, clean prompts, proper citation counts, no tag leakage

---

### 2025-10-11: MEMORY_BANK Migration to Structured Tools

#### Complete Architecture Overhaul: Inline Tags → Structured Tool Calls

**Problem Solved:**
- Inline tags `[MEMORY_BANK: tag="..." content="..."]` were leaking to UI during streaming
- Tags could split across chunks: `"[MEMO" + "RY_BANK: ..."`
- Complex regex filtering across 4 yield points (fragile, edge cases)
- Mixed control flow with user-facing content (architectural anti-pattern)

**Solution: Tool-Based Memory Operations**

1. **Added 3 Memory Bank Tools** - `tool_definitions.py:37-97`
   ```python
   create_memory(content="fact", tag="identity|preference|goal|context")
   update_memory(old_content="old fact", new_content="new fact")
   archive_memory(content="outdated fact")
   ```
   - Structured JSON tool calls (like `search_memory`)
   - Clean separation from text response
   - Impossible for tags to leak to UI

2. **Updated System Prompt** - `agent_chat.py:1310-1321`
   - Removed inline tag syntax completely
   - Tool-based examples: `create_memory(content="...", tag="...")`
   - Clear directive: "DO NOT say 'I'll remember' without calling the tool"

3. **Tool Execution Handlers** - `agent_chat.py:852-952`
   - `create_memory`: Stores with metadata (tags, importance, confidence, status, etc.)
   - `update_memory`: Semantic search → update with reason="llm_update"
   - `archive_memory`: Semantic search → archive item
   - Proper logging: `[MEMORY_BANK TOOL] Created/Updated/Archived`

4. **Removed ALL Streaming Filters**
   - Line 705: Tool continuation filter → removed
   - Line 720: Batch yielding filter → removed
   - Line 830: Native tool filter → removed
   - Line 962: Remaining buffer filter → removed
   - **Result**: Zero filtering complexity, zero tag leakage

5. **Backwards Compatibility** - `agent_chat.py:965-976`
   - Old inline tag extraction kept as fallback (DEPRECATED)
   - Warns if LLM outputs old-style tags
   - Smooth transition, no breaking changes

**Architecture Comparison:**

Before (Inline Tags):
```
LLM → "Text [MEMORY_BANK: tag='x' content='y']"
    → Stream chunks: "[ME" + "MORY" + "_BA" + "NK..."
    → Regex filter (fails on splits)
    → UI sees leaked tags ❌
```

After (Structured Tools):
```
LLM → "Text" + TOOL_CALL(create_memory, {content, tag})
    → Stream text: "Text" (clean)
    → Execute tool separately
    → UI never sees tags ✅
```

**Benefits:**
- ✅ **Zero tag leakage** - Architecturally impossible
- ✅ **Clean architecture** - Matches industry standards (ChatGPT, Claude API)
- ✅ **No filtering complexity** - Removed 4 regex filter points
- ✅ **Proper streaming** - No delays, no edge cases
- ✅ **Maintainable** - Standard tool pattern, easy to extend

**Tech Debt Resolved:**
- Inline control tokens → Structured output
- Regex streaming filters → No filtering needed
- Tag leakage risk → Impossible by design

**Files Modified:**
- `utils/tool_definitions.py`: Added create_memory, update_memory, archive_memory
- `app/routers/agent_chat.py`: Tool handlers, prompt update, filter removal
- All 4 streaming filter points eliminated

**Impact**: Production-ready memory bank system with clean separation of concerns, following modern AI agent best practices.

---

### 2025-10-08: Confidence Scoring Fix & Persistence Improvements

#### Citation Confidence Calculation Fixed (Backend Learning Layer)
**Context**: This fix applies to BACKEND confidence calculation for internal learning (outcome detection, memory promotion). Confidence scores are NOT displayed in UI (removed Oct 7, 2025).

**Problem**: Backend confidence calculations showing 0.00% regardless of relevance
**Root Cause**: Formula assumed ChromaDB distances of 0-2.0, but actual distances were 300-500+

**Investigation Findings**:
- Old formula: `confidence = 1.0 / (1.0 + distance)`
- With distance=381: `1.0 / 382 = 0.0026` → displayed as "0.00%"
- ChromaDB returns raw L2/cosine distances in high-dimensional space (not normalized)

**Solution**: Exponential decay formula that works with any distance range
```python
# New formula (works for all LLMs and embedding models)
CONFIDENCE_SCALE_FACTOR = 100.0
confidence = math.exp(-distance / CONFIDENCE_SCALE_FACTOR)
```

**Results with Actual Data**:
- distance=320 → confidence=0.041 (4.1%) - close match
- distance=381 → confidence=0.022 (2.2%) - moderate match
- distance=474 → confidence=0.009 (0.9%) - distant match

**Code Changes**:
- `app/routers/agent_chat.py:79` - Updated `_format_search_results_as_citations()`
- `modules/memory/unified_memory_system.py:651, 665` - Updated transparency context tracking
- Added detailed docstring explaining scale factor tuning

**Impact**:
- Enables accurate memory system learning (outcome tracking relies on confidence)
- Allows filtering of low-quality results during retrieval
- Works consistently across different LLMs and embedding models
- **Note**: Users do not see these scores - purely for backend decision-making

#### Tool Call Message Persistence Fixed
**Problem**: Messages with search_memory tool calls disappeared after page refresh
**Root Cause**: Early return in streaming path bypassed ALL persistence logic

**Investigation Findings**:
- Tool execution path returned early at line 1573 (old code)
- Skipped: session file save, memory storage, title generation, doc_id tracking
- Only messages WITHOUT tool calls persisted correctly

**Solution**: Refactored into single source of truth for persistence
```python
# New shared function (agent_chat.py:859-1016)
async def _persist_conversation_turn(
    conversation_id, user_message, response_content,
    thinking_content, thinking_sent, search_results, session_file
) -> (exchange_doc_id, title)
```

**Code Changes**:
- Created `_persist_conversation_turn()` shared method
- Both normal path (line 1905) and tool call path (line 1717) now use same function
- Zero code duplication, single maintenance point
- Proper error handling with user-facing warnings

**Impact**:
- Citations now persist across refresh
- Memory system learns from tool-call conversations
- Outcome scoring works for all message types
- Cleaner, more maintainable codebase

**Outcome Detection Character Limit Removed** - `agent_chat.py:1366-1372`
- **Issue**: 10-character minimum threshold blocked meaningful short feedback ("TERRIBLE", "WOW TY")
- **Root Cause**: Arbitrary filter meant to save LLM calls but prevented critical user feedback
- **Fix**: Removed character limit guard - OutcomeDetector LLM already handles noise by returning "unknown"
- **Impact**: All user messages now eligible for outcome detection regardless of length

---

### 2025-10-07: UI/UX Improvements & Bug Fixes (Earlier)

#### Enhanced Transparency & Flow
**Problem**: Users couldn't see what the LLM was doing or which memories were being used
**Solution**: Complete overhaul of visual feedback system

**UI Changes**:
1. **Citation Confidence Scores Removed** - `TerminalMessageThread.tsx:46-50`
   - Issue: ChromaDB distance metrics showing 0% or 67% for all results (unreliable)
   - Fix: Show only collection names (color-coded), no confidence scores

2. **Tool Execution Display Added** - `TerminalMessageThread.tsx:378-396`
   - Feature: Compact pill badges `tool_name + status_icon`
   - Visual: ✓ completed, spinner running, ✗ failed
   - Order: Renders BEFORE thinking block for chronological clarity

3. **Citations Redesigned** - `TerminalMessageThread.tsx:7-71`
   - Changed from: Inline preview with broken scores
   - Changed to: Collapsible "X memories" section with full text previews

4. **Markdown Support Added** - `TerminalMessageThread.tsx:277-363`
   - LLM can use: **bold**, *italic*, `code`, headings, lists, code blocks, callouts
   - Dependencies: `react-markdown`, `remark-gfm`, `rehype-raw`
   - Custom callouts: `:::success`, `:::warning`, `:::info`

**Backend Fixes**:
1. **Cross-Conversation Promotion Enabled** - `unified_memory_system.py:1673-1675`
   - Removed conversation ID filter blocking cross-conversation learning
   - Valuable memories from ANY conversation now promote to history/patterns

2. **Memory Bank Capacity Limit** - `unified_memory_system.py:2341-2357`
   - Added 500-item limit with clear error message
   - Prevents LLM spam, requires user cleanup

3. **Embedding Cache** - `embedding_service.py:32-117`
   - MD5-based cache with 200-item LRU eviction
   - ~30% reduction in redundant embedding generation

4. **Citation Formatting** - `agent_chat.py:46-75`
   - Created helper function to convert distance → confidence
   - Fixed backend/frontend data format mismatch

**Data Flow Fixes**:
1. **Citation Handler** - `useChatStore.ts:813-824`
   - Added missing handler for citation events
2. **Message Mapping** - `ConnectedChat.tsx:1193`
   - Pass toolExecutions through to components
3. **Tool Events** - `useChatStore.ts:777-811`
   - Capture and update tool execution states

**Impact**: Real-time tool feedback, proper citations, markdown formatting, cross-conversation learning, better performance

---

## License

Apache 2.0 License - See LICENSE file for details
---

## Known Issues

### Memory_bank Doc_ID Mismatch (FIXED - 2025-11-26)

**Location**: `unified_memory_system.py:store_memory_bank()`

**Issue**: Returned doc_id didn't match stored doc_id, causing retrieval failures

**Root Cause**:
- `store_memory_bank()` generated ID: `memory_bank_xxxxx`
- Called `store()` which regenerated ID: `memory_bank_xxxxx_timestamp`
- Returned first ID, but document stored under second ID
- Result: Any code trying to retrieve/update using returned ID would fail

**Evidence**:
- Storyteller test showed "Document not found" errors for all create_memory operations
- LLM couldn't search newly created memories (0% success rate)
- Deduplication failed (couldn't find existing docs to check similarity)
- Caused spam of duplicate memories as LLM retried failed operations

**Fix Implemented (2025-11-26)**:
- `store_memory_bank()` now captures doc_id returned by `store()` instead of pre-generating
- Line 3637: `doc_id = await self.store(...)` instead of generating then ignoring return value
- All create_memory operations now return correct retrievable IDs
- Deduplication works properly (can find existing docs)

**Impact**: Fixed memory_bank reliability, deduplication, and LLM tool success rates

### Model Switching Timeout

**Location**: `app/routers/model_switcher.py:204-221`

**Issue**: Model switching frequently fails with timeout error

**Root Cause**:
- Backend performs a health check after model switch (10-second timeout)
- Large models (7B+) not in VRAM take 5-20 seconds to load
- First inference after loading is slowest (cold start)
- 10-second timeout is too aggressive for cold starts

**Workaround**: Try switching again - second attempt usually succeeds as model is cached

**Fix Implemented (2025-10-16)**:
- Health check now uses lightweight parameters (`num_ctx: 2048`, `num_predict: 10`)
- Model switching succeeds on all hardware
- Actual chat uses full context (32768) but gracefully degrades if OOM

## Out-of-Memory (OOM) Graceful Degradation (NEW - 2025-10-16)

### Problem: Ollama Crashes with Large Context on Weak Hardware

**Issue**: Models configured with large context windows (32K+) cause Ollama to crash on systems with insufficient RAM/VRAM. Error: `"llama runner process has terminated: exit status 2"`

**Solution**: 3-tier error handling with automatic retry and user guidance.

### Architecture

#### 1. Lightweight Health Check ([model_switcher.py:262-282](../app/routers/model_switcher.py))

Health check uses minimal parameters to avoid OOM during model switching:

```python
health_payload = {
    "model": model_name,
    "messages": [{"role": "user", "content": "test"}],
    "stream": False,
    "options": {
        "num_ctx": 2048,      # Minimal context for health check only
        "num_predict": 10     # Just need a few tokens to verify it works
    }
}
```

**Result**: All hardware can successfully switch models, regardless of RAM limitations.

#### 2. OOM Detection & Auto-Retry ([ollama_client.py:162-176](../modules/llm/ollama_client.py))

Detects "terminated" error from Ollama and automatically retries with reduced context:

```python
if api_response.status_code == 500:
    error_text = api_response.text
    if "terminated" in error_text.lower():
        logger.warning(f"OOM detected, retrying with reduced context (2048)")
        original_ctx = payload["options"].get("num_ctx", "unknown")
        payload["options"]["num_ctx"] = 2048  # Retry with minimal context
        api_response = await self.client.post("/api/chat", json=payload)
        payload["_oom_recovered"] = True  # Mark for warning message
```

**Applies to both:**
- Non-streaming: `generate_response()` (line 162-176)
- Streaming: `stream_response_with_tools()` (line 680-726)

#### 3. User Guidance ([ollama_client.py:330-341](../modules/llm/ollama_client.py))

Prepends warning message to response with actionable fix:

```python
warning_msg = (
    f"⚠️ **Memory Limit Reached**\n\n"
    f"This model ran out of memory with {original_ctx} context window. "
    f"Reduced to 2048 tokens for this response.\n\n"
    f"**To fix permanently:** Open Settings → Context Window Settings → "
    f"Lower context for `{actual_model}` to 8K or less.\n\n"
    f"---\n\n"
)
```

### Behavior

**Strong Hardware (Most Users)**:
- Model uses full configured context (e.g., 32768)
- No OOM errors, no warnings, no performance penalty

**Weak Hardware (Insufficient RAM/VRAM)**:
- First message of each session:
  1. Tries full context (32768) → Ollama crashes (~5-8 sec)
  2. Auto-retries with minimal context (2048) → succeeds
  3. Shows warning with instructions
- User can permanently lower context via Settings UI
- Subsequent messages in same session repeat OOM cycle (no session memory by design)

### Integration with Context Management

Works seamlessly with existing context window system:
- Uses `get_context_size()` from `config/model_contexts.py` ✅
- User can adjust via `POST /api/model/context/{model_name}` API ✅
- Frontend ModelContextSettings.tsx provides UI control ✅
- OOM fix intercepts AFTER context configuration, BEFORE Ollama crash ✅

### Benefits

1. **Universal Compatibility**: Works on all hardware without configuration
2. **Graceful Degradation**: Strong PCs get full performance, weak PCs still work
3. **User Empowerment**: Clear instructions to fix permanently
4. **No Special Casing**: Same codebase for all hardware tiers
5. **Minimal Tech Debt**: Only 25 lines of code across 2 methods

**Files Modified**:
- `app/routers/model_switcher.py` - Lightweight health check (lines 262-282)
- `modules/llm/ollama_client.py` - OOM detection (lines 162-176, 680-726), user warning (lines 330-341)

**Additional Fixes (2025-10-22)**:
1. Memory refresh blocked by streaming state - Event dispatch moved outside streaming check (useChatStore.ts:723)
2. Tool indicators disappeared during streaming - toolExecutions explicitly preserved at 3 update points (lines 581, 604, 745)
3. archive_memory indicator disappeared - Messages with tools but no text content are preserved (lines 732-748)
4. Backend memory flag conditional - Changed to memory_updated: True on ALL responses (agent_chat.py:830)

