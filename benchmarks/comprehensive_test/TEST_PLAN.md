# Comprehensive Memory System Test Plan

## Purpose
Validate EVERY feature of the Roampal memory system through exhaustive testing.
- **No LLM dependencies** - Pure Python, deterministic
- **Start from empty state** - Build KGs from scratch
- **Stress test all tiers** - Exercise full lifecycle (store → promote → demote → delete)
- **Validate all 3 KGs** - Routing, Content, Action-Effectiveness

---

## Test Execution Requirements

### Environment
- **Python 3.10+**
- **Dependencies**: `asyncio`, `pytest`, `json`, `datetime`
- **ChromaDB**: Running on localhost:8003 (or embedded mode)
- **No LLM required** - All test data is pre-defined

### Setup
```bash
cd benchmarks/comprehensive_test
python test_comprehensive.py
```

### Expected Runtime
- **< 5 minutes** for full test suite
- **Deterministic** - Same inputs = same outputs

### Cleanup
- Creates `./test_data/` directory
- Auto-deletes after test completion
- Can keep for debugging with `--keep-data` flag

---

## Feature Coverage Matrix

### 1. STORAGE OPERATIONS

#### 1.1 Basic Storage
- [x] Store to working collection
- [x] Store to history collection
- [x] Store to patterns collection
- [x] Store to books collection
- [x] Store to memory_bank collection
- [x] Generate unique doc IDs
- [x] Generate embeddings (768d vectors)
- [x] Store metadata correctly

**Pass Criteria:**
- All 5 collections accept stores
- Doc IDs follow format: `{collection}_{uuid}_{timestamp}`
- Embeddings are 768-dimensional floats
- Metadata persists with all fields intact

#### 1.2 Deduplication (Memory_bank & Patterns)
- [x] Detect 95% similarity duplicates
- [x] Keep higher quality version (importance × confidence)
- [x] Increment `mentioned_count` on duplicate
- [x] Archive lower quality version
- [x] NO deduplication for working/history/books

**Pass Criteria:**
- Store duplicate with quality=0.7 → increments count
- Store duplicate with quality=0.95 → archives old, stores new
- Working/history accept duplicates without merging
- `SIMILARITY_THRESHOLD = 0.95` enforced

#### 1.3 Contextual Retrieval (v0.2.1)
- [x] Generate contextual prefix (if LLM available, mock in test)
- [x] Fallback to original text if LLM unavailable
- [x] Prefix format: "{context}, {collection type}: {text}"
- [x] Skip for text < 50 chars

**Pass Criteria:**
- Mocked prefix generation works
- Original text preserved in metadata
- Embedding uses contextualized text
- Graceful fallback when mocked LLM fails

---

### 2. RETRIEVAL OPERATIONS

#### 2.1 Basic Search
- [x] Search single collection
- [x] Search multiple collections
- [x] Search with limit parameter
- [x] Search with metadata filters
- [x] Return results with distance/score

**Pass Criteria:**
- Query returns top-k results
- Results sorted by relevance
- Metadata filters work (ChromaDB `where` syntax)
- Empty collections return []

#### 2.2 Search Depth Multiplier
- [x] All collections use `limit × 3` multiplier
- [x] Working: fetches 15 for limit=5
- [x] Memory_bank: fetches 15 for limit=5
- [x] Books: fetches 15 for limit=5
- [x] History/Patterns: fetches 15 for limit=5

**Pass Criteria:**
- Verify 3× depth in all collection searches
- Final results correctly limited to `limit` parameter
- Fair competition across collections

#### 2.3 Hybrid Search (BM25 + Vector + RRF)
- [x] Vector search (embeddings)
- [x] BM25 search (lexical) - if available
- [x] Reciprocal Rank Fusion (k=60)
- [x] Graceful fallback if BM25 unavailable

**Pass Criteria:**
- Pure vector search works (baseline)
- If BM25 available: hybrid fusion works
- RRF formula: `score = Σ(1/(rank+60))`
- Fallback to vector-only if BM25 missing

#### 2.4 Cross-Encoder Reranking
- [x] Rerank top-30 results - if available
- [x] Blend: 40% original + 60% cross-encoder
- [x] Graceful fallback if cross-encoder unavailable

**Pass Criteria:**
- Pure ranking works (baseline)
- If cross-encoder available: reranking improves precision
- Fallback to original ranking if unavailable

#### 2.5 Memory_bank Quality Ranking
- [x] Boost by `importance × confidence × 0.5`
- [x] Adjusted distance: `distance × (1 - quality × 0.5)`
- [x] Max 50% boost for quality=1.0
- [x] Content KG entity boost (additional 50%)

**Pass Criteria:**
- High-quality memory (0.9, 0.9) ranks higher than low-quality (0.3, 0.4) with same distance
- Entity boost applies only to memory_bank
- Formula validated: `quality = importance × confidence`

---

### 3. OUTCOME-BASED SCORING

#### 3.1 Score Updates
- [x] "worked" → +0.2 (max 1.0)
- [x] "failed" → -0.3 (min 0.1)
- [x] "partial" → +0.05
- [x] "unknown" → no change
- [x] Score clamped to [0.1, 1.0]

**Pass Criteria:**
- Initial score: 0.5
- After "worked": 0.7
- After "failed": 0.4
- After 5× "worked": caps at 1.0
- After 5× "failed": floors at 0.1

#### 3.2 Collections Using Outcome Scoring
- [x] Working: YES (score exists)
- [x] History: YES (score exists)
- [x] Patterns: YES (score exists)
- [x] Books: NO (pure distance)
- [x] Memory_bank: NO (quality = importance × confidence)

**Pass Criteria:**
- Working/history/patterns metadata includes `score` field
- Books/memory_bank do NOT have `score` field
- Outcome scoring only updates scoreable collections

---

### 4. PROMOTION & DEMOTION

#### 4.1 Working → History
- [x] Threshold: score ≥ 0.7 AND uses ≥ 2
- [x] Auto-promotion every 20 messages
- [x] Promotion on conversation switch
- [x] Background task (hourly)

**Pass Criteria:**
- Memory with score=0.8, uses=3 promotes
- Memory with score=0.6, uses=3 does NOT promote
- Memory with score=0.8, uses=1 does NOT promote
- Promotion updates metadata (promoted_from, promotion_history)

#### 4.2 History → Patterns
- [x] Threshold: score ≥ 0.9 AND uses ≥ 3
- [x] Copies to patterns collection
- [x] Deletes from history

**Pass Criteria:**
- Memory with score=0.95, uses=4 promotes
- Memory with score=0.85, uses=4 does NOT promote
- New doc_id format: `patterns_{uuid}`
- Metadata includes `promoted_from: "history"`

#### 4.3 Fast-Track Promotion (Working → Patterns)
- [x] Threshold: score ≥ 0.9 AND uses ≥ 3 AND last 3 outcomes = "worked"
- [x] Skips history tier
- [x] Marks as `fast_tracked: true`

**Pass Criteria:**
- Memory with score=0.95, uses=3, outcomes=[worked, worked, worked] → fast-tracks
- Memory with score=0.95, uses=3, outcomes=[worked, worked, partial] → does NOT fast-track
- Metadata includes `fast_tracked: true`

#### 4.4 Patterns → History (Demotion)
- [x] Threshold: score < 0.3
- [x] Copies back to history
- [x] Deletes from patterns

**Pass Criteria:**
- Pattern with score=0.2 demotes
- Pattern with score=0.4 does NOT demote
- New doc_id format: `history_{uuid}`
- Metadata includes `demoted_from: "patterns"`

#### 4.5 Deletion
- [x] Threshold: score < 0.2 (for items > 7 days old)
- [x] Threshold: score < 0.1 (for items < 7 days old)
- [x] Deletes from collection
- [x] Does NOT delete high-value (score ≥ 0.9)

**Pass Criteria:**
- Old memory (10 days) with score=0.15 → deletes
- New memory (3 days) with score=0.15 → does NOT delete
- High-value memory with score=0.95 → never deletes
- Working/history only (patterns permanent)

---

### 5. KNOWLEDGE GRAPHS

#### 5.1 Routing KG
- [x] Stores concept → collection mappings
- [x] Tracks success_rates per collection
- [x] Tracks failure_patterns
- [x] Updates on search outcomes
- [x] Learning phases: exploration → confidence → mastery

**Pass Criteria:**
- After 10 searches with concept "authentication", KG learns best collection
- Exploration phase (score < 0.5): searches all 5 collections
- Confidence phase (0.5 ≤ score < 2.0): searches top 2-3
- Mastery phase (score ≥ 2.0): searches top 1-2

**Structure Validation:**
```json
{
  "routing_patterns": {
    "authentication": {
      "best_collection": "patterns",
      "score": 3.5,
      "uses": 15
    }
  },
  "success_rates": {
    "working": 0.7,
    "history": 0.8,
    "patterns": 0.95,
    "books": 0.6,
    "memory_bank": 0.85
  }
}
```

#### 5.2 Content KG
- [x] Extracts entities from memory_bank content
- [x] Builds entity relationships (co-occurrence)
- [x] Tracks entity quality (importance × confidence)
- [x] Entity boost for memory_bank searches
- [x] Cleanup on archive/delete

**Pass Criteria:**
- Store "User prefers Docker for development" → extracts ["docker", "development"]
- Builds relationship: docker <-> development
- Entity quality tracked: avg_quality = Σ(importance × confidence) / mentions
- Boost calculation: up to 1.5× (50% boost)
- Archive memory → entities cleaned up

**Structure Validation:**
```json
{
  "entities": {
    "docker": {
      "mentions": 5,
      "collections": {"memory_bank": 5},
      "documents": ["mem_001", "mem_002"],
      "avg_quality": 0.85
    }
  },
  "relationships": {
    "docker__development": {
      "entities": ["docker", "development"],
      "strength": 4.47,
      "co_occurrences": 5
    }
  }
}
```

#### 5.3 Action-Effectiveness KG
- [x] Tracks (context, action, collection) → success_rate
- [x] Updates on every action outcome
- [x] Context detection (LLM-classified)
- [x] Conversation boundary detection

**Pass Criteria:**
- After 10 `search_memory` calls in "coding" context with 9 successes:
  - `"coding|search_memory|patterns"` → 90% success_rate
- Different contexts tracked separately:
  - `"coding|search_memory|patterns"` vs `"fitness|search_memory|patterns"`
- Boundary detection clears cache on topic shift

**Structure Validation:**
```json
{
  "context_action_effectiveness": {
    "coding|search_memory|patterns": {
      "success_count": 9,
      "failure_count": 1,
      "success_rate": 0.9,
      "total_uses": 10
    }
  }
}
```

---

### 6. TIER-SPECIFIC FEATURES

#### 6.1 Books Collection
- [x] Permanent storage (never decays)
- [x] Pure distance ranking (no score)
- [x] No outcome-based scoring
- [x] Searchable by title/author metadata

**Pass Criteria:**
- Books never deleted
- No `score` field in metadata
- Metadata filters work: `{"title": "architecture"}`

#### 6.2 Working Collection
- [x] 24-hour retention (mocked time)
- [x] Global search (across all conversations)
- [x] Auto-promotion every 20 messages
- [x] Outcome-based scoring

**Pass Criteria:**
- Memories > 24h deleted (unless high-value)
- Search returns results from different conversation_ids
- Message count triggers promotion

#### 6.3 History Collection
- [x] 30-day retention (mocked time)
- [x] High-value preservation (score ≥ 0.9)
- [x] Promotes to patterns
- [x] `clear_old_history()` deletes old items

**Pass Criteria:**
- Memories > 30 days deleted (unless high-value)
- High-value memories (score ≥ 0.9) preserved indefinitely
- Decay test:
  ```python
  # Day 1: Store history item (score=0.6)
  doc_id = store("Old memory", collection="history", score=0.6)

  # Day 35: Advance time
  advance_time(35 days)
  await clear_old_history(days=30)

  # Result: doc_id deleted (> 30 days, score < 0.9)
  assert not exists(doc_id)

  # But high-value memory survives:
  doc_id_2 = store("Important memory", collection="history", score=0.95)
  advance_time(100 days)
  await clear_old_history(days=30)
  assert exists(doc_id_2)  # Still there!
  ```

#### 6.4 Patterns Collection
- [x] Permanent storage
- [x] Demotes to history if score drops
- [x] Deduplication (95% threshold)

**Pass Criteria:**
- Patterns never auto-deleted
- Score < 0.3 triggers demotion
- Duplicates merged

#### 6.5 Memory_bank Collection
- [x] Permanent storage
- [x] 500-item capacity limit
- [x] Quality ranking (importance × confidence)
- [x] Deduplication (95% threshold)
- [x] Content KG entity extraction
- [x] Archive/restore functionality

**Pass Criteria:**
- 501st item rejected with error
- Quality boost formula validated
- Duplicates merged by quality
- Archive sets `status: "archived"`
- Restore sets `status: "active"`

---

### 7. EDGE CASES & ROBUSTNESS

#### 7.1 Empty States
- [x] Search empty collection → []
- [x] Promote with no eligible items → no-op
- [x] Delete non-existent doc → graceful error
- [x] Archive non-existent memory_bank item → error

**Pass Criteria:**
- No crashes on empty operations
- Appropriate error messages

#### 7.2 Boundary Conditions
- [x] Store with empty text → error
- [x] Search with empty query → error
- [x] Score at min (0.1) → further decreases ignored
- [x] Score at max (1.0) → further increases ignored
- [x] Memory_bank at 500 items → 501st rejected

**Pass Criteria:**
- Validation errors raised
- Score clamping works
- Capacity enforced

#### 7.3 Concurrent Operations
- [x] Simultaneous stores to same collection
- [x] Promotion during active search
- [x] KG updates during promotion
- [x] No race conditions in auto-promotion lock

**Pass Criteria:**
- No data corruption
- Async locks prevent races
- All operations complete successfully

#### 7.4 Data Integrity
- [x] Metadata JSON fields (lists, dicts) serialize correctly
- [x] Embeddings persist and retrieve correctly
- [x] Doc IDs are unique across all operations
- [x] Timestamps are valid ISO format

**Pass Criteria:**
- JSON fields don't get stringified incorrectly
- Embeddings remain 768-dimensional
- No duplicate doc IDs
- Timestamps parseable by `datetime.fromisoformat()`

---

## Test Output Format

### Console Output
```
========================================
ROAMPAL COMPREHENSIVE MEMORY SYSTEM TEST
========================================

[1/7] STORAGE OPERATIONS
  [1.1] Basic Storage...................... ✓ PASS (5/5 tests)
  [1.2] Deduplication...................... ✓ PASS (5/5 tests)
  [1.3] Contextual Retrieval............... ✓ PASS (4/4 tests)

[2/7] RETRIEVAL OPERATIONS
  [2.1] Basic Search....................... ✓ PASS (5/5 tests)
  [2.2] Search Depth Multiplier............ ✓ PASS (5/5 tests)
  [2.3] Hybrid Search...................... ✓ PASS (4/4 tests)
  [2.4] Cross-Encoder Reranking............ ✓ PASS (3/3 tests)
  [2.5] Memory_bank Quality Ranking........ ✓ PASS (4/4 tests)

[3/7] OUTCOME-BASED SCORING
  [3.1] Score Updates...................... ✓ PASS (5/5 tests)
  [3.2] Collections Using Outcome Scoring.. ✓ PASS (2/2 tests)

[4/7] PROMOTION & DEMOTION
  [4.1] Working → History.................. ✓ PASS (4/4 tests)
  [4.2] History → Patterns................. ✓ PASS (3/3 tests)
  [4.3] Fast-Track Promotion............... ✓ PASS (2/2 tests)
  [4.4] Patterns → History (Demotion)...... ✓ PASS (2/2 tests)
  [4.5] Deletion........................... ✓ PASS (4/4 tests)

[5/7] KNOWLEDGE GRAPHS
  [5.1] Routing KG......................... ✓ PASS (4/4 tests)
  [5.2] Content KG......................... ✓ PASS (5/5 tests)
  [5.3] Action-Effectiveness KG............ ✓ PASS (3/3 tests)

[6/7] TIER-SPECIFIC FEATURES
  [6.1] Books Collection................... ✓ PASS (4/4 tests)
  [6.2] Working Collection................. ✓ PASS (4/4 tests)
  [6.3] History Collection................. ✓ PASS (2/2 tests)
  [6.4] Patterns Collection................ ✓ PASS (3/3 tests)
  [6.5] Memory_bank Collection............. ✓ PASS (6/6 tests)

[7/7] EDGE CASES & ROBUSTNESS
  [7.1] Empty States....................... ✓ PASS (4/4 tests)
  [7.2] Boundary Conditions................ ✓ PASS (5/5 tests)
  [7.3] Concurrent Operations.............. ✓ PASS (4/4 tests)
  [7.4] Data Integrity..................... ✓ PASS (4/4 tests)

========================================
RESULTS: 97/97 tests passed (100%)
Runtime: 4m 23s
========================================

KG State Saved:
  - test_data/routing_kg.json (15 concepts learned)
  - test_data/content_kg.json (47 entities, 123 relationships)
  - test_data/action_effectiveness_kg.json (24 action patterns)

Tier Counts:
  - books: 10 items
  - working: 23 items (12 promoted to history)
  - history: 15 items (3 promoted to patterns)
  - patterns: 8 items
  - memory_bank: 15 items
```

---

## Passing Criteria Summary

### Critical (Must Pass 100%)
- All storage operations work
- All retrieval operations return correct results
- Outcome scoring math correct (+0.2/-0.3)
- Promotion/demotion thresholds enforced
- All 3 KGs build correctly
- No data corruption or crashes

### Important (Must Pass 95%+)
- Enhanced retrieval (hybrid, reranking) with graceful fallback
- Edge cases handled gracefully
- Concurrent operations safe

### Nice-to-Have (Must Pass 80%+)
- Performance < 5 minutes
- Clean console output
- Detailed KG inspection files

---

## Test Data Strategy

### Deterministic Test Data
All test data is pre-defined (no LLM generation):

```python
# Storage test data
STORAGE_TESTS = [
    {"text": "Python best practices", "collection": "books"},
    {"text": "User asked about Docker", "collection": "working"},
    {"text": "Solved authentication bug", "collection": "patterns"},
    # ... 50 predefined entries
]

# Search test queries
SEARCH_QUERIES = [
    {"query": "authentication", "expected_collection": "patterns"},
    {"query": "Docker setup", "expected_collection": "working"},
    # ... 20 predefined queries
]

# Outcome test cases
OUTCOME_TESTS = [
    {"doc_id": "working_001", "outcome": "worked", "expected_score": 0.7},
    {"doc_id": "working_002", "outcome": "failed", "expected_score": 0.2},
    # ... 30 predefined cases
]
```

### Mock Utilities
```python
def mock_embedding(text: str) -> List[float]:
    """Generate deterministic 768d embedding from text hash"""
    # Use hash(text) to create reproducible vector

def mock_llm_contextual_prefix(text: str, metadata: dict) -> str:
    """Generate deterministic contextual prefix"""
    # Use rule-based prefix generation (no actual LLM)

def advance_time(days: int):
    """Mock time advancement for decay testing"""
    # Patch datetime.now() for deterministic time-based tests
```

---

## Implementation Notes

### File Structure
```
benchmarks/comprehensive_test/
  ├── TEST_PLAN.md              # This file
  ├── test_comprehensive.py     # Main test script
  ├── test_data_fixtures.py     # Pre-defined test data
  ├── mock_utilities.py         # Mock functions (embeddings, LLM, time)
  ├── validators.py             # Assertion helpers
  └── README.md                 # Quick start guide
```

### Dependencies
```python
# test_comprehensive.py
import sys
import os
sys.path.insert(0, '../ui-implementation/src-tauri/backend')

from modules.memory.unified_memory_system import UnifiedMemorySystem
from modules.memory.content_graph import ContentGraph
from modules.embedding.embedding_service import EmbeddingService
```

### Execution Flow
1. **Setup**: Create test_data/ directory, initialize empty memory system
2. **Run Tests**: Execute all 97 tests in order
3. **Validate**: Check assertions, save KG states
4. **Report**: Print results, save detailed logs
5. **Cleanup**: Delete test_data/ (unless --keep-data flag)

---

## Success Metrics

### Test Passes
- **97/97 tests pass** → System is production-ready
- **90-96/97 tests pass** → Minor issues, review failures
- **< 90/97 tests pass** → Major issues, do not deploy

### Performance
- **< 3 minutes** → Excellent
- **3-5 minutes** → Acceptable
- **> 5 minutes** → Needs optimization

### Coverage
- **100% of documented features tested** → Complete
- **All 3 KGs validated** → Intelligence layer working
- **All 5 tiers exercised** → Full lifecycle tested
- **Edge cases covered** → Robust
