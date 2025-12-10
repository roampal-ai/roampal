# UnifiedMemorySystem Refactoring Plan

> **Date:** December 2024
> **Status:** Planning
> **Related:** [ARCHITECTURAL_CRITIQUE.md](./ARCHITECTURAL_CRITIQUE.md)
> **Fact-Checked:** December 9, 2024 (all line numbers and counts verified against codebase)

---

## Codebase Context

### Two Versions Exist

| Location | Lines | Wilson Score | Status |
|----------|-------|--------------|--------|
| `c:/ROAMPAL/modules/memory/unified_memory_system.py` | 2,584 | ❌ Missing | Stripped/older - **DO NOT USE** |
| `c:/ROAMPAL/ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py` | 4,746 | ✅ Present | **CANONICAL SOURCE** |

**Why two versions?** The `ui-implementation` path is structured for Tauri bundling. The root-level version appears to be an older/stripped copy.

**Action:** All refactoring work targets the `ui-implementation/src-tauri/backend/` codebase.

---

## Build Strategy

### Isolated Build Approach

Build the new architecture in a separate folder to avoid risk to production:

```
C:\ROAMPAL-REFACTOR\
├── modules/
│   └── memory/
│       ├── unified_memory_system.py   # ~800 line facade
│       ├── config.py                  # MemoryConfig dataclass
│       ├── types.py                   # OutcomeHistory, PromotionRecord, etc.
│       ├── scoring_service.py         # Wilson score, final score calc
│       ├── promotion_service.py       # Promotion/demotion logic
│       ├── routing_service.py         # KG-based collection routing
│       ├── kg_service.py              # Knowledge graph ops (with Lock fix)
│       ├── search_service.py          # Hybrid search, reranking
│       ├── memory_bank_service.py     # Memory bank CRUD
│       ├── context_service.py         # Context detection
│       └── outcome_service.py         # Outcome recording
├── tests/
│   ├── characterization/              # Captured current behavior
│   │   ├── test_search_behavior.py
│   │   ├── test_outcome_behavior.py
│   │   └── test_mcp_tools.py
│   └── unit/                          # New service unit tests
└── benchmarks/
    └── compare_old_vs_new.py          # Regression check
```

**Benefits:**
- No risk to production until swap
- Side-by-side comparison possible
- Easy rollback (just don't swap)
- Clean git history

**Swap Criteria:**
1. Characterization tests pass against BOTH implementations
2. Benchmarks match within ±1%
3. All 7 MCP tools return identical shapes
4. Stress test (1000 ops) passes without errors

---

## Dependency Strategy

### Import from Original (Don't Copy)

The refactor targets `UnifiedMemorySystem` decomposition only. Stable dependencies stay in place:

```python
# In C:\ROAMPAL-REFACTOR\modules\memory\scoring_service.py
import sys
sys.path.insert(0, "C:/ROAMPAL/ui-implementation/src-tauri/backend")

from modules.embedding.embedding_service import EmbeddingService  # Import from original
from modules.memory.chromadb_adapter import ChromaDBAdapter        # Import from original
```

**Rationale:** Keep scope tight. We're restructuring the God class, not rebuilding ChromaDB or embedding logic.

### Use Existing Python Environment

```bash
# Run tests using the bundled Python with all deps installed
cd C:\ROAMPAL-REFACTOR
C:\ROAMPAL\ui-implementation\src-tauri\binaries\python\python.exe -m pytest tests/
```

**Rationale:** Avoid version mismatches. The existing environment has all dependencies (chromadb, sentence-transformers, scipy, etc.) already working.

---

## Executive Summary

This document outlines the decomposition of `UnifiedMemorySystem` from a 4,746-line God class into 11 focused services. The refactoring preserves all functionality while addressing architectural debt identified in the critique.

**Current State:** 4,746 lines, 52 async methods, 1 God class
**Target State:** ~800 line facade + 10 services (~4,050 lines total)

---

## Phase 1: Configuration & Types (Days 1-2)

### 1.1 Create `MemoryConfig` dataclass
**New file:** `modules/memory/config.py`

Extract all magic numbers from the codebase:

| Constant | Current Location | Value | Purpose |
|----------|------------------|-------|---------|
| HIGH_VALUE_THRESHOLD | Line 202 | 0.9 | Preserve memories above this |
| PROMOTION_SCORE_THRESHOLD | Line 203 | 0.7 | Min score for promotion |
| DEMOTION_SCORE_THRESHOLD | Line 204 | 0.4 | Below this, patterns demote |
| DELETION_SCORE_THRESHOLD | Line 205 | 0.2 | Below this, delete |
| NEW_ITEM_DELETION_THRESHOLD | Line 206 | 0.1 | Lenient for new items |
| CROSS_ENCODER_BLEND_RATIO | Line 709 | 0.6 | Cross-encoder weight |
| DEFAULT_IMPORTANCE | Line 993 | 0.7 | Default memory importance |
| EMBEDDING_WEIGHT_PROVEN | Line 1580 | 0.2 | Embedding weight for proven |
| LEARNED_WEIGHT_PROVEN | Line 1581 | 0.8 | Learned weight for proven |
| KG_DEBOUNCE_SECONDS | Line 398 | 5 | KG save debounce |
| MAX_MEMORY_BANK_ITEMS | Line 4221 | 1000 | Capacity limit |
| SEARCH_MULTIPLIER | Lines 1359, 1398, 1413, 1421 | 3 | Search depth multiplier (limit × 3) |

> **Note:** `SEARCH_MULTIPLIER` is currently hardcoded in 4 locations. Architecture.md incorrectly references lines 1090, 1129, 1144, 1152 (stale - drifted ~270 lines). Actual locations verified Dec 9, 2024.

```python
@dataclass
class MemoryConfig:
    high_value_threshold: float = 0.9
    promotion_score_threshold: float = 0.7
    demotion_score_threshold: float = 0.4
    deletion_score_threshold: float = 0.2
    new_item_deletion_threshold: float = 0.1
    cross_encoder_blend_ratio: float = 0.6
    default_importance: float = 0.7
    kg_debounce_seconds: int = 5
    max_memory_bank_items: int = 1000
    search_multiplier: int = 3  # Currently hardcoded as `limit * 3` in 4 locations
```

### 1.2 Create typed dataclasses for JSON strings
**New file:** `modules/memory/types.py`

Replace 16 JSON serialization occurrences:

```python
@dataclass
class OutcomeEntry:
    outcome: Literal["worked", "failed", "partial", "unknown"]
    timestamp: str
    context: Optional[str] = None

@dataclass
class OutcomeHistory:
    entries: List[OutcomeEntry] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self.entries])

    @classmethod
    def from_json(cls, data: str) -> "OutcomeHistory":
        entries = [OutcomeEntry(**e) for e in json.loads(data)]
        return cls(entries=entries)

@dataclass
class PromotionRecord:
    from_collection: str
    to_collection: str
    timestamp: str
    score: float
    uses: int

@dataclass
class PromotionHistory:
    promotions: List[PromotionRecord] = field(default_factory=list)
    # ... similar serialization methods
```

**Lines affected:** 928, 929, 932, 1529, 2989, 3028, etc.

---

## Phase 2: Extract ScoringService (Days 3-4)

### 2.1 Create `ScoringService`
**New file:** `modules/memory/scoring_service.py`

**Extract from `unified_memory_system.py`:**

| Method/Code | Lines | Purpose |
|-------------|-------|---------|
| `wilson_score_lower()` | 47-90 | Statistical confidence scoring |
| Scoring logic in `search()` | 1514-1656 | Calculate final rank score |
| Dynamic weight calculation | 1570-1644 | Weight embedding vs learned |

**Interface:** Implement existing `core/interfaces/scoring_engine_interface.py`

```python
class ScoringService(ScoringEngineInterface):
    def __init__(self, config: MemoryConfig):
        self.config = config

    @staticmethod
    def wilson_score_lower(successes: float, total: int, confidence: float = 0.95) -> float:
        """Calculate Wilson score confidence interval lower bound."""
        ...

    def calculate_final_score(
        self,
        metadata: Dict[str, Any],
        distance: float,
        collection: str
    ) -> Dict[str, float]:
        """
        Returns:
            {
                "final_rank_score": combined score,
                "wilson_score": statistical confidence,
                "embedding_similarity": 1/(1+distance),
                "learned_score": outcome-based score,
                "embedding_weight": weight used,
                "learned_weight": weight used
            }
        """
        ...

    def get_dynamic_weights(
        self,
        uses: int,
        score: float,
        collection: str
    ) -> Tuple[float, float]:
        """Return (embedding_weight, learned_weight) based on memory maturity."""
        ...
```

**Lines removed from UnifiedMemorySystem:** ~200

---

## Phase 3: Extract PromotionService (Days 5-6)

### 3.1 Create `PromotionService`
**New file:** `modules/memory/promotion_service.py`

**Extract from `unified_memory_system.py`:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `_handle_promotion()` | 2959-3070 | Handle auto promotion/demotion |
| `_promote_valuable_working_memory()` | 3489-3600+ | Batch promote valuable memories |
| `_promote_item()` | 2921-2957 | Move item between collections |
| `_handle_promotion_error()` | 3482-3487 | Error callback for async tasks |

```python
class PromotionService:
    def __init__(
        self,
        config: MemoryConfig,
        collections: Dict[str, ChromaDBAdapter],
        embedding_service: EmbeddingServiceInterface,
        relationship_tracker: Any  # For _add_relationship calls
    ):
        self.config = config
        self.collections = collections
        self.embedding_service = embedding_service
        self._promotion_lock = asyncio.Lock()

    async def handle_promotion(
        self,
        doc_id: str,
        collection: str,
        score: float,
        uses: int,
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Handle automatic promotion/demotion.
        Returns new doc_id if promoted/demoted, None otherwise.
        """
        ...

    async def promote_valuable_working_memory(
        self,
        conversation_id: Optional[str] = None
    ) -> int:
        """
        Promote valuable working memories to history.
        Returns count of promoted items.
        """
        ...
```

**Lines removed from UnifiedMemorySystem:** ~400

---

## Phase 4: Extract RoutingService (Day 7)

### 4.1 Create `RoutingService`
**New file:** `modules/memory/routing_service.py`

**Extract from `unified_memory_system.py`:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `_route_query()` | 2131-2217 | Intelligent collection routing |
| `_calculate_tier_scores()` | 2082-2129 | Score collections for query |
| `get_tier_recommendations()` | 2802-2860 | Recommendations for insights |
| `_extract_concepts()` | 2219-2299 | N-gram extraction |
| `ACRONYM_DICT` | 1036-1156 | Acronym expansion |

```python
class RoutingService:
    def __init__(self, knowledge_graph: Dict, config: MemoryConfig):
        self.kg = knowledge_graph
        self.config = config
        self.acronym_dict = self._load_acronyms()

    def route_query(self, query: str) -> List[str]:
        """
        Intelligent routing using learned KG patterns.

        Returns list of collection names to search based on:
        - Phase 1 (Exploration): total_score < 0.5 → all 5 collections
        - Phase 2 (Medium): 0.5 ≤ score < 2.0 → top 2-3 collections
        - Phase 3 (High): score ≥ 2.0 → top 1-2 collections
        """
        ...

    def extract_concepts(self, text: str) -> List[str]:
        """Extract unigrams, bigrams, trigrams for KG routing."""
        ...

    def preprocess_query(self, query: str) -> str:
        """Expand acronyms, normalize whitespace."""
        ...
```

**Lines removed from UnifiedMemorySystem:** ~400

---

## Phase 5: Extract KnowledgeGraphService (Day 8)

### 5.1 Create `KnowledgeGraphService`
**New file:** `modules/memory/kg_service.py`

**CRITICAL: This phase fixes the race condition at lines 385-405**

**Extract from `unified_memory_system.py`:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `_load_kg()` | 312-336 | Load routing KG from disk |
| `_save_kg_sync()` | 352-378 | Sync save with file lock |
| `_save_kg()` | 380-383 | Async wrapper |
| `_debounced_save_kg()` | 385-405 | **RACE CONDITION** |
| `_update_kg_routing()` | 3092-3162 | Update patterns on outcome |
| `_build_concept_relationships()` | 3071-3090 | Build concept graph |
| `_track_problem_solution()` | 3365-3432 | Track problem→solution |
| `_find_known_solutions()` | 3295-3363 | Find known solutions |
| `get_kg_entities()` | 4524-4669 | Get entities for UI |
| `get_kg_relationships()` | 4671-4720 | Get relationships for UI |

**Race condition fix:**

```python
class KnowledgeGraphService:
    def __init__(self, data_dir: Path, config: MemoryConfig):
        self.data_dir = data_dir
        self.config = config
        self.kg_path = data_dir / "knowledge_graph.json"
        self.knowledge_graph = self._load_kg()

        # FIX: Add proper locking for async operations
        self._kg_lock = asyncio.Lock()
        self._kg_save_task: Optional[asyncio.Task] = None
        self._kg_save_pending = False

    async def debounced_save(self):
        """
        Debounce KG saves with proper locking to prevent race conditions.

        FIX: The original code (lines 385-405) had a race condition where:
        1. Task A starts delayed_save(), sleeps for 5s
        2. Task B cancels Task A, starts new delayed_save()
        3. If Task A was mid-write when cancelled, data could be lost

        Solution: Use asyncio.Lock() to ensure only one save operation
        can be in progress at a time.
        """
        async with self._kg_lock:
            # Cancel existing pending save
            if self._kg_save_task and not self._kg_save_task.done():
                self._kg_save_task.cancel()
                try:
                    await self._kg_save_task
                except asyncio.CancelledError:
                    pass

            async def delayed_save():
                try:
                    await asyncio.sleep(self.config.kg_debounce_seconds)
                    await self._save_kg()
                    self._kg_save_pending = False
                except asyncio.CancelledError:
                    pass

            self._kg_save_pending = True
            self._kg_save_task = asyncio.create_task(delayed_save())
```

**Lines removed from UnifiedMemorySystem:** ~600

---

## Phase 6: Extract SearchService (Day 9)

### 6.1 Create `SearchService`
**New file:** `modules/memory/search_service.py`

**Extract from `unified_memory_system.py`:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `search()` | ~700-900 | Main search with hybrid ranking |
| `search_books()` | 1759-1799 | Book-specific search |
| `_rerank_with_cross_encoder()` | ~1661 | Cross-encoder reranking |
| `_find_similar_by_embedding()` | ~850-910 | Deduplication check |

```python
class SearchService:
    def __init__(
        self,
        collections: Dict[str, ChromaDBAdapter],
        scoring_service: ScoringService,
        routing_service: RoutingService,
        kg_service: KnowledgeGraphService,
        embedding_service: EmbeddingServiceInterface,
        config: MemoryConfig,
        reranker: Optional[CrossEncoder] = None
    ):
        ...

    async def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        limit: int = 5,
        offset: int = 0,
        filters: Optional[Dict] = None,
        return_metadata: bool = False,
        transparency_context: Optional[Any] = None
    ) -> Union[List[Dict], Dict]:
        """
        Hybrid search with:
        1. KG-based routing (or explicit collection list)
        2. Vector similarity search
        3. BM25 fallback (if enabled)
        4. Wilson score ranking
        5. Cross-encoder reranking (if available)
        """
        ...
```

**Lines removed from UnifiedMemorySystem:** ~500

---

## Phase 7: Wire Up Dependency Injection (Day 10)

### 7.1 Update UnifiedMemorySystem constructor

```python
from core.interfaces.embedding_service_interface import EmbeddingServiceInterface
from core.interfaces.scoring_engine_interface import ScoringEngineInterface

class UnifiedMemorySystem:
    """
    Facade for the memory system.

    After refactoring, this class delegates to specialized services
    while maintaining the same public API for backwards compatibility.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        config: Optional[MemoryConfig] = None,
        # Optional DI for testing:
        scoring_service: Optional[ScoringService] = None,
        promotion_service: Optional[PromotionService] = None,
        routing_service: Optional[RoutingService] = None,
        kg_service: Optional[KnowledgeGraphService] = None,
        search_service: Optional[SearchService] = None,
        embedding_service: Optional[EmbeddingServiceInterface] = None,
    ):
        self.config = config or MemoryConfig()
        self.data_dir = Path(data_dir)

        # Initialize or inject services
        self.embedding_service = embedding_service or EmbeddingService()
        self.kg_service = kg_service or KnowledgeGraphService(
            self.data_dir, self.config
        )
        self.scoring_service = scoring_service or ScoringService(self.config)
        self.routing_service = routing_service or RoutingService(
            self.kg_service.knowledge_graph, self.config
        )
        # ... initialize remaining services
```

### 7.2 Leverage existing interfaces

The codebase already has interfaces in `core/interfaces/`:
- `embedding_service_interface.py`
- `scoring_engine_interface.py`
- `memory_adapter_interface.py`
- `vector_db_interface.py`

Update concrete classes to implement these interfaces for proper DI.

---

## Phase 8: Memory Bank Operations (Day 11)

### 8.1 Create `MemoryBankService`
**New file:** `modules/memory/memory_bank_service.py`

**Extract from `unified_memory_system.py`:**

| Method | Lines | Purpose |
|--------|-------|---------|
| `store_memory_bank()` | 4200-4279 | Store user memory |
| `update_memory_bank()` | 4281-4378 | Update with auto-archive |
| `archive_memory_bank()` | 4380-4416 | Soft delete |
| `search_memory_bank()` | 4418-4470 | Search with filters |
| `user_restore_memory()` | 4472-4494 | Restore archived |
| `user_delete_memory()` | 4496-4522 | Hard delete |

**Lines removed from UnifiedMemorySystem:** ~350

---

## Phase 9: Context & Outcome Tracking (Day 12)

### 9.1 Create `ContextService`
**New file:** `modules/memory/context_service.py`

**Extract:**
- `detect_context_type()` (lines 438-533)
- `analyze_conversation_context()` (lines 3164-3293)
- `_get_cold_start_context()` (lines ~1900-2011)
- `_format_cold_start_results()` (lines 2032-2080)

### 9.2 Create `OutcomeService`
**New file:** `modules/memory/outcome_service.py`

**Extract:**
- `record_outcome()` logic
- Cached doc_id tracking (`_cached_doc_ids`)
- Action effectiveness updates

**Lines removed from UnifiedMemorySystem:** ~400

---

## Final File Structure

```
modules/memory/
├── unified_memory_system.py  # ~800 lines (facade)
├── config.py                 # ~100 lines
├── types.py                  # ~150 lines
├── scoring_service.py        # ~250 lines
├── promotion_service.py      # ~400 lines
├── routing_service.py        # ~400 lines
├── kg_service.py             # ~600 lines
├── search_service.py         # ~500 lines
├── memory_bank_service.py    # ~350 lines
├── context_service.py        # ~300 lines
└── outcome_service.py        # ~200 lines
```

**Total:** ~4,050 lines across 11 files

---

## Critical Fixes Included

| Issue | Phase | Fix |
|-------|-------|-----|
| Race condition (lines 385-405) | Phase 5 | Add `asyncio.Lock()` to debounced_save |
| Magic numbers scattered | Phase 1 | Consolidate in `MemoryConfig` |
| JSON strings in ChromaDB | Phase 1 | Typed dataclasses with serialization |
| Untestable architecture | Phase 7 | Constructor injection via interfaces |
| Unused interfaces | Phase 7 | Implement existing `core/interfaces/` ABCs |

---

## Migration Strategy

### Step-by-Step Approach

1. **Extract one service at a time** - never do big bang refactoring
2. **Keep facade working** - `UnifiedMemorySystem` delegates to services
3. **Run benchmarks after each extraction** - ensure no regression
4. **Write tests for extracted services** - now possible with DI
5. **Delete old code only after tests pass**

### Testing Each Phase

```bash
# After each phase:
python -m pytest tests/memory/
python benchmarks/memory_benchmark.py

# Verify MCP still works:
python -c "from modules.memory import UnifiedMemorySystem; print('OK')"
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing behavior | Benchmark before/after each phase |
| MCP integration breaks | MCP calls facade (unchanged public API) |
| Data corruption | All services use same KG/ChromaDB paths |
| Merge conflicts | Feature branch, merge phases incrementally |
| Performance regression | Profile before/after, especially search() |

---

## Validation Checklist

After each phase, verify:

- [ ] All existing tests pass
- [ ] Benchmark scores unchanged
- [ ] MCP tools still work
- [ ] UI still functions
- [ ] No new errors in logs
- [ ] Memory usage acceptable

---

## Book References

This plan aligns with principles from:

**Clean Architecture (Robert Martin):**
> "The Dependency Rule: Source code dependencies must point only inward, toward higher-level policies."

**The Pragmatic Programmer (Hunt & Thomas):**
> "DRY - Don't Repeat Yourself"
> "Orthogonality - Eliminate effects between unrelated things"

**Designing Data-Intensive Applications (Kleppmann):**
> "The key to managing complexity is to keep the parts of the system as independent as possible."

---

## Part II: Risk-Ordered Action Plan

> **Book Guidance (The Pragmatic Programmer):**
> "Take short, deliberate steps: move a field from one class to another, fuse two similar methods into a superclass. Refactoring often involves making many localized changes that result in a larger-scale change. If you keep your steps small, and test after each step, you will avoid prolonged debugging."

> **Book Guidance (The Pragmatic Programmer):**
> "A regression test compares the output of the current test with previous (or known) values. We can ensure that bugs we fixed today didn't break things that were working yesterday. This is an important safety net."

---

### Pre-Flight: Establish Safety Net (Before Any Code Changes)

**Why first:** You cannot safely refactor without tests. The books are unanimous on this.

| Step | Action | Verification |
|------|--------|--------------|
| 0.1 | Run existing benchmarks, save baseline | `python benchmarks/memory_benchmark.py > baseline.txt` |
| 0.2 | Create characterization test for `search()` | Capture current output for 10 known queries |
| 0.3 | Create characterization test for `record_outcome()` | Capture score changes for known scenarios |
| 0.4 | Create characterization test for MCP tools | Test all 7 MCP tools return expected shapes |
| 0.5 | Commit baseline tests | `git commit -m "Add refactoring safety net tests"` |

**Exit criteria:** All characterization tests pass. This is your regression suite.

---

### Sprint 1: Zero-Risk Foundation (Days 1-3)

**Principle:** Add new files only. Touch ZERO existing code. No risk of breaking anything.

| Step | Action | Risk | Verification |
|------|--------|------|--------------|
| 1.1 | Create `modules/memory/config.py` with `MemoryConfig` | **NONE** | File exists, imports work |
| 1.2 | Create `modules/memory/types.py` with dataclasses | **NONE** | File exists, imports work |
| 1.3 | Create `modules/memory/scoring_service.py` (copy, don't move) | **NONE** | Unit tests pass |
| 1.4 | Write unit tests for `ScoringService.wilson_score_lower()` | **NONE** | Tests pass |
| 1.5 | Write unit tests for `ScoringService.calculate_final_score()` | **NONE** | Tests pass |

**Key insight:** At this point, `UnifiedMemorySystem` is unchanged. The new services exist alongside it. Nothing can break.

**Commit:** `git commit -m "Add config, types, and ScoringService (no integration yet)"`

---

### Sprint 2: First Integration - Scoring (Days 4-5)

**Principle:** Wire up one service. Keep fallback to old code available.

| Step | Action | Risk | Verification |
|------|--------|------|--------------|
| 2.1 | Add `MemoryConfig` import to `UnifiedMemorySystem.__init__` | **LOW** | Still uses class constants |
| 2.2 | Add `self.config = MemoryConfig()` | **LOW** | Neutral - not used yet |
| 2.3 | Replace `self.HIGH_VALUE_THRESHOLD` with `self.config.high_value_threshold` | **LOW** | Same values, just indirection |
| 2.4 | Repeat for all 5 threshold constants | **LOW** | Benchmarks unchanged |
| 2.5 | Add `ScoringService` instantiation to `__init__` | **LOW** | Not called yet |
| 2.6 | Replace inline Wilson calculation with `self.scoring_service.wilson_score_lower()` | **MEDIUM** | Run regression tests |
| 2.7 | Replace inline score calculation with `self.scoring_service.calculate_final_score()` | **MEDIUM** | Run benchmarks |

**Rollback point:** If benchmarks regress, revert 2.6-2.7 only. Config changes are safe.

**Commit:** `git commit -m "Integrate ScoringService into UnifiedMemorySystem"`

---

### Sprint 3: Fix the Race Condition (Day 6)

**Principle:** Fix the data corruption risk BEFORE extracting more services.

> **Book Guidance (DDIA):**
> "Concurrency issues (race conditions) only come into play when one transaction reads data that is concurrently modified."

| Step | Action | Risk | Verification |
|------|--------|------|--------------|
| 3.1 | Add `self._kg_lock = asyncio.Lock()` to `__init__` | **NONE** | Just a new attribute |
| 3.2 | Wrap `_debounced_save_kg` body in `async with self._kg_lock:` | **LOW** | Serializes saves |
| 3.3 | Stress test with concurrent writes | **LOW** | Run 100 parallel record_outcome calls |
| 3.4 | Verify KG file integrity after stress test | **LOW** | JSON loads without error |

**Why now:** This fix is small, isolated, and prevents silent data loss. Do it before extracting KG service so the fix is part of the extracted code.

**Commit:** `git commit -m "Fix race condition in _debounced_save_kg (lines 385-405)"`

---

### Sprint 4: Extract KnowledgeGraphService (Days 7-9)

**Principle:** Extract the most dangerous code next, while the fix is fresh.

| Step | Action | Risk | Verification |
|------|--------|------|--------------|
| 4.1 | Create `modules/memory/kg_service.py` (copy all KG methods) | **NONE** | File exists |
| 4.2 | Include the Lock fix from Sprint 3 | **NONE** | Already tested |
| 4.3 | Write unit tests for `KnowledgeGraphService` | **NONE** | Tests pass in isolation |
| 4.4 | Add `KnowledgeGraphService` instantiation to `UnifiedMemorySystem.__init__` | **LOW** | Not used yet |
| 4.5 | Replace `self.knowledge_graph` access with `self.kg_service.knowledge_graph` | **MEDIUM** | Run regression tests |
| 4.6 | Replace `self._debounced_save_kg()` with `self.kg_service.debounced_save()` | **MEDIUM** | Run stress test again |
| 4.7 | Delete old KG methods from `UnifiedMemorySystem` | **HIGH** | Full test suite + benchmarks |

**Rollback point:** If 4.7 causes issues, revert and keep both implementations temporarily.

**Commit:** `git commit -m "Extract KnowledgeGraphService with race condition fix"`

---

### Sprint 5: Extract RoutingService (Days 10-11)

**Principle:** Routing depends on KG, so extract after KG is stable.

> **Book Guidance (Clean Architecture):**
> "It is impossible to follow the dependency relationships and wind up back at that component. This structure has no cycles. It is a directed acyclic graph (DAG)."

**Dependency order matters:**
```
RoutingService → KnowledgeGraphService → (config)
```

| Step | Action | Risk | Verification |
|------|--------|------|--------------|
| 5.1 | Create `modules/memory/routing_service.py` | **NONE** | File exists |
| 5.2 | Pass `kg_service.knowledge_graph` to `RoutingService` constructor | **LOW** | Explicit dependency |
| 5.3 | Write unit tests with mock KG | **NONE** | Tests pass |
| 5.4 | Integrate into `UnifiedMemorySystem` | **MEDIUM** | Regression tests |
| 5.5 | Delete old routing methods | **HIGH** | Full test suite |

**Commit:** `git commit -m "Extract RoutingService"`

---

### Sprint 6: Extract SearchService (Days 12-14)

**Principle:** Search is the highest-risk extraction. It touches everything.

**Dependencies:**
```
SearchService → ScoringService
             → RoutingService
             → KnowledgeGraphService
             → EmbeddingService
             → Collections (ChromaDB)
             → Reranker (optional)
```

| Step | Action | Risk | Verification |
|------|--------|------|--------------|
| 6.1 | Create `modules/memory/search_service.py` | **NONE** | File exists |
| 6.2 | Constructor takes all dependencies explicitly | **LOW** | DI pattern |
| 6.3 | Write integration test with real ChromaDB | **MEDIUM** | Matches characterization test output |
| 6.4 | Wire into `UnifiedMemorySystem.search()` as delegate | **MEDIUM** | Benchmarks unchanged |
| 6.5 | **STOP. Run full benchmark suite.** | - | Must match baseline ±1% |
| 6.6 | If benchmarks pass, delete old search code | **HIGH** | Final verification |

**Why careful:** `search()` is called by MCP, UI, and internal promotion. Breaking it breaks everything.

**Commit:** `git commit -m "Extract SearchService"`

---

### Sprint 7: Extract PromotionService (Days 15-16)

**Principle:** Promotion is less risky - it runs in background, failures are recoverable.

| Step | Action | Risk | Verification |
|------|--------|------|--------------|
| 7.1 | Create `modules/memory/promotion_service.py` | **NONE** | File exists |
| 7.2 | Extract `_handle_promotion`, `_promote_valuable_working_memory` | **LOW** | Background task |
| 7.3 | Wire into `UnifiedMemorySystem` | **MEDIUM** | Promotion still happens |
| 7.4 | Delete old promotion methods | **MEDIUM** | Watch logs for errors |

**Commit:** `git commit -m "Extract PromotionService"`

---

### Sprint 8: Extract Remaining Services (Days 17-20)

**Principle:** Lower-risk extractions, can be done in parallel if needed.

| Service | Risk | Reason |
|---------|------|--------|
| `MemoryBankService` | **LOW** | Isolated CRUD operations |
| `ContextService` | **LOW** | Read-only analysis |
| `OutcomeService` | **MEDIUM** | Writes to KG, but already fixed |

**Commit each separately.**

---

### Sprint 9: Final Cleanup (Days 21-22)

| Step | Action | Verification |
|------|--------|--------------|
| 9.1 | Update all imports to use new services | All tests pass |
| 9.2 | Remove dead code from `UnifiedMemorySystem` | Line count ≤ 800 |
| 9.3 | Add deprecation warnings to old method stubs (if any) | Logs show warnings |
| 9.4 | Run full benchmark suite | Matches baseline |
| 9.5 | Run stress test (1000 operations) | No errors |
| 9.6 | Run MCP integration test | All 7 tools work |

---

## Dependency Graph (Extract in This Order)

```
                    ┌──────────────┐
                    │ MemoryConfig │  ← Extract FIRST (no dependencies)
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌──────────┐  ┌─────────────────┐
        │  types  │  │ Scoring  │  │ KnowledgeGraph  │ ← Fix race condition HERE
        │ .py     │  │ Service  │  │ Service         │
        └─────────┘  └────┬─────┘  └────────┬────────┘
                          │                  │
                          │    ┌─────────────┘
                          │    │
                          ▼    ▼
                    ┌──────────────┐
                    │   Routing    │  ← Depends on KG
                    │   Service    │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    Search    │  ← Depends on Scoring, Routing, KG
                    │    Service   │  ← HIGHEST RISK - extract last of core
                    └──────┬───────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   ┌───────────┐    ┌───────────┐    ┌───────────┐
   │ Promotion │    │  Context  │    │  Outcome  │  ← Lower risk, extract last
   │  Service  │    │  Service  │    │  Service  │
   └───────────┘    └───────────┘    └───────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │MemoryBank   │  ← Isolated, can extract anytime
                    │  Service    │
                    └──────────────┘
```

---

## Abort Conditions

**STOP and reassess if any of these occur:**

1. **Benchmark regression > 5%** - Something fundamental is wrong
2. **MCP tools return different shapes** - API contract broken
3. **KG file corruption** - Race condition not properly fixed
4. **Memory usage doubles** - Service instantiation overhead
5. **Two sprints without a green test suite** - Refactoring too fast

> **Book Guidance (The Pragmatic Programmer):**
> "If a bug slips through the net of existing tests, you need to add a new test to catch it next time."

---

## Definition of Done

The refactoring is complete when:

- [ ] `UnifiedMemorySystem` is ≤ 800 lines
- [ ] All 10 services exist and are unit tested
- [ ] Benchmark scores match baseline ±1%
- [ ] All 7 MCP tools pass integration tests
- [ ] Race condition fix is in production
- [ ] No cyclic dependencies between services
- [ ] Each service can be instantiated with mock dependencies

---

*Created: December 2024*
*Updated: December 9, 2024 (Added appendices A-G)*
*Status: Ready for implementation*

---

## Appendix A: Module-Level Items

| Item | Line | Destination |
|------|------|-------------|
| `CollectionName` | 45 | `types.py` |
| `wilson_score_lower()` | 47-90 | `ScoringService` |
| `ContextType = str` | 94 | `types.py` |
| `ActionOutcome` dataclass | 97-153 | `types.py` |
| `with_retry()` decorator | 155-175 | `utils.py` or facade |
| `AutonomousRouter = None` | 41 | **DELETE** (dead) |

---

## Appendix B: Instantiation Patterns

| Mode | File | Line | Pattern |
|------|------|------|---------|
| FastAPI | `main.py` | 259 | `app.state.memory` |
| MCP | `main.py` | 784 | Local var in `run_mcp_server()` |

---

## Appendix C: Uncovered Methods (Updated Dec 10, 2024)

> **Complete inventory of all 72 methods with service assignments**

### Facade Methods (remain in UnifiedMemorySystem)

| Method | Line | Notes |
|--------|------|-------|
| `__init__()` | 208 | Orchestrates service creation |
| `initialize()` | 535 | Async initialization |
| `set_llm_service()` | 428 | LLM injection |
| `store()` | 801 | Main storage entry point |
| `delete_by_conversation()` | 1831 | Cleanup |
| `load_conversation_history()` | 3781 | Delegates to adapter |
| `clear_session()` | 3796 | Session cleanup |
| `ingest_book()` | 3806 | Delegates to BookProcessor |
| `get_recent_conversation()` | 3811 | Query helper |
| `save_conversation_turn()` | 3886 | Conversation persistence |
| `get_stats()` | 3974 | System statistics |
| `mark_persistent()` | 4028 | Mark memory as persistent |
| `export_backup()` | 4040 | Backup export |
| `import_backup()` | 4062 | Backup import |
| `_startup_cleanup()` | 4081 | Init-time cleanup |
| `_doc_exists()` | 4182 | Helper |
| `cleanup()` | 4722 | Shutdown cleanup |

### ScoringService Methods

| Method | Line | Notes |
|--------|------|-------|
| `wilson_score_lower()` | 47 | Static function (module-level) |
| Score calculation in `search()` | 1514-1656 | Extract scoring logic |

### KnowledgeGraphService Methods

| Method | Line | Notes |
|--------|------|-------|
| `_load_kg()` | 312 | Load from disk |
| `_load_content_graph()` | 298 | Content graph loading |
| `_load_relationships()` | 338 | Relationship loading |
| `_save_kg_sync()` | 352 | Sync save with lock |
| `_save_kg()` | 380 | Async wrapper |
| `_debounced_save_kg()` | 385 | **RACE CONDITION FIX** |
| `_save_relationships_sync()` | 407 | Relationship persistence |
| `_save_relationships()` | 423 | Async wrapper |
| `_build_concept_relationships()` | 3071 | Concept graph building |
| `_update_kg_routing()` | 3092 | Update on outcome |
| `_track_problem_solution()` | 3365 | Problem→solution tracking |
| `_find_known_solutions()` | 3295 | Solution lookup |
| `_cleanup_kg_dead_references()` | 4099 | Dead reference cleanup |
| `cleanup_action_kg_for_doc_ids()` | 4141 | Action KG cleanup |
| `_add_relationship()` | 4002 | Add relationship |
| `_get_related_docs()` | 4010 | Get related docs |
| `get_kg_entities()` | 4524 | UI: entity list |
| `get_kg_relationships()` | 4671 | UI: relationship list |

### RoutingService Methods

| Method | Line | Notes |
|--------|------|-------|
| `_route_query()` | 2131 | Main routing logic |
| `_calculate_tier_scores()` | 2082 | Collection scoring |
| `get_tier_recommendations()` | 2802 | Recommendations for insights |
| `_extract_concepts()` | 2219 | N-gram extraction |
| `_preprocess_query()` | 1161 | Acronym expansion |
| `ACRONYM_DICT` | 1036-1156 | Class constant |

### SearchService Methods

| Method | Line | Notes |
|--------|------|-------|
| `search()` | 1205 | Main hybrid search |
| `search_books()` | 1759 | Book-specific search |
| `_rerank_with_cross_encoder()` | 656 | Cross-encoder reranking |
| `_calculate_entity_boost()` | 749 | Entity quality boost |
| `get_cold_start_context()` | 1884 | Cold start handling |
| `_smart_truncate()` | 2013 | Text truncation |
| `_format_cold_start_results()` | 2032 | Format cold start |
| `get_facts_for_entities()` | 2862 | Entity fact lookup |

### PromotionService Methods

| Method | Line | Notes |
|--------|------|-------|
| `_promote_item()` | 2921 | Move between collections |
| `_handle_promotion()` | 2959 | Auto promotion/demotion |
| `_handle_promotion_error()` | 3482 | Error callback |
| `_promote_valuable_working_memory()` | 3489 | Batch promotion |
| `cleanup_old_working_memory()` | 3437 | Age-based cleanup |
| `clear_old_history()` | 3673 | History cleanup |
| `clear_old_working_memory()` | 3721 | Working memory cleanup |

### ContextService Methods

| Method | Line | Notes |
|--------|------|-------|
| `detect_context_type()` | 438 | Context detection |
| `analyze_conversation_context()` | 3164 | Conversation analysis |
| `_generate_contextual_prefix()` | 571 | Prefix generation |
| `get_working_context()` | 3837 | Working context retrieval |

### OutcomeService Methods

| Method | Line | Notes |
|--------|------|-------|
| `record_outcome()` | 2317 | Main outcome recording |
| `record_action_outcome()` | 2619 | Action-specific outcomes |
| `get_action_effectiveness()` | 2713 | Effectiveness query |
| `should_avoid_action()` | 2733 | Avoidance check |
| `get_doc_effectiveness()` | 2763 | Doc-level effectiveness |
| `_track_usage()` | 2305 | Usage tracking |
| `_check_implicit_outcomes()` | 3567 | Implicit outcome detection |
| `detect_conversation_outcome()` | 3932 | Conversation outcome |

### MemoryBankService Methods

| Method | Line | Notes |
|--------|------|-------|
| `store_memory_bank()` | 4200 | Store user memory |
| `update_memory_bank()` | 4281 | Update with auto-archive |
| `archive_memory_bank()` | 4380 | Soft delete |
| `search_memory_bank()` | 4418 | Search with filters |
| `user_restore_memory()` | 4472 | Restore archived |
| `user_delete_memory()` | 4496 | Hard delete |

### Module-Level Items

| Item | Line | Destination |
|------|------|-------------|
| `CollectionName` | 45 | `types.py` |
| `ContextType = str` | 94 | `types.py` |
| `ActionOutcome` dataclass | 97-153 | `types.py` |
| `with_retry()` decorator | 155-175 | `utils.py` |
| `AutonomousRouter = None` | 41 | **DELETE** (dead code) |

---

## Appendix D: Error Handling (14 bare except:)

Lines: 334, 344, 722, 1384, 1493, 1537, 3061, 3532, 3701, 3748, 3997, 4057, 4192, 4743

**Fix:** Replace with specific exceptions + logging.

---

## Appendix E: Additional Risks

| Risk | Mitigation |
|------|------------|
| Circular imports | Use `TYPE_CHECKING` |
| `sys.path.insert()` fragility | Test from multiple dirs |
| ChromaDB lock contention | Share collection dict |

---

## Appendix F: Attribute Ownership

| Attribute | Owner |
|-----------|-------|
| `self.data_dir` | Facade |
| `self.collections` | Facade (shared) |
| `self.reranker` | `SearchService` |
| `self.knowledge_graph` | `KGService` |
| `self._promotion_lock` | `PromotionService` |
| `self._kg_save_*` | `KGService` |
| `self._cached_doc_ids` | `OutcomeService` |
| `self.outcome_*` | `OutcomeService` |

---

## Appendix G: Characterization Queries

```python
QUERIES = [
    "how do I search memory",
    "python async patterns",
    "",  # empty
    "user preference for dark mode",
    "API rate limiting",
    "remember my name is John",
]
```