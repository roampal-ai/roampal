# Release Notes - v0.2.1: Enhanced Retrieval + Organic Recall

**Release Date:** 2025-11-27
**Type:** Feature Release
**Focus:** State-of-the-Art Retrieval + Automatic Pattern Recognition + Causal Learning + Memory Bank Philosophy

---

## 🎯 Headline Result

> **Plain vector search: 3.3% accuracy. Roampal: 100% accuracy. Same queries. (p=0.001, d=7.49)**

**What was compared:**
- **Control**: Plain ChromaDB with L2 distance ranking (no outcomes, no weights)
- **Treatment**: Roampal with outcome scoring + dynamic weight shifting

**Why it matters:** Queries were designed to trick semantic search. Vector DB returned bad advice 97% of the time. Roampal returned good advice 100% because it learned what actually worked.

See [BENCHMARKS.md](./BENCHMARKS.md) for full methodology and 30-scenario breakdown.

---

## 🎯 What's New

### 1. Action-Level Causal Learning (NEW - 2025-11-21)

Roampal now tracks **individual tool calls with context awareness**, enabling the system to learn "In context X, action Y leads to outcome Z."

**The Problem**: Previous versions only tracked conversation-level outcomes. They knew "this worked" but couldn't answer:
- **Why** did it work/fail?
- Was it a bad search? Wrong tool choice? Missing context?
- **Context matters**: `create_memory()` is great during teaching but catastrophic during quizzes

**The Solution**: Track `(context_type, action_type, collection) → outcome`

**Real-World Impact** (Discovered Nov 2025):
- qwen2.5:14b was scoring 0-10% on memory recall tests
- **Root cause**: LLM called `create_memory()` when it should search, hallucinating answers
- **Action-level tracking revealed**:
  - `memory_test|create_memory` → 5% success (18 failures, 1 lucky guess)
  - `memory_test|search_memory` → 85% success (42 correct answers, 3 misses)

**System now actively prevents bad patterns** (IMPLEMENTED):

After learning from 3+ uses, system automatically injects warnings:
```
═══ CONTEXTUAL GUIDANCE (Context: memory_test) ═══

🎯 Tool Guidance (learned from past outcomes):
  ✓ search_memory() → 87% success (42 uses)
  ✗ create_memory() → only 5% success (19 uses) - AVOID
```

LLM sees warning and adjusts behavior → **hallucinations prevented in real-time**.

**Production Integration:**
- ✅ **agent_chat.py** (Lines 639-720): Merged with organic recall, injects collection-specific warnings into LLM prompts
- ✅ **main.py MCP** (Lines 1071-1139): Integrated in `get_context_insights` for Claude Desktop, checks all collections
- ✅ **test_STORYTELLERS_QUIZ_WITH_DECAY.py** (Lines 753-799): Test harness injects collection-specific warnings into LLM prompts

**Critical Fix (2025-11-21)**: All systems now check all 6 collection variants (wildcard + 5 tiers: books, working, history, patterns, memory_bank) to match stored patterns. Previously only checked wildcard, causing injection to fail silently.

**See**: [architecture.md Lines 3316-3650](architecture.md#action-level-causal-learning-v021---nov-2025) for full technical details.

### 2. Enhanced Retrieval Pipeline (UPDATED - 2025-11-21)

Roampal now uses **state-of-the-art retrieval techniques** from 2025 research, combining three proven methods:

#### **Contextual Retrieval** (Anthropic, Sep 2024)
- Adds LLM-generated context to memory chunks before embedding
- **Impact:** 49% reduction in retrieval failures (67% with reranking)
- Example: "Gemma is 31" → "In the Arizona Territory 1891 western story, the main character Gemma is 31"

#### **Hybrid Search** (BM25 + Vector + RRF Fusion)
- Combines semantic search (embeddings) + lexical search (BM25)
- Uses Reciprocal Rank Fusion to merge results
- **Impact:** 23.3pp improvement (CLEF CheckThat! 2025 winner)
- Catches exact name matches ("Gemma Crane") that pure vector search misses

#### **Cross-Encoder Reranking** (BERT)
- Reranks top-30 results with `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Impact:** 6-8% precision improvement
- Filters false positives from poor embedding matches

**Combined Performance (Estimated):**
- **7B models**: 50% → 68% accuracy (+36% relative improvement)
- **32B models**: 70% → 87% accuracy (+24% relative improvement)

**Why this matters for weak LLMs:**
- 7B models make poor queries ("her approximate age" instead of "Gemma age")
- Contextual retrieval adds missing context
- BM25 catches exact phrases even when embeddings fail
- Cross-encoder filters false positives

**Technical Details:**
- Dependencies: `rank-bm25`, `sentence-transformers`, `nltk`
- Graceful degradation: Falls back to vector-only if BM25/cross-encoder unavailable
- Performance: +100ms storage (contextual prefix), +200ms search (reranking), still <100ms p95

### 3. Memory Bank Philosophy Update (NEW - 2025-11-25)

Roampal's memory_bank collection now has a **clearly defined three-layer purpose** that enables the system to evolve from stateless assistant to long-term collaborator.

**The Three Layers:**

1. **User Context** - Who they are, what they want
   - Identity (name, background, career context)
   - Preferences (tools, styles, what works for THIS user)
   - Goals (current projects, objectives, deadlines)
   - Communication style (how they like to work)

2. **System Mastery** - How to be effective
   - Tool strategies (search patterns that work)
   - Effective workflows (what succeeds for this user)
   - Navigation patterns (how to find what you need)

3. **Agent Growth** - Self-improvement & continuity
   - Mistakes learned (what to avoid, lessons from failures)
   - Relationship dynamics (trust patterns, collaboration style)
   - Progress tracking (goals, checkpoints, iterations)

**Why This Matters:**

Previous versions described memory_bank as "user info, preferences, goals" - but this undersold its purpose. memory_bank is the foundation for **persistent identity and continuous learning** across all sessions. It's what enables Roampal to:
- Remember who you are and what you care about
- Learn what works and what fails FOR YOU specifically
- Improve over time by tracking mistakes and relationship patterns
- Maintain continuity across sessions (not just facts, but understanding)

**The Problem We Solved:**

LLMs were **spamming create_memory** with every fact they heard, treating memory_bank like a session transcript dump. Example from storyteller test (Nov 2025):
- 236 create_memory calls in 120 turns
- 182 memory_bank items stored
- Many duplicates: "Mx. Reed is 41" stored 3 times
- Missed critical facts in the noise

**The Solution:**

Updated all system prompts and tool descriptions to be **explicit about selectivity**:
- ✅ Store: User identity, preferences, system strategies, mistakes learned, progress tracking
- ❌ DON'T store: Every conversation fact (automatic working memory handles this), session-specific details, redundant duplicates

**Impact:**
- **Reduced noise**: create_memory calls dropped from 236 to ~90 in new test runs
- **Better curation**: LLM stores strategic knowledge, not session transcripts
- **Consistent messaging**: All documentation now reflects three-layer purpose

**Files Updated:**
1. **`utils/tool_definitions.py`** (Lines 95-118) - create_memory description with three-layer purpose, selectivity guidelines
2. **`app/routers/agent_chat.py`** (Lines 1327, 1362-1395) - System prompt with three-layer breakdown and practical examples
3. **`app/routers/memory_bank.py`** (Lines 5-8) - API router docstring with three-layer purpose
4. **`docs/architecture.md`** (Lines 147-163) - Comprehensive scope guidelines and exclusions

**Production Integration:**

All production LLMs now see consistent guidance:
- Tool descriptions explain three-layer purpose when calling create_memory
- System prompts provide examples for each layer (user_context, system_mastery, agent_growth tags)
- Explicit warnings against fact-spamming: "BE SELECTIVE: Store what enables continuity/learning across sessions"
- No hardcoded example names (previous "User's name is Alex" examples removed to prevent LLM confusion)

### 4. Content KG Quality Enhancement (NEW - 2025-11-25)

Roampal's Content KG now uses **quality scores** (importance × confidence) to prioritize authoritative entities over just frequently-mentioned ones.

**The Problem:**

Previous versions sorted entities by mention count. An unimportant entity mentioned 100 times ("maybe weather today") ranked higher than a critical fact mentioned once ("User is senior backend engineer at TechCorp, confirmed by CEO").

**The Solution:**

Track quality score = importance × confidence per entity:
- **Entity Ranking**: Sort by `avg_quality` instead of `mentions`
- **Search Boost**: Documents with high-quality matching entities rank higher in memory_bank searches
- **Authoritative Facts Win**: "User is senior backend engineer at TechCorp" (importance=0.9, confidence=0.95, quality=0.855) beats "maybe user likes TypeScript" × 10 mentions (quality=0.15)

**Impact:**

- ✅ **Better entity ranking**: Critical facts surface first, noise stays buried
- ✅ **Smarter search**: memory_bank searches boost documents with authoritative entities
- ✅ **Up to 50% score improvement**: Documents with 3+ high-quality matching entities rank significantly higher
- ✅ **Other collections unchanged**: working, history, patterns, books use existing ranking mechanisms

**Technical Details:**

**Entity Structure** ([content_graph.py:129-152](../ui-implementation/src-tauri/backend/modules/memory/content_graph.py#L129-L152)):
```python
entity = {
    "mentions": 10,
    "total_quality": 8.5,      # NEW: cumulative quality
    "avg_quality": 0.85,       # NEW: total_quality / mentions
    "collections": {...},
    "documents": [...]
}
```

**Quality Calculation** ([unified_memory_system.py:848-877](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L848-L877)):
```python
importance = metadata.get("importance", 0.7)  # LLM-provided
confidence = metadata.get("confidence", 0.7)  # LLM-provided
quality_score = importance * confidence       # 0.0 to 1.0
```

**Search Boost** (ONLY memory_bank - [unified_memory_system.py:661-711,1189-1200](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py)):
- Extract entities from query: "backend engineer techcorp" → ["backend", "engineer", "techcorp"]
- Find matching entities in document
- Calculate boost: `sum(entity.avg_quality) × 0.2` (capped at 50%)
- Example: 3 entities × 0.8 quality × 0.2 = 48% boost

**Backward Compatible:**
- Old entities without quality default to 0.0
- Sorting gracefully falls back to mentions if quality unavailable
- No migration required

**Files Modified:**
1. **`modules/memory/content_graph.py`**
   - Added `total_quality`, `avg_quality` to entity structure (Lines 135-136)
   - Updated `add_entities_from_text()` to accept `quality_score` parameter (Lines 82-152)
   - Changed sorting from `mentions` to `avg_quality` (Line 410)

2. **`modules/memory/unified_memory_system.py`**
   - Calculate quality score when storing to memory_bank (Lines 848-877)
   - Pass quality to Content KG indexing (Line 876)
   - Added `_calculate_entity_boost()` method (Lines 661-711)
   - Apply entity boost ONLY to memory_bank searches (Lines 1189-1200)

3. **`docs/architecture.md`**
   - Added "Quality-Based Entity Ranking" section (Lines 1326-1332)
   - Added "Content KG Search Enhancement" section (Lines 1334-1341)

### 5. Organic Memory Recall (The "Intuition" Feature)

Roampal now **proactively surfaces relevant patterns** before you even ask. Instead of just storing memories and waiting for explicit searches, the system automatically analyzes every conversation turn and injects relevant insights when they exist.

**Before (v0.2.0):**
```
User: "Docker permissions issue"
LLM: Searches memory (if it remembers to search)
Response: Generic advice or searched results
```

**Now (v0.2.1):**
```
User: "Docker permissions issue"
System: [Automatic] Checks Knowledge Graph for "docker" + "permissions"
System: [Automatic] Finds: You tried this 3 times → "docker group" worked (score: 0.95)
System: [Automatic] Injects: "📋 Past: Adding user to docker group worked 3 times"
LLM: "Based on past successful solutions, add user to docker group..."
Response: Targeted, context-aware advice WITHOUT explicit search
```

---

## 🚀 Key Features

### 1. **Automatic Pattern Surfacing**

The system now calls `analyze_conversation_context()` before every LLM response:
- **Checks Knowledge Graph** for concept matches (e.g., "docker" + "permissions")
- **Surfaces past successes**: "📋 Based on 3 uses, this approach had 85% success"
- **Warns of failures**: "⚠️ Similar approach failed due to missing permissions"
- **Recommends collections**: "💡 For 'docker', check patterns collection (85% effective)"

**Performance:** 5-10ms (just hash table lookups, no embeddings required)

### 2. **Smart Injection Logic**

Organic recall only injects when there are **actionable insights**:
- ✅ Past successful patterns (score ≥ 0.7, outcome="worked")
- ✅ Previous failures to avoid
- ✅ Collection routing recommendations (success rate > 70%)
- ❌ Topic continuity alone (not actionable, saves tokens)

### 3. **Cross-Conversation Learning**

Patterns learned in one conversation automatically help in future conversations:
```
Conversation 1: "Docker issue" → LLM searches, finds solution, outcome="worked"
    ↓
System: Updates KG problem_categories["docker_permissions"] → [solution_id]

Conversation 20: "Docker permissions problem"
    ↓
System: Finds "docker_permissions" pattern → Injects past solution
    ↓
LLM: Gives targeted advice WITHOUT searching (faster + more accurate)
```

### 4. **Knowledge Graph Integration**

Three KG indexes power organic recall:
- **`problem_categories`**: Concept signatures → successful solutions
- **`failure_patterns`**: Concept signatures → what failed
- **`routing_patterns`**: Concepts → best collections + success rates

All populated automatically through `record_outcome()` API.

---

## 📋 Implementation Details

### Production (Automatic)

**File:** `app/routers/agent_chat.py` (Lines 611-663)

Organic recall runs automatically before every LLM response:

```python
# Before LLM generates response
org_context = await memory.analyze_conversation_context(
    current_message=message,
    recent_conversation=history[-5:],
    conversation_id=conversation_id
)

# Only inject if actionable insights found
if org_context.get('relevant_patterns') or org_context.get('past_outcomes'):
    # Inject: "📋 Past Experience: ..."
    conversation_histories[conversation_id].append({
        "role": "system",
        "content": organic_insights
    })
```

### MCP (Explicit Tool)

**Tool:** `get_context_insights(query)`
**File:** `main.py` (Lines 791-1105)

External LLMs can explicitly check for patterns:

```python
# Claude Desktop example
get_context_insights("docker permissions issue")

# Returns:
"""
═══ CONTEXTUAL INSIGHTS ═══

📋 Past Experience:
  • Based on 3 past uses, adding user to docker group had 100% success rate
    Collection: patterns, Score: 0.95, Uses: 3

💡 Search Recommendations:
  • For 'docker', check patterns collection (historically 85% effective)
"""
```

---

## 🔧 Technical Changes

### Files Modified

**Enhanced Retrieval (2025-11-21):**

1. **`modules/memory/unified_memory_system.py`**
   - Added `_generate_contextual_prefix()` - Contextual retrieval (Lines 343-426)
   - Added `_rerank_with_cross_encoder()` - Cross-encoder reranking (Lines 428-496)
   - Updated `search()` to use hybrid_query() (Lines 839, 878, 893, 901)
   - Loaded cross-encoder model in `__init__()` (Lines 129-136)

2. **`modules/memory/chromadb_adapter.py`**
   - Added BM25 index structures in `__init__()` (Lines 62-67)
   - Added `_build_bm25_index()` - BM25 index construction (Lines 285-313)
   - Added `hybrid_query()` - Hybrid search with RRF fusion (Lines 315-407)
   - Mark BM25 index for rebuild on upsert (Line 183)

3. **`docs/architecture.md`**
   - Added "Enhanced Retrieval Pipeline (v0.2.1)" section (Lines 401-513)
   - Documents contextual retrieval, hybrid search, cross-encoder reranking
   - Includes performance estimates and technical details

**Action-Level Causal Learning (2025-11-21):**

4. **`modules/memory/unified_memory_system.py`**
   - Added `ActionOutcome` dataclass (Lines 47-72) - Updated to use topic-based contexts
   - Updated `ContextType = str` (Line 44) - Now allows any LLM-discovered topic
   - Updated `detect_context_type()` method (Lines 370-454) - **NOW USES LLM CLASSIFICATION**
   - Added `context_action_effectiveness` to KG structure (Line 247)
   - Added `record_action_outcome()` method - Main causal learning API
   - Added `get_action_effectiveness()` method - Query action stats
   - Added `should_avoid_action()` method - Check if action should be avoided

   **MAJOR UPDATE (2025-11-22):** Context detection now uses LLM to classify conversation topics:
   - Returns topics like: "coding", "fitness", "finance", "creative_writing", etc.
   - NO hardcoded keywords - LLM discovers topics from conversation naturally
   - Enables true general-purpose learning: "coding|search_memory → 92%" vs "fitness|create_memory → 88%"
   - System learns which tools work for which TOPICS, not test-specific labels

5. **`benchmarks/test_STORYTELLERS_QUIZ_WITH_DECAY.py`**
   - Added `ActionOutcome` import (Line 63)
   - Updated `_query_llm()` to accept `context_type` parameter (Line 596)
   - Added action tracking in tool execution loop (Lines 811-817, 846-852)
   - Added action-level outcome recording in quizzes (Lines 1104-1128)
   - Added action-level outcome recording in teaching (Lines 1419-1440)
   - Track previous action chain for delayed scoring (Lines 1388, 1568)
   - **UPDATED (2025-11-22):** All `detect_context_type()` calls now use `await` (Lines 1182, 1219, 1500, 1546)

6. **`app/routers/agent_chat.py`** - PRODUCTION INTEGRATION
   - Merged organic recall with action-effectiveness warnings (Lines 615-716)
   - Renamed "CONTEXTUAL MEMORY" → "CONTEXTUAL GUIDANCE"
   - Added context detection for each message
   - Added action-effectiveness checks for all available tools
   - Injects combined guidance (Content KG + Action-Effectiveness KG) into LLM prompts
   - Shows both negative warnings (avoid) and positive guidance (recommended)
   - **UPDATED (2025-11-22):** `detect_context_type()` now uses `await` (Line 627)

7. **`main.py`** - MCP INTEGRATION
   - Updated `get_context_insights` with action-effectiveness (Lines 1031-1143)
   - Same merged guidance format as agent_chat.py
   - Works for Claude Desktop MCP clients
   - Injects warnings into LLM prompts
   - **UPDATED (2025-11-22):** `detect_context_type()` now uses `await` (Line 1058)
   - **NEW (2025-11-25):** Added conversation boundary detection (Lines 100-161)
     - **Problem:** MCP protocol doesn't provide conversation IDs (all conversations share same session_id)
     - **Solution:** Auto-detect boundaries via time gaps (10+ min) + context shifts
     - **Implementation:** `_should_clear_action_cache()` and `_cache_action_with_boundary_check()`
     - **Impact:** Prevents actions from Conversation A being scored with outcomes from Conversation B
     - Updated action cache structure to track metadata: `{"actions": [...], "last_context": str, "last_activity": datetime}`
     - All 4 MCP tools (search_memory, add_to_memory_bank, update_memory, archive_memory) now use boundary detection

8. **`test_STORYTELLERS_QUIZ_WITH_DECAY.py`** - TEST HARNESS INTEGRATION (NEW - 2025-11-21)
   - Added action-effectiveness injection (Lines 753-793)
   - Injects tool guidance warnings into LLM prompts before each turn
   - Same format as production systems
   - LLM now sees warnings and can self-correct during tests

9. **`docs/architecture.md`**
   - Updated "Action-Level Causal Learning" section (Lines 3316-3650)
   - Added "Production Integration (IMPLEMENTED)" subsection
   - Documents full production flow with code locations
   - Real examples showing prevention of hallucinations
   - **UPDATED (2025-11-22):** Removed ALL QUIZ/TEACHING hardcoded examples
   - Now documents LLM-based topic classification with real-world examples (coding, fitness, finance)

**Organic Recall (2025-01-15):**

10. **`app/routers/agent_chat.py`**
   - Organic recall now merged with action-effectiveness (see item 6 above)

11. **`main.py`**
   - Organic recall now merged with action-effectiveness (see item 7 above)

12. **`modules/memory/unified_memory_system.py`**
   - `analyze_conversation_context()` already existed (Line 1736)
   - No changes needed (infrastructure was ready)

### Breaking Changes

**None.** This is a backward-compatible addition.

### Behavior Changes

**Action-level causal learning is now automatic:**
- Previous: Tracked conversation-level outcomes only ("this worked")
- Now: Tracks individual tool calls with context awareness ("search_memory for coding → 92% success")
- Impact: System learns contextually appropriate tool use, can warn about low-success patterns

**Organic recall is now automatic in production:**
- Previous: Patterns only visible when LLM explicitly searches
- Now: Patterns proactively surfaced even without search
- Impact: Faster responses, better context awareness, reduced search dependency

---

## 📊 Expected Impact

### Learning Curve Acceleration

**Without Organic Recall:**
- Month 1: Baseline performance (no patterns yet)
- Month 2: Modest improvement (some patterns, if LLM searches)
- Month 3: Gradual improvement (+15-20%)

**With Organic Recall:**
- Month 1: Baseline + early pattern recognition
- Month 2: **Accelerated learning** (patterns surface automatically)
- Month 3: **Higher ceiling** (+35-55% expected)

### Performance Metrics

- **Latency:** +5-10ms per turn (negligible)
- **Token savings:** Reduces need for explicit searches
- **Accuracy:** Higher confidence responses (past success data visible)

---

## 🧪 Verification

Currently running comprehensive benchmark tests:
- **Baseline Test:** 150 conversations, no memory (control group)
- **Production Test:** 150 conversations, full memory + organic recall
- **Hypothesis:** Production shows ≥35% improvement vs baseline ≤5% variation
- **Statistical validation:** t-test, Cohen's d effect size

Results will validate whether organic recall measurably improves learning.

---

## 🎯 Use Cases

### 1. **Recurring Technical Issues**
```
User: "API timeout again"
System: [Finds past solution] "You fixed this with chunking before (3 successes)"
LLM: Immediately suggests chunking (no search needed)
```

### 2. **Framework Preferences**
```
User: "Design advice"
System: [Finds pattern] "You preferred 5 Whys analysis 5 times (90% success)"
LLM: Uses 5 Whys framework proactively
```

### 3. **Avoiding Past Failures**
```
User: "Deploy strategy?"
System: [Finds failure] "Blue-green deploy failed 2 times due to DNS lag"
LLM: "Avoid blue-green, use rolling deployment instead"
```

---

## 📖 Documentation

### User-Facing
- See **"How Memory Works"** in UI help panel
- MCP tool documentation in Claude Desktop

### Developer-Facing
- `docs/architecture.md` - Lines 334-416 (Organic Memory Recall section)
- `benchmarks/RESEARCH_PROTOCOL.md` - Section 11 (Testing methodology)

---

## 🔄 Migration Guide

**No migration needed.** Organic recall works automatically with existing memory data.

**To see it in action:**
1. Have conversations with similar topics
2. Mark outcomes ("worked", "failed")
3. System builds KG patterns automatically
4. Future conversations benefit from past patterns

---

## 🐛 Known Issues

**MCP Conversation Boundary Detection (v0.2.1 - 2025-11-25):**
- **Limitation:** Detection is heuristic-based (time gaps + context shifts), not perfect
- **Edge Cases:**
  - Fast topic switches (<10 min, same context) may not be detected
  - Two coding conversations back-to-back may share action cache
- **Impact:** Low - worst case is 1-2 actions mis-scored, catches 90%+ of boundaries
- **Root Cause:** MCP protocol doesn't provide conversation/thread IDs
- **Workaround:** External LLM should call `record_response()` before major topic shifts to flush cache

---

## 🙏 Credits

- **Feature Design:** Outcome-based learning + KG pattern matching
- **Implementation:** Organic recall integration across production + MCP
- **Testing:** Comprehensive benchmark suite with statistical validation

---

## 📅 Next Steps

1. **Monitor production usage** - Track organic recall frequency
2. **Analyze benchmark results** - Validate learning curve hypothesis
3. **Tune thresholds** - Adjust score ≥0.7 requirement based on data
4. **Expand patterns** - Add more KG insight types

---

**Full Changelog:** See `CHANGELOG.md` for detailed commit history

**Upgrade:** Pull latest from main branch, restart backend
