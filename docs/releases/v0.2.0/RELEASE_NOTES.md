# Roampal v0.2.0 - Learning-Based Knowledge Graph Routing

**Release Date:** November 5, 2025
**Build:** `Roampal_0.2.0_x64-setup.exe`


## 🚀 Major Features

### Memory Bank Metadata Standardization (v0.2.0)
- **Standardized Parameters**: Both Internal and MCP tools now have identical memory_bank parameters
  - `content` (required) - The fact/information to store
  - `tags` (optional, array) - Categories (e.g., identity, preference, goal, context)
  - `importance` (optional, default: 0.7) - How critical is this fact
  - `confidence` (optional, default: 0.7) - How certain about this fact
- **Internal Tool**: `create_memory` changed from singular `tag` to plural `tags` array
- **MCP Tool**: `add_to_memory_bank` added `confidence` parameter
- **No Scoring**: memory_bank stores user facts/useful information with metadata, NOT scored
- **Outcome-Based Scoring**: Only applies to working/history/patterns collections
  - ✅ `worked`: +0.2 score
  - ❌ `failed`: -0.3 score
  - ⚠️ `partial`: +0.05 score

### Learning-Based Knowledge Graph Routing
- **NEW**: Pure learning-based routing with ZERO hardcoded keywords
- **How It Works**: Learns which memory tiers (books, working, history, patterns, memory_bank) successfully answer which types of queries
- **N-gram Concept Extraction**: Unigrams, bigrams, trigrams extracted from queries without stopword filtering
- **Confidence Evolution**:
  - **Cold Start** (0-10 queries): Explores all 5 tiers
  - **Learning** (10-20 queries): Focuses on top 2-3 tiers based on success patterns
  - **Confident** (20+ queries): Routes to single best tier with 80%+ success rate
- **Safety Fallback**: Automatically expands to all tiers if <3 results found

### MCP Integration - Semantic Learning Storage (v0.2.0)
- **MAJOR**: `record_response` now uses semantic learnings with **explicit outcome scoring**
  - **Two Parameters**: `key_takeaway` (semantic summary, required) + `outcome` (explicit scoring, optional, defaults to "unknown")
  - **Semantic Storage**: External LLM summarizes what was learned (1-2 sentences)
  - **Explicit Outcome**: External LLM provides explicit outcome based on user feedback (worked/failed/partial/unknown)
  - **Why Explicit vs Auto-Detection**: External LLMs don't have full context of Roampal's internal conversations, so explicit scoring is more reliable
  - **Scores CURRENT not PREVIOUS**: Unlike internal system, MCP scores the learning being recorded immediately (allows optional tool calling - LLM only calls when clear outcomes)
  - **Why Semantic Instead of Verbatim**:
    - LLMs excel at summarization (core strength)
    - Avoids truncation/paraphrasing errors from copying exact text
    - Better for memory search (semantic concepts vs exact phrases)
    - Focuses on learnings, not chat logs
  - **How It Works**:
    1. External LLM calls `record_response(key_takeaway="User confirmed deploy worked using docker-compose.", outcome="worked")`
    2. Calculates initial score from outcome: worked=0.7, failed=0.2, partial=0.55, unknown=0.5
    3. Stores semantic summary to working memory with calculated score
    4. Scores all retrieved memories from last search with same outcome (upvote helpful, downvote bad advice)
    5. Updates KG routing if query was cached (learns which collections answer which queries)
  - **Example**:
    - Tool call: `key_takeaway="Suggested docker-compose for deployment. User confirmed it worked perfectly."`, `outcome="worked"`
    - Result: Stored with initial score 0.7 (0.5 base + 0.2 for worked outcome)
    - Optional: LLM can skip calling when no clear outcome occurs
- **REMOVED**: `list_memory_bank` tool (redundant - use `search_memory` with `collections=["memory_bank"]`)
- **REMOVED**: KG query tools (users explore KG via Roampal UI)
- **FIXED**: `collections=["all"]` bug - now properly triggers KG routing
- **Principle**: External LLMs function identically to internal Roampal LLM
- **Impact**: Cross-client knowledge sharing with automatic promotion (Claude → Cursor → Roampal)

## 🔧 Technical Implementation

### N-gram Concept Extraction
Location: `unified_memory_system.py:930-975`

```python
# Example: "show me books about investing"
Unigrams:  ["show", "me", "books", "about", "investing"]
Bigrams:   ["show_me", "me_books", "books_about", "about_investing"]
Trigrams:  ["show_me_books", "me_books_about", "books_about_investing"]
Technical: ["CamelCase", "snake_case", "ErrorTypes"]
→ Returns top 15 concepts (prioritizes longer phrases)
```

### Scoring Formula
Location: `unified_memory_system.py:896-908`

```python
success_rate = successes / (successes + failures)  # Excludes partials
confidence = min(total_uses / 10.0, 1.0)           # Grows with usage
tier_score = success_rate * confidence

# Example:
# "books" concept: 8 successes, 2 failures, 10 total uses
# books_tier_score = (8/10) * min(10/10, 1.0) = 0.8 * 1.0 = 0.8
```

### Routing Thresholds
Location: `unified_memory_system.py:909-928`

- `total_score < 0.5`: Search all 5 tiers (exploration)
- `0.5 ≤ total_score < 2.0`: Top 2-3 tiers (medium confidence)
- `total_score ≥ 2.0`: Top 1-2 tiers (high confidence)

### Safety Fallback
Location: `unified_memory_system.py:664-690`

```python
if len(results) < 3 and not_searching_all_tiers:
    # Re-search remaining tiers to ensure adequate results
    # Prevents over-aggressive routing from missing relevant content
```

### Outcome Learning
Location: `unified_memory_system.py:1430-1457`

- Tracks `successes`, `failures`, `total` per tier per concept
- Updates on every `record_outcome(worked/failed/partial)` call
- Partial outcomes increment `total` but don't affect success rate
- Builds concept relationships for knowledge graph visualization

## 🛠️ File Changes

### Backend Changes
- `backend/main.py` (lines 756-799): `record_response` tool definition - semantic learning with 2 parameters (key_takeaway + outcome)
- `backend/main.py` (lines 965-1050): Complete rewrite of `record_response` handler
  - Stores CURRENT learning: semantic summary from external LLM
  - Scores PREVIOUS learning using external LLM's outcome assessment
  - Leverages LLM's summarization strength instead of verbatim storage
- `backend/main.py` (lines 719-729): Removed `list_memory_bank` tool (redundant)
- `backend/main.py` (lines 891-902): Removed `list_memory_bank` handler
- `backend/main.py` (lines 831-851): Fixed MCP `["all"]` bug + routing logs
- `backend/modules/memory/unified_memory_system.py` (lines 870-928): Learning-based `_route_query()`
- `backend/modules/memory/unified_memory_system.py` (lines 930-975): N-gram `_extract_concepts()`
- `backend/modules/memory/unified_memory_system.py` (lines 1033-1050): `_track_usage()` now saves KG
- `backend/modules/memory/unified_memory_system.py` (lines 664-690): Fallback safety net
- `backend/modules/memory/unified_memory_system.py` (lines 1430-1457): Enhanced outcome learning
- `backend/modules/memory/unified_memory_system.py` (lines 2342-2398): `store_memory_bank()` - accepts `importance` and `confidence`
- `backend/utils/tool_definitions.py` (lines 64-82): Changed `tag` to `tags` (array), standardized defaults
- `backend/app/routers/agent_chat.py` (lines 2114-2135): Handler updated for `tags` array
- `backend/main.py` (lines 726-728): MCP `add_to_memory_bank` added `confidence` parameter

### Documentation
- `docs/architecture.md` (lines 225-244): Outcome-based scoring documentation (working/history/patterns only)
- `docs/architecture.md` (lines 445-580): Comprehensive KG routing documentation
- `docs/architecture.md` (lines 1684-1813): Updated MCP section with automatic outcome detection
- `docs/architecture.md` (lines 2041-2053): MCP `add_to_memory_bank` tool with `confidence` parameter
- `docs/MCP_IMPLEMENTATION.md`: Complete guide for `record_response` tool
- `docs/KG_ROUTING_IMPLEMENTATION.md`: Complete implementation guide (reference)

## 📊 Performance & Learning Metrics

### Expected Learning Curve
- **0-3 queries**: Cold start, exploring all tiers
- **3-10 queries**: Building initial patterns, routing to 2-3 tiers
- **10-20 queries**: Medium confidence, consistent 2-tier routing
- **20+ queries**: High confidence, single-tier routing for common queries

### Example Learning Progression
```
Query 1: "show me books" → Routes to all 5 tiers (cold start)
Query 5: "show me books" → Routes to [books, working] (learning)
Query 15: "show me books" → Routes to [books] (confident)

Success rates improve over time:
- Cold start: 60% success (searching wrong tiers)
- After learning: 90% success (focused on right tier)
```

## 🔄 Upgrade Notes

### For Users Upgrading from v0.1.6 or Earlier
- **Manual KG Reset Recommended**: If you experience poor routing performance:
  - **Option 1 (Easier)**: Use Settings UI → Delete knowledge graph data
  - **Option 2**: Manually delete `knowledge_graph.json` from your data folder
- **Fresh Learning**: KG will rebuild from scratch with improved routing logic
- **Data Preserved**: All memories in ChromaDB (books, working, history, patterns, memory_bank) remain intact
- **Settings Preserved**: All user preferences and MCP session history preserved

### For New Users
- **Optimal Experience**: Starts with empty KG, no legacy data
- **Learning Begins**: Routing learns from first search

## 🎯 Use Cases

### Improved Search Performance
1. **Book Queries**: Learns to route "books about X" directly to books tier
2. **Recent Context**: Routes "what did I say about X" to working/history tiers
3. **Personal Info**: Routes "my name/preferences" to memory_bank tier
4. **Patterns**: Routes "how do I X" to patterns tier for proven solutions

### MCP Integration Benefits
- **Claude Desktop**: Automatic intelligent routing without manual tier specification
- **Cursor IDE**: Fast, focused searches in development context
- **External LLMs**: Benefit from Roampal's learned routing patterns

### Knowledge Graph Cleanup on Deletion (v0.2.0)
- **NEW**: Automatic KG cleanup prevents stale data from deleted memories
- **Content KG Cleanup**: Automatically called on ALL memory_bank deletions
  - `archive_memory_bank()` → `content_graph.remove_entity_mention(doc_id)`
  - `user_delete_memory()` → `content_graph.remove_entity_mention(doc_id)`
  - `delete_by_conversation()` → batch cleanup for all deleted memory_bank items
- **Routing KG Cleanup**: Called on bulk deletions
  - `delete_by_conversation()` → `_cleanup_kg_dead_references()`
- **Fallback Protection**: Cold-start has fallback to vector search when Content KG has stale data
  - Prevents crashes when Content KG references deleted memories
  - Automatically falls back to vector search if Content KG retrieval returns empty
- **Why This Matters**:
  - Prevents crashes from Content KG pointing to deleted memory_bank IDs
  - ChromaDB returns `None` for deleted docs → would crash without fallback
  - Keeps both Routing KG and Content KG clean and accurate
- **Implementation**:
  - `unified_memory_system.py:877-892` - Content KG cleanup in `delete_by_conversation()`
  - `unified_memory_system.py:2816-2842` - Content KG cleanup in `user_delete_memory()`
  - `unified_memory_system.py:2703` - Content KG cleanup in `archive_memory_bank()`
  - `unified_memory_system.py:964-976` - Fallback to vector search when Content KG stale

### Enhanced MCP Outcome Scoring Prompt (v0.2.0)
- **IMPROVED**: Comprehensive outcome detection prompt for external LLMs
- **Signal Detection Checklist**: Explicit patterns to match user feedback
  - ✅ Positive signals: "worked", "perfect", "thanks", "ok cool" → worked
  - ❌ Negative signals: "failed", "error", "wrong" → failed
  - ⚠️ Clarification signals: "what?", "confused", "hmm" → partial
  - ❓ No signal: First message or zero feedback → unknown
- **Edge Case Handling**: Specific guidance for tricky scenarios
  - "ok" → worked (neutral acceptance)
  - "hmm" → partial (thinking/unsure)
  - User repeats question → failed (you didn't answer properly)
  - Sarcasm detection ("yeah, that totally worked...") → failed
- **Multi-Turn Examples**: 5-turn conversation showing signal analysis at each step
- **Common Mistakes Section**: Clear "don't do this" / "do this" guidance
  - ❌ Don't default to "unknown" when lazy
  - ❌ Don't score based on YOUR judgment
  - ✅ Do check EVERY message for signals
  - ✅ Do score even subtle signals
- **Confidence Check**: Self-validation mechanism (80%+ confidence threshold)
  - If <80% confident → default to "unknown" and explain why
- **Recovery Path**: Instructions for when LLM realizes it scored wrong
- **Meta-Learning Motivation**: Explains why accurate scoring matters (system learns from outcomes)
- **Quick Reference TL;DR**: Summary at bottom for rapid lookup after first read
- **Expected Impact**: Significantly improved outcome tracking accuracy (target: 95%+)
- **Implementation**: `main.py:818-993` - Complete rewrite of `record_response` tool description


## 🐛 Bug Fixes

- **FIXED**: MCP `collections=["all"]` now properly triggers KG routing (was searching no tiers)
- **FIXED**: Success rate calculation excludes partials (was diluting rates)
- **FIXED**: Old KG data auto-migrated (was biasing routing toward "working" tier)
- **FIXED**: Content KG stale data crashes (fallback to vector search when entities point to deleted docs)

## 📦 Embedding Model Simplification

### Bundled Embedding Model (v0.2.0)
- **NEW**: All users now use bundled `paraphrase-multilingual-mpnet-base-v2`
- **REMOVED**: Ollama embedding dependency - no longer uses `nomic-embed-text`
- **Benefit**: Consistent embeddings for all users (chat + MCP server)
- **Benefit**: Works offline out-of-the-box, no Ollama required for embeddings
- **Benefit**: Simpler codebase (~100 lines removed from embedding_service.py)
- **Compatible**: Same 768 dimensions - existing ChromaDB collections still work
- **License**: Apache 2.0 - commercially safe to bundle and redistribute

### Technical Details
- Model location: `binaries/models/paraphrase-multilingual-mpnet-base-v2/`
- Loads on startup instead of lazy loading
- Single code path (no Ollama fallback complexity)
- 50+ languages supported
- Native 768-dimensional embeddings (no padding needed)

## 🎨 UI Enhancements

### Knowledge Graph Visualization
- **NEW**: Dynamic header text reflects active filters ("Top 20 most recent concepts from today")
- **NEW**: Time-based filtering (All Time, Today, This Week, This Session)
- **NEW**: Sort options (Importance, Recent, Oldest)
- **NEW**: Timestamp tracking (created_at, last_used) for all concepts
- **IMPROVED**: Smooth panel resizing with debounced layout updates
- **FIXED**: Success rate bug - now correctly shows 100% for concepts with all "worked" outcomes
- **FIXED**: Empty state UX - filter controls always visible even when no data
- **ENHANCED**: Concept modal shows routing breakdown (collections searched, success rates per tier)

### Concept Success Rates
- **FIXED**: Success rate calculation: `successes / (successes + failures)` instead of `successes / total`
- **NEW**: 50% neutral baseline for concepts without feedback (was showing 0%)
- **Why**: Only counts explicit outcomes (worked/failed), not total searches without feedback
- **Example**: 1 worked + 0 failed = 100% (not 6.7% from 1/15 total searches)

## 🔮 Next Steps (v0.2.0 Completion)

### Content Knowledge Graph (Major Feature - Implemented)
- **NEW**: Dual KG system - routing KG (✅ implemented) + content KG (✅ implemented)
- **Routing KG**: Used for ALL collections (books, working, history, patterns, memory_bank) - learns which collection answers which query
- **Content KG**: Used for memory_bank only - indexes entity relationships from text content, not query patterns
- **Benefits**:
  - Fixes MCP integration issues (entities will have real connections, not 0)
  - Enables `get_kg_path("logan", "everbright")` to find relationships
  - Creates "turbo user profile graph" - visual representation of user's knowledge
  - Foundation for future NER, fact extraction, semantic network features
- **UI Enhancement**:
  - 🔵 Blue nodes = routing patterns (query-based)
  - 🟢 Green nodes = memory entities (content-based)
  - 🟣 Purple nodes = both
- **Implementation**: [modules/memory/content_graph.py](../ui-implementation/src-tauri/backend/modules/memory/content_graph.py)
- **Details**: See [docs/KG_UPGRADE.md](KG_UPGRADE.md) Appendix C

### v0.2.0 Polish
- Export/import learned routing patterns
- Multi-session KG learning aggregation
- Advanced graph layout algorithms (force-directed)
- Concept relationship strength indicators

---

**Full Changelog**: https://github.com/[your-repo]/roampal/compare/v0.1.7...v0.2.0
**Download**: [Release Assets](https://github.com/[your-repo]/roampal/releases/tag/v0.2.0)
