# Release Notes - v0.2.6: Unified Learning + Directive Insights

**Release Date:** December 2025
**Type:** Feature Release
**Focus:** Complete the learning loop across all interfaces + make insights actionable

---

## Headlines

> **Internal LLM now contributes to Action KG** - Tool effectiveness tracking unified across all interfaces
> **Document-Level Insights** - Action KG examples surface which specific docs lead to success
> **Directive Insights** - get_context_insights tells you WHAT TO DO, not just what happened before
> **Model-Agnostic Prompts** - Workflow-based tool descriptions that work across all LLMs

---

## Part 1: Internal Action KG Contribution

### The Problem

Prior to v0.2.6, Action-Effectiveness KG tracking had a gap:

| Interface | Reads Action KG? | Writes to Action KG? |
|-----------|------------------|---------------------|
| MCP Server (Claude Code, Cursor, etc.) | ✅ Yes | ✅ Yes |
| Internal Agent Chat (Roampal UI) | ✅ Yes | ❌ **No** |

The internal LLM:
- **Saw** action stats: "search_memory on patterns has 88% success rate"
- **Didn't contribute** to those stats when it used tools

This meant Action KG metrics were **only built from external MCP clients**, but the internal agent benefited from reading those stats without giving back. A one-way relationship.

**Note:** architecture.md line 930 incorrectly stated internal system tracked actions. This was aspirational, not implemented.

### Current Architecture (v0.2.5)

**MCP Server** ([main.py:1145](../ui-implementation/src-tauri/backend/main.py#L1145)):
```python
# Creates ActionOutcome and caches for later scoring
action = ActionOutcome(
    action_type="search_memory",
    context_type=context_type,
    outcome="unknown",
    ...
)
_cache_action_with_boundary_check(session_id, action, context_type)
```

**Agent Chat** ([agent_chat.py:2219](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py#L2219)):
```python
# Calls memory system directly - NO ActionOutcome created
if tool_name == "search_memory":
    tool_results = await self._search_memory_with_collections(...)
```

### The Fix

#### 1. Import ActionOutcome

**File:** [agent_chat.py:49](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py#L49)

```python
from modules.memory.unified_memory_system import UnifiedMemorySystem, ActionOutcome
```

#### 2. Add Action Cache

**File:** agent_chat.py (module level, ~line 80)

```python
_agent_action_cache: Dict[str, List[ActionOutcome]] = {}
```

#### 3. Track ALL Tools at Common Exit Point

**File:** agent_chat.py (~line 2504, after tool executes, before yielding result)

```python
# Track ANY tool for Action KG (built-in or external MCP)
action = ActionOutcome(
    action_type=tool_name,  # "search_memory", "mcp__github__list_issues", etc.
    context_type=context_type,  # already detected at line 589
    outcome="unknown",
    action_params=tool_args,
    collection=tool_args.get("collections", [None])[0] if tool_name == "search_memory" else None
)
_agent_action_cache.setdefault(conversation_id, []).append(action)
```

Key insight: By placing tracking at the **common exit point** instead of inside each `if tool_name == "X"` block, we capture ALL tools automatically:
- Built-in: `search_memory`, `create_memory`, `update_memory`, `archive_memory`
- External MCP: `mcp__github__create_issue`, `mcp__filesystem__read_file`, etc.
- Any future tools added

#### 4. Score Actions When Outcome Detected

**File:** agent_chat.py (~line 1022, where outcome is determined)

```python
# After: if outcome_result.get("outcome") in ["worked", "failed", "partial"]:
if conversation_id in _agent_action_cache:
    for action in _agent_action_cache[conversation_id]:
        action.outcome = outcome
        await self.memory.record_action_outcome(action)
    del _agent_action_cache[conversation_id]
```

This hooks into the existing outcome detection system ([agent_chat.py:986-1051](../ui-implementation/src-tauri/backend/app/routers/agent_chat.py#L986-L1051)).

### Result: Unified Tracking

| Interface | Reads Action KG? | Writes to Action KG? |
|-----------|------------------|---------------------|
| MCP Server (Claude Code, Cursor, etc.) | ✅ Yes | ✅ Yes |
| Internal Agent Chat (Roampal UI) | ✅ Yes | ✅ **Yes** |

---

## Part 2: Document-Level Insights (Extending Action KG)

### The Problem

memory_bank and books are **static collections** - they don't have outcome-based scores like working/history/patterns. How do we learn which specific docs are useful?

| Collection | Has outcome scores? | Before v0.2.6 |
|------------|---------------------|---------------|
| working/history/patterns | ✅ Per-document scores | Learns from outcomes |
| memory_bank | ❌ Has importance/confidence only | No learning |
| books | ❌ Static chunks | No learning |

### The Insight: Action KG Already Tracks Doc IDs

Looking at [unified_memory_system.py:2650-2658](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L2650-L2658):

```python
# Action KG already stores doc_ids in examples!
example = {
    "timestamp": action.timestamp.isoformat(),
    "outcome": action.outcome,
    "doc_id": action.doc_id,  # ← Already tracked!
    "params": action.action_params,
    ...
}
stats["examples"] = (stats.get("examples", []) + [example])[-5:]
```

**We don't need a 4th KG.** The data is already being captured - we just need to use it.

### The Solution: Aggregate Doc Success from Action KG

#### New Helper Method

**File:** unified_memory_system.py

```python
def get_doc_effectiveness(self, doc_id: str) -> Optional[Dict]:
    """
    Aggregate success rate for a specific doc from Action KG examples.
    Scans all context_action_effectiveness entries for this doc_id.
    """
    successes = 0
    failures = 0
    partials = 0

    for key, stats in self.knowledge_graph["context_action_effectiveness"].items():
        for example in stats.get("examples", []):
            if example.get("doc_id") == doc_id:
                if example["outcome"] == "worked":
                    successes += 1
                elif example["outcome"] == "failed":
                    failures += 1
                else:
                    partials += 1

    total = successes + failures + partials
    if total == 0:
        return None

    return {
        "successes": successes,
        "failures": failures,
        "partials": partials,
        "total_uses": total,
        "success_rate": (successes + partials * 0.5) / total
    }
```

#### Use During Search (Optional Boost)

```python
# In unified_memory_system.py search()
if collection in ["memory_bank", "books"]:
    doc_stats = self.get_doc_effectiveness(doc_id)
    if doc_stats and doc_stats["total_uses"] >= 3:
        # Boost based on success rate (max 15% boost)
        effectiveness_boost = doc_stats["success_rate"] * 0.15
        adjusted_score = base_score * (1 + effectiveness_boost)
```

### MCP Fixes (Unified with Internal System)

Two fixes to align MCP with internal agent_chat.py behavior:

**Fix 1: Cache ALL doc_ids** ([main.py:1091-1094](../ui-implementation/src-tauri/backend/main.py#L1091))

```python
# Before: Only cached working/history/patterns
# if collection in ['working', 'history', 'patterns'] and doc_id:

# After: Cache ALL doc_ids for Action KG tracking
if doc_id:
    cached_doc_ids.append(doc_id)
```

**Fix 2: Set doc_id in ActionOutcome** ([main.py:1170](../ui-implementation/src-tauri/backend/main.py#L1170))

```python
# Before: search_memory ActionOutcome had no doc_id
action = ActionOutcome(
    action_type="search_memory",
    ...
    collection=coll if coll != "all" else None
    # NO doc_id!
)

# After: Include first result's doc_id for Action KG examples
action = ActionOutcome(
    ...
    collection=coll if coll != "all" else None,
    doc_id=cached_doc_ids[0] if cached_doc_ids else None
)
```

### Benefits

- **No new KG** - uses existing Action KG data
- memory_bank facts gain learned effectiveness
- books chunks that help get prioritized
- **Unified tracking** - both internal and MCP now cache all doc_ids AND set doc_id in ActionOutcome
- `get_doc_effectiveness()` now works for books/memory_bank searches via MCP
- Simpler implementation, less maintenance

---

## Part 3: Directive Insights

### The Problem

Current `get_context_insights` output is **retrospective**, not **directive**:

```
# Current (v0.2.5)
📊 Action Outcome Stats:
  • search_memory() on books: 90% success (79 uses)
  • create_memory() on memory_bank: 92% success (70 uses)
```

This tells you what worked before, but doesn't prompt you to **take action now**.

### The Solution: Actionable Output

```
# New (v0.2.6)
📌 RECOMMENDED ACTIONS:
  • search_memory(collections=["patterns"]) - 3 matches for "docker"
  • record_response() after reply - required for learning

💡 YOU ALREADY KNOW THIS (from memory_bank):
  • "User prefers Docker Compose over raw Docker" (83% helpful)

📋 PAST EXPERIENCE:
  • "Adding user to docker group" worked 3 times (score: 0.95)

📊 TOOL STATS:
  • search_memory() on patterns: 92% success (45 uses)

⚠️ REMINDER: Call record_response() to complete the learning loop
```

### Changes to get_context_insights

**File:** [main.py:1319-1375](../ui-implementation/src-tauri/backend/main.py#L1319-L1375)

#### 1. Add Directive Prompts

```python
# After entity extraction from query
matched_concepts = org_context.get('matched_concepts', [])
if matched_concepts:
    response += "📌 RECOMMENDED ACTIONS:\n"

    # Get routing recommendations from Routing KG
    routing = memory.get_tier_recommendations(matched_concepts)
    if routing and routing.get('top_collections'):
        collections = routing['top_collections'][:2]
        match_count = routing.get('match_count', 0)
        response += f"  • search_memory(collections={collections}) - {match_count} patterns found\n"

    response += "  • record_response(outcome=...) after reply - required for learning\n\n"
```

#### 2. Surface Relevant memory_bank Facts

```python
# Query Content KG for matching entities → pull memory_bank facts
relevant_facts = await memory.get_facts_for_entities(matched_concepts, limit=2)
if relevant_facts:
    response += "💡 YOU ALREADY KNOW THIS (from memory_bank):\n"
    for fact in relevant_facts:
        response += f"  • \"{fact['content'][:80]}...\"\n"
    response += "\n"
```

#### 3. Add Explicit record_response Reminder

```python
# At end of insights output
response += "\n═══ TO COMPLETE THIS INTERACTION ═══\n"
response += "After responding → record_response(key_takeaway=\"...\", outcome=\"worked|failed|partial\")"
```

### New Helper Methods

**File:** unified_memory_system.py

```python
def get_tier_recommendations(self, concepts: List[str]) -> Dict:
    """Query Routing KG for best collections given concepts"""
    # Returns: {"top_collections": ["patterns", "history"], "match_count": 3}

async def get_facts_for_entities(self, entities: List[str], limit: int = 2) -> List[Dict]:
    """Query Content KG → retrieve matching memory_bank facts"""
    # Returns: [{"doc_id": "...", "content": "...", "entity": "docker"}]

def get_doc_effectiveness(self, doc_id: str) -> Optional[Dict]:
    """Aggregate success rate for a doc from Action KG examples"""
    # Returns: {"success_rate": 0.83, "total_uses": 5, ...}
```

---

## The Complete Learning Loop (v0.2.6)

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
▼                                                                  │
QUERY: "How do I fix Docker permissions?"                          │
    │                                                              │
    ▼                                                              │
get_context_insights() ← queries 3 Knowledge Graphs                │
    │                                                              │
    ├── Routing KG → "search patterns for docker"                  │
    ├── Content KG → "docker" entity → pull memory_bank facts      │
    └── Action KG → stats + doc-level insights from examples       │
    │                                                              │
    ▼                                                              │
OUTPUT: "📌 SEARCH patterns | 💡 You know: prefers Compose"        │
    │                                                              │
    ▼                                                              │
search_memory(collections=["patterns"]) → retrieves docs           │
    │                                                              │
    ├── Cache: [patterns_abc, memory_bank_789, book_chunk_234]     │
    │                                                              │
    ▼                                                              │
LLM RESPONDS using retrieved context                               │
    │                                                              │
    ▼                                                              │
record_response(outcome="worked")                                  │
    │                                                              │
    ├── Routing KG: docker + patterns → success                    │
    ├── Action KG: coding|search_memory|patterns → success         │
    ├── Action KG examples: doc_ids stored with outcome            │
    └── Memory scores: patterns_abc score increased                │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
                    SYSTEM GETS SMARTER EVERY LOOP
```

---

## Part 4: Model-Agnostic Prompt Design

### The Problem

Previous MCP tool descriptions used "motivation" framing:
- "You are stateless. This makes you stateful."
- "This is how you become a better assistant."

This doesn't work reliably because:
- LLMs don't have goals or self-improvement desires
- "Motivation" framing is model-specific (works for Claude, maybe not Llama)
- Description is read once; LLMs forget to follow through

### The Solution: Workflow-Based Prompts

Focus on **instruction-following mechanics** that work across all LLMs:

| Technique | Why It's Universal |
|-----------|-------------------|
| **Clear triggers** ("when X → do Y") | All instruction-tuned models pattern match |
| **Explicit workflow steps** | Models follow numbered sequences reliably |
| **In-output reminders** | Fresh context > stale description (recency bias) |
| **Task completion framing** | Models are trained to complete tasks, not leave them hanging |

### Updated Tool Descriptions

#### get_context_insights (main.py)

**Before:**
```
⚡ USE BEFORE searching - Get organic insights from Content KG + Action-Effectiveness KG.

Returns: Past solutions, failure warnings, tool effectiveness stats, recommended collections

This is your "intuition" - pattern matching in knowledge graphs (5-10ms, no embeddings).

Workflow: get_context_insights(q) → read insights → search_memory(recommended_collections) → respond
```

**After:**
```
Search your memory before responding. Returns what you know about this user/topic.

WORKFLOW (follow these steps):
1. get_context_insights(query) ← YOU ARE HERE
2. Read the context returned
3. search_memory() if you need more details
4. Respond to user
5. record_response() to complete

Returns: Known facts, past solutions, recommended collections, tool stats.
Fast lookup (5-10ms) - no embedding search, just pattern matching.
```

#### record_response (main.py)

**Before:**
```
🔴 REQUIRED: Call after EVERY response - Stores semantic learning + scores based on user satisfaction.
```

**After:**
```
Complete the interaction. Call this after responding to the user.

WORKFLOW:
1. get_context_insights() ✓
2. search_memory() if needed ✓
3. Respond to user ✓
4. record_response() ← YOU ARE HERE (completes the interaction)

Parameters:
• key_takeaway: 1-2 sentence summary of what happened
• outcome: "worked" | "failed" | "partial" | "unknown"

OUTCOME DETECTION (read user's reaction):
✓ worked = user satisfied, says thanks, moves on
✗ failed = user corrects you, says "no", "that's wrong", provides the right answer
~ partial = user says "kind of" or takes some but not all of your answer
? unknown = no clear signal from user

⚠️ CRITICAL - "failed" OUTCOMES ARE ESSENTIAL:
• If user says you were wrong → outcome="failed"
• If memory you retrieved was outdated → outcome="failed"
• If user had to correct you → outcome="failed"
• If you gave advice that didn't help → outcome="failed"

Failed outcomes are how bad memories get deleted. Without them, wrong info persists forever.
Don't default to "worked" just to be optimistic. Wrong memories MUST be demoted.

This closes the loop. Without it, the system can't learn what worked OR what didn't.
```

#### search_memory (main.py)

**Before:**
```
Search your 5-tier persistent memory (books, working, history, patterns, memory_bank).
```

**After:**
```
Search your persistent memory. Use when you need details beyond what get_context_insights returned.

WHEN TO SEARCH:
• User says "remember", "I told you", "we discussed" → search immediately
• get_context_insights recommended a collection → search that collection
• You need more detail than the context provided

WHEN NOT TO SEARCH:
• General knowledge questions (use your training)
• get_context_insights already gave you the answer

Collections: memory_bank (user facts), books (docs), patterns (proven solutions), history (past), working (recent)
Omit 'collections' parameter for auto-routing (recommended).
```

### Updated get_context_insights Output

The output now ends with an explicit next-step reminder:

```
═══ KNOWN CONTEXT ═══
[User Profile]
- Logan, crypto researcher, prefers Docker Compose

[Relevant Patterns]
- "docker permissions" → add user to docker group (worked 3x)

[Recommended]
- search_memory(collections=["patterns"]) for more solutions

═══ TO COMPLETE THIS INTERACTION ═══
After responding → record_response(key_takeaway="...", outcome="worked|failed|partial")
```

This creates an **open loop** that prompts the LLM to close it.

### Why This Works Across Models

| Model Size | Behavior |
|------------|----------|
| 7B models | Follow simple numbered workflows |
| 13-30B models | Understand task completion framing |
| 70B+ models | Won't be annoyed by explicit instructions |

The key insight: **Don't try to motivate LLMs. Give them clear instructions and open loops to close.**

### Files Modified

| File | Changes |
|------|---------|
| `main.py` | Updated tool descriptions with workflow framing |
| `main.py` | get_context_insights output includes "TO COMPLETE" section |

---

## Part 5: Contextual Book Embeddings

### The Problem

Book chunks are embedded as raw text without context:
```
"Every piece of knowledge must have a single..."
```

When searching for "DRY principle", this chunk doesn't match well because the acronym isn't in the text. Testing showed searches for generic terms like "DRY principle" returned irrelevant results (Marcus Aurelius, Cialdini) instead of Pragmatic Programmer.

### The Fix

Prepend contextual prefix before embedding:
```
"Book: Pragmatic Programmer, Section: DRY Principle. Every piece of knowledge must have a single..."
```

**File:** [smart_book_processor.py:388-391](../ui-implementation/src-tauri/backend/modules/memory/smart_book_processor.py#L388)

```python
batch_texts = [
    f"Book: {title}, Section: {chunk_data.get('source_context', title)}. {chunk_data['text']}"
    for chunk_data in batch_chunk_data
]
```

### Impact

- ~49% retrieval improvement (per Anthropic's Contextual Retrieval research, Sep 2024)
- Ambiguous queries now match correct book sections
- No schema change - only affects embedding generation
- Zero cost at query time (prefix added at ingestion)

### Limitation

**Existing books need re-upload to benefit.** Only new uploads get contextual embeddings.

---

## Part 6: Action KG Cleanup on Book Deletion

### The Problem

When books are deleted, Action KG examples that reference deleted book chunk `doc_id`s become stale:
- `get_doc_effectiveness(doc_id)` returns stats for non-existent documents
- Examples in `context_action_effectiveness` retain dead `doc_id` references
- Content KG doesn't need cleanup (books aren't indexed there - only `memory_bank`)

### The Fix

Added `cleanup_action_kg_for_doc_ids()` method and integrated it into book deletion flow.

**Files:**
- [unified_memory_system.py:4114-4153](../ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py#L4114) - New cleanup method
- [book_upload_api.py:705-714](../ui-implementation/src-tauri/backend/backend/api/book_upload_api.py#L705) - Call cleanup on delete

```python
async def cleanup_action_kg_for_doc_ids(self, doc_ids: List[str]) -> int:
    """Remove Action KG examples referencing specific doc_ids."""
    doc_id_set = set(doc_ids)
    cleaned = 0
    for key, stats in self.knowledge_graph.get("context_action_effectiveness", {}).items():
        examples = stats.get("examples", [])
        stats["examples"] = [ex for ex in examples if ex.get("doc_id") not in doc_id_set]
        cleaned += len(examples) - len(stats["examples"])
    return cleaned
```

### Impact

- Prevents stale doc_id references in Action KG
- `get_doc_effectiveness()` no longer returns data for deleted books
- Consistent with `memory_bank` deletion which already cleans Content KG

---

## Summary

### Three Knowledge Graphs (Extended)

| KG | Question It Answers | Updated in v0.2.6? |
|----|---------------------|-------------------|
| **Routing KG** | "Search patterns or history?" | ✅ Exposed in get_context_insights recommendations |
| **Content KG** | "What entities relate to query?" | ✅ Facts surfaced in get_context_insights output |
| **Action KG** | "Which tool works here?" | ✅ Internal LLM contributes + doc-level insights + tracks ALL tools |

**Action KG now tracks ANY tool** - built-in (`search_memory`, `create_memory`) and external MCP tools (`mcp__github__create_issue`, etc.).

### Files Modified

| File | Changes |
|------|---------|
| `app/routers/agent_chat.py` | Import ActionOutcome, add cache, track tools, score on outcome |
| `main.py` | Enhanced get_context_insights output, workflow-based tool descriptions |
| `unified_memory_system.py` | Add get_doc_effectiveness(), get_facts_for_entities(), cleanup_action_kg_for_doc_ids() |
| `smart_book_processor.py` | Add contextual prefix before embedding for improved retrieval |
| `book_upload_api.py` | Call cleanup_action_kg_for_doc_ids() on book deletion |
| `docs/architecture.md` | Fix line 930 (internal tracking was not implemented) |

### Six Parts Summary

| Part | Feature | Impact |
|------|---------|--------|
| **Part 1** | Internal Action KG | Internal LLM now contributes to Action KG |
| **Part 2** | Document-Level Insights | Aggregate doc success from existing Action KG examples |
| **Part 3** | Directive Insights | get_context_insights outputs actionable prompts + facts |
| **Part 4** | Model-Agnostic Prompts | Workflow-based descriptions that work across all LLMs |
| **Part 5** | Contextual Book Embeddings | ~49% improved retrieval for ambiguous book queries |
| **Part 6** | Action KG Cleanup | Prevent stale doc_ids when books deleted |

---

## Upgrade Notes

- No database migration needed
- No configuration changes required
- Existing KG data preserved
- New tracking begins immediately after upgrade
- Document effectiveness starts at 0 and builds over time
- **Books:** Existing books retain old embeddings; re-upload to get contextual prefix benefit

---

## Previous Release

See [v0.2.5 Release Notes](RELEASE_NOTES_0.2.5.md) for MCP Client Integration, Wilson Scoring, and Multilingual Reranking.