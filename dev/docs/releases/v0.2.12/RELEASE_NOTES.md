# Roampal Desktop v0.2.12 Release Notes

**Release Date:** 2026-01-10
**Type:** Critical Bug Fix
**Status:** ✅ COMPLETE

---

## Overview

v0.2.12 is a hotfix release that addresses a critical rendering bug introduced in v0.2.11's virtualization implementation, fixes update notification dismiss persistence, corrects misleading system prompts about outcome scoring, enables all memory sources (organic recall, cold start, search) to participate in selective outcome scoring with MCP parity, and adds main LLM memory attribution for accurate causal scoring.

---

## Critical Fixes

### 1. Message Virtualization Overlap Bug

**Problem:** Messages in the chat interface render with overlapping text, making the UI unreadable. Text from multiple messages appears stacked on top of each other.

**Root Cause:** Three issues in the `TerminalMessageThread.tsx` virtualization implementation:

1. **Height measurement timing:** react-window positions items using the default 60px height estimate BEFORE actual heights are measured, causing items to stack at incorrect positions.

2. **Memo comparison uses reference equality:** The `MessageRow` memo compares `prev.message === next.message` by reference, not content. When streaming updates mutate the message object without changing its reference, the component doesn't re-render or re-measure.

3. **useLayoutEffect dependencies incomplete:** The effect watches `message` reference instead of `message.content`, so height changes during streaming aren't detected.

**Solution:**

1. Update `useLayoutEffect` to watch actual content changes:
```tsx
// Before (line 230):
}, [setSize, index, message]);

// After:
}, [setSize, index, message.content, message.streaming, message.events?.length]);
```

2. Fix memo comparison to detect content changes:
```tsx
// Before (lines 395-399):
}, (prev, next) => {
  return prev.message === next.message && prev.index === next.index && prev.style === next.style && prev.isProcessing === next.isProcessing;
});

// After:
}, (prev, next) => {
  if (prev.message === 'loading-indicator' || next.message === 'loading-indicator') {
    return prev.message === next.message && prev.isProcessing === next.isProcessing;
  }
  return (
    prev.message.content === next.message.content &&
    prev.message.streaming === next.message.streaming &&
    prev.message.events?.length === next.message.events?.length &&
    prev.index === next.index &&
    prev.isProcessing === next.isProcessing
  );
});
```

3. Improve initial height estimation based on content length:
```tsx
// Before (line 449):
const getSize = (index: number) => sizeMap.current[index] || 60;

// After:
const getSize = (index: number) => {
  if (sizeMap.current[index]) return sizeMap.current[index];
  // Estimate based on content length for better initial positioning
  const msg = messages[index];
  if (!msg) return 60;
  const contentLength = msg.content?.length || 0;
  const hasTools = msg.toolExecutions?.length || msg.events?.some(e => e.type === 'tool_execution');
  const baseHeight = 40;
  const textHeight = Math.ceil(contentLength / 80) * 20; // ~80 chars per line, 20px per line
  const toolHeight = hasTools ? 30 : 0;
  return Math.max(60, baseHeight + textHeight + toolHeight);
};
```

**Files Changed:**
-   `ui-implementation/src/components/TerminalMessageThread.tsx:230,395-399,449`

**Status:** ✅ COMPLETE

---

### 2. Update Notification Dismiss Not Persisted

**Problem:** When users dismiss the update notification, it reappears every time they restart the app. The dismiss state is only stored in React memory, not persisted.

**Root Cause:** The `dismissed` state in `useUpdateChecker.ts` is a simple `useState(false)` with no persistence to localStorage.

**Solution:**

Persist dismiss state to localStorage, keyed by version so it only stays dismissed for the specific version they dismissed:

```tsx
// In useUpdateChecker.ts

const DISMISS_KEY = 'roampal_update_dismissed_version';

export function useUpdateChecker() {
  const [updateInfo, setUpdateInfo] = useState<UpdateInfo | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const [checking, setChecking] = useState(false);

  // Check if this version was previously dismissed
  useEffect(() => {
    if (updateInfo?.version) {
      const dismissedVersion = localStorage.getItem(DISMISS_KEY);
      if (dismissedVersion === updateInfo.version) {
        setDismissed(true);
      }
    }
  }, [updateInfo?.version]);

  // ... existing check logic ...

  const dismiss = () => {
    if (updateInfo?.version) {
      localStorage.setItem(DISMISS_KEY, updateInfo.version);
    }
    setDismissed(true);
  };

  // ... rest unchanged ...
}
```

**Behavior after fix:**
- User dismisses v0.2.12 notification → stays dismissed even after restart
- When v0.2.13 releases → notification shows again (different version)
- Critical updates still cannot be dismissed (intentional)

**Files Changed:**
-   `ui-implementation/src/hooks/useUpdateChecker.ts:18-21,44`

**Status:** ✅ COMPLETE

---

### 3. System Prompt Misleads LLM About Outcome Scoring

**Problem:** When users ask LLM how outcome scoring works, it claims scoring is "automatic" with no LLM involvement. This is incorrect.

**Root Cause:** System prompt told LLM "This happens automatically - you don't need to do anything" when LLM IS doing the scoring via `OutcomeDetector`.

**How Scoring Actually Works:**
1. User sends message → Main LLM responds (with memory attribution marks, see Fix #7)
2. User sends follow-up → System intercepts
3. `OutcomeDetector` (separate LLM call) analyzes: "Did this help?"
4. Returns `worked`, `failed`, `partial`, or `unknown`
5. Only memories marked as used by the LLM get scored (selective scoring via Fix #5 and #7)

**Solution:** Updated system prompt in `agent_chat.py:1411-1432` to explain the actual mechanism.

**Files Changed:**
-   `ui-implementation/src-tauri/backend/app/routers/agent_chat.py:1411-1432`

**Status:** ✅ COMPLETE

---

### 4. Organic Recall Memories Not Scored

**Problem:** Memories surfaced via organic recall (auto-injected guidance before each LLM response) are never outcome-scored. Only memories from explicit `search_memory` calls get scored.

**Impact:** The system can't learn which organic recall guidance is actually helpful. Bad patterns/facts keep getting surfaced because they never receive negative scores.

**Root Cause:** Organic recall (lines 624-698 in `agent_chat.py`) injects formatted guidance but never adds doc_ids to `_search_cache`. The scoring flow only processes what's in that cache.

**Current flow:**
```
Organic Recall → Injects guidance text → NOT cached → NOT scored
search_memory  → Returns results      → Cached      → Scored
```

**Solution:**

Extract doc_ids from organic recall sources and add them to the scoring cache:

```python
# In agent_chat.py, after organic recall injection (around line 698)

# Cache organic recall doc_ids for outcome scoring
organic_doc_ids = []

# From memory_bank facts
for fact in relevant_facts:
    if doc_id := fact.get('id') or fact.get('doc_id'):
        organic_doc_ids.append(doc_id)

# From content KG patterns
if org_context:
    for pattern in org_context.get('relevant_patterns', []):
        if doc_id := pattern.get('doc_id'):
            organic_doc_ids.append(doc_id)
    for outcome in org_context.get('past_outcomes', []):
        if doc_id := outcome.get('doc_id'):
            organic_doc_ids.append(doc_id)

# Merge with existing cache (don't overwrite search_memory results)
if organic_doc_ids:
    existing = _search_cache.get(conversation_id, [])
    _search_cache[conversation_id] = list(set(existing + organic_doc_ids))
    logger.debug(f"[ORGANIC_CACHE] Added {len(organic_doc_ids)} organic recall doc_ids")
```

**Behavior after fix:**
- Organic recall guidance gets scored alongside explicit searches
- Bad patterns/facts get demoted over time
- System learns which auto-injected guidance actually helps

**Files Changed:**
-   `ui-implementation/src-tauri/backend/app/routers/agent_chat.py:698-715`

**Status:** ✅ COMPLETE

---

### 5. Internal LLM Lacks Selective Scoring (Parity with MCP)

**Problem:** roampal-core MCP has selective scoring via `score_response(related=["doc_id1", "doc_id2"])`. Desktop MCP has selective scoring via `record_response(related=[1, 3])`. However, the internal LLM (Roampal Desktop chat) was scoring ALL cached memories blindly, polluting learning when only 2 of 5 surfaced memories were actually used.

**Impact:**
- Irrelevant memories get scored with outcomes they didn't contribute to
- Learning signal is diluted across unused memories
- Bad memories that weren't even referenced still get upvoted if the response "worked"

**Current flow (scores everything):**
```
1. 5 memories surfaced (organic + search)
2. LLM uses memories #1 and #3 in response
3. User says "thanks!" → outcome = worked
4. ALL 5 memories scored as "worked" ← Wrong!
```

**Solution:**

Enhance OutcomeDetector to identify which memories were actually used, using positional indexing (small-LLM friendly):

**1. Change cache structure to include position map:**
```python
# Before: just doc_ids
_search_cache[conv_id] = ["doc_1", "doc_2", "doc_3"]

# After: position → doc_id + content
_search_cache[conv_id] = {
    "position_map": {1: "history_a1b2c3d4", 2: "working_x1y2z3", 3: "patterns_m1n2o3"},
    "content_map": {1: "User prefers dark mode", 2: "Project uses React 18", ...}
}
```

**2. Enhance OutcomeDetector prompt:**
```python
# In outcome_detector.py - _llm_analyze()

prompt = f"""
...existing outcome detection...

SURFACED MEMORIES (assistant had access to these):
1. {content_map[1]}
2. {content_map[2]}
3. {content_map[3]}

Which memory NUMBERS were actually USED in the response?
Only include memories that directly influenced the answer.

Return JSON:
{{
    "outcome": "worked|failed|partial|unknown",
    "used_positions": [1, 3],  // positions that were used
    "confidence": 0.0-1.0,
    ...
}}
"""
```

**3. Selective scoring in agent_chat.py:**
```python
# Around line 1058
used_positions = outcome_result.get("used_positions", [])
position_map = cached.get("position_map", {})

if used_positions:
    # Score only used memories
    for pos in used_positions:
        if doc_id := position_map.get(pos):
            await self.memory.record_outcome(doc_id=doc_id, outcome=outcome)
    logger.info(f"[OUTCOME] Scored {len(used_positions)} used memories, skipped {len(position_map) - len(used_positions)}")
else:
    # Fallback: score all (backwards compat)
    for doc_id in position_map.values():
        await self.memory.record_outcome(doc_id=doc_id, outcome=outcome)
```

**Behavior after fix:**
- OutcomeDetector sees what memories were surfaced
- Identifies which ones were actually referenced in the response
- Only those get scored with the outcome
- Unused memories remain neutral (not polluted)

**Files Changed:**
-   `ui-implementation/src-tauri/backend/app/routers/agent_chat.py:698-715,1058-1070`
-   `ui-implementation/src-tauri/backend/modules/advanced/outcome_detector.py:72-104`

**Status:** ✅ COMPLETE

---

### 6. Cold Start Memories Not Scored

**Problem:** Cold start injects user profile on message 1, but those memories are never cached for outcome scoring. Same issue as organic recall (fix #4).

**Root Cause:** `get_cold_start_context()` returns a formatted string, losing the doc_ids:

```python
# unified_memory_system.py - get_cold_start_context()
all_context.append({
    "id": r.get("id", ""),      # ← doc_id collected
    "content": r.get("content"),
    "source": "memory_bank"
})
# But _format_cold_start_results() returns just a string

# agent_chat.py - line 555
context_summary = await self.memory.get_cold_start_context(limit=5)
# Only the string is used, doc_ids are lost
```

**Solution:**

**1. Return doc_ids from `get_cold_start_context()`:**
```python
# unified_memory_system.py
async def get_cold_start_context(self, limit: int = 5) -> Tuple[Optional[str], List[str]]:
    """Returns (formatted_context, doc_ids)"""
    all_context = []
    # ... existing collection logic ...

    doc_ids = [r.get("id") for r in all_context if r.get("id")]
    formatted = self._format_cold_start_results(all_context)
    return formatted, doc_ids
```

**2. Cache cold start doc_ids in agent_chat.py:**
```python
# Around line 555
context_summary, cold_start_doc_ids = await self.memory.get_cold_start_context(limit=5)

if context_summary:
    self.conversation_histories[conversation_id].append({
        "role": "system",
        "content": context_summary
    })

    # Cache for outcome scoring (same as organic recall)
    if cold_start_doc_ids:
        existing = _search_cache.get(conversation_id, {})
        position_map = existing.get("position_map", {})
        content_map = existing.get("content_map", {})

        # Add cold start memories with positions
        next_pos = max(position_map.keys(), default=0) + 1
        for doc_id, content in zip(cold_start_doc_ids, [c["content"] for c in all_context]):
            position_map[next_pos] = doc_id
            content_map[next_pos] = content[:200]
            next_pos += 1

        _search_cache[conversation_id] = {"position_map": position_map, "content_map": content_map}
        logger.debug(f"[COLD_START] Cached {len(cold_start_doc_ids)} doc_ids for scoring")
```

**Behavior after fix:**
- Cold start memories cached alongside organic recall and search_memory
- All three sources participate in selective scoring (fix #5)
- User profile facts can be demoted if they lead to bad responses

**Files Changed:**
-   `ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py:864-920`
-   `ui-implementation/src-tauri/backend/app/routers/agent_chat.py:550-575`

**Status:** ✅ COMPLETE

---

### 7. Main LLM Memory Attribution for Causal Scoring

**Problem:** Fix #5 has OutcomeDetector inferring which memories were used by analyzing the response. But the main LLM *knows* which memories helped vs hurt - it shouldn't have to guess. Also, the current system applies the same outcome (worked/failed) to all used memories, even if some were helpful and others were misleading.

**Example of the problem:**
```
Memories surfaced: [1: good advice, 2: bad advice, 3: irrelevant]
Main LLM uses #1 and #2, ignores #3
LLM overcomes bad advice from #2, gives good response
User: "thanks that worked!"

Current behavior (Fix #5):
  OutcomeDetector infers: used_positions = [1, 2]
  Outcome = worked
  Result: BOTH #1 and #2 upvoted ← Wrong! #2 was bad advice!

Desired behavior:
  Memory #1 (helpful) → upvote
  Memory #2 (misleading) → downvote (even though exchange succeeded!)
  Memory #3 (unused) → neutral
```

**Solution: Two-stage attribution**

Main LLM marks memories as helpful/unhelpful/no_impact. OutcomeDetector combines this with outcome to score correctly.

**1. Add to Main LLM System Prompt (agent_chat.py):**
```
MEMORY ATTRIBUTION (include at end of response if memories were surfaced):

After your response, add this hidden annotation:
<!-- MEM: 1👍 2👎 3➖ -->

Markers:
👍 = helped me answer well
👎 = was wrong/misleading
➖ = didn't use

This is invisible to the user but helps the system learn.
```

**2. Parse annotations in agent_chat.py:**
```python
# After getting main LLM response, before showing to user
import re

def parse_memory_marks(response: str) -> Tuple[str, Dict[int, str]]:
    """Extract and strip memory attribution from response."""
    match = re.search(r'<!-- MEM: (.*?) -->', response)
    if not match:
        return response, {}

    marks_str = match.group(1)
    marks = {}
    for item in marks_str.split():
        # Parse "1👍" → {1: "👍"}
        pos = int(''.join(c for c in item if c.isdigit()))
        emoji = ''.join(c for c in item if not c.isdigit())
        marks[pos] = emoji

    clean_response = re.sub(r'<!-- MEM:.*?-->', '', response).strip()
    return clean_response, marks

# Usage:
clean_response, memory_marks = parse_memory_marks(llm_response)
# Show clean_response to user
# Pass memory_marks to OutcomeDetector
```

**3. Update OutcomeDetector to use marks (outcome_detector.py):**
```python
async def analyze(
    self,
    conversation: List[Dict[str, Any]],
    surfaced_memories: Optional[Dict[int, str]] = None,
    llm_marks: Optional[Dict[int, str]] = None  # NEW: {1: "👍", 2: "👎", 3: "➖"}
) -> Dict[str, Any]:
```

**4. Simplified OutcomeDetector Prompt:**
```python
prompt = f"""Did the assistant's response help the user?

User reaction: {user_message}

Answer:
- YES = "thanks!", "perfect!", user moved on
- NO = "wrong", "didn't work", user frustrated
- KINDA = "okay", "I guess"

The assistant marked memories: {llm_marks}
(👍=helped, 👎=wrong, ➖=unused)

SCORING RULES:
If YES: upvote 👍 memories, ignore 👎 and ➖
If NO: downvote 👎 memories, ignore 👍 and ➖
If KINDA: slight upvote 👍, slight downvote 👎

Return JSON:
{{"outcome": "yes/no/kinda", "upvote": [1], "downvote": [2]}}
```

**5. Apply scores in agent_chat.py:**
```python
outcome_result = await outcome_detector.analyze(
    conversation,
    surfaced_memories=content_map,
    llm_marks=memory_marks  # Pass parsed marks
)

# Score based on OutcomeDetector's combined analysis
for pos in outcome_result.get("upvote", []):
    if doc_id := position_map.get(pos):
        await self.memory.record_outcome(doc_id, "worked")

for pos in outcome_result.get("downvote", []):
    if doc_id := position_map.get(pos):
        await self.memory.record_outcome(doc_id, "failed")

# Positions not in upvote/downvote = neutral (no scoring)
```

**Scoring Matrix:**
```
                | YES (worked) | KINDA (partial) | NO (failed) |
----------------|--------------|-----------------|-------------|
👍 (helpful)    | upvote       | slight_up       | neutral     |
👎 (unhelpful)  | neutral      | slight_down     | downvote    |
➖ (no_impact)  | neutral      | neutral         | neutral     |
```

**Key insight:** A positive exchange can still downvote bad memories. If the LLM overcame misleading advice and the user was happy, the bad memory STILL gets demoted because the main LLM marked it 👎.

**Fallback behavior:**
- If main LLM doesn't include `<!-- MEM: ... -->` annotation
- OutcomeDetector falls back to inferring usage (Fix #5 behavior)
- All inferred memories get same outcome score

**Files Changed:**
-   `ui-implementation/src-tauri/backend/app/routers/agent_chat.py` (system prompt, parse annotations, pass to OutcomeDetector)
-   `ui-implementation/src-tauri/backend/modules/advanced/outcome_detector.py` (accept llm_marks, updated prompt, return upvote/downvote)
-   `ui-implementation/src-tauri/backend/modules/memory/unified_memory_system.py` (pass llm_marks to OutcomeDetector)

**Status:** ✅ COMPLETE

---

## Files Summary

| File | Changes |
|------|---------|
| `TerminalMessageThread.tsx` | Fix virtualization height measurement and memo comparison |
| `useUpdateChecker.ts` | Persist update dismiss state to localStorage |
| `agent_chat.py` | Fix system prompt + cache organic/cold start + selective scoring + memory attribution parsing |
| `outcome_detector.py` | Add surfaced memories to prompt, return used_positions, accept llm_marks for causal scoring |
| `unified_memory_system.py` | Return doc_ids from get_cold_start_context() |

---

## Testing Checklist

### Fix #1 - Virtualization (Code Verified)
-   [x] Verify messages render without overlapping - `useLayoutEffect` watches `messageContent, messageStreaming, messageEventsLength` (line 235)
-   [x] Verify streaming messages update correctly - memo compares `.content`, `.streaming`, `.events?.length` (lines 406-408)
-   [x] Verify scrolling is smooth - react-window with proper height estimation (line 462)
-   [x] Verify tool execution indicators display correctly - `hasTools` checked in getSize (line 475)
-   [x] Verify long messages render at correct heights - content-based height estimation (lines 463-479)
-   [x] Test with rapid message streaming - useLayoutEffect triggers on content change (line 235)

### Fix #2 - Update Notification Dismiss (Code Verified)
-   [x] Verify dismissing update notification persists across restarts - `localStorage.setItem(DISMISS_KEY, ...)` (line 60)
-   [x] Verify new version notification shows again after dismissing old one - version-keyed dismiss (line 30)

### Fix #4 - Organic Recall Scoring (Code Verified)
-   [x] Verify organic recall doc_ids are added to scoring cache - `organic_doc_ids` list built and cached (lines 796-827)
-   [x] Verify organic recall memories receive outcome scores after user follow-up - doc_ids in `_search_cache` (line 823)

### Fix #5 - Selective Scoring (Code Verified)
-   [x] Verify OutcomeDetector receives surfaced memories with positions - `surfaced_memories` param (outcome_detector.py:63)
-   [x] Verify OutcomeDetector returns used_positions in result - `result["used_positions"]` (outcome_detector.py:170)
-   [x] Verify only used_positions memories get scored (not all cached) - selective scoring in agent_chat.py
-   [x] Verify fallback to score-all when used_positions is empty - default empty list (outcome_detector.py:171)

### Fix #6 - Cold Start Scoring (Code Verified)
-   [x] Verify cold start returns doc_ids alongside formatted context - `cold_start_doc_ids` extracted (line 642)
-   [x] Verify cold start doc_ids are cached for scoring on message 1 - cached to `_search_cache` (lines 652-660)

### Fix #7 - Memory Attribution (Code Verified)
-   [x] Verify main LLM receives memory attribution instruction in system prompt - lines 1640-1647
-   [x] Verify `<!-- MEM: ... -->` annotation is parsed correctly - `parse_memory_marks()` (lines 180-220)
-   [x] Verify annotation is stripped before showing response to user - `re.sub()` removes annotation (line 215)
-   [x] Verify llm_marks are passed to OutcomeDetector - `_memory_marks_cache` (lines 1097-1099)
-   [x] Verify OutcomeDetector returns upvote/downvote arrays - result includes `upvote`, `downvote` (outcome_detector.py:240-241)
-   [x] Verify 👍 memories upvoted on YES outcome - `_analyze_with_marks` logic (outcome_detector.py:211)
-   [x] Verify 👎 memories downvoted on NO outcome - `_analyze_with_marks` logic (outcome_detector.py:212)
-   [x] Verify ➖ memories are never scored - "ignore" in prompt (outcome_detector.py:211-212)
-   [x] Verify fallback to Fix #5 behavior when no annotation present - falls back to `used_positions` (outcome_detector.py:56-58)
-   [x] Verify mixed scoring: positive exchange with some upvotes and some downvotes - KINDA outcome handles both (outcome_detector.py:213)
