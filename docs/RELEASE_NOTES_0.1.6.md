# Roampal v0.1.6 - Hotfix Release

**Release Date:** November 3, 2025
**Build:** `Roampal_0.1.6_x64-setup.exe`

## Bug Fixes

### Collection-Specific Memory Search (CRITICAL FIX)
- **FIXED**: The collection-specific memory search feature mentioned in v0.1.5 release notes now actually works
- **Issue**: v0.1.5 code was hardcoded to always search all collections, ignoring the AI's collection choices
- **Resolution**: AI can now properly target specific memory collections:
  - Search only `memory_bank` for user preferences and identity
  - Search only `patterns` for proven solutions
  - Search only `books` for uploaded documentation
  - Search only `history` or `working` for recent conversations
  - Or search `all` collections (remains the default)

### Knowledge Graph Success Rate Accuracy
- **FIXED**: Concept success rates now calculated accurately
- **Issue**: "Partial" outcomes (searches without user feedback) were diluting success percentages
  - Example: "what" concept showed 9% (2/23) instead of actual 18% (2/11)
- **Resolution**:
  - Backend now calculates: `success_rate = successes / (successes + failures)`
  - Partials excluded from denominator but tracked separately as contextual data
  - UI clearly shows: "✓ 5 worked ✗ 32 failed → 14% success" + "Plus 13 partial results"
- **Impact**: More accurate routing decisions for AI, better transparency for users

## Improvements

### Enhanced Knowledge Graph Concept Modal
- **Improved clarity** of concept detail display:
  - Header: "Learned routing behavior for queries containing 'X'" (was confusing "When you mention 'X', I...")
  - Section title: "Collections Searched" (was vague "Search Strategy")
  - Per-collection stats: "37 with feedback, 14% success" (shows only confirmed outcomes)
  - Track Record: Success rate displayed separately from partial count
  - Explanation: "Plus X partial results (still useful data, just not counted in success rate)"
- **Better context** for both AI routing decisions and user understanding

## Technical Details

### Collection-Specific Search Fix
**Changed File**: `app/routers/agent_chat.py` (line 1959)
- **Before**: `collections = ["all"]` (hardcoded override)
- **After**: `collections = tool_args.get("collections", ["all"])` (respects LLM choice)

**Impact**:
- More efficient searches (fewer collections = faster results)
- Better token usage (targeted results vs everything)
- AI can now implement strategic search patterns (check memory_bank first, fall back to patterns, etc.)

### Knowledge Graph Success Rate Fix
**Backend Changes**: `modules/memory/unified_memory_system.py` (lines 1303-1312)
- Success rate calculation changed from `successes / total` to `successes / (successes + failures)`
- Partials tracked in `total` but excluded from rate calculation
- Fully backward compatible (no schema changes)

**Frontend Changes**: `ui-implementation/src/components/KnowledgeGraph.tsx`
- Updated concept modal UI (lines 657-726)
- Success rate calculation mirrors backend logic
- Separate display of success rate and partial count
- Enhanced clarity for all text labels

## Upgrade Notes

- No breaking changes
- Fully backward compatible (default behavior unchanged)
- All existing data and configurations preserved
- Simply overwrites v0.1.5 installation

## Note to Users

We apologize for the discrepancy in v0.1.5 release notes. This feature was documented but the implementation was inadvertently omitted from the build. v0.1.6 corrects this oversight and delivers the functionality as originally described.
