# Learning Curve Test

## Purpose

Prove that Roampal's memory system **learns over time** by showing measurable performance improvement across repeated interactions.

## What This Test Proves

### ✅ Claims Being Tested:
1. **Long-term memory retention** - System remembers information from early conversations
2. **Preference learning** - System learns user/sensei preferences through outcome scoring
3. **Retrieval quality improves** - More data = better, more relevant search results
4. **Promotion/KG routing helps** - High-value memories surface faster over time

### ❌ What This Does NOT Test:
- LLM response quality (we only test memory retrieval)
- Real-time conversation (simulated interactions)
- Multi-user scenarios (single user story)

## Test Design

### Dataset: Storyteller Conversations

Uses `conversation_dataset_storytellers_150.json` which contains:
- **12 different stories** with 12-13 visits each
- **138/150 interactions require memory** of previous visits
- **Different senseis** with distinct preferences (ground truth)

Example story: "Memory Thieves Don't Cry" (cyberpunk noir, 13 visits)
- Visit 1: User introduces story premise
- Visit 3: User asks about character development
- Visit 7: User asks about tone/style
- Visit 13: User asks about ending (needs context from all previous visits)

### Test Methodology

```
1. BASELINE (Cold Start)
   - Fresh system with no memories
   - Test: Can it answer questions about the story? (Expected: ~10-20%)

2. LEARNING PHASE (Feed 13 visits)
   For each visit:
   - User asks question
   - System searches memory
   - Store conversation in 'history'
   - Record outcome (worked/failed based on relevance)
   - Store sensei preferences in 'memory_bank'

3. CHECKPOINTS (After 3, 6, 9, 13 visits)
   - Test: Can it answer questions about the story?
   - Measure accuracy improvement

4. FINAL TEST (After 13 visits)
   - Test: Can it answer questions about the story? (Expected: 60-80%)

5. PROOF OF LEARNING
   - Plot learning curve (accuracy vs. # visits)
   - Assert: Final accuracy > Baseline + 30%
```

### Test Questions

For each story, we test:

1. **Story recall**: "What's the protagonist's profession?" → "memory-hacker"
2. **Preference recall**: "What tone does sensei prefer?" → "morally gray"
3. **Conversation history**: "What did we discuss in visit 3?" → [visit 3 content]
4. **Genre/domain**: "What genre is this?" → "cyberpunk_noir"
5. **Constraints**: "What should I avoid?" → "clean resolutions"

### Success Criteria

**PASS** if:
- Learning curve shows consistent upward trend
- Final accuracy > Baseline accuracy + 30%
- No regression (accuracy never drops at later checkpoints)

**FAIL** if:
- Learning curve is flat (no improvement)
- Final accuracy < Baseline + 30%
- Performance degrades with more data

## Files in This Folder

- `README.md` - This file
- `DESIGN.md` - Detailed test design and methodology
- `test_learning_curve.py` - Main test script
- `storyteller_simulator.py` - Automated conversation simulator
- `learning_metrics.py` - Accuracy measurement and scoring
- `results/` - Test results and learning curve graphs

## Running the Test

```bash
cd benchmarks/comprehensive_test/learning_curve_test
python test_learning_curve.py
```

**Expected runtime:** ~2-3 minutes with real embeddings

**Output:**
```
Testing story: Memory Thieves Don't Cry (13 visits)

Baseline (0 visits): 15.0% accuracy
After 3 visits: 33.3% accuracy
After 6 visits: 50.0% accuracy
After 9 visits: 66.7% accuracy
After 13 visits: 78.3% accuracy

✅ LEARNING PROVEN: 15.0% → 78.3% (+63.3%)
Learning curve saved to results/learning_curve.png
```

## Dependencies

- Uses **real embeddings** (sentence-transformers) for semantic similarity
- First run will download `all-MiniLM-L6-v2` model (~80MB)
- Optional: Can fall back to keyword matching if model unavailable

## What Success Looks Like

```
Accuracy
   ^
80%|                            ●  (visit 13)
   |                        ●
60%|                    ●
   |                ●
40%|            ●
   |        ●
20%|    ●
   |●
 0%+----------------------------> Visits
   0   3   6   9   12

📈 LEARNING CURVE: System gets smarter with more data
```

## Ablation Studies (Optional)

Test what features drive learning by disabling them:

1. **Disable promotion**: Does outcome-based promotion matter?
2. **Disable KG routing**: Do knowledge graphs help?
3. **Disable scoring**: Is outcome scoring critical?

Each should show reduced learning if the feature matters.
