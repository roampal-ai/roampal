# Deprecated Tests

This folder contains experimental and outdated tests that are kept for reference but **should not be used** for benchmarking.

## Why These Tests Are Deprecated

### LOCOMO Tests (All variants)
**Files**: `test_locomo_*.py`

**Problem**: LOCOMO (ACL 2024) is too easy to be a meaningful benchmark.
- Only ~26K tokens per question (fits in modern LLM context windows)
- Doesn't actually test long-term memory - just good embeddings + good LLM
- `test_locomo_full_evidence.py` got 100% but used SimpleMemoryStore (basic Python list), not Roampal's real system

**Why kept**: Shows that simple vector search works on clean data. Useful for understanding what DOESN'T prove learning works.

### LongMemEval Failed Attempts
**Files**: `test_longmemeval_REAL.py`, `test_longmemeval_FULL_ROAMPAL.py`, `test_longmemeval_ACCUMULATE.py`, `test_longmemeval_roampal.py`

**Problems**:
1. **Wrong dataset**: Used `longmemeval_s_cleaned.json` (no dialogue content)
2. **Bulk cramming**: Dumped all data at once, no incremental learning
3. **No feedback loop**: Couldn't test outcome-based learning

**Fixed version**: `test_longmemeval_LEARNING.py` (in parent directory)

### Utility Scripts
**Files**: `rejudge_with_claude.py`, `test_sample_predictions.py`, `verify_results.py`

**Problem**: One-off utility scripts for debugging, not actual benchmarks.

### Experimental Tests
**Files**: `test_mem0_v1_comparison.py`, `test_precision_ceiling.py`, `test_precision_stress.py`

**Problem**: Experimental/stress tests that don't validate core functionality.

---

## What to Use Instead

See `../README.md` for legitimate tests:
- **`test_longmemeval_LEARNING.py`** - The real test (incremental learning with feedback)
- **`test_longmemeval_OFFICIAL.py`** - Official benchmark comparison
- **`test_standard_metrics.py`** - Core memory system validation
- **`test_learning_curve_REAL.py`** - Learning effectiveness

---

## Key Lessons Learned

1. **LOCOMO is a marketing benchmark** - Too easy, doesn't test memory systems
2. **LongMemEval is the real benchmark** - 115K tokens, tests actual long-term memory
3. **Bulk data ingestion doesn't test learning** - Need incremental feedback loops
4. **SimpleMemoryStore beating UnifiedMemorySystem** - Proves clean data doesn't need learning (complexity hurts on easy problems)

---

*These tests are kept for historical reference and to avoid repeating the same mistakes.*
