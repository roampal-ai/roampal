# Roampal Benchmark Suite

Comprehensive tests validating Roampal's memory system architecture and performance.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run standard benchmarks
pytest benchmarks/ -v

# Run specific test
pytest benchmarks/test_kg_routing.py -v
```

## Core System Tests (Working)

### Memory Architecture Tests
- **`test_standard_metrics.py`** ✅ - Comprehensive memory system validation
  - Memory storage, retrieval, ranking
  - Cross-collection search
  - Metadata filtering
  - **Status**: 6/6 tests passing

- **`test_kg_routing.py`** ✅ - Knowledge Graph routing accuracy
  - Collection routing based on query type
  - Learning from outcome patterns
  - Routing confidence metrics
  - **Status**: 7/7 tests passing

- **`test_learning_curve_REAL.py`** ✅ - Learning effectiveness over time
  - Does accuracy improve with feedback?
  - Incremental learning validation (semantic confusion attack)
  - Learning curve analysis
  - **Status**: 1/1 tests passing

### Experimental Tests
- **`test_natural_conversation.py`** ⚠️ - Natural feedback simulation
  - Simulates real usage with conversational feedback
  - **Status**: 100% accuracy (unrealistic - needs improvement)
  - **Note**: Test is too lenient, marked for revision

## Academic Benchmarks

**Note**: LongMemEval benchmark tests have been moved to `deprecated/` as they don't accurately reflect real-world usage patterns. The tests were attempting to measure learning over 500 questions, but the methodology didn't match how the system is actually used.

**Previous results**:
- test_longmemeval_LEARNING.py: 25.60% (vs GPT-4o: 64%, EmergenceMem: 82.4%)
- Performance was poor due to test methodology not matching system design

**Future work**: Need to develop better benchmarks that test real conversational usage patterns rather than artificial academic datasets.

---

## Deprecated Tests

See `benchmarks/deprecated/` for experimental/outdated tests:
- **API-broken tests**: test_outcome_tracking.py, test_cold_start.py, test_memory_ranking.py, test_books_search.py, test_stale_data.py (outdated APIs)
- **Failed academic benchmarks**: test_longmemeval_LEARNING.py, test_longmemeval_OFFICIAL.py (methodology mismatch)
- **Experimental tests**: test_natural_learning.py, test_pure_learning.py (redundant experiments)
- **LOCOMO attempts**: Various LOCOMO test files (benchmark too easy - 26K tokens fits in context)

**Note**: Many tests were deprecated due to API changes in the memory system or because the benchmark methodology didn't match real-world usage patterns.

## Running Tests

### Quick Validation
```bash
# Run all working tests
pytest benchmarks/ -v

# Run specific test
pytest benchmarks/test_kg_routing.py -v

# Run with benchmark output
pytest benchmarks/test_standard_metrics.py -v --benchmark-only
```

### Test Results
- **test_standard_metrics.py**: ~20s runtime, validates core memory operations
- **test_kg_routing.py**: ~8s runtime, validates routing accuracy
- **test_learning_curve_REAL.py**: ~6s runtime, validates learning under semantic confusion

## Current Limitations

The benchmark suite currently focuses on **unit-level validation** rather than end-to-end real-world usage:
- Tests validate individual components (storage, retrieval, routing)
- Missing comprehensive long-term learning benchmarks
- Need better simulation of real conversational patterns

**Future work**: Develop realistic multi-session benchmarks that test actual usage patterns over time.
