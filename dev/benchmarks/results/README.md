# Benchmark Results

Versioned benchmark results from key tests. Each file is named with the release version that produced it.

---

## Roampal vs Vector DB (30 adversarial coding scenarios)

**Test**: `test_roampal_vs_vector_db.py`

| Version | File | Roampal | Vector DB | p-value | Cohen's d |
|---------|------|---------|-----------|---------|-----------|
| v0.2.3 | `statistical_results_v0.2.3.json` | **100%** (30/30) | 3.3% (1/30) | 0.001 | 7.49 |
| v0.2.6 | `statistical_results_v0.2.6.json` | **97%** (29/30) | 0% (0/30) | 0.001 | 7.49 |

**Note**: v0.2.6 used 768d embeddings (all-mpnet-base-v2), v0.2.3 used 384d (all-MiniLM-L6-v2).

---

## Token Efficiency (100 adversarial finance scenarios)

**Test**: `test_token_efficiency.py`

| Version | File | Roampal | RAG | Tokens/Query |
|---------|------|---------|-----|--------------|
| v0.2.3 | `token_efficiency_results_v0.2.3.json` | **100%** (100/100) | 0% | 20 |
| v0.2.6 | `token_efficiency_results_v0.2.6.json` | **96%** (96/100) | 1% | 20 |

---

## 4-Way Comprehensive Benchmark (200 tests)

**Test**: `test_comprehensive_benchmark.py`

| Version | File | Description |
|---------|------|-------------|
| v0.2.6 | `comprehensive_4way_benchmark_v0.2.6.json` | 4 conditions × 5 maturity levels × 10 scenarios |

**Results Summary**:

| Condition | Top-1 | MRR | nDCG@5 |
|-----------|-------|-----|--------|
| RAG Baseline | 10% | 0.55 | 0.67 |
| Reranker Only | 20% | 0.60 | 0.70 |
| Outcomes Only | **50%** | **0.75** | **0.82** |
| Full Roampal | 44% | 0.72 | 0.79 |

**Key Finding**: Outcome learning (+40 pts) dominates reranker (+10 pts) by 4×.

---

## Learning Curve (10 scenarios × 5 maturity levels)

**Test**: `test_learning_curve.py`

| Version | File | Cold Start | After 3 Uses | Improvement |
|---------|------|------------|--------------|-------------|
| v0.2.5 | `learning_curve_results_v0.2.5.json` | 10% | **100%** | +90pp |

**Key Finding**: Just 3 uses is enough for 100% accuracy on adversarial queries.

---

## Statistical Significance (Real Embeddings)

**Test**: `learning_curve_test/test_statistical_significance_synthetic.py`

| Version | File | Before | After | p-value | Cohen's d |
|---------|------|--------|-------|---------|-----------|
| v0.2.3 | `learning_curve_real_embeddings_v0.2.3.json` | 0% | **93.3%** | 0.005 | 13.4 |

---

## Test Methodology

All tests use:
- Real semantic embeddings (sentence-transformers)
- Adversarial queries designed to trick vector search
- Synthetic scenarios with known ground truth
- Statistical significance testing (paired t-test, Cohen's d)

See [../README.md](../README.md) for full test documentation.
