# Roampal Benchmarks

**Last Updated**: 2026-04-15
**Benchmark Suite**: [roampal-labs](https://github.com/roampal-ai/roampal-labs)

---

## Executive Summary

Roampal's retrieval system has been validated on the **LoCoMo dataset** (1,537 non-adversarial questions from real conversations) using the roampal-labs benchmark suite.

**Headline Result**:
> **TagCascade retrieval: 27.3% Hit@1 (p<0.0001 vs Wilson+CE blend). Wilson scoring hurts retrieval by 4.3 points.**

---

## LoCoMo Evaluation (v0.3.1)

| Config | Hit@1 Clean | Hit@1 Poison | p-value |
|--------|-------------|--------------|---------|
| **TagCascade + cosine** | **27.3%** | **29.0%** | **baseline** |
| Overlap + cosine | 25.8% | 28.0% | p=0.0003 |
| Pure CE | 25.4% | 28.4% | — |
| TagCascade + Wilson | 23.0% | 25.0% | p<0.0001 |

### Key Findings

- **Wilson scoring hurts retrieval** by 4.3 points in every configuration tested (p<0.0001). Wilson is still used for display and promotion thresholds, but removed from ranking.
- **Two-lane retrieval** adds +6.1 Hit@1 (p<0.0001). Separating summaries and facts into dedicated lanes improves results.
- **Nursery slot**: zero benefit (p=1.0). Removed.
- **TagCascade** can never perform worse than cosine — worst case falls back to unfiltered cosine search.

### What the Numbers Mean

- **Hit@1**: Did the system retrieve the right memory as the top result?
- **Clean vs Poison**: Tests resilience against adversarial/contradictory memories
- **TagCascade**: Tag-routed search with overlap counting, then cross-encoder reranking (ONNX)

---

## Architecture Under Test

```
Query
  |
Tag matching (word-boundary regex against known tag index)
  |
Tag-routed search (if tags match) OR cosine fallback
  Per-tag ChromaDB queries with noun_tags $contains filter
  Overlap counting: how many query tags each result matches
  Tier-fill pool: highest overlap first, cosine tiebreak within tier
  Cosine fill remaining slots from unfiltered search
  |
CE reranks top 40 candidates (ONNX, mmarco-mMiniLMv2-L12-H384-v1)
  Raw CE score as final ranking (NO Wilson blend)
  |
Two-lane injection:
  Lane 1: 4 summaries (memory_type != "fact")
  Lane 2: 4 facts (memory_type == "fact")
  Total: 8 memories per context
```

---

## Running Benchmarks

Benchmarks live in the [roampal-labs](https://github.com/roampal-ai/roampal-labs) repository, not in this repo. See roampal-labs README for setup and execution instructions.

---

## Historical Notes

Previous internal benchmarks (v0.2.5 era, synthetic adversarial tests) were removed in v0.3.1. The LoCoMo evaluation provides a more rigorous, standardized evaluation using real conversation data rather than synthetic scenarios.

The key architectural change from v0.2.x to v0.3.1:
- **Removed**: Knowledge Graphs (Routing KG, Content KG, Action KG), Wilson-blended ranking, emoji-based scoring
- **Added**: TagCascade retrieval, sidecar LLM processing, ONNX cross-encoder, two-lane injection
- **Result**: Simpler architecture with better benchmark results
