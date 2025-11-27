"""
A/B Test: Outcome-Based Learning vs Plain Vector Search

This test proves that outcome-based scoring actually improves retrieval quality,
not just that "having memory is better than no memory."

Test Design:
- Condition A (Treatment): Full Roampal with outcome scoring
- Condition B (Control): Same storage, but search ignores scores (pure vector similarity)

Both conditions:
- Use identical embeddings (paraphrase-multilingual-mpnet-base-v2)
- Store the same memories
- Use the same ChromaDB backend

The ONLY difference: whether search results are re-ranked by outcome scores.

This isolates the value of outcome-based learning.
"""

import asyncio
import json
import os
import sys
import statistics
import math
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "ui-implementation" / "src-tauri" / "backend"))
sys.path.insert(0, str(Path(__file__).parent))

from modules.memory.unified_memory_system import UnifiedMemorySystem
from mock_utilities import MockLLMService

# Try to import real embeddings
try:
    from learning_curve_test.real_embedding_service import RealEmbeddingService
    HAS_REAL_EMBEDDINGS = True
except ImportError:
    HAS_REAL_EMBEDDINGS = False
    print("WARNING: Real embeddings not available. Install sentence-transformers.")


# ====================================================================================
# TEST SCENARIOS - Designed to show outcome learning value
# ====================================================================================

TEST_SCENARIOS = [
    {
        "name": "Debugging Advice Quality",
        "description": "User asks about debugging. Some advice worked, some failed.",
        "memories": [
            {"text": "For Python debugging, use print statements to trace variable values", "outcome": "failed", "reason": "Too slow for complex bugs"},
            {"text": "For Python debugging, use pdb with breakpoints for step-through debugging", "outcome": "worked"},
            {"text": "For Python debugging, use logging module with DEBUG level", "outcome": "worked"},
            {"text": "For Python debugging, just read the code carefully", "outcome": "failed", "reason": "Missed the actual bug"},
            {"text": "For Python debugging, use VS Code debugger with watch expressions", "outcome": "worked"},
        ],
        "query": "How should I debug my Python code?",
        "good_answers": ["pdb", "logging", "VS Code debugger"],
        "bad_answers": ["print statements", "read the code"],
    },
    {
        "name": "API Design Patterns",
        "description": "User asked about REST API design. Some patterns worked better.",
        "memories": [
            {"text": "Use nested URLs like /users/123/posts/456 for related resources", "outcome": "failed", "reason": "Got too complex with deep nesting"},
            {"text": "Use flat URLs with query params like /posts?user_id=123", "outcome": "worked"},
            {"text": "Always return 200 OK with error details in body", "outcome": "failed", "reason": "Broke client error handling"},
            {"text": "Use proper HTTP status codes: 201 Created, 404 Not Found, etc", "outcome": "worked"},
            {"text": "Version APIs in URL path like /v1/users", "outcome": "worked"},
        ],
        "query": "What's the best way to design REST API endpoints?",
        "good_answers": ["flat URLs", "HTTP status codes", "Version APIs"],
        "bad_answers": ["nested URLs", "200 OK with error"],
    },
    {
        "name": "Database Optimization",
        "description": "User asked about slow database queries.",
        "memories": [
            {"text": "Add indexes on frequently queried columns to speed up SELECT", "outcome": "worked"},
            {"text": "Use SELECT * to get all data, then filter in application code", "outcome": "failed", "reason": "Made it slower, transferred too much data"},
            {"text": "Use EXPLAIN ANALYZE to understand query execution plan", "outcome": "worked"},
            {"text": "Increase database connection pool size to handle more queries", "outcome": "failed", "reason": "Didn't fix the slow query, just added connections"},
            {"text": "Denormalize tables for read-heavy workloads", "outcome": "worked"},
        ],
        "query": "My database queries are slow, how do I fix them?",
        "good_answers": ["indexes", "EXPLAIN ANALYZE", "Denormalize"],
        "bad_answers": ["SELECT *", "connection pool"],
    },
    {
        "name": "Git Workflow",
        "description": "User asked about Git branching strategies.",
        "memories": [
            {"text": "Commit directly to main branch for faster iteration", "outcome": "failed", "reason": "Broke production twice"},
            {"text": "Use feature branches and pull requests for code review", "outcome": "worked"},
            {"text": "Rebase feature branches before merging to keep history clean", "outcome": "worked"},
            {"text": "Use git push --force to fix mistakes quickly", "outcome": "failed", "reason": "Lost teammate's commits"},
            {"text": "Write descriptive commit messages explaining why, not just what", "outcome": "worked"},
        ],
        "query": "What's a good Git workflow for a team?",
        "good_answers": ["feature branches", "pull requests", "Rebase", "commit messages"],
        "bad_answers": ["directly to main", "push --force"],
    },
    {
        "name": "Error Handling",
        "description": "User asked about handling errors in code.",
        "memories": [
            {"text": "Catch all exceptions with a generic except: block to prevent crashes", "outcome": "failed", "reason": "Hid real bugs, made debugging impossible"},
            {"text": "Use specific exception types and handle each appropriately", "outcome": "worked"},
            {"text": "Log errors with full stack traces for debugging", "outcome": "worked"},
            {"text": "Return None on error and let caller figure it out", "outcome": "failed", "reason": "Caused NoneType errors downstream"},
            {"text": "Use Result types or Either monads for explicit error handling", "outcome": "worked"},
        ],
        "query": "How should I handle errors in my application?",
        "good_answers": ["specific exception", "Log errors", "Result types"],
        "bad_answers": ["generic except", "Return None"],
    },
]


# ====================================================================================
# STATISTICAL FUNCTIONS
# ====================================================================================

def cohens_d(treatment: List[float], control: List[float]) -> float:
    """Calculate Cohen's d effect size"""
    if not treatment or not control:
        return 0.0
    mean_t = statistics.mean(treatment)
    mean_c = statistics.mean(control)

    # Handle edge case where all values are the same
    try:
        var_t = statistics.variance(treatment) if len(treatment) > 1 else 0
        var_c = statistics.variance(control) if len(control) > 1 else 0
    except:
        var_t, var_c = 0, 0

    n_t, n_c = len(treatment), len(control)

    # Pooled standard deviation
    if n_t + n_c <= 2:
        return 0.0
    pooled_var = ((n_t - 1) * var_t + (n_c - 1) * var_c) / (n_t + n_c - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 0.001

    return (mean_t - mean_c) / pooled_std


def paired_t_test(treatment: List[float], control: List[float]) -> Tuple[float, float]:
    """Simple paired t-test returning (t_statistic, p_value)"""
    if len(treatment) != len(control) or len(treatment) < 2:
        return (0.0, 1.0)

    differences = [t - c for t, c in zip(treatment, control)]
    mean_diff = statistics.mean(differences)

    try:
        std_diff = statistics.stdev(differences)
    except:
        std_diff = 0.001

    if std_diff == 0:
        return (float('inf') if mean_diff > 0 else 0.0, 0.005 if mean_diff > 0 else 1.0)

    n = len(differences)
    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Approximate p-value from t-statistic (df = n-1)
    abs_t = abs(t_stat)
    if abs_t > 4.0:
        p_value = 0.005
    elif abs_t > 3.0:
        p_value = 0.01
    elif abs_t > 2.5:
        p_value = 0.025
    elif abs_t > 2.0:
        p_value = 0.05
    elif abs_t > 1.5:
        p_value = 0.1
    else:
        p_value = 0.2

    return (t_stat, p_value)


# ====================================================================================
# TEST FUNCTIONS
# ====================================================================================

async def run_scenario_with_outcomes(
    scenario: Dict,
    data_dir: str,
    embedding_service
) -> Dict:
    """
    Run scenario WITH outcome-based ranking (Treatment condition).

    Memories are scored based on outcomes, and search results are ranked by score.
    """
    name = scenario["name"]

    # Create memory system
    system = UnifiedMemorySystem(
        data_dir=data_dir,
        use_server=False,
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = embedding_service

    # Store all memories
    doc_ids = []
    for mem in scenario["memories"]:
        doc_id = await system.store(
            text=mem["text"],
            collection="working",
            metadata={"scenario": name}
        )
        doc_ids.append((doc_id, mem.get("outcome", "unknown")))

    # Apply outcomes TWICE to simulate real usage (not 3x, to avoid deletion)
    # worked: 0.5 -> 0.7 -> 0.9 (near promotion)
    # failed: 0.5 -> 0.2 -> 0.0 (very low, but not deleted)
    for _ in range(2):
        for doc_id, outcome in doc_ids:
            if outcome in ["worked", "failed"]:
                await system.record_outcome(
                    doc_id=doc_id,
                    outcome=outcome
                )

    # Search and evaluate
    results = await system.search(scenario["query"], collections=["working"], limit=3)

    # Debug: show scores
    print(f"      [DEBUG-Treatment] Scores: {[(r.get('text', '')[:30], r.get('metadata', {}).get('score', 'N/A')) for r in results]}")

    # Score: how many top-3 results are "good" answers?
    good_count = 0
    bad_count = 0

    for result in results:
        text = result.get("text", "").lower()

        for good in scenario["good_answers"]:
            if good.lower() in text:
                good_count += 1
                break

        for bad in scenario["bad_answers"]:
            if bad.lower() in text:
                bad_count += 1
                break

    # Precision: good / (good + bad), or good / 3 if no bad found
    total_relevant = good_count + bad_count
    precision = good_count / 3  # Out of top 3 results

    return {
        "scenario": name,
        "condition": "with_outcomes",
        "top_3_good": good_count,
        "top_3_bad": bad_count,
        "precision": precision,
        "results": [r.get("text", "")[:80] for r in results]
    }


async def run_scenario_without_outcomes(
    scenario: Dict,
    data_dir: str,
    embedding_service
) -> Dict:
    """
    Run scenario WITHOUT outcome-based ranking (Control condition).

    Same memories stored, but we DON'T apply outcome scoring.
    Search returns pure vector similarity ranking.
    """
    name = scenario["name"]

    # Create memory system
    system = UnifiedMemorySystem(
        data_dir=data_dir,
        use_server=False,
        llm_service=MockLLMService()
    )
    await system.initialize()
    system.embedding_service = embedding_service

    # Store all memories (but DON'T record outcomes)
    for mem in scenario["memories"]:
        await system.store(
            text=mem["text"],
            collection="working",
            metadata={"scenario": name}
        )

    # NO outcome recording - pure vector search

    # Search and evaluate
    results = await system.search(scenario["query"], collections=["working"], limit=3)

    # Debug: show scores
    print(f"      [DEBUG-Control] Scores: {[(r.get('text', '')[:30], r.get('metadata', {}).get('score', 'N/A')) for r in results]}")

    # Score: how many top-3 results are "good" answers?
    good_count = 0
    bad_count = 0

    for result in results:
        text = result.get("text", "").lower()

        for good in scenario["good_answers"]:
            if good.lower() in text:
                good_count += 1
                break

        for bad in scenario["bad_answers"]:
            if bad.lower() in text:
                bad_count += 1
                break

    precision = good_count / 3

    return {
        "scenario": name,
        "condition": "without_outcomes",
        "top_3_good": good_count,
        "top_3_bad": bad_count,
        "precision": precision,
        "results": [r.get("text", "")[:80] for r in results]
    }


# ====================================================================================
# MAIN TEST
# ====================================================================================

async def main():
    print("=" * 70)
    print("A/B TEST: Outcome-Based Learning vs Plain Vector Search")
    print("=" * 70)
    print()
    print("This test proves that outcome scoring improves retrieval quality.")
    print("Both conditions use identical embeddings and storage.")
    print("The ONLY difference: whether search results are ranked by outcome scores.")
    print()

    if not HAS_REAL_EMBEDDINGS:
        print("ERROR: This test requires real embeddings.")
        print("Install: pip install sentence-transformers")
        return

    # Initialize embedding service
    print("Loading embedding model (paraphrase-multilingual-mpnet-base-v2)...")
    embedding_service = RealEmbeddingService()
    print("Model loaded.\n")

    # Create test directories
    test_dir = Path(__file__).parent / "ab_test_data"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    treatment_results = []
    control_results = []

    print("-" * 70)
    print("Running scenarios...")
    print("-" * 70)

    for i, scenario in enumerate(TEST_SCENARIOS):
        print(f"\n[{i+1}/{len(TEST_SCENARIOS)}] {scenario['name']}")
        print(f"    Query: \"{scenario['query']}\"")

        # Treatment: WITH outcome scoring
        treatment_dir = str(test_dir / f"treatment_{i}")
        os.makedirs(treatment_dir, exist_ok=True)
        treatment = await run_scenario_with_outcomes(scenario, treatment_dir, embedding_service)
        treatment_results.append(treatment)
        print(f"    [WITH outcomes]    Good: {treatment['top_3_good']}/3, Bad: {treatment['top_3_bad']}/3, Precision: {treatment['precision']:.0%}")

        # Control: WITHOUT outcome scoring
        control_dir = str(test_dir / f"control_{i}")
        os.makedirs(control_dir, exist_ok=True)
        control = await run_scenario_without_outcomes(scenario, control_dir, embedding_service)
        control_results.append(control)
        print(f"    [WITHOUT outcomes] Good: {control['top_3_good']}/3, Bad: {control['top_3_bad']}/3, Precision: {control['precision']:.0%}")

        diff = treatment['precision'] - control['precision']
        if diff > 0:
            print(f"    -> Outcome scoring improved precision by {diff:.0%}")
        elif diff < 0:
            print(f"    -> Outcome scoring decreased precision by {abs(diff):.0%}")
        else:
            print(f"    -> No difference")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    treatment_precisions = [r['precision'] for r in treatment_results]
    control_precisions = [r['precision'] for r in control_results]

    mean_treatment = statistics.mean(treatment_precisions)
    mean_control = statistics.mean(control_precisions)

    print(f"\nMean Precision (WITH outcomes):    {mean_treatment:.1%}")
    print(f"Mean Precision (WITHOUT outcomes): {mean_control:.1%}")
    print(f"Improvement:                       {mean_treatment - mean_control:+.1%}")

    # Statistical analysis
    d = cohens_d(treatment_precisions, control_precisions)
    t_stat, p_value = paired_t_test(treatment_precisions, control_precisions)

    print(f"\nStatistical Analysis:")
    print(f"  Cohen's d:    {d:.2f}", end="")
    if d >= 0.8:
        print(" (LARGE effect)")
    elif d >= 0.5:
        print(" (medium effect)")
    elif d >= 0.2:
        print(" (small effect)")
    else:
        print(" (negligible)")

    print(f"  t-statistic:  {t_stat:.2f}")
    print(f"  p-value:      {p_value}")

    if p_value < 0.05:
        print("\n✅ STATISTICALLY SIGNIFICANT: Outcome scoring improves retrieval quality.")
    else:
        print("\n⚠️  NOT STATISTICALLY SIGNIFICANT: Need more scenarios or larger effect.")

    # Save results
    results_file = test_dir / "ab_test_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "treatment_results": treatment_results,
            "control_results": control_results,
            "statistics": {
                "mean_treatment": mean_treatment,
                "mean_control": mean_control,
                "improvement": mean_treatment - mean_control,
                "cohens_d": d,
                "t_statistic": t_stat,
                "p_value": p_value
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Cleanup
    print("\nCleaning up test data...")
    shutil.rmtree(test_dir)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
