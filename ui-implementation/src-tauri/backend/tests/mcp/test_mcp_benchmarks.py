"""
MCP Tool Performance Benchmarks

Measures performance of the MCP tool layer, including:
- Tool call latency (p50, p95, p99)
- End-to-end workflow latency (search → record cycle)
- Action-Effectiveness KG overhead
- Cache management overhead

Ported from dev/benchmarks/ to test MCP interface layer.

Usage:
    cd ui-implementation/src-tauri/backend
    python -m pytest tests/mcp/test_mcp_benchmarks.py -v

Or run directly:
    python tests/mcp/test_mcp_benchmarks.py
"""

import os
os.environ["ROAMPAL_BENCHMARK_MODE"] = "true"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import time
import statistics
import shutil
import pytest
from datetime import datetime

from modules.memory.unified_memory_system import UnifiedMemorySystem


class MCPBenchmarkSuite:
    """Test harness for MCP tool benchmarks."""

    def __init__(self, test_name: str = "mcp_benchmark"):
        self.memory_system = None
        self.test_data_dir = Path(f"./test_data_{test_name}")

    async def initialize(self):
        """Initialize fresh memory system."""
        if self.test_data_dir.exists():
            try:
                shutil.rmtree(self.test_data_dir)
            except:
                pass

        self.test_data_dir.mkdir(exist_ok=True)

        self.memory_system = UnifiedMemorySystem(
            data_dir=str(self.test_data_dir),
            use_server=False
        )
        await self.memory_system.initialize()
        print(f"[INIT] Memory system ready at {self.test_data_dir}")

    async def cleanup(self):
        """Cleanup test data."""
        if self.memory_system:
            await asyncio.sleep(0.3)
        if self.test_data_dir.exists():
            try:
                shutil.rmtree(self.test_data_dir)
            except:
                pass


# =============================================================================
# Benchmark 1: search_memory Tool Latency
# =============================================================================

@pytest.mark.asyncio
async def test_search_memory_latency():
    """
    Benchmark: search_memory tool latency
    Target: <100ms p95 (same as service-level target)
    Overhead: MCP formatting + cache management

    Tests the full search_memory code path including:
    - Query routing
    - Embedding generation
    - Vector search
    - Result formatting
    - Doc_id caching
    - Action tracking
    """
    from tests.mcp.mcp_tool_harness import call_search_memory, clear_all_caches

    suite = MCPBenchmarkSuite("search_latency")
    await suite.initialize()

    # Seed 100 memories
    print("\n[SEARCH BENCHMARK] Seeding 100 memories...")
    for i in range(100):
        await suite.memory_system.store_memory_bank(
            text=f"Memory {i}: Information about topic {i % 10} with context",
            tags=[f"topic_{i % 10}"],
            importance=0.8,
            confidence=0.9
        )

    # Run 100 searches
    print("[SEARCH BENCHMARK] Running 100 search queries...")
    latencies = []
    queries = [
        "python programming", "machine learning", "web development",
        "data science", "system design", "cloud computing",
        "cybersecurity", "mobile development", "devops", "databases"
    ]

    for i in range(100):
        clear_all_caches()  # Fresh cache each search
        query = queries[i % len(queries)]

        start = time.perf_counter()
        result = await call_search_memory(
            memory=suite.memory_system,
            arguments={"query": query, "limit": 5},
            session_id=f"bench_{i}"
        )
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    # Calculate percentiles
    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    avg = statistics.mean(latencies)

    print(f"\n[RESULTS - search_memory]")
    print(f"  p50: {p50:.2f}ms")
    print(f"  p95: {p95:.2f}ms")
    print(f"  p99: {p99:.2f}ms")
    print(f"  avg: {avg:.2f}ms")
    print(f"  Target: <100ms p95")

    await suite.cleanup()

    # Note: We don't assert hard targets in benchmarks since they're hardware-dependent
    # But we log if target is exceeded
    if p95 > 100:
        print(f"  [WARNING] p95 {p95:.2f}ms exceeds 100ms target")
    else:
        print(f"  [PASS] p95 within target")

    return {"p50": p50, "p95": p95, "p99": p99, "avg": avg}


# =============================================================================
# Benchmark 2: add_to_memory_bank Tool Latency
# =============================================================================

@pytest.mark.asyncio
async def test_add_memory_latency():
    """
    Benchmark: add_to_memory_bank tool latency
    Measures store + action tracking overhead
    """
    from tests.mcp.mcp_tool_harness import call_add_to_memory_bank, clear_all_caches

    suite = MCPBenchmarkSuite("add_latency")
    await suite.initialize()

    print("\n[ADD BENCHMARK] Running 50 store operations...")
    latencies = []

    for i in range(50):
        clear_all_caches()

        start = time.perf_counter()
        result = await call_add_to_memory_bank(
            memory=suite.memory_system,
            arguments={
                "content": f"Test fact {i}: User preference number {i}",
                "tags": ["preference", f"test_{i % 5}"],
                "importance": 0.7 + (i % 3) * 0.1,
                "confidence": 0.8
            },
            session_id=f"bench_add_{i}"
        )
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    avg = statistics.mean(latencies)

    print(f"\n[RESULTS - add_to_memory_bank]")
    print(f"  p50: {p50:.2f}ms")
    print(f"  p95: {p95:.2f}ms")
    print(f"  avg: {avg:.2f}ms")

    await suite.cleanup()

    return {"p50": p50, "p95": p95, "avg": avg}


# =============================================================================
# Benchmark 3: Full Workflow Latency (Search → Record Cycle)
# =============================================================================

@pytest.mark.asyncio
async def test_full_workflow_latency():
    """
    Benchmark: Full MCP workflow (get_context → search → record)

    This is the realistic usage pattern:
    1. get_context_insights (fast lookup)
    2. search_memory (embedding search)
    3. record_response (score + store)
    """
    from tests.mcp.mcp_tool_harness import (
        call_search_memory, call_record_response,
        call_get_context_insights, clear_all_caches
    )

    suite = MCPBenchmarkSuite("workflow_latency")
    await suite.initialize()

    # Seed data
    print("\n[WORKFLOW BENCHMARK] Seeding 50 memories...")
    for i in range(50):
        await suite.memory_system.store_memory_bank(
            text=f"User fact {i}: Information about {i % 5}",
            tags=[f"topic_{i % 5}"]
        )

    print("[WORKFLOW BENCHMARK] Running 20 full workflows...")
    workflow_latencies = []
    step_latencies = {"context": [], "search": [], "record": []}

    for i in range(20):
        clear_all_caches()
        session_id = f"workflow_{i}"
        query = f"topic {i % 5}"

        workflow_start = time.perf_counter()

        # Step 1: get_context_insights
        ctx_start = time.perf_counter()
        await call_get_context_insights(
            memory=suite.memory_system,
            arguments={"query": query},
            session_id=session_id,
            data_path=suite.test_data_dir
        )
        step_latencies["context"].append((time.perf_counter() - ctx_start) * 1000)

        # Step 2: search_memory
        search_start = time.perf_counter()
        await call_search_memory(
            memory=suite.memory_system,
            arguments={"query": query, "limit": 5},
            session_id=session_id
        )
        step_latencies["search"].append((time.perf_counter() - search_start) * 1000)

        # Step 3: record_response
        record_start = time.perf_counter()
        await call_record_response(
            memory=suite.memory_system,
            arguments={
                "key_takeaway": f"Learned about topic {i % 5}",
                "outcome": "worked"
            },
            session_id=session_id
        )
        step_latencies["record"].append((time.perf_counter() - record_start) * 1000)

        workflow_end = time.perf_counter()
        workflow_latencies.append((workflow_end - workflow_start) * 1000)

    # Calculate stats
    print(f"\n[RESULTS - Full Workflow]")
    print(f"  Total workflow:")
    print(f"    p50: {statistics.median(workflow_latencies):.2f}ms")
    print(f"    p95: {sorted(workflow_latencies)[int(len(workflow_latencies) * 0.95)]:.2f}ms")
    print(f"    avg: {statistics.mean(workflow_latencies):.2f}ms")

    print(f"\n  By step:")
    for step, lats in step_latencies.items():
        print(f"    {step}: p50={statistics.median(lats):.2f}ms, avg={statistics.mean(lats):.2f}ms")

    await suite.cleanup()

    return {
        "workflow_p50": statistics.median(workflow_latencies),
        "workflow_p95": sorted(workflow_latencies)[int(len(workflow_latencies) * 0.95)],
        "steps": {k: statistics.median(v) for k, v in step_latencies.items()}
    }


# =============================================================================
# Benchmark 4: Cache Management Overhead
# =============================================================================

@pytest.mark.asyncio
async def test_cache_overhead():
    """
    Benchmark: Cache management overhead

    Measures the cost of:
    - Doc_id caching per search
    - Action tracking per tool call
    - Cache cleanup on record_response
    """
    from tests.mcp.mcp_tool_harness import (
        call_search_memory, call_record_response,
        get_search_cache, get_action_cache, clear_all_caches
    )

    suite = MCPBenchmarkSuite("cache_overhead")
    await suite.initialize()

    # Seed data
    for i in range(20):
        await suite.memory_system.store_memory_bank(
            text=f"Test fact {i}",
            tags=["test"]
        )

    print("\n[CACHE BENCHMARK] Measuring cache operations...")

    # Measure search with caching
    caching_latencies = []
    for i in range(50):
        clear_all_caches()

        start = time.perf_counter()
        await call_search_memory(
            memory=suite.memory_system,
            arguments={"query": "test", "limit": 5},
            session_id=f"cache_{i}"
        )
        end = time.perf_counter()
        caching_latencies.append((end - start) * 1000)

        # Verify cache populated
        cache = get_search_cache(f"cache_{i}")
        assert cache is not None
        assert len(cache["doc_ids"]) > 0

    # Measure record with cache cleanup
    cleanup_latencies = []
    for i in range(50):
        # Pre-populate caches
        await call_search_memory(
            memory=suite.memory_system,
            arguments={"query": "test", "limit": 5},
            session_id=f"cleanup_{i}"
        )

        start = time.perf_counter()
        await call_record_response(
            memory=suite.memory_system,
            arguments={"key_takeaway": "Test", "outcome": "worked"},
            session_id=f"cleanup_{i}"
        )
        end = time.perf_counter()
        cleanup_latencies.append((end - start) * 1000)

        # Verify cache cleared
        cache = get_search_cache(f"cleanup_{i}")
        assert cache is None or len(cache.get("doc_ids", [])) == 0

    print(f"\n[RESULTS - Cache Overhead]")
    print(f"  Search (with caching): p50={statistics.median(caching_latencies):.2f}ms")
    print(f"  Record (with cleanup): p50={statistics.median(cleanup_latencies):.2f}ms")

    await suite.cleanup()

    return {
        "search_with_cache_p50": statistics.median(caching_latencies),
        "record_with_cleanup_p50": statistics.median(cleanup_latencies)
    }


# =============================================================================
# Benchmark 5: Scaling Test (Memory Count Impact)
# =============================================================================

@pytest.mark.asyncio
async def test_scaling_by_memory_count():
    """
    Benchmark: How latency scales with memory count

    Tests search latency at different memory counts:
    - 10, 50, 100, 200 memories
    """
    from tests.mcp.mcp_tool_harness import call_search_memory, clear_all_caches

    print("\n" + "=" * 80)
    print("MCP TOOL SCALING BENCHMARK")
    print("=" * 80)

    test_sizes = [10, 50, 100, 200]
    results = {}

    for size in test_sizes:
        suite = MCPBenchmarkSuite(f"scale_{size}")
        await suite.initialize()

        # Seed memories
        print(f"\n[{size} memories]")
        print(f"  Seeding...", end=" ", flush=True)
        for i in range(size):
            await suite.memory_system.store_memory_bank(
                text=f"Memory {i}: Topic {i % 10} details",
                tags=[f"topic_{i % 10}"]
            )
        print("OK")

        # Run searches
        print(f"  Benchmarking...", end=" ", flush=True)
        latencies = []
        for i in range(50):
            clear_all_caches()
            query = f"topic {i % 10}"

            start = time.perf_counter()
            await call_search_memory(
                memory=suite.memory_system,
                arguments={"query": query, "limit": 5},
                session_id=f"scale_{size}_{i}"
            )
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

        results[size] = {
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "avg": statistics.mean(latencies)
        }

        print(f"p50={results[size]['p50']:.1f}ms, p95={results[size]['p95']:.1f}ms")

        await suite.cleanup()

    # Summary table
    print("\n" + "=" * 80)
    print("SCALING SUMMARY")
    print("=" * 80)
    print(f"{'Memories':<12} {'p50 (ms)':<12} {'p95 (ms)':<12} {'avg (ms)':<12}")
    print("-" * 80)

    for size in test_sizes:
        r = results[size]
        print(f"{size:<12} {r['p50']:<12.2f} {r['p95']:<12.2f} {r['avg']:<12.2f}")

    return results


# =============================================================================
# Main Runner
# =============================================================================

async def run_all_benchmarks():
    """Run all MCP benchmarks and print summary."""
    print("=" * 80)
    print("ROAMPAL MCP TOOL BENCHMARKS")
    print("=" * 80)
    print()

    results = {}

    print("Running search_memory benchmark...")
    results["search"] = await test_search_memory_latency()
    print()

    print("Running add_to_memory_bank benchmark...")
    results["add"] = await test_add_memory_latency()
    print()

    print("Running full workflow benchmark...")
    results["workflow"] = await test_full_workflow_latency()
    print()

    print("Running cache overhead benchmark...")
    results["cache"] = await test_cache_overhead()
    print()

    print("Running scaling benchmark...")
    results["scaling"] = await test_scaling_by_memory_count()
    print()

    # Final summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"\nsearch_memory p95: {results['search']['p95']:.2f}ms")
    print(f"add_to_memory_bank p95: {results['add']['p95']:.2f}ms")
    print(f"Full workflow p95: {results['workflow']['workflow_p95']:.2f}ms")
    print(f"\nNote: Results are hardware-dependent.")
    print("=" * 80)

    return results


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
