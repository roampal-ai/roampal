"""
Benchmark: Mem0 v1.0.0 vs Roampal - HARD MODE Semantic Confusion Test
Tests both systems against 4:1 noise ratio (5 correct Python docs, 20 wrong language docs)
"""
import asyncio
import time
import os
from datetime import datetime

# Mem0 v1.0.0
from mem0 import Memory

# Roampal
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ui-implementation', 'src-tauri', 'backend'))
from modules.memory.unified_memory_system import UnifiedMemorySystem


# Test data: 5 Python docs (CORRECT) + 20 other language docs (NOISE)
PYTHON_DOCS = [
    "Python async/await tutorial: Use async def to define coroutines, await to call them",
    "Python asyncio event loop: asyncio.run() starts the event loop and runs coroutines",
    "Python async context managers: Use async with for database connections and files",
    "Python async iterators: Use async for to iterate over async generators",
    "Python asyncio tasks: asyncio.create_task() schedules coroutines concurrently"
]

NOISE_DOCS = [
    # JavaScript (5 docs)
    "JavaScript async/await: Use async function and await keyword for promises",
    "JavaScript Promise.all(): Run multiple async operations concurrently",
    "JavaScript event loop: Microtasks and macrotasks execution order",
    "Node.js async patterns: Callbacks, promises, and async/await evolution",
    "JavaScript async generators: for await...of loops for async iteration",

    # Rust (5 docs)
    "Rust async/.await: Use async fn and .await for futures in Tokio runtime",
    "Rust tokio runtime: Multi-threaded async executor for Rust applications",
    "Rust async traits: Define async methods in traits with async-trait crate",
    "Rust futures: poll() method and Pin<Box<dyn Future>> for async operations",
    "Rust async streams: Stream trait for async iteration patterns",

    # Go (5 docs)
    "Go goroutines: Lightweight concurrent functions with go keyword",
    "Go channels: Communication between goroutines with buffered channels",
    "Go select statement: Multiplexing channel operations concurrently",
    "Go context package: Cancellation and timeout for goroutine coordination",
    "Go WaitGroup: Synchronization primitive for waiting on goroutines",

    # Ruby (5 docs)
    "Ruby async gems: async, concurrent-ruby for asynchronous programming",
    "Ruby Fiber: Lightweight concurrency with manual scheduling",
    "Ruby EventMachine: Event-driven I/O for concurrent connections",
    "Ruby async-await syntax: Using fibers and promises for async code",
    "Ruby concurrent-ruby: Thread pools and futures for parallelism"
]

QUERY = "python async programming"


async def test_mem0_v1():
    """Test Mem0 v1.0.0 with HARD MODE semantic confusion"""
    print("\n" + "="*60)
    print("TESTING MEM0 V1.0.0")
    print("="*60)

    # Initialize Mem0
    memory = Memory()

    # Store all documents (5 Python + 20 noise)
    print("\nStoring documents...")
    user_id = "test_user"

    for i, doc in enumerate(PYTHON_DOCS):
        memory.add(doc, user_id=user_id, metadata={"type": "python", "index": i})

    for i, doc in enumerate(NOISE_DOCS):
        lang = "javascript" if i < 5 else "rust" if i < 10 else "go" if i < 15 else "ruby"
        memory.add(doc, user_id=user_id, metadata={"type": lang, "index": i})

    print(f"Stored 25 documents (5 Python, 20 noise)")

    # Run 20 queries and measure precision
    print(f"\nRunning 20 queries: '{QUERY}'")

    results = []
    total_latency = 0

    for query_num in range(1, 21):
        start = time.time()

        # Search with Mem0 v1.0.0
        search_results = memory.search(QUERY, user_id=user_id, limit=5)

        latency = time.time() - start
        total_latency += latency

        # Calculate precision: how many results are Python docs?
        python_count = sum(1 for r in search_results.get('results', [])
                          if 'Python' in r.get('memory', ''))
        precision = python_count / 5.0

        results.append({
            'query': query_num,
            'precision': precision,
            'python_count': python_count,
            'latency_ms': latency * 1000
        })

        if query_num <= 3 or query_num == 20:
            print(f"  Query {query_num}: {python_count}/5 Python docs = {precision*100:.1f}% precision ({latency*1000:.1f}ms)")

    # Calculate stats
    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_latency = total_latency / len(results) * 1000

    print(f"\n" + "-"*60)
    print(f"MEM0 V1.0.0 RESULTS:")
    print(f"  Average Precision: {avg_precision*100:.1f}%")
    print(f"  Average Latency: {avg_latency:.1f}ms")
    print(f"  First Query Precision: {results[0]['precision']*100:.1f}%")
    print(f"  Last Query Precision: {results[-1]['precision']*100:.1f}%")
    print("-"*60)

    return {
        'system': 'Mem0 v1.0.0',
        'avg_precision': avg_precision,
        'avg_latency_ms': avg_latency,
        'first_precision': results[0]['precision'],
        'last_precision': results[-1]['precision']
    }


async def test_roampal():
    """Test Roampal with HARD MODE semantic confusion"""
    print("\n" + "="*60)
    print("TESTING ROAMPAL V0.2.0")
    print("="*60)

    # Initialize Roampal
    memory = UnifiedMemorySystem(data_dir="./benchmarks/.benchmarks/mem0_comparison")

    # Store Python docs in memory_bank (correct collection)
    print("\nStoring documents...")
    for doc in PYTHON_DOCS:
        await memory.store_memory_bank(
            text=doc,
            tags=["python", "async"],
            importance=0.9,
            confidence=0.9
        )

    # Store noise docs in working/history
    for i, doc in enumerate(NOISE_DOCS):
        if i < 10:
            await memory.store(doc, collection="working", metadata={"role": "assistant"})
        else:
            await memory.store(doc, collection="history", metadata={"role": "assistant", "score": 0.6})

    print(f"Stored 25 documents (5 Python in memory_bank, 20 noise in working/history)")

    # Run 20 queries and measure precision
    print(f"\nRunning 20 queries: '{QUERY}'")

    results = []
    total_latency = 0

    for query_num in range(1, 21):
        start = time.time()

        # Search with Roampal (all collections)
        search_results = await memory.search(QUERY, collections=["all"], limit=5)

        latency = time.time() - start
        total_latency += latency

        # Calculate precision: how many results are from memory_bank (Python)?
        python_count = sum(1 for r in search_results
                          if r.get('collection') == 'memory_bank')
        precision = python_count / 5.0

        results.append({
            'query': query_num,
            'precision': precision,
            'python_count': python_count,
            'latency_ms': latency * 1000
        })

        if query_num <= 3 or query_num == 20:
            print(f"  Query {query_num}: {python_count}/5 from memory_bank = {precision*100:.1f}% precision ({latency*1000:.1f}ms)")

    # Calculate stats
    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_latency = total_latency / len(results) * 1000

    print(f"\n" + "-"*60)
    print(f"ROAMPAL V0.2.0 RESULTS:")
    print(f"  Average Precision: {avg_precision*100:.1f}%")
    print(f"  Average Latency: {avg_latency:.1f}ms")
    print(f"  First Query Precision: {results[0]['precision']*100:.1f}%")
    print(f"  Last Query Precision: {results[-1]['precision']*100:.1f}%")
    print("-"*60)

    return {
        'system': 'Roampal v0.2.0',
        'avg_precision': avg_precision,
        'avg_latency_ms': avg_latency,
        'first_precision': results[0]['precision'],
        'last_precision': results[-1]['precision']
    }


async def main():
    """Run comparison benchmark"""
    print("\n" + "="*60)
    print("MEM0 V1.0.0 VS ROAMPAL V0.2.0")
    print("HARD MODE: 4:1 Noise Ratio Semantic Confusion Test")
    print("="*60)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mem0 Version: 1.0.0")
    print(f"Roampal Version: 0.2.0")

    # Run both tests
    mem0_results = await test_mem0_v1()
    roampal_results = await test_roampal()

    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\nPrecision (higher is better):")
    print(f"  Mem0 v1.0.0:    {mem0_results['avg_precision']*100:.1f}%")
    print(f"  Roampal v0.2.0: {roampal_results['avg_precision']*100:.1f}%")
    print(f"  Winner: {'Roampal' if roampal_results['avg_precision'] > mem0_results['avg_precision'] else 'Mem0'} by {abs(roampal_results['avg_precision'] - mem0_results['avg_precision'])*100:.1f}%")

    print(f"\nLatency (lower is better):")
    print(f"  Mem0 v1.0.0:    {mem0_results['avg_latency_ms']:.1f}ms")
    print(f"  Roampal v0.2.0: {roampal_results['avg_latency_ms']:.1f}ms")
    print(f"  Winner: {'Roampal' if roampal_results['avg_latency_ms'] < mem0_results['avg_latency_ms'] else 'Mem0'} ({roampal_results['avg_latency_ms']/mem0_results['avg_latency_ms']:.1f}x faster)" if roampal_results['avg_latency_ms'] < mem0_results['avg_latency_ms'] else f"  Winner: Mem0 ({mem0_results['avg_latency_ms']/roampal_results['avg_latency_ms']:.1f}x faster)")

    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(main())
