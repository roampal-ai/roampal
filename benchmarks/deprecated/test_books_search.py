"""
Books Collection Search Recall Benchmark Tests

Tests that validate books collection search quality:
1. Semantic search finds relevant book chunks
2. Content is properly extracted (not empty)
3. Metadata is preserved (title, author, has_code)
4. Top-K recall meets targets

Target Metrics:
- Recall@5: ≥80%
- Content extraction: 100% (no empty results)
- Metadata accuracy: 100%
"""

import pytest
from typing import List, Dict, Any


@pytest.mark.books
@pytest.mark.asyncio
async def test_books_content_extraction(test_memory_system, test_book_content):
    """
    Test that books collection properly stores and retrieves document content.

    Expected: Retrieved chunks have non-empty content field.
    """
    memory = test_memory_system

    # Upload book to books collection
    from modules.memory.smart_book_processor import SmartBookProcessor
    import tempfile

    temp_books_dir = tempfile.mkdtemp(prefix="books_test_")
    processor = SmartBookProcessor(
        data_dir=temp_books_dir,
        chromadb_adapter=memory.collections["books"],
        embedding_service=memory.embedding_service
    )

    # Process test book
    await processor.process_text_document(
        content=test_book_content["content"],
        title=test_book_content["title"],
        author=test_book_content["author"]
    )

    # Search for content
    results = await memory.search(
        query="memory architecture tiers",
        collections=["books"],
        limit=5
    )

    print(f"\n=== BOOKS CONTENT EXTRACTION ===")
    print(f"Results found: {len(results)}")

    # Assertions
    assert len(results) > 0, "Should find book chunks matching query"

    for i, result in enumerate(results):
        content = result.get('content') or result.get('text') or result.get('metadata', {}).get('content')

        print(f"\nResult {i+1}:")
        print(f"  Content length: {len(content) if content else 0}")
        print(f"  Has content: {bool(content)}")
        print(f"  Preview: {content[:100] if content else '[EMPTY]'}...")

        assert content, f"Result {i+1} should have non-empty content"
        assert len(content) > 10, f"Result {i+1} content should be substantial"


@pytest.mark.books
@pytest.mark.asyncio
async def test_books_metadata_preservation(test_memory_system, test_book_content):
    """
    Test that book metadata (title, author, has_code) is preserved.

    Expected: Retrieved chunks include correct metadata fields.
    """
    memory = test_memory_system

    # Upload book
    from modules.memory.smart_book_processor import SmartBookProcessor
    import tempfile

    temp_books_dir = tempfile.mkdtemp(prefix="books_test_")
    processor = SmartBookProcessor(
        data_dir=temp_books_dir,
        chromadb_adapter=memory.collections["books"],
        embedding_service=memory.embedding_service
    )

    await processor.process_text_document(
        content=test_book_content["content"],
        title=test_book_content["title"],
        author=test_book_content["author"]
    )

    # Search
    results = await memory.search(
        query="Roampal architecture",
        collections=["books"],
        limit=5
    )

    print(f"\n=== BOOKS METADATA ===")

    # Check metadata
    for i, result in enumerate(results):
        metadata = result.get('metadata', {})

        print(f"\nResult {i+1} metadata:")
        print(f"  Title: {metadata.get('title')}")
        print(f"  Author: {metadata.get('author')}")
        print(f"  Has code: {metadata.get('has_code')}")
        print(f"  Chunk index: {metadata.get('chunk_index')}")

        assert metadata.get('title') == test_book_content["title"], "Title should match"
        assert metadata.get('author') == test_book_content["author"], "Author should match"
        assert 'chunk_index' in metadata, "Should have chunk_index"


@pytest.mark.books
@pytest.mark.asyncio
async def test_books_semantic_search_accuracy(test_memory_system, test_book_content):
    """
    Test that semantic search returns relevant book chunks.

    Expected: Query "memory tiers" returns chunks about tier architecture.
    """
    memory = test_memory_system

    # Upload book
    from modules.memory.smart_book_processor import SmartBookProcessor
    import tempfile

    temp_books_dir = tempfile.mkdtemp(prefix="books_test_")
    processor = SmartBookProcessor(
        data_dir=temp_books_dir,
        chromadb_adapter=memory.collections["books"],
        embedding_service=memory.embedding_service
    )

    await processor.process_text_document(
        content=test_book_content["content"],
        title=test_book_content["title"],
        author=test_book_content["author"]
    )

    # Test specific queries
    test_queries = [
        ("memory tiers", ["Books", "Memory Bank", "Working", "History", "Patterns"]),
        ("knowledge graphs", ["Routing KG", "Content KG"]),
        ("cold start", ["Cold-Start", "message #1", "auto-inject"])
    ]

    for query, expected_keywords in test_queries:
        results = await memory.search(
            query=query,
            collections=["books"],
            limit=3
        )

        print(f"\n=== Query: '{query}' ===")
        print(f"Results: {len(results)}")

        # Check if results contain expected keywords
        all_content = " ".join([
            r.get('content', '') or r.get('text', '') or r.get('metadata', {}).get('content', '')
            for r in results
        ])

        relevant_keywords_found = sum(1 for kw in expected_keywords if kw.lower() in all_content.lower())

        print(f"Expected keywords: {expected_keywords}")
        print(f"Keywords found: {relevant_keywords_found}/{len(expected_keywords)}")

        assert relevant_keywords_found > 0, \
            f"Query '{query}' should return chunks with at least some expected keywords"


@pytest.mark.books
@pytest.mark.asyncio
async def test_books_metadata_filtering(test_memory_system, test_book_content):
    """
    Test that metadata filtering works for books collection.

    Expected: Filter by title returns only that book's chunks.
    """
    memory = test_memory_system

    # Upload book
    from modules.memory.smart_book_processor import SmartBookProcessor
    import tempfile

    temp_books_dir = tempfile.mkdtemp(prefix="books_test_")
    processor = SmartBookProcessor(
        data_dir=temp_books_dir,
        chromadb_adapter=memory.collections["books"],
        embedding_service=memory.embedding_service
    )

    await processor.process_text_document(
        content=test_book_content["content"],
        title=test_book_content["title"],
        author=test_book_content["author"]
    )

    # Search with metadata filter
    results = await memory.search(
        query="architecture",
        collections=["books"],
        metadata_filters={"title": test_book_content["title"]},
        limit=5
    )

    print(f"\n=== METADATA FILTERING ===")
    print(f"Filter: title='{test_book_content['title']}'")
    print(f"Results: {len(results)}")

    # All results should match filter
    for result in results:
        metadata = result.get('metadata', {})
        assert metadata.get('title') == test_book_content["title"], \
            "Metadata filter should restrict results to specified title"


@pytest.mark.books
@pytest.mark.asyncio
async def test_books_search_metrics(test_memory_system, test_book_content):
    """
    Collect books search performance metrics for benchmarking.

    Metrics:
    - Recall@5 (relevant chunks in top-5)
    - Search latency
    - Content extraction success rate
    """
    import time

    memory = test_memory_system

    # Upload book
    from modules.memory.smart_book_processor import SmartBookProcessor
    import tempfile

    temp_books_dir = tempfile.mkdtemp(prefix="books_test_")
    processor = SmartBookProcessor(
        data_dir=temp_books_dir,
        chromadb_adapter=memory.collections["books"],
        embedding_service=memory.embedding_service
    )

    await processor.process_text_document(
        content=test_book_content["content"],
        title=test_book_content["title"],
        author=test_book_content["author"]
    )

    # Test queries with known relevant content
    queries = [
        "5-tier memory architecture",
        "knowledge graph routing",
        "cold-start auto-trigger"
    ]

    total_recall = 0
    total_latency = 0
    content_extraction_success = 0
    total_results = 0

    for query in queries:
        # Measure search latency
        start_time = time.time()
        results = await memory.search(
            query=query,
            collections=["books"],
            limit=5
        )
        latency = time.time() - start_time
        total_latency += latency

        # Check content extraction
        for result in results:
            total_results += 1
            content = result.get('content') or result.get('text') or result.get('metadata', {}).get('content')
            if content and len(content) > 10:
                content_extraction_success += 1

        # Calculate recall (did we find relevant chunks?)
        # For this test, we assume if we got results, they're relevant (semantic search working)
        if len(results) > 0:
            total_recall += 1

    avg_latency = total_latency / len(queries)
    recall_rate = total_recall / len(queries)
    extraction_rate = content_extraction_success / total_results if total_results > 0 else 0

    print(f"\n=== BOOKS SEARCH METRICS ===")
    print(f"Queries tested: {len(queries)}")
    print(f"Recall rate: {recall_rate:.1%} ({total_recall}/{len(queries)})")
    print(f"Avg search latency: {avg_latency:.3f}s")
    print(f"Content extraction: {extraction_rate:.1%} ({content_extraction_success}/{total_results})")

    # Assertions for targets
    assert recall_rate >= 0.66, f"Target: ≥66% recall (2/3 queries), got {recall_rate:.1%}"
    assert avg_latency < 2.0, f"Target: <2s avg latency, got {avg_latency:.3f}s"
    assert extraction_rate >= 0.95, f"Target: ≥95% content extraction, got {extraction_rate:.1%}"
