"""
Tests for sidecar_queue.py — v0.3.1 client locking and retry queue.
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from modules.memory.sidecar_queue import (
    get_client_id,
    get_client_lock,
    execute_with_client_lock,
    queue_sidecar_retry,
    get_queue_stats,
    _sidecar_retry_queue,
    _client_locks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear global queue and locks between tests."""
    _sidecar_retry_queue.clear()
    _client_locks.clear()
    yield
    _sidecar_retry_queue.clear()
    _client_locks.clear()


def _make_client(model="qwen2.5:3b", base_url="http://localhost:11434"):
    client = MagicMock()
    client.model_name = model
    client._client = MagicMock()
    client._client.base_url = base_url
    return client


# ---------------------------------------------------------------------------
# get_client_id
# ---------------------------------------------------------------------------

class TestGetClientId:
    def test_with_model_and_client(self):
        client = _make_client()
        cid = get_client_id(client)
        assert "qwen2.5:3b" in cid
        assert "localhost:11434" in cid

    def test_without_model_name(self):
        client = MagicMock(spec=[])  # no attributes
        cid = get_client_id(client)
        assert cid  # falls back to id()

    def test_different_models_different_ids(self):
        c1 = _make_client(model="qwen2.5:3b")
        c2 = _make_client(model="llama3.2:3b")
        assert get_client_id(c1) != get_client_id(c2)

    def test_same_model_same_id(self):
        c1 = _make_client()
        c2 = _make_client()
        assert get_client_id(c1) == get_client_id(c2)


# ---------------------------------------------------------------------------
# get_client_lock
# ---------------------------------------------------------------------------

class TestGetClientLock:
    def test_creates_lock(self):
        lock = get_client_lock("client_1")
        assert isinstance(lock, asyncio.Lock)

    def test_returns_same_lock(self):
        lock1 = get_client_lock("client_1")
        lock2 = get_client_lock("client_1")
        assert lock1 is lock2

    def test_different_clients_different_locks(self):
        lock1 = get_client_lock("client_1")
        lock2 = get_client_lock("client_2")
        assert lock1 is not lock2


# ---------------------------------------------------------------------------
# execute_with_client_lock
# ---------------------------------------------------------------------------

class TestExecuteWithClientLock:
    @pytest.mark.asyncio
    async def test_success(self):
        client = _make_client()
        result = await execute_with_client_lock(
            client=client,
            task_func=AsyncMock(return_value={"summary": "test"}),
            task_name="test_task",
        )
        assert result == {"summary": "test"}

    @pytest.mark.asyncio
    async def test_failure_returns_none(self):
        client = _make_client()
        result = await execute_with_client_lock(
            client=client,
            task_func=AsyncMock(side_effect=Exception("boom")),
            task_name="test_task",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Verify tasks for same client run sequentially, not concurrently."""
        client = _make_client()
        execution_order = []

        async def task_a():
            execution_order.append("a_start")
            await asyncio.sleep(0.05)
            execution_order.append("a_end")
            return "a"

        async def task_b():
            execution_order.append("b_start")
            await asyncio.sleep(0.01)
            execution_order.append("b_end")
            return "b"

        # Run both concurrently — lock should serialize them
        results = await asyncio.gather(
            execute_with_client_lock(client, task_a, "task_a"),
            execute_with_client_lock(client, task_b, "task_b"),
        )

        # One must finish before the other starts
        a_start = execution_order.index("a_start")
        a_end = execution_order.index("a_end")
        b_start = execution_order.index("b_start")
        b_end = execution_order.index("b_end")

        # Either A finishes before B starts, or B finishes before A starts
        assert (a_end < b_start) or (b_end < a_start)


# ---------------------------------------------------------------------------
# queue_sidecar_retry
# ---------------------------------------------------------------------------

class TestQueueSidecarRetry:
    def test_adds_to_queue(self):
        task_data = {"task_type": "score_exchange", "doc_id": "doc_1"}
        queue_sidecar_retry(task_data, "Test error")
        assert len(_sidecar_retry_queue) == 1
        item = _sidecar_retry_queue[0]
        assert item["task_type"] == "score_exchange"
        assert item["doc_id"] == "doc_1"
        assert item["retry_count"] == 0
        assert item["max_retries"] == 3

    def test_deduplicates_same_task(self):
        task_data = {"task_type": "score_exchange", "doc_id": "doc_1"}
        queue_sidecar_retry(task_data, "Error 1")
        queue_sidecar_retry(task_data, "Error 2")
        assert len(_sidecar_retry_queue) == 1

    def test_different_doc_ids_not_deduped(self):
        queue_sidecar_retry({"task_type": "score_exchange", "doc_id": "doc_1"}, "err")
        queue_sidecar_retry({"task_type": "score_exchange", "doc_id": "doc_2"}, "err")
        assert len(_sidecar_retry_queue) == 2

    def test_initial_retry_delay_60s(self):
        now = time.time()
        queue_sidecar_retry({"task_type": "test", "doc_id": "d"}, "err")
        item = _sidecar_retry_queue[0]
        assert item["next_retry"] >= now + 59  # allow 1s tolerance

    def test_custom_max_retries(self):
        queue_sidecar_retry({"task_type": "test", "doc_id": "d"}, "err", max_retries=5)
        assert _sidecar_retry_queue[0]["max_retries"] == 5


# ---------------------------------------------------------------------------
# get_queue_stats
# ---------------------------------------------------------------------------

class TestGetQueueStats:
    def test_empty_queue(self):
        stats = get_queue_stats()
        assert stats["total_queued"] == 0
        assert stats["pending_retries"] == 0
        assert stats["failed_permanently"] == 0

    def test_with_items(self):
        queue_sidecar_retry({"task_type": "score_exchange", "doc_id": "d1"}, "err")
        queue_sidecar_retry({"task_type": "extract_facts", "doc_id": "d2"}, "err")
        stats = get_queue_stats()
        assert stats["total_queued"] == 2
        assert stats["pending_retries"] == 2
        assert len(stats["items"]) == 2

    def test_permanently_failed(self):
        queue_sidecar_retry({"task_type": "test", "doc_id": "d1"}, "err")
        _sidecar_retry_queue[0]["retry_count"] = 3  # exhausted
        stats = get_queue_stats()
        assert stats["failed_permanently"] == 1
        assert stats["pending_retries"] == 0
