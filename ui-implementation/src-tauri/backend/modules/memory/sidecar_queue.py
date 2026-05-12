"""
Sidecar Queue Manager — Ensures sidecar always runs with retry logic.

v0.3.1.3: Persistent queuing for failed sidecar tasks with exponential backoff.
Single-user desktop app: Simple global queue with retry.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

# Global queue for failed sidecar tasks
_sidecar_retry_queue: List[Dict[str, Any]] = []
# Track busy clients to avoid concurrent model loading
_busy_clients = set()
_client_locks = {}  # per-client locks for sequential execution

# v0.3.3 Defect 14: callbacks for terminal-failure notification + retry success.
# The retry queue is decoupled from app.state, so the agent_chat side registers
# callbacks at startup that mutate `app.state.sidecar_last_error`. This keeps
# the queue module free of HTTP/Tauri concerns while letting the UI's
# sidecar-status indicator reflect the queue's actual outcome (NOT the first
# transient failure that the queue is designed to absorb).
_terminal_failure_callback: Optional[Callable[[Dict[str, Any]], None]] = None
_retry_success_callback: Optional[Callable[[Dict[str, Any]], None]] = None


def register_terminal_failure_callback(cb: Callable[[Dict[str, Any]], None]) -> None:
    """Set the callback fired when a retry task exhausts its budget.

    Callback receives the dead-letter item: `{task_type, doc_id, retry_count,
    last_error, ...}`. Use it to surface a user-visible "sidecar dropped a
    score" indicator. NOT fired on transient single-attempt failures.
    """
    global _terminal_failure_callback
    _terminal_failure_callback = cb


def register_retry_success_callback(cb: Callable[[Dict[str, Any]], None]) -> None:
    """Set the callback fired when a previously-queued task succeeds on retry.

    Callback receives the successful item. Use it to clear a previously-set
    failure indicator now that the queue self-healed.
    """
    global _retry_success_callback
    _retry_success_callback = cb


def get_client_lock(client_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific client."""
    if client_id not in _client_locks:
        _client_locks[client_id] = asyncio.Lock()
    return _client_locks[client_id]


def get_client_id(client: Any) -> str:
    """Generate a unique ID for a client."""
    if hasattr(client, "model_name") and hasattr(client, "_client"):
        base_url = str(client._client.base_url) if client._client else "unknown"
        return f"{client.model_name}@{base_url}"
    return str(id(client))


async def execute_with_client_lock(
    client: Any,
    task_func: Callable[[], Awaitable[Any]],
    task_name: str = "sidecar_task",
) -> Any:
    """
    Execute a task with client locking to prevent concurrent model loading.

    Args:
        client: OllamaClient instance
        task_func: Async function to execute
        task_name: Name for logging

    Returns:
        Task result or None if failed
    """
    client_id = get_client_id(client)
    lock = get_client_lock(client_id)

    try:
        async with lock:
            logger.debug(f"[SIDECAR] Acquired lock for {client_id} - {task_name}")
            return await task_func()
    except Exception as e:
        logger.error(f"[SIDECAR] Task failed {task_name}: {e}")
        return None


def queue_sidecar_retry(
    task_data: Dict[str, Any], error: str, max_retries: int = 3
) -> None:
    """
    Queue a failed sidecar task for retry with exponential backoff.

    Args:
        task_data: Original task data
        error: Error message
        max_retries: Maximum number of retry attempts
    """
    # Check if similar task already queued
    for item in _sidecar_retry_queue:
        if item.get("task_type") == task_data.get("task_type") and item.get(
            "doc_id"
        ) == task_data.get("doc_id"):
            logger.debug(f"[SIDECAR] Task already queued: {task_data.get('doc_id')}")
            return

    retry_item = {
        "task": task_data,
        "task_type": task_data.get("task_type", "unknown"),
        "doc_id": task_data.get("doc_id", "unknown"),
        "retry_count": 0,
        "max_retries": max_retries,
        "next_retry": time.time() + 60,  # 1 minute initial delay
        "last_error": error,
        "created_at": time.time(),
    }

    _sidecar_retry_queue.append(retry_item)
    logger.warning(
        f"[SIDECAR] Queued for retry: {task_data.get('doc_id', 'unknown')} - {error}"
    )


async def process_retry_queue() -> None:
    """Background task to process retry queue."""
    while True:
        await asyncio.sleep(30)  # Check every 30 seconds

        now = time.time()
        processed = []

        for item in list(_sidecar_retry_queue):
            if item["next_retry"] <= now and item["retry_count"] < item["max_retries"]:
                try:
                    logger.info(
                        f"[SIDECAR] Retrying {item['task_type']} for {item['doc_id']} (attempt {item['retry_count'] + 1})"
                    )

                    # Import here to avoid circular imports
                    from .sidecar_service import (
                        score_exchange,
                        extract_facts,
                        extract_noun_tags,
                        summarize_only,
                    )

                    task = {k: v for k, v in item["task"].items() if k not in ("task_type", "doc_id")}
                    task_type = item["task_type"]

                    if task_type == "score_exchange":
                        result = await score_exchange(**task)
                    elif task_type == "extract_facts":
                        result = await extract_facts(**task)
                    elif task_type == "extract_noun_tags":
                        result = await extract_noun_tags(**task)
                    elif task_type == "summarize_only":
                        result = await summarize_only(**task)
                    else:
                        logger.error(f"[SIDECAR] Unknown task type: {task_type}")
                        processed.append(item)
                        continue

                    if result is not None:
                        logger.info(f"[SIDECAR] Retry successful for {item['doc_id']}")
                        # v0.3.3 Defect 14: notify the status surface that the
                        # previously-queued task self-healed — clears any stale
                        # last-error message on app.state.
                        if _retry_success_callback is not None:
                            try:
                                _retry_success_callback(item)
                            except Exception as cb_err:
                                logger.warning(
                                    f"[SIDECAR] retry_success_callback raised (non-fatal): {cb_err}"
                                )
                        processed.append(item)
                    else:
                        # Task failed again
                        item["retry_count"] += 1
                        item["next_retry"] = now + (
                            60 * (2 ** item["retry_count"])
                        )  # Exponential
                        item["last_error"] = "Task returned None"

                except Exception as e:
                    item["retry_count"] += 1
                    item["next_retry"] = now + (
                        60 * (2 ** item["retry_count"])
                    )  # Exponential
                    item["last_error"] = str(e)
                    logger.error(f"[SIDECAR] Retry failed for {item['doc_id']}: {e}")

            elif item["retry_count"] >= item["max_retries"]:
                logger.error(f"[SIDECAR] Max retries exceeded for {item['doc_id']}")
                # v0.3.3 Defect 14: terminal failure — queue exhausted its
                # budget and is dropping this score. Notify the status surface
                # so the UI's sidecar-status indicator can flip red. This is
                # the only path that should alarm the user; transient single-
                # attempt failures get absorbed silently by the retry queue.
                if _terminal_failure_callback is not None:
                    try:
                        _terminal_failure_callback(item)
                    except Exception as cb_err:
                        logger.warning(
                            f"[SIDECAR] terminal_failure_callback raised (non-fatal): {cb_err}"
                        )
                processed.append(item)

        # Remove processed items
        for item in processed:
            if item in _sidecar_retry_queue:
                _sidecar_retry_queue.remove(item)


def get_queue_stats() -> Dict[str, Any]:
    """Get statistics about the retry queue."""
    return {
        "total_queued": len(_sidecar_retry_queue),
        "pending_retries": sum(
            1
            for item in _sidecar_retry_queue
            if item["retry_count"] < item["max_retries"]
        ),
        "failed_permanently": sum(
            1
            for item in _sidecar_retry_queue
            if item["retry_count"] >= item["max_retries"]
        ),
        "oldest_item": min(
            (item["created_at"] for item in _sidecar_retry_queue), default=0
        ),
        "items": [
            {
                "task_type": item["task_type"],
                "doc_id": item["doc_id"],
                "retry_count": item["retry_count"],
                "max_retries": item["max_retries"],
                "next_retry_in": max(0, item["next_retry"] - time.time()),
                "last_error": item["last_error"][:100] if item["last_error"] else None,
            }
            for item in _sidecar_retry_queue[:10]  # First 10 items
        ],
    }
