"""
Memory Bank API Router
Provides user control over memory_bank collection (5th collection)

memory_bank is the persistent identity layer enabling continuity and growth:
1. User Context - Who they are, what they want (identity, preferences, goals, projects)
2. System Mastery - How to be effective (tool strategies, search patterns, what works/fails)
3. Agent Growth - Self-improvement (mistakes learned, relationship dynamics, progress tracking)

LLM has full autonomy, user has override via Settings UI
"""

from fastapi import APIRouter, HTTPException, Request
from typing import List, Optional
from pydantic import BaseModel
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory-bank", tags=["memory-bank"])


class MemoryCreate(BaseModel):
    text: str
    tags: List[str]
    importance: float = 0.7
    confidence: float = 0.7


class MemoryUpdate(BaseModel):
    text: str
    tags: Optional[List[str]] = None
    importance: Optional[float] = None


@router.get("/list")
async def list_memories(
    request: Request,
    include_archived: bool = False,
    tags: Optional[str] = None,
    limit: int = 1000  # v0.2.5: increased from 50 to match storage capacity
):
    """
    List all memories from memory_bank collection.

    Args:
        include_archived: Include archived memories
        tags: Comma-separated tags to filter by
        limit: Max number of results

    Returns:
        List of memories with metadata
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        tag_list = tags.split(",") if tags else None

        # Search memory_bank collection
        results = await memory.search_memory_bank(
            query=None,  # Get all
            tags=tag_list,
            include_archived=include_archived,
            limit=limit
        )

        # Format results for UI
        formatted = []
        for r in results:
            metadata = r.get("metadata", {})
            formatted.append({
                "id": r.get("id"),
                "text": r.get("content", ""),
                "tags": json.loads(metadata.get("tags", "[]")),
                # Note: importance/confidence removed - memory_bank is authoritative facts, not scored
                "status": metadata.get("status", "active"),
                "created_at": metadata.get("created_at"),
                "updated_at": metadata.get("updated_at"),
                "last_mentioned": metadata.get("last_mentioned"),
                "mentioned_count": metadata.get("mentioned_count", 1),
                "archive_reason": metadata.get("archive_reason"),
                "archived_at": metadata.get("archived_at")
            })

        return {
            "memories": formatted,
            "total": len(formatted)
        }

    except Exception as e:
        logger.error(f"Error listing memories: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to list memories: {str(e)}")


@router.get("/archived")
async def get_archived_memories(
    request: Request,
    limit: int = 1000  # v0.2.5: increased from 50 to match storage capacity
):
    """
    Get only archived memories.

    Args:
        limit: Max number of results

    Returns:
        List of archived memories
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        results = await memory.search_memory_bank(
            query=None,
            tags=None,
            include_archived=True,
            limit=limit * 2
        )

        # Filter to only archived
        archived = []
        for r in results:
            metadata = r.get("metadata", {})
            if metadata.get("status") == "archived":
                archived.append({
                    "id": r.get("id"),
                    "text": r.get("content", ""),
                    "tags": json.loads(metadata.get("tags", "[]")),
                    "archived_at": metadata.get("archived_at"),
                    "archive_reason": metadata.get("archive_reason"),
                    "original_id": metadata.get("original_id")
                })

        return {
            "memories": archived[:limit],
            "total": len(archived)
        }

    except Exception as e:
        logger.error(f"Error getting archived memories: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to get archived memories: {str(e)}")


@router.post("/restore/{doc_id}")
async def restore_memory(
    request: Request,
    doc_id: str
):
    """
    User manually restores archived memory.

    Args:
        doc_id: Memory ID to restore

    Returns:
        Success status
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        success = await memory.user_restore_memory(doc_id)
        if not success:
            raise HTTPException(404, "Memory not found")

        return {
            "status": "restored",
            "doc_id": doc_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring memory: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to restore memory: {str(e)}")


@router.delete("/delete/{doc_id}")
async def delete_memory(
    request: Request,
    doc_id: str
):
    """
    User permanently deletes memory (hard delete).

    Args:
        doc_id: Memory ID to delete

    Returns:
        Success status
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        success = await memory.user_delete_memory(doc_id)
        if not success:
            raise HTTPException(404, "Memory not found")

        return {
            "status": "deleted",
            "doc_id": doc_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to delete memory: {str(e)}")


@router.get("/search")
async def search_memories(
    request: Request,
    q: str,
    tags: Optional[str] = None,
    limit: int = 20
):
    """
    Semantic search across memory_bank.

    Args:
        q: Search query
        tags: Optional comma-separated tags filter
        limit: Max results

    Returns:
        Relevant memories ranked by similarity
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        tag_list = tags.split(",") if tags else None

        results = await memory.search_memory_bank(
            query=q,
            tags=tag_list,
            include_archived=False,
            limit=limit
        )

        # Format results
        formatted = []
        for r in results:
            metadata = r.get("metadata", {})
            formatted.append({
                "id": r.get("id"),
                "text": r.get("content", ""),
                "tags": json.loads(metadata.get("tags", "[]")),
                "relevance": 1.0 / (1.0 + r.get("distance", 0.5))  # Convert distance to relevance
                # Note: confidence removed - memory_bank is authoritative facts, not probabilistic
            })

        return {
            "query": q,
            "results": formatted,
            "total": len(formatted)
        }

    except Exception as e:
        logger.error(f"Error searching memories: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to search memories: {str(e)}")


@router.get("/stats")
async def get_memory_stats(request: Request):
    """
    Get memory_bank statistics.

    Returns:
        Stats about memory_bank collection
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        # Get all memories
        all_memories = await memory.search_memory_bank(
            query=None,
            include_archived=True,
            limit=1000
        )

        # Count by status
        active = sum(1 for m in all_memories if m.get("metadata", {}).get("status") == "active")
        archived = sum(1 for m in all_memories if m.get("metadata", {}).get("status") == "archived")

        # Collect all tags
        all_tags = set()
        for m in all_memories:
            tags = json.loads(m.get("metadata", {}).get("tags", "[]"))
            all_tags.update(tags)

        return {
            "total_memories": len(all_memories),
            "active": active,
            "archived": archived,
            "unique_tags": len(all_tags),
            "tags": sorted(list(all_tags))
        }

    except Exception as e:
        logger.error(f"Error getting memory stats: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to get memory stats: {str(e)}")


@router.post("/create")
async def create_memory(
    request: Request,
    memory_data: MemoryCreate
):
    """
    Create a new memory in memory_bank.
    Used by MCP bridge and external tools.

    Args:
        memory_data: Memory content, tags, importance, confidence

    Returns:
        Created memory ID
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        doc_id = await memory.store_memory_bank(
            text=memory_data.text,
            tags=memory_data.tags,
            importance=memory_data.importance,
            confidence=memory_data.confidence
        )

        return {
            "status": "created",
            "doc_id": doc_id,
            "text": memory_data.text
        }

    except ValueError as e:
        # Capacity errors
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Error creating memory: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to create memory: {str(e)}")


@router.put("/update/{doc_id}")
async def update_memory(
    request: Request,
    doc_id: str,
    memory_data: MemoryUpdate
):
    """
    Update an existing memory in memory_bank.
    Old version is automatically archived.
    Used by MCP bridge and external tools.

    Args:
        doc_id: Memory ID to update
        memory_data: New text and optional metadata

    Returns:
        Updated memory ID
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        updated_id = await memory.update_memory_bank(
            doc_id=doc_id,
            new_text=memory_data.text,
            reason="api_update"
        )

        # Update tags/importance if provided
        if memory_data.tags is not None or memory_data.importance is not None:
            doc = memory.collections["memory_bank"].get_fragment(updated_id)
            if doc:
                metadata = doc.get("metadata", {})
                if memory_data.tags is not None:
                    metadata["tags"] = json.dumps(memory_data.tags)
                if memory_data.importance is not None:
                    metadata["importance"] = memory_data.importance
                memory.collections["memory_bank"].update_fragment_metadata(updated_id, metadata)

        return {
            "status": "updated",
            "doc_id": updated_id,
            "text": memory_data.text
        }

    except Exception as e:
        logger.error(f"Error updating memory: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to update memory: {str(e)}")


@router.post("/archive/{doc_id}")
async def archive_memory(
    request: Request,
    doc_id: str
):
    """
    Archive a memory (soft delete).
    Used by MCP bridge and external tools.

    Args:
        doc_id: Memory ID to archive

    Returns:
        Success status
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        success = await memory.archive_memory_bank(
            doc_id=doc_id,
            reason="api_archive"
        )

        if not success:
            raise HTTPException(404, "Memory not found")

        return {
            "status": "archived",
            "doc_id": doc_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error archiving memory: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to archive memory: {str(e)}")
