"""
Data Management API Router

Provides endpoints for clearing/deleting user data collections.
All operations are destructive and permanent.

v0.3.1: Added memory summarization/migration endpoints.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import json

# Ghost registry for clearing ghost IDs after collection nuke (v0.2.9)
from modules.memory.ghost_registry import get_ghost_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/data", tags=["data-management"])

# v0.3.1: Cancel flag for migration summarization
_cancel_flags: Dict[str, bool] = {}


@router.get("/stats")
async def get_data_stats(request: Request):
    """
    Get counts and stats for all data types.

    Returns:
        {
            "memory_bank": {"count": 14, "active": 11, "archived": 3},
            "working": {"count": 11},
            "history": {"count": 2},
            "patterns": {"count": 0},
            "books": {"count": 221},
            "sessions": {"count": 67},
            "outcomes": {"exists": true}
        }
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    stats = {}

    try:
        # Get ChromaDB collection counts
        for collection_name in [
            "memory_bank",
            "working",
            "history",
            "patterns",
            "books",
        ]:
            if collection_name in memory.collections:
                adapter = memory.collections[collection_name]
                count = await adapter.get_collection_count()

                # Memory bank has active/archived split
                if collection_name == "memory_bank":
                    try:
                        results = adapter.collection.get(
                            where={"status": "active"}, include=[]
                        )
                        active_count = len(results.get("ids", []))
                        archived_count = count - active_count
                        stats[collection_name] = {
                            "count": count,
                            "active": active_count,
                            "archived": archived_count,
                        }
                    except Exception:
                        stats[collection_name] = {"count": count}
                else:
                    stats[collection_name] = {"count": count}

        # Count session files (use memory.data_dir for correct path)
        # v0.3.0: Include archive folder in count
        sessions_dir = memory.data_dir / "sessions"
        if sessions_dir.exists():
            session_files = list(sessions_dir.glob("**/*.jsonl"))
            stats["sessions"] = {"count": len(session_files)}
        else:
            stats["sessions"] = {"count": 0}

        # v0.3.0: Check outcomes.db exists
        outcomes_db = memory.data_dir / "outcomes.db"
        stats["outcomes"] = {"exists": outcomes_db.exists()}

        return stats

    except Exception as e:
        logger.error(f"Error fetching data stats: {e}", exc_info=True)
        logger.error(f"Failed to get stats: {e}"); raise HTTPException(500, "Failed to get stats")


@router.post("/clear/memory_bank")
async def clear_memory_bank(request: Request):
    """Clear all memory_bank entries (active and archived)."""
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        if "memory_bank" not in memory.collections:
            raise HTTPException(404, "Memory bank collection not found")

        adapter = memory.collections["memory_bank"]
        count_before = await adapter.get_collection_count()

        # Delete all documents in collection (preserves schema)
        # ChromaDB requires either getting all IDs first or using where_document
        if count_before > 0:
            # Get all IDs and delete them
            all_docs = adapter.collection.get(include=[])
            if all_docs.get("ids"):
                # Delete in batches to avoid ChromaDB batch size limits (max 166)
                batch_size = 100
                all_ids = all_docs["ids"]
                for i in range(0, len(all_ids), batch_size):
                    batch = all_ids[i : i + batch_size]
                    adapter.collection.delete(ids=batch)

        logger.info(f"Cleared memory_bank collection ({count_before} entries deleted)")

        return {
            "status": "success",
            "collection": "memory_bank",
            "deleted_count": count_before,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing memory_bank: {e}", exc_info=True)
        logger.error(f"Failed to clear memory_bank: {e}"); raise HTTPException(500, "Failed to clear memory_bank")


@router.post("/clear/working")
async def clear_working_memory(request: Request):
    """Clear working memory (current conversation context)."""
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        if "working" not in memory.collections:
            raise HTTPException(404, "Working memory collection not found")

        adapter = memory.collections["working"]
        count_before = await adapter.get_collection_count()

        if count_before > 0:
            all_docs = adapter.collection.get(include=[])
            if all_docs.get("ids"):
                # Delete in batches to avoid ChromaDB batch size limits (max 166)
                batch_size = 100
                all_ids = all_docs["ids"]
                for i in range(0, len(all_ids), batch_size):
                    batch = all_ids[i : i + batch_size]
                    adapter.collection.delete(ids=batch)

        logger.info(f"Cleared working memory ({count_before} entries deleted)")

        return {
            "status": "success",
            "collection": "working",
            "deleted_count": count_before,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing working memory: {e}", exc_info=True)
        logger.error(f"Failed to clear working memory: {e}"); raise HTTPException(500, "Failed to clear working memory")


@router.post("/clear/history")
async def clear_history(request: Request):
    """Clear conversation history (30-day past conversations)."""
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        if "history" not in memory.collections:
            raise HTTPException(404, "History collection not found")

        adapter = memory.collections["history"]
        count_before = await adapter.get_collection_count()

        if count_before > 0:
            all_docs = adapter.collection.get(include=[])
            if all_docs.get("ids"):
                # Delete in batches to avoid ChromaDB batch size limits (max 166)
                batch_size = 100
                all_ids = all_docs["ids"]
                for i in range(0, len(all_ids), batch_size):
                    batch = all_ids[i : i + batch_size]
                    adapter.collection.delete(ids=batch)

        logger.info(f"Cleared history collection ({count_before} entries deleted)")

        return {
            "status": "success",
            "collection": "history",
            "deleted_count": count_before,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing history: {e}", exc_info=True)
        logger.error(f"Failed to clear history: {e}"); raise HTTPException(500, "Failed to clear history")


@router.post("/clear/patterns")
async def clear_patterns(request: Request):
    """Clear proven solution patterns."""
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        if "patterns" not in memory.collections:
            raise HTTPException(404, "Patterns collection not found")

        adapter = memory.collections["patterns"]
        count_before = await adapter.get_collection_count()

        if count_before > 0:
            all_docs = adapter.collection.get(include=[])
            if all_docs.get("ids"):
                # Delete in batches to avoid ChromaDB batch size limits (max 166)
                batch_size = 100
                all_ids = all_docs["ids"]
                for i in range(0, len(all_ids), batch_size):
                    batch = all_ids[i : i + batch_size]
                    adapter.collection.delete(ids=batch)

        logger.info(f"Cleared patterns collection ({count_before} entries deleted)")

        return {
            "status": "success",
            "collection": "patterns",
            "deleted_count": count_before,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing patterns: {e}", exc_info=True)
        logger.error(f"Failed to clear patterns: {e}"); raise HTTPException(500, "Failed to clear patterns")


@router.post("/clear/books")
async def clear_books(request: Request):
    """Clear uploaded books and reference documents (both ChromaDB and SQLite)."""
    memory = request.app.state.memory
    book_processor = request.app.state.book_processor

    if not memory:
        raise HTTPException(503, "Memory system not available")

    try:
        if "books" not in memory.collections:
            raise HTTPException(404, "Books collection not found")

        adapter = memory.collections["books"]
        count_before = await adapter.get_collection_count()

        # Step 1: Nuke and recreate ChromaDB collection (v0.2.9)
        # This fully rebuilds the HNSW index, eliminating ghost vectors
        # that remain after regular delete() operations
        if count_before > 0:
            collection_name = adapter.collection_name
            client = adapter.client

            # Delete the entire collection (removes HNSW index completely)
            client.delete_collection(name=collection_name)
            logger.info(f"Nuked books collection '{collection_name}'")

            # Recreate with same settings
            adapter.collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,  # We provide our own embeddings
                metadata={"hnsw:space": "l2"},
            )
            logger.info(f"Recreated books collection '{collection_name}'")

        logger.info(
            f"Cleared books ChromaDB collection ({count_before} entries deleted)"
        )

        # Step 1b: Clear ghost registry (v0.2.9)
        # Since we nuked the collection, no ghosts remain - clear the blacklist
        from config.settings import settings

        ghost_registry = get_ghost_registry(settings.paths.data_dir)
        ghosts_cleared = ghost_registry.clear()
        if ghosts_cleared > 0:
            logger.info(f"Cleared {ghosts_cleared} ghost IDs from registry")

        # Step 2: Clear SQLite database (book metadata and chunks)
        # Note: ghosts_cleared used in return value below
        sqlite_deleted = 0
        metadata_deleted = 0
        if book_processor:
            import aiosqlite

            db_path = book_processor.db_path
            if db_path.exists():
                async with aiosqlite.connect(str(db_path)) as db:
                    # Count books before deletion
                    async with db.execute("SELECT COUNT(*) FROM books") as cursor:
                        row = await cursor.fetchone()
                        sqlite_deleted = row[0] if row else 0

                    # Delete all books and chunks
                    await db.execute("DELETE FROM books")
                    await db.execute("DELETE FROM chunks")
                    await db.commit()

                logger.info(f"Cleared SQLite database ({sqlite_deleted} books deleted)")

            # Step 3: Clear metadata JSON files
            metadata_dir = book_processor.data_dir / "metadata"
            if metadata_dir.exists():
                import shutil

                metadata_files = list(metadata_dir.glob("*.json"))
                metadata_deleted = len(metadata_files)
                shutil.rmtree(metadata_dir)
                metadata_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Cleared metadata directory ({metadata_deleted} files deleted)"
                )

            # Step 4: Clear upload files
            uploads_dir = book_processor.data_dir / "uploads"
            uploads_deleted = 0
            if uploads_dir.exists():
                import shutil

                upload_files = list(uploads_dir.glob("*"))
                uploads_deleted = len(upload_files)
                shutil.rmtree(uploads_dir)
                uploads_dir.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"Cleared uploads directory ({uploads_deleted} files deleted)"
                )

        return {
            "status": "success",
            "collection": "books",
            "chromadb_deleted": count_before,
            "sqlite_deleted": sqlite_deleted,
            "metadata_deleted": metadata_deleted,
            "uploads_deleted": uploads_deleted,
            "ghosts_cleared": ghosts_cleared,  # v0.2.9
            "deleted_count": count_before,  # For UI backward compatibility
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing books: {e}", exc_info=True)
        logger.error(f"Failed to clear books: {e}"); raise HTTPException(500, "Failed to clear books")


@router.post("/clear/sessions")
async def clear_sessions(request: Request):
    """
    Delete all session/conversation files including archived sessions.

    Safety: Prevents deletion if any session is currently active.
    """
    memory = request.app.state.memory

    try:
        # Use AppData paths, not bundled data folder
        sessions_dir = memory.data_dir / "sessions" if memory else Path("data/sessions")
        if not sessions_dir.exists():
            return {
                "status": "success",
                "message": "No sessions directory found",
                "deleted_count": 0,
            }

        # Get active conversation ID if memory system is available
        active_conversation_id = None
        if memory and hasattr(memory, "conversation_id"):
            active_conversation_id = memory.conversation_id

        # v0.3.0: Include archive folder - use recursive glob
        session_files = list(sessions_dir.glob("**/*.jsonl"))
        deleted_count = 0
        skipped_active = False

        for session_file in session_files:
            session_id = session_file.stem

            # Skip active conversation
            if active_conversation_id and session_id == active_conversation_id:
                logger.warning(f"Skipping active conversation: {session_id}")
                skipped_active = True
                continue

            try:
                session_file.unlink()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete session {session_id}: {e}")

        # Clean up empty archive folder if it exists
        archive_dir = sessions_dir / "archive"
        if archive_dir.exists() and not any(archive_dir.iterdir()):
            try:
                archive_dir.rmdir()
                logger.info("Removed empty archive directory")
            except Exception:
                pass  # Not critical

        logger.info(f"Cleared {deleted_count} session files (including archive)")

        result = {"status": "success", "deleted_count": deleted_count}

        if skipped_active:
            result["warning"] = "Active conversation was preserved"
            result["active_conversation_id"] = active_conversation_id

        return result

    except Exception as e:
        logger.error(f"Error clearing sessions: {e}", exc_info=True)
        logger.error(f"Failed to clear sessions: {e}"); raise HTTPException(500, "Failed to clear sessions")


@router.post("/clear/outcomes")
async def clear_outcomes(request: Request):
    """
    Clear outcome tracking data (outcomes.db SQLite database).

    This removes all outcome scoring history used for memory ranking.
    """
    memory = request.app.state.memory

    try:
        data_dir = memory.data_dir if memory else Path("data")
        outcomes_db = data_dir / "outcomes.db"

        if not outcomes_db.exists():
            return {
                "status": "success",
                "message": "No outcomes database found",
                "deleted_count": 0,
            }

        try:
            outcomes_db.unlink()
            logger.info("Cleared outcomes.db")
            return {
                "status": "success",
                "deleted_count": 1,
                "message": "Outcomes database cleared",
            }
        except Exception as e:
            logger.error(f"Failed to delete outcomes.db: {e}")
            logger.error(f"Failed to delete outcomes.db: {e}"); raise HTTPException(500, "Failed to delete outcomes.db")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing outcomes: {e}", exc_info=True)
        logger.error(f"Failed to clear outcomes: {e}"); raise HTTPException(500, "Failed to clear outcomes")


@router.post("/compact-database")
async def compact_database(request: Request):
    """
    Compact ChromaDB to reclaim disk space from deleted items.
    VACUUM the ChromaDB SQLite database to free up space from deleted embeddings.
    """
    import sqlite3

    memory = request.app.state.memory

    try:
        # Use AppData paths, not bundled data folder
        chroma_db_path = (
            memory.data_dir / "chromadb/chroma.sqlite3"
            if memory
            else Path("data/chromadb/chroma.sqlite3")
        )

        if not chroma_db_path.exists():
            return {
                "status": "success",
                "message": "ChromaDB does not exist",
                "space_reclaimed": 0,
            }

        # Get size before compaction
        size_before = chroma_db_path.stat().st_size / (1024 * 1024)  # MB

        # Connect and VACUUM
        conn = sqlite3.connect(str(chroma_db_path))
        conn.execute("VACUUM")
        conn.close()

        # Get size after compaction
        size_after = chroma_db_path.stat().st_size / (1024 * 1024)  # MB
        space_reclaimed = size_before - size_after

        logger.info(
            f"ChromaDB compacted: {size_before:.1f}MB → {size_after:.1f}MB (reclaimed {space_reclaimed:.1f}MB)"
        )

        return {
            "status": "success",
            "size_before_mb": round(size_before, 2),
            "size_after_mb": round(size_after, 2),
            "space_reclaimed_mb": round(space_reclaimed, 2),
            "message": f"Reclaimed {space_reclaimed:.1f} MB",
        }

    except Exception as e:
        logger.error(f"Error compacting database: {e}", exc_info=True)
        logger.error(f"Failed to compact database: {e}"); raise HTTPException(500, "Failed to compact database")


# ---------------------------------------------------------------------------
# v0.3.1: Memory summarization / migration
# ---------------------------------------------------------------------------

SUMMARIZE_THRESHOLD = 400  # Characters — memories longer than this are candidates


@router.get("/summarize/scan")
async def summarize_scan(request: Request):
    """
    Scan for unsummarized memories. Lightweight — no LLM calls.
    Returns candidate counts per collection and sidecar status.
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    sidecar_configured = bool(getattr(request.app.state, "sidecar_model", ""))

    candidates: Dict[str, int] = {}
    for coll_name in ["working", "history", "patterns"]:
        adapter = memory.collections.get(coll_name)
        if not adapter or not adapter.collection:
            candidates[coll_name] = 0
            continue

        try:
            data = adapter.collection.get(include=["documents", "metadatas"])
            ids = data.get("ids", [])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])

            count = 0
            for i in range(len(ids)):
                doc = docs[i] if i < len(docs) and docs[i] else ""
                meta = metas[i] if i < len(metas) and metas[i] else {}
                if len(doc) > SUMMARIZE_THRESHOLD and not meta.get("summarized_at"):
                    count += 1
            candidates[coll_name] = count
        except Exception as e:
            logger.warning(f"[MIGRATION] Failed to scan {coll_name}: {e}")
            candidates[coll_name] = 0

    return {
        "candidates": candidates,
        "total": sum(candidates.values()),
        "sidecar_configured": sidecar_configured,
    }


@router.post("/summarize/run")
async def summarize_run(request: Request):
    """
    Summarize legacy memories via SSE stream.
    Per-memory: summarize → extract tags (LLM) → extract facts → update → store facts.
    Matches sidecar architecture: LLM tags for summaries and facts.
    """
    from modules.memory.sidecar_service import (
        summarize_only_with_retry,
        extract_facts_with_retry,
        extract_noun_tags_with_retry,
    )

    sidecar_client = getattr(request.app.state, "sidecar_client", None)
    sidecar_model = getattr(request.app.state, "sidecar_model", "")
    if not sidecar_client or not sidecar_model:
        raise HTTPException(400, "Sidecar model not configured")

    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    async def generate():
        cancel_key = "summarize_migration"
        _cancel_flags[cancel_key] = False

        # Phase 1: Scan candidates
        candidates = []
        for coll_name in ["working", "history", "patterns"]:
            adapter = memory.collections.get(coll_name)
            if not adapter or not adapter.collection:
                continue
            try:
                data = adapter.collection.get(include=["documents", "metadatas"])
                ids = data.get("ids", [])
                docs = data.get("documents", [])
                metas = data.get("metadatas", [])
                for i in range(len(ids)):
                    doc = docs[i] if i < len(docs) and docs[i] else ""
                    meta = metas[i] if i < len(metas) and metas[i] else {}
                    if len(doc) > SUMMARIZE_THRESHOLD and not meta.get("summarized_at"):
                        candidates.append((coll_name, ids[i], doc, meta))
            except Exception as e:
                logger.warning(f"[MIGRATION] Scan failed for {coll_name}: {e}")

        total = len(candidates)
        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

        if total == 0:
            yield f"data: {json.dumps({'type': 'complete', 'summarized': 0, 'facts': 0, 'tags': 0})}\n\n"
            return

        # Phase 2: Process each candidate
        summarized = 0
        facts_total = 0
        tags_total = 0

        for idx, (coll_name, doc_id, content, meta) in enumerate(candidates):
            if _cancel_flags.get(cancel_key):
                yield f"data: {json.dumps({'type': 'cancelled', 'summarized': summarized, 'facts': facts_total, 'tags': tags_total})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'progress', 'current': idx + 1, 'total': total, 'collection': coll_name, 'message': f'Summarizing {idx + 1}/{total}...'})}\n\n"

            try:
                # Step 1: Summarize
                summary = await summarize_only_with_retry(
                    content, sidecar_client, sidecar_model, doc_id=doc_id
                )
                if not summary:
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to summarize {doc_id}', 'current': idx + 1, 'total': total})}\n\n"
                    continue

                # Enforce length — prevent re-summarization loop
                if len(summary) > SUMMARIZE_THRESHOLD:
                    summary = summary[: SUMMARIZE_THRESHOLD - 20] + "... [truncated]"

                # Step 2: Extract tags (LLM, matching sidecar architecture)
                tags = (
                    await extract_noun_tags_with_retry(
                        summary, sidecar_client, sidecar_model, doc_id=doc_id
                    )
                    or []
                )

                # Step 3: Extract facts (LLM)
                facts = (
                    await extract_facts_with_retry(
                        content, sidecar_client, sidecar_model, doc_id=doc_id
                    )
                    or []
                )

                # Step 4: Update memory in ChromaDB
                adapter = memory.collections.get(coll_name)
                if adapter:
                    adapter.update_fragment_metadata(
                        doc_id,
                        {
                            "text": summary,
                            "content": summary,
                            "summarized_at": datetime.now().isoformat(),
                            "original_length": len(content),
                            "noun_tags": json.dumps(tags),
                        },
                    )

                # Step 5: Store extracted facts as new working memories (with LLM tags)
                for fact_text in facts:
                    fact_tags = (
                        await extract_noun_tags_with_retry(
                            fact_text,
                            sidecar_client,
                            sidecar_model,
                            doc_id=f"{doc_id}_fact",
                        )
                        or []
                    )
                    try:
                        await memory.store(
                            text=fact_text,
                            collection="working",
                            metadata={
                                "memory_type": "fact",
                                "role": "fact",
                                "source": "sidecar",
                                "score": 0.5,
                                "noun_tags": json.dumps(fact_tags),
                            },
                        )
                    except Exception as e:
                        logger.warning(f"[MIGRATION] Failed to store fact: {e}")
                    facts_total += 1

                # Step 6: Register tags with tag service
                if tags and hasattr(memory, "_tag_service") and memory._tag_service:
                    memory._tag_service.add_known_tags(tags)
                tags_total += len(tags)

                summarized += 1

            except Exception as e:
                logger.warning(f"[MIGRATION] Failed to process {doc_id}: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': f'Error: {doc_id}', 'current': idx + 1, 'total': total})}\n\n"

        yield f"data: {json.dumps({'type': 'complete', 'summarized': summarized, 'facts': facts_total, 'tags': tags_total})}\n\n"
        _cancel_flags.pop(cancel_key, None)

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.post("/summarize/cancel")
async def summarize_cancel():
    """Cancel an in-progress summarization migration."""
    _cancel_flags["summarize_migration"] = True
    return {"status": "cancelling"}
