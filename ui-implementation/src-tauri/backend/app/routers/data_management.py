"""
Data Management API Router

Provides endpoints for clearing/deleting user data collections.
All operations are destructive and permanent.
"""

import logging
from pathlib import Path
from typing import Dict
from fastapi import APIRouter, Request, HTTPException
import json

# Ghost registry for clearing ghost IDs after collection nuke (v0.2.9)
from modules.memory.ghost_registry import get_ghost_registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/data", tags=["data-management"])


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
            "knowledge_graph": {"nodes": 45, "edges": 23}
        }
    """
    memory = request.app.state.memory
    if not memory:
        raise HTTPException(503, "Memory system not available")

    stats = {}

    try:
        # Get ChromaDB collection counts
        for collection_name in ["memory_bank", "working", "history", "patterns", "books"]:
            if collection_name in memory.collections:
                adapter = memory.collections[collection_name]
                count = await adapter.get_collection_count()

                # Memory bank has active/archived split
                if collection_name == "memory_bank":
                    try:
                        results = adapter.collection.get(
                            where={"status": "active"},
                            include=[]
                        )
                        active_count = len(results.get("ids", []))
                        archived_count = count - active_count
                        stats[collection_name] = {
                            "count": count,
                            "active": active_count,
                            "archived": archived_count
                        }
                    except:
                        stats[collection_name] = {"count": count}
                else:
                    stats[collection_name] = {"count": count}

        # Count session files (use memory.data_dir for correct path)
        sessions_dir = memory.data_dir / "sessions"
        if sessions_dir.exists():
            session_files = list(sessions_dir.glob("*.jsonl"))
            stats["sessions"] = {"count": len(session_files)}
        else:
            stats["sessions"] = {"count": 0}

        # Parse knowledge graph (use memory.kg_path for correct location)
        kg_path = memory.kg_path
        if kg_path and kg_path.exists():
            try:
                with open(kg_path, 'r', encoding='utf-8') as f:
                    kg_data = json.load(f)

                # Count actual KG data (not just legacy nodes/edges)
                nodes = len(kg_data.get("routing_patterns", {}))
                nodes += len(kg_data.get("problem_solutions", {}))
                nodes += len(kg_data.get("solution_patterns", {}))

                # Count edges from memory_relationships.json (document relationships)
                edges = 0
                rel_path = memory.data_dir / "memory_relationships.json"
                if rel_path.exists():
                    try:
                        with open(rel_path, 'r', encoding='utf-8') as f:
                            rel_data = json.load(f)
                            # Count document relationships
                            edges = len(rel_data.get("related", {}))
                            edges += len(rel_data.get("evolution", {}))
                            edges += len(rel_data.get("conflicts", {}))
                    except:
                        pass

                stats["knowledge_graph"] = {
                    "nodes": nodes,
                    "edges": edges
                }
            except:
                stats["knowledge_graph"] = {"nodes": 0, "edges": 0}
        else:
            stats["knowledge_graph"] = {"nodes": 0, "edges": 0}

        return stats

    except Exception as e:
        logger.error(f"Error fetching data stats: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to get stats: {str(e)}")


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
                    batch = all_ids[i:i + batch_size]
                    adapter.collection.delete(ids=batch)

        logger.info(f"Cleared memory_bank collection ({count_before} entries deleted)")

        return {
            "status": "success",
            "collection": "memory_bank",
            "deleted_count": count_before
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing memory_bank: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to clear memory_bank: {str(e)}")


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
                    batch = all_ids[i:i + batch_size]
                    adapter.collection.delete(ids=batch)

        logger.info(f"Cleared working memory ({count_before} entries deleted)")

        return {
            "status": "success",
            "collection": "working",
            "deleted_count": count_before
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing working memory: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to clear working memory: {str(e)}")


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
                    batch = all_ids[i:i + batch_size]
                    adapter.collection.delete(ids=batch)

        logger.info(f"Cleared history collection ({count_before} entries deleted)")

        return {
            "status": "success",
            "collection": "history",
            "deleted_count": count_before
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing history: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to clear history: {str(e)}")


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
                    batch = all_ids[i:i + batch_size]
                    adapter.collection.delete(ids=batch)

        logger.info(f"Cleared patterns collection ({count_before} entries deleted)")

        return {
            "status": "success",
            "collection": "patterns",
            "deleted_count": count_before
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing patterns: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to clear patterns: {str(e)}")


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
                metadata={"hnsw:space": "l2"}
            )
            logger.info(f"Recreated books collection '{collection_name}'")

        logger.info(f"Cleared books ChromaDB collection ({count_before} entries deleted)")

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
                logger.info(f"Cleared metadata directory ({metadata_deleted} files deleted)")

            # Step 4: Clear upload files
            uploads_dir = book_processor.data_dir / "uploads"
            uploads_deleted = 0
            if uploads_dir.exists():
                import shutil
                upload_files = list(uploads_dir.glob("*"))
                uploads_deleted = len(upload_files)
                shutil.rmtree(uploads_dir)
                uploads_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Cleared uploads directory ({uploads_deleted} files deleted)")

        return {
            "status": "success",
            "collection": "books",
            "chromadb_deleted": count_before,
            "sqlite_deleted": sqlite_deleted,
            "metadata_deleted": metadata_deleted,
            "uploads_deleted": uploads_deleted,
            "ghosts_cleared": ghosts_cleared,  # v0.2.9
            "deleted_count": count_before  # For UI backward compatibility
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing books: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to clear books: {str(e)}")


@router.post("/clear/sessions")
async def clear_sessions(request: Request):
    """
    Delete all session/conversation files.

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
                "deleted_count": 0
            }

        # Get active conversation ID if memory system is available
        active_conversation_id = None
        if memory and hasattr(memory, 'conversation_id'):
            active_conversation_id = memory.conversation_id

        session_files = list(sessions_dir.glob("*.jsonl"))
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

        logger.info(f"Cleared {deleted_count} session files")

        result = {
            "status": "success",
            "deleted_count": deleted_count
        }

        if skipped_active:
            result["warning"] = "Active conversation was preserved"
            result["active_conversation_id"] = active_conversation_id

        return result

    except Exception as e:
        logger.error(f"Error clearing sessions: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to clear sessions: {str(e)}")


@router.post("/clear/knowledge-graph")
async def clear_knowledge_graph(request: Request):
    """Clear the knowledge graph (concept relationships)."""
    memory = request.app.state.memory

    try:
        # Use AppData paths, not bundled data folder
        kg_path = memory.kg_path if memory else Path("data/knowledge_graph.json")

        if not kg_path.exists():
            return {
                "status": "success",
                "message": "Knowledge graph file does not exist",
                "cleared": False
            }

        # Read existing to get count (use actual KG structure, not legacy nodes/edges)
        nodes_count = 0
        edges_count = 0
        try:
            with open(kg_path, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
            # Count actual KG data
            nodes_count = len(kg_data.get("routing_patterns", {}))
            nodes_count += len(kg_data.get("problem_solutions", {}))
            nodes_count += len(kg_data.get("solution_patterns", {}))
            edges_count = len(kg_data.get("relationships", {}))
        except:
            pass

        # Overwrite with empty structure (use proper schema from unified_memory_system)
        empty_kg = {
            "routing_patterns": {},
            "success_rates": {},
            "failure_patterns": {},
            "problem_categories": {},
            "problem_solutions": {},
            "solution_patterns": {}
        }

        with open(kg_path, 'w', encoding='utf-8') as f:
            json.dump(empty_kg, f, indent=2)

        # Also clear memory_relationships.json
        rel_path = memory.data_dir / "memory_relationships.json" if memory else Path("data/memory_relationships.json")
        empty_rel = {
            "related": {},
            "evolution": {},
            "conflicts": {}
        }
        if rel_path.exists():
            with open(rel_path, 'w', encoding='utf-8') as f:
                json.dump(empty_rel, f, indent=2)

        # CRITICAL: Clear in-memory cache in the UnifiedMemorySystem
        if memory:
            if hasattr(memory, 'knowledge_graph'):
                memory.knowledge_graph = empty_kg
                logger.info("Cleared in-memory knowledge graph cache")
            if hasattr(memory, 'relationships'):
                memory.relationships = empty_rel
                logger.info("Cleared in-memory relationships cache")

        logger.info(f"Cleared knowledge graph ({nodes_count} nodes, {edges_count} edges)")

        return {
            "status": "success",
            "cleared": True,
            "nodes_cleared": nodes_count,
            "edges_cleared": edges_count
        }

    except Exception as e:
        logger.error(f"Error clearing knowledge graph: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to clear knowledge graph: {str(e)}")


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
        chroma_db_path = memory.data_dir / "chromadb/chroma.sqlite3" if memory else Path("data/chromadb/chroma.sqlite3")

        if not chroma_db_path.exists():
            return {
                "status": "success",
                "message": "ChromaDB does not exist",
                "space_reclaimed": 0
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

        logger.info(f"ChromaDB compacted: {size_before:.1f}MB → {size_after:.1f}MB (reclaimed {space_reclaimed:.1f}MB)")

        return {
            "status": "success",
            "size_before_mb": round(size_before, 2),
            "size_after_mb": round(size_after, 2),
            "space_reclaimed_mb": round(space_reclaimed, 2),
            "message": f"Reclaimed {space_reclaimed:.1f} MB"
        }

    except Exception as e:
        logger.error(f"Error compacting database: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to compact database: {str(e)}")
