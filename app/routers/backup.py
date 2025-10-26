"""
Complete backup and restore functionality for RoamPal
Handles full system backup including ChromaDB, sessions, books, and knowledge graph
Supports selective export with granular control
"""
import logging
import json
import zipfile
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from config.settings import DATA_PATH

logger = logging.getLogger(__name__)
router = APIRouter()


class BackupInfo(BaseModel):
    """Information about a backup"""
    filename: str
    size_mb: float
    created_at: str
    contains: Dict[str, bool]


class RestoreResponse(BaseModel):
    """Response from restore operation"""
    status: str
    message: str
    restored_items: Dict[str, int]


class ExportSizeEstimate(BaseModel):
    """Size estimate for export"""
    total_mb: float
    breakdown: Dict[str, float]
    file_counts: Dict[str, int]


# Valid export data types
VALID_EXPORT_TYPES = {"sessions", "memory", "books", "knowledge"}


def _backup_sessions(zipf: zipfile.ZipFile) -> int:
    """Backup sessions to zip file. Returns count of sessions backed up."""
    count = 0
    sessions_dir = Path(DATA_PATH) / "sessions"
    if sessions_dir.exists():
        for file in sessions_dir.glob("*.jsonl"):
            zipf.write(file, f"sessions/{file.name}")
            count += 1
    return count


def _backup_memory(zipf: zipfile.ZipFile) -> int:
    """Backup ChromaDB to zip file. Returns count of files backed up."""
    count = 0
    chromadb_dir = Path(DATA_PATH) / "chromadb"
    if chromadb_dir.exists():
        for file in chromadb_dir.rglob("*"):
            if file.is_file():
                rel_path = file.relative_to(str(Path(DATA_PATH) / "chromadb"))
                zipf.write(file, f"chromadb/{rel_path}")
                count += 1
    return count


def _backup_books(zipf: zipfile.ZipFile) -> int:
    """Backup books to zip file. Returns count of files backed up."""
    count = 0
    books_dir = Path(DATA_PATH) / "books"
    if books_dir.exists():
        # SQLite database
        books_db = books_dir / "books.db"
        if books_db.exists():
            zipf.write(books_db, "books/books.db")
            count += 1

        # Book files (archive, uploads, metadata)
        for subdir in ["archive", "uploads", "metadata"]:
            subdir_path = books_dir / subdir
            if subdir_path.exists():
                for file in subdir_path.rglob("*"):
                    if file.is_file():
                        rel_path = file.relative_to(str(Path(DATA_PATH) / "books"))
                        zipf.write(file, f"books/{rel_path}")
                        count += 1
    return count


def _backup_knowledge(zipf: zipfile.ZipFile) -> int:
    """Backup knowledge graph and outcomes. Returns count of files backed up."""
    count = 0

    # Knowledge graph
    kg_file = Path(DATA_PATH) / "knowledge_graph.json"
    if kg_file.exists():
        zipf.write(kg_file, "knowledge_graph.json")
        count += 1

    # Memory relationships
    rel_file = Path(DATA_PATH) / "memory_relationships.json"
    if rel_file.exists():
        zipf.write(rel_file, "memory_relationships.json")
        count += 1

    # Outcomes database
    outcomes_db = Path(DATA_PATH) / "outcomes.db"
    if outcomes_db.exists():
        zipf.write(outcomes_db, "outcomes.db")
        count += 1

    return count


@router.post("/create")
async def create_backup(
    include: Optional[str] = Query(None, description="Comma-separated list: sessions,memory,books,knowledge (default: all)")
) -> FileResponse:
    """
    Create system backup as downloadable zip with selective export support

    Query Parameters:
    - include: Comma-separated list of data types to include
      - "sessions": Conversation history
      - "memory": ChromaDB vector embeddings (the actual memory)
      - "books": Books database and uploaded files
      - "knowledge": Knowledge graph, relationships, outcomes
      - Default: All data types included

    Examples:
    - POST /api/backup/create (full backup)
    - POST /api/backup/create?include=sessions,memory (only conversations and memory)
    - POST /api/backup/create?include=sessions (only conversations)
    """
    try:
        # Parse include parameter
        if include:
            include_types = set(t.strip().lower() for t in include.split(","))
            # Validate types
            invalid_types = include_types - VALID_EXPORT_TYPES
            if invalid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid export types: {', '.join(invalid_types)}. Valid types: {', '.join(VALID_EXPORT_TYPES)}"
                )
        else:
            # Default: include everything
            include_types = VALID_EXPORT_TYPES.copy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create descriptive filename based on what's included
        if include_types == VALID_EXPORT_TYPES:
            backup_name = f"roampal_backup_{timestamp}.zip"
        else:
            types_str = "_".join(sorted(include_types))
            backup_name = f"roampal_{types_str}_{timestamp}.zip"

        # Create backups directory
        backups_dir = Path(DATA_PATH) / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backups_dir / backup_name

        logger.info(f"Creating backup: {backup_name} (including: {', '.join(include_types)})")

        # Create backup with selected data types
        backup_stats = {}

        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if "sessions" in include_types:
                backup_stats["sessions"] = _backup_sessions(zipf)
                logger.info(f"Backed up {backup_stats['sessions']} sessions")

            if "memory" in include_types:
                backup_stats["chromadb_files"] = _backup_memory(zipf)
                logger.info(f"Backed up {backup_stats['chromadb_files']} ChromaDB files")

            if "books" in include_types:
                backup_stats["books"] = _backup_books(zipf)
                logger.info(f"Backed up {backup_stats['books']} book files")

            if "knowledge" in include_types:
                backup_stats["knowledge_files"] = _backup_knowledge(zipf)
                logger.info(f"Backed up {backup_stats['knowledge_files']} knowledge files")

            # Add metadata
            metadata = {
                "backup_date": timestamp,
                "backup_version": "1.1",  # Updated version for selective export
                "roampal_version": "2.0",
                "backup_type": "selective" if include_types != VALID_EXPORT_TYPES else "full",
                "included_types": list(include_types),
                "backup_stats": backup_stats,
                "total_files": sum(backup_stats.values())
            }
            zipf.writestr("backup_info.json", json.dumps(metadata, indent=2))

        # Get file size
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        logger.info(f"Backup created successfully: {backup_name} ({size_mb:.2f} MB)")

        return FileResponse(
            path=str(backup_path),
            media_type="application/zip",
            filename=backup_name,
            headers={
                "Content-Disposition": f"attachment; filename={backup_name}"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backup creation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Backup failed: {str(e)}"
        )


@router.get("/estimate")
async def estimate_export_size(
    include: Optional[str] = Query(None, description="Comma-separated list: sessions,memory,books,knowledge (default: all)")
) -> ExportSizeEstimate:
    """
    Estimate size of export before creating backup

    Returns breakdown by data type and file counts for informed decisions
    """
    try:
        # Parse include parameter
        if include:
            include_types = set(t.strip().lower() for t in include.split(","))
            invalid_types = include_types - VALID_EXPORT_TYPES
            if invalid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid export types: {', '.join(invalid_types)}"
                )
        else:
            include_types = VALID_EXPORT_TYPES.copy()

        breakdown = {}
        file_counts = {}

        # Calculate sessions size
        if "sessions" in include_types:
            sessions_dir = Path(DATA_PATH) / "sessions"
            if sessions_dir.exists():
                size_bytes = sum(f.stat().st_size for f in sessions_dir.glob("*.jsonl"))
                file_count = len(list(sessions_dir.glob("*.jsonl")))
                breakdown["sessions_mb"] = round(size_bytes / (1024**2), 2)
                file_counts["sessions"] = file_count
            else:
                breakdown["sessions_mb"] = 0
                file_counts["sessions"] = 0

        # Calculate ChromaDB size
        if "memory" in include_types:
            chromadb_dir = Path(DATA_PATH) / "chromadb"
            if chromadb_dir.exists():
                size_bytes = sum(
                    f.stat().st_size
                    for f in chromadb_dir.rglob("*")
                    if f.is_file()
                )
                file_count = sum(1 for _ in chromadb_dir.rglob("*") if _.is_file())
                breakdown["memory_mb"] = round(size_bytes / (1024**2), 2)
                file_counts["memory"] = file_count
            else:
                breakdown["memory_mb"] = 0
                file_counts["memory"] = 0

        # Calculate books size
        if "books" in include_types:
            books_dir = Path(DATA_PATH) / "books"
            if books_dir.exists():
                size_bytes = sum(
                    f.stat().st_size
                    for f in books_dir.rglob("*")
                    if f.is_file()
                )
                file_count = sum(1 for _ in books_dir.rglob("*") if _.is_file())
                breakdown["books_mb"] = round(size_bytes / (1024**2), 2)
                file_counts["books"] = file_count
            else:
                breakdown["books_mb"] = 0
                file_counts["books"] = 0

        # Calculate knowledge size
        if "knowledge" in include_types:
            size_bytes = 0
            file_count = 0

            for file_path in [
                Path(DATA_PATH) / "knowledge_graph.json",
                Path(DATA_PATH) / "memory_relationships.json",
                Path(DATA_PATH) / "outcomes.db"
            ]:
                if file_path.exists():
                    size_bytes += file_path.stat().st_size
                    file_count += 1

            breakdown["knowledge_mb"] = round(size_bytes / (1024**2), 2)
            file_counts["knowledge"] = file_count

        total_mb = round(sum(breakdown.values()), 2)

        return ExportSizeEstimate(
            total_mb=total_mb,
            breakdown=breakdown,
            file_counts=file_counts
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Size estimation failed: {e}")
        raise HTTPException(500, detail=str(e))


@router.get("/list")
async def list_backups():
    """List all available backups"""
    try:
        backups_dir = Path(DATA_PATH) / "backups"
        if not backups_dir.exists():
            return {"backups": []}

        backups = []
        for backup_file in sorted(backups_dir.glob("roampal_backup_*.zip"), reverse=True):
            try:
                # Try to read metadata from backup
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    if "backup_info.json" in zipf.namelist():
                        metadata = json.loads(zipf.read("backup_info.json"))
                        contains = {
                            "sessions": metadata.get("backup_stats", {}).get("sessions", 0) > 0,
                            "chromadb": metadata.get("backup_stats", {}).get("chromadb_files", 0) > 0,
                            "books": metadata.get("backup_stats", {}).get("books", 0) > 0,
                            "knowledge_graph": metadata.get("backup_stats", {}).get("knowledge_files", 0) > 0
                        }
                    else:
                        contains = {"sessions": True, "chromadb": True, "books": True, "knowledge_graph": True}

                backups.append({
                    "filename": backup_file.name,
                    "size_mb": backup_file.stat().st_size / (1024 * 1024),
                    "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                    "contains": contains
                })
            except Exception as e:
                logger.warning(f"Could not read backup metadata for {backup_file.name}: {e}")
                # Still include in list
                backups.append({
                    "filename": backup_file.name,
                    "size_mb": backup_file.stat().st_size / (1024 * 1024),
                    "created_at": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
                    "contains": {}
                })

        return {"backups": backups}

    except Exception as e:
        logger.error(f"Failed to list backups: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore")
async def restore_from_backup(file: UploadFile = File(...)) -> RestoreResponse:
    """
    Restore from backup zip

    CAUTION: This will replace all current data with backup data.
    Current data will be backed up to pre_restore_<timestamp> before restoration.
    """
    temp_dir = None
    try:
        logger.info(f"Starting restore from backup: {file.filename}")

        # Create temp directory for extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="roampal_restore_"))

        # Save uploaded file
        upload_path = temp_dir / "backup.zip"
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract and validate
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()

        with zipfile.ZipFile(upload_path, 'r') as zipf:
            zipf.extractall(extract_dir)

        # Validate backup structure
        if not (extract_dir / "backup_info.json").exists():
            raise ValueError("Invalid backup: missing backup_info.json")

        # Read metadata
        with open(extract_dir / "backup_info.json") as f:
            metadata = json.load(f)

        logger.info(f"Backup metadata: {metadata}")

        # Create backup of current data before restoration
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pre_restore_backup = Path(DATA_PATH) / f"backups/pre_restore_{backup_timestamp}.zip"
        pre_restore_backup.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Backing up current data to: {pre_restore_backup.name}")

        # Backup current data (excluding backups folder itself)
        data_dir = Path(DATA_PATH)
        if data_dir.exists():
            for item in data_dir.iterdir():
                if item.name != "backups":
                    try:
                        dest = pre_restore_backup / item.name
                        if item.is_dir():
                            shutil.copytree(item, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dest)
                    except Exception as e:
                        logger.warning(f"Could not backup {item}: {e}")

        # Restore data
        restored_items = {
            "sessions": 0,
            "chromadb_files": 0,
            "books": 0,
            "knowledge_files": 0
        }

        # 1. Restore sessions
        sessions_src = extract_dir / "sessions"
        if sessions_src.exists():
            sessions_dest = Path(DATA_PATH) / "sessions"
            sessions_dest.mkdir(parents=True, exist_ok=True)

            # Clear existing sessions
            for old_file in sessions_dest.glob("*.jsonl"):
                old_file.unlink()

            # Copy from backup
            for file in sessions_src.glob("*.jsonl"):
                shutil.copy2(file, sessions_dest / file.name)
                restored_items["sessions"] += 1
            logger.info(f"Restored {restored_items['sessions']} sessions")

        # 2. Restore ChromaDB
        chromadb_src = extract_dir / "chromadb"
        if chromadb_src.exists():
            chromadb_dest = Path(DATA_PATH) / "chromadb"

            # Remove existing ChromaDB
            if chromadb_dest.exists():
                shutil.rmtree(chromadb_dest)

            # Copy from backup
            shutil.copytree(chromadb_src, chromadb_dest)
            restored_items["chromadb_files"] = sum(1 for _ in chromadb_dest.rglob("*") if _.is_file())
            logger.info(f"Restored {restored_items['chromadb_files']} ChromaDB files")

        # 3. Restore books
        books_src = extract_dir / "books"
        if books_src.exists():
            books_dest = Path(DATA_PATH) / "books"
            books_dest.mkdir(parents=True, exist_ok=True)

            for item in books_src.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(books_src)
                    dest_file = books_dest / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_file)
                    restored_items["books"] += 1
            logger.info(f"Restored {restored_items['books']} book files")

        # 4. Restore knowledge graph and outcomes
        for filename in ["knowledge_graph.json", "memory_relationships.json", "outcomes.db"]:
            src_file = extract_dir / filename
            if src_file.exists():
                dest_file = Path(DATA_PATH) / filename
                shutil.copy2(src_file, dest_file)
                restored_items["knowledge_files"] += 1

        logger.info(f"Restored {restored_items['knowledge_files']} knowledge files")

        logger.info("Restore completed successfully")

        return RestoreResponse(
            status="success",
            message=f"Backup restored successfully. Previous data saved to {pre_restore_backup.name}. Please restart the application.",
            restored_items=restored_items
        )

    except Exception as e:
        logger.error(f"Restore failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Restore failed: {str(e)}"
        )

    finally:
        # Cleanup temp directory
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Could not cleanup temp directory: {e}")


@router.delete("/cleanup")
async def cleanup_old_backups(keep: int = 7):
    """
    Clean up old backups, keeping only the N most recent

    Args:
        keep: Number of backups to keep (default: 7)
    """
    try:
        backups_dir = Path(DATA_PATH) / "backups"
        if not backups_dir.exists():
            return {"deleted": 0, "kept": 0}

        # Get all backup files (excluding pre_restore backups)
        backups = sorted(
            [f for f in backups_dir.glob("roampal_backup_*.zip")],
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )

        deleted = 0
        for old_backup in backups[keep:]:
            old_backup.unlink()
            logger.info(f"Deleted old backup: {old_backup.name}")
            deleted += 1

        return {
            "deleted": deleted,
            "kept": len(backups[:keep]),
            "message": f"Kept {len(backups[:keep])} most recent backups, deleted {deleted} old backups"
        }

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
