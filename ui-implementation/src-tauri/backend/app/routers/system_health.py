"""
System Health Router - Monitor system health and resources
"""
import logging
import shutil
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
router = APIRouter()


class SystemHealthResponse(BaseModel):
    """System health status response"""
    status: str  # "healthy", "warning", "error"
    timestamp: str
    disk: Dict[str, Any]
    data_sizes: Dict[str, Any]
    warnings: list[str]
    integrity_check: Optional[Dict[str, Any]] = None


@router.get("/health")
async def get_system_health(request: Request) -> SystemHealthResponse:
    """
    Get comprehensive system health status

    Returns:
        System health with disk space, data sizes, and warnings
    """
    try:
        warnings = []

        # Check disk space
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        used_percent = (used / total) * 100

        if free_gb < 0.5:
            warnings.append("❌ Critical: Less than 500MB free disk space")
        elif free_gb < 1.0:
            warnings.append("⚠️ Low disk space: Less than 1GB free")
        elif free_gb < 5.0:
            warnings.append("⚠️ Disk space getting low: Less than 5GB free")

        disk_info = {
            "total_gb": round(total_gb, 2),
            "used_gb": round((total - free) / (1024**3), 2),
            "free_gb": round(free_gb, 2),
            "used_percent": round(used_percent, 1)
        }

        # Calculate data sizes
        data_sizes = {}

        # ChromaDB size
        chromadb_dir = Path("data/chromadb")
        if chromadb_dir.exists():
            chromadb_size = sum(
                f.stat().st_size
                for f in chromadb_dir.rglob("*")
                if f.is_file()
            ) / (1024**2)  # MB
            data_sizes["chromadb_mb"] = round(chromadb_size, 2)

            if chromadb_size > 1000:  # 1GB
                warnings.append(f"⚠️ ChromaDB size large: {chromadb_size:.0f}MB")
        else:
            data_sizes["chromadb_mb"] = 0

        # Sessions count
        sessions_dir = Path("data/sessions")
        if sessions_dir.exists():
            session_count = len(list(sessions_dir.glob("*.jsonl")))
            data_sizes["sessions_count"] = session_count

            if session_count > 1000:
                warnings.append(f"⚠️ Many sessions: {session_count} (consider cleanup)")
        else:
            data_sizes["sessions_count"] = 0

        # Books database size
        books_db = Path("data/books/books.db")
        if books_db.exists():
            books_size = books_db.stat().st_size / (1024**2)  # MB
            data_sizes["books_db_mb"] = round(books_size, 2)
        else:
            data_sizes["books_db_mb"] = 0

        # Backups size
        backups_dir = Path("data/backups")
        if backups_dir.exists():
            backup_count = len(list(backups_dir.glob("*.zip")))
            backup_size = sum(
                f.stat().st_size
                for f in backups_dir.glob("*.zip")
            ) / (1024**2)  # MB
            data_sizes["backups_count"] = backup_count
            data_sizes["backups_mb"] = round(backup_size, 2)

            if backup_size > 500:  # 500MB
                warnings.append(f"⚠️ Large backup folder: {backup_size:.0f}MB (consider cleanup)")
        else:
            data_sizes["backups_count"] = 0
            data_sizes["backups_mb"] = 0

        # Get integrity check results if available
        integrity_check = None
        if hasattr(request.app.state, 'integrity_check'):
            integrity_check = request.app.state.integrity_check

        # Determine overall status
        if any("Critical" in w or "❌" in w for w in warnings):
            status = "error"
        elif warnings:
            status = "warning"
        else:
            status = "healthy"

        return SystemHealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            disk=disk_info,
            data_sizes=data_sizes,
            warnings=warnings if warnings else ["✅ All systems normal"],
            integrity_check=integrity_check
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return SystemHealthResponse(
            status="error",
            timestamp=datetime.now().isoformat(),
            disk={},
            data_sizes={},
            warnings=[f"❌ Health check failed: {str(e)}"]
        )


@router.get("/disk-space")
async def get_disk_space():
    """
    Get detailed disk space information

    Returns:
        Disk space metrics
    """
    try:
        total, used, free = shutil.disk_usage(".")

        return {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "used_percent": round((used / total) * 100, 1),
            "free_percent": round((free / total) * 100, 1),
            "warning": free / (1024**3) < 1.0,  # Less than 1GB free
            "critical": free / (1024**3) < 0.5  # Less than 500MB free
        }
    except Exception as e:
        logger.error(f"Disk space check failed: {e}")
        return {
            "error": str(e)
        }


@router.get("/data-sizes")
async def get_data_sizes():
    """
    Get sizes of all data directories

    Returns:
        Breakdown of data storage usage
    """
    try:
        sizes = {}

        # Check all data subdirectories
        data_dirs = {
            "chromadb": Path("data/chromadb"),
            "sessions": Path("data/sessions"),
            "books": Path("data/books"),
            "backups": Path("data/backups")
        }

        for name, dir_path in data_dirs.items():
            if dir_path.exists():
                size_bytes = sum(
                    f.stat().st_size
                    for f in dir_path.rglob("*")
                    if f.is_file()
                )
                file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())

                sizes[name] = {
                    "size_mb": round(size_bytes / (1024**2), 2),
                    "size_gb": round(size_bytes / (1024**3), 3),
                    "file_count": file_count
                }
            else:
                sizes[name] = {
                    "size_mb": 0,
                    "size_gb": 0,
                    "file_count": 0
                }

        # Calculate total
        total_mb = sum(s["size_mb"] for s in sizes.values())

        return {
            "breakdown": sizes,
            "total_mb": round(total_mb, 2),
            "total_gb": round(total_mb / 1024, 3)
        }

    except Exception as e:
        logger.error(f"Data sizes check failed: {e}")
        return {
            "error": str(e)
        }


@router.get("/migration-notice")
async def get_migration_notice():
    """
    Check if there's a pending embedding migration notice for v0.1.7
    Returns migration info if notice exists, null otherwise
    """
    try:
        from config.settings import DATA_PATH
        notice_path = Path(DATA_PATH) / ".embedding_migration_notice"

        if notice_path.exists():
            import json
            notice_data = json.loads(notice_path.read_text())
            return {
                "has_notice": True,
                "notice": notice_data
            }

        return {"has_notice": False}

    except Exception as e:
        logger.error(f"Error checking migration notice: {e}")
        return {"has_notice": False, "error": str(e)}


@router.post("/migration-notice/dismiss")
async def dismiss_migration_notice():
    """
    Dismiss the embedding migration notice after user acknowledges it
    """
    try:
        from config.settings import DATA_PATH
        notice_path = Path(DATA_PATH) / ".embedding_migration_notice"
        flag_path = Path(DATA_PATH) / ".embedding_migration_v017"

        # Remove notice
        if notice_path.exists():
            notice_path.unlink()

        # Mark migration as acknowledged
        flag_path.touch()

        logger.info("[Migration] User acknowledged embedding migration notice")
        return {"success": True, "message": "Migration notice dismissed"}

    except Exception as e:
        logger.error(f"Error dismissing migration notice: {e}")
        return {"success": False, "error": str(e)}
