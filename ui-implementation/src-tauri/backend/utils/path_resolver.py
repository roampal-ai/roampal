"""Path resolution utility for dual-mode file system (legacy and auth-based)"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Cache for user lookup
_users_cache = None
_cache_timestamp = 0

def load_users_cache():
    """Load users from storage for path resolution"""
    global _users_cache, _cache_timestamp
    
    users_file = Path("backend/data/users.json")
    if not users_file.exists():
        return {}
    
    # Check if cache is still valid (file hasn't changed)
    current_mtime = users_file.stat().st_mtime
    if _users_cache is None or current_mtime > _cache_timestamp:
        try:
            _users_cache = json.loads(users_file.read_text())
            _cache_timestamp = current_mtime
        except:
            _users_cache = {}
    
    return _users_cache

def get_user_uuid(username: str) -> Optional[str]:
    """Get UUID for a username"""
    users = load_users_cache()
    user_data = users.get(username)
    return user_data.get("user_id") if user_data else None

def resolve_user_path(username: str, shard: str, path_type: str = "sessions", filename: Optional[str] = None) -> Tuple[Path, Path]:
    """
    Resolve user path for both new (UUID-based) and legacy systems
    
    Returns: (new_path, legacy_path) tuple
    
    path_type can be: 'sessions', 'images', 'shards', etc.
    """
    # Legacy path (always available)
    if path_type == "sessions":
        legacy_base = Path(f"backend/data/shards/{shard}/sessions")
        if filename:
            legacy_path = legacy_base / f"{username}_{filename}"
        else:
            legacy_path = legacy_base
    elif path_type == "images":
        legacy_base = Path(f"backend/data/shards/{shard}/images")
        if filename:
            legacy_path = legacy_base / f"{username}_{filename}"
        else:
            legacy_path = legacy_base
    else:
        legacy_path = Path(f"backend/data/shards/{shard}/{path_type}")
        if filename:
            legacy_path = legacy_path / filename
    
    # New UUID-based path (if user has UUID)
    user_uuid = get_user_uuid(username)
    if user_uuid:
        new_base = Path(f"backend/data/users/{user_uuid}/shards/{shard}/{path_type}")
        if filename:
            new_path = new_base / filename
        else:
            new_path = new_base
    else:
        new_path = None
    
    return (new_path, legacy_path)

def get_user_data_path(username: str, shard: str, path_type: str = "sessions", filename: Optional[str] = None, write: bool = False) -> Path:
    """
    Get the appropriate path for user data (read or write)
    
    For reads: Check new path first, fall back to legacy
    For writes: Use new path if user has UUID, otherwise use legacy
    """
    new_path, legacy_path = resolve_user_path(username, shard, path_type, filename)
    
    if write:
        # For writes, prefer new path if available
        if new_path:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            return new_path
        else:
            legacy_path.parent.mkdir(parents=True, exist_ok=True)
            return legacy_path
    else:
        # For reads, check both paths
        if new_path and new_path.exists():
            return new_path
        elif legacy_path.exists():
            return legacy_path
        else:
            # Return the path that would be used for writing
            return new_path if new_path else legacy_path

def migrate_user_data(username: str, shard: str):
    """
    Migrate user data from legacy to new UUID-based structure
    """
    user_uuid = get_user_uuid(username)
    if not user_uuid:
        logger.warning(f"Cannot migrate {username} - no UUID found")
        return False
    
    migrations = []
    
    # Migrate sessions
    legacy_sessions = Path(f"backend/data/shards/{shard}/sessions")
    if legacy_sessions.exists():
        for session_file in legacy_sessions.glob(f"{username}_*.json"):
            new_name = session_file.name.replace(f"{username}_", "")
            new_path = Path(f"backend/data/users/{user_uuid}/shards/{shard}/sessions/{new_name}")
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not new_path.exists():
                new_path.write_bytes(session_file.read_bytes())
                migrations.append(f"Session: {session_file.name}")
    
    # Migrate images
    legacy_images = Path(f"backend/data/shards/{shard}/images")
    if legacy_images.exists():
        for image_file in legacy_images.glob(f"{username}_*"):
            new_name = image_file.name.replace(f"{username}_", "")
            new_path = Path(f"backend/data/users/{user_uuid}/shards/{shard}/images/{new_name}")
            new_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not new_path.exists():
                new_path.write_bytes(image_file.read_bytes())
                migrations.append(f"Image: {image_file.name}")
    
    if migrations:
        logger.info(f"Migrated {len(migrations)} files for {username} ({user_uuid})")
        return True
    return False

def list_user_files(username: str, shard: str, path_type: str = "sessions") -> list:
    """
    List all files for a user (from both new and legacy paths)
    """
    files = []
    
    new_path, legacy_path = resolve_user_path(username, shard, path_type)
    
    # Check new path
    if new_path and new_path.exists():
        files.extend(list(new_path.glob("*")))
    
    # Check legacy path
    if legacy_path.exists():
        if path_type in ["sessions", "images"]:
            # For sessions/images, filter by username prefix
            pattern = f"{username}_*"
            files.extend(list(legacy_path.glob(pattern)))
        else:
            files.extend(list(legacy_path.glob("*")))
    
    # Remove duplicates (by filename)
    seen = set()
    unique_files = []
    for f in files:
        name = f.name.replace(f"{username}_", "") if f"{username}_" in f.name else f.name
        if name not in seen:
            seen.add(name)
            unique_files.append(f)
    
    return unique_files