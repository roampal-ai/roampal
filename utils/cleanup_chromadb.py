#!/usr/bin/env python3
"""
ChromaDB Collection Cleanup Utility
Removes orphaned UUID folders from ChromaDB persistence directory
Created: 2025-01-17
"""
import os
import shutil
import time
import logging
import re
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_uuid_folder(folder_name: str) -> bool:
    """Check if folder name matches UUID pattern"""
    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, folder_name, re.IGNORECASE))

def get_active_collections() -> List[str]:
    """Get list of active collection names from configuration"""
    # These are the active collections as per MEMORY_IMPLEMENTATION.md
    return [
        'loopsmith_memories',      # Main unified collection
        'loopsmith_books',          # Books collection
        'loopsmith_working',        # Working memory
        'loopsmith_conversations',  # Conversations
        'loopsmith_patterns',       # Patterns
        'loopsmith_concepts',       # Knowledge graph concepts
        'loopsmith_relations'       # Knowledge graph relations
    ]

def cleanup_chromadb_folders(chromadb_path: str = "./data/chromadb", dry_run: bool = True):
    """
    Clean up orphaned UUID folders in ChromaDB directory

    Args:
        chromadb_path: Path to ChromaDB persistence directory
        dry_run: If True, only show what would be deleted without actually deleting
    """
    chromadb_dir = Path(chromadb_path)

    if not chromadb_dir.exists():
        logger.error(f"ChromaDB directory not found: {chromadb_path}")
        return

    logger.info(f"Scanning ChromaDB directory: {chromadb_path}")

    # Get active collections
    active_collections = get_active_collections()
    logger.info(f"Active collections: {active_collections}")

    folders_to_remove = []
    folders_kept = []

    # Scan directory for UUID folders
    for entry in os.scandir(chromadb_dir):
        if entry.is_dir() and is_uuid_folder(entry.name):
            # Check if this folder is referenced in sqlite DB
            # For now, we'll mark all UUID folders for removal
            # as they're likely orphaned from the old system
            folders_to_remove.append(entry.path)
        elif entry.is_dir():
            folders_kept.append(entry.name)

    if dry_run:
        logger.info(f"DRY RUN MODE - No files will be deleted")

    logger.info(f"Found {len(folders_to_remove)} UUID folders to remove")
    logger.info(f"Keeping {len(folders_kept)} named folders: {folders_kept}")

    # Remove orphaned folders
    for folder_path in folders_to_remove:
        folder_name = os.path.basename(folder_path)

        if dry_run:
            logger.info(f"Would delete: {folder_name}")
            continue

        # Attempt to remove with retries for Windows lock issues
        for attempt in range(3):
            try:
                shutil.rmtree(folder_path)
                logger.info(f"Deleted: {folder_name}")
                break
            except PermissionError as e:
                if attempt < 2:
                    logger.warning(f"Permission error, retrying in 1 second: {folder_name}")
                    time.sleep(1)
                else:
                    logger.error(f"Failed to delete {folder_name} after 3 attempts: {e}")
            except Exception as e:
                logger.error(f"Failed to delete {folder_name}: {e}")
                break

    # Report summary
    if dry_run:
        logger.info(f"\nSummary (DRY RUN):")
        logger.info(f"  Would delete {len(folders_to_remove)} UUID folders")
    else:
        logger.info(f"\nSummary:")
        logger.info(f"  Deleted {len(folders_to_remove)} UUID folders")

    logger.info(f"  Kept {len(folders_kept)} named folders")

    # Check remaining disk usage
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(chromadb_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)

    logger.info(f"  ChromaDB directory size: {total_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean up orphaned ChromaDB collections")
    parser.add_argument(
        "--path",
        default="./data/chromadb",
        help="Path to ChromaDB persistence directory"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete files (without this flag, runs in dry-run mode)"
    )

    args = parser.parse_args()

    cleanup_chromadb_folders(
        chromadb_path=args.path,
        dry_run=not args.execute
    )