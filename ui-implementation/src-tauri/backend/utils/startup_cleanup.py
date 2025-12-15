"""
Startup cleanup utilities for managing soft-deleted shards and file locks
"""
import os
import shutil
import logging
import time
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


async def cleanup_soft_deleted_shards() -> Dict[str, any]:
    """
    Clean up shards that were soft-deleted in previous sessions.
    This runs on startup when file locks should be released.
    Also cleans up ChromaDB collections for deleted shards.
    """
    results = {
        "cleaned": [],
        "failed": [],
        "skipped": [],
        "chromadb_collections_deleted": []
    }
    
    # Standardized path
    shards_dir = Path("backend/data/shards")
    if not shards_dir.exists():
        logger.warning(f"Shards directory not found at 'backend/data/shards'")
        return results
    
    for shard_dir in shards_dir.iterdir():
        if not shard_dir.is_dir():
            continue
            
        # Check for deletion markers
        deletion_marker = shard_dir / "SHARD_DELETED.txt"
        if deletion_marker.exists():
            shard_name = shard_dir.name
            logger.info(f"Found soft-deleted shard: {shard_name}")
            
            # Delete ChromaDB collections for this shard first
            collections_deleted = await delete_chromadb_collections_for_shard(shard_name)
            if collections_deleted:
                results["chromadb_collections_deleted"].extend(collections_deleted)
            
            try:
                # Attempt full deletion now that server was restarted
                shutil.rmtree(shard_dir)
                results["cleaned"].append(shard_name)
                logger.info(f"Successfully cleaned up soft-deleted shard: {shard_name}")
            except PermissionError:
                # Still locked, skip for now
                results["skipped"].append(shard_name)
                logger.warning(f"Shard {shard_name} still has locked files, skipping cleanup")
            except Exception as e:
                results["failed"].append({
                    "shard": shard_name,
                    "error": str(e)
                })
                logger.error(f"Failed to clean up {shard_name}: {e}")
    
    return results


async def disconnect_chromadb_for_shard(shard_name: str, app_state) -> bool:
    """
    Disconnect ChromaDB connections for a specific shard to allow deletion.
    """
    try:
        # Check if shard has a multi-tier adapter
        adapter_attr = f"{shard_name}_vector_adapter"
        if hasattr(app_state, adapter_attr):
            adapter = getattr(app_state, adapter_attr)
            
            # Close global DB connection
            if hasattr(adapter, 'global_db') and adapter.global_db:
                try:
                    # ChromaDB doesn't have explicit close, but we can del the reference
                    del adapter.global_db
                    logger.info(f"Disconnected global ChromaDB for {shard_name}")
                except Exception as e:
                    logger.warning(f"Could not disconnect global DB: {e}")
            
            # Close all private DB connections
            if hasattr(adapter, 'private_dbs'):
                for user_id in list(adapter.private_dbs.keys()):
                    try:
                        del adapter.private_dbs[user_id]
                        logger.info(f"Disconnected private ChromaDB for {shard_name} user {user_id}")
                    except Exception as e:
                        logger.warning(f"Could not disconnect private DB for user {user_id}: {e}")
                adapter.private_dbs.clear()
            
            # Remove the adapter itself
            delattr(app_state, adapter_attr)
            logger.info(f"Removed vector adapter for {shard_name}")
            
        # Also check for any book processor
        book_processor_attr = f"{shard_name}_book_processor"
        if hasattr(app_state, book_processor_attr):
            processor = getattr(app_state, book_processor_attr)
            # Release any references to memory adapter
            if hasattr(processor, 'memory_adapter'):
                processor.memory_adapter = None
            delattr(app_state, book_processor_attr)
            logger.info(f"Removed book processor for {shard_name}")
            
        # Check for soul layer manager
        soul_manager_attr = f"{shard_name}_soul_layer_manager"
        if hasattr(app_state, soul_manager_attr):
            delattr(app_state, soul_manager_attr)
            logger.info(f"Removed soul layer manager for {shard_name}")
            
        # Give OS time to release file handles
        await asyncio.sleep(0.5)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to disconnect ChromaDB for {shard_name}: {e}")
        return False


async def attempt_online_deletion(shard_name: str, app_state) -> Dict[str, any]:
    """
    Attempt to delete a shard while the system is online by disconnecting its resources first.
    Also deletes ChromaDB collections from the server.
    """
    result = {
        "success": False,
        "shard": shard_name,
        "steps": []
    }
    
    # Step 1: Disconnect ChromaDB and other resources
    disconnected = await disconnect_chromadb_for_shard(shard_name, app_state)
    result["steps"].append({
        "action": "disconnect_chromadb",
        "success": disconnected
    })
    
    if not disconnected:
        result["message"] = "Could not disconnect ChromaDB connections"
        return result
    
    # Step 2: Delete ChromaDB collections from server
    collections_deleted = await delete_chromadb_collections_for_shard(shard_name)
    result["steps"].append({
        "action": "delete_chromadb_collections",
        "success": len(collections_deleted) > 0 or True,  # Success even if no collections found
        "collections_deleted": collections_deleted
    })
    
    # Step 3: Try to delete the shard directory
    shard_dir = Path(f"data/shards/{shard_name}")
    if not shard_dir.exists():
        shard_dir = Path(f"backend/data/shards/{shard_name}")
    
    try:
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
            result["success"] = True
            result["message"] = f"Successfully deleted shard {shard_name}"
            result["steps"].append({
                "action": "delete_directory",
                "success": True
            })
            logger.info(f"Successfully deleted shard directory: {shard_name}")
        else:
            result["success"] = True
            result["message"] = f"Shard {shard_name} directory not found"
            
    except PermissionError as e:
        # If still locked, mark for soft deletion
        try:
            marker_file = shard_dir / "SHARD_DELETED.txt"
            with open(marker_file, 'w') as f:
                f.write(f"SOFT DELETION - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Will be cleaned up on next server restart\n")
            
            result["success"] = True
            result["message"] = f"Shard {shard_name} marked for deletion on next restart"
            result["steps"].append({
                "action": "soft_delete_marker",
                "success": True
            })
            
        except Exception as marker_error:
            result["message"] = f"Could not mark shard for deletion: {marker_error}"
            result["steps"].append({
                "action": "soft_delete_marker",
                "success": False,
                "error": str(marker_error)
            })
            
    except Exception as e:
        result["message"] = f"Failed to delete shard: {e}"
        result["steps"].append({
            "action": "delete_directory",
            "success": False,
            "error": str(e)
        })
    
    return result


import asyncio


async def delete_chromadb_collections_for_shard(shard_name: str) -> List[str]:
    """
    Delete all ChromaDB collections associated with a shard.
    This includes both global and private user collections.
    
    Returns:
        List of collection names that were deleted
    """
    deleted_collections = []
    
    try:
        # Check if ChromaDB server mode is enabled
        use_server = os.environ.get('CHROMADB_USE_SERVER', 'true').lower() == 'true'
        
        if use_server:
            import chromadb
            from chromadb.config import Settings
            
            # Try to reuse existing client or create new one
            client = None
            try:
                # First, try to import and reuse the memory visualization router's client
                from app.routers.memory_visualization_router import get_chroma_client
                client = get_chroma_client()
                logger.info(f"Reusing existing ChromaDB client for shard {shard_name} deletion")
            except Exception as e:
                logger.debug(f"Could not reuse existing client: {e}")
            
            if client is None:
                # Reset the chromadb system to avoid conflicts
                try:
                    chromadb.api.client.SharedSystemClient.clear()
                except:
                    pass
                
                # Create new client with proper settings
                client = chromadb.HttpClient(
                    host="localhost", 
                    port=8003,
                    settings=Settings(anonymized_telemetry=False)
                )
            
            # Get all collections
            all_collections = client.list_collections()
            
            # Find collections for this shard
            for collection in all_collections:
                collection_name = collection.name
                
                # Check if this collection belongs to the shard
                # Format: global_{shard_name}_fragments or user_{id}_{shard_name}_fragments
                # Also check for other shard-specific collections like conversations and knowledge_graph
                if (f"global_{shard_name}_" in collection_name or
                    f"_{shard_name}_fragments" in collection_name or
                    f"_{shard_name}_conversations" in collection_name or
                    f"_{shard_name}_knowledge_graph" in collection_name):
                    
                    try:
                        client.delete_collection(name=collection_name)
                        deleted_collections.append(collection_name)
                        logger.info(f"Deleted ChromaDB collection: {collection_name}")
                    except Exception as e:
                        logger.warning(f"Could not delete collection {collection_name}: {e}")
            
            if deleted_collections:
                logger.info(f"Deleted {len(deleted_collections)} ChromaDB collections for shard {shard_name}")
            else:
                logger.info(f"No ChromaDB collections found for shard {shard_name}")
                
        else:
            logger.info("ChromaDB server mode not enabled, skipping collection cleanup")
            
    except Exception as e:
        logger.error(f"Error deleting ChromaDB collections for shard {shard_name}: {e}")
    
    return deleted_collections