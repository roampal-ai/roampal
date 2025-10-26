"""
Script to build knowledge graphs from existing memory fragments
Run: python backend/scripts/build_knowledge_graphs.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.knowledge_graph_builder import KnowledgeGraphBuilder
import chromadb
from chromadb import Settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Build knowledge graphs for all shards and users"""
    
    kg_builder = KnowledgeGraphBuilder()
    
    if not kg_builder.chroma_client:
        logger.error("Could not connect to ChromaDB. Make sure it's running on port 8003")
        return
    
    # Get all collections
    try:
        collections = kg_builder.chroma_client.list_collections()
        
        # Track unique shard/user combinations
        processed = set()
        shards = set()
        users = set()
        
        for collection in collections:
            name = collection.name
            
            # Parse collection names
            if name.startswith("user_") and "_fragments" in name:
                # Format: user_{user_id}_{shard_id}_fragments
                parts = name.replace("_fragments", "").split("_")
                if len(parts) >= 3:
                    user_id = "_".join(parts[1:-1])  # Handle user IDs with underscores
                    shard_id = parts[-1]
                    
                    users.add(user_id)
                    shards.add(shard_id)
                    
                    key = f"{user_id}:{shard_id}"
                    if key not in processed:
                        logger.info(f"Building graphs for user={user_id}, shard={shard_id}")
                        result = kg_builder.update_knowledge_graphs(user_id, shard_id)
                        logger.info(f"  Result: {result}")
                        processed.add(key)
                        
            elif name.startswith("global_") and "_fragments" in name:
                # Format: global_{shard_id}_fragments
                parts = name.replace("_fragments", "").split("_")
                if len(parts) >= 2:
                    shard_id = parts[-1]
                    shards.add(shard_id)
        
        # Build global graphs for all shards
        for shard_id in shards:
            logger.info(f"Ensuring global graph exists for shard={shard_id}")
            # Use a dummy user to trigger global graph building
            kg_builder.update_knowledge_graphs("system", shard_id)
        
        logger.info(f"\nSummary:")
        logger.info(f"  Shards processed: {', '.join(shards)}")
        logger.info(f"  Users processed: {', '.join(users)}")
        logger.info(f"  Total graphs built: {len(processed) + len(shards)}")
        
    except Exception as e:
        logger.error(f"Error building knowledge graphs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()