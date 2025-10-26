"""
Build semantic knowledge graphs with meanings for all users and shards
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from pathlib import Path
from services.knowledge_graph_builder import KnowledgeGraphBuilder
import chromadb
from chromadb import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_semantic_graphs():
    """Build semantic knowledge graphs for all shards."""
    
    builder = KnowledgeGraphBuilder()
    
    if not builder.chroma_client:
        logger.error("ChromaDB not available. Cannot build knowledge graphs.")
        return
    
    # Get all collections
    try:
        collections = builder.chroma_client.list_collections()
        logger.info(f"Found {len(collections)} collections")
        
        # Group collections by user and shard
        user_shards = {}
        for collection in collections:
            name = collection.name
            
            # Parse collection name
            if name.startswith('user_') and '_fragments' in name:
                parts = name.replace('user_', '').replace('_fragments', '').rsplit('_', 1)
                if len(parts) == 2:
                    user_id, shard_id = parts
                    if user_id not in user_shards:
                        user_shards[user_id] = set()
                    user_shards[user_id].add(shard_id)
        
        logger.info(f"Found {len(user_shards)} users with shards")
        
        # Build semantic graphs for each user/shard
        for user_id, shards in user_shards.items():
            for shard_id in shards:
                logger.info(f"Building semantic graph for {user_id}/{shard_id}")
                
                try:
                    # Get private fragments
                    private_collection_name = f"user_{user_id}_{shard_id}_fragments"
                    private_collection = builder.chroma_client.get_collection(private_collection_name)
                    
                    # Get fragments
                    private_items = private_collection.get(limit=100, include=['documents', 'metadatas'])
                    
                    if len(private_items['ids']) > 0:
                        private_fragments = []
                        for i in range(len(private_items['ids'])):
                            private_fragments.append({
                                'id': private_items['ids'][i],
                                'text': private_items['documents'][i] if private_items['documents'] else '',
                                'metadata': private_items['metadatas'][i] if private_items['metadatas'] else {},
                                'score': (private_items['metadatas'][i] or {}).get('composite_score', 0.5),
                                'created_at': (private_items['metadatas'][i] or {}).get('created_at', '')
                            })
                        
                        # Build semantic graph
                        logger.info(f"Building semantic graph from {len(private_fragments)} fragments")
                        private_graph = builder.build_semantic_graph_from_fragments(
                            private_fragments, 
                            is_private=True,
                            use_semantic=True
                        )
                        
                        # Save graph
                        builder.save_graph(user_id, shard_id, private_graph, is_private=True)
                        
                        logger.info(f"Saved semantic graph for {user_id}/{shard_id}: "
                                  f"{len(private_graph['concepts'])} concepts with meanings")
                    
                except Exception as e:
                    logger.error(f"Failed to build graph for {user_id}/{shard_id}: {e}")
                
                # Build global graph
                try:
                    global_collection_name = f"global_{shard_id}_fragments"
                    global_collection = builder.chroma_client.get_collection(global_collection_name)
                    
                    global_items = global_collection.get(limit=50, include=['documents', 'metadatas'])
                    
                    if len(global_items['ids']) > 0:
                        global_fragments = []
                        for i in range(len(global_items['ids'])):
                            global_fragments.append({
                                'id': global_items['ids'][i],
                                'text': global_items['documents'][i] if global_items['documents'] else '',
                                'metadata': global_items['metadatas'][i] if global_items['metadatas'] else {},
                                'score': (global_items['metadatas'][i] or {}).get('composite_score', 0.5),
                                'created_at': (global_items['metadatas'][i] or {}).get('created_at', '')
                            })
                        
                        logger.info(f"Building global semantic graph from {len(global_fragments)} fragments")
                        global_graph = builder.build_semantic_graph_from_fragments(
                            global_fragments,
                            is_private=False,
                            use_semantic=True
                        )
                        
                        builder.save_graph('global', shard_id, global_graph, is_private=False)
                        
                        logger.info(f"Saved global semantic graph for {shard_id}: "
                                  f"{len(global_graph['concepts'])} concepts with meanings")
                        
                except Exception as e:
                    logger.debug(f"No global collection for {shard_id} or error: {e}")
                    
    except Exception as e:
        logger.error(f"Failed to build semantic graphs: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("Starting semantic knowledge graph building...")
    build_semantic_graphs()
    logger.info("Semantic knowledge graph building complete!")