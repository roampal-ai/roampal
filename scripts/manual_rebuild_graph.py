"""
Manual script to rebuild knowledge graph
Run this to rebuild the knowledge graph for a shard
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.knowledge_graph_builder import KnowledgeGraphBuilder
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_graph(user_id="system", shard_id="tester"):
    """Rebuild knowledge graph for a user and shard"""
    
    logger.info(f"Rebuilding knowledge graph for user={user_id}, shard={shard_id}")
    
    builder = KnowledgeGraphBuilder()
    
    # Build private graph
    logger.info("Building private knowledge graph...")
    private_graph = builder.build_graph_from_chromadb(user_id, shard_id, is_private=True)
    
    # Build global graph
    logger.info("Building global knowledge graph...")
    global_graph = builder.build_graph_from_chromadb(user_id, shard_id, is_private=False)
    
    # Print results
    print("\n=== Knowledge Graph Rebuilt ===")
    print(f"Private concepts: {len(private_graph.get('concepts', {}))}")
    print(f"Private relations: {len(private_graph.get('relations', []))}")
    print(f"Global concepts: {len(global_graph.get('concepts', {}))}")
    print(f"Global relations: {len(global_graph.get('relations', []))}")
    
    # Show sample concepts
    all_concepts = list(private_graph.get('concepts', {}).keys())[:10]
    all_concepts.extend(list(global_graph.get('concepts', {}).keys())[:10])
    
    if all_concepts:
        print(f"\nSample concepts found:")
        for concept in all_concepts[:20]:
            print(f"  - {concept}")
    
    return {
        "private": private_graph,
        "global": global_graph
    }

if __name__ == "__main__":
    # Get parameters from command line or use defaults
    user_id = sys.argv[1] if len(sys.argv) > 1 else "system"
    shard_id = sys.argv[2] if len(sys.argv) > 2 else "tester"
    
    rebuild_graph(user_id, shard_id)