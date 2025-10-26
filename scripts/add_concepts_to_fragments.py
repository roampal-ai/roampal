"""
Script to add concepts to existing fragments in ChromaDB
This retroactively processes fragments that were stored before concept extraction was implemented
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from services.concept_extractor import ConceptExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_collection(client, collection_name):
    """Process a collection and add concepts to fragments"""
    try:
        collection = client.get_collection(collection_name)
        
        # Get all fragments
        results = collection.get(
            include=["documents", "metadatas"],
            limit=1000  # Process up to 1000 fragments
        )
        
        if not results['ids']:
            logger.info(f"No fragments in {collection_name}")
            return 0
        
        extractor = ConceptExtractor()
        updated = 0
        
        for i in range(len(results['ids'])):
            fragment_id = results['ids'][i]
            document = results['documents'][i] if results['documents'] else ""
            metadata = results['metadatas'][i] if results['metadatas'] else {}
            
            # Skip if already has concepts
            if 'concepts' in metadata and metadata['concepts']:
                continue
            
            # Extract concepts
            concepts = extractor.extract_concepts_light(document, max_concepts=10)
            
            if concepts:
                # Update metadata with concepts (as JSON string for ChromaDB)
                import json
                metadata['concepts'] = json.dumps(concepts)  # Store as JSON string
                metadata['concept_count'] = len(concepts)
                metadata['needs_semantic'] = True
                metadata['semantic_processed'] = False
                
                # Update in ChromaDB
                collection.update(
                    ids=[fragment_id],
                    metadatas=[metadata]
                )
                updated += 1
                logger.info(f"Added {len(concepts)} concepts to fragment {fragment_id[:8]}...")
        
        logger.info(f"Updated {updated} fragments in {collection_name}")
        return updated
        
    except Exception as e:
        logger.error(f"Error processing {collection_name}: {e}")
        return 0

def main():
    # Connect to ChromaDB
    client = chromadb.HttpClient(host="localhost", port=8003)
    
    # Get all collections
    collections = client.list_collections()
    
    total_updated = 0
    for collection in collections:
        if 'fragments' in collection.name:
            logger.info(f"Processing collection: {collection.name}")
            updated = process_collection(client, collection.name)
            total_updated += updated
    
    logger.info(f"Total fragments updated: {total_updated}")

if __name__ == "__main__":
    main()