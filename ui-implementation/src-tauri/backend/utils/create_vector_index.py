import asyncio
import logging
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# --- Boilerplate to enable running script from project root ---
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
# -------------------------------------------------------------

from modules.embedding.embedding_service import EmbeddingService
from modules.memory.chromadb_adapter import ChromaDBAdapter
from modules.soul_manager.soul_layer_manager import SoulLayerManager
from app.dependencies_initializers import initialize_fragment_memory_adapter
from config.settings import settings

COLLECTION_NAME = "roampal_og_soul_fragments"
BATCH_SIZE = 64
GARBAGE_PHRASES = [
    "here are three direct and impactful quotes",
    "here are the named models/frameworks/methods",
    "(this quote highlights the importance"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_document(doc: Dict[str, Any], text_key: str) -> bool:
    text = doc.get(text_key)
    if not text or not isinstance(text, str) or not text.strip():
        return False
    lower_text = text.lower()
    if any(phrase in lower_text for phrase in GARBAGE_PHRASES):
        logger.warning(f"Skipping noisy document with ID {doc.get('chunk_id') or doc.get('quote_id') or doc.get('model_id')}: '{text[:50]}...'")
        return False
    return True

def get_doc_id(doc: Dict[str, Any], text: str, kind: str) -> str:
    # Try all possible id fields; fall back to a stable string composite
    for k in ["chunk_id", "summary_id", "quote_id", "model_id", "learning_id", "id"]:
        if doc.get(k):
            return str(doc[k])
    # Fallback: composite string
    return f"{kind}_{hash(text)}"

async def main(full_reindex: bool):
    logger.info("Starting Semantic Indexing Process for Roampal OG...")

    embedding_service = EmbeddingService()
    vector_db = ChromaDBAdapter(persistence_directory=settings.paths.get_vector_db_dir("roampal"))
    roampal_memory_adapter = await initialize_fragment_memory_adapter("roampal", settings)
    soul_manager = SoulLayerManager(memory_adapter=roampal_memory_adapter, shard_id="roampal")

    await vector_db.initialize(collection_name=COLLECTION_NAME)

    if full_reindex:
        logger.warning(f"FULL REINDEX requested. Deleting existing collection '{COLLECTION_NAME}'...")
        await asyncio.to_thread(vector_db.client.delete_collection, name=COLLECTION_NAME)
        await vector_db.initialize(collection_name=COLLECTION_NAME)
        logger.info("Collection deleted and re-initialized for a clean slate.")

    all_docs_to_index = []
    skipped_docs = 0

    logger.info("Loading documents from soul layer files...")
    summaries = await soul_manager.get_all_foundational_summaries()
    quotes = await soul_manager.get_all_quotes()
    models = await soul_manager.get_all_models()
    learnings = await soul_manager.get_all_learnings()

    embedding_meta = embedding_service.get_embedding_metadata

    # Summaries
    for doc in summaries:
        text = doc.get('structured_summary') or doc.get('summary_text')
        if is_valid_document(doc, 'structured_summary'):
            doc_id = get_doc_id(doc, text, "summary")
            all_docs_to_index.append({
                "id": doc_id,
                "text_to_embed": text,
                "metadata": {
                    "source_type": "summary",
                    "original_text": text,
                    "book_title": doc.get('book_title'),
                    "chunk_id": doc.get('chunk_id'),
                    **embedding_meta
                }
            })
        else:
            skipped_docs += 1

    # Quotes
    for doc in quotes:
        text = doc.get('text') or doc.get('quote')
        if is_valid_document(doc, 'text'):
            doc_id = get_doc_id(doc, text, "quote")
            all_docs_to_index.append({
                "id": doc_id,
                "text_to_embed": text,
                "metadata": {
                    "source_type": "quote",
                    "original_text": text,
                    "book_title": doc.get('book_title'),
                    "chunk_id": doc.get('chunk_id'),
                    **embedding_meta
                }
            })
        else:
            skipped_docs += 1

    # Models
    for doc in models:
        name = doc.get('name', '')
        desc = doc.get('description_snippet') or doc.get('description', '')
        model_text = f"Model: {name}. Description: {desc}"
        if name or desc:
            doc_id = get_doc_id(doc, model_text, "model")
            all_docs_to_index.append({
                "id": doc_id,
                "text_to_embed": model_text,
                "metadata": {
                    "source_type": "model",
                    "original_text": model_text,
                    "book_title": doc.get('book_title'),
                    "chunk_id": doc.get('chunk_id'),
                    **embedding_meta
                }
            })
        else:
            skipped_docs += 1

    # Learnings
    for doc in learnings:
        text = doc.get('learning_content') or doc.get('text')
        if text and isinstance(text, str) and text.strip():
            doc_id = get_doc_id(doc, text, "learning")
            all_docs_to_index.append({
                "id": doc_id,
                "text_to_embed": text,
                "metadata": {
                    "source_type": "learning",
                    "original_text": text,
                    "topic": doc.get('topic'),
                    **embedding_meta
                }
            })
        else:
            skipped_docs += 1

    logger.info(f"Found a total of {len(all_docs_to_index)} valid documents to process. Skipped {skipped_docs} noisy or empty docs.")

    for i in range(0, len(all_docs_to_index), BATCH_SIZE):
        batch = all_docs_to_index[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1} of {len(all_docs_to_index)//BATCH_SIZE + 1}...")
        ids_batch = [doc['id'] for doc in batch]
        texts_batch = [doc['text_to_embed'] for doc in batch]
        metadata_batch = [doc['metadata'] for doc in batch]
        try:
            # Use async for embedding, even if service is not truly async (won't hurt)
            vectors_batch = await asyncio.gather(*(embedding_service.embed_text(text) for text in texts_batch))
            await vector_db.upsert_vectors(
                ids=ids_batch,
                vectors=vectors_batch,
                metadatas=metadata_batch
            )
            logger.info(f"Batch {i//BATCH_SIZE + 1} upserted successfully.")
        except Exception as e:
            logger.error(f"Error processing batch {i//BATCH_SIZE + 1}: {e}", exc_info=True)

    logger.info("Semantic Indexing Process Completed Successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and store semantic vectors for Roampal OG's soul layer.")
    parser.add_argument(
        "--full-reindex",
        action="store_true",
        help="If set, deletes the existing collection before indexing to ensure a clean slate."
    )
    args = parser.parse_args()
    asyncio.run(main(full_reindex=args.full_reindex))
