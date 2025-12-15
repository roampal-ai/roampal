import sys
import os
import logging
from pathlib import Path
import json
import asyncio

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = THIS_DIR if THIS_DIR.endswith("backend") else os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from modules.memory.chromadb_adapter import ChromaDBAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("clean_chromadb")

VDB_COLLECTION = "roampal_og_soul_fragments"
SOUL_LAYERS_DIR = Path(__file__).parent.parent / "data" / "og_data" / "arbitrary_store" / "soul_layers" / "og"
SUMMARIES_PATH = SOUL_LAYERS_DIR / "summaries.jsonl"
MODELS_PATH = SOUL_LAYERS_DIR / "models.jsonl"
QUOTES_PATH = SOUL_LAYERS_DIR / "quotes.jsonl"

def load_all_ids_from_jsonl(paths):
    all_ids = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "chunk_id" in data:
                        all_ids.add(data["chunk_id"])
                    if "chunk_id" in data and data.get("structured_summary"):
                        all_ids.add(data["chunk_id"] + "_summary")
                    if "name" in data:
                        all_ids.add(data["name"])
                    if "quote" in data:
                        all_ids.add(data["quote"])
                except Exception:
                    continue
    return all_ids

async def main():
    logger.info("--- ChromaDB Cleanup Starting ---")
    canonical_ids = load_all_ids_from_jsonl([SUMMARIES_PATH, MODELS_PATH, QUOTES_PATH])
    logger.info(f"Canonical IDs loaded: {len(canonical_ids)} unique IDs")

    # Initialize ChromaDB
    vdb = ChromaDBAdapter(persistence_directory=settings.paths.get_vector_db_dir("roampal"))
    await vdb.initialize(collection_name=VDB_COLLECTION)
    logger.info(f"ChromaDB client initialized for path: {getattr(vdb, 'db_path', '[unknown path]')}")
    logger.info(f"Getting or creating ChromaDB collection: '{VDB_COLLECTION}'...")
    logger.info(f"Collection '{VDB_COLLECTION}' is ready.")

    # List all vectors in collection
    all_vectors = vdb.list_all_ids()
    logger.info(f"Vector DB contains {len(all_vectors)} vectors")

    # Find stragglers
    extra_ids = set(all_vectors) - canonical_ids
    logger.info(f"Found {len(extra_ids)} orphaned vectors not in canonical data.")

    if extra_ids:
        # Actually delete from collection
        logger.info(f"Deleting {len(extra_ids)} orphaned vectors from ChromaDB...")
        vdb.delete_vectors(list(extra_ids))
        logger.info("Cleanup complete! All orphaned vectors removed.")
    else:
        logger.info("No orphaned vectors found. No action needed.")

if __name__ == "__main__":
    asyncio.run(main())
