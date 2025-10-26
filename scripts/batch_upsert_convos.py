import os
import json
from pathlib import Path
import asyncio

from backend.modules.soul_manager.soul_layer_manager import SoulLayerManager, upsert_to_chroma
from backend.modules.memory.chromadb_adapter import ChromaDBAdapter
from backend.modules.embedding.embedding_service import EmbeddingService

async def batch_upsert_conversations(convo_dir):
    vector_db = ChromaDBAdapter(persistence_directory="backend/data/shards/roampal/vector_store")
    embedding_service = EmbeddingService(model_name="all-MiniLM-L6-v2")
    upserted = 0

    for file in Path(convo_dir).glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    frag = json.loads(line)
                    # Use message text/content or whatever is the right key for your fragment
                    text = frag.get("content") or frag.get("text") or frag.get("message") or ""
                    if not text:
                        continue
                    id = frag.get("chunk_id") or frag.get("id") or str(hash(text))
                    meta = {k: v for k, v in frag.items() if k not in ("content", "text", "message")}
                    # Flatten metadata (remove nested dicts) for Chroma
                    meta = {k: (str(v) if isinstance(v, dict) else v) for k, v in meta.items()}
                    vector = embedding_service.embed_text(text)
                    await vector_db.upsert_vectors(
                        ids=[id],
                        vectors=[vector],
                        metadatas=[meta]
                    )
                    upserted += 1
                except Exception as e:
                    print(f"Error upserting from file {file}: {e}")
    print(f"Upserted {upserted} conversation fragments from {convo_dir}.")

if __name__ == "__main__":
    import sys
    convo_dir = "C:/RoampalAI/backend/data/og_data/arbitrary_store/conversations"
    asyncio.run(batch_upsert_conversations(convo_dir))
