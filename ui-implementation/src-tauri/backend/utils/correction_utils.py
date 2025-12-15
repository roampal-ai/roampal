import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from config.settings import settings  # Import settings for corrections_jsonl_path

logger = logging.getLogger(__name__)

def log_correction(chunk_id: str, user_prompt: str, reason: str):
    entry = {
        "chunk_id": chunk_id,
        "user_prompt": user_prompt,
        "reason": reason
    }
    settings.paths.corrections_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.paths.corrections_jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info(f"Logged correction for chunk {chunk_id}")

def load_corrections() -> List[Dict[str, Any]]:
    if not settings.paths.corrections_jsonl_path.exists():
        return []
    with open(settings.paths.corrections_jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_penalized_chunks() -> set:
    return {entry["chunk_id"] for entry in load_corrections()}