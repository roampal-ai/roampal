from typing import Dict, Any, Optional

def format_fragment_for_ui(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes a raw result from a vector DB query and formats it into a
    standardized dictionary for the frontend UI.
    """
    metadata = result.get('metadata', {})
    distance = result.get('distance', 2.0)
    
    # Converts L2 distance to a 0-100 confidence score
    confidence = round((1 / (1 + distance)) * 100)
    
    return {
        "text": metadata.get("original_text", "N/A"),
        "confidence": confidence,
        "source": metadata.get("source_type", "Unknown"),
        "chunk_id": metadata.get("chunk_id") or metadata.get("learning_id") or result.get("id", "")
    }
