import re
from typing import List, Dict, Any, Set
from backend.config.settings import settings

# --- Original Content ---

def strip_fluff_phrases(text: str) -> str:
    # Moved to settings.text.fluff_phrases
    # fluff_phrases = [
    #     "in your opinion", "what do you think about", "can you tell me", "i'd like to know",
    #     "could you please explain", "tell me more about", "i'm curious about", "what are your thoughts on",
    #     "can we discuss", "let's talk about"
    # ]
    lower_text = text.lower()
    for phrase in settings.text.fluff_phrases:
        if lower_text.startswith(phrase):
            # Remove the phrase and any leading/trailing whitespace
            return text[len(phrase):].lstrip(" ,.?!")
    return text

# Moved to settings.text.english_stop_words
# ENGLISH_STOP_WORDS: Set[str] = {
#     "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
#     "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
#     "can", "did", "do", "does", "doing", "down", "during",
#     "each", "few", "for", "from", "further",
#     "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
#     "i", "if", "in", "into", "is", "it", "its", "itself",
#     "just", "know",
#     "me", "more", "most", "my", "myself",
#     "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
#     "s", "same", "she", "should", "so", "some", "such",
#     "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
#     "under", "until", "up",
#     "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
#     "you", "your", "yours", "yourself", "yourselves"
# }

def approximate_token_count(text: str) -> int:
    """A simplified token counter based on splitting by spaces."""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())

def keyword_based_search(query_text: str, documents: List[Dict[str, Any]], text_key: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    A simple keyword search as a fallback for semantic search.
    """
    query_keywords = {word for word in query_text.lower().split() if word not in settings.text.english_stop_words}
    if not query_keywords:
        return []

    scored_docs = []
    for doc in documents:
        doc_text = doc.get(text_key, "").lower()
        if not doc_text:
            continue
        
        matched_keywords = query_keywords.intersection(doc_text.split())
        if matched_keywords:
            score = len(matched_keywords)
            scored_docs.append({"doc": doc, "score": score})

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in scored_docs[:top_k]]

# --- New function stub to fix the ImportError ---

def extract_entities_for_action(text: str, action_type: Any) -> Dict[str, Any]:
    """
    Placeholder function to extract entities from text for a given action.
    The indexing script only needs this to exist for imports to work; its logic is not used during indexing.
    """
    # A real implementation would use regex or NLP to parse the text.
    # For now, returning an empty dict is sufficient to resolve the error.
    return {}