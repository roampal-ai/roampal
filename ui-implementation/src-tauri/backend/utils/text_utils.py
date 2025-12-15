import re
from typing import List, Dict, Any, Set, Tuple, Optional
from config.settings import settings

# --- Original Content ---

def strip_fluff_phrases(text: str) -> str:
    lower_text = text.lower()
    for phrase in settings.text.fluff_phrases:
        if lower_text.startswith(phrase):
            return text[len(phrase):].lstrip(" ,.?!")
    return text

def approximate_token_count(text: str) -> int:
    """A simplified token counter based on splitting by spaces."""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())

def keyword_based_search(query_text: str, documents: List[Dict[str, Any]], text_key: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """A simple keyword search as a fallback for semantic search."""
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

# --- v0.2.5: Thinking Extraction (Clean Architecture - Single Responsibility) ---

# Regex pattern for thinking tags (supports <think> and <thinking> variants)
THINKING_TAG_PATTERN = re.compile(r'<think(?:ing)?>(.*?)</think(?:ing)?>', re.DOTALL)
# Strip pattern handles BOTH properly closed AND unclosed thinking blocks
THINKING_TAG_STRIP_PATTERN = re.compile(r'<think(?:ing)?>(.*?)</think(?:ing)?>|<think(?:ing)?>.*?(?=\n\n|$)', re.DOTALL)


def extract_thinking(response: str) -> Tuple[Optional[str], str]:
    """
    Extract thinking content from LLM response and return clean response.
    """
    if not response:
        return None, ""

    # Extract thinking content
    thinking_match = THINKING_TAG_PATTERN.search(response)
    thinking_content = thinking_match.group(1).strip() if thinking_match else None

    # Strip all thinking tags
    clean_response = THINKING_TAG_STRIP_PATTERN.sub('', response).strip()

    return thinking_content, clean_response


def extract_entities_for_action(text: str, action_type: Any) -> Dict[str, Any]:
    """Placeholder function for imports."""
    return {}
