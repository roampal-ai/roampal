"""
Tag Service — Noun tag extraction and matching for tag-routed retrieval.

v0.4.5: Replaces Knowledge Graph with simpler, faster tag-based routing.

Two extraction modes:
- Regex: Deterministic, <1ms, no dependencies. Used for migration and query matching.
- LLM: Sidecar/main LLM extraction. Gets common nouns (kickboxing, pottery class) that
  regex misses. Used for ongoing storage when available.

Query matching always uses regex (zero latency in search path).
"""

import json
import logging
import re
from typing import Callable, List, Optional, Set

logger = logging.getLogger(__name__)


# Common English words that start sentences but aren't proper nouns
_COMMON_SENTENCE_STARTERS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "i",
    "we",
    "he",
    "she",
    "they",
    "you",
    "my",
    "our",
    "his",
    "her",
    "there",
    "here",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "how",
    "why",
    "if",
    "but",
    "and",
    "or",
    "so",
    "yet",
    "for",
    "nor",
    "not",
    "no",
    "yes",
    "just",
    "also",
    "still",
    "even",
    "only",
    "after",
    "before",
    "since",
    "while",
    "during",
    "until",
    "once",
    "both",
    "each",
    "every",
    "all",
    "any",
    "some",
    "most",
    "many",
    "much",
    "few",
    "several",
    "other",
    "another",
    "such",
    "same",
    "however",
    "therefore",
    "meanwhile",
    "otherwise",
    "furthermore",
    "although",
    "because",
    "unless",
    "whether",
    "instead",
    "despite",
    "agreed",
    "asked",
    "confirmed",
    "explained",
    "found",
    "noted",
    "fixed",
    "added",
    "removed",
    "updated",
    "replaced",
    "checked",
    "went",
    "got",
    "made",
    "took",
    "came",
    "said",
    "told",
    "gave",
    "ran",
    "set",
    "put",
    "let",
    "cut",
    "run",
    "get",
    "see",
    "saw",
    "pretty",
    "really",
    "very",
    "quite",
    "rather",
    "always",
    "never",
    "often",
    "sometimes",
    "recently",
    "currently",
    "finally",
    "already",
    "overall",
    "basically",
    "essentially",
    "apparently",
    "generally",
    "stop",
    "start",
    "keep",
    "done",
    "doing",
    "being",
}

# Tags that are noise — meta-words, months, generic adjectives
_NOISE_TAGS = {
    # Meta/abstract terms
    "source",
    "answer",
    "details",
    "accuracy",
    "response",
    "question",
    "information",
    "content",
    "data",
    "system",
    "memory",
    "result",
    "user",
    "assistant",
    "context",
    "type",
    "value",
    "name",
    "list",
    "example",
    "note",
    "item",
    "thing",
    "way",
    "time",
    "part",
    "topic",
    "conversation",
    "discussion",
    "exchange",
    "summary",
    "key",
    "takeaway",
    "outcome",
    "score",
    "status",
    "update",
    "focus",
    "approach",
    "method",
    "process",
    "based",
    "level",
    "architecture",
    "benchmark",
    "release",
    "version",
    # Months/days
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    # Generic adjectives
    "old",
    "new",
    "big",
    "small",
    "long",
    "short",
    "high",
    "low",
    "good",
    "great",
    "nice",
    "bad",
    "best",
    "worst",
    "first",
    "last",
}

# Short acronyms that ARE valid tags
_VALID_SHORT_TAGS = {"ai", "ml", "uk", "us", "eu", "nz"}


def _is_adjective_not_place(word: str) -> bool:
    """Check if word is a nationality/language adjective, not a place name."""
    suffixes = ("ian", "ish", "ese", "ean", "ic")
    return any(word.endswith(s) for s in suffixes) and len(word) > 4


def _strip_file_extensions(word: str) -> str:
    """Remove common file extensions stuck to words."""
    return re.sub(
        r"\.(md|py|js|ts|json|yaml|yml|toml|txt|csv|html|css)$",
        "",
        word,
        flags=re.IGNORECASE,
    )


def _dedup_substrings(tags: List[str]) -> List[str]:
    """If 'ford mustang' is a tag, remove 'ford' and 'mustang' individually."""
    multi_word = [t for t in tags if " " in t]
    if not multi_word:
        return tags
    result = []
    for tag in tags:
        if " " not in tag:
            if any(tag in mw.split() for mw in multi_word):
                continue
        result.append(tag)
    return result


def extract_tags_regex(text: str) -> List[str]:
    """
    Extract noun tags from text using regex. No LLM needed.

    Extracts: proper nouns, multi-word capitalized sequences, quoted strings.
    Filters: noise words, nationality adjectives, file extensions, substrings.

    Returns: List of lowercase tags, sorted by specificity (longer first), max 8.
    """
    if not text or len(text) < 3:
        return []

    tags: Set[str] = set()

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # 1. Capitalized words — filter common starters for first word only
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = sent.split()
        for idx, word in enumerate(words):
            word = _strip_file_extensions(word)
            clean = re.sub(r"[^a-zA-Z'\-]", "", word)
            if not clean or not clean[0].isupper() or len(clean) < 2:
                continue
            lower = clean.lower()

            if idx == 0:
                if lower in _COMMON_SENTENCE_STARTERS:
                    continue

            if lower in _NOISE_TAGS:
                continue

            if _is_adjective_not_place(lower):
                continue

            tags.add(lower)

    # 2. Multi-word proper nouns (consecutive capitalized words)
    for match in re.finditer(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text):
        phrase = match.group(1).lower()
        if len(phrase) >= 4 and phrase not in _NOISE_TAGS:
            parts = phrase.split()
            if not all(p in _NOISE_TAGS or _is_adjective_not_place(p) for p in parts):
                tags.add(phrase)

    # 3. Quoted strings
    for match in re.finditer(r'"(.*?)"', text):
        quoted = match.group(1).strip().lower()
        if 2 <= len(quoted) <= 50 and quoted not in _NOISE_TAGS:
            tags.add(quoted)

    # 4. Final filter
    filtered = []
    for tag in tags:
        if len(tag) < 2:
            continue
        if re.match(r"^\d+$", tag):
            continue
        if len(tag) <= 2 and tag not in _VALID_SHORT_TAGS:
            continue
        filtered.append(tag)

    # 5. Deduplicate substrings
    filtered = _dedup_substrings(filtered)

    # Sort: longer (more specific) first, cap at 8
    filtered.sort(key=len, reverse=True)
    return filtered[:8]


class TagService:
    """
    Tag extraction and matching for tag-routed retrieval.

    Manages:
    - Noun tag extraction (regex + optional LLM)
    - Known tag index (in-memory, rebuilt from ChromaDB on init)
    - Query-to-tag matching (always regex, word-boundary safe)
    """

    def __init__(
        self, llm_extract_fn: Optional[Callable[[str], Optional[List[str]]]] = None
    ):
        """
        Args:
            llm_extract_fn: Optional callable(text) -> List[str] for LLM extraction.
                            If None or returns None, falls back to regex.
        """
        self._llm_extract_fn = llm_extract_fn
        self._known_tags: Set[str] = set()

    def set_llm_extract_fn(
        self, llm_extract_fn: Optional[Callable[[str], Optional[List[str]]]]
    ):
        """Update the LLM extraction function after initialization."""
        self._llm_extract_fn = llm_extract_fn

    # --- Extraction ---

    def extract_tags(self, text: str) -> List[str]:
        """
        Extract noun tags from text. Tries LLM first, returns [] on failure (benchmark-aligned).

        Returns: List of lowercase tag strings, max 8.
        """
        if self._llm_extract_fn:
            # v0.3.2 (Bug 5 guard): a sync call site can't execute an async
            # llm_extract_fn. Previously returned a coroutine that got passed
            # to _normalize_llm_tags, AttributeError/TypeError was caught at
            # DEBUG level, and the call site got [] with no visibility. Fail
            # loudly so future regressions surface instead of silently empty.
            import inspect
            if inspect.iscoroutinefunction(self._llm_extract_fn):
                logger.warning(
                    "TagService.extract_tags() called with an async llm_extract_fn — "
                    "sync path can't await it. Use extract_tags_async() instead. "
                    "Returning []."
                )
                return []
            try:
                tags = self._llm_extract_fn(text)
                if tags:  # tags is [] on LLM failure (benchmark behavior)
                    tags = self._normalize_llm_tags(tags)
                    self._known_tags.update(tags)
                    return tags
                # tags is [] → return [] (no regex fallback, benchmark-aligned)
                return []
            except Exception as e:
                logger.warning(f"LLM tag extraction failed: {e}")
                return []  # Empty list on exception (benchmark-aligned)

        # No LLM extract function configured → use regex (for migration/pre-sidecar)
        return self.extract_tags_regex_and_register(text)

    async def extract_tags_async(self, text: str) -> List[str]:
        """
        Async version of extract_tags for use in async contexts.

        Returns: List of lowercase tag strings, max 8.
        """
        if self._llm_extract_fn:
            try:
                # Check if llm_extract_fn is async
                import inspect

                if inspect.iscoroutinefunction(self._llm_extract_fn):
                    tags = await self._llm_extract_fn(text)
                else:
                    tags = self._llm_extract_fn(text)
                if tags:  # tags is [] on LLM failure (benchmark behavior)
                    tags = self._normalize_llm_tags(tags)
                    self._known_tags.update(tags)
                    return tags
                # tags is [] → return [] (no regex fallback, benchmark-aligned)
                return []
            except Exception as e:
                logger.debug(f"LLM tag extraction failed: {e}")
                return []  # Empty list on exception (benchmark-aligned)

        # No LLM extract function configured → use regex (for migration/pre-sidecar)
        return self.extract_tags_regex_and_register(text)

    def extract_tags_regex_and_register(self, text: str) -> List[str]:
        """Extract tags via regex and register them in known_tags."""
        tags = extract_tags_regex(text)
        self._known_tags.update(tags)
        return tags

    @staticmethod
    def _normalize_llm_tags(tags: List[str]) -> List[str]:
        """Normalize LLM-extracted tags: lowercase, dedup, filter noise, cap at 8."""
        seen = set()
        result = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            tag = tag.lower().strip()
            if len(tag) < 2 or tag in _NOISE_TAGS or tag in seen:
                continue
            seen.add(tag)
            result.append(tag)
        result.sort(key=len, reverse=True)
        return result[:8]

    # --- Query Matching ---

    def match_query_tags(self, query: str) -> List[str]:
        """
        Match query text against known tag index.

        Always uses regex (fast, no LLM latency in search path).
        Uses word-boundary matching to prevent "log" matching "logan".

        Returns: List of matched known tags, sorted by length (most specific first).
        """
        if not query or not self._known_tags:
            return []

        query_lower = query.lower()
        matches = []

        # Sort known tags by length descending (longer/more specific first)
        sorted_tags = sorted(self._known_tags, key=len, reverse=True)

        for tag in sorted_tags:
            if len(tag) < 2:
                continue
            # Word boundary match
            pattern = r"\b" + re.escape(tag) + r"\b"
            if re.search(pattern, query_lower):
                matches.append(tag)

        return matches[:8]

    # --- Known Tag Index ---

    def rebuild_known_tags(self, collections: dict):
        """
        Scan all ChromaDB collections and rebuild in-memory known_tags set.

        Called once during initialization. Fast — reads metadata only.
        """
        self._known_tags.clear()
        total = 0

        for coll_name, adapter in collections.items():
            try:
                collection_obj = adapter.collection
                if not collection_obj:
                    continue
                items = collection_obj.get(limit=10000, include=["metadatas"])
                for meta in items.get("metadatas", []):
                    if not meta:
                        continue
                    noun_tags_raw = meta.get("noun_tags", "")
                    if not noun_tags_raw:
                        continue
                    try:
                        if isinstance(noun_tags_raw, str):
                            tags = json.loads(noun_tags_raw)
                        else:
                            tags = noun_tags_raw
                        if isinstance(tags, list):
                            self._known_tags.update(
                                t for t in tags if isinstance(t, str)
                            )
                            total += len(tags)
                    except (json.JSONDecodeError, TypeError):
                        pass
            except Exception as e:
                logger.warning(f"Failed to scan {coll_name} for tags: {e}")

        logger.info(
            f"Rebuilt known tag index: {len(self._known_tags)} unique tags from {total} total"
        )

    def add_known_tags(self, tags: List[str]):
        """Add tags to the known index (called after storing a new memory)."""
        self._known_tags.update(tags)

    @property
    def known_tags(self) -> Set[str]:
        """Get a copy of the known tag set."""
        return self._known_tags.copy()

    @property
    def known_tag_count(self) -> int:
        return len(self._known_tags)
