"""
Sidecar-backed TagService LLM extractor factory.

v0.3.2 (Bug 4): Produces an async wrapper that reads
`app_state.sidecar_client` / `sidecar_model` at CALL time. Prior
inline closure at main.py captured boot-time values into local
vars — after any `/sidecar/set`, `/sidecar/mirror`, or
`/switch` (when `mirror_chat=True`) update, the closure kept
invoking the stale client while app.state already held the new
one. Calls failed silently (DEBUG-level log) and every post-swap
memory landed without LLM-extracted noun_tags.

Factoring this into a helper makes the pattern unit-testable: we
can mutate the fake app_state and assert the wrapper sees the
new values on the next call.
"""

import logging
from typing import List

from modules.memory.sidecar_service import extract_noun_tags

logger = logging.getLogger(__name__)


def make_llm_tag_extractor(app_state):
    """Return an async `extract(text) -> List[str]` closure that
    dynamically reads `app_state.sidecar_client` and
    `app_state.sidecar_model` on each invocation.

    Returns [] (benchmark-aligned, no regex fallback) when sidecar
    is unavailable or the extraction call raises. Failures log at
    WARNING so silent regressions surface at the default log
    level.
    """

    async def async_llm_tag_extractor(text: str) -> List[str]:
        client = getattr(app_state, "sidecar_client", None)
        model = getattr(app_state, "sidecar_model", None)
        if not client or not model:
            return []
        try:
            tags = await extract_noun_tags(
                text=text,
                client=client,
                model=model,
            )
            return tags if tags else []
        except Exception as e:
            logger.warning(f"LLM tag extraction failed: {e}")
            return []

    return async_llm_tag_extractor
