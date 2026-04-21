"""
Sidecar Service — Background LLM for exchange scoring and fact extraction.

v0.3.1.3: 2-call architecture with client locking and retry queue.
- Client locking prevents concurrent model loading (same provider/model)
- Retry queue with exponential backoff for failed tasks
- Shared client support to avoid GPU memory duplication

Live sidecar operations (2 LLM calls):
1. score_exchange() — Summary + outcome + memory scores
2. extract_facts() — Dedicated atomic fact extraction

Utility (kept for migration tool):
- extract_noun_tags() — LLM-based tag extraction (not used in live flow)
- summarize_only() — Retroactive summarization of long memories
"""

import json
import logging
import re
import asyncio
from typing import Any, Dict, List, Optional

from .sidecar_queue import execute_with_client_lock, queue_sidecar_retry

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text that may contain markdown fences or other wrapping."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (```json ... ```)
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find any JSON object in the text (greedy — handles nested braces)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


async def _call_llm(
    prompt: str,
    client: Any,
    model: str,
    system_prompt: str = "You are part of a memory system. Return ONLY valid JSON. No other text. Be concise.",
) -> Optional[Dict[str, Any]]:
    """Call LLM via OllamaClient and extract JSON response."""
    try:
        response = await client.generate_response(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        )
        if not response:
            return None
        return _extract_json(response)
    except Exception as e:
        logger.warning(f"Sidecar LLM call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Scoring prompt — summary + outcome + memory_scores only
# ---------------------------------------------------------------------------


def _build_scoring_prompt(
    user_msg: str,
    assistant_msg: str,
    followup: str,
    memories: List[Dict[str, str]],
) -> str:
    """Build scoring prompt — summary + outcome + memory_scores only.

    v0.3.1.2: Tags extracted server-side via regex (matching core v0.4.8).
    Facts extracted via dedicated extract_facts() call.
    Small models produce better results with focused single-task prompts.
    """

    # Memory section (only if memories present)
    memory_section = ""
    memory_score_template = ""
    memory_instructions = ""

    if memories:
        memory_lines = "\n".join(f'- {m["id"]}: "{m["content"]}"' for m in memories)
        memory_section = f"\nThese memories were injected into your context for this exchange:\n{memory_lines}\n"

        # Build memory_scores template with placeholder per memory
        score_entries = ", ".join(
            f'"{m["id"]}": "<worked|failed|partial|unknown>"' for m in memories
        )
        memory_score_template = f', "memory_scores": {{{score_entries}}}'

        memory_instructions = """
MEMORY SCORES: For each memory, judge based on topic relevance and exchange outcome.
1. Memory is NOT about the topic discussed -> "unknown"
2. Memory IS about the topic AND outcome is "worked" -> "worked"
3. Memory IS about the topic AND outcome is "failed":
   - Your response echoed/relied on info from this memory -> "failed"
   - The failure seems unrelated to this memory's content -> "unknown"
4. Memory IS about the topic AND outcome is "partial" -> "partial"
5. Your response contradicts what the memory says and the exchange worked -> "unknown"
6. Memory contains good advice/instructions the response IGNORED (didn't follow) -> "unknown" not "failed". "failed" means the memory's content was WRONG and caused a bad response, not that the model failed to follow good advice.
7. When in doubt -> "unknown"
"""

    prompt = f"""You are part of a memory system for an AI assistant. You ARE the main AI in this system — write as if you are making a note for your future self.

The user said:
"{user_msg[:8000]}"

You responded:
"{assistant_msg[:8000]}"

The user then followed up with:
"{followup[:4000]}"
{memory_section}
Respond with ONLY a JSON object:
{{ "summary": "<~300 chars>", "outcome": "<worked|failed|partial|unknown>"{memory_score_template} }}

SUMMARY (under 300 chars): Capture what happened AND what changed. Summaries provide context and continuity — the story behind the facts.
- Include names, topics, and the flow of the conversation
- Note corrections, decisions, and new information alongside the context
- Help future retrieval understand WHY something matters, not just WHAT
BAD: "User and assistant had a conversation" (empty, no content)
BAD: "Temperature is 350F" (that's a fact, not a summary)
GOOD: "User corrected the baking temp from 375F to 350F while adapting the recipe for a convection oven — first attempt burned the edges"
GOOD: "Discussed switching API from REST to GraphQL after mobile team reported nested query issues with the current setup"

OUTCOME: Based on the user's follow-up:
- worked: user confirmed, moved on, or was satisfied
- failed: user corrected you, got frustrated, or asked to redo
- partial: helped but incomplete or needed adjustment
- unknown: no clear signal
{memory_instructions}"""

    return prompt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def score_exchange_with_retry(
    user_msg: str,
    assistant_msg: str,
    followup: str,
    memories: List[Dict[str, str]],
    client: Any,
    model: str,
    doc_id: str = "unknown",
) -> Optional[Dict[str, Any]]:
    """
    Sidecar scoring call with client locking and retry queue.

    Wraps the actual score_exchange function with:
    1. Client locking to prevent concurrent model loading
    2. Automatic retry queuing on failure

    Args:
        user_msg: User's message from the exchange being scored
        assistant_msg: Assistant's response from the exchange being scored
        followup: User's NEXT message (the signal for outcome detection)
        memories: Cached memories from the exchange [{id, content}, ...]
        client: OllamaClient instance
        model: Sidecar model name
        doc_id: Document ID for tracking retries

    Returns:
        {summary, outcome, memory_scores} or None on failure
    """

    async def _score_task():
        return await score_exchange(
            user_msg=user_msg,
            assistant_msg=assistant_msg,
            followup=followup,
            memories=memories,
            client=client,
            model=model,
        )

    result = await execute_with_client_lock(
        client=client, task_func=_score_task, task_name=f"score_exchange_{doc_id}"
    )

    if result is None:
        # Queue for retry
        task_data = {
            "task_type": "score_exchange",
            "doc_id": doc_id,
            "user_msg": user_msg,
            "assistant_msg": assistant_msg,
            "followup": followup,
            "memories": memories,
            "client": client,
            "model": model,
        }
        queue_sidecar_retry(task_data, "Task returned None")

    return result


async def score_exchange(
    user_msg: str,
    assistant_msg: str,
    followup: str,
    memories: List[Dict[str, str]],
    client: Any,
    model: str,
) -> Optional[Dict[str, Any]]:
    """
    Sidecar scoring call — summary + outcome + memory_scores.

    v0.3.1.2: Tags extracted server-side via regex (matching core v0.4.8).
    Facts extracted via dedicated extract_facts() call.

    Args:
        user_msg: User's message from the exchange being scored
        assistant_msg: Assistant's response from the exchange being scored
        followup: User's NEXT message (the signal for outcome detection)
        memories: Cached memories from the exchange [{id, content}, ...]
        client: OllamaClient instance
        model: Sidecar model name

    Returns:
        {summary, outcome, memory_scores} or None on failure
    """
    prompt = _build_scoring_prompt(
        user_msg=user_msg,
        assistant_msg=assistant_msg[:8000],
        followup=followup[:4000],
        memories=memories,
    )

    result = await _call_llm(prompt, client, model)
    if not result:
        return None

    # Validate and clean fields
    summary = result.get("summary", "")
    if not isinstance(summary, str) or len(summary) < 10:
        return None  # Summary is required

    outcome = result.get("outcome", "unknown")
    if outcome not in ("worked", "failed", "partial", "unknown"):
        outcome = "unknown"

    # Clean memory_scores
    memory_scores = {}
    raw_scores = result.get("memory_scores", {})
    if isinstance(raw_scores, dict):
        for mem_id, score in raw_scores.items():
            if isinstance(score, str) and score in (
                "worked",
                "failed",
                "partial",
                "unknown",
            ):
                memory_scores[str(mem_id)] = score
            else:
                memory_scores[str(mem_id)] = "unknown"

    return {
        "summary": summary[:2000],
        "outcome": outcome,
        "memory_scores": memory_scores,
    }


async def extract_noun_tags(
    text: str,
    client: Any,
    model: str,
) -> Optional[List[str]]:
    """
    Extract key noun tags from text using a dedicated sidecar call.

    v0.3.1.1: Split from combined prompt. Matches core v0.4.5 extract_tags().
    Focused single-task prompt produces better tags from small models.

    Returns:
        List of lowercase tag strings (max 8), or None on failure
    """
    prompt = (
        "Extract the key TOPIC nouns from this text — people's names, places, objects, "
        "and specific things the text is actually about. "
        'Return ONLY a JSON object like: {"tags": ["calvin", "muscle car", "boston"]}\n'
        "Rules:\n"
        "- Use actual names, not pronouns (skip 'he', 'she', 'they', 'user', 'assistant')\n"
        "- Keep each tag as a short noun phrase (1-3 words)\n"
        "- Include both proper nouns and important common nouns\n"
        "- Skip meta-words about the conversation itself: source, answer, details, accuracy, "
        "response, question, topic, context, information, correction, update, memory\n"
        "- Skip generic verbs/actions: said, told, mentioned, discussed, talked, asked\n"
        "- Focus on WHO and WHAT the text is about, not how it was communicated\n"
        f'- Maximum 8 tags\n\nText: "{text[:2000]}"'
    )

    result = await _call_llm(
        prompt,
        client,
        model,
        system_prompt="You are part of a memory system. Return ONLY valid JSON. Be concise.",
    )
    if not result:
        return None

    # v0.3.2: same bare-array tolerance as extract_facts — small models
    # (qwen2.5:3b etc.) sometimes emit ["tag1", "tag2"] instead of the
    # schema-wrapped {"tags": [...]}. Pre-fix, .get() raised AttributeError
    # and the tag lane silently died, leaving TagCascade retrieval with
    # empty tag lists for any memories scored by that model.
    tags = result if isinstance(result, list) else result.get("tags")
    if not isinstance(tags, list):
        return None

    skip = {
        "he",
        "she",
        "they",
        "it",
        "user",
        "assistant",
        "the user",
        "the assistant",
        "i",
        "you",
        "we",
        "them",
        "his",
        "her",
    }
    cleaned = [
        t.lower().strip()
        for t in tags
        if isinstance(t, str) and t.lower().strip() not in skip and len(t.strip()) >= 2
    ]
    return cleaned[:8] if cleaned else None


async def summarize_only(
    content: str,
    client: Any,
    model: str,
) -> Optional[str]:
    """
    Summarize a memory's content (for retroactive summarization).

    Returns:
        Summary string (~300 chars) or None on failure
    """
    prompt = f"""You are part of a memory system for an AI assistant. You ARE the main AI — write a first-person note to your future self (~300 chars). Capture the important details from this exchange.

Exchange:
"{content}"

Respond with ONLY a JSON object:
{{"summary": "<~300 chars>"}}"""

    result = await _call_llm(
        prompt,
        client,
        model,
        system_prompt="/no_think\nYou are part of a memory system. Return ONLY valid JSON. Be concise.",
    )
    return result.get("summary") if result else None


async def extract_facts(
    content: str,
    client: Any,
    model: str,
) -> Optional[List[str]]:
    """
    Extract atomic facts from exchange content using a dedicated sidecar call.

    v0.3.1.1: Now used for both live exchanges and retroactive migration.
    Matches core v0.4.5 extract_facts() prompt.

    Returns:
        List of fact strings (max 10, each ≤150 chars) or None on failure
    """
    prompt = f"""You are part of a memory system. Extract key facts worth remembering from this exchange.

Rules:
- Include WHO or WHAT each fact is about — names, projects, topics
- Combine related details into ONE rich fact rather than many fragments
- Include specifics: dates, versions, preferences, decisions, reasons
- ONE fact per entry, max 150 characters
- Skip vague feelings, pleasantries, or generic observations
- If no useful facts, return empty array

Exchange:
"{content[:8000]}"

Respond with ONLY a JSON object:
{{"facts": ["fact 1", "fact 2", ...]}}"""

    result = await _call_llm(
        prompt,
        client,
        model,
        system_prompt="/no_think\nYou are part of a memory system. Return ONLY valid JSON.",
    )
    if not result:
        return None

    # v0.3.2: tolerate bare JSON arrays from small models (e.g. qwen2.5:3b)
    # which sometimes return ["fact 1", "fact 2"] instead of {"facts": [...]}.
    # Without this guard `.get()` raises AttributeError on list.
    raw_facts = result if isinstance(result, list) else result.get("facts", [])
    if not isinstance(raw_facts, list):
        return None

    # Clean facts — same logic as score_exchange() lines 228-234
    facts = [
        f.strip().lstrip("•-*0123456789. ")
        for f in raw_facts
        if isinstance(f, str) and len(f.strip()) > 10
    ][:10]

    return facts if facts else None


async def extract_facts_with_retry(
    content: str,
    client: Any,
    model: str,
    doc_id: str = "unknown",
) -> Optional[List[str]]:
    """
    Extract facts with client locking and retry queue.
    """

    async def _extract_task():
        return await extract_facts(content=content, client=client, model=model)

    result = await execute_with_client_lock(
        client=client, task_func=_extract_task, task_name=f"extract_facts_{doc_id}"
    )

    if result is None:
        task_data = {
            "task_type": "extract_facts",
            "doc_id": doc_id,
            "content": content,
            "client": client,
            "model": model,
        }
        queue_sidecar_retry(task_data, "Task returned None")

    return result


async def extract_noun_tags_with_retry(
    text: str,
    client: Any,
    model: str,
    doc_id: str = "unknown",
) -> Optional[List[str]]:
    """
    Extract noun tags with client locking and retry queue.
    """

    async def _extract_task():
        return await extract_noun_tags(text=text, client=client, model=model)

    result = await execute_with_client_lock(
        client=client, task_func=_extract_task, task_name=f"extract_noun_tags_{doc_id}"
    )

    if result is None:
        task_data = {
            "task_type": "extract_noun_tags",
            "doc_id": doc_id,
            "text": text,
            "client": client,
            "model": model,
        }
        queue_sidecar_retry(task_data, "Task returned None")

    return result


async def summarize_only_with_retry(
    content: str,
    client: Any,
    model: str,
    doc_id: str = "unknown",
) -> Optional[str]:
    """
    Summarize content with client locking and retry queue.
    """

    async def _summarize_task():
        return await summarize_only(content=content, client=client, model=model)

    result = await execute_with_client_lock(
        client=client, task_func=_summarize_task, task_name=f"summarize_only_{doc_id}"
    )

    if result is None:
        task_data = {
            "task_type": "summarize_only",
            "doc_id": doc_id,
            "content": content,
            "client": client,
            "model": model,
        }
        queue_sidecar_retry(task_data, "Task returned None")

    return result
