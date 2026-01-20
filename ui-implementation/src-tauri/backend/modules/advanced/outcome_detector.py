"""
OutcomeDetector - LLM-only outcome detection
Analyzes conversation patterns to detect implicit success/failure signals
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class OutcomeDetector:
    """
    LLM-based outcome detector with nuanced understanding of conversation context.
    Returns {outcome: "worked|failed|partial|unknown", confidence: 0-1, indicators: [...]}

    Note: Requires LLM service to function. No heuristic fallback.
    """

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        if not llm_service:
            logger.warning("OutcomeDetector initialized without LLM service - outcome detection will not work")

    async def analyze(
        self,
        conversation: List[Dict[str, Any]],
        surfaced_memories: Optional[Dict[int, str]] = None,
        llm_marks: Optional[Dict[int, str]] = None  # v0.2.12 Fix #7: {pos: emoji}
    ) -> Dict[str, Any]:
        """
        Analyze conversation to detect outcome using LLM.

        Args:
            conversation: List of turns [{role, content, timestamp}, ...]
            surfaced_memories: v0.2.12 - Optional {position: content} for selective scoring
            llm_marks: v0.2.12 Fix #7 - Main LLM's attribution {pos: '👍'/'👎'/'➖'}

        Returns:
            {
                "outcome": "worked|failed|partial|unknown",
                "confidence": 0.0-1.0,
                "indicators": ["explicit_thanks", "topic_change_30s", ...],
                "reasoning": "User said thanks and moved to new topic",
                "used_positions": [1, 3],  # v0.2.12: which memories were actually used (fallback)
                "upvote": [1],              # v0.2.12 Fix #7: positions to upvote
                "downvote": [2]             # v0.2.12 Fix #7: positions to downvote
            }
        """
        if not conversation or len(conversation) < 2:
            return {
                "outcome": "unknown",
                "confidence": 0.0,
                "indicators": [],
                "reasoning": "Insufficient conversation history",
                "used_positions": [],
                "upvote": [],
                "downvote": []
            }

        # LLM-only analysis
        if self.llm_service:
            llm_result = await self._llm_analyze(conversation, surfaced_memories, llm_marks)
            if llm_result:
                return llm_result

        # No LLM = no outcome detection
        logger.debug("No LLM service available for outcome detection")
        return {
            "outcome": "unknown",
            "confidence": 0.0,
            "indicators": [],
            "reasoning": "LLM service unavailable",
            "used_positions": [],
            "upvote": [],
            "downvote": []
        }

    async def _llm_analyze(
        self,
        conversation: List[Dict[str, Any]],
        surfaced_memories: Optional[Dict[int, str]] = None,
        llm_marks: Optional[Dict[int, str]] = None  # v0.2.12 Fix #7
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze conversation outcome with full context understanding.

        v0.2.12: Now supports selective scoring - identifies which memories were actually used.
        v0.2.12 Fix #7: When llm_marks provided, uses causal scoring (combines outcome + attribution).
        """
        if not self.llm_service:
            return None

        try:
            # Format conversation for LLM
            conv_text = self._format_conversation(conversation)

            # v0.2.12 Fix #7: If main LLM provided marks, use simplified causal scoring prompt
            if llm_marks:
                return await self._analyze_with_marks(conv_text, llm_marks)

            # v0.3.0: Simplified prompt - just detect outcome, no memory tracking
            prompt = f"""Based on how the user responded, grade this exchange.

{conv_text}

Grade the USER'S REACTION (not the assistant's quality):
- worked = user satisfied (thanks, great, perfect, got it)
- failed = user unhappy/correcting (no, wrong, didn't work)
- partial = lukewarm (ok, I guess, sure)
- unknown = no clear signal

Return JSON: {{"outcome": "worked|failed|partial|unknown"}}"""

            # Call LLM
            response = await self.llm_service.generate_response(prompt, format="json")

            # Extract JSON from anywhere in response (handles thinking blocks, markdown, etc)
            import re
            import json

            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON found in response: {response[:200]}")

            # Parse the extracted JSON
            result = json.loads(json_match.group(0))

            # Validate structure - only outcome is required, fill defaults for rest
            if "outcome" in result:
                result.setdefault("confidence", 0.8)
                result.setdefault("indicators", [])
                result.setdefault("reasoning", "")
                result.setdefault("used_positions", [])
                result["upvote"] = []
                result["downvote"] = []
                return result

        except Exception as e:
            logger.warning(f"LLM outcome analysis failed: {e}")

        return None

    async def _analyze_with_marks(
        self,
        conv_text: str,
        llm_marks: Dict[int, str]
    ) -> Optional[Dict[str, Any]]:
        """v0.3.0: Detect overall outcome from user reaction.

        Note: Per-memory scoring now uses direct emoji→outcome mapping in agent_chat.py.
        This method only determines the overall outcome for logging/key_takeaway.
        """
        import re
        import json

        # Format marks for display
        marks_display = " ".join(f"{pos}{emoji}" for pos, emoji in sorted(llm_marks.items()))

        prompt = f"""Did the assistant's response help the user?

USER'S REACTION:
{conv_text}

Answer:
- YES = "thanks!", "perfect!", user moved on to new topic
- NO = "wrong", "didn't work", user frustrated or corrected
- KINDA = "okay", "I guess", lukewarm
- UNKNOWN = no clear signal

The assistant marked memories as: {marks_display}
(👍=definitely helped, 🤷=kinda helped, 👎=misleading, ➖=unused)

Return JSON only:
{{"outcome": "yes/no/kinda/unknown", "reasoning": "brief"}}"""

        try:
            response = await self.llm_service.generate_response(prompt, format="json")

            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError(f"No JSON found in response: {response[:200]}")

            result = json.loads(json_match.group(0))

            # Normalize outcome to standard format
            outcome_map = {"yes": "worked", "no": "failed", "kinda": "partial"}
            raw_outcome = result.get("outcome", "unknown").lower()
            normalized_outcome = outcome_map.get(raw_outcome, raw_outcome)

            # Build complete result (v0.3.0: upvote/downvote no longer needed, using direct emoji→outcome)
            return {
                "outcome": normalized_outcome,
                "confidence": 0.8 if normalized_outcome != "unknown" else 0.0,
                "indicators": ["llm_marks_direct"],
                "reasoning": result.get("reasoning", "Overall outcome from user reaction"),
                "used_positions": [],  # Not used - direct scoring via emoji
                "upvote": [],  # Not used - direct scoring via emoji
                "downvote": []  # Not used - direct scoring via emoji
            }

        except Exception as e:
            logger.warning(f"LLM outcome detection failed: {e}, falling back to inference mode")
            return None

    def _format_conversation(self, conversation: List[Dict[str, Any]]) -> str:
        """Format conversation for LLM analysis"""
        lines = []
        for turn in conversation:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"{role}: {content[:500]}")  # Truncate long messages
        return "\n".join(lines)