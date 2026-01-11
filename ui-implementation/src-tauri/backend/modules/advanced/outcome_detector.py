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

            # v0.2.12: Build memory section if surfaced_memories provided (fallback to inference)
            memory_section = ""
            if surfaced_memories:
                memory_lines = []
                for pos in sorted(surfaced_memories.keys()):
                    content = surfaced_memories[pos][:200]  # Truncate for prompt size
                    memory_lines.append(f"{pos}. {content}")
                memory_section = f"""

SURFACED MEMORIES (assistant had access to these):
{chr(10).join(memory_lines)}

Which memory NUMBERS were actually USED or REFERENCED in the assistant's response?
Only include memories that directly influenced the answer. Return empty list if none were used.
"""

            prompt = f"""Evaluate if the assistant's response was helpful. Judge both user feedback AND response quality.

"worked": ENTHUSIASTIC satisfaction or clear success
  • "thanks!", "perfect!", "awesome!", "that worked!"
  • User moves to NEW topic (indicates previous was resolved)
  • NOT worked: "yea pretty good", "okay", follow-up questions

"failed": Dissatisfaction, criticism, or confusion
  • "no", "nah", "wrong", "didn't work"
  • Criticism: "why are you...", "stop doing..."
  • Repeated questions about SAME issue (solution didn't work)
  • Follow-up questions expressing confusion

"partial": Lukewarm (positive but not enthusiastic)
  • "yea pretty good", "okay", "sure", "I guess", "kinda"
  • Helped somewhat but incomplete

"unknown": No clear signal yet
  • No user response after answer
  • Pure neutral: "hm", "noted"

CRITICAL: Follow-up questions are NOT success signals. User continuing conversation ≠ satisfaction.

CONVERSATION:
{conv_text}
{memory_section}
Return JSON only:
{{
    "outcome": "worked|failed|partial|unknown",
    "confidence": 0.0-1.0,
    "indicators": ["signals"],
    "reasoning": "brief why",
    "used_positions": [1, 3]
}}"""

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

            # Validate structure
            if all(k in result for k in ["outcome", "confidence", "indicators", "reasoning"]):
                # v0.2.12: Ensure used_positions is always present (default to empty list)
                if "used_positions" not in result:
                    result["used_positions"] = []
                # v0.2.12 Fix #7: Add empty upvote/downvote for consistency
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
        """v0.2.12 Fix #7: Causal scoring using main LLM's attribution marks.

        Simplified prompt - just detect outcome, then combine with marks for upvote/downvote.
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

The assistant marked memories as: {marks_display}
(👍=helpful, 👎=wrong/misleading, ➖=unused)

SCORING RULES:
If YES: upvote 👍 memories, ignore 👎 and ➖
If NO: downvote 👎 memories, ignore 👍 and ➖
If KINDA: slight upvote 👍, slight downvote 👎

Return JSON only:
{{"outcome": "yes/no/kinda/unknown", "upvote": [1], "downvote": [2], "reasoning": "brief"}}"""

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

            # Build complete result
            return {
                "outcome": normalized_outcome,
                "confidence": 0.8 if normalized_outcome != "unknown" else 0.0,
                "indicators": ["llm_marks_causal"],
                "reasoning": result.get("reasoning", "Causal scoring with main LLM marks"),
                "used_positions": [],  # Not used in causal mode
                "upvote": result.get("upvote", []),
                "downvote": result.get("downvote", [])
            }

        except Exception as e:
            logger.warning(f"LLM causal scoring failed: {e}, falling back to inference mode")
            return None

    def _format_conversation(self, conversation: List[Dict[str, Any]]) -> str:
        """Format conversation for LLM analysis"""
        lines = []
        for turn in conversation:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"{role}: {content[:500]}")  # Truncate long messages
        return "\n".join(lines)