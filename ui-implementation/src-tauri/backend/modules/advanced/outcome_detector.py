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

    async def analyze(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze conversation to detect outcome using LLM.

        Args:
            conversation: List of turns [{role, content, timestamp}, ...]

        Returns:
            {
                "outcome": "worked|failed|partial|unknown",
                "confidence": 0.0-1.0,
                "indicators": ["explicit_thanks", "topic_change_30s", ...],
                "reasoning": "User said thanks and moved to new topic"
            }
        """
        if not conversation or len(conversation) < 2:
            return {
                "outcome": "unknown",
                "confidence": 0.0,
                "indicators": [],
                "reasoning": "Insufficient conversation history"
            }

        # LLM-only analysis
        if self.llm_service:
            llm_result = await self._llm_analyze(conversation)
            if llm_result:
                return llm_result

        # No LLM = no outcome detection
        logger.debug("No LLM service available for outcome detection")
        return {
            "outcome": "unknown",
            "confidence": 0.0,
            "indicators": [],
            "reasoning": "LLM service unavailable"
        }

    async def _llm_analyze(self, conversation: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze conversation outcome with full context understanding"""
        if not self.llm_service:
            return None

        try:
            # Format conversation for LLM
            conv_text = self._format_conversation(conversation)

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

Return JSON only:
{{
    "outcome": "worked|failed|partial|unknown",
    "confidence": 0.0-1.0,
    "indicators": ["signals"],
    "reasoning": "brief why"
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
                return result

        except Exception as e:
            logger.warning(f"LLM outcome analysis failed: {e}")

        return None

    def _format_conversation(self, conversation: List[Dict[str, Any]]) -> str:
        """Format conversation for LLM analysis"""
        lines = []
        for turn in conversation:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"{role}: {content[:500]}")  # Truncate long messages
        return "\n".join(lines)