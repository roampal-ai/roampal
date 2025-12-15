import logging
from typing import List, Optional, Dict, Any
import re

from core.interfaces.intent_router_interface import IntentRouterInterface
from core.types.common_types import Action, ActionType, Message

logger = logging.getLogger(__name__)

OG_SPECIFIC_COMMANDS_KEYWORDS = {
    ActionType.SAVE_LEARNING: ["/save_learning", "/remember_insight", "/log_this"],
    ActionType.PROVIDE_HELP: ["/og help", "/og_commands"]
}

# Keyword stubs for quick factual-query detection (temporary heuristic)
FACTUAL_TRIGGERS = [
    "price of", "who won", "latest", "current", "today", "news", "weather",
    "when is", "where is", "who is", "how do I", "how to", "top", "rank",
    "is [x] banned", "is [x] legal", "has [x] changed",
    "what is going on", "what's happening", "what happened", "what's new",
    "update on", "status of", "recent", "breaking", "developments",
    "trump", "biden", "politics", "election", "congress", "senate",
    "white house", "dc", "washington", "government", "policy"
]

BOOKS = [
    {"title": "Meditations", "author": "Marcus Aurelius", "keywords": ["stoicism", "suffering", "meaning"]},
    {"title": "Clean Architecture", "author": "Robert C. Martin", "keywords": ["uncle bob", "dependency inversion", "layers"]},
    {"title": "The Art of War", "author": "Sun Tzu", "keywords": ["strategy", "know thyself", "enemy"]},
    {"title": "Man's Search for Meaning", "author": "Viktor Frankl", "keywords": ["logotherapy", "concentration camp"]},
    {"title": "Tao Te Ching", "author": "Lao Tzu", "keywords": ["wu wei", "flow", "dao"]},
    {"title": "Zero to One", "author": "Peter Thiel", "keywords": ["monopolies", "startups", "innovation"]}
]

BOOK_KEYWORDS = [b['title'].lower() for b in BOOKS] + [b['author'].lower() for b in BOOKS] + [kw.lower() for b in BOOKS for kw in b['keywords']]

class OGIntentRouter(IntentRouterInterface):
    def __init__(self):
        self.keywords: Dict[ActionType, List[str]] = {}
        self.initialized = False
        logger.debug("OGIntentRouter created (uninitialized).")

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.initialized:
            return
        self.keywords = OG_SPECIFIC_COMMANDS_KEYWORDS.copy()
        logger.info(f"OGIntentRouter initialized with {len(self.keywords)} patterns.")
        self.initialized = True

    def _parse_save_learning_params(self, input_after_command: str) -> Dict[str, Any]:
        params = {"learning_content": ""}
        content_match = re.search(r'"([^"]+)"', input_after_command)
        remaining = input_after_command
        if content_match:
            params["learning_content"] = content_match.group(1).strip()
            remaining = input_after_command.replace(content_match.group(0), "").strip()
        else:
            parts = input_after_command.split(" --", 1)
            params["learning_content"] = parts[0].strip()
            remaining = f"--{parts[1]}" if len(parts) > 1 else ""

        topic_match = re.search(r'--topic\s+"?([^"]+)"?', remaining, re.IGNORECASE)
        if topic_match:
            params["topic"] = topic_match.group(1).strip()

        keywords_match = re.search(r'--keywords\s+"?([^"]+)"?', remaining, re.IGNORECASE)
        if keywords_match:
            kws = [kw.strip() for kw in keywords_match.group(1).split(",") if kw.strip()]
            if kws:
                params["keywords"] = kws

        return params

    def _should_trigger_web_search(self, user_input: str) -> bool:
        inp_lower = user_input.lower()
        logger.info(f"OGIntentRouter: Checking web search triggers for: '{user_input}'")
        logger.info(f"OGIntentRouter: Lowercase input: '{inp_lower}'")
        
        # Skip web search for image analysis requests
        image_analysis_indicators = [
            "[image context:", "image content:", "about this image",
            "analyze this image", "analyze the image", "please analyze",
            "image:", "screenshot", "picture shows", "image shows",
            "image depicts", "image displays", "visual shows",
            "computer screen with", "browser window", "multiple windows",
            "foreground", "background", "tabs opened",
            "please analyze this image"  # Added for consistency with router format
        ]
        
        for indicator in image_analysis_indicators:
            if indicator in inp_lower:
                logger.info(f"OGIntentRouter: Detected image analysis context ('{indicator}'), skipping web search")
                return False
        
        # Skip web search for conversation context queries
        conversation_context_indicators = [
            "what were we talking about", "what were we discussing", 
            "what was our conversation", "what did we discuss",
            "what were we saying", "what was our chat about",
            "what were we discussing", "what was our topic",
            "what were we covering", "what was our subject"
        ]
        
        for indicator in conversation_context_indicators:
            if indicator in inp_lower:
                logger.info(f"OGIntentRouter: Detected conversation context query, skipping web search")
                return False
        
        for phrase in FACTUAL_TRIGGERS:
            logger.info(f"OGIntentRouter: Checking phrase: '{phrase}'")
            if phrase in inp_lower:
                logger.info(f"OGIntentRouter: Triggered web search from phrase match: '{phrase}'")
                return True
        logger.info(f"OGIntentRouter: No web search triggers found")
        return False

    def _detect_book_lookup(self, user_input: str) -> Optional[Dict[str, str]]:
        inp = user_input.lower()
        
        # First check if user is explicitly asking for book content
        book_request_patterns = [
            r"quote.*from.*book",
            r"what.*book.*say",
            r"from.*book",
            r"in.*book",
            r"book.*quote",
            r"book.*model",
            r"book.*summary"
        ]
        
        is_book_request = any(re.search(pattern, inp) for pattern in book_request_patterns)
        
        if is_book_request:
            matched_kw = [kw for kw in BOOK_KEYWORDS if kw in inp]
            logger.info(f"Matched keywords in detect_book_lookup: {matched_kw}")
            if matched_kw:
                typ = None
                if "quote" in inp: typ = "quote"
                elif "model" in inp or "concept" in inp: typ = "model"
                elif "lesson" in inp or "learning" in inp or "summary" in inp: typ = "summary"
                else: typ = "summary"  # Default to summary if no specific type
                
                book = next((b['title'] for b in BOOKS if any(kw in [b['title'].lower(), b['author'].lower()] + [k.lower() for k in b['keywords']] for kw in matched_kw)), None)
                if book and typ:
                    return {"book": book, "type": typ}
        return None

    async def determine_action(
        self,
        user_input: str,
        conversation_history: Optional[List[Message]] = None,
        user_profile: Optional[Dict[str, Any]] = None
    ) -> Action:
        if not self.initialized:
            return Action(type=ActionType.RESPOND_WITH_LLM,
                          parameters={"reason": "Router not initialized"})

        inp = user_input.strip()
        # Sanitize input for safe logging
        from utils.safe_logger import sanitize_for_logging
        safe_inp = sanitize_for_logging(inp)
        logger.info(f"OGIntentRouter: Input received: '{safe_inp}'")

        # 1) Structured-summary command
        m = re.search(r'\bstructured summary for\s+(.+)', inp, re.IGNORECASE)
        if m:
            chunk_id = m.group(1).strip().rstrip(".")
            logger.info("OGIntentRouter: Matched RESPOND_WITH_STRUCTURED_SUMMARY")
            return Action(
                type=ActionType.RESPOND_WITH_STRUCTURED_SUMMARY,
                parameters={"chunk_id": chunk_id}
            )

        # 2) Slash-commands
        for atype, kws in self.keywords.items():
            for kw in kws:
                if inp.lower().startswith(kw.lower()):
                    logger.info(f"OGIntentRouter: Matched {atype.name}")
                    params = {}
                    tail = inp[len(kw):].strip()
                    if atype == ActionType.SAVE_LEARNING:
                        params = self._parse_save_learning_params(tail)
                    return Action(type=atype, parameters=params)

        # Detect multi-part queries and split - more specific detection
        # Only trigger if we have clear numbered lists or semicolons, not just "and"
        # IMPORTANT: Skip multi-part detection for image context messages
        if not inp.startswith("Based on this image content:"):
            if (';' in inp or 
                (('1.' in inp or '2.' in inp or '3.' in inp) and len([x for x in ['1.', '2.', '3.', '4.', '5.'] if x in inp]) >= 2) or
                (inp.count('?') >= 2)):  # Multiple questions
                return Action(type=ActionType.MULTI_PART_QUERY, parameters={"original_input": inp})

        # Detect book lookups
        book_lookup = self._detect_book_lookup(inp)
        if book_lookup:
            logger.info(f"OGIntentRouter: Book lookup detected for {book_lookup}")
            return Action(type=ActionType.SOUL_BOOK_LOOKUP, parameters=book_lookup)

        # 3) Heuristic Web Search Detection
        logger.info(f"OGIntentRouter: About to check web search triggers for: '{safe_inp}'")
        if self._should_trigger_web_search(inp):
            logger.info(f"OGIntentRouter: Web search triggered, returning PERFORM_WEB_SEARCH action")
            return Action(
                type=ActionType.PERFORM_WEB_SEARCH,
                parameters={"query": inp, "confidence": 0.85}
            )
        else:
            logger.info(f"OGIntentRouter: Web search not triggered, continuing to fallback")

        # 4) Fallback: use LLM
        logger.warning("OGIntentRouter: No match; defaulting to RESPOND_WITH_LLM")
        return Action(
            type=ActionType.RESPOND_WITH_LLM,
            parameters={"original_input": user_input}
        )

    async def handle_multi_part_query(self, input_text: str) -> List[Dict]:
        """
        Split multi-part query into sub-queries for parallel processing.
        Robust: Handle varying separators, trim whitespace, log splits.
        """
        try:
            parts = [p.strip() for p in input_text.split(';') if p.strip()]
            if not parts:
                parts = [p.strip() for p in re.split(r'\d+\.', input_text) if p.strip()]  # Split on numbered lists
            
            sub_queries = []
            for part in parts:
                sub_queries.append({'sub_query': part, 'action': await self.determine_action(part)})
            
            logger.info(f"Split multi-part query into {len(sub_queries)} sub-queries")
            return sub_queries
        except Exception as e:
            logger.error(f"Error splitting multi-part: {e}")
            return [{'sub_query': input_text, 'action': Action(type=ActionType.RESPOND_WITH_LLM)}]

    async def get_available_commands(self) -> List[str]:
        cmds = ["Use natural language for most queries."]
        if ActionType.SAVE_LEARNING in self.keywords:
            cmds.append(
                '/save_learning "Your insight here" --topic "optional" --keywords "k1,k2"'
            )
        cmds.append("/og help")
        return cmds