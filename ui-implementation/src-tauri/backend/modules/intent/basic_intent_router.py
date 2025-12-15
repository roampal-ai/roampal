# backend/modules/intent/basic_intent_router.py
import logging
from typing import List, Optional, Dict, Any
import re

from core.interfaces.intent_router_interface import IntentRouterInterface
from core.types.common_types import Action, ActionType, Message
from utils.text_utils import strip_fluff_phrases, extract_entities_for_action

logger = logging.getLogger(__name__)

DEFAULT_KEYWORDS = {
    ActionType.ADD_GOAL: ["add goal", "new goal", "set goal", "my goal is", "i want to achieve", "my objective is"],
    ActionType.GET_GOALS: ["get goals", "what are my goals", "show goals", "list goals", "view goals"],
    ActionType.REMOVE_GOAL: ["remove goal", "delete goal", "clear goal"],
    ActionType.ADD_VALUE: ["add value", "new value", "my value is", "set value"],
    ActionType.GET_VALUES: ["get values", "what are my values", "show values", "list values", "view values"],
    ActionType.REMOVE_VALUE: ["remove value", "delete value", "clear value"],
    ActionType.PERFORM_WEB_SEARCH: [
        "search for", "look up", "find information on", "what is", "who is", 
        "when did", "how does", "tell me about", "search web", "google", "bing", 
        "duckduckgo", "what's the weather", "latest news on"
    ],
    ActionType.PROVIDE_HELP: [
        "help roampal", "roampal help", "user manual", "show me commands", 
        "what are your commands", "assistance please", "i need help", "can you help", "help"
    ], # Made 'help' more specific
    ActionType.REMEMBER_INFO: ["remember that", "make a note", "store this info", "save this information"],
}

class BasicIntentRouter(IntentRouterInterface):
    def __init__(self):
        self.keywords: Dict[ActionType, List[str]] = {}
        self.initialized = False
        logger.debug("BasicIntentRouter instance created (uninitialized).")

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        if self.initialized:
            logger.info("BasicIntentRouter already initialized.")
            return
        self.keywords = DEFAULT_KEYWORDS.copy() # Use a copy to allow modification
        if config and "custom_keywords" in config:
            for action_type_str, kw_list in config["custom_keywords"].items():
                try:
                    action_type = ActionType[action_type_str.upper()]
                    self.keywords[action_type] = kw_list # Override or add
                except KeyError:
                    logger.warning(f"Unknown ActionType '{action_type_str}' in custom_keywords config.")
        
        logger.info(f"BasicIntentRouter initialized with {len(self.keywords)} intent patterns.")
        self.initialized = True

    async def determine_action(self, 
                               user_input: str, 
                               conversation_history: Optional[List[Message]] = None,
                               user_profile: Optional[Dict[str, Any]] = None) -> Action:
        if not self.initialized:
            logger.warning("BasicIntentRouter not initialized. Defaulting to RESPOND_WITH_LLM.")
            return Action(type=ActionType.RESPOND_WITH_LLM, parameters={"reason": "Router not initialized"})

        # Use your more comprehensive strip_fluff_phrases
        processed_input = strip_fluff_phrases(user_input)
        logger.info(f"BasicIntentRouter - Original input: '{user_input}', Processed input for intent: '{processed_input}'")

        # Sort keywords by length (longest first) for more specific matching
        # This helps if one keyword is a substring of another for a different action
        # We'll iterate through action types, and within each, check its keywords.
        # For now, the order of ActionTypes in DEFAULT_KEYWORDS implies some priority.
        
        for action_type, kw_list in self.keywords.items():
            # Sort keywords for this action_type by length, longest first
            sorted_kw_list = sorted(kw_list, key=len, reverse=True)
            for keyword in sorted_kw_list:
                # Use regex for case-insensitive, whole word/phrase matching
                # \b ensures word boundaries, but might be too restrictive for phrases.
                # A simpler check: if keyword is in processed_input and forms a distinct phrase.
                # For phrases, we want to match the phrase as a whole.
                # For single words, \b is good.
                
                # If keyword contains spaces, it's a phrase, look for exact phrase (case insensitive)
                if ' ' in keyword:
                    if keyword in processed_input: # Already lowercased
                        logger.info(f"Intent identified: {action_type.name} (Phrase Keyword: '{keyword}' matched in '{processed_input}')")
                        params = await extract_entities_for_action(action_type, user_input, processed_input, keyword)
                        if action_type == ActionType.PERFORM_WEB_SEARCH and (not params or not params.get("query")):
                            params = params or {}; params["query"] = processed_input.split(keyword, 1)[-1].strip() or processed_input
                        return Action(type=action_type, parameters=params if params else None)
                else: # Single word keyword, use word boundary
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    if re.search(pattern, processed_input, re.IGNORECASE):
                        logger.info(f"Intent identified: {action_type.name} (Word Keyword: '{keyword}' matched in '{processed_input}')")
                        params = await extract_entities_for_action(action_type, user_input, processed_input, keyword)
                        if action_type == ActionType.PERFORM_WEB_SEARCH and (not params or not params.get("query")):
                             params = params or {}; params["query"] = processed_input.split(keyword, 1)[-1].strip() or processed_input
                        return Action(type=action_type, parameters=params if params else None)

        logger.info(f"No specific keyword intent found for processed input '{processed_input}'. Defaulting to RESPOND_WITH_LLM.")
        return Action(type=ActionType.RESPOND_WITH_LLM, parameters={"original_input": user_input})

    async def get_available_commands(self) -> List[str]:
        commands = []
        for action_type, kws in self.keywords.items():
            if action_type not in [ActionType.PERFORM_WEB_SEARCH, ActionType.RESPOND_WITH_LLM, ActionType.UNKNOWN_INTENT, ActionType.NO_ACTION]:
                example_trigger = kws[0]
                if "goal" in example_trigger: example = f"{example_trigger} <your goal description>"
                elif "value" in example_trigger: example = f"{example_trigger} <your value description>"
                elif "remember" in example_trigger: example = f"{example_trigger} <information to remember>"
                else: example = example_trigger
                commands.append(f"'{example}' (Action: {action_type.name})")
        commands.append("'search for <your query>' (To perform a web search)")
        return commands
