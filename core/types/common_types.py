from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class ActionType(Enum):
    ADD_GOAL = "add_goal"
    GET_GOALS = "get_goals"
    REMOVE_GOAL = "remove_goal"
    ADD_VALUE = "add_value"
    GET_VALUES = "get_values"
    REMOVE_VALUE = "remove_value"
    PERFORM_WEB_SEARCH = "perform_web_search"
    PERFORM_WEB_SEARCH_INTERNAL = "perform_web_search_internal"
    PROVIDE_HELP = "provide_help"
    RESPOND_WITH_LLM = "respond_with_llm"
    REMEMBER_INFO = "remember_info"
    QUERY_INGESTED_BOOKS = "query_ingested_books"
    SAVE_LEARNING = "save_learning"
    UNKNOWN_INTENT = "unknown_intent"
    NO_ACTION = "no_action"
    RESPOND_WITH_STRUCTURED_SUMMARY = "respond_with_structured_summary"  # <— NEW
    SOUL_BOOK_LOOKUP = "soul_book_lookup"  # <— PATCHED: Enables book fragment lookup
    MULTI_PART_QUERY = "multi_part_query"

class Action(BaseModel):
    type: ActionType
    parameters: Optional[Dict[str, Any]] = None

class Message(BaseModel):
    role: str
    content: str

class InteractionType(Enum):
    USER_CHAT = "USER_CHAT"
    OG_CHAT = "OG_CHAT"
    ADD_GOAL = ActionType.ADD_GOAL.value
    GET_GOALS = ActionType.GET_GOALS.value
    REMOVE_GOAL = ActionType.REMOVE_GOAL.value
    ADD_VALUE = ActionType.ADD_VALUE.value
    GET_VALUES = ActionType.GET_VALUES.value
    REMOVE_VALUE = ActionType.REMOVE_VALUE.value
    PERFORM_WEB_SEARCH = ActionType.PERFORM_WEB_SEARCH.value
    PERFORM_WEB_SEARCH_INTERNAL = ActionType.PERFORM_WEB_SEARCH_INTERNAL.value
    PROVIDE_HELP = ActionType.PROVIDE_HELP.value
    RESPOND_WITH_LLM = ActionType.RESPOND_WITH_LLM.value
    REMEMBER_INFO = ActionType.REMEMBER_INFO.value
    QUERY_INGESTED_BOOKS = ActionType.QUERY_INGESTED_BOOKS.value
    SAVE_LEARNING = ActionType.SAVE_LEARNING.value
    RESPOND_WITH_STRUCTURED_SUMMARY = ActionType.RESPOND_WITH_STRUCTURED_SUMMARY.value  # <— NEW

    OG_ADD_GOAL = f"OG_{ActionType.ADD_GOAL.name}"
    OG_GET_GOALS = f"OG_{ActionType.GET_GOALS.name}"
    OG_PERFORM_WEB_SEARCH = f"OG_{ActionType.PERFORM_WEB_SEARCH.name}"
    OG_PERFORM_WEB_SEARCH_INTERNAL = f"OG_{ActionType.PERFORM_WEB_SEARCH_INTERNAL.name}"
    OG_PROVIDE_HELP = f"OG_{ActionType.PROVIDE_HELP.name}"
    OG_RESPOND_WITH_LLM = f"OG_{ActionType.RESPOND_WITH_LLM.name}"
    OG_QUERY_INGESTED_BOOKS = f"OG_{ActionType.QUERY_INGESTED_BOOKS.name}"
    OG_SAVE_LEARNING = f"OG_{ActionType.SAVE_LEARNING.name}"
    OG_RESPOND_WITH_STRUCTURED_SUMMARY = f"OG_{ActionType.RESPOND_WITH_STRUCTURED_SUMMARY.name}"  # <— NEW
    OG_SOUL_BOOK_LOOKUP = f"OG_{ActionType.SOUL_BOOK_LOOKUP.name}"
    OG_MULTI_PART_QUERY = f"OG_{ActionType.MULTI_PART_QUERY.name}"
    
    @classmethod
    def from_action_type(cls, action_type: ActionType, is_og: bool = False) -> 'InteractionType':
        prefix = "OG_" if is_og else ""
        specific = f"{prefix}{action_type.name}"
        try:
            return cls[specific]
        except KeyError:
            try:
                return cls(action_type.value)
            except ValueError:
                logger.warning(
                    f"Could not map ActionType '{action_type.name}' to InteractionType '{specific}'."
                )
                return cls.OG_CHAT if is_og else cls.USER_CHAT