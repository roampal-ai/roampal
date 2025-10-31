# Tool definitions for Roampal
# Currently only search_memory is implemented and active

# Active tools (currently implemented)
AVAILABLE_TOOLS = [
    # Memory Search Tool - The only active tool in Roampal
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search the 5-tier memory system (books, working, history, patterns, memory_bank) for relevant information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query - use the user's EXACT words/phrases, do NOT simplify or extract keywords. If user provides a long paragraph, use the full paragraph as the query."
                    },
                    "collections": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["books", "working", "history", "patterns", "memory_bank", "all"]},
                        "description": """Which collections to search. Choose based on keywords:
- 'books' = user says: book, document, pdf, chapter, uploaded file
- 'working' = user says: working memory, recent context, current project
- 'history' = user says: conversation, chat history, what we discussed
- 'patterns' = user says: patterns, learned behaviors, user preferences
- 'memory_bank' = user says: remember, stored fact, personal info, identity
- 'all' = user doesn't specify OR wants comprehensive search
Examples: "search my books" → ["books"], "what did we discuss" → ["history"], "do you remember my name" → ["memory_bank"]""",
                        "default": ["all"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (1-20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"],
            },
        },
    },
    # Memory Bank Management Tools
    {
        "type": "function",
        "function": {
            "name": "create_memory",
            "description": "Store important user facts in memory bank for long-term recall (identity, preferences, goals, context)",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact to remember (be specific and complete)"
                    },
                    "tag": {
                        "type": "string",
                        "enum": ["identity", "preference", "goal", "context"],
                        "description": "Category: identity (who they are), preference (what they like), goal (what they're building), context (relevant details)"
                    }
                },
                "required": ["content", "tag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": "Update an existing memory when user corrects or clarifies information",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_content": {
                        "type": "string",
                        "description": "The old/incorrect fact to find and replace"
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The corrected/updated fact"
                    }
                },
                "required": ["old_content", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "archive_memory",
            "description": "Archive outdated or no longer relevant memories",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact to archive (find by semantic match)"
                    }
                },
                "required": ["content"],
            },
        },
    }
]

# Future tools (not yet implemented - kept for reference)
# These tool definitions are disabled to avoid confusing the LLM
FUTURE_TOOLS = [
    # Web search, vision, speech, document processing, code analysis tools
    # will be added here when implemented
]
