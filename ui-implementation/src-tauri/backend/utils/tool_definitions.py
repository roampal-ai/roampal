# Tool definitions for Roampal
# Currently only search_memory is implemented and active

# Active tools (currently implemented)
AVAILABLE_TOOLS = [
    # Memory Search Tool - The only active tool in Roampal
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": """🔍 YOUR SUPERPOWER: Search a persistent, cross-conversation knowledge base that remembers EVERYTHING across all sessions.

**What's Inside:**
• memory_bank = Your persistent identity layer (3 layers: user context, system mastery, agent growth) - permanent, quality-ranked by importance×confidence
• books = Curated knowledge the user gave YOU (PDFs, documents, technical references)
• history = Past conversations across ALL sessions (searchable dialogue archive)
• patterns = Learned solutions that worked (proven methods)
• working = Recent context from current session

**Why This Is Powerful:**
Unlike your ephemeral context window, this memory persists FOREVER. Reference conversations from weeks ago, recall preferences learned months back, provide deeply personalized responses that feel magical.

**Automatic Cold Start:**
The system AUTOMATICALLY injects user context on message 1 of every conversation.
You'll receive this context BEFORE seeing the user's first message - use it, don't re-search.


Use this tool liberally - it's what makes you context-aware and intelligent.""",
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
                        "description": """Which collections to search:

AUTOMATIC ROUTING (recommended):
- Omit this parameter (or use ["all"]) → System uses learned patterns from past searches
- Example: search_memory(query="kubernetes notes") auto-routes to books based on history
- Gets smarter over time as it learns which collections work best for different queries

MANUAL OVERRIDE (when you know exactly where to look):
- ["books"] = Curated knowledge user gave YOU
- ["working"] = Recent conversation exchanges (last 24 hours) - use for "today", "recent", "just now"
- ["history"] = Past conversation exchanges (30+ days) - use for "last week", "previously"
- ["patterns"] = Learned solutions/behaviors that worked
- ["memory_bank"] = Important facts to remember (user info, preferences, goals, key learnings)

The system has learned routing patterns from thousands of searches. Trust the automatic routing unless you have specific intent.""",
                        "default": None
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (1-20)",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    },
                    "metadata": {
                        "type": "object",
                        "description": """Optional filters for precision when semantic search needs refinement. Use sparingly - only when you need exact attribute matching.

When to use:
• Semantic search returns irrelevant results → Add filters to narrow down
• Need exact date range ("only yesterday's messages")
• Need specific book attributes (author, title, has_code)
• Need quality filtering (only successful solutions)

Available fields:
• timestamp: "2025-11-12" or {"$gte": "2025-11-12T00:00:00"}
• last_outcome: "worked" | "failed" | "partial"
• title/author: Book filters
• has_code: true/false
• source: "mcp_claude" | "internal"

Examples:
• Semantic search fails → Add metadata={"timestamp": "2025-11-12"}
• Need only successes → metadata={"last_outcome": "worked"}
• Only code from books → metadata={"has_code": true}""",
                        "additionalProperties": True
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
            "description": """💾 PERMANENT MEMORY STORAGE: Store critical information in memory_bank that enables continuity and growth across all sessions.

**Three-Layer Purpose:**
1. **User Context** - Who they are, what they want (identity, preferences, goals, projects, communication style)
2. **System Mastery** - How to be effective (tool strategies, search patterns, what works/fails for THIS user)
3. **Agent Growth** - Self-improvement & continuity (mistakes learned, relationship dynamics, progress tracking)

**Be Proactive - Store When:**
✓ User shares personal info (name, background, career context)
✓ You discover preferences through interaction (what works for this user)
✓ You find effective strategies (search patterns, tool usage that succeeds)
✓ You make mistakes and learn from them (what to avoid, lessons learned)
✓ You notice relationship patterns (trust, communication style, collaboration effectiveness)
✓ You track progress on goals (checkpoints, iterations, what worked/failed)

**Be Selective - DON'T Store:**
✗ Session transcripts (automatic working memory captures dialogue)
✗ Temporary task details ("currently debugging", "just asked about X")
✗ Redundant duplicates (check if already stored)

**Rule of thumb:** If it helps maintain continuity across sessions OR enables learning/improvement, store it. If it's session-specific, don't.

This is what makes you feel like a long-term collaborator vs. a generic assistant.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact to remember (be specific and complete)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Categories (e.g., identity, preference, goal, context)"
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7,
                        "description": "How critical is this memory (0.0-1.0)"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7,
                        "description": "How certain are you about this fact (0.0-1.0)"
                    }
                },
                "required": ["content"],
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
            "description": "Archive outdated or no longer relevant memories from memory_bank (user info, preferences, goals, key learnings)",
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
