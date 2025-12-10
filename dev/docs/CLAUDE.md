
# CLAUDE.md

AI coding assistant instructions for Roampal. For system design, see `ARCHITECTURE.md`.

## Quick Start

```bash
# Backend
python main.py                    # Start FastAPI (port 8000)

# Frontend
cd ui-implementation && npm run tauri:dev  # Desktop app

# Validate
python validate_system.py         # Check everything works
```

## Core Rules

### MUST (Break these = broken system)
- Run `validate_system.py` after core changes
- Use `conversation_id` in all chat requests
- Call `/api/chat/switch-conversation` when switching
- Include `TransparencyContext` in new operations

### NEVER (Will cause problems)
- Modify `unified_memory_system.py` directly (use API)
- Create new routers (extend `agent_chat.py`)
- Add files to `modules/advanced/`
- Change ChromaDB collections directly
- Leave `console.log` in production

### SHOULD (Best practices)
- Promote memories every 20 messages
- Check existing utilities before creating
- Delete dead code immediately, just be careful its actually 
 dead not just disconnected.


## Code Principles

### Core Philosophy
"Optimize for the reader, not the writer" - Google Style Guide
Code will be read 100x more than written. Make it obvious.

### SIMPLICITY (KISS)
- Single purpose per function (<20 lines)
- Obvious names > clever names
- If it needs a comment to explain WHAT, refactor it
- Prefer composition over inheritance

### ROBUSTNESS (Fail Fast)
- Guard clauses at function entry
- Validate at system boundaries ONLY
- Return early on invalid input
- Explicit error messages with context

### NO REDUNDANCY (DRY)
- Extract when used 3+ times (Rule of Three)
- Single source of truth: `config/settings.py`
- Check for existing utilities first
- Constants in ONE place only

### NO PREMATURE CODE (YAGNI)
- Build what's needed NOW
- Delete dead code immediately
- No "just in case" abstractions
- Features on demand, not speculation

### TECHNICAL DEBT PREVENTION
- No TODO without issue # and date
- Refactor on 3rd duplication
- Fix broken windows immediately
- Tests alongside implementation
- Update docs with code changes

### CODE SMELLS TO AVOID
- Classes >300 lines → Split
- Nesting >3 levels → Extract method
- Magic numbers → Named constants
- Duplicate logic → Extract utility
- Mixed abstractions → Separate layers

## File Map

```
Feature → File:Line (approx)

Chat Processing → app/routers/agent_chat.py:58-650
Memory Search → modules/memory/unified_memory_system.py:234-380
Memory Storage → modules/memory/unified_memory_system.py:156-230
Memory Promotion → main.py:59-95
Model Switch → app/routers/model_switcher.py:78-150
Session CRUD → app/routers/sessions.py:45-180
Book Upload → backend/api/book_upload_api.py:32-213

UI Chat → ui-implementation/src/components/ConnectedChat.tsx:200-800
UI Messages → ui-implementation/src/components/EnhancedChatMessage.tsx:50-250
UI Sidebar → ui-implementation/src/components/Sidebar.tsx:100-400
Chat Store → ui-implementation/src/stores/useChatStore.ts:150-600
WebSocket → ui-implementation/src/stores/useChatStore.ts:800-950
```

## Common Tasks

### Add API Endpoint
1. Add to `app/routers/agent_chat.py`
2. Update OpenAPI schema if needed
3. Run `python validate_system.py`
4. Test with `curl localhost:8000/docs`

### Fix Memory Issues
1. Check ChromaDB: `ls data/chromadb/`
2. Review sessions: `ls data/sessions/*.jsonl`
3. Test memory: `python -c "from modules.memory.unified_memory_system import *; ..."`
4. Check stats: `/api/memory/stats`

### Debug Chat Flow
1. Enable debug logs: `export DEBUG=true`
2. Watch backend: `tail -f logs/app.log`
3. Check WebSocket: Browser DevTools → Network → WS
4. Verify memory context in response metadata

### Add UI Component
1. Create in `ui-implementation/src/components/`
2. Follow existing patterns (check `ConnectedChat.tsx`)
3. Update store if needed: `src/stores/useChatStore.ts`
4. Run: `npm run lint && npm run build`

### Switch Models
```bash
# Check available models from all providers
curl localhost:8000/api/model/providers/all/models

# Switch model (auto-detects provider)
curl -X POST localhost:8000/api/model/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name":"qwen2.5:7b", "provider":"ollama"}'

# Or for LM Studio
curl -X POST localhost:8000/api/model/switch \
  -H "Content-Type: application/json" \
  -d '{"model_name":"qwen2.5-7b-instruct", "provider":"lmstudio"}'
```

## Testing

```bash
# Backend
pytest tests/test_core.py -v      # Core functionality
pytest tests/test_memory.py -v    # Memory system
python validate_system.py          # Full validation

# Frontend
cd ui-implementation
npm run lint                      # Linting
npm run build                     # Type check
npm run test                      # Unit tests

# Integration
python test_clean_system.py       # End-to-end test
```

## Current State

```yaml
# System Configuration
Providers: Ollama (port 11434) | LM Studio (port 1234)
Model: Auto-detected from available providers
Mode: Chat with Memory (memory + learning enabled)
Port: 8000 (backend), 5174 (frontend dev)

# Feature Flags
Memory: true
Knowledge Graph: true
Outcome Detection: true
Autonomy: false (chat-only system)

# Memory Status (typical)
Working: ~60 items (24hr retention)
History: ~5 items (30-day retention)
Patterns: 0-2 items (score-based)
Books: Persistent reference docs

# Active Services
- ChromaDB in embedded mode (data/chromadb/)
- LLM Providers: Ollama (11434) or LM Studio (1234)
- SQLite for outcomes tracking
```

## Project Structure

```
Roampal/
├── app/routers/          # API endpoints
│   └── agent_chat.py     # THE main router
├── modules/memory/       # Memory system
│   └── unified_memory_system.py  # THE memory system
├── services/            # Service layer
│   └── transparency_context.py   # Action tracking
├── ui-implementation/   # Tauri + React frontend
│   ├── src/components/  # React components
│   └── src/stores/      # Zustand stores
└── data/               # Persistent storage
    ├── chromadb/       # Vector DB
    ├── sessions/       # JSONL conversations
    └── books/          # Uploaded docs
```

## Gotchas & Known Issues

### Memory Not Promoting
- Check 20-message threshold reached
- Verify `/api/chat/switch-conversation` called
- Look for `auto_promote()` in logs

### WebSocket Disconnects
- Check `session_id` consistency
- Verify no CORS issues (port 8000/5173)
- Browser console for WS errors

### Model Switching Fails
- Ensure model downloaded: `ollama list`
- Check `.env` has correct OLLAMA_HOST
- Restart backend after model pull

### UI Not Updating
- Check WebSocket connection in DevTools
- Verify `conversation_id` in requests
- Clear browser cache if needed

### Tests Failing
- ChromaDB data exists? Check data/chromadb/
- Ollama running? Port 11434
- Clean state: `rm -rf data/chromadb && python main.py`

## API Quick Reference

```python
# Chat
POST /api/agent/chat
  {"message": "...", "conversation_id": "...", "mode": "learning"}

# Memory
GET /api/memory/stats
POST /api/chat/switch-conversation
  {"old_conversation_id": "...", "new_conversation_id": "..."}

# Models
GET /api/model/available
POST /api/model/switch {"model_name": "..."}

# Sessions
GET /api/sessions/list
GET /api/sessions/{id}
POST /api/chat/create-conversation

# Books
POST /api/book-upload/upload (multipart)
GET /api/book-upload/books
DELETE /api/book-upload/book/{id}
```

## Environment Variables

```bash
# Required
OLLAMA_HOST=http://localhost:11434
ROAMPAL_LLM_OLLAMA_MODEL=qwen3:8b  # Or OLLAMA_MODEL for fallback

# Core Features
ROAMPAL_ENABLE_MEMORY=true
ROAMPAL_ENABLE_OUTCOME_TRACKING=true
ROAMPAL_ENABLE_KG=true

# Optional
ROAMPAL_LOG_LEVEL=INFO
ROAMPAL_PORT=8000
```

## When to Edit What

| Need to... | Edit this file |
|------------|---------------|
| Add chat endpoint | `app/routers/agent_chat.py` |
| Change memory logic | Use API, not `unified_memory_system.py` |
| Add UI component | `ui-implementation/src/components/` |
| Update chat store | `src/stores/useChatStore.ts` |
| Add feature | Extend `app/routers/agent_chat.py` |
| Change model | Use `/api/model/switch` endpoint |
| Debug memory | Check `/api/memory/stats` first |

---
*For architecture details and design philosophy, see `ARCHITECTURE.md`*