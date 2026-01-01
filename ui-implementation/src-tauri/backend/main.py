#!/usr/bin/env python3
"""
Roampal - Memory-Enhanced Chatbot
Intelligent chatbot with persistent memory and learning capabilities
"""

import asyncio
import logging
import os
import sys
import json
import uuid
from pathlib import Path

# Fix module imports for bundled production builds
if __name__ == "__main__":
    # Add current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time

# Windows subprocess support - Set ProactorEventLoop on Windows for asyncio.subprocess
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional

# Core services
from modules.embedding.embedding_service import EmbeddingService
from modules.memory.unified_memory_system import UnifiedMemorySystem
from modules.memory.types import ActionOutcome
from modules.llm.ollama_client import OllamaClient
from services.unified_image_service import UnifiedImageService
from config.feature_flag_validator import FeatureFlagValidator
from config.settings import DATA_PATH

# MCP (Model Context Protocol) server
from mcp.server import Server
from mcp.server.stdio import stdio_server

# API routers - Clean architecture with single agent router
from app.routers.agent_chat import router as agent_router
from app.routers.model_switcher import router as model_switcher_router
from app.routers.model_registry import router as model_registry_router
from app.routers.model_contexts import router as model_contexts_router
from app.routers.memory_visualization_enhanced import router as memory_enhanced_router
from app.routers.sessions import router as sessions_router
from app.routers.personality_manager import router as personality_router
from app.routers.backup import router as backup_router
from app.routers.memory_bank import router as memory_bank_router
from app.routers.system_health import router as system_health_router
from app.routers.data_management import router as data_management_router
from backend.api.book_upload_api import router as book_upload_router
from app.routers.mcp import router as mcp_router
from app.routers.mcp_servers import router as mcp_servers_router  # v0.2.5: External MCP tool servers

# Configure logging for production with rotation
# IMPORTANT: Logs go to AppData, NOT the install directory
# This prevents personal info (username in paths) from being included in releases
from logging.handlers import RotatingFileHandler
log_level = os.getenv('ROAMPAL_LOG_LEVEL', 'INFO')

# Create logs directory in AppData (same parent as DATA_PATH)
logs_dir = Path(DATA_PATH).parent / 'logs'
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / 'roampal.log'

# Create rotating file handler (10MB max, keep 3 backups)
file_handler = RotatingFileHandler(
    str(log_file_path),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=3,          # Keep 3 old files (roampal.log.1, .2, .3)
    encoding='utf-8'
)

# Check if running in MCP mode - if so, only log to file (not console/stderr)
# MCP uses stdio for JSON-RPC protocol, console logs would corrupt it
handlers = [file_handler]
if "--mcp" not in sys.argv:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger(__name__)

# MCP session-based caches for outcome scoring and cold-start injection
_mcp_search_cache = {}  # {session_id: {"doc_ids": [...], "query": "...", "timestamp": ...}}
_mcp_first_tool_call = set()  # Track first tool call per session for cold-start injection
_mcp_action_cache = {}  # {session_id: {"actions": [...], "last_context": str, "last_activity": datetime}} - Track tool actions for Action-Effectiveness KG

# MCP conversation boundary detection configuration
MCP_CACHE_EXPIRY_SECONDS = 600  # 10 minutes - clear cache if no activity (new conversation likely started)

def _should_clear_action_cache(session_id: str, new_context: str) -> tuple[bool, str]:
    """
    Detect conversation boundaries to prevent cross-conversation action scoring.

    MCP protocol doesn't provide conversation IDs, so we infer boundaries from:
    1. Time gaps (10+ minutes = likely new conversation)
    2. Context shifts (coding → fitness = likely new conversation)

    Returns:
        (should_clear: bool, reason: str)
    """
    if session_id not in _mcp_action_cache:
        return False, "No existing cache"

    cache = _mcp_action_cache[session_id]
    last_activity = cache.get("last_activity")
    last_context = cache.get("last_context")

    # Signal 1: TIME GAP - 10+ minutes since last tool call
    if last_activity:
        time_gap = (datetime.now() - last_activity).total_seconds()
        if time_gap > MCP_CACHE_EXPIRY_SECONDS:
            return True, f"time_gap={time_gap:.0f}s (>{MCP_CACHE_EXPIRY_SECONDS}s)"

    # Signal 2: CONTEXT SHIFT - Topic changed significantly
    # Ignore shifts to/from "general" (too noisy, not reliable)
    if last_context and new_context != last_context:
        if last_context != "general" and new_context != "general":
            return True, f"context_shift: {last_context} → {new_context}"

    return False, "same_conversation"

def _cache_action_with_boundary_check(session_id: str, action: "ActionOutcome", context_type: str):
    """
    Cache action with automatic conversation boundary detection.

    Clears cache if conversation boundary detected, then caches the new action.
    """
    # Check for conversation boundary
    should_clear, reason = _should_clear_action_cache(session_id, context_type)
    if should_clear:
        actions_discarded = len(_mcp_action_cache[session_id].get("actions", []))
        logger.warning(
            f"[MCP] Conversation boundary detected ({reason}). "
            f"Clearing {actions_discarded} cached actions to prevent cross-conversation scoring."
        )
        del _mcp_action_cache[session_id]

    # Initialize cache if needed
    if session_id not in _mcp_action_cache:
        _mcp_action_cache[session_id] = {
            "actions": [],
            "last_context": context_type,
            "last_activity": datetime.now()
        }

    # Add action
    _mcp_action_cache[session_id]["actions"].append(action)

    # Update metadata
    _mcp_action_cache[session_id]["last_activity"] = datetime.now()
    _mcp_action_cache[session_id]["last_context"] = context_type

async def _inject_cold_start_if_needed(session_id: str, tool_response: str, memory_system) -> str:
    """
    Prepend user profile to first tool response (MCP cold-start injection).

    Per architecture.md line 2143-2150: ALWAYS inject on first tool call (any tool).
    Uses Content KG to get top entities, retrieves their memory_bank documents.
    """
    if session_id not in _mcp_first_tool_call:
        _mcp_first_tool_call.add(session_id)

        try:
            context_summary = await asyncio.wait_for(
                memory_system.get_cold_start_context(limit=5),
                timeout=10.0
            )

            if context_summary:
                logger.info(f"[MCP] Cold-start injection for {session_id}: {len(context_summary)} chars")
                return f"""═══ KNOWN CONTEXT (auto-loaded) ═══
{context_summary}

═══ Tool Response ═══
{tool_response}"""
            else:
                logger.info(f"[MCP] No cold-start context available for {session_id}")
        except asyncio.TimeoutError:
            logger.warning(f"[MCP] Cold-start timeout for {session_id}")
        except Exception as e:
            logger.warning(f"[MCP] Cold-start failed for {session_id}: {e}")

    return tool_response

async def memory_promotion_task(memory: UnifiedMemorySystem):
    """Background task to promote valuable working memory to history"""
    while True:
        try:
            # Run promotion immediately on startup, then every 30 minutes
            logger.info("Running scheduled memory promotion...")

            # Promote valuable working memory (also cleans up items > 24h old)
            await memory.promote_valuable_working_memory()

            # Clean dead references from knowledge graph
            cleaned_refs = await memory._cleanup_kg_dead_references()
            if cleaned_refs > 0:
                logger.info(f"Cleaned {cleaned_refs} dead references from knowledge graph")

            # Update last check time
            memory._last_promotion_check = datetime.now()

            # Get stats for logging
            stats = memory.get_stats()
            logger.info(f"Memory promotion complete - Working: {stats['collections']['working']}, "
                       f"History: {stats['collections']['history']}, "
                       f"Patterns: {stats['collections']['patterns']}")

            # Wait 30 minutes before next check
            await asyncio.sleep(1800)  # 30 minutes in seconds

        except asyncio.CancelledError:
            logger.info("Memory promotion task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in memory promotion task: {e}", exc_info=True)
            # Don't crash the task on error
            await asyncio.sleep(60)  # Wait a minute before retrying

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Roampal services"""
    logger.info("""
    ╔══════════════════════════════════════╗
    ║         Roampal v1.0                 ║
    ║    Memory-Enhanced Chatbot           ║
    ╚══════════════════════════════════════╝
    """)

    try:
        # Try to initialize memory system with retry logic
        max_retries = 3
        retry_delay = 2  # Start with 2 seconds
        app.state.memory = None

        for attempt in range(max_retries):
            try:
                use_chromadb_server = os.getenv('CHROMADB_USE_SERVER', 'false').lower() == 'true'
                app.state.memory = UnifiedMemorySystem(
                    data_dir=DATA_PATH,
                    use_server=use_chromadb_server
                )
                await app.state.memory.initialize()
                logger.info(f"✓ UnifiedMemorySystem initialized (server mode: {use_chromadb_server})")

                # Initialize session cleanup manager
                sessions_dir = Path(DATA_PATH) / "sessions"
                sessions_dir.mkdir(parents=True, exist_ok=True)

                # Embedding service from memory system
                app.state.embedding_service = app.state.memory.embedding_service

                # Aliases for backward compatibility
                app.state.memory_collections = app.state.memory
                app.state.memory_adapter = app.state.memory
                break  # Success, exit retry loop

            except (ConnectionError, TimeoutError) as mem_error:
                if attempt < max_retries - 1:
                    logger.warning(f"Memory system connection failed (attempt {attempt + 1}/{max_retries}): {mem_error}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)  # Exponential backoff with max 30s cap
                else:
                    logger.warning(f"Memory system connection failed after {max_retries} attempts: {mem_error}")
                    logger.warning("⚠️  ChromaDB may not be running")

            except ImportError as mem_error:
                logger.warning(f"Memory system dependency missing: {mem_error}")
                break  # No point retrying import errors

            except Exception as mem_error:
                logger.error(f"Memory system initialization failed: {mem_error}", exc_info=True)
                break  # Unexpected error, don't retry

        # If memory system failed to initialize
        if app.state.memory is None:
            logger.warning("⚠️  IMPORTANT: Running without memory system!")
            logger.warning("⚠️  - Conversations will NOT be remembered")
            logger.warning("⚠️  - No learning or pattern recognition")
            logger.warning("⚠️  - To enable: Start ChromaDB with 'chroma run'")
            app.state.memory_collections = None
            app.state.memory_adapter = None

            # Try to create standalone embedding service (may fail if Ollama not installed)
            try:
                from modules.embedding.embedding_service import EmbeddingService
                app.state.embedding_service = EmbeddingService()
                logger.info("✓ Standalone embedding service initialized")
            except Exception as embed_error:
                logger.warning(f"⚠️  Embedding service unavailable: {embed_error}")
                logger.warning("⚠️  - This is expected if Ollama is not installed")
                logger.warning("⚠️  - Install Ollama from https://ollama.com to enable AI features")
                app.state.embedding_service = None

        # ==================== MULTI-PROVIDER LLM INITIALIZATION ====================
        logger.info("🔍 Detecting available LLM providers...")

        from app.routers.model_switcher import PROVIDERS, detect_provider, get_provider_models

        # Detect all running providers
        detected_providers = {}
        for provider_name, provider_config in PROVIDERS.items():
            provider_info = await detect_provider(provider_name, provider_config)
            if provider_info:
                models = await get_provider_models(provider_name, provider_config)
                provider_info['models'] = models
                detected_providers[provider_name] = provider_info
                logger.info(f"✓ Detected {provider_name} on port {provider_config['port']} with {len(models)} models")

        if not detected_providers:
            logger.warning("⚠️  No LLM providers detected")
            port_list = ', '.join([f"{name}:{cfg['port']}" for name, cfg in PROVIDERS.items()])
            logger.warning(f"   Checked ports: {port_list}")
            logger.warning("   Starting in setup mode - user will be prompted to install a provider")
            app.state.llm_client = None
        else:
            # Select provider (priority: configured > first available)
            configured_provider = os.getenv('ROAMPAL_LLM_PROVIDER', 'ollama')

            if configured_provider in detected_providers:
                active_provider = configured_provider
                logger.info(f"✓ Using configured provider: {active_provider}")
            else:
                active_provider = list(detected_providers.keys())[0]
                logger.info(f"✓ Configured provider '{configured_provider}' not available, using: {active_provider}")

            # Select model from active provider
            available_models = detected_providers[active_provider]['models']
            configured_model = os.getenv('ROAMPAL_LLM_OLLAMA_MODEL') or os.getenv('OLLAMA_MODEL')

            if configured_model and configured_model in available_models:
                selected_model = configured_model
                logger.info(f"✓ Using configured model: {selected_model}")
            elif available_models:
                selected_model = available_models[0]
                logger.info(f"✓ Using first available model: {selected_model}")
            else:
                # Active provider has no models - try other providers
                logger.warning(f"⚠️  Provider {active_provider} has no models - checking other providers")
                selected_model = None
                for other_provider, other_info in detected_providers.items():
                    if other_provider != active_provider and other_info['models']:
                        active_provider = other_provider
                        available_models = other_info['models']
                        selected_model = available_models[0]
                        logger.info(f"✓ Switched to {active_provider} - using first model: {selected_model}")
                        break

                if not selected_model:
                    logger.warning(f"⚠️  No models found in ANY provider")
                    app.state.llm_client = None

            # Initialize LLM client with selected provider
            if selected_model:
                provider_config = PROVIDERS[active_provider]
                base_url = f"http://localhost:{provider_config['port']}"

                from modules.llm.ollama_client import OllamaClient
                app.state.llm_client = OllamaClient()
                app.state.llm_client.base_url = base_url
                app.state.llm_client.model_name = selected_model
                app.state.llm_client.api_style = provider_config['api_style']

                await app.state.llm_client.initialize({
                    "ollama_base_url": base_url,
                    "ollama_model": selected_model
                })

                # Reinitialize httpx client with new base URL (CRITICAL for requests to work)
                if hasattr(app.state.llm_client, '_recycle_client'):
                    await app.state.llm_client._recycle_client()

                logger.info(f"✓ LLM initialized: {active_provider}:{selected_model} (API: {provider_config['api_style']})")

                # Save preferences
                os.environ['ROAMPAL_LLM_PROVIDER'] = active_provider
                os.environ['OLLAMA_MODEL'] = selected_model
                os.environ['ROAMPAL_LLM_OLLAMA_MODEL'] = selected_model
            else:
                app.state.llm_client = None

        # Store detected providers for API access
        app.state.detected_providers = detected_providers
        # ==================== END MULTI-PROVIDER INITIALIZATION ====================

        # Inject LLM service into memory system for outcome detection
        if app.state.memory and app.state.llm_client:
            app.state.memory.llm_service = app.state.llm_client
            logger.info("✓ LLM service connected to memory system")

        # Initialize book processor for document uploads (after LLM client)
        try:
            from modules.memory.smart_book_processor import SmartBookProcessor
            from config.settings import settings
            books_dir = settings.paths.get_book_folder_path()
            books_dir.mkdir(parents=True, exist_ok=True)

            # Get books collection from UnifiedMemorySystem
            books_adapter = None
            if app.state.memory and hasattr(app.state.memory, 'collections'):
                books_adapter = app.state.memory.collections.get('books')

            app.state.book_processor = SmartBookProcessor(
                data_dir=str(books_dir),
                chromadb_adapter=books_adapter,
                embedding_service=app.state.embedding_service
            )
            await app.state.book_processor.initialize()
            logger.info("✓ Book processor initialized")

            # Backfill timestamps for existing books
            await app.state.book_processor.backfill_book_timestamps()
            logger.info("✓ Book timestamps backfilled")
        except Exception as book_error:
            logger.warning(f"⚠️  Book processor unavailable: {book_error}")
            logger.warning("⚠️  - Document upload features will be disabled")
            app.state.book_processor = None

        # Image service
        try:
            app.state.image_service = UnifiedImageService(
                embedding_service=app.state.embedding_service,
                memory_adapter=None
            )
            logger.info("✓ Image service initialized")
        except Exception as img_error:
            logger.warning(f"⚠️  Image service unavailable: {img_error}")
            logger.warning("⚠️  - Image processing features will be disabled")
            app.state.image_service = None

        # Clean architecture - no longer need OGChatService
        # Agent router handles all chat operations with memory-enhanced responses
        logger.info("Using clean agent_chat router (memory-only mode)")

        # Validate feature flags for production
        from config.feature_flags import get_flag_manager
        flag_manager = get_flag_manager()
        current_flags = flag_manager.get_safe_config()
        is_production = os.getenv("ROAMPAL_PROFILE", "production") == "production"

        if is_production:
            # Sanitize flags for production safety
            sanitized_flags = FeatureFlagValidator.sanitize_for_production(current_flags)
            for key, value in sanitized_flags.items():
                if current_flags.get(key) != value:
                    flag_manager.set_flag(key, value)

        # Validate final configuration
        is_valid = FeatureFlagValidator.validate_and_log(flag_manager.get_safe_config(), is_production)
        if not is_valid and is_production:
            logger.error("Feature flag validation failed for production - applying safe defaults")
            safe_config = FeatureFlagValidator.get_safe_production_config()
            for key, value in safe_config.items():
                flag_manager.set_flag(key, value)


        # Initialize agent service ONCE at startup
        from app.routers.agent_chat import AgentChatService
        import app.routers.agent_chat as agent_chat_module

        agent_chat_module.agent_service = AgentChatService(
            memory=app.state.memory,
            llm=app.state.llm_client
        )
        logger.info("✓ Agent service initialized at startup")

        # v0.2.5: Initialize MCP Client Manager for external tool servers
        try:
            from modules.mcp_client.manager import MCPClientManager, set_mcp_manager
            mcp_manager = MCPClientManager(Path(DATA_PATH))
            await mcp_manager.initialize()
            set_mcp_manager(mcp_manager)
            app.state.mcp_manager = mcp_manager
            server_count = len([s for s in mcp_manager.servers.values() if s.status == "connected"])
            tool_count = len(mcp_manager.get_all_tools())
            if server_count > 0:
                logger.info(f"✓ MCP Client Manager initialized ({server_count} servers, {tool_count} external tools)")
            else:
                logger.info("✓ MCP Client Manager initialized (no servers configured)")
        except Exception as e:
            logger.warning(f"⚠️  MCP Client Manager initialization failed: {e}")
            app.state.mcp_manager = None

        if app.state.memory:
            logger.info("✓ Memory system successfully connected")
        else:
            logger.warning("⚠️  Memory system not connected - learning disabled")

        # No longer need enhanced chat initialization - agent router handles everything

        logger.info("✓ UnifiedMemorySystem is THE ONLY memory system")

        # Check system status and warn about missing components
        if not app.state.memory:
            logger.warning("\n" + "="*50)
            logger.warning("⚠️  SYSTEM RUNNING IN DEGRADED MODE")
            logger.warning("⚠️  Memory system is NOT available")
            logger.warning("⚠️  To enable full functionality:")
            logger.warning("⚠️  1. Start ChromaDB: chroma run --path ./data/chromadb --port 8003")
            logger.warning("⚠️  2. Restart Roampal")
            logger.warning("="*50 + "\n")

        # Model configuration status (already checked above, this is just informational)
        if not os.getenv('OLLAMA_MODEL') and not os.getenv('ROAMPAL_LLM_OLLAMA_MODEL') and not os.getenv('ROAMPAL_LLM_OLLAMA_MODEL'):
            logger.warning("⚠️  No LLM model configured in environment!")
            logger.warning("⚠️  Set with: ROAMPAL_LLM_OLLAMA_MODEL=<your-model> or OLLAMA_MODEL=<your-model>")

        logger.info("✓ Services initialized (check warnings above)")

        # Start background task for memory promotion if memory system is available
        if app.state.memory:
            asyncio.create_task(memory_promotion_task(app.state.memory))
            logger.info("✓ Memory promotion task started (runs every 30 minutes)")

    except Exception as e:
        logger.critical(f"System initialization failed: {e}", exc_info=True)
        # Set minimal state to prevent crashes
        app.state.memory = None
        app.state.llm_client = None
        app.state.chat_service = None
        logger.error("⚠️  System running in EMERGENCY mode - most features disabled")
        # Don't raise to allow health endpoint to work

    yield

    # Cleanup
    logger.info("Shutting down Roampal...")

    # Clean shutdown of memory system
    if hasattr(app.state, 'memory') and app.state.memory:
        try:
            logger.info("Cleaning up memory system...")
            await app.state.memory.cleanup()
            logger.info("✓ Memory system cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}", exc_info=True)

    # Close LLM client if exists
    if hasattr(app.state, 'llm_client') and app.state.llm_client:
        try:
            logger.info("Closing LLM client...")
            if hasattr(app.state.llm_client, 'close'):
                await app.state.llm_client.close()
            logger.info("✓ LLM client closed")
        except Exception as e:
            logger.error(f"Error closing LLM client: {e}", exc_info=True)

    # v0.2.5: Cleanup MCP connections
    if hasattr(app.state, 'mcp_manager') and app.state.mcp_manager:
        try:
            logger.info("Disconnecting MCP servers...")
            await app.state.mcp_manager.disconnect_all()
            logger.info("✓ MCP servers disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting MCP servers: {e}", exc_info=True)

    logger.info("✓ Roampal shutdown complete")

# Create app
app = FastAPI(
    title="Roampal",
    description="Memory-Enhanced Chatbot with Learning",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
allowed_origins = os.getenv(
    'ROAMPAL_ALLOWED_ORIGINS',
    os.getenv('ROAMPAL_ALLOWED_ORIGINS', 'http://localhost:5173,http://localhost:5174,http://localhost:3000,tauri://localhost')
).split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for WebSocket support in Tauri
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# WebSocket imports
from fastapi import WebSocket, WebSocketDisconnect

# Mount routers
# Clean architecture - single agent router handles all chat operations
app.include_router(agent_router, prefix="/api/agent", tags=["agent"])  # Main agent chat endpoint
app.include_router(agent_router, prefix="/api/chat", tags=["chat"])  # Compatibility alias for UI

# Supporting routers
app.include_router(model_switcher_router, prefix="/api/model", tags=["model-switch"])  # Runtime model switching & installation
app.include_router(model_registry_router)  # Unified model registry (uses /api/model prefix internally)
app.include_router(model_contexts_router)  # Model context window management (uses /api/model prefix internally)
app.include_router(memory_enhanced_router, prefix="/api/memory", tags=["memory"])  # Memory visualization
app.include_router(memory_bank_router)  # Memory bank (5th collection) - user control over persistent memories
app.include_router(sessions_router, prefix="/api/sessions", tags=["sessions"])  # Session management
app.include_router(personality_router)  # Personality customization (has its own prefix)
app.include_router(backup_router, prefix="/api/backup", tags=["backup"])  # Backup and restore with selective export
app.include_router(data_management_router)  # Data management (export/delete collections)
app.include_router(book_upload_router)  # Document processor (books collection)
app.include_router(system_health_router, prefix="/api/system", tags=["system"])  # System health and disk monitoring
app.include_router(mcp_router)  # MCP integrations (Claude Desktop, Claude Code, Cursor)
app.include_router(mcp_servers_router)  # v0.2.5: External MCP tool server management

@app.get("/health")
async def health():
    import os
    return {
        "status": "healthy",
        "service": "Roampal",
        "safe_mode": os.getenv('ROAMPAL_SAFE_MODE', os.getenv('ROAMPAL_SAFE_MODE', 'not set')),
        "safe_mode_enabled": os.getenv('ROAMPAL_SAFE_MODE', os.getenv('ROAMPAL_SAFE_MODE', 'false')).lower() == 'true'
    }


# v0.2.8: Update Notification System
@app.get("/api/check-update")
async def check_update():
    """Check for available updates (called on app startup)"""
    try:
        from utils.update_checker import check_for_updates, get_current_version
        update_info = await check_for_updates()
        if update_info:
            return {"available": True, "current_version": get_current_version(), **update_info}
        return {"available": False, "current_version": get_current_version()}
    except Exception as e:
        logger.debug(f"[UPDATE] Check failed: {e}")
        return {"available": False, "error": str(e)}

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        if hasattr(app.state, 'memory') and app.state.memory:
            stats = app.state.memory.get_stats()
            return {
                "fragments": sum(stats["collections"].values()),
                "collections": stats["collections"],
                "mode": "single-user",
                "learning": True
            }
        else:
            logger.warning("Memory system not available for stats")
            return {"fragments": 0, "mode": "single-user", "status": "degraded"}
    except AttributeError as e:
        logger.error(f"AttributeError getting stats: {e}")
        return {"fragments": 0, "mode": "single-user", "error": "memory_not_initialized"}
    except Exception as e:
        logger.error(f"Unexpected error getting stats: {e}", exc_info=True)
        return {"fragments": 0, "mode": "single-user", "error": "internal_error"}

@app.get("/api/metrics")
async def get_metrics():
    """Get performance metrics"""
    from services.metrics_service import get_metrics
    metrics = get_metrics()
    return metrics.get_summary()


# Simple rate limiting
rate_limit_storage = defaultdict(lambda: deque(maxlen=10000))
RATE_LIMIT = 10000  # requests per minute (increased for benchmarking)


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Simple rate limiting middleware - 100 requests per minute per session"""
    # Skip rate limiting for health, metrics, and WebSocket endpoints
    if request.url.path in ["/health", "/api/metrics", "/api/stats"] or "/ws/" in request.url.path:
        return await call_next(request)

    # Get session identifier (use session_id from headers or IP)
    session_id = request.headers.get("X-Session-Id", str(request.client.host))

    # Get current minute
    current_minute = datetime.now().replace(second=0, microsecond=0)

    # Get request history for this session
    request_times = rate_limit_storage[session_id]

    # Count requests in current minute
    recent_requests = sum(1 for t in request_times if t >= current_minute)

    if recent_requests >= RATE_LIMIT:
        from fastapi import HTTPException
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait before making more requests.")

    # Add current request
    request_times.append(datetime.now())

    # Process request
    response = await call_next(request)
    return response

@app.get("/api/backup")
async def export_backup():
    """Export memory system backup"""
    try:
        if hasattr(app.state, 'memory') and app.state.memory:
            backup = await app.state.memory.export_backup()
            return JSONResponse(content=backup)
        else:
            raise HTTPException(status_code=503, detail="Memory system not available")
    except Exception as e:
        logger.error(f"Backup export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/restore")
async def import_backup(backup_data: Dict[str, Any]):
    """Restore memory system from backup"""
    try:
        if hasattr(app.state, 'memory') and app.state.memory:
            success = await app.state.memory.import_backup(backup_data)
            if success:
                return {"status": "success", "message": "Backup restored successfully"}
            else:
                raise HTTPException(status_code=400, detail="Failed to restore backup")
        else:
            raise HTTPException(status_code=503, detail="Memory system not available")
    except Exception as e:
        logger.error(f"Backup restore failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))



# Unified WebSocket endpoint for all conversation updates
@app.websocket("/ws/conversation/{conversation_id}")
async def websocket_conversation(websocket: WebSocket, conversation_id: str):
    """Unified WebSocket endpoint for conversation updates"""
    try:
        await websocket.accept()
        # Store connection
        if not hasattr(app.state, 'websockets'):
            app.state.websockets = {}
        app.state.websockets[conversation_id] = websocket

        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "conversation_id": conversation_id
        })

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
            else:
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "handshake":
                        # Sync memory system if available
                        if hasattr(app.state, 'memory') and app.state.memory:
                            app.state.memory.conversation_id = conversation_id
                            logger.info(f"WebSocket synced memory conversation to {conversation_id}")
                except:
                    pass
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        logger.warning(f"WebSocket error for conversation {conversation_id}: {e}")
    finally:
        # Clean up connection
        if hasattr(app.state, 'websockets') and conversation_id in app.state.websockets:
            del app.state.websockets[conversation_id]



async def run_mcp_server():
    """Run Roampal as MCP server for AI tool integrations"""
    logger.info("[MCP] Starting Roampal MCP Server...")

    # Initialize memory system (use embedded ChromaDB for MCP mode)
    data_path = Path(DATA_PATH)  # Local reference for use throughout MCP server
    memory = UnifiedMemorySystem(data_dir=str(DATA_PATH), use_server=False)
    await memory.initialize()

    # Pre-warm bundled embedding model (paraphrase-multilingual-mpnet-base-v2)
    # Loads model on startup to avoid ~30s delay on first search
    logger.info("[MCP] Pre-warming bundled embedding model (paraphrase-multilingual-mpnet-base-v2)...")
    try:
        await memory.embedding_service.embed_text("test")
        logger.info("[MCP] ✓ Bundled embedding model ready")
    except Exception as e:
        logger.warning(f"[MCP] Embedding pre-warm failed (first search will be slow): {e}")

    # Initialize MCP Session Manager with memory reference (CRITICAL for automatic outcome detection)
    from modules.mcp.session_manager import MCPSessionManager
    from modules.mcp.client_detector import detect_mcp_client, get_client_display_name

    mcp_session_manager = MCPSessionManager(DATA_PATH, memory)
    logger.info("[MCP] Session manager initialized with automatic outcome detection")

    # Create MCP server
    server = Server("roampal-memory")

    # Import MCP types for proper typing
    import mcp.types as types
    from typing import Any

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        """List available MCP tools"""
        return [
            types.Tool(
                name="search_memory",
                description="""Search your persistent memory. Use when you need details beyond what get_context_insights returned.

WHEN TO SEARCH:
• User says "remember", "I told you", "we discussed" → search immediately
• get_context_insights recommended a collection → search that collection
• You need more detail than the context provided

WHEN NOT TO SEARCH:
• General knowledge questions (use your training)
• get_context_insights already gave you the answer

Collections: working (24h then auto-promotes), history (30d scored), patterns (permanent scored), memory_bank (permanent), books (permanent docs)
Omit 'collections' parameter for auto-routing (recommended).""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query - use the users EXACT words/phrases, do NOT simplify or extract keywords"
                        },
                        "collections": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["books", "working", "history", "patterns", "memory_bank"]},
                            "description": "Which collections to search. Omit for auto-routing (recommended). Manual: books, working, history, patterns, memory_bank",
                            "default": None
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results (1-20)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["relevance", "recency", "score"],
                            "description": "Sort order. 'recency' for temporal queries like 'last thing we did'. Auto-detected if omitted.",
                            "default": None
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional filters. Use sparingly. Examples: timestamp='2025-11-12', last_outcome='worked', has_code=true",
                            "additionalProperties": True
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="add_to_memory_bank",
                description="""Store PERMANENT facts (user identity, preferences, goals, learned strategies).

Store: Learning about user, discovering what works, tracking progress
Don't: Session transcripts (auto-captured), temporary tasks

Examples: "User's name is X", "User prefers Y style", "Full queries work better than keywords for this user"

Note: memory_bank facts are NOT auto-scored like search results. They persist until archived.
Use this for stable user info, not session learnings (those go in record_response).""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact to remember"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "Categories: identity, preference, goal, project, system_mastery, agent_growth"},
                        "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "How critical (0.0-1.0)"},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7, "description": "How certain (0.0-1.0)"}
                    },
                    "required": ["content"]
                }
            ),
            types.Tool(
                name="update_memory",
                description="Update existing memory when information changes or needs correction.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "old_content": {"type": "string", "description": "Old/incorrect fact to find"},
                        "new_content": {"type": "string", "description": "Corrected/updated fact"}
                    },
                    "required": ["old_content", "new_content"]
                }
            ),
            types.Tool(
                name="archive_memory",
                description="Archive outdated/irrelevant memories from memory_bank.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory to archive (semantic match)"}
                    },
                    "required": ["content"]
                }
            ),
            types.Tool(
                name="get_context_insights",
                description="""Search your memory before responding. Returns what you know about this user/topic.

WORKFLOW (follow these steps):
1. get_context_insights(query) ← YOU ARE HERE
2. Read the context returned
3. search_memory() if you need more details
4. Respond to user
5. record_response() to complete

Returns: Known facts, past solutions, recommended collections, tool stats.
Fast lookup (5-10ms) - no embedding search, just pattern matching.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query/topic you're considering (use user's exact words)"
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="record_response",
                description="""Complete the interaction. Call this after responding to the user.

WORKFLOW:
1. get_context_insights() ✓
2. search_memory() if needed ✓
3. Respond to user ✓
4. record_response() ← YOU ARE HERE (completes the interaction)

Parameters:
• key_takeaway: 1-2 sentence summary of what happened
• outcome: "worked" | "failed" | "partial" | "unknown"
• related: (Optional) Which search results were actually helpful (positions 1, 2, 3 or doc_ids)

OUTCOME DETECTION (read user's reaction):
✓ worked = user satisfied, says thanks, moves on
✗ failed = user corrects you, says "no", "that's wrong", provides the right answer
~ partial = user says "kind of" or takes some but not all of your answer
? unknown = no clear signal from user

⚠️ CRITICAL - "failed" OUTCOMES ARE ESSENTIAL:
• If user says you were wrong → outcome="failed"
• If memory you retrieved was outdated → outcome="failed"
• If user had to correct you → outcome="failed"
• If you gave advice that didn't help → outcome="failed"

Failed outcomes are how bad memories get deleted. Without them, wrong info persists forever.
Don't default to "worked" just to be optimistic. Wrong memories MUST be demoted.

SELECTIVE SCORING (optional):
If you retrieved 5 memories but only used 2, specify which with related=[1, 3].
Unrelated memories get 0 (neutral) - they're not penalized, just skipped.

This closes the loop. Without it, the system can't learn what worked OR what didn't.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key_takeaway": {
                            "type": "string",
                            "description": "1-2 sentence semantic summary of this exchange"
                        },
                        "outcome": {
                            "type": "string",
                            "enum": ["worked", "failed", "partial", "unknown"],
                            "description": "How helpful was your response based on user's reaction",
                            "default": "unknown"
                        },
                        "related": {
                            "type": "array",
                            "items": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
                            "description": "Which results were helpful. Use positions (1, 2, 3) or doc_ids. Omit to score all.",
                            "default": None
                        }
                    },
                    "required": ["key_takeaway"]
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        """Handle tool calls"""
        # Log all tool calls for analytics (non-blocking)
        session_id = detect_mcp_client()
        client_name = get_client_display_name(session_id)
        logger.info(f"[MCP] Tool called: {name} from {client_name}")

        # Record tool call to analytics file (background, non-blocking)
        async def log_tool_call():
            try:
                analytics_file = data_path / "mcp_tool_calls.jsonl"
                with open(analytics_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "tool": name,
                        "session_id": session_id,
                        "client": client_name,
                        "timestamp": datetime.now().isoformat(),
                        "args_summary": {k: str(v)[:100] for k, v in arguments.items()}  # Truncate long args
                    }) + "\n")
            except Exception as e:
                logger.error(f"[MCP] Failed to log tool call: {e}")

        asyncio.create_task(log_tool_call())

        # Detect context type for Action-Effectiveness KG tracking (v0.2.1)
        session_id = detect_mcp_client()
        context_type = "general"  # Default fallback

        # Get recent conversation for context detection
        session_file = data_path / "mcp_sessions" / f"{session_id}.json"
        recent_conv = []
        if session_file.exists():
            try:
                session_data = json.loads(session_file.read_text(encoding="utf-8"))
                turns = session_data.get("turns", [])[-5:]  # Last 5 turns
                for turn in turns:
                    user_msg = turn.get("user_message", "")
                    ai_msg = turn.get("ai_response", "")
                    if user_msg:
                        recent_conv.append({"role": "user", "content": user_msg})
                    if ai_msg:
                        recent_conv.append({"role": "assistant", "content": ai_msg})
            except Exception as e:
                logger.warning(f"[MCP] Failed to load session for context detection: {e}")

        # Detect context for this interaction
        try:
            context_type = await memory.detect_context_type(
                system_prompts=[],
                recent_messages=recent_conv
            )
            logger.info(f"[MCP] Context detected: {context_type}")
        except Exception as e:
            logger.warning(f"[MCP] Context detection failed: {e}")

        try:
            if name == "search_memory":
                query = arguments.get("query")
                # Fix: "all" is not a valid collection name - pass None to trigger KG routing
                collections = arguments.get("collections", None)
                if collections == ["all"]:
                    collections = None
                # Handle both string and int (Claude Desktop sometimes sends "5" instead of 5)
                limit = int(arguments.get("limit", 5)) if arguments.get("limit") else 5
                # Extract metadata filters
                metadata = arguments.get("metadata", None)

                # v0.2.9: sort_by parameter with auto-detection
                sort_by = arguments.get("sort_by", None)

                # Auto-detect temporal queries if sort_by not specified
                if sort_by is None:
                    temporal_keywords = [
                        "last", "recent", "yesterday", "today", "earlier",
                        "previous", "before", "when did", "how long ago",
                        "last time", "previously", "lately", "just now"
                    ]
                    query_lower = query.lower()
                    if any(kw in query_lower for kw in temporal_keywords):
                        sort_by = "recency"
                        logger.info(f"[MCP] search_memory: Auto-detected temporal query, using recency sort")

                # Quick check: if memory system isn't initialized, return early (avoids slow embedding model loading)
                if not memory.initialized:
                    return types.CallToolResult(
                        content=[types.TextContent(
                            type="text",
                            text=f"No results found for '{query}' in all collections.\n\nNote: Memory system is empty. Upload documents or store memories first to enable search."
                        )]
                    )

                # Log routing decision (will be None if KG should route)
                if collections is None:
                    logger.info(f"[MCP] search_memory: KG will route query '{query[:50]}'")
                else:
                    logger.info(f"[MCP] search_memory: LLM specified collections: {collections}")

                if metadata:
                    logger.info(f"[MCP] search_memory: Using metadata filters: {metadata}")

                # Wrap search with timeout to prevent MCP hanging (ChromaDB can be slow)
                try:
                    results = await asyncio.wait_for(
                        memory.search(query=query, collections=collections, limit=limit, metadata_filters=metadata),
                        timeout=25.0  # 25 seconds - less than MCP's ~240s timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[MCP] search_memory timed out after 25s for query: {query}")
                    return types.CallToolResult(
                        content=[types.TextContent(
                            type="text",
                            text="⚠️ Search timed out (25s). Embedding model is loading on first use (takes ~1 minute). Please wait and try again."
                        )],
                        isError=True
                    )

                # v0.2.9: Apply sort_by if specified
                if results and sort_by:
                    if sort_by == "recency":
                        # Sort by timestamp (newest first)
                        def get_timestamp(r):
                            meta = r.get('metadata', {})
                            ts = meta.get('timestamp') or meta.get('created_at') or ''
                            return ts if ts else ''
                        results = sorted(results, key=get_timestamp, reverse=True)
                        logger.info(f"[MCP] search_memory: Sorted {len(results)} results by recency")
                    elif sort_by == "score":
                        # Sort by outcome score (highest first)
                        def get_score(r):
                            meta = r.get('metadata', {})
                            return float(meta.get('score', 0.5))
                        results = sorted(results, key=get_score, reverse=True)
                        logger.info(f"[MCP] search_memory: Sorted {len(results)} results by score")
                    # sort_by == "relevance" is default (no re-sorting needed, vector similarity order)

                # Cache doc_ids from scorable collections for outcome-based scoring
                # Per architecture.md line 572-573: Cache doc_ids + query + collections for record_response scoring
                cached_doc_ids = []
                result_collections = set()  # Track which collections actually returned results (ALL, for KG routing)
                if results:
                    for r in results:
                        metadata = r.get('metadata', {})
                        collection = r.get('collection') or metadata.get('collection', 'unknown')
                        doc_id = r.get('doc_id') or r.get('id')

                        # Track ALL collections for KG routing updates (architecture.md line 1088-1104)
                        result_collections.add(collection)

                        # Cache ALL doc_ids for Action KG tracking (v0.2.6 - unified with internal system)
                        # Books and memory_bank now tracked for doc-level effectiveness insights
                        if doc_id:
                            cached_doc_ids.append(doc_id)

                # v0.2.9: Build position mapping for selective scoring (related=[1, 3] support)
                positions = {}
                for idx, doc_id in enumerate(cached_doc_ids, 1):
                    positions[idx] = doc_id

                _mcp_search_cache[session_id] = {
                    "doc_ids": cached_doc_ids,
                    "positions": positions,  # v0.2.9: Position -> doc_id mapping for related param
                    "query": query,
                    "collections": list(result_collections),  # Cache ACTUAL collections that returned results
                    "timestamp": datetime.now()
                }
                logger.info(f"[MCP] Cached {len(cached_doc_ids)} doc_ids from query '{query[:50]}' (result collections: {result_collections}, positions: 1-{len(cached_doc_ids)})")

                if not results:
                    collection_str = ", ".join(collections) if collections != ["all"] else "all collections"
                    text = f"No results found for '{query}' in {collection_str}.\n\nNote: Make sure you have uploaded documents or stored memories first."
                else:
                    text = f"Found {len(results)} result(s) for '{query}':\n\n"
                    for i, r in enumerate(results[:5], 1):
                        # Try both 'content' and 'text' fields (different adapters use different names)
                        content = r.get('content') or r.get('text', '')
                        # v0.2.8: Return full content (was truncated to 300 chars)
                        content_preview = content if content else '[No content]'
                        # Get collection from metadata or root level
                        metadata = r.get('metadata', {})
                        collection = r.get('collection') or metadata.get('collection', 'unknown')

                        # Extract metadata for LLM context (v0.2.3 enhancement)
                        score = metadata.get('score')
                        uses = metadata.get('uses', 0)
                        timestamp = metadata.get('timestamp')
                        last_outcome = metadata.get('last_outcome')
                        doc_id = r.get('doc_id') or r.get('id', '')

                        # Build metadata line for LLM scoring decisions
                        meta_parts = []
                        if score is not None:
                            meta_parts.append(f"score:{score:.2f}")
                        if uses > 0:
                            meta_parts.append(f"uses:{uses}")
                        if last_outcome:
                            meta_parts.append(f"last:{last_outcome}")
                        if timestamp:
                            try:
                                from datetime import datetime as dt
                                if isinstance(timestamp, str):
                                    ts = dt.fromisoformat(timestamp.replace('Z', '+00:00'))
                                else:
                                    ts = timestamp
                                age_days = (datetime.now(ts.tzinfo) if ts.tzinfo else datetime.now() - ts).days if hasattr(ts, 'days') else (datetime.now() - ts).days
                                if age_days == 0:
                                    meta_parts.append("age:today")
                                elif age_days == 1:
                                    meta_parts.append("age:1d")
                                elif age_days < 7:
                                    meta_parts.append(f"age:{age_days}d")
                                elif age_days < 30:
                                    meta_parts.append(f"age:{age_days//7}w")
                                else:
                                    meta_parts.append(f"age:{age_days//30}mo")
                            except:
                                pass  # Skip age if parsing fails

                        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
                        id_hint = f" [id:{doc_id}]" if doc_id else ""
                        text += f"{i}. [{collection}]{meta_line}{id_hint} {content_preview}\n\n"

                # Apply cold-start injection if this is first tool call
                text = await _inject_cold_start_if_needed(session_id, text, memory)

                # Track action for Action-Effectiveness KG (will be scored on record_response)
                collections_used = list(result_collections) if result_collections else (collections if collections else ["all"])
                for coll in collections_used:
                    action = ActionOutcome(
                        action_type="search_memory",
                        context_type=context_type,
                        outcome="unknown",  # Will be set when record_response is called
                        action_params={"query": query, "limit": limit},
                        collection=coll if coll != "all" else None,
                        doc_id=cached_doc_ids[0] if cached_doc_ids else None  # v0.2.6: Track first result for doc-level insights
                    )
                    _cache_action_with_boundary_check(session_id, action, context_type)

                logger.info(f"[MCP] Cached {len(collections_used)} search_memory actions (context={context_type}, total_cached={len(_mcp_action_cache[session_id]['actions'])})")

                return types.CallToolResult(content=[types.TextContent(type="text", text=text)])

            elif name == "add_to_memory_bank":
                content = arguments.get("content")
                tags = arguments.get("tags", [])
                importance = arguments.get("importance", 0.7)
                confidence = arguments.get("confidence", 0.7)
                session_id = detect_mcp_client()

                doc_id = await memory.store_memory_bank(text=content, tags=tags, importance=importance, confidence=confidence)

                text = f"Added to memory bank (ID: {doc_id})"
                text = await _inject_cold_start_if_needed(session_id, text, memory)

                # Track action for Action-Effectiveness KG
                action = ActionOutcome(
                    action_type="create_memory",
                    context_type=context_type,
                    outcome="unknown",
                    action_params={"content_preview": content[:50]},
                    doc_id=doc_id,
                    collection="memory_bank"
                )
                _cache_action_with_boundary_check(session_id, action, context_type)
                logger.info(f"[MCP] Cached create_memory action (context={context_type}, total_cached={len(_mcp_action_cache[session_id]['actions'])})")

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=text)]
                )

            elif name == "update_memory":
                old_content = arguments.get("old_content", "")
                new_content = arguments.get("new_content", "")
                session_id = detect_mcp_client()

                # Find the old memory by semantic search
                results = await memory.search_memory_bank(query=old_content, limit=1, include_archived=False)

                if results:
                    doc_id = results[0].get("id")
                    await memory.update_memory_bank(doc_id=doc_id, new_text=new_content, reason="mcp_update")
                    logger.info(f"[MCP] Updated memory: {doc_id}")

                    text = f"Updated memory (ID: {doc_id})"
                    text = await _inject_cold_start_if_needed(session_id, text, memory)

                    # Track action for Action-Effectiveness KG
                    action = ActionOutcome(
                        action_type="update_memory",
                        context_type=context_type,
                        outcome="unknown",
                        action_params={"new_content_preview": new_content[:50]},
                        doc_id=doc_id,
                        collection="memory_bank"
                    )
                    _cache_action_with_boundary_check(session_id, action, context_type)
                    logger.info(f"[MCP] Cached update_memory action (context={context_type}, total_cached={len(_mcp_action_cache[session_id]['actions'])})")

                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=text)]
                    )
                else:
                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text="Memory not found for update")],
                        isError=True
                    )

            elif name == "archive_memory":
                content = arguments.get("content", "")
                session_id = detect_mcp_client()

                # Find memory by semantic search
                results = await memory.search_memory_bank(query=content, limit=1, include_archived=False)

                if results:
                    doc_id = results[0].get("id")
                    await memory.archive_memory_bank(doc_id=doc_id, reason="mcp_archive")
                    logger.info(f"[MCP] Archived memory: {doc_id}")

                    text = f"Archived memory (ID: {doc_id})"
                    text = await _inject_cold_start_if_needed(session_id, text, memory)

                    # Track action for Action-Effectiveness KG
                    action = ActionOutcome(
                        action_type="archive_memory",
                        context_type=context_type,
                        outcome="unknown",
                        action_params={"content_preview": content[:50]},
                        doc_id=doc_id,
                        collection="memory_bank"
                    )
                    _cache_action_with_boundary_check(session_id, action, context_type)
                    logger.info(f"[MCP] Cached archive_memory action (context={context_type}, total_cached={len(_mcp_action_cache[session_id]['actions'])})")

                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=text)]
                    )
                else:
                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text="Memory not found for archiving")],
                        isError=True
                    )

            elif name == "get_context_insights":
                query = arguments.get("query", "")
                session_id = detect_mcp_client()

                logger.info(f"[MCP] get_context_insights: query='{query[:50]}'")

                # Get recent conversation from session file
                session_file = data_path / "mcp_sessions" / f"{session_id}.json"
                recent_conv = []
                system_prompts = []
                if session_file.exists():
                    try:
                        session_data = json.loads(session_file.read_text(encoding="utf-8"))
                        # Extract last 5 turns for context
                        turns = session_data.get("turns", [])[-5:]
                        for turn in turns:
                            # Convert session turn format to conversation format
                            user_msg = turn.get("user_message", "")
                            ai_msg = turn.get("ai_response", "")
                            if user_msg:
                                recent_conv.append({"role": "user", "content": user_msg})
                            if ai_msg:
                                recent_conv.append({"role": "assistant", "content": ai_msg})
                    except Exception as e:
                        logger.error(f"[MCP] Failed to load session file: {e}")

                # Detect context type for action-effectiveness guidance (v0.2.1)
                context_type = await memory.detect_context_type(
                    system_prompts=system_prompts,
                    recent_messages=recent_conv
                )

                # Analyze conversation for organic insights (Content KG)
                try:
                    org_context = await memory.analyze_conversation_context(
                        current_message=query,
                        recent_conversation=recent_conv,
                        conversation_id=session_id
                    )

                    # Check action-effectiveness for tool guidance (Action-Effectiveness KG)
                    action_stats = []
                    available_actions = ["search_memory", "create_memory", "update_memory", "archive_memory"]
                    collections_to_check = [None, "books", "working", "history", "patterns", "memory_bank"]  # Check all 5 tiers + wildcard

                    for action in available_actions:
                        # Check across all collections
                        for collection in collections_to_check:
                            stats = memory.get_action_effectiveness(context_type, action, collection)
                            # Only surface stats if we have enough data (10+ uses)
                            if stats and stats['total_uses'] >= 10:
                                coll_suffix = f" on {collection}" if collection else ""
                                action_stats.append({
                                    'action': action,
                                    'collection': collection,
                                    'success_rate': stats['success_rate'],
                                    'uses': stats['total_uses'],
                                    'suffix': coll_suffix
                                })

                    # v0.2.6: Get directive insights using new helper methods
                    matched_concepts = org_context.get('matched_concepts', []) if org_context else []

                    # Get routing recommendations from Routing KG
                    routing = memory.get_tier_recommendations(matched_concepts)

                    # Get relevant memory_bank facts from Content KG
                    relevant_facts = []
                    if matched_concepts:
                        try:
                            relevant_facts = await memory.get_facts_for_entities(matched_concepts[:5], limit=2)
                        except Exception as e:
                            logger.warning(f"[MCP] Failed to get facts for entities: {e}")

                    # Format insights
                    has_actionable_insights = (
                        org_context and (
                            org_context.get('relevant_patterns') or
                            org_context.get('past_outcomes') or
                            org_context.get('proactive_insights')
                        )
                    )

                    if not has_actionable_insights and not action_stats and not relevant_facts:
                        text = f"No relevant patterns found in knowledge graph for '{query}'.\n\nThis appears to be a new type of query. The system will learn from your interactions and build patterns over time."
                    else:
                        response = f"═══ KNOWN CONTEXT (Topic: {context_type}) ═══\n\n"

                        # v0.2.6: DIRECTIVE - Recommended actions first
                        response += "📌 RECOMMENDED ACTIONS:\n"
                        if routing and routing.get('top_collections'):
                            collections = routing['top_collections'][:2]
                            match_count = routing.get('match_count', 0)
                            confidence = routing.get('confidence_level', 'exploration')
                            if match_count > 0:
                                response += f"  • search_memory(collections={collections}) - {match_count} patterns matched ({confidence} confidence)\n"
                            else:
                                response += f"  • search_memory() - auto-routing will select collections\n"
                        response += "  • record_response(outcome=...) after reply - required for learning\n\n"

                        # v0.2.6: Surface relevant memory_bank facts
                        # v0.2.8: Full content, no truncation
                        if relevant_facts:
                            response += "💡 YOU ALREADY KNOW THIS (from memory_bank):\n"
                            for fact in relevant_facts:
                                content = fact.get('content', '')
                                eff = fact.get('effectiveness')
                                eff_str = f" ({int(eff['success_rate']*100)}% helpful)" if eff and eff.get('total_uses', 0) >= 3 else ""
                                response += f"  • \"{content}\"{eff_str}\n"
                            response += "\n"

                        if org_context and org_context.get('relevant_patterns'):
                            response += "📋 PAST EXPERIENCE:\n"
                            for pattern in org_context['relevant_patterns'][:3]:
                                response += f"  • {pattern['insight']}\n"
                                response += f"    Collection: {pattern['collection']}, Score: {pattern['score']:.2f}, Uses: {pattern['uses']}\n"
                                response += f"    → {pattern['text']}\n\n"

                        if org_context and org_context.get('past_outcomes'):
                            response += "⚠️ PAST FAILURES TO AVOID:\n"
                            for outcome in org_context['past_outcomes'][:2]:
                                response += f"  • {outcome['insight']}\n\n"

                        # Action-Effectiveness KG stats (v0.2.1 - informational only)
                        if action_stats:
                            # Sort by uses (most frequently used first)
                            action_stats.sort(key=lambda x: x['uses'], reverse=True)
                            top_stats = action_stats[:5]  # Show top 5 most-used

                            response += "📊 TOOL STATS:\n"
                            for stat in top_stats:
                                suffix = stat.get('suffix', '')
                                success_rate = int(stat['success_rate']*100)
                                uses = stat['uses']
                                response += f"  • {stat['action']}(){suffix}: {success_rate}% success ({uses} uses)\n"

                            response += "\n"

                        if org_context and org_context.get('proactive_insights'):
                            response += "💡 SEARCH RECOMMENDATIONS:\n"
                            for insight in org_context['proactive_insights'][:3]:
                                rec = insight.get('recommendation', '')
                                if rec:
                                    response += f"  • {rec}\n"

                        if org_context and org_context.get('topic_continuity'):
                            for topic in org_context['topic_continuity'][:1]:
                                response += f"\n🔗 {topic['insight']}\n"

                        # v0.2.6: Explicit completion reminder (open loop)
                        response += "\n═══ TO COMPLETE THIS INTERACTION ═══\n"
                        response += "After responding → record_response(key_takeaway=\"...\", outcome=\"worked|failed|partial\")"
                        text = response

                except Exception as e:
                    logger.error(f"[MCP] get_context_insights error: {e}", exc_info=True)
                    text = f"Error analyzing context: {str(e)}"

                # Inject cold-start context if first tool call
                text = await _inject_cold_start_if_needed(session_id, text, memory)

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=text)]
                )

            elif name == "record_response":
                # Per architecture.md line 574-586: Store semantic summary, score CURRENT learning
                key_takeaway = arguments.get("key_takeaway")
                outcome = arguments.get("outcome", "unknown")  # Default to "unknown"
                related = arguments.get("related", None)  # v0.2.9: Selective scoring
                session_id = detect_mcp_client()

                logger.info(f"[MCP] record_response: session={session_id}, outcome={outcome}, related={related}")
                logger.info(f"[MCP] Takeaway: {key_takeaway[:100]}...")

                # Get cached query from last search (for KG routing updates)
                cached_query = ""
                if session_id in _mcp_search_cache:
                    cached_query = _mcp_search_cache[session_id].get("query", "")

                # Calculate initial score based on outcome (architecture.md line 576-577)
                initial_scores = {
                    "worked": 0.7,
                    "failed": 0.2,
                    "partial": 0.55,
                    "unknown": 0.5
                }
                initial_score = initial_scores.get(outcome, 0.5)

                # Store CURRENT semantic summary with initial score
                doc_id = await memory.store(
                    text=key_takeaway,
                    collection="working",
                    metadata={
                        "role": "learning",
                        "source": session_id,
                        "score": initial_score,
                        "query": cached_query,  # Enables KG routing updates
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                logger.info(f"[MCP] Stored learning with doc_id={doc_id}, initial_score={initial_score}")

                # Score cached memories from last search (architecture.md line 578)
                # v0.2.9: Support selective scoring via `related` parameter
                cached_memories_scored = 0
                skipped_memories = 0
                if outcome in ["worked", "failed", "partial"] and session_id in _mcp_search_cache:
                    cached = _mcp_search_cache[session_id]
                    cached_doc_ids = cached.get("doc_ids", [])
                    positions = cached.get("positions", {})  # v0.2.9: Position -> doc_id mapping

                    # v0.2.9: Resolve `related` to doc_ids (supports positions and doc_ids)
                    doc_ids_to_score = None  # None = score all (default)
                    if related is not None and len(related) > 0:
                        doc_ids_to_score = set()
                        for item in related:
                            if isinstance(item, int):
                                # Position-based (e.g., 1, 2, 3)
                                doc_id_from_pos = positions.get(item)
                                if doc_id_from_pos:
                                    doc_ids_to_score.add(doc_id_from_pos)
                                    logger.info(f"[MCP] Resolved position {item} → {doc_id_from_pos}")
                                else:
                                    logger.warning(f"[MCP] Position {item} not found in cache (valid: 1-{len(cached_doc_ids)})")
                            elif isinstance(item, str):
                                # Doc ID-based (e.g., "history_abc123")
                                if item in cached_doc_ids:
                                    doc_ids_to_score.add(item)
                                    logger.info(f"[MCP] Using doc_id: {item}")
                                else:
                                    logger.warning(f"[MCP] Doc ID {item} not found in cache")

                        if not doc_ids_to_score:
                            # All specified items were invalid - fall back to scoring all
                            logger.warning(f"[MCP] No valid doc_ids resolved from related={related}, falling back to score all")
                            doc_ids_to_score = None

                    logger.info(f"[MCP] Scoring strategy: {'selective' if doc_ids_to_score else 'all'} ({len(doc_ids_to_score) if doc_ids_to_score else len(cached_doc_ids)} memories)")

                    # Get collections that were searched
                    cached_collections = cached.get("collections", ["all"])

                    for cached_doc_id in cached_doc_ids:
                        # v0.2.9: Only score if in related list (or related not specified)
                        if doc_ids_to_score is not None and cached_doc_id not in doc_ids_to_score:
                            skipped_memories += 1
                            continue

                        try:
                            await memory.record_outcome(doc_id=cached_doc_id, outcome=outcome)
                            cached_memories_scored += 1
                        except Exception as e:
                            logger.warning(f"[MCP] Failed to score cached memory {cached_doc_id}: {e}")

                    # KG learns which collections worked for this query type
                    # Update KG routing for each collection that returned results
                    if cached_query and outcome in ["worked", "failed", "partial"] and cached_collections:
                        for collection in cached_collections:
                            try:
                                await memory._update_kg_routing(cached_query, collection, outcome)
                                logger.info(f"[MCP] KG routing updated: '{cached_query[:50]}' → {collection} ({outcome})")
                            except Exception as e:
                                logger.warning(f"[MCP] Failed to update KG routing: {e}")

                # Score Action-Effectiveness KG (v0.2.1) - Update tool effectiveness stats
                actions_scored = 0
                if outcome in ["worked", "failed", "partial"] and session_id in _mcp_action_cache:
                    cache = _mcp_action_cache[session_id]
                    cached_actions = cache.get("actions", [])
                    logger.info(f"[MCP] Scoring {len(cached_actions)} cached actions for Action-Effectiveness KG with outcome={outcome}")

                    for action in cached_actions:
                        # Update outcome from "unknown" to actual result
                        action.outcome = outcome
                        try:
                            await memory.record_action_outcome(action)
                            actions_scored += 1
                            logger.info(f"[MCP] Action-Effectiveness KG updated: {action.context_type}|{action.action_type}|{action.collection or '*'} → {outcome}")
                        except Exception as e:
                            logger.warning(f"[MCP] Failed to record action outcome: {e}")

                # Always clear caches after record_response (architecture.md line 549)
                # Even if outcome="unknown", prevents stale cache buildup
                if session_id in _mcp_search_cache:
                    del _mcp_search_cache[session_id]
                    logger.debug(f"[MCP] Cleared search cache for {session_id}")

                if session_id in _mcp_action_cache:
                    del _mcp_action_cache[session_id]
                    logger.debug(f"[MCP] Cleared action cache for {session_id}")

                # Record to session file
                result = await mcp_session_manager.record_exchange(
                    session_id=session_id,
                    user_msg=key_takeaway,  # Store semantic summary
                    assistant_msg="",  # Not used in MCP mode
                    doc_id=doc_id,
                    outcome=outcome
                )

                # Build response text (v0.2.3: enriched summary)
                client_name = get_client_display_name(session_id)
                response_text = f"✓ Learning recorded for {client_name}\n"
                response_text += f"Doc ID: {doc_id}\n"
                response_text += f"Initial score: {initial_score} (outcome={outcome})\n"

                # v0.2.9: Enhanced reporting for selective scoring
                if cached_memories_scored > 0 or skipped_memories > 0:
                    if skipped_memories > 0:
                        response_text += f"Scored {cached_memories_scored} memories (skipped {skipped_memories} unrelated)\n"
                    else:
                        response_text += f"Scored {cached_memories_scored} cached memories\n"

                if actions_scored > 0:
                    response_text += f"Updated {actions_scored} tool effectiveness stats\n"

                # v0.2.3: Add system learning summary for LLM context
                response_text += "\n--- System Learning Summary ---\n"
                response_text += f"Your takeaway stored in working memory (will promote to history if score stays ≥0.7)\n"
                if outcome == "worked":
                    response_text += "Outcome 'worked': High initial score (0.7) - this learning is on track for promotion\n"
                elif outcome == "failed":
                    response_text += "Outcome 'failed': Low initial score (0.2) - consider what went wrong for future\n"
                elif outcome == "partial":
                    response_text += "Outcome 'partial': Medium score (0.55) - needs more positive outcomes to promote\n"
                else:
                    response_text += "Outcome 'unknown': Neutral score (0.5) - will adjust based on future use\n"

                if cached_memories_scored > 0:
                    if skipped_memories > 0:
                        response_text += f"Selective scoring: {cached_memories_scored} memories scored with '{outcome}', {skipped_memories} skipped (neutral)\n"
                    else:
                        response_text += f"The {cached_memories_scored} memories from your last search were also scored with '{outcome}'\n"

                logger.info(f"[MCP] {client_name} recorded learning: {doc_id}")

                # Apply cold-start injection if this is first tool call
                response_text = await _inject_cold_start_if_needed(session_id, response_text, memory)

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=response_text)]
                )

            else:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"Unknown tool: {name}")],
                    isError=True
                )

        except Exception as e:
            logger.error(f"[MCP] Tool call error for {name}: {e}", exc_info=True)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True
            )

    # Run MCP server via stdio
    logger.info("[MCP] Server initialized with 6 tools: search_memory, add_to_memory_bank, update_memory, archive_memory, get_context_insights, record_response")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def print_mcp_banner():
    """Print informative banner to console (stderr) for MCP mode.

    MCP uses stdout for JSON-RPC, so all console output MUST go to stderr.
    This gives users visibility that MCP server is running.
    """
    import sys
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                 ROAMPAL MCP SERVER RUNNING                   ║
╠══════════════════════════════════════════════════════════════╣
║  Status: Connected to AI tool (Claude Desktop, Cursor, etc)  ║
║                                                              ║
║  This window provides memory to your AI assistant.           ║
║  Closing this window will disconnect memory access.          ║
║                                                              ║
║  To stop: Close this window or press Ctrl+C                  ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner, file=sys.stderr)


if __name__ == "__main__":
    # Check for MCP mode
    if "--mcp" in sys.argv:
        print_mcp_banner()  # Show informative console message
        logger.info("[MCP] Running in MCP server mode")
        asyncio.run(run_mcp_server())
    else:
        # Run FastAPI server (normal mode)
        # Port configurable via ROAMPAL_API_PORT env var (default: 8001)
        import uvicorn
        api_port = int(os.getenv('ROAMPAL_API_PORT', '8001'))
        logger.info(f"Starting FastAPI server on port {api_port}")
        uvicorn.run(app, host="127.0.0.1", port=api_port)