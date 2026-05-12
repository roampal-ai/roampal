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
if sys.platform == "win32":
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
from modules.llm.ollama_client import OllamaClient
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
from app.routers.attachments import router as attachments_router  # v0.3.3 §4 Defect 4
from backend.api.book_upload_api import router as book_upload_router
from app.routers.mcp import router as mcp_router
from app.routers.mcp_servers import (
    router as mcp_servers_router,
)  # v0.2.5: External MCP tool servers

# Configure logging for production with rotation
# IMPORTANT: Logs go to AppData, NOT the install directory
# This prevents personal info (username in paths) from being included in releases
from logging.handlers import RotatingFileHandler

log_level = os.getenv("ROAMPAL_LOG_LEVEL", "INFO")

# Create logs directory in AppData (same parent as DATA_PATH)
logs_dir = Path(DATA_PATH).parent / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
log_file_path = logs_dir / "roampal.log"

# Create rotating file handler (10MB max, keep 3 backups)
file_handler = RotatingFileHandler(
    str(log_file_path),
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=3,  # Keep 3 old files (roampal.log.1, .2, .3)
    encoding="utf-8",
)

# Check if running in MCP mode - if so, only log to file (not console/stderr)
# MCP uses stdio for JSON-RPC protocol, console logs would corrupt it
handlers = [file_handler]
if "--mcp" not in sys.argv:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)

# v0.3.3 §12: _mcp_search_cache and _mcp_action_cache removed — they were
# leftovers from the v0.2.6 Action-KG plan (KG removed in v0.3.1). Both had
# no readers; record_action_outcome was a no-op stub. Cold-start dedup state
# kept because _inject_cold_start_if_needed actually reads it.
_mcp_first_tool_call: set[str] = set()


async def _inject_cold_start_if_needed(
    session_id: str, tool_response: str, memory_system
) -> str:
    """
    Prepend user profile to first tool response (MCP cold-start injection).

    Per architecture.md line 2143-2150: ALWAYS inject on first tool call (any tool).
    Uses Content KG to get top entities, retrieves their memory_bank documents.
    """
    if session_id not in _mcp_first_tool_call:
        _mcp_first_tool_call.add(session_id)

        try:
            # get_cold_start_context returns tuple: (profile_string, doc_ids, raw_facts)
            # mode="mcp" uses add_to_memory_bank tool name (vs create_memory for internal)
            result = await asyncio.wait_for(
                memory_system.get_cold_start_context(limit=5, mode="mcp"), timeout=10.0
            )

            # Unpack the tuple - we only need the profile string for injection
            profile_string = result[0] if result else None

            if profile_string:
                logger.info(
                    f"[MCP] Cold-start injection for {session_id}: {len(profile_string)} chars"
                )
                return f"""═══ KNOWN CONTEXT (auto-loaded) ═══
{profile_string}

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

            # KG cleanup removed in v0.3.1

            # Update last check time
            memory._last_promotion_check = datetime.now()

            # Get stats for logging
            stats = memory.get_stats()
            logger.info(
                f"Memory promotion complete - Working: {stats['collections']['working']}, "
                f"History: {stats['collections']['history']}, "
                f"Patterns: {stats['collections']['patterns']}"
            )

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
                use_chromadb_server = (
                    os.getenv("CHROMADB_USE_SERVER", "false").lower() == "true"
                )
                app.state.memory = UnifiedMemorySystem(
                    data_dir=DATA_PATH, use_server=use_chromadb_server
                )
                await app.state.memory.initialize()
                logger.info(
                    f"✓ UnifiedMemorySystem initialized (server mode: {use_chromadb_server})"
                )

                # v0.3.3: Clean up phantom ChromaDB entries from pre-fix hard deletes
                await app.state.memory._startup_cleanup_memory_bank_phantoms()

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
                    logger.warning(
                        f"Memory system connection failed (attempt {attempt + 1}/{max_retries}): {mem_error}"
                    )
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(
                        retry_delay * 2, 30
                    )  # Exponential backoff with max 30s cap
                else:
                    logger.warning(
                        f"Memory system connection failed after {max_retries} attempts: {mem_error}"
                    )
                    logger.warning("⚠️  ChromaDB may not be running")

            except ImportError as mem_error:
                logger.warning(f"Memory system dependency missing: {mem_error}")
                break  # No point retrying import errors

            except Exception as mem_error:
                logger.error(
                    f"Memory system initialization failed: {mem_error}", exc_info=True
                )
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
                logger.warning(
                    "⚠️  - Install Ollama from https://ollama.com to enable AI features"
                )
                app.state.embedding_service = None

        # ==================== MULTI-PROVIDER LLM INITIALIZATION ====================
        logger.info("🔍 Detecting available LLM providers...")

        from app.routers.model_switcher import (
            PROVIDERS,
            detect_provider,
            get_provider_models,
        )

        # Detect all running providers
        detected_providers = {}
        for provider_name, provider_config in PROVIDERS.items():
            provider_info = await detect_provider(provider_name, provider_config)
            if provider_info:
                models = await get_provider_models(provider_name, provider_config)
                provider_info["models"] = models
                detected_providers[provider_name] = provider_info
                logger.info(
                    f"✓ Detected {provider_name} on port {provider_config['port']} with {len(models)} models"
                )

        if not detected_providers:
            logger.warning("⚠️  No LLM providers detected")
            port_list = ", ".join(
                [f"{name}:{cfg['port']}" for name, cfg in PROVIDERS.items()]
            )
            logger.warning(f"   Checked ports: {port_list}")
            logger.warning(
                "   Starting in setup mode - user will be prompted to install a provider"
            )
            app.state.llm_client = None
        else:
            # v0.3.2 (0b follow-up): Persisted-selection file also drives
            # which PROVIDER we boot with. Earlier logic locked provider to
            # the env-var default and then skipped persisted if its provider
            # didn't match — meaning if you switched to LM Studio last
            # session, restart would fall back to first-available Ollama.
            configured_provider = os.getenv("ROAMPAL_LLM_PROVIDER", "ollama")

            from app.routers.model_switcher import (
                load_main_model_config,
                save_main_model_config,
            )
            from utils.startup_model_selection import (
                select_startup_model, should_migrate_env_to_json,
                select_active_provider,
            )
            persisted = load_main_model_config()

            active_provider, provider_source = select_active_provider(
                detected_providers=detected_providers,
                persisted=persisted,
                configured_env_provider=configured_provider,
            )
            if provider_source == "persisted":
                logger.info(
                    f"✓ Using persisted provider: {active_provider} "
                    f"(model: {persisted['model']})"
                )
            elif provider_source == "env":
                logger.info(f"✓ Using configured provider: {active_provider}")
            elif provider_source == "first_detected":
                logger.info(
                    f"✓ Configured provider '{configured_provider}' not available, using: {active_provider}"
                )

            # Select model from active provider
            available_models = detected_providers[active_provider]["models"]
            configured_model = os.getenv("ROAMPAL_LLM_OLLAMA_MODEL") or os.getenv(
                "OLLAMA_MODEL"
            )

            # Migration: if no JSON exists yet but env vars are set, seed
            # the JSON once so future switches persist via the new path.
            # Only migrate when the env model is actually installed — prevents
            # writing stale env vars (e.g. OLLAMA_MODEL=codellama:latest) when
            # that model isn't pulled anymore.
            if should_migrate_env_to_json(persisted, configured_model, available_models=available_models):
                try:
                    save_main_model_config(
                        model=configured_model, provider=active_provider
                    )
                    persisted = {"model": configured_model, "provider": active_provider}
                except Exception as seed_err:
                    logger.warning(f"Could not seed main_model_config.json: {seed_err}")

            selected_model, source = select_startup_model(
                available_models=available_models,
                configured_env_model=configured_model,
                persisted=persisted,
                active_provider=active_provider,
            )
            if source == "env":
                logger.info(f"✓ Using configured model (env): {selected_model}")
            elif source == "persisted":
                logger.info(f"✓ Using persisted model: {selected_model}")
            elif source == "fallback":
                logger.info(f"✓ Using first available model: {selected_model}")
            else:
                # Active provider has no models - try other providers
                logger.warning(
                    f"⚠️  Provider {active_provider} has no models - checking other providers"
                )
                selected_model = None
                for other_provider, other_info in detected_providers.items():
                    if other_provider != active_provider and other_info["models"]:
                        active_provider = other_provider
                        available_models = other_info["models"]
                        selected_model = available_models[0]
                        logger.info(
                            f"✓ Switched to {active_provider} - using first model: {selected_model}"
                        )
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
                app.state.llm_client.api_style = provider_config["api_style"]

                await app.state.llm_client.initialize(
                    {"ollama_base_url": base_url, "ollama_model": selected_model}
                )

                # Reinitialize httpx client with new base URL (CRITICAL for requests to work)
                if hasattr(app.state.llm_client, "_recycle_client"):
                    await app.state.llm_client._recycle_client()

                logger.info(
                    f"✓ LLM initialized: {active_provider}:{selected_model} (API: {provider_config['api_style']})"
                )

                # Save preferences
                os.environ["ROAMPAL_LLM_PROVIDER"] = active_provider
                os.environ["OLLAMA_MODEL"] = selected_model
                os.environ["ROAMPAL_LLM_OLLAMA_MODEL"] = selected_model
            else:
                app.state.llm_client = None

        # Store detected providers for API access
        app.state.detected_providers = detected_providers
        # ==================== END MULTI-PROVIDER INITIALIZATION ====================

        # ==================== GPU VRAM DETECTION (v0.3.2 0g) ====================
        try:
            from utils.gpu_detection import detect_gpu
            gpu_info = detect_gpu()
            app.state.gpu_vram_gb = gpu_info.vram_gb
            app.state.gpu_count = gpu_info.count
            app.state.gpu_source = gpu_info.source
        except Exception as gpu_err:
            logger.warning(f"GPU detection failed: {gpu_err}")
            app.state.gpu_vram_gb = None
            app.state.gpu_count = 0
            app.state.gpu_source = "none"
        # ========================================================================

        # ==================== SIDECAR INITIALIZATION (v0.3.1) ====================
        app.state.sidecar_model = ""
        app.state.sidecar_provider = ""
        app.state.sidecar_client = None
        app.state.sidecar_last_error = ""  # v0.3.1.3: Track last sidecar failure for UI

        sidecar_config_path = Path(DATA_PATH) / "sidecar_config.json"

        # v0.3.2 (0f): First-run seeding. If no sidecar config exists yet,
        # mirror the chat model so single-GPU users don't accidentally end up
        # running two different models at once. Seeded before the existing
        # config-load branch so the rest of the lifespan sees a warm config.
        if not sidecar_config_path.exists() and app.state.llm_client:
            try:
                from app.routers.model_switcher import _atomic_write_json
                chat_model = app.state.llm_client.model_name
                chat_api = getattr(app.state.llm_client, "api_style", "ollama")
                chat_provider = "lmstudio" if chat_api == "openai" else "ollama"
                if chat_model:
                    _atomic_write_json(
                        sidecar_config_path,
                        {
                            "enabled": True,
                            "model": chat_model,
                            "provider": chat_provider,
                            "mirror_chat": True,
                        },
                    )
                    logger.info(
                        f"✓ Sidecar seeded to mirror chat: {chat_provider}/{chat_model}"
                    )
            except Exception as seed_err:
                logger.warning(f"Could not seed sidecar_config.json: {seed_err}")

        # v0.3.2 (0f) migration: existing v0.3.1 configs predate mirror_chat.
        # Infer a safe default from model equality so we don't silently flip
        # users who had distinct chat + sidecar models into mirror mode.
        #
        # v0.3.2 follow-up: defer migration when the chat LLM client hasn't
        # finished loading its model yet (e.g. provider-fallback in progress).
        # At that point chat_model == "" and bool("") would lock mirror_chat
        # permanently to False — the realign block below is gated on
        # mirror_chat=True, so the lock-in is unrecoverable without a manual
        # toggle. Leaving the key absent is safe: other readers treat missing
        # as falsy (same as False) and the migration retries next boot once
        # the chat client is populated.
        if sidecar_config_path.exists():
            try:
                existing = json.loads(sidecar_config_path.read_text(encoding="utf-8"))
                if "mirror_chat" not in existing:
                    chat_model = getattr(app.state.llm_client, "model_name", "") if app.state.llm_client else ""
                    if not chat_model:
                        logger.info(
                            "Sidecar mirror_chat migration deferred: chat model not loaded yet"
                        )
                    else:
                        matches = existing.get("model") == chat_model
                        existing["mirror_chat"] = matches
                        from app.routers.model_switcher import _atomic_write_json
                        _atomic_write_json(sidecar_config_path, existing)
                        logger.info(
                            "✓ Sidecar config migrated: mirror_chat=%s (sidecar=%s, chat=%s)",
                            matches, existing.get("model"), chat_model,
                        )
            except Exception as mig_err:
                logger.warning(f"Sidecar config migration skipped: {mig_err}")

        if sidecar_config_path.exists():
            try:
                sc = json.loads(sidecar_config_path.read_text(encoding="utf-8"))

                # v0.3.2 follow-up: enforce mirror_chat at BOOT, not just on
                # switch. If the user's persisted sidecar was the previous
                # chat model (now stale/deleted/swapped), align it with the
                # current chat model at startup so we don't boot a dead
                # sidecar. Re-persist so the file reflects actual state.
                chat_now = getattr(app.state.llm_client, "model_name", "") if app.state.llm_client else ""
                chat_api = getattr(app.state.llm_client, "api_style", "ollama") if app.state.llm_client else "ollama"
                chat_provider_now = "lmstudio" if chat_api == "openai" else "ollama"
                if sc.get("mirror_chat") and chat_now and (
                    sc.get("model") != chat_now or sc.get("provider") != chat_provider_now
                ):
                    from app.routers.model_switcher import _atomic_write_json
                    old_model = sc.get("model")
                    sc["enabled"] = True
                    sc["model"] = chat_now
                    sc["provider"] = chat_provider_now
                    _atomic_write_json(sidecar_config_path, sc)
                    logger.info(
                        "✓ Sidecar realigned at boot: mirror_chat=true forced %s → %s",
                        old_model, chat_now,
                    )

                if sc.get("enabled") and sc.get("model"):
                    from modules.llm.ollama_client import OllamaClient

                    sidecar_provider = sc.get("provider", "ollama")
                    if sidecar_provider == "lmstudio":
                        sc_url, sc_api = "http://localhost:1234", "openai"
                    else:
                        sc_url, sc_api = "http://localhost:11434", "ollama"

                    sidecar_client = OllamaClient()
                    await sidecar_client.initialize(
                        {
                            "ollama_base_url": sc_url,
                            "ollama_model": sc.get("model"),
                        }
                    )
                    sidecar_client.api_style = sc_api

                    app.state.sidecar_model = sc.get("model")
                    app.state.sidecar_provider = sidecar_provider
                    app.state.sidecar_client = sidecar_client
                    logger.info(
                        f"✓ Sidecar loaded: {sc.get('model')} ({sidecar_provider})"
                    )

                    # Wire TagService with LLM extraction function for benchmark-aligned tagging.
                    # v0.3.2 (Bug 4): Factory reads app.state.sidecar_client/model at CALL time,
                    # not at boot. See utils/sidecar_tag_wrapper.py for the rationale and test.
                    if app.state.memory and hasattr(app.state.memory, "_tag_service"):
                        from utils.sidecar_tag_wrapper import make_llm_tag_extractor

                        app.state.memory._tag_service.set_llm_extract_fn(
                            make_llm_tag_extractor(app.state)
                        )
                        logger.info(
                            "✓ TagService wired with LLM extraction (benchmark-aligned)"
                        )
            except Exception as e:
                logger.warning(f"⚠️ Failed to load sidecar config: {e}")
        # ==================== END SIDECAR INITIALIZATION ====================

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
            if app.state.memory and hasattr(app.state.memory, "collections"):
                books_adapter = app.state.memory.collections.get("books")

            app.state.book_processor = SmartBookProcessor(
                data_dir=str(books_dir),
                chromadb_adapter=books_adapter,
                embedding_service=app.state.embedding_service,
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
            sanitized_flags = FeatureFlagValidator.sanitize_for_production(
                current_flags
            )
            for key, value in sanitized_flags.items():
                if current_flags.get(key) != value:
                    flag_manager.set_flag(key, value)

        # Validate final configuration
        is_valid = FeatureFlagValidator.validate_and_log(
            flag_manager.get_safe_config(), is_production
        )
        if not is_valid and is_production:
            logger.error(
                "Feature flag validation failed for production - applying safe defaults"
            )
            safe_config = FeatureFlagValidator.get_safe_production_config()
            for key, value in safe_config.items():
                flag_manager.set_flag(key, value)

        # Initialize agent service ONCE at startup
        from app.routers.agent_chat import AgentChatService
        import app.routers.agent_chat as agent_chat_module

        agent_chat_module.agent_service = AgentChatService(
            memory=app.state.memory, llm=app.state.llm_client
        )
        logger.info("✓ Agent service initialized at startup")

        # v0.2.5: Initialize MCP Client Manager for external tool servers
        try:
            from modules.mcp_client.manager import MCPClientManager, set_mcp_manager

            mcp_manager = MCPClientManager(Path(DATA_PATH))
            await mcp_manager.initialize()
            set_mcp_manager(mcp_manager)
            app.state.mcp_manager = mcp_manager
            server_count = len(
                [s for s in mcp_manager.servers.values() if s.status == "connected"]
            )
            tool_count = len(mcp_manager.get_all_tools())
            if server_count > 0:
                logger.info(
                    f"✓ MCP Client Manager initialized ({server_count} servers, {tool_count} external tools)"
                )
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
            logger.warning("\n" + "=" * 50)
            logger.warning("⚠️  SYSTEM RUNNING IN DEGRADED MODE")
            logger.warning("⚠️  Memory system is NOT available")
            logger.warning("⚠️  To enable full functionality:")
            logger.warning(
                "⚠️  1. Start ChromaDB: chroma run --path ./data/chromadb --port 8003"
            )
            logger.warning("⚠️  2. Restart Roampal")
            logger.warning("=" * 50 + "\n")

        # Model configuration status (already checked above, this is just informational)
        if (
            not os.getenv("OLLAMA_MODEL")
            and not os.getenv("ROAMPAL_LLM_OLLAMA_MODEL")
            and not os.getenv("ROAMPAL_LLM_OLLAMA_MODEL")
        ):
            logger.warning("⚠️  No LLM model configured in environment!")
            logger.warning(
                "⚠️  Set with: ROAMPAL_LLM_OLLAMA_MODEL=<your-model> or OLLAMA_MODEL=<your-model>"
            )

        logger.info("✓ Services initialized (check warnings above)")

        # Start background task for memory promotion if memory system is available
        if app.state.memory:
            asyncio.create_task(memory_promotion_task(app.state.memory))
            logger.info("✓ Memory promotion task started (runs every 30 minutes)")

        # Start sidecar retry queue processor
        try:
            from modules.memory.sidecar_queue import (
                process_retry_queue,
                register_terminal_failure_callback,
                register_retry_success_callback,
            )

            # v0.3.3 Defect 14: wire the queue's outcome callbacks to app.state
            # so the UI's /api/model/sidecar/status indicator only alarms when
            # the queue actually drops a score (3-attempt budget exhausted),
            # not on the inevitable single-attempt transient failures the
            # queue is designed to absorb.
            def _on_terminal_failure(item: Dict[str, Any]) -> None:
                doc_id = item.get("doc_id", "unknown")
                task_type = item.get("task_type", "task")
                last_err = item.get("last_error", "unknown error")
                # Use the short form so the chat-header badge / status panel
                # stays readable; the full traceback is in the backend log.
                app.state.sidecar_last_error = (
                    f"Dropped {task_type} for {doc_id} after retries: "
                    f"{str(last_err)[:120]}"
                )
                logger.error(
                    f"[SIDECAR] Terminal failure surfaced to UI: {doc_id} ({task_type})"
                )

            def _on_retry_success(item: Dict[str, Any]) -> None:
                # Queue self-healed — clear any sticky error from a prior
                # terminal failure or from older code paths.
                if getattr(app.state, "sidecar_last_error", ""):
                    app.state.sidecar_last_error = ""
                    logger.info(
                        f"[SIDECAR] Retry self-healed — cleared sidecar_last_error "
                        f"(was set from earlier terminal failure)"
                    )

            register_terminal_failure_callback(_on_terminal_failure)
            register_retry_success_callback(_on_retry_success)

            asyncio.create_task(process_retry_queue())
            logger.info("✓ Sidecar retry queue processor started with status callbacks (Defect 14)")
        except ImportError as e:
            logger.warning(f"Sidecar queue module not available: {e}")

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
    if hasattr(app.state, "memory") and app.state.memory:
        try:
            logger.info("Cleaning up memory system...")
            await app.state.memory.cleanup()
            logger.info("✓ Memory system cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}", exc_info=True)

    # Close LLM client if exists
    if hasattr(app.state, "llm_client") and app.state.llm_client:
        try:
            logger.info("Closing LLM client...")
            if hasattr(app.state.llm_client, "close"):
                await app.state.llm_client.close()
            logger.info("✓ LLM client closed")
        except Exception as e:
            logger.error(f"Error closing LLM client: {e}", exc_info=True)

    # v0.2.5: Cleanup MCP connections
    if hasattr(app.state, "mcp_manager") and app.state.mcp_manager:
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
    lifespan=lifespan,
)

# CORS configuration
allowed_origins = os.getenv(
    "ROAMPAL_ALLOWED_ORIGINS",
    os.getenv(
        "ROAMPAL_ALLOWED_ORIGINS",
        "http://localhost:5173,http://localhost:5174,http://localhost:3000,tauri://localhost",
    ),
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins + ["https://tauri.localhost"],  # v0.3.1.3: No wildcard
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)

# WebSocket imports
from fastapi import WebSocket, WebSocketDisconnect

# Mount routers
# Clean architecture - single agent router handles all chat operations
app.include_router(
    agent_router, prefix="/api/agent", tags=["agent"]
)  # Main agent chat endpoint
app.include_router(
    agent_router, prefix="/api/chat", tags=["chat"]
)  # Compatibility alias for UI

# Supporting routers
app.include_router(
    model_switcher_router, prefix="/api/model", tags=["model-switch"]
)  # Runtime model switching & installation
app.include_router(
    model_registry_router
)  # Unified model registry (uses /api/model prefix internally)
app.include_router(
    model_contexts_router
)  # Model context window management (uses /api/model prefix internally)
app.include_router(
    memory_enhanced_router, prefix="/api/memory", tags=["memory"]
)  # Memory visualization
app.include_router(
    memory_bank_router
)  # Memory bank (5th collection) - user control over persistent memories
app.include_router(
    sessions_router, prefix="/api/sessions", tags=["sessions"]
)  # Session management
app.include_router(personality_router)  # Personality customization (has its own prefix)
app.include_router(
    backup_router, prefix="/api/backup", tags=["backup"]
)  # Backup and restore with selective export
app.include_router(
    data_management_router
)  # Data management (export/delete collections)
app.include_router(book_upload_router)  # Document processor (books collection)
app.include_router(
    system_health_router, prefix="/api/system", tags=["system"]
)  # System health and disk monitoring
app.include_router(mcp_router)  # MCP integrations (Claude Desktop, Claude Code, Cursor)
app.include_router(mcp_servers_router)  # v0.2.5: External MCP tool server management
app.include_router(attachments_router)  # v0.3.3 §4 Defect 4: image attachment bytes endpoint


@app.get("/health")
async def health():
    import os

    return {
        "status": "healthy",
        "service": "Roampal",
        "safe_mode": os.getenv(
            "ROAMPAL_SAFE_MODE", os.getenv("ROAMPAL_SAFE_MODE", "not set")
        ),
        "safe_mode_enabled": os.getenv(
            "ROAMPAL_SAFE_MODE", os.getenv("ROAMPAL_SAFE_MODE", "false")
        ).lower()
        == "true",
    }


# v0.2.8: Update Notification System
@app.get("/api/check-update")
async def check_update():
    """Check for available updates (called on app startup)"""
    try:
        from utils.update_checker import check_for_updates, get_current_version

        update_info = await check_for_updates()
        if update_info:
            return {
                "available": True,
                "current_version": get_current_version(),
                **update_info,
            }
        return {"available": False, "current_version": get_current_version()}
    except Exception as e:
        logger.debug(f"[UPDATE] Check failed: {e}")
        return {"available": False, "error": str(e)}


@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        if hasattr(app.state, "memory") and app.state.memory:
            stats = app.state.memory.get_stats()
            return {
                "fragments": sum(stats["collections"].values()),
                "collections": stats["collections"],
                "mode": "single-user",
                "learning": True,
            }
        else:
            logger.warning("Memory system not available for stats")
            return {"fragments": 0, "mode": "single-user", "status": "degraded"}
    except AttributeError as e:
        logger.error(f"AttributeError getting stats: {e}")
        return {
            "fragments": 0,
            "mode": "single-user",
            "error": "memory_not_initialized",
        }
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
try:
    RATE_LIMIT = int(os.getenv("ROAMPAL_RATE_LIMIT", "200"))
except (ValueError, TypeError):
    RATE_LIMIT = 200


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Simple rate limiting middleware - 100 requests per minute per session"""
    # Skip rate limiting for health, metrics, and WebSocket endpoints
    if (
        request.url.path in ["/health", "/api/metrics", "/api/stats"]
        or "/ws/" in request.url.path
    ):
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

        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making more requests.",
        )

    # Add current request
    request_times.append(datetime.now())

    # Process request
    response = await call_next(request)
    return response


@app.get("/api/backup")
async def export_backup():
    """Export memory system backup"""
    try:
        if hasattr(app.state, "memory") and app.state.memory:
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
        if hasattr(app.state, "memory") and app.state.memory:
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
        if not hasattr(app.state, "websockets"):
            app.state.websockets = {}
        app.state.websockets[conversation_id] = websocket

        # Send connection confirmation
        await websocket.send_json(
            {"type": "connected", "conversation_id": conversation_id}
        )

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
                        if hasattr(app.state, "memory") and app.state.memory:
                            app.state.memory.conversation_id = conversation_id
                            logger.info(
                                f"WebSocket synced memory conversation to {conversation_id}"
                            )
                except:
                    pass
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        logger.warning(f"WebSocket error for conversation {conversation_id}: {e}")
    finally:
        # Clean up connection
        if hasattr(app.state, "websockets") and conversation_id in app.state.websockets:
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
    logger.info(
        "[MCP] Pre-warming bundled embedding model (paraphrase-multilingual-mpnet-base-v2)..."
    )
    try:
        await memory.embedding_service.embed_text("test")
        logger.info("[MCP] ✓ Bundled embedding model ready")
    except Exception as e:
        logger.warning(
            f"[MCP] Embedding pre-warm failed (first search will be slow): {e}"
        )

    from modules.mcp.client_detector import detect_mcp_client, get_client_display_name

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
                            "description": "Search query - use the users EXACT words/phrases, do NOT simplify or extract keywords",
                        },
                        "collections": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "books",
                                    "working",
                                    "history",
                                    "patterns",
                                    "memory_bank",
                                ],
                            },
                            "description": "Which collections to search. Omit for auto-routing (recommended). Manual: books, working, history, patterns, memory_bank",
                            "default": None,
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results (1-20)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["relevance", "recency", "score"],
                            "description": "Sort order. 'recency' for temporal queries like 'last thing we did'. Auto-detected if omitted.",
                            "default": None,
                        },
                        "type": {
                            "type": "string",
                            "enum": ["fact", "summary"],
                            "description": "Filter by memory type: 'fact' for atomic facts, 'summary' for exchange summaries. Omit for all.",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional filters. Use sparingly. Examples: timestamp='2025-11-12', last_outcome='worked', has_code=true",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="add_to_memory_bank",
                description="""Store PERMANENT facts (user identity, preferences, goals, learned strategies).

Store: Learning about user, discovering what works, tracking progress
Don't: Session transcripts (auto-captured), temporary tasks

Examples: "User's name is X", "User prefers Y style", "Full queries work better than keywords for this user"

SIZE GUIDANCE:
• Keep facts AS SMALL AS POSSIBLE - aim for ~300 chars or less
• The first ~300 chars of each fact appear in cold start profile summaries
• Longer facts work but only the beginning shows on cold start - put the key info first
• Research dumps belong in books collection, not memory_bank
• If you notice massive facts (1000+ chars), offer to condense them
• One concept per fact - split multi-topic content into separate memories

Note: memory_bank facts are NOT auto-scored like search results. They persist until archived.
Use this for stable user info, not session learnings (those go in record_response).""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The fact to remember",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Categories: identity, preference, goal, project, system_mastery, agent_growth",
                        },
                        "noun_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": 'Content nouns for tag-routed retrieval (auto-extracted if omitted). Example: ["python", "asyncio"]',
                        },
                        "importance": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.7,
                            "description": "How critical (0.0-1.0)",
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 0.7,
                            "description": "How certain (0.0-1.0)",
                        },
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="update_memory",
                description="Update existing memory when information changes or needs correction.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "old_content": {
                            "type": "string",
                            "description": "Old/incorrect fact to find",
                        },
                        "new_content": {
                            "type": "string",
                            "description": "Corrected/updated fact",
                        },
                    },
                    "required": ["old_content", "new_content"],
                },
            ),
            types.Tool(
                name="archive_memory",
                description="Archive outdated/irrelevant memories from memory_bank.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Memory to archive (semantic match)",
                        }
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="get_context_insights",
                description="""⚠️ REQUIRED - Call this BEFORE answering the user's question.

This is how you access the memory system. Without it, you have no context about this user.

═══ WORKFLOW ═══
1. get_context_insights(query) ← YOU ARE HERE (start every interaction here)
2. Read the context returned
3. search_memory() if you need more details
4. Respond to user
5. record_response() to complete the loop

═══ WHAT YOU GET ═══
• User profile (identity, preferences, goals, projects)
• Relevant memories from past sessions
• Proven solutions ranked by success rate
• Doc IDs for scoring when you call record_response()

Fast lookup (5-10ms). Use the user's exact words as query.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query/topic you're considering (use user's exact words)",
                        }
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="record_response",
                description="""Store a key takeaway when the transcript alone won't capture important learning.

OPTIONAL - Only use for significant exchanges:
• Major decisions made
• Complex solutions that worked
• User corrections (what you got wrong and why)
• Important context that would be lost

Most routine exchanges don't need this - the transcript is enough.
NOT for permanent preferences or standing rules — use add_to_memory_bank for those.

Key takeaways start at 0.7 (user explicitly asked to remember = higher confidence).
Scoring happens automatically on subsequent turns: +0.2 worked, +0.05 partial, -0.3 failed.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key_takeaway": {
                            "type": "string",
                            "description": "1-2 sentence summary of the important learning",
                        },
                        "noun_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key TOPIC nouns from the takeaway. Use actual names not pronouns (skip he/she/they/user/assistant). Lowercase, 1-3 words each, max 8. Include proper nouns and important common nouns. Skip meta-words (source, answer, details, accuracy, response, question, topic, context, information, correction, update, memory) and verbs (said, told, mentioned, discussed, talked, asked). Focus on WHO and WHAT, not how it was communicated.",
                        },
                    },
                    "required": ["key_takeaway", "noun_tags"],
                },
            ),
            types.Tool(
                name="score_memories",
                description="""Score individual cached memories from your previous context.

Your job here is to:
1. Score each cached memory individually — was it helpful, misleading, or unused?
2. Summarize the previous exchange in ~300 chars
3. Rate the exchange outcome (worked/failed/partial/unknown)

FINDING MEMORY IDs:
Memory IDs correspond to [id:...] tags in the KNOWN CONTEXT block from the previous turn.
Look at KNOWN CONTEXT to see what each memory contained, then score based on whether it helped your response.

PER-MEMORY SCORING:
You MUST score each memory ID:
• worked = this memory was helpful
• partial = somewhat helpful
• unknown = didn't use this memory
• failed = this memory was MISLEADING (gave bad advice that led you astray)

You MAY also score any other memory visible in KNOWN CONTEXT or earlier conversation.

⚠️ CRITICAL - "failed" means MISLEADING, not just unused.
If you didn't use a memory, mark it "unknown" not "failed".

ACTIVE MEMORY MANAGEMENT:
• "failed" is not punishment — it's pruning. A memory scored "failed"
  3 times will be demoted or deleted. This is GOOD. Use it.
• If a memory was MISLEADING in your response (you gave wrong info
  because of it), score it "failed" even if the exchange went okay
• If you notice a pattern memory surfacing repeatedly with "unknown"
  scores, consider: is this memory actually useful? If not, scoring
  "failed" once helps the system stop wasting context on it
• You are the gardener. Pull the weeds.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_scores": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "string",
                                "enum": ["worked", "failed", "partial", "unknown"],
                            },
                            "description": "Score for each memory: doc_id -> outcome. MUST include all cached memories. MAY include extras from context.",
                        },
                        "exchange_summary": {
                            "type": "string",
                            "description": "~300 char summary of the previous exchange",
                        },
                        "exchange_outcome": {
                            "type": "string",
                            "enum": ["worked", "failed", "partial", "unknown"],
                            "description": "Was the previous response effective?",
                        },
                        "noun_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Key TOPIC nouns from the exchange. Use actual names not pronouns (skip he/she/they/user/assistant). Lowercase, 1-3 words each, max 8. Include proper nouns and important common nouns. Skip meta-words (source, answer, details, accuracy, response, question, topic, context, information, correction, update, memory) and verbs (said, told, mentioned, discussed, talked, asked). Focus on WHO and WHAT, not how it was communicated.",
                        },
                        "facts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Atomic facts from this exchange. One fact per string, max 150 chars. Include WHO/WHAT, specifics (dates, names, preferences, decisions). Skip vague observations.",
                        },
                    },
                    "required": ["memory_scores"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        """Handle tool calls"""
        # Log all tool calls for analytics (non-blocking)
        session_id = detect_mcp_client()
        client_name = get_client_display_name(session_id)
        logger.info(f"[MCP] Tool called: {name} from {client_name}")

        try:
            if name == "search_memory":
                query = arguments.get("query")
                # Fix: "all" is not a valid collection name - pass None to trigger routing
                collections = arguments.get("collections", None)
                if collections == ["all"]:
                    collections = None
                # Handle both string and int (Claude Desktop sometimes sends "5" instead of 5)
                limit = int(arguments.get("limit", 5)) if arguments.get("limit") else 5
                # Extract metadata filters
                metadata = arguments.get("metadata", None)
                # v0.3.1: type filter for fact/summary distinction
                type_filter = arguments.get("type")
                if type_filter:
                    metadata = dict(metadata) if metadata else {}
                    if type_filter == "fact":
                        metadata["memory_type"] = "fact"
                    elif type_filter == "summary":
                        metadata["memory_type"] = {"$ne": "fact"}

                # v0.2.9: sort_by parameter with auto-detection
                sort_by = arguments.get("sort_by", None)

                # Auto-detect temporal queries if sort_by not specified
                if sort_by is None:
                    temporal_keywords = [
                        "last",
                        "recent",
                        "yesterday",
                        "today",
                        "earlier",
                        "previous",
                        "before",
                        "when did",
                        "how long ago",
                        "last time",
                        "previously",
                        "lately",
                        "just now",
                    ]
                    query_lower = query.lower()
                    if any(kw in query_lower for kw in temporal_keywords):
                        sort_by = "recency"
                        logger.info(
                            f"[MCP] search_memory: Auto-detected temporal query, using recency sort"
                        )

                # Quick check: if memory system isn't initialized, return early (avoids slow embedding model loading)
                if not memory.initialized:
                    return types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text=f"No results found for '{query}' in all collections.\n\nNote: Memory system is empty. Upload documents or store memories first to enable search.",
                            )
                        ]
                    )

                # Log routing decision (will be None if KG should route)
                if collections is None:
                    logger.info(
                        f"[MCP] search_memory: KG will route query '{query[:50]}'"
                    )
                else:
                    logger.info(
                        f"[MCP] search_memory: LLM specified collections: {collections}"
                    )

                if metadata:
                    logger.info(
                        f"[MCP] search_memory: Using metadata filters: {metadata}"
                    )

                # Wrap search with timeout to prevent MCP hanging (ChromaDB can be slow)
                try:
                    results = await asyncio.wait_for(
                        memory.search(
                            query=query,
                            collections=collections,
                            limit=limit,
                            metadata_filters=metadata,
                        ),
                        timeout=25.0,  # 25 seconds - less than MCP's ~240s timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"[MCP] search_memory timed out after 25s for query: {query}"
                    )
                    return types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text",
                                text="⚠️ Search timed out (25s). Embedding model is loading on first use (takes ~1 minute). Please wait and try again.",
                            )
                        ],
                        isError=True,
                    )

                # v0.2.9: Apply sort_by if specified
                if results and sort_by:
                    if sort_by == "recency":
                        # Sort by timestamp (newest first)
                        def get_timestamp(r):
                            meta = r.get("metadata", {})
                            ts = meta.get("timestamp") or meta.get("created_at") or ""
                            return ts if ts else ""

                        results = sorted(results, key=get_timestamp, reverse=True)
                        logger.info(
                            f"[MCP] search_memory: Sorted {len(results)} results by recency"
                        )
                    elif sort_by == "score":
                        # Sort by outcome score (highest first)
                        def get_score(r):
                            meta = r.get("metadata", {})
                            return float(meta.get("score", 0.5))

                        results = sorted(results, key=get_score, reverse=True)
                        logger.info(
                            f"[MCP] search_memory: Sorted {len(results)} results by score"
                        )
                    # sort_by == "relevance" is default (no re-sorting needed, vector similarity order)

                if not results:
                    collection_str = (
                        ", ".join(collections)
                        if collections != ["all"]
                        else "all collections"
                    )
                    text = f"No results found for '{query}' in {collection_str}.\n\nNote: Make sure you have uploaded documents or stored memories first."
                else:
                    text = f"Found {len(results)} result(s) for '{query}':\n\n"
                    for i, r in enumerate(results[:5], 1):
                        # Try both 'content' and 'text' fields (different adapters use different names)
                        content = r.get("content") or r.get("text", "")
                        # v0.2.8: Return full content (was truncated to 300 chars)
                        content_preview = content if content else "[No content]"
                        # Get collection from metadata or root level
                        metadata = r.get("metadata", {})
                        collection = r.get("collection") or metadata.get(
                            "collection", "unknown"
                        )

                        # Extract metadata for LLM context (v0.2.3 enhancement)
                        score = metadata.get("score")
                        uses = metadata.get("uses", 0)
                        timestamp = metadata.get("timestamp")
                        last_outcome = metadata.get("last_outcome")
                        doc_id = r.get("doc_id") or r.get("id", "")

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
                                    ts = dt.fromisoformat(
                                        timestamp.replace("Z", "+00:00")
                                    )
                                else:
                                    ts = timestamp
                                age_days = (
                                    (
                                        datetime.now(ts.tzinfo)
                                        if ts.tzinfo
                                        else datetime.now() - ts
                                    ).days
                                    if hasattr(ts, "days")
                                    else (datetime.now() - ts).days
                                )
                                if age_days == 0:
                                    meta_parts.append("age:today")
                                elif age_days == 1:
                                    meta_parts.append("age:1d")
                                elif age_days < 7:
                                    meta_parts.append(f"age:{age_days}d")
                                elif age_days < 30:
                                    meta_parts.append(f"age:{age_days // 7}w")
                                else:
                                    meta_parts.append(f"age:{age_days // 30}mo")
                            except:
                                pass  # Skip age if parsing fails

                        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
                        id_hint = f" [id:{doc_id}]" if doc_id else ""
                        text += f"{i}. [{collection}]{meta_line}{id_hint} {content_preview}\n\n"

                # Apply cold-start injection if this is first tool call
                text = await _inject_cold_start_if_needed(session_id, text, memory)

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=text)]
                )

            elif name == "add_to_memory_bank":
                content = arguments.get("content")
                tags = arguments.get("tags", [])
                noun_tags = arguments.get(
                    "noun_tags"
                )  # v0.3.1: content nouns for TagCascade
                importance = arguments.get("importance", 0.7)
                confidence = arguments.get("confidence", 0.7)
                session_id = detect_mcp_client()

                # v0.3.1: Safety cap matching roampal-core
                MAX_MEMORY_CHARS = 2000
                if content and len(content) > MAX_MEMORY_CHARS:
                    content = content[:MAX_MEMORY_CHARS]
                    logger.warning(
                        f"[MCP] Memory content truncated from {len(arguments.get('content', ''))} to {MAX_MEMORY_CHARS} chars (safety cap)"
                    )

                doc_id = await memory.store_memory_bank(
                    text=content,
                    tags=tags,
                    noun_tags=noun_tags,
                    importance=importance,
                    confidence=confidence,
                )

                text = f"Added to memory bank (ID: {doc_id})"
                text = await _inject_cold_start_if_needed(session_id, text, memory)

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=text)]
                )

            elif name == "update_memory":
                old_content = arguments.get("old_content", "")
                new_content = arguments.get("new_content", "")
                session_id = detect_mcp_client()

                # v0.3.1: Safety cap matching roampal-core
                MAX_MEMORY_CHARS = 2000
                if new_content and len(new_content) > MAX_MEMORY_CHARS:
                    new_content = new_content[:MAX_MEMORY_CHARS]
                    logger.warning(
                        f"[MCP] Updated memory content truncated from {len(arguments.get('new_content', ''))} to {MAX_MEMORY_CHARS} chars (safety cap)"
                    )

                # Find the old memory by semantic search
                results = await memory.search_memory_bank(
                    query=old_content, limit=1, include_archived=False
                )

                if results:
                    doc_id = results[0].get("id")
                    await memory.update_memory_bank(
                        doc_id=doc_id, new_text=new_content, reason="mcp_update"
                    )
                    logger.info(f"[MCP] Updated memory: {doc_id}")

                    text = f"Updated memory (ID: {doc_id})"
                    text = await _inject_cold_start_if_needed(session_id, text, memory)

                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=text)]
                    )
                else:
                    return types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text", text="Memory not found for update"
                            )
                        ],
                        isError=True,
                    )

            elif name == "archive_memory":
                content = arguments.get("content", "")
                session_id = detect_mcp_client()

                # Find memory by semantic search
                results = await memory.search_memory_bank(
                    query=content, limit=1, include_archived=False
                )

                if results:
                    doc_id = results[0].get("id")
                    await memory.archive_memory_bank(
                        doc_id=doc_id, reason="mcp_archive"
                    )
                    logger.info(f"[MCP] Archived memory: {doc_id}")

                    text = f"Archived memory (ID: {doc_id})"
                    text = await _inject_cold_start_if_needed(session_id, text, memory)

                    return types.CallToolResult(
                        content=[types.TextContent(type="text", text=text)]
                    )
                else:
                    return types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text", text="Memory not found for archiving"
                            )
                        ],
                        isError=True,
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
                        session_data = json.loads(
                            session_file.read_text(encoding="utf-8")
                        )
                        # Extract last 5 turns for context
                        turns = session_data.get("turns", [])[-5:]
                        for turn in turns:
                            # Convert session turn format to conversation format
                            user_msg = turn.get("user_message", "")
                            ai_msg = turn.get("ai_response", "")
                            if user_msg:
                                recent_conv.append(
                                    {"role": "user", "content": user_msg}
                                )
                            if ai_msg:
                                recent_conv.append(
                                    {"role": "assistant", "content": ai_msg}
                                )
                    except Exception as e:
                        logger.error(f"[MCP] Failed to load session file: {e}")

                # v0.3.0: Use unified context injection (ported from roampal-core)
                try:
                    context = await memory.get_context_for_injection(
                        query=query,
                        conversation_id=session_id,
                        recent_conversation=recent_conv,
                    )

                    # Format response
                    user_facts = context.get("user_facts", [])
                    memories = context.get("memories", [])

                    text = f"Known Context for '{query}':\n\n"

                    if user_facts:
                        text += "**Memory Bank (always injected):**\n"
                        for fact in user_facts:
                            text += f"• {fact.get('content', '')}\n"
                        text += "\n"

                    if memories:
                        text += "**Relevant Memories (top 3 by Wilson score):**\n"
                        for mem in memories:
                            coll = mem.get("collection", "unknown")
                            content = mem.get("content") or mem.get("text", "")
                            wilson = mem.get("wilson_score", 0)
                            doc_id = mem.get("id", "")
                            id_hint = f" [id:{doc_id}]" if doc_id else ""
                            if wilson >= 0.7:
                                text += f"• [{coll}]{id_hint} ({int(wilson * 100)}% proven) {content}\n"
                            else:
                                text += f"• [{coll}]{id_hint} {content}\n"
                        text += "\n"

                    if not user_facts and not memories:
                        text += "No relevant context found. This may be a new topic or first interaction.\n"

                    text += (
                        f"\n_Cached {len(cached_doc_ids)} doc_ids for outcome scoring._"
                    )

                except Exception as e:
                    logger.error(
                        f"[MCP] get_context_insights error: {e}", exc_info=True
                    )
                    text = f"Error analyzing context: {str(e)}"

                # Inject cold-start context if first tool call
                text = await _inject_cold_start_if_needed(session_id, text, memory)

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=text)]
                )

            elif name == "record_response":
                # v0.3.1: Simplified — just stores key takeaway. Scoring moved to score_memories tool.
                key_takeaway = arguments.get("key_takeaway")
                noun_tags = arguments.get("noun_tags", [])
                session_id = detect_mcp_client()

                # v0.3.1: Safety cap matching roampal-core
                MAX_MEMORY_CHARS = 2000
                if key_takeaway and len(key_takeaway) > MAX_MEMORY_CHARS:
                    key_takeaway = key_takeaway[:MAX_MEMORY_CHARS]
                    logger.warning(
                        f"[MCP] Key takeaway truncated from {len(arguments.get('key_takeaway', ''))} to {MAX_MEMORY_CHARS} chars (safety cap)"
                    )

                if not key_takeaway:
                    return types.CallToolResult(
                        content=[
                            types.TextContent(
                                type="text", text="Error: 'key_takeaway' is required"
                            )
                        ],
                        isError=True,
                    )

                logger.info(
                    f"[MCP] record_response: session={session_id}, takeaway={key_takeaway[:100]}..."
                )

                # Store key takeaway as working memory with initial score 0.7
                metadata = {
                    "role": "learning",
                    "source": session_id,
                    "score": 0.7,
                    "created_at": datetime.now().isoformat(),
                }
                # Extract noun_tags for TagCascade (LLM via TagService, no regex)
                if noun_tags:
                    metadata["noun_tags"] = json.dumps(noun_tags)
                elif memory and hasattr(memory, "_tag_service") and memory._tag_service:
                    auto_tags = await memory._tag_service.extract_tags_async(key_takeaway)
                    if auto_tags:
                        metadata["noun_tags"] = json.dumps(auto_tags)

                doc_id = await memory.store(
                    text=key_takeaway, collection="working", metadata=metadata
                )

                logger.info(
                    f"[MCP] Stored takeaway (score=0.7): {key_takeaway[:50]}..."
                )

                response_text = f"Recorded: {key_takeaway}"
                response_text = await _inject_cold_start_if_needed(
                    session_id, response_text, memory
                )

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=response_text)]
                )

            elif name == "score_memories":
                # v0.3.1: Per-memory scoring — matches roampal-core score_memories tool
                memory_scores = arguments.get("memory_scores", {})
                exchange_summary = arguments.get("exchange_summary")
                exchange_outcome = arguments.get("exchange_outcome", "unknown")
                noun_tags = arguments.get("noun_tags", [])
                facts = arguments.get("facts", [])
                session_id = detect_mcp_client()

                logger.info(
                    f"[MCP] score_memories: session={session_id}, scores={len(memory_scores)}, outcome={exchange_outcome}, facts={len(facts)}"
                )

                # 1. Score each memory individually
                scored_count = 0
                for mem_doc_id, mem_outcome in memory_scores.items():
                    if mem_outcome in ["worked", "failed", "partial", "unknown"]:
                        try:
                            await memory.record_outcome(
                                doc_id=mem_doc_id, outcome=mem_outcome
                            )
                            scored_count += 1
                        except Exception as e:
                            logger.warning(
                                f"[MCP] Failed to score memory {mem_doc_id}: {e}"
                            )

                # 2. Store exchange summary if provided
                summary_doc_id = None
                if exchange_summary and len(exchange_summary.strip()) >= 10:
                    summary_metadata = {
                        "role": "learning",
                        "source": session_id,
                        "score": 0.5,
                        "created_at": datetime.now().isoformat(),
                    }
                    if noun_tags:
                        summary_metadata["noun_tags"] = json.dumps(noun_tags)
                    elif memory and hasattr(memory, "_tag_service") and memory._tag_service:
                        auto_tags = await memory._tag_service.extract_tags_async(exchange_summary)
                        if auto_tags:
                            summary_metadata["noun_tags"] = json.dumps(auto_tags)

                    summary_doc_id = await memory.store(
                        text=exchange_summary,
                        collection="working",
                        metadata=summary_metadata,
                    )

                # 3. Store atomic facts (two-lane retrieval)
                stored_facts = 0
                if facts:
                    for fact_text in facts:
                        if not fact_text or len(fact_text.strip()) < 10:
                            continue
                        try:
                            fact_noun_tags = []
                            if memory and hasattr(memory, "_tag_service") and memory._tag_service:
                                fact_noun_tags = await memory._tag_service.extract_tags_async(fact_text)
                            fact_metadata = {
                                "memory_type": "fact",
                                "role": "fact",
                                "source": session_id,
                                "score": 0.5,
                                "created_at": datetime.now().isoformat(),
                            }
                            if fact_noun_tags:
                                fact_metadata["noun_tags"] = json.dumps(fact_noun_tags)
                            await memory.store(
                                text=fact_text,
                                collection="working",
                                metadata=fact_metadata,
                            )
                            stored_facts += 1
                        except Exception as e:
                            logger.warning(f"Failed to store fact: {e}")

                # Build response
                parts = [f"Scored ({scored_count} memories updated)"]
                if exchange_summary:
                    parts.append(f"Summary stored ({len(exchange_summary)} chars)")
                if stored_facts:
                    parts.append(f"{stored_facts} facts stored")

                logger.info(
                    f"[MCP] score_memories complete: {scored_count} scored, summary={'yes' if exchange_summary else 'no'}, facts={stored_facts}"
                )

                response_text = ". ".join(parts)
                response_text = await _inject_cold_start_if_needed(
                    session_id, response_text, memory
                )

                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=response_text)]
                )

            else:
                return types.CallToolResult(
                    content=[
                        types.TextContent(type="text", text=f"Unknown tool: {name}")
                    ],
                    isError=True,
                )

        except Exception as e:
            logger.error(f"[MCP] Tool call error for {name}: {e}", exc_info=True)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True,
            )

    # Run MCP server via stdio
    logger.info(
        "[MCP] Server initialized with 7 tools: search_memory, add_to_memory_bank, update_memory, archive_memory, get_context_insights, record_response, score_memories"
    )
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


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

        api_port = int(os.getenv("ROAMPAL_API_PORT", "8001"))
        logger.info(f"Starting FastAPI server on port {api_port}")
        uvicorn.run(app, host="127.0.0.1", port=api_port)
