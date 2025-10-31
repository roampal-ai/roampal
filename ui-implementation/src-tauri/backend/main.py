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
from modules.memory.session_cleanup import SessionCleanupManager
from modules.llm.ollama_client import OllamaClient
from services.unified_image_service import UnifiedImageService
from config.feature_flag_validator import FeatureFlagValidator
from config.settings import DATA_PATH

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

# Configure logging for production with rotation
from logging.handlers import RotatingFileHandler
log_level = os.getenv('ROAMPAL_LOG_LEVEL', 'INFO')

# Create rotating file handler (10MB max, keep 3 backups)
file_handler = RotatingFileHandler(
    'roampal.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=3,          # Keep 3 old files (roampal.log.1, .2, .3)
    encoding='utf-8'
)

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        file_handler
    ]
)
logger = logging.getLogger(__name__)


async def memory_promotion_task(memory: UnifiedMemorySystem):
    """Background task to promote valuable working memory to history"""
    while True:
        try:
            # Wait 30 minutes between promotion checks
            await asyncio.sleep(1800)  # 30 minutes in seconds

            logger.info("Running scheduled memory promotion...")

            # Promote valuable working memory
            await memory._promote_valuable_working_memory()

            # Clean old working memory (older than 24 hours)
            await memory.clear_old_working_memory(hours=24)

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
                app.state.session_cleanup = SessionCleanupManager(sessions_dir)
                await app.state.session_cleanup.initialize()
                logger.info(f"✓ Session cleanup manager initialized (max sessions: {app.state.session_cleanup.max_sessions})")

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
            app.state.memory.set_llm_service(app.state.llm_client)
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

@app.get("/health")
async def health():
    import os
    return {
        "status": "healthy",
        "service": "Roampal",
        "safe_mode": os.getenv('ROAMPAL_SAFE_MODE', os.getenv('ROAMPAL_SAFE_MODE', 'not set')),
        "safe_mode_enabled": os.getenv('ROAMPAL_SAFE_MODE', os.getenv('ROAMPAL_SAFE_MODE', 'false')).lower() == 'true'
    }

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
rate_limit_storage = defaultdict(lambda: deque(maxlen=100))
RATE_LIMIT = 100  # requests per minute


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)