import logging
from config.settings import settings
from pathlib import Path
from typing import cast, Optional, Dict, Any, List
import os
import asyncio
import re

from config.settings import (
    Settings, LLMSettings, FragmentFileMemorySettings, WebSearchSettings,
    OGPromptSettings, ScoringSettings, BOOKS
)

from core.interfaces.llm_client_interface import LLMClientInterface
from core.interfaces.memory_adapter_interface import MemoryAdapterInterface
from core.interfaces.web_scraper_interface import WebScraperInterface
from core.interfaces.prompt_engine_interface import PromptEngineInterface
from core.interfaces.intent_router_interface import IntentRouterInterface
from core.interfaces.scoring_engine_interface import ScoringEngineInterface
from core.interfaces.book_processor_interface import BookProcessorInterface
# Removed: soul_layer_manager_interface - using enhanced memory collections
from core.interfaces.ingestion_manager_interface import IngestionManagerInterface

from modules.llm.ollama_client import OllamaClient
from modules.memory.chromadb_adapter import ChromaDBAdapter
from modules.memory.file_memory_adapter import FileMemoryAdapter
# Multi-tier memory adapter removed - using simple ChromaDB adapter
from modules.web_search.playwright_web_scraper import PlaywrightWebScraper
from modules.prompt.prompt_engine import PromptEngine
from modules.intent.og_intent_router import OGIntentRouter
# REMOVED: Dead code - ScoringEngine module doesn't exist
# from modules.scoring.scoring_engine import ScoringEngine
from modules.memory.smart_book_processor import SmartBookProcessor
# Removed: soul_layer_manager - using enhanced memory collections
from modules.ingestion.ingestion_manager import IngestionManager
# REMOVED: from modules.scoring.self_debate_engine import SelfDebateEngine  # Overengineered - user feedback validates fragments naturally
# Removed: soul_injector - using outcome-based scoring
from modules.embedding.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# REMOVED: CANONICAL_VECTOR_STORE_PATH - now using fragment-specific paths

def load_dynamic_books(folder='data/og_books'):
    BOOKS.clear()
    og_books_dir = os.path.join(os.path.dirname(__file__), '..', folder)
    for filename in os.listdir(og_books_dir):
        if filename.endswith('.txt'):
            title = filename.replace('.txt', '').replace('_', ' ').title()
            author = 'Unknown'
            keywords = [title.lower()] + (['coding'] if any(kw in title.lower() for kw in ['sicp', 'programming', 'mit', 'cleanarchitecture']) else ['philosophy'])
            BOOKS.append({
                'title': title,
                'author': author,
                'keywords': keywords
            })
            logger.info(f"Added dynamic book: {title}")

async def initialize_llm_client(config: LLMSettings) -> LLMClientInterface:
    logger.debug(f"Initializing LLMClient with provider: {config.provider}, model: {config.ollama_model}")
    if config.provider == "ollama":
        client = cast(LLMClientInterface, OllamaClient())
        await client.initialize(config=config.model_dump())
        return client
    else:
        msg = f"Unsupported LLM provider: '{config.provider}'"
        logger.critical(msg)
        raise ValueError(msg)

async def initialize_fragment_memory_adapter(
    fragment_id: str,
    global_settings: Settings,
    adapter_type: str = "chromadb"
) -> MemoryAdapterInterface:
    # Validate fragment_id to prevent path traversal
    if not fragment_id or not re.match(r'^[a-zA-Z0-9_-]+$', fragment_id):
        raise ValueError(f"Invalid fragment_id: '{fragment_id}'. Only alphanumeric characters, hyphens and underscores allowed.")
    
    logger.info(f"Attempting to initialize MemoryAdapter for fragment: '{fragment_id}' ({adapter_type})")
    fragment_memory_config_obj = global_settings.og_memory
    # FIXED: Use fragment-specific base path instead of hardcoded "og_data"  
    base_path = f"data/shards/{fragment_id}"

    if adapter_type == "chromadb":
        # FIXED: Use fragment-specific vector store path instead of hardcoded "og"
        # Ensure path is safe and normalized
        vector_store_path = os.path.abspath(settings.paths.get_vector_db_dir(fragment_id))
        # Validate that the path is within expected directory
        expected_base = os.path.abspath("data/vector_stores")
        if not vector_store_path.startswith(expected_base):
            raise ValueError(f"Invalid vector store path: {vector_store_path}")
        logger.info(f"VECTOR_PATH_FIX: Using vector store path for fragment '{fragment_id}': {vector_store_path}")
        adapter = cast(MemoryAdapterInterface, ChromaDBAdapter(persistence_directory=vector_store_path))
        collection_name = getattr(fragment_memory_config_obj, "collection_name", None) or f"roampal_{fragment_id}_soul_fragments"
        await adapter.initialize(collection_name=collection_name)
        logger.info(f"Initialized ChromaDBAdapter for fragment '{fragment_id}' at path '{vector_store_path}' in collection '{collection_name}'.")
        return adapter

    elif adapter_type == "file":
        config_dict = fragment_memory_config_obj.model_dump()
        config_dict['fragment_id'] = fragment_id
        config_dict['base_data_path'] = base_path
        adapter = FileMemoryAdapter()
        await adapter.initialize(config=config_dict)
        logger.info(f"Initialized FileMemoryAdapter for fragment '{fragment_id}' at '{getattr(adapter, 'data_path', None)}'.")
        return adapter

    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

# Multi-tier memory adapter function removed - use initialize_fragment_memory_adapter instead

async def initialize_web_scraper(config: WebSearchSettings) -> WebScraperInterface:
    logger.debug("Initializing WebScraper")
    scraper = PlaywrightWebScraper()
    await scraper.initialize(service_client_config=config.model_dump())
    return scraper

def _resolve_prompt_dirs_from_config(config: Any, default_dir: Optional[str] = None) -> List[str]:
    prompt_dirs = []
    if hasattr(config, "template_directories") and config.template_directories:
        for d in config.template_directories:
            d_path = Path(d)
            if not d_path.is_absolute():
                base_dir = Path(__file__).parent.parent.parent.resolve()
                d_path = (base_dir / d_path).resolve()
            prompt_dirs.append(str(d_path))
    elif hasattr(config, "template_directory") and config.template_directory:
        d_path = Path(config.template_directory)
        if not d_path.is_absolute():
            base_dir = Path(__file__).parent.parent.parent.resolve()
            d_path = (base_dir / d_path).resolve()
        prompt_dirs.append(str(d_path))
    elif default_dir:
        prompt_dirs.append(str(default_dir))
    return prompt_dirs

async def initialize_og_prompt_engine(config: OGPromptSettings) -> PromptEngineInterface:
    logger.info(f"Initializing PromptEngine for Roampal, default prompt file: {config.default_system_prompt_name}.txt")
    engine = PromptEngine()

    # Use Roampal template directory
    roampal_template_dir = settings.paths.get_soul_layers_dir()
    prompt_dirs = [str(roampal_template_dir), str(settings.paths.prompt_template_dir)]

    logger.info(f"Using prompt template directories: {prompt_dirs}")
    await engine.initialize(
        template_directories=prompt_dirs,
        config={"default_system_prompt_name": config.default_system_prompt_name}
    )
    return engine

async def initialize_og_intent_router() -> IntentRouterInterface:
    logger.info("Initializing OG IntentRouter (OGIntentRouter)")
    router = OGIntentRouter()
    await router.initialize(config=None)
    return router

async def initialize_scoring_engine(config: ScoringSettings) -> ScoringEngineInterface:
    logger.debug("Initializing ScoringEngine")
    engine = ScoringEngine()
    config_dict = config.model_dump() if hasattr(config, "model_dump") else dict(config)
    config_dict["data_path"] = "data/loopsmith/neuron_scores.jsonl"
    await engine.initialize(config=config_dict)
    return engine

async def initialize_book_processor_for_fragment(
    fragment_id: str,
    global_settings: Settings,
    llm_client: LLMClientInterface,
    embedding_service: EmbeddingService,
    multi_tier_adapter: Optional[MultiTierMemoryAdapter] = None
) -> BookProcessorInterface:
    # Use MultiTierMemoryAdapter if provided, otherwise fall back to regular adapter
    if multi_tier_adapter:
        memory_adapter = multi_tier_adapter
        logger.info(f"Initializing BookProcessor for fragment '{fragment_id}' with MultiTierMemoryAdapter for GLOBAL book seeding")
    else:
        memory_adapter = await initialize_fragment_memory_adapter(fragment_id, global_settings, adapter_type="chromadb")
        logger.warning(f"Initializing BookProcessor for fragment '{fragment_id}' with regular ChromaDB adapter (books won't be in global memory!)")
    
    processor = SmartBookProcessor(
        data_dir=f"data/shards/{fragment_id}/memory",
        chromadb_adapter=vector_db,
        embedding_service=embedding_service,
        llm_client=llm_client
    )
    logger.info(f"SmartBookProcessor instance for fragment '{fragment_id}' created.")
    return cast(BookProcessorInterface, processor)

# Soul layer manager removed - using enhanced memory collections instead

async def initialize_ingestion_manager(
    settings_instance: Settings,
    llm_client: LLMClientInterface
) -> IngestionManagerInterface:
    logger.info("Initializing IngestionManager...")
    manager = IngestionManager(settings=settings_instance, llm_client=llm_client)
    await manager.initialize()
    logger.info("IngestionManager instance created and initialized.")
    return cast(IngestionManagerInterface, manager)

# REMOVED: initialize_self_debate_engine - overengineered feature removed
# User feedback naturally validates fragments through scoring adjustments
