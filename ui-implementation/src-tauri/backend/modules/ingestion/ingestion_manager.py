# backend/modules/ingestion/ingestion_manager.py
# DEPRECATED: This module is not used anywhere in the codebase.
# Book processing is now handled by SmartBookProcessor via book_upload_api.py
# Kept for backwards compatibility but should be removed in future cleanup.

import logging
import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

from config.settings import Settings
from core.interfaces.llm_client_interface import LLMClientInterface # For type hinting __init__
from core.interfaces.memory_adapter_interface import MemoryAdapterInterface
from core.interfaces.ingestion_manager_interface import IngestionManagerInterface
from modules.memory.chromadb_adapter import ChromaDBAdapter  # <-- REPLACE FileMemoryAdapter
# from utils.book_processor import BookProcessor  # REMOVED: Old processor no longer exists
from modules.memory.smart_book_processor import SmartBookProcessor
from .models import IngestionJob, IngestionJobCreateRequest, IngestionJobForceFlags

logger = logging.getLogger(__name__)

JOB_QUEUE: List[IngestionJob] = []
PROCESSED_JOBS: Dict[str, IngestionJob] = {}

class IngestionManager(IngestionManagerInterface):
    def __init__(self, settings: Settings, llm_client: LLMClientInterface):
        self.settings = settings
        self.llm_client = llm_client
        self.is_initialized = False # Will be set in initialize
        logger.debug("IngestionManager instance created (uninitialized).")

    async def initialize(self): # Matches interface
        # In a real scenario, might load existing pending jobs, connect to a queue, etc.
        self.is_initialized = True
        logger.info("IngestionManager initialized.")

    async def _get_memory_adapter_for_target(self, target_memory_config_key: str) -> Optional[MemoryAdapterInterface]:
        logger.info(f"Attempting to get memory adapter for target_key: '{target_memory_config_key}'")
        mem_settings_dict: Optional[Dict[str, Any]] = None
        collection_name = None
        # --- PATCH: Always use OG vector store path ---
        from config.settings import settings
        OG_VECTOR_STORE_PATH = settings.paths.get_vector_db_dir("og")

        if target_memory_config_key == "user_memory":
            mem_settings_dict = self.settings.user_memory.model_dump()
            collection_name = mem_settings_dict.get("collection_name", "roampal_user_soul_fragments")
            data_path = OG_VECTOR_STORE_PATH
        elif target_memory_config_key == "og_memory":
            mem_settings_dict = self.settings.og_memory.model_dump()
            collection_name = mem_settings_dict.get("collection_name", "roampal_og_soul_fragments")
            data_path = OG_VECTOR_STORE_PATH
        elif target_memory_config_key in self.settings.fragment_memory_configs:
            fragment_mem_settings = self.settings.fragment_memory_configs[target_memory_config_key]
            mem_settings_dict = fragment_mem_settings.model_dump()
            collection_name = mem_settings_dict.get("collection_name", f"roampal_{target_memory_config_key}_soul_fragments")
            data_path = OG_VECTOR_STORE_PATH  # PATCH: Use OG path for all fragments too

        if mem_settings_dict:
            adapter = ChromaDBAdapter(persistence_directory=data_path)
            await adapter.initialize(collection_name=collection_name, fragment_id=target_memory_config_key)
            logger.info(f"Initialized ChromaDBAdapter for target '{target_memory_config_key}' (collection={collection_name})")
            return adapter
        else:
            logger.error(f"No memory configuration found for target_key: '{target_memory_config_key}' in settings.")
            return None

    async def submit_job_request(self, job_request_data: Dict[str, Any]) -> Dict[str, Any]:
        # Validate with Pydantic model
        try:
            job_create_model = IngestionJobCreateRequest(**job_request_data)
        except Exception as e: # Pydantic ValidationError
            logger.error(f"Invalid job request data: {e}", exc_info=True)
            raise ValueError(f"Invalid job request data: {e}")

        job = IngestionJob(
            source_uri=job_create_model.source_uri,
            source_type=job_create_model.source_type,
            title=job_create_model.title,
            author=job_create_model.author,
            tags=job_create_model.tags or [],
            is_foundational=job_create_model.is_foundational if job_create_model.is_foundational is not None else False,
            fragment_associated=job_create_model.fragment_associated,
            license_info=job_create_model.license_info,
            target_memory_config_key=job_create_model.target_memory_config_key,
            force_flags=job_create_model.force_flags if job_create_model.force_flags is not None else IngestionJobForceFlags()
        )
        JOB_QUEUE.append(job)
        PROCESSED_JOBS[job.job_id] = job
        logger.info(f"Submitted new ingestion job: {job.job_id} for '{job.title}' target: {job.target_memory_config_key}")
        return job.model_dump() # Return as dict

    async def process_single_job(self, job_data: Dict[str, Any]) -> Dict[str, Any]:
        # Re-create IngestionJob from dict for type safety if needed, or assume job_data is already an IngestionJob instance
        job = IngestionJob(**job_data) if isinstance(job_data, dict) else job_data

        logger.info(f"Starting processing for job ID: {job.job_id}, Title: '{job.title}'")
        job.status = "preprocessing"; job.updated_at = datetime.datetime.now(datetime.timezone.utc)
        PROCESSED_JOBS[job.job_id] = job

        target_memory_adapter = await self._get_memory_adapter_for_target(job.target_memory_config_key)
        if not target_memory_adapter:
            job.status = "failed"; job.error_message = "Could not initialize target memory adapter."
            PROCESSED_JOBS[job.job_id] = job; return job.model_dump()

        # NOTE: SmartBookProcessor has different constructor signature
        # This code path is never executed (IngestionManager is unused)
        # processor = SmartBookProcessor(data_dir="data", llm_client=self.llm_client)
        raise NotImplementedError("IngestionManager is deprecated. Use book_upload_api.py instead.")
        local_source_file_path = job.source_uri
        if job.source_type != "file":
            job.status = "failed"; job.error_message = f"Source type '{job.source_type}' not yet supported for download."
            PROCESSED_JOBS[job.job_id] = job; return job.model_dump()
            
        try:
            job.status = "processing_registry"; PROCESSED_JOBS[job.job_id] = job
            book_meta = await processor.add_book_to_registry(
                source_filepath_str=local_source_file_path, title=job.title, author=job.author,
                tags=job.tags, is_foundational=job.is_foundational,
                fragment_associated=job.fragment_associated, license_info=job.license_info
            )
            if not book_meta: raise Exception("Failed to add book to registry.")
            book_id = book_meta["book_id"]

            if book_meta.get("chunk_status") != "chunked" or job.force_flags.force_reprocess_chunks:
                job.status = "chunking"; PROCESSED_JOBS[job.job_id] = job
                await processor.chunk_book_by_chapters_and_size(book_id) # Uses defaults from BookProcessor
            
            current_meta = await processor._get_book_metadata(book_id) # Refresh meta
            if current_meta and (current_meta.get("summary_status") not in ["fully_summarized"] or job.force_flags.force_resummarize_chunks):
                job.status = "summarizing_chunks"; PROCESSED_JOBS[job.job_id] = job
                await processor.process_all_chunks_for_book(book_id, force_resummarize=job.force_flags.force_resummarize_chunks)

            current_meta = await processor._get_book_metadata(book_id)
            if current_meta and (current_meta.get("summary_status") != "full_summary_generated" or job.force_flags.force_regenerate_full_summary):
                if current_meta.get("summary_status") in ["fully_summarized", "partially_summarized", "chunked"] or job.force_flags.force_regenerate_full_summary:
                    job.status = "generating_full_summary"; PROCESSED_JOBS[job.job_id] = job
                    await processor.generate_full_book_summary(book_id, force_overwrite_foundational_kb=job.force_flags.force_update_foundational_kb)
            
            current_meta = await processor._get_book_metadata(book_id)
            if current_meta and (job.force_flags.force_reextract_content or current_meta.get("models_status") != "extracted"):
                job.status = "extracting_models"; PROCESSED_JOBS[job.job_id] = job
                await processor.extract_models_from_book(book_id)
            
            current_meta = await processor._get_book_metadata(book_id)
            if current_meta and (job.force_flags.force_reextract_content or current_meta.get("quotes_status") != "extracted"):
                job.status = "extracting_quotes"; PROCESSED_JOBS[job.job_id] = job
                await processor.extract_quotes_from_book(book_id)

            job.status = "completed"
            logger.info(f"Successfully processed job ID: {job.job_id}, Title: '{job.title}'")
        except Exception as e:
            logger.error(f"Failed to process job ID: {job.job_id}, Title: '{job.title}'. Error: {e}", exc_info=True)
            job.status = "failed"; job.error_message = str(e)

        job.updated_at = datetime.datetime.now(datetime.timezone.utc)
        PROCESSED_JOBS[job.job_id] = job
        return job.model_dump()

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = PROCESSED_JOBS.get(job_id)
        return job.model_dump() if job else None
