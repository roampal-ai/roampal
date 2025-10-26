# backend/modules/ingestion/models.py
import datetime
import uuid
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

class IngestionJobForceFlags(BaseModel):
    force_reprocess_chunks: bool = False
    force_resummarize_chunks: bool = False
    force_regenerate_full_summary: bool = False
    force_reextract_content: bool = False
    force_update_foundational_kb: bool = False

class IngestionJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_uri: str  
    source_type: Literal["file", "url", "s3_object"] = "file"
    
    title: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    is_foundational: bool = False 
    fragment_associated: Optional[str] = None 
    license_info: Optional[str] = None
    
    target_memory_config_key: str 
                                  
    force_flags: IngestionJobForceFlags = Field(default_factory=IngestionJobForceFlags)
    
    status: Literal[
        "pending", "preprocessing", "downloading", 
        "processing_registry", "chunking", "summarizing_chunks", 
        "generating_full_summary", "extracting_models", "extracting_quotes", 
        "completed", "failed", "partial_success"
    ] = "pending"
    error_message: Optional[str] = None
    processing_log: List[str] = Field(default_factory=list)
    
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))

class IngestionJobCreateRequest(BaseModel): 
    source_uri: str
    source_type: Literal["file", "url", "s3_object"] = "file"
    title: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    is_foundational: Optional[bool] = False
    fragment_associated: Optional[str] = None
    license_info: Optional[str] = None
    target_memory_config_key: str 
    force_flags: Optional[IngestionJobForceFlags] = None

    class Config:
        extra = 'forbid' # To catch unexpected fields in the request
