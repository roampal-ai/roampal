"""
Input validation models for robustness
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import re


class ChatRequest(BaseModel):
    """Validated chat request model"""
    message: str = Field(..., min_length=1, max_length=100000)
    conversation_id: Optional[str] = Field(None, pattern=r'^conv_[a-z0-9_]{1,50}$')
    session_id: Optional[str] = Field(None, max_length=100)
    use_memory: bool = Field(True)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        # Strip whitespace
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        # Check for control characters
        if re.search(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', v):
            raise ValueError("Message contains invalid control characters")
        return v


class BookUploadRequest(BaseModel):
    """Validated book upload request"""
    title: Optional[str] = Field(None, max_length=500)
    author: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    check_duplicate: bool = Field(True)

    @field_validator('title', 'author')
    @classmethod
    def sanitize_text(cls, v: Optional[str]) -> Optional[str]:
        if v:
            # Remove potentially dangerous characters
            v = re.sub(r'[<>:"\\|?*]', '', v)
            v = v.strip()
        return v


class MemorySearchRequest(BaseModel):
    """Validated memory search request"""
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(10, ge=1, le=100)
    collections: Optional[List[str]] = Field(None)
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator('collections')
    @classmethod
    def validate_collections(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        valid_collections = {'books', 'working', 'history', 'patterns', 'memory_bank'}
        if v:
            invalid = set(v) - valid_collections
            if invalid:
                raise ValueError(f"Invalid collections: {invalid}")
        return v


class FeedbackRequest(BaseModel):
    """Validated feedback request"""
    doc_id: str = Field(..., pattern=r'^[a-zA-Z]+_[a-f0-9]{8,}$')
    outcome: str = Field(..., pattern=r'^(worked|failed|partial)$')
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = Field(None)
    failure_reason: Optional[str] = Field(None, max_length=500)

    @field_validator('failure_reason')
    @classmethod
    def validate_failure_reason(cls, v: Optional[str], values: Dict) -> Optional[str]:
        if values.get('outcome') == 'failed' and not v:
            raise ValueError("Failure reason required when outcome is 'failed'")
        return v


class ConversationSwitchRequest(BaseModel):
    """Validated conversation switch request"""
    old_conversation_id: str = Field(..., pattern=r'^conv_[a-z0-9_]{1,50}$')
    new_conversation_id: str = Field(..., pattern=r'^conv_[a-z0-9_]{1,50}$')

    @field_validator('new_conversation_id')
    @classmethod
    def validate_different_ids(cls, v: str, values: Dict) -> str:
        if v == values.get('old_conversation_id'):
            raise ValueError("New conversation ID must be different from old")
        return v


class CommandRequest(BaseModel):
    """Validated command request"""
    command: str = Field(..., pattern=r'^/[a-z]+(\s.*)?$')
    args: Optional[str] = Field(None, max_length=1000)

    @field_validator('command')
    @classmethod
    def validate_command(cls, v: str) -> str:
        allowed_commands = {
            '/help', '/clear', '/memory', '/stats', '/run',
            '/test', '/fix', '/save', '/load', '/search'
        }
        cmd = v.split()[0]
        if cmd not in allowed_commands:
            raise ValueError(f"Unknown command: {cmd}")
        return v