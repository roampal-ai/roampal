"""
API endpoints for model context window management.
Allows users to view and customize context sizes per model.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

# Import centralized context configuration
from config.model_contexts import (
    get_context_size,
    get_model_info,
    get_all_model_contexts,
    save_user_override,
    delete_user_override
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model-contexts"])


class ContextUpdateRequest(BaseModel):
    """Request model for updating model context."""
    context_size: int


@router.get("/api/model/contexts")
async def get_all_contexts() -> Dict[str, Any]:
    """
    Get context window information for all configured models.

    Returns default and maximum values for each model type.
    """
    try:
        contexts = get_all_model_contexts()
        return {
            "status": "success",
            "contexts": contexts
        }
    except Exception as e:
        logger.error(f"Failed to get model contexts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/model/context/{model_name}")
async def get_model_context(model_name: str) -> Dict[str, Any]:
    """
    Get current context window setting for a specific model.

    Args:
        model_name: Full model name (e.g., "qwen2.5:14b")

    Returns:
        Current context size, default, maximum, and whether user has overridden it.
    """
    try:
        info = get_model_info(model_name)
        return {
            "status": "success",
            "model": model_name,
            **info
        }
    except Exception as e:
        logger.error(f"Failed to get context for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/model/context/{model_name}")
async def set_model_context(
    model_name: str,
    request: ContextUpdateRequest
) -> Dict[str, Any]:
    """
    Set a custom context window size for a specific model.

    Args:
        model_name: Full model name (e.g., "qwen2.5:14b")
        request: New context size

    Returns:
        Success status and new context info.
    """
    try:
        # Validate context size (must be positive and reasonable)
        if request.context_size < 512:
            raise ValueError("Context size must be at least 512 tokens")
        if request.context_size > 200000:
            raise ValueError("Context size cannot exceed 200,000 tokens")

        # Save the override
        success = save_user_override(model_name, request.context_size)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save context override"
            )

        # Return updated info
        info = get_model_info(model_name)
        return {
            "status": "success",
            "message": f"Context window for {model_name} set to {request.context_size}",
            **info
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to set context for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/api/model/context/{model_name}")
async def reset_model_context(model_name: str) -> Dict[str, Any]:
    """
    Reset a model's context window to its default value.

    Args:
        model_name: Full model name (e.g., "qwen2.5:14b")

    Returns:
        Success status and default context info.
    """
    try:
        # Properly delete the override entry from settings file
        success = delete_user_override(model_name)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to reset context override"
            )

        info = get_model_info(model_name)
        return {
            "status": "success",
            "message": f"Context window for {model_name} reset to default",
            **info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset context for {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))