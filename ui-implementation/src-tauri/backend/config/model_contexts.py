"""
Centralized model context window configuration.
Single source of truth for context sizes across the application.
"""

import json
import os
import threading
from typing import Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Thread-safe file locking for concurrent write operations
_settings_lock = threading.Lock()

# Model-specific optimal contexts
# default: Safe value that works on most hardware
# max: Theoretical maximum the model supports
MODEL_CONTEXTS = {
    "gpt-oss": {"default": 32768, "max": 128000},
    "llama3.1": {"default": 32768, "max": 131072},
    "llama3.2": {"default": 32768, "max": 131072},
    "llama3.3": {"default": 32768, "max": 131072},
    "llama3": {"default": 32768, "max": 131072},  # Generic llama3
    "qwen3-coder": {"default": 32768, "max": 262144},  # MoE 30B, 256K context
    "qwen3": {"default": 32768, "max": 32768},  # Qwen3 models
    "qwen2.5": {"default": 32768, "max": 32768},
    "qwen2": {"default": 32768, "max": 32768},
    "qwen": {"default": 32768, "max": 32768},
    "mistral": {"default": 16384, "max": 32768},
    "mixtral": {"default": 16384, "max": 32768},
    "phi-4": {"default": 16384, "max": 128000},
    "phi": {"default": 16384, "max": 128000},
    "dolphin": {"default": 16384, "max": 32768},
    "firefunction": {"default": 16384, "max": 32768},
    "command-r": {"default": 32768, "max": 128000},
    "command": {"default": 32768, "max": 128000},
}

# User settings file path - Use AppData in production
from config.settings import DATA_PATH
SETTINGS_FILE = str(Path(DATA_PATH) / "user_model_contexts.json")

def load_user_overrides() -> Dict[str, int]:
    """Load user-specified context overrides from settings file."""
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)
                return data.get("model_overrides", {})
        except Exception as e:
            logger.error(f"Failed to load user context overrides: {e}")
    return {}

def save_user_override(model_name: str, context_size: int) -> bool:
    """Save user's context override for a specific model."""
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)

        # Thread-safe file operations to prevent race conditions
        with _settings_lock:
            # Load existing settings
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
            else:
                data = {}

            # Update model override
            if "model_overrides" not in data:
                data["model_overrides"] = {}
            data["model_overrides"][model_name] = context_size

            # Save back
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(data, f, indent=2)

        logger.info(f"Saved context override for {model_name}: {context_size}")
        return True
    except Exception as e:
        logger.error(f"Failed to save context override: {e}")
        return False

def delete_user_override(model_name: str) -> bool:
    """Delete user's context override for a specific model, restoring default."""
    try:
        if not os.path.exists(SETTINGS_FILE):
            return True  # Nothing to delete, already at defaults

        # Thread-safe file operations to prevent race conditions
        with _settings_lock:
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)

            # Remove the override if it exists
            if "model_overrides" in data and model_name in data["model_overrides"]:
                del data["model_overrides"][model_name]

                # Save back to file
                with open(SETTINGS_FILE, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Deleted context override for {model_name}, restored to default")
            else:
                logger.info(f"No override found for {model_name}, already at default")

        return True
    except Exception as e:
        logger.error(f"Failed to delete context override for {model_name}: {e}")
        return False

def get_context_size(model_name: str, user_override: Optional[int] = None) -> int:
    """
    Get the appropriate context size for a model.

    Priority order:
    1. Runtime override (if provided)
    2. User's saved preference
    3. Model-specific default
    4. Safe fallback (8192)
    """
    # Runtime override takes precedence
    if user_override:
        return user_override

    # Check user's saved preferences
    user_overrides = load_user_overrides()
    if model_name in user_overrides:
        return user_overrides[model_name]

    # Find model-specific default
    model_lower = model_name.lower()
    for prefix, config in MODEL_CONTEXTS.items():
        if prefix in model_lower:
            return config["default"]

    # Safe fallback for unknown models
    return 8192

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get full context information for a model."""
    model_lower = model_name.lower()

    # Find matching config
    for prefix, config in MODEL_CONTEXTS.items():
        if prefix in model_lower:
            current = get_context_size(model_name)
            return {
                "model": model_name,
                "current": current,
                "default": config["default"],
                "max": config["max"],
                "is_override": model_name in load_user_overrides()
            }

    # Unknown model
    return {
        "model": model_name,
        "current": get_context_size(model_name),
        "default": 8192,
        "max": 128000,  # Optimistic max
        "is_override": model_name in load_user_overrides()
    }

def get_all_model_contexts() -> Dict[str, Dict[str, Any]]:
    """Get context info for all configured models."""
    result = {}
    for prefix, config in MODEL_CONTEXTS.items():
        result[prefix] = {
            "default": config["default"],
            "max": config["max"]
        }
    return result