"""
Centralized model context window configuration.
Single source of truth for context sizes across the application.
"""

import json
import os
import time
import threading
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import logging

import httpx

from utils.atomic_json import write_json_atomic

logger = logging.getLogger(__name__)

# Thread-safe file locking for concurrent write operations
_settings_lock = threading.Lock()

# User settings file path - Use AppData in production
from config.settings import DATA_PATH
SETTINGS_FILE = str(Path(DATA_PATH) / "user_model_contexts.json")

# Disk cache for dynamic context fetch (Section 6, v0.3.3)
MODEL_METADATA_CACHE_FILE = str(Path(DATA_PATH) / "model_metadata_cache.json")

# Tier 1: in-memory LRU with TTL - fast hot-path lookup per session
_context_cache: Dict[Tuple[str, str], Tuple[int, float]] = {}
_CACHE_TTL = 300  # 5 minutes
_DISK_CACHE_TTL = 86400  # 24 hours

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
        os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)

        with _settings_lock:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    data = json.load(f)
            else:
                data = {}

            if "model_overrides" not in data:
                data["model_overrides"] = {}
            data["model_overrides"][model_name] = context_size

            # v0.3.3 Defect 15: SETTINGS_FILE is a str (line 25), but
            # write_json_atomic signature expects a Path and calls
            # .parent internally. Wrap to avoid `'str' object has no
            # attribute 'parent'` (which silently 500s on every slider
            # commit, so user-set context overrides never persisted).
            write_json_atomic(Path(SETTINGS_FILE), data)

        logger.info(f"Saved context override for {model_name}: {context_size}")
        return True
    except Exception as e:
        logger.error(f"Failed to save context override: {e}")
        return False


def delete_user_override(model_name: str) -> bool:
    """Delete user's context override for a specific model, restoring default."""
    try:
        if not os.path.exists(SETTINGS_FILE):
            return True

        with _settings_lock:
            with open(SETTINGS_FILE, 'r') as f:
                data = json.load(f)

            if "model_overrides" in data and model_name in data["model_overrides"]:
                del data["model_overrides"][model_name]

                # v0.3.3 Defect 15: SETTINGS_FILE→Path wrap (see save_user_override above).
                write_json_atomic(Path(SETTINGS_FILE), data)

                logger.info(f"Deleted context override for {model_name}, restored to default")
            else:
                logger.info(f"No override found for {model_name}, already at default")

        return True
    except Exception as e:
        logger.error(f"Failed to delete context override for {model_name}: {e}")
        return False


# ---- Dynamic context fetch helpers (Section 6, v0.3.3) ----

def _load_disk_cache() -> Dict[str, Any]:
    """Load disk cache from file, filtering out expired entries."""
    if not os.path.exists(MODEL_METADATA_CACHE_FILE):
        return {}
    try:
        with open(MODEL_METADATA_CACHE_FILE, 'r') as f:
            data = json.load(f)
        now = time.time()
        valid = {k: v for k, v in data.items() if v.get("expires_at", 0) > now}
        if len(valid) != len(data):
            write_json_atomic(Path(MODEL_METADATA_CACHE_FILE), valid)
        return valid
    except Exception as e:
        logger.debug(f"Failed to load disk cache for model metadata: {e}")
        return {}


def _save_to_disk_cache(key: str, context_length: int) -> None:
    """Save a single entry to the disk cache with 24h expiry."""
    try:
        cache = _load_disk_cache()
        cache[key] = {
            "context_length": context_length,
            "expires_at": time.time() + _DISK_CACHE_TTL
        }
        os.makedirs(os.path.dirname(MODEL_METADATA_CACHE_FILE), exist_ok=True)
        write_json_atomic(Path(MODEL_METADATA_CACHE_FILE), cache)
    except Exception as e:
        logger.debug(f"Failed to save disk cache entry for {key}: {e}")


def _check_memory_cache(provider: str, model_name: str) -> Optional[int]:
    """Check in-memory cache. Returns value if hit and not expired, else None."""
    key = (provider, model_name)
    entry = _context_cache.get(key)
    if entry is None:
        return None
    value, expires_at = entry
    if time.time() < expires_at:
        return value
    del _context_cache[key]
    return None


def _set_memory_cache(provider: str, model_name: str, value: int) -> None:
    """Set in-memory cache entry with TTL."""
    key = (provider, model_name)
    _context_cache[key] = (value, time.time() + _CACHE_TTL)


async def fetch_context_from_provider(
    model_name: str, provider: str, base_url: str, timeout: float = 5.0
) -> Optional[int]:
    """Return the model's max context window by asking the provider.

    Returns None on timeout, parse failure, or model-not-found.
    Callers fall back to MODEL_CONTEXTS table on None.
    """
    try:
        if provider == "ollama":
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    f"{base_url.rstrip('/')}/api/show",
                    json={"name": model_name},
                )
                r.raise_for_status()
                model_info = r.json().get("model_info", {})
                for key, value in model_info.items():
                    if key.endswith(".context_length"):
                        return int(value)
                return None
        elif provider == "lmstudio":
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(
                    f"{base_url.rstrip('/')}/v1/models/{model_name}"
                )
                r.raise_for_status()
                data = r.json()
                value = (
                    data.get("loaded_context_length")
                    or data.get("max_context_length")
                    or data.get("context_length")
                )
                return int(value) if value else None
        else:
            return None
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as e:
        logger.debug(f"Dynamic context fetch failed for {model_name} via {provider}: {e}")
        return None


async def get_context_size_async(
    model_name: str,
    user_override: Optional[int] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> int:
    """Async context size resolution with dynamic provider fetch.

    Priority order:
    1. Runtime override (if provided)
    2. User's saved preference
    3. Dynamic fetch from provider (cached, 5-min TTL / 24h disk)
    4. MODEL_CONTEXTS prefix match
    5. 8192 fallback

    When provider is None, skips dynamic fetch and falls to step 4.
    """
    if user_override:
        return user_override

    user_overrides = load_user_overrides()
    if model_name in user_overrides:
        return user_overrides[model_name]

    # Dynamic fetch (step 3) - only when provider info is available
    if provider and base_url:
        cache_key = f"{provider}::{model_name}"
        disk_cache = _load_disk_cache()

        # Check disk cache first
        disk_entry = disk_cache.get(cache_key)
        if disk_entry and disk_entry.get("expires_at", 0) > time.time():
            val = disk_entry["context_length"]
            _set_memory_cache(provider, model_name, val)
            return val

        # Check memory cache
        mem_val = _check_memory_cache(provider, model_name)
        if mem_val is not None:
            return mem_val

        # Fetch from provider
        fetched = await fetch_context_from_provider(model_name, provider, base_url)
        if fetched is not None:
            _set_memory_cache(provider, model_name, fetched)
            _save_to_disk_cache(cache_key, fetched)
            return fetched

    # MODEL_CONTEXTS prefix match (step 4)
    model_lower = model_name.lower()
    for prefix, config in MODEL_CONTEXTS.items():
        if prefix in model_lower:
            return config["default"]

    # Safe fallback (step 5)
    return 8192


def get_context_size(model_name: str, user_override: Optional[int] = None) -> int:
    """
    Get the appropriate context size for a model.

    Priority order:
    1. Runtime override (if provided)
    2. User's saved preference
    3. Model-specific default
    4. Safe fallback (8192)
    """
    if user_override:
        return user_override

    user_overrides = load_user_overrides()
    if model_name in user_overrides:
        return user_overrides[model_name]

    model_lower = model_name.lower()
    for prefix, config in MODEL_CONTEXTS.items():
        if prefix in model_lower:
            return config["default"]

    return 8192


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get full context information for a model."""
    model_lower = model_name.lower()

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

    return {
        "model": model_name,
        "current": get_context_size(model_name),
        "default": 8192,
        "max": 128000,
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


# ---- Vision canary probe (Section 4 deferred, v0.3.4) ----

# Tiny 1x1 transparent PNG as base64 - used for vision capability detection
_VISION_CANARY = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQD"
    "wAEhQGAWVLMkQAAAABJRU5ErkJggg=="
)

# v0.3.3 Section 4H — capability detection is fully dynamic.
# Ollama's /api/show exposes a `capabilities` array (e.g. ['vision', 'tools',
# 'thinking', 'completion']) self-declared by the model. We cache that on the
# disk metadata cache (shared with the context-length cache) and use it for
# every runtime gate. NO hardcoded model-name enumeration for vision or tools
# — those rot the day a new model family ships.
#
# LM Studio's OpenAI-compatible /v1/models doesn't expose capabilities per spec.
# For LM Studio: vision falls back to the existing async probe_vision (canary
# 1×1 PNG upload). Tools default to True; runtime tool-incompatibility
# detection in ollama_client retries without tools if the model 400s.


def _caps_cache_key(model_name: str, provider: str) -> str:
    return f"caps::{provider}::{model_name}"


def _load_caps_cache() -> Dict[str, Any]:
    """Load capability cache entries from the shared disk cache file."""
    if not os.path.exists(MODEL_METADATA_CACHE_FILE):
        return {}
    try:
        with open(MODEL_METADATA_CACHE_FILE, 'r') as f:
            data = json.load(f)
        now = time.time()
        return {
            k: v for k, v in data.items()
            if k.startswith("caps::") and v.get("expires_at", 0) > now
        }
    except Exception as e:
        logger.debug(f"Failed to load capabilities cache: {e}")
        return {}


def _save_caps_cache(key: str, capabilities: set) -> None:
    """Save a capability set with 24h expiry. Sets are stored as sorted lists."""
    try:
        cache = _load_disk_cache()
        cache[key] = {
            "capabilities": sorted(capabilities),
            "expires_at": time.time() + _DISK_CACHE_TTL
        }
        os.makedirs(os.path.dirname(MODEL_METADATA_CACHE_FILE), exist_ok=True)
        write_json_atomic(Path(MODEL_METADATA_CACHE_FILE), cache)
    except Exception as e:
        logger.debug(f"Failed to save capability cache entry for {key}: {e}")


async def fetch_ollama_capabilities(
    model_name: str, base_url: str, timeout: float = 5.0
) -> Optional[set]:
    """Ask Ollama what the model says it can do.

    POSTs /api/show and returns the `capabilities` array as a set
    (e.g. {'vision', 'tools', 'thinking', 'completion'}). Returns None on
    network failure or non-200; returns empty set if /api/show responds but
    has no capabilities key (older Ollama versions).

    Result is cached on disk for 24h via _save_caps_cache.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{base_url.rstrip('/')}/api/show",
                json={"name": model_name},
            )
            r.raise_for_status()
            data = r.json()
            raw = data.get("capabilities") or []
            caps = set(raw) if isinstance(raw, list) else set()
            _save_caps_cache(_caps_cache_key(model_name, "ollama"), caps)
            return caps
    except (httpx.HTTPError, ValueError, TypeError) as e:
        logger.debug(f"fetch_ollama_capabilities failed for {model_name}: {e}")
        return None


def get_cached_capabilities(model_name: str, provider: str) -> set:
    """Synchronous read of the capability cache. No network calls.

    Returns the cached capability set or empty set on miss. Cold-start (before
    the first registry refresh populates the cache) returns empty — callers
    should treat empty as "unknown, default False" rather than "no capabilities."
    """
    entry = _load_caps_cache().get(_caps_cache_key(model_name, provider))
    if entry and isinstance(entry.get("capabilities"), list):
        return set(entry["capabilities"])
    return set()


def get_cached_vision(model_name: str, provider: str) -> bool:
    """Sync cache reader for the vision capability."""
    return "vision" in get_cached_capabilities(model_name, provider)


def get_cached_tools(model_name: str, provider: str) -> bool:
    """Sync cache reader for the tools capability.

    For LM Studio: defaults to True because the OpenAI-compatible API has no
    capability metadata, and runtime tool-incompatibility detection in
    ollama_client retries without tools on 400. Preemptively gating tools off
    would hide tool-capable models from the UI.
    """
    if provider == "lmstudio" and not get_cached_capabilities(model_name, provider):
        return True
    return "tools" in get_cached_capabilities(model_name, provider)


async def probe_vision(
    model_name: str, provider: str, base_url: str, timeout: float = 10.0
) -> Optional[bool]:
    """Probe whether a model accepts image input by sending a canary image.

    Returns True if the model processes images without error.
    Returns False if the model rejects images with a known error pattern.
    Returns None on timeout, network failure, or ambiguous response.

    Results are cached on disk for 24 hours to avoid repeated probes.
    """
    # Check capability cache first (populated by fetch_ollama_capabilities on
    # registry refresh, or by a previous probe_vision call below).
    cached_caps = get_cached_capabilities(model_name, provider)
    if cached_caps:
        return "vision" in cached_caps

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "ok"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{_VISION_CANARY}"}
                            },
                        ],
                    }
                ],
                "stream": False,
                "options": {"num_predict": 1},
            }

            if provider == "ollama":
                r = await client.post(
                    f"{base_url.rstrip('/')}/api/chat", json=payload
                )
            elif provider == "lmstudio":
                # LMStudio uses OpenAI-compatible API
                lm_payload = {
                    "model": model_name,
                    "messages": payload["messages"],
                    "max_tokens": 1,
                }
                r = await client.post(
                    f"{base_url.rstrip('/')}/v1/chat/completions", json=lm_payload
                )
            else:
                return None

            if r.status_code == 200:
                result = True
            elif "image" in (r.text.lower() if hasattr(r, 'text') else str(r).lower()):
                result = False
            else:
                logger.debug(
                    f"Vision probe ambiguous for {model_name} via {provider}: "
                    f"status={r.status_code}"
                )
                return None

        # Write probe result into the shared capability cache so future calls
        # (including any sync get_cached_vision lookups) read the same source.
        _save_caps_cache(
            _caps_cache_key(model_name, provider),
            {"vision"} if result else set(),
        )
        logger.info(f"Vision probe for {model_name} via {provider}: capable={result}")
        return result

    except httpx.TimeoutException:
        logger.debug(f"Vision probe timed out for {model_name} via {provider}")
        return None
    except (httpx.HTTPError, ValueError, TypeError) as e:
        err_text = str(e).lower()
        if any(kw in err_text for kw in ["image", "vision", "multimodal"]):
            _save_caps_cache(_caps_cache_key(model_name, provider), set())
            return False
        logger.debug(f"Vision probe failed for {model_name} via {provider}: {e}")
        return None
