"""
Unified Model Registry API
Single source of truth for model metadata across all providers.
"""
from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
import logging

from config.model_contexts import MODEL_CONTEXTS, get_context_size

logger = logging.getLogger(__name__)
router = APIRouter(tags=["model-registry"])

# Tool-capable model families (whitelist based on testing)
TOOL_CAPABLE_FAMILIES = {
    "verified": [
        # Models personally tested with reliable tool calling
        "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b", "qwen2.5:72b",
        "llama3.3:70b", "llama3.1:70b",
        "gpt-oss:20b", "gpt-oss:120b",
        "mixtral:8x7b",
        "command-r", "command-r-plus"
    ],
    "compatible": [
        # Same family as verified, likely works but untested
        "qwen2.5:3b",  # Other qwen2.5 variants
        "llama3.2:3b", "llama3.2:8b", "llama3.1:8b",  # Llama 3.x family
        "mistral:7b", "phi-4", "phi3"
    ],
    "experimental": [
        # Available but known issues with tools
        "deepseek-r1", "deepseek-coder", "dolphin"
    ]
}

def get_model_tier(model_name: str) -> str:
    """Determine model tier (verified/compatible/experimental)"""
    model_lower = model_name.lower()

    # Check verified first (highest confidence)
    for verified_model in TOOL_CAPABLE_FAMILIES["verified"]:
        if verified_model in model_lower:
            return "verified"

    # Check compatible (same family, likely works)
    for compatible_model in TOOL_CAPABLE_FAMILIES["compatible"]:
        if compatible_model in model_lower:
            return "compatible"

    # Check experimental (known issues)
    for experimental_model in TOOL_CAPABLE_FAMILIES["experimental"]:
        if experimental_model in model_lower:
            return "experimental"

    # Unknown models default to experimental
    return "experimental"

def is_tool_capable(model_name: str) -> bool:
    """Check if model supports reliable tool calling"""
    tier = get_model_tier(model_name)
    # Only verified and compatible models are considered tool-capable
    return tier in ["verified", "compatible"]

def get_model_description(model_name: str, tier: str) -> str:
    """Generate description based on model name and tier"""
    descriptions = {
        # Qwen 2.5 Series
        "qwen2.5:7b": "Best-in-class tool calling",
        "qwen2.5:14b": "Larger Qwen - Great tool performance",
        "qwen2.5:32b": "Powerful Qwen - Excellent tools",
        "qwen2.5:72b": "Massive Qwen - Superior tool calling",
        "qwen2.5:3b": "Efficient with tool support",

        # Llama 3 Series
        "llama3.3:70b": "Meta's latest 70B - Native tools, 128K context",
        "llama3.1:70b": "Meta's flagship - Excellent tools, 128K context",
        "llama3.1:8b": "Compact Llama - Good tools, 128K context",
        "llama3.2:8b": "Latest compact Llama - Tool support",
        "llama3.2:3b": "Ultra-compact Llama - May have inconsistent tools",

        # OpenAI Open Source
        "gpt-oss:20b": "OpenAI efficient model - Excellent tools",
        "gpt-oss:120b": "OpenAI flagship open model - Native tools",

        # Mistral Family
        "mistral:7b": "Fast and efficient with tools",
        "mixtral:8x7b": "Mixture of Experts - Native tools",

        # Other
        "command-r": "Cohere's command model - Strong tools",
        "phi-4": "Microsoft's efficient model",
        "phi3": "Compact Microsoft model",
    }

    # Get base description
    desc = None
    model_lower = model_name.lower()
    for key, description in descriptions.items():
        if key in model_lower:
            desc = description
            break

    if not desc:
        desc = "Custom model"

    # Add tier indicator
    if tier == "verified":
        return f"✅ {desc}"
    elif tier == "compatible":
        return f"⚠️ {desc} (untested)"
    else:
        return f"🧪 {desc} (experimental - may not work with tools)"

def extract_quantization(file_name: str) -> Optional[str]:
    """Extract quantization level from GGUF filename"""
    if not file_name or "Q" not in file_name:
        return None

    # Look for patterns like Q4_K_M, Q5_K_S, Q8_0, etc.
    import re
    match = re.search(r'Q(\d+)_([A-Z0-9_]+)', file_name)
    if match:
        return f"Q{match.group(1)}_{match.group(2)}"

    return None

@router.get("/api/model/registry")
async def get_model_registry(
    provider: Optional[str] = None,
    tool_capable_only: bool = True
) -> Dict[str, Any]:
    """
    Get unified model registry with metadata from all config sources.

    Query Parameters:
    - provider: Filter by provider (ollama/lmstudio)
    - tool_capable_only: Only return models with reliable tool support (default: True)

    Returns:
    - models: List of model metadata
    - count: Total number of models
    - providers: List of available providers
    """
    from app.routers.model_switcher import MODEL_TO_HUGGINGFACE, PROVIDERS, detect_provider

    registry = []
    available_providers = []

    # Detect which providers are running
    for provider_name, provider_config in PROVIDERS.items():
        if provider and provider != provider_name:
            continue

        try:
            provider_info = await detect_provider(provider_name, provider_config)
            if not provider_info:
                logger.debug(f"Provider {provider_name} not available")
                continue

            available_providers.append(provider_name)

            # Get models from provider
            if provider_name == "ollama":
                # Use Ollama API to get installed models
                import httpx
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"http://localhost:{provider_config['port']}/api/tags")
                        if response.status_code == 200:
                            data = response.json()
                            models = [m['name'] for m in data.get('models', [])]
                        else:
                            models = []
                except Exception as e:
                    logger.error(f"Failed to fetch Ollama models: {e}")
                    models = []

            elif provider_name == "lmstudio":
                # Use LM Studio API to get loaded models
                import httpx
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"http://localhost:{provider_config['port']}/v1/models")
                        if response.status_code == 200:
                            data = response.json()
                            models = [m['id'] for m in data.get('data', [])]
                        else:
                            models = []
                except Exception as e:
                    logger.error(f"Failed to fetch LM Studio models: {e}")
                    models = []
            else:
                models = []

            # Process each model
            for model_name in models:
                # Skip non-chat models (embeddings, vision, etc.)
                if any(x in model_name.lower() for x in ['embed', 'llava', 'nomic', 'bge-', 'minilm']):
                    continue

                # Check tool capability
                tier = get_model_tier(model_name)
                if tool_capable_only and not is_tool_capable(model_name):
                    continue

                # Get context window
                context_size = get_context_size(model_name)

                # Get GGUF info if available
                gguf_info = MODEL_TO_HUGGINGFACE.get(model_name, {})
                size_gb = gguf_info.get("size_gb")
                quantization = extract_quantization(gguf_info.get("file", ""))

                # Get max context from MODEL_CONTEXTS
                model_family = model_name.split(':')[0]
                max_context = MODEL_CONTEXTS.get(model_family, {}).get("max", context_size)

                registry.append({
                    "name": model_name,
                    "provider": provider_name,
                    "tier": tier,
                    "size_gb": size_gb,
                    "context_window": context_size,
                    "max_context": max_context,
                    "quantization": quantization,
                    "capabilities": {
                        "tools": is_tool_capable(model_name),
                        "streaming": True,
                        "vision": "llava" in model_name.lower() or "gemma3" in model_name.lower()
                    },
                    "description": get_model_description(model_name, tier)
                })

        except Exception as e:
            logger.error(f"Error processing provider {provider_name}: {e}", exc_info=True)
            continue

    return {
        "models": registry,
        "count": len(registry),
        "providers": available_providers,
        "tool_capable_families": TOOL_CAPABLE_FAMILIES
    }


@router.get("/api/model/tiers")
async def get_model_tiers() -> Dict[str, Any]:
    """
    Get information about model tiers and tool capability.

    Returns tier definitions and which models belong to each tier.
    """
    return {
        "tiers": {
            "verified": {
                "name": "Verified",
                "description": "Models personally tested with reliable tool calling",
                "badge": "✅",
                "models": TOOL_CAPABLE_FAMILIES["verified"]
            },
            "compatible": {
                "name": "Compatible",
                "description": "Same family as verified models, likely works but untested",
                "badge": "⚠️",
                "models": TOOL_CAPABLE_FAMILIES["compatible"]
            },
            "experimental": {
                "name": "Experimental",
                "description": "Available but may have issues with tool calling",
                "badge": "🧪",
                "models": TOOL_CAPABLE_FAMILIES["experimental"]
            }
        }
    }
