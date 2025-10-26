"""
Model-Mode Compatibility Matrix
Defines which modes each model can support based on token limits
"""

# Model to supported modes mapping
# Based on token limits: <8K = basic only, 8K-12K = basic/learning, 12K+ = all modes
MODEL_MODE_MATRIX = {
    # Small models (<8K tokens) - Basic only
    "qwen3:1b": ["basic"],
    "qwen3:3b": ["basic"],
    "tinyllama": ["basic"],
    "phi": ["basic"],

    # Medium models (8K-12K tokens) - Basic and Learning
    "qwen3:8b": ["basic", "learning"],
    "mistral:7b": ["basic", "learning"],
    "mistral:7b-instruct": ["basic", "learning"],
    "gemma:7b": ["basic", "learning"],
    "llama3:8b": ["basic", "learning"],
    "codellama:7b": ["basic", "learning"],

    # Large models (12K+ tokens) - All modes
    "llama3.3:latest": ["basic", "learning", "agent"],
    "llama3.3:70b": ["basic", "learning", "agent"],
    "llama3.1:8b": ["basic", "learning", "agent"],
    "llama3.1:70b": ["basic", "learning", "agent"],
    "qwen2.5-coder:32b": ["basic", "learning", "agent"],
    "qwen2.5:14b": ["basic", "learning", "agent"],
    "qwen2.5:32b": ["basic", "learning", "agent"],
    "deepseek-r1:7b": ["basic", "learning", "agent"],
    "deepseek-r1:8b": ["basic", "learning", "agent"],
    "deepseek-r1:14b": ["basic", "learning", "agent"],
    "deepseek-r1:32b": ["basic", "learning", "agent"],
    "deepseek-r1:70b": ["basic", "learning", "agent"],
    "deepseek-coder-v2:16b": ["basic", "learning", "agent"],
    "deepseek-coder-v2:236b": ["basic", "learning", "agent"],
    "codellama:13b": ["basic", "learning", "agent"],
    "codellama:34b": ["basic", "learning", "agent"],
    "codellama:70b": ["basic", "learning", "agent"],
    "mistral-nemo:12b": ["basic", "learning", "agent"],
    "mistral-large": ["basic", "learning", "agent"],

    # Default for unknown models - conservative
    "default": ["basic", "learning"]
}

def get_supported_modes(model_name: str) -> list[str]:
    """
    Get list of supported modes for a model

    Args:
        model_name: Name of the model

    Returns:
        List of supported mode names
    """
    # Check exact match first
    if model_name in MODEL_MODE_MATRIX:
        return MODEL_MODE_MATRIX[model_name]

    # Check partial matches (e.g., "llama3.3" matches "llama3.3:latest")
    for pattern, modes in MODEL_MODE_MATRIX.items():
        if pattern in model_name or model_name in pattern:
            return modes

    # Return default
    return MODEL_MODE_MATRIX["default"]

def can_use_mode(model_name: str, mode: str) -> bool:
    """
    Check if a model can use a specific mode

    Args:
        model_name: Name of the model
        mode: Mode to check (basic/learning/agent)

    Returns:
        True if model supports the mode
    """
    return mode in get_supported_modes(model_name)

def enforce_mode(model_name: str, requested_mode: str) -> str:
    """
    Enforce mode based on model capabilities

    Args:
        model_name: Name of the model
        requested_mode: The mode user wants

    Returns:
        Actual mode that will be used (may be downgraded)
    """
    supported = get_supported_modes(model_name)

    if requested_mode in supported:
        return requested_mode

    # Downgrade to best available
    if "learning" in supported and requested_mode == "agent":
        return "learning"
    elif "basic" in supported:
        return "basic"

    # Fallback to basic
    return "basic"