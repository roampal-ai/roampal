"""
Model Context Limits Configuration
Provides dynamic context and iteration limits based on model capabilities.

Based on research:
- "Lost in the Middle" (Liu et al. 2023): Use 70% of max context for safety
- "Chain-of-Thought" (Wei et al. 2022): Quality vs depth tradeoffs
- Token estimation: ~4 characters per token average
"""

import os
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import time
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)

# Cache for model configurations
_model_cache = {}
_cache_lock = threading.Lock()

# Model configurations: (max_tokens, safe_char_limit, per_tool_limit, base_iterations)
# Updated Dec 2025 with latest models from registry
MODEL_CONFIGURATIONS = {
    # Small models (4K-8K context)
    "codellama:7b": (4096, 8000, 1500, 4),
    "qwen3:4b": (8192, 10000, 2000, 5),
    "qwen3:8b": (8192, 10000, 2000, 5),
    "qwen2.5:3b": (8192, 10000, 2000, 5),  # Small Qwen 2.5
    "llama3.2:3b": (8192, 10000, 2000, 5),  # Small Llama 3.2
    "mistral:7b-instruct-v0.3": (8192, 10000, 2000, 5),

    # Medium models (16K-32K context)
    "codellama:13b": (16384, 40000, 4000, 6),
    "deepseek-r1:7b": (32768, 80000, 8000, 7),
    "deepseek-r1:8b": (32768, 80000, 8000, 7),
    "deepseek-r1:14b": (32768, 80000, 8000, 7),
    "deepseek-r1:32b": (32768, 80000, 8000, 7),
    "qwen3:14b": (32768, 80000, 8000, 7),
    "qwen3:32b": (32768, 80000, 8000, 7),
    "qwen2.5:7b": (32768, 80000, 8000, 7),
    "qwen2.5:14b": (32768, 80000, 8000, 7),
    "qwen2.5:32b": (32768, 80000, 8000, 7),
    "qwen2.5-coder:32b": (32768, 80000, 8000, 7),
    "mixtral:8x7b": (32768, 80000, 8000, 8),  # MoE - slightly more iterations
    "mistral:7b": (32768, 80000, 8000, 7),
    "gemma3:1b": (32768, 80000, 8000, 5),  # Text-only, limited capability

    # Large models (128K+ context)
    "deepseek-coder-v2:16b": (128000, 300000, 30000, 10),
    "gemma3:4b": (128000, 300000, 30000, 8),
    "gemma3:12b": (128000, 300000, 30000, 9),
    "gemma3:27b": (128000, 300000, 30000, 10),
    "llama3.3:70b": (131072, 320000, 32000, 10),
    "llama3.1:8b": (131072, 320000, 32000, 10),
    "llama3.1:70b": (131072, 320000, 32000, 10),
    "llama3.2:8b": (131072, 320000, 32000, 8),  # Llama 3.2 medium
    "qwen2.5:72b": (131072, 320000, 32000, 10),  # Large Qwen 2.5
    "phi3:3.8b": (128000, 300000, 30000, 6),  # Large context but limited reasoning
    "gpt-oss:20b": (128000, 300000, 30000, 10),  # OpenAI OSS efficient model

    # Extra Large models (256K+ context)
    "qwen3-coder:30b": (262144, 600000, 60000, 10),  # MoE 30B, 256K context, 3.3B active

    # Massive models (1M+ context)
    "llama4:maverick": (1048576, 2500000, 250000, 12),  # MoE 401B, 128 experts, 1M context
    "gpt-oss:120b": (131072, 320000, 32000, 12),  # OpenAI OSS flagship

    # Ultra context models (10M+)
    "llama4:scout": (10485760, 25000000, 2500000, 12),  # MoE 109B, 10M context

    # Reference models (not in Ollama)
    "claude-3-opus": (200000, 500000, 50000, 12),  # Reference only
    "gpt-4-turbo": (128000, 300000, 30000, 10),    # Reference only

    # Default fallback for unknown models
    "default": (8192, 10000, 2000, 5),
}

# Model-specific generation parameters for stable output
# Format: {model_pattern: {param: value}}
MODEL_GENERATION_PARAMS = {
    "deepseek-r1": {
        "num_predict": 3072,  # Extra room for reasoning models (thinking + response)
        "temperature": 0.3,  # Lower temperature for focused output
        "repeat_penalty": 1.2,  # Penalize repetition
        "stop": ["You asking", "\n\n\n\n", "system:"]  # Stop on loops
    },
    "default": {
        "num_predict": 2048,  # Standard response length
        "temperature": 0.7,  # Balanced creativity
        "repeat_penalty": 1.1,  # Light repetition penalty
        "stop": []  # No special stop sequences
    }
}

# Safety margin based on research (70% of max for optimal quality)
SAFETY_MARGIN = float(os.getenv("ROAMPAL_CONTEXT_SAFETY_MARGIN", "0.7"))

# Quality thresholds
QUALITY_THRESHOLD = float(os.getenv("ROAMPAL_QUALITY_THRESHOLD", "0.6"))
LATENCY_THRESHOLD = float(os.getenv("ROAMPAL_LATENCY_THRESHOLD", "30"))  # seconds


def get_generation_params(model_name: str) -> Dict[str, Any]:
    """
    Get model-specific generation parameters.

    Args:
        model_name: Name of the model (e.g., "deepseek-r1:8b")

    Returns:
        Dictionary of generation parameters (num_predict, temperature, etc.)
    """
    # Check for exact match first
    if model_name in MODEL_GENERATION_PARAMS:
        return MODEL_GENERATION_PARAMS[model_name].copy()

    # Check for partial match (e.g., "deepseek-r1" in "deepseek-r1:8b")
    for pattern, params in MODEL_GENERATION_PARAMS.items():
        if pattern in model_name.lower():
            return params.copy()

    # Return default params
    return MODEL_GENERATION_PARAMS["default"].copy()

# Task complexity configurations
TASK_COMPLEXITY = {
    "simple": {
        "keywords": ["list", "show", "what", "display", "get", "find file", "check"],
        "iteration_multiplier": 0.6,  # 60% of base iterations
        "min_iterations": 3,  # Need at least 3: execute tools, interpret results, final response
        "max_iterations": 5  # Allow more room for simple tasks with multiple tools
    },
    "medium": {
        "keywords": ["analyze", "explain", "debug", "review", "understand", "compare"],
        "iteration_multiplier": 1.0,  # 100% of base iterations
        "min_iterations": 3,
        "max_iterations": 7
    },
    "complex": {
        "keywords": ["refactor", "implement", "fix all", "entire codebase", "comprehensive",
                    "architecture", "optimize", "migrate", "redesign", "all files"],
        "iteration_multiplier": 1.5,  # 150% of base iterations
        "min_iterations": 5,
        "max_iterations": 15
    }
}


@dataclass
class ModelLimits:
    """Container for model-specific limits"""
    model_name: str
    max_tokens: int
    context_char_limit: int
    per_tool_limit: int
    base_iterations: int
    effective_iterations: int
    task_complexity: str
    safety_margin: float

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "context_limit": self.context_char_limit,
            "per_tool": self.per_tool_limit,
            "iterations": self.effective_iterations,
            "complexity": self.task_complexity
        }


@dataclass
class IterationMetrics:
    """Metrics for quality monitoring per iteration"""
    iteration: int
    latency: float
    token_estimate: int
    response_length: int
    tools_used: int
    coherence_score: float
    timestamp: float

    def should_continue(self) -> bool:
        """Determine if iterations should continue based on metrics"""
        if self.latency > LATENCY_THRESHOLD:
            logger.warning(f"Iteration {self.iteration}: High latency {self.latency}s")
            return False

        if self.coherence_score < QUALITY_THRESHOLD:
            logger.warning(f"Iteration {self.iteration}: Low coherence {self.coherence_score}")
            return False

        return True


@lru_cache(maxsize=128)
def get_model_config(model_name: str) -> Tuple[int, int, int, int]:
    """
    Get configuration for a specific model with caching.
    Returns: (max_tokens, context_char_limit, per_tool_limit, base_iterations)
    """
    # Check cache first
    with _cache_lock:
        if model_name in _model_cache:
            logger.debug(f"Using cached config for {model_name}")
            return _model_cache[model_name]

    model_lower = model_name.lower()

    # Try exact match first
    if model_lower in MODEL_CONFIGURATIONS:
        config = MODEL_CONFIGURATIONS[model_lower]
        with _cache_lock:
            _model_cache[model_name] = config
        return config

    # Try partial match
    for key, config in MODEL_CONFIGURATIONS.items():
        if key in model_lower or model_lower in key:
            logger.info(f"Matched model '{model_name}' to configuration '{key}'")
            with _cache_lock:
                _model_cache[model_name] = config
            return config

    # Check for model size indicators
    if "70b" in model_lower or "65b" in model_lower or "405b" in model_lower:
        # Large model detected
        config = MODEL_CONFIGURATIONS["llama3.1:70b"]
    elif "34b" in model_lower or "30b" in model_lower:
        # Medium-large model
        config = MODEL_CONFIGURATIONS["qwen3:32b"]
    elif "13b" in model_lower or "14b" in model_lower:
        # Medium model
        config = MODEL_CONFIGURATIONS["codellama:13b"]
    else:
        config = None

    if config:
        with _cache_lock:
            _model_cache[model_name] = config
        return config

    logger.warning(f"Unknown model '{model_name}', using default configuration")
    config = MODEL_CONFIGURATIONS["default"]

    # Cache the result
    with _cache_lock:
        _model_cache[model_name] = config

    return config


@lru_cache(maxsize=512)
def detect_task_complexity(message: str) -> str:
    """
    Analyze user message to determine task complexity.
    Returns: 'simple', 'medium', or 'complex'
    Cached for repeated messages.
    """
    message_lower = message.lower()

    # Check for complex indicators first (most restrictive)
    for keyword in TASK_COMPLEXITY["complex"]["keywords"]:
        if keyword in message_lower:
            logger.info(f"Detected complex task from keyword: '{keyword}'")
            return "complex"

    # Check for simple indicators
    for keyword in TASK_COMPLEXITY["simple"]["keywords"]:
        if keyword in message_lower:
            # Additional check: if asking for multiple things, upgrade to medium
            if any(word in message_lower for word in ["and", "also", "then", "plus"]):
                return "medium"
            logger.info(f"Detected simple task from keyword: '{keyword}'")
            return "simple"

    # Default to medium
    logger.info("Defaulting to medium complexity")
    return "medium"


@lru_cache(maxsize=256)
def get_model_limits(model_name: str, user_message: str) -> ModelLimits:
    """
    Get comprehensive limits for a model based on its capabilities and task complexity.
    Cached for performance.
    """
    # Get base configuration (already cached)
    max_tokens, base_context, per_tool, base_iterations = get_model_config(model_name)

    # Apply safety margin to context
    safe_context = int(base_context * SAFETY_MARGIN)
    safe_per_tool = int(per_tool * SAFETY_MARGIN)

    # Detect task complexity
    complexity = detect_task_complexity(user_message)
    complexity_config = TASK_COMPLEXITY[complexity]

    # Calculate effective iterations
    effective_iterations = int(base_iterations * complexity_config["iteration_multiplier"])
    effective_iterations = max(
        complexity_config["min_iterations"],
        min(effective_iterations, complexity_config["max_iterations"])
    )

    logger.info(f"Model limits for {model_name}: context={safe_context}, "
                f"iterations={effective_iterations} ({complexity} task)")

    return ModelLimits(
        model_name=model_name,
        max_tokens=max_tokens,
        context_char_limit=safe_context,
        per_tool_limit=safe_per_tool,
        base_iterations=base_iterations,
        effective_iterations=effective_iterations,
        task_complexity=complexity,
        safety_margin=SAFETY_MARGIN
    )


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text.
    Using ~4 characters per token as a conservative estimate.
    """
    # More accurate estimation could use tiktoken if available
    try:
        import tiktoken
        # Try to use tiktoken for more accurate counting
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback to character-based estimation
        return len(text) // 4


def calculate_token_budget(current_context: str, model_limits: ModelLimits) -> float:
    """
    Calculate remaining token budget as a percentage.
    Returns: float between 0 and 1 (e.g., 0.3 = 30% remaining)
    """
    current_tokens = estimate_tokens(current_context)
    max_tokens = model_limits.max_tokens

    # Reserve 25% for response generation
    usable_tokens = int(max_tokens * 0.75)

    remaining = max(0, usable_tokens - current_tokens)
    budget_percentage = remaining / usable_tokens

    if budget_percentage < 0.2:
        logger.warning(f"Low token budget: {budget_percentage:.1%} remaining")

    return budget_percentage


def smart_truncate(content: str, content_type: str, model_limits: ModelLimits) -> str:
    """
    Intelligently truncate content based on type and model limits.
    """
    limit = model_limits.per_tool_limit

    if len(content) <= limit:
        return content

    if content_type == "list_directory":
        lines = content.split('\n') if isinstance(content, str) else content
        if isinstance(lines, list) and len(lines) > 30:
            # For large directories, show sample + count
            sample = lines[:20] if isinstance(lines[0], str) else [str(item) for item in lines[:20]]
            sample_text = '\n'.join(sample) if isinstance(sample[0], str) else str(sample)
            return f"Directory contains {len(lines)} items. First 20:\n{sample_text}\n... and {len(lines)-20} more items"
        return str(content)[:limit]

    elif content_type == "read_file":
        # For files, show beginning and end
        third = limit // 3
        return (f"{content[:third]}\n\n"
                f"... [File truncated: {len(content)} total chars, showing beginning and end] ...\n\n"
                f"{content[-third:]}")

    elif content_type == "search_results":
        # For search results, prioritize matches
        lines = content.split('\n')
        result_lines = []
        current_size = 0

        for line in lines:
            line_size = len(line)
            if current_size + line_size > limit:
                result_lines.append(f"... [{len(lines) - len(result_lines)} more results truncated]")
                break
            result_lines.append(line)
            current_size += line_size

        return '\n'.join(result_lines)

    else:
        # Generic truncation with ellipsis
        return f"{content[:limit-50]}... [truncated from {len(content)} chars]"


def calculate_coherence_score(response: str, iteration: int, expected_tools: int = 0) -> float:
    """
    Calculate a coherence score for the response.
    Returns: float between 0 and 1
    """
    score = 1.0

    # Check response length (too short might indicate truncation)
    if len(response) < 50:
        score *= 0.7

    # Check for incomplete sentences
    if response and not response.rstrip().endswith(('.', '!', '?', '```', ')', ']', '}')):
        score *= 0.9

    # Check for cut-off indicators
    cutoff_indicators = ["Let me", "I will", "To", "The", "This", "First,", "Now,"]
    if any(response.rstrip().endswith(indicator) for indicator in cutoff_indicators):
        score *= 0.6

    # Iteration penalty (slight degradation over iterations)
    iteration_penalty = 0.95 ** max(0, iteration - 3)
    score *= iteration_penalty

    # Check for error messages
    if "error" in response.lower() or "failed" in response.lower():
        score *= 0.8

    return min(1.0, max(0.0, score))


def should_continue_iterations(
    iteration: int,
    model_limits: ModelLimits,
    metrics: IterationMetrics,
    token_budget: float,
    has_tools: bool
) -> Tuple[bool, str]:
    """
    Determine if iterations should continue based on multiple factors.
    Returns: (should_continue, reason)
    """
    # Check max iterations
    if iteration >= model_limits.effective_iterations:
        return False, f"Reached maximum iterations ({model_limits.effective_iterations})"

    # Check if task is complete (no tools used)
    if not has_tools and iteration > 0:
        return False, "Task complete (no tools needed)"

    # Check token budget
    if token_budget < 0.2:
        return False, f"Low token budget ({token_budget:.1%} remaining)"

    # Check quality metrics
    if not metrics.should_continue():
        return False, f"Quality degradation detected"

    # Check hard limits
    if iteration >= 15:  # Absolute maximum
        return False, "Reached hard iteration limit (15)"

    return True, "Continue"


def load_calibration_data(model_name: str) -> Optional[Dict]:
    """
    Load calibration data for a model if it exists.
    """
    calibration_file = f"data/model_calibrations/{model_name.replace(':', '_')}.json"

    try:
        if os.path.exists(calibration_file):
            with open(calibration_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded calibration data for {model_name}")
                return data
    except Exception as e:
        logger.error(f"Failed to load calibration data: {e}")

    return None


def save_calibration_data(model_name: str, calibration: Dict):
    """
    Save calibration data for future use.
    """
    os.makedirs("data/model_calibrations", exist_ok=True)
    calibration_file = f"data/model_calibrations/{model_name.replace(':', '_')}.json"

    try:
        with open(calibration_file, 'w') as f:
            json.dump(calibration, f, indent=2)
            logger.info(f"Saved calibration data for {model_name}")
    except Exception as e:
        logger.error(f"Failed to save calibration data: {e}")


# Chain strategy management
CHAIN_STRATEGY = os.getenv("ROAMPAL_CHAIN_STRATEGY", "dynamic").lower()

def get_chain_strategy() -> str:
    """Get the current chain strategy (fixed/dynamic/hybrid)"""
    return CHAIN_STRATEGY

def use_dynamic_limits() -> bool:
    """Check if dynamic limits should be used"""
    return CHAIN_STRATEGY in ["dynamic", "hybrid"]

def clear_model_cache():
    """Clear the model configuration cache"""
    global _model_cache
    with _cache_lock:
        _model_cache.clear()
    get_model_config.cache_clear()
    get_model_limits.cache_clear()
    detect_task_complexity.cache_clear()
    logger.info("Cleared model configuration cache")

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring"""
    return {
        "model_config_cache_size": len(_model_cache),
        "model_config_cache_info": get_model_config.cache_info()._asdict(),
        "model_limits_cache_info": get_model_limits.cache_info()._asdict(),
        "complexity_cache_info": detect_task_complexity.cache_info()._asdict()
    }