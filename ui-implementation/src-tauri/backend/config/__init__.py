"""
Configuration module for Roampal
"""

from .model_limits import (
    ModelLimits,
    IterationMetrics,
    get_model_limits,
    detect_task_complexity,
    estimate_tokens,
    calculate_token_budget,
    smart_truncate,
    calculate_coherence_score,
    should_continue_iterations,
    use_dynamic_limits,
    get_chain_strategy
)

__all__ = [
    "ModelLimits",
    "IterationMetrics",
    "get_model_limits",
    "detect_task_complexity",
    "estimate_tokens",
    "calculate_token_budget",
    "smart_truncate",
    "calculate_coherence_score",
    "should_continue_iterations",
    "use_dynamic_limits",
    "get_chain_strategy"
]