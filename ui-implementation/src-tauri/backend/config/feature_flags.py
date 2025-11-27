"""
Feature Flags Configuration System
Controls which advanced features are enabled/disabled
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlags:
    """Feature flags with safe defaults (all advanced features OFF)"""

    # Core memory features (always on)
    ENABLE_MEMORY: bool = True
    ENABLE_SEARCH: bool = True
    ENABLE_OUTCOME_TRACKING: bool = True

    # Advanced features (default OFF for safety)
    ENABLE_KG: bool = True  # Knowledge graph enhancements (ENABLED for better routing)
    ENABLE_AUTONOMY: bool = False  # Autonomous actions
    ENABLE_PATTERN_CRON: bool = False  # Background pattern mining
    ENABLE_REFLECTION_WRITE: bool = False  # Reflection can write files
    ENABLE_AUTO_REFACTOR: bool = False  # Automatic code refactoring
    ENABLE_GIT_OPERATIONS: bool = False  # Git commit/push operations

    # Streaming features (2025-10-16)
    ENABLE_WEBSOCKET_STREAMING: bool = True  # Token-by-token streaming via WebSocket (REQUIRED for UI functionality)

    # Hybrid features (safe partial enablement)
    ENABLE_OUTCOME_DETECTION: bool = True  # Enable outcome detection system
    ENABLE_LLM_OUTCOME_DETECTION: bool = True  # LLM-only detection (no heuristic fallback as of 2025-10-05)
    ENABLE_LLM_AUTONOMOUS_ROUTING: bool = True  # Full LLM autonomy, zero overrides (2025-10-02)
    ENABLE_PROBLEM_SOLUTION_INDEX: bool = True  # Track but don't auto-apply
    ENABLE_AUTO_APPLY_SOLUTIONS: bool = False  # Auto-apply known solutions

    # Dry-run modes (safe exploration)
    PLANNER_DRY_RUN: bool = True  # Plan but don't execute
    ORCHESTRATOR_DRY_RUN: bool = True  # Orchestrate but don't act
    REFLECTION_DRY_RUN: bool = True  # Log insights but don't write

    # Observability
    ENABLE_METRICS: bool = True  # Collect metrics
    ENABLE_DETAILED_LOGGING: bool = False  # Verbose logging
    METRICS_EXPORT: bool = False  # Export metrics to external system

    # Performance tuning
    MAX_MEMORY_SEARCH_RESULTS: int = 10
    MAX_KG_PATH_LENGTH: int = 5
    SOLUTION_MIN_SUCCESS_RATE: float = 0.6
    PROMOTION_THRESHOLD: float = 0.7
    DEMOTION_THRESHOLD: float = 0.3

    # Safety limits
    MAX_AUTONOMOUS_ACTIONS: int = 0  # 0 = disabled
    MAX_FILE_WRITES_PER_SESSION: int = 10
    MAX_REFACTOR_SIZE_KB: int = 100
    REQUIRE_CONFIRMATION: bool = True  # Require user confirmation


class FeatureFlagManager:
    """Manages feature flags with environment override support"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("data/feature_flags.json")
        self.flags = FeatureFlags()
        self.load_flags()

    def load_flags(self):
        """Load flags from file and environment"""
        # 1. Start with defaults
        self.flags = FeatureFlags()

        # 2. Load from JSON config if exists
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    for key, value in config.items():
                        if hasattr(self.flags, key):
                            setattr(self.flags, key, value)
                logger.info(f"Loaded feature flags from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load feature flags: {e}")

        # 3. Override with environment variables (highest priority)
        # Check both ROAMPAL_ and ROAMPAL_ (for backward compatibility)
        for key in dir(self.flags):
            if key.startswith('_'):
                continue

            # Try ROAMPAL_ first, fallback to ROAMPAL_ for backward compatibility
            env_key_roampal = f"ROAMPAL_{key}"
            env_key_loopsmith = f"ROAMPAL_{key}"

            env_key = None
            if env_key_roampal in os.environ:
                env_key = env_key_roampal
            elif env_key_loopsmith in os.environ:
                env_key = env_key_loopsmith

            if env_key:
                value = os.environ[env_key]
                # Convert string to appropriate type
                current_value = getattr(self.flags, key)
                if isinstance(current_value, bool):
                    setattr(self.flags, key, value.lower() in ('true', '1', 'yes'))
                elif isinstance(current_value, int):
                    setattr(self.flags, key, int(value))
                elif isinstance(current_value, float):
                    setattr(self.flags, key, float(value))
                else:
                    setattr(self.flags, key, value)
                logger.info(f"Override {key} from environment: {value}")

    def save_flags(self):
        """Save current flags to file"""
        try:
            self.config_path.parent.mkdir(exist_ok=True)
            with open(self.config_path, 'w') as f:
                # Convert to dict, excluding private attributes
                config = {
                    k: v for k, v in asdict(self.flags).items()
                    if not k.startswith('_')
                }
                json.dump(config, f, indent=2)
            logger.info(f"Saved feature flags to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save feature flags: {e}")

    def get(self, flag_name: str, default: Any = None) -> Any:
        """Get flag value with fallback"""
        return getattr(self.flags, flag_name, default)

    def set(self, flag_name: str, value: Any) -> bool:
        """Set flag value (runtime only, doesn't persist)"""
        if hasattr(self.flags, flag_name):
            setattr(self.flags, flag_name, value)
            logger.info(f"Set {flag_name} = {value}")
            return True
        return False

    # Alias for set() - used by main.py sanitization code
    def set_flag(self, flag_name: str, value: Any) -> bool:
        """Alias for set() - Set flag value (runtime only, doesnt persist)"""
        return self.set(flag_name, value)

    def is_enabled(self, flag_name: str) -> bool:
        """Check if a feature is enabled"""
        value = self.get(flag_name, False)
        return bool(value) if not isinstance(value, bool) else value

    def get_safe_config(self) -> Dict[str, Any]:
        """Get configuration safe for display/export"""
        return {
            k: v for k, v in asdict(self.flags).items()
            if not k.startswith('_') and 'key' not in k.lower() and 'secret' not in k.lower()
        }

    def validate(self) -> Dict[str, str]:
        """Validate flag combinations for safety"""
        issues = {}

        # Autonomous features require confirmation disabled = dangerous
        if self.flags.ENABLE_AUTONOMY and not self.flags.REQUIRE_CONFIRMATION:
            issues["AUTONOMY_NO_CONFIRM"] = "Autonomous mode without confirmation is dangerous"

        # Auto-apply solutions requires problem-solution index
        if self.flags.ENABLE_AUTO_APPLY_SOLUTIONS and not self.flags.ENABLE_PROBLEM_SOLUTION_INDEX:
            issues["AUTO_APPLY_NO_INDEX"] = "Auto-apply requires problem-solution index"

        # Reflection write requires reflection enabled
        if self.flags.ENABLE_REFLECTION_WRITE and self.flags.REFLECTION_DRY_RUN:
            issues["REFLECTION_CONFLICT"] = "Reflection write enabled but in dry-run mode"

        # Pattern cron requires KG
        if self.flags.ENABLE_PATTERN_CRON and not self.flags.ENABLE_KG:
            issues["PATTERN_NO_KG"] = "Pattern mining requires knowledge graph"

        # Git operations are high risk
        if self.flags.ENABLE_GIT_OPERATIONS:
            issues["GIT_WARNING"] = "Git operations enabled - ensure proper access controls"

        return issues

    def apply_safety_profile(self, profile: str):
        """Apply predefined safety profiles"""
        profiles = {
            "development": {
                "ENABLE_KG": True,
                "ENABLE_OUTCOME_DETECTION": True,
                "ENABLE_PROBLEM_SOLUTION_INDEX": True,
                "ENABLE_WEBSOCKET_STREAMING": True,  # REQUIRED for UI
                "PLANNER_DRY_RUN": False,
                "ORCHESTRATOR_DRY_RUN": False,
                "ENABLE_DETAILED_LOGGING": True
            },
            "testing": {
                "ENABLE_KG": True,
                "ENABLE_LLM_OUTCOME_DETECTION": True,
                "ENABLE_PROBLEM_SOLUTION_INDEX": True,
                "ENABLE_WEBSOCKET_STREAMING": True,  # REQUIRED for UI
                "ENABLE_AUTO_APPLY_SOLUTIONS": False,
                "PLANNER_DRY_RUN": True,
                "ORCHESTRATOR_DRY_RUN": True
            },
            "production": {
                "ENABLE_KG": True,
                "ENABLE_OUTCOME_DETECTION": True,
                "ENABLE_PROBLEM_SOLUTION_INDEX": True,
                "ENABLE_WEBSOCKET_STREAMING": True,  # REQUIRED for UI
                "ENABLE_METRICS": True,
                "METRICS_EXPORT": True,
                "ENABLE_DETAILED_LOGGING": False,
                "REQUIRE_CONFIRMATION": True
            },
            "experimental": {
                "ENABLE_KG": True,
                "ENABLE_AUTONOMY": True,
                "ENABLE_LLM_OUTCOME_DETECTION": True,
                "ENABLE_AUTO_APPLY_SOLUTIONS": True,
                "PLANNER_DRY_RUN": False,
                "MAX_AUTONOMOUS_ACTIONS": 5,
                "REQUIRE_CONFIRMATION": False  # Living dangerously
            }
        }

        if profile in profiles:
            for key, value in profiles[profile].items():
                self.set(key, value)
            logger.info(f"Applied safety profile: {profile}")

            # Validate after applying
            issues = self.validate()
            if issues:
                logger.warning(f"Profile {profile} has warnings: {issues}")
        else:
            logger.error(f"Unknown profile: {profile}")


# Global singleton instance
_flag_manager = None


def get_flag_manager(config_path: Optional[str] = None) -> FeatureFlagManager:
    """Get or create the global flag manager"""
    global _flag_manager
    if _flag_manager is None:
        _flag_manager = FeatureFlagManager(config_path)
    return _flag_manager


def is_enabled(flag_name: str) -> bool:
    """Quick check if a feature is enabled"""
    return get_flag_manager().is_enabled(flag_name)


def get_flag(flag_name: str, default: Any = None) -> Any:
    """Quick get flag value"""
    return get_flag_manager().get(flag_name, default)