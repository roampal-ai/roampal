"""
Feature Flag Validator - Production safeguards for dangerous flag combinations
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


class FeatureFlagValidator:
    """
    Validates feature flag combinations to prevent dangerous configurations in production.
    Enforces safety rules and provides clear warnings.
    """

    # Define dangerous combinations
    DANGEROUS_COMBINATIONS = [
        # Autonomy without confirmation is dangerous
        ({"ENABLE_AUTONOMY": True, "REQUIRE_CONFIRMATION": False},
         "Autonomy enabled without confirmation - HIGH RISK of unintended changes"),

        # File writes without dry run in production
        ({"MAX_FILE_WRITES_PER_SESSION": lambda x: x > 0, "PLANNER_DRY_RUN": False},
         "File writes enabled without dry-run mode - Risk of data loss"),

        # Git operations without confirmation
        ({"ENABLE_GIT_OPERATIONS": True, "REQUIRE_CONFIRMATION": False},
         "Git operations without confirmation - Risk of unwanted commits"),

        # Auto-apply solutions without limits
        ({"ENABLE_AUTO_APPLY_SOLUTIONS": True, "MAX_AUTONOMOUS_ACTIONS": lambda x: x > 10},
         "Auto-apply with high autonomous action limit - Risk of runaway automation"),
    ]

    @staticmethod
    def validate_and_log(config: Dict[str, Any], is_production: bool) -> bool:
        """
        Validate feature flag configuration and log warnings.

        Args:
            config: Feature flag configuration dictionary
            is_production: Whether running in production mode

        Returns:
            True if configuration is valid, False if dangerous combinations detected
        """
        warnings = []

        for conditions, warning_message in FeatureFlagValidator.DANGEROUS_COMBINATIONS:
            # Check if all conditions in this combination are met
            matches = True
            for flag_name, expected_value in conditions.items():
                actual_value = config.get(flag_name)

                # Handle callable conditions (lambda functions)
                if callable(expected_value):
                    if not expected_value(actual_value):
                        matches = False
                        break
                else:
                    if actual_value != expected_value:
                        matches = False
                        break

            if matches:
                warnings.append(warning_message)
                logger.warning(f"⚠️  DANGEROUS CONFIGURATION: {warning_message}")

        if warnings and is_production:
            logger.critical("❌ PRODUCTION MODE: Dangerous flag combinations detected!")
            for warning in warnings:
                logger.critical(f"  - {warning}")
            return False

        if warnings:
            logger.warning("⚠️  Dangerous flag combinations detected (non-production mode)")
        else:
            logger.info("✓ Feature flag configuration validated - no dangerous combinations")

        return True

    @staticmethod
    def get_safe_production_config() -> Dict[str, Any]:
        """
        Returns safe default configuration for production deployment.
        All autonomous/dangerous features are disabled.
        """
        return {
            "ENABLE_AUTONOMY": False,
            "ENABLE_AUTO_APPLY_SOLUTIONS": False,
            "ENABLE_AUTO_REFACTOR": False,
            "ENABLE_GIT_OPERATIONS": False,
            "MAX_AUTONOMOUS_ACTIONS": 0,
            "MAX_FILE_WRITES_PER_SESSION": 0,
            "REQUIRE_CONFIRMATION": True,
            "PLANNER_DRY_RUN": True,
            "ORCHESTRATOR_DRY_RUN": True,
            "REFLECTION_DRY_RUN": True,
            "ENABLE_PATTERN_CRON": False,
            "ENABLE_REFLECTION_WRITE": False,
            # Keep these enabled as they're read-only/safe
            "ENABLE_MEMORY": True,
            "ENABLE_SEARCH": True,
            "ENABLE_KG": True,
            "ENABLE_METRICS": True,
            "ENABLE_OUTCOME_DETECTION": True,
            "ENABLE_OUTCOME_TRACKING": True,
            "ENABLE_LLM_OUTCOME_DETECTION": True,
            "ENABLE_PROBLEM_SOLUTION_INDEX": True,
        }

    @staticmethod
    def sanitize_for_production(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize configuration for production by disabling dangerous features.
        Returns a new config dict with safe values enforced.
        """
        safe_defaults = FeatureFlagValidator.get_safe_production_config()
        sanitized = dict(config)

        # Force disable dangerous features for production
        dangerous_flags = [
            "ENABLE_AUTONOMY",
            "ENABLE_AUTO_APPLY_SOLUTIONS",
            "ENABLE_AUTO_REFACTOR",
            "ENABLE_GIT_OPERATIONS",
        ]

        for flag in dangerous_flags:
            if config.get(flag, False):
                logger.warning(f"⚠️  Production safety: Disabling {flag}")
                sanitized[flag] = False

        # Enforce safe limits
        if config.get("MAX_AUTONOMOUS_ACTIONS", 0) > 0:
            logger.warning("⚠️  Production safety: Setting MAX_AUTONOMOUS_ACTIONS to 0")
            sanitized["MAX_AUTONOMOUS_ACTIONS"] = 0

        if config.get("MAX_FILE_WRITES_PER_SESSION", 0) > 0:
            logger.warning("⚠️  Production safety: Setting MAX_FILE_WRITES_PER_SESSION to 0")
            sanitized["MAX_FILE_WRITES_PER_SESSION"] = 0

        # Ensure confirmation is required
        if not config.get("REQUIRE_CONFIRMATION", True):
            logger.warning("⚠️  Production safety: Enabling REQUIRE_CONFIRMATION")
            sanitized["REQUIRE_CONFIRMATION"] = True

        # Ensure dry-run modes for safety
        for dry_run_flag in ["PLANNER_DRY_RUN", "ORCHESTRATOR_DRY_RUN", "REFLECTION_DRY_RUN"]:
            if not config.get(dry_run_flag, True):
                logger.warning(f"⚠️  Production safety: Enabling {dry_run_flag}")
                sanitized[dry_run_flag] = True

        return sanitized
