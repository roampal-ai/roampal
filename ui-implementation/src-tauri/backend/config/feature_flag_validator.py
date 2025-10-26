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
