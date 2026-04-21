"""v0.3.2 (0b): pure helpers for startup model selection.

Extracted from main.py lifespan so the precedence rule (env → persisted
JSON → first available) can be unit-tested without booting the whole app.
"""

from __future__ import annotations

from typing import Optional, Tuple


def select_startup_model(
    available_models: list[str],
    configured_env_model: Optional[str],
    persisted: Optional[dict],
    active_provider: str,
) -> Tuple[Optional[str], str]:
    """Return (selected_model, source) for this startup.

    Precedence:
      1. env var (configured_env_model) — if present AND in available_models
      2. persisted JSON — if provider matches AND model is in available_models
      3. available_models[0] — fall-through, only when list non-empty
      4. (None, "none") — nothing to select

    `source` is one of "env" | "persisted" | "fallback" | "none".
    """
    if configured_env_model and configured_env_model in available_models:
        return configured_env_model, "env"

    if (
        persisted
        and persisted.get("provider") == active_provider
        and persisted.get("model") in available_models
    ):
        return persisted["model"], "persisted"

    if available_models:
        return available_models[0], "fallback"

    return None, "none"


def select_active_provider(
    detected_providers: dict,
    persisted: Optional[dict],
    configured_env_provider: Optional[str],
) -> Tuple[Optional[str], str]:
    """v0.3.2: decide which provider to boot with.

    Fixes the "switched to LM Studio last session, restart falls back to
    first-available Ollama model" bug: if the user persisted a model on a
    specific provider AND that provider is currently detected, honor their
    choice over the env-var default. Otherwise fall through to the prior
    behavior (env > first detected).

    Returns (provider_name, source) where source is
    "persisted" | "env" | "first_detected" | "none".
    """
    # 1. Persisted selection drives the active provider when it's reachable.
    if persisted:
        p_provider = persisted.get("provider")
        if p_provider and p_provider in detected_providers:
            return p_provider, "persisted"

    # 2. Env-configured provider, if detected.
    if configured_env_provider and configured_env_provider in detected_providers:
        return configured_env_provider, "env"

    # 3. First detected provider (deterministic by dict order).
    if detected_providers:
        return next(iter(detected_providers)), "first_detected"

    return None, "none"


def should_migrate_env_to_json(
    persisted: Optional[dict],
    configured_env_model: Optional[str],
    available_models: Optional[list[str]] = None,
) -> bool:
    """One-shot migration: seed main_model_config.json from env vars on first
    v0.3.2 launch when no JSON file exists yet but env vars are set. Subsequent
    starts read the JSON, so this fires at most once per user.

    v0.3.2 follow-up: only migrate when the env-configured model is actually
    installed. Prevents the file from being seeded with stale env vars like
    OLLAMA_MODEL=codellama:latest when codellama isn't pulled anymore.
    """
    if persisted is not None or not configured_env_model:
        return False
    if available_models is None:
        # Caller didn't pass the list — preserve legacy behavior for tests.
        return True
    return configured_env_model in available_models
