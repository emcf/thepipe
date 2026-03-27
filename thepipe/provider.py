"""LLM provider presets and client factory for thepipe.

Provides a clean abstraction over different OpenAI-compatible LLM providers,
allowing users to switch between OpenAI, MiniMax, and others via a single
``--provider`` flag or ``LLM_PROVIDER`` environment variable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, Optional

from openai import OpenAI


@dataclass(frozen=True)
class ProviderPreset:
    """Immutable configuration for an OpenAI-compatible LLM provider."""

    name: str
    base_url: str
    default_model: str
    api_key_env: str
    models: Dict[str, str] = field(default_factory=dict)
    temperature_min: float = 0.0
    temperature_max: float = 2.0


# ---------------------------------------------------------------------------
# Built-in provider presets
# ---------------------------------------------------------------------------

OPENAI_PRESET = ProviderPreset(
    name="openai",
    base_url="https://api.openai.com/v1",
    default_model="gpt-4o",
    api_key_env="OPENAI_API_KEY",
    models={
        "gpt-4o": "GPT-4o (latest)",
        "gpt-4o-mini": "GPT-4o Mini",
        "gpt-4-turbo": "GPT-4 Turbo",
    },
)

MINIMAX_PRESET = ProviderPreset(
    name="minimax",
    base_url="https://api.minimax.io/v1",
    default_model="MiniMax-M2.7",
    api_key_env="MINIMAX_API_KEY",
    models={
        "MiniMax-M2.7": "MiniMax M2.7 (latest, 1M context)",
        "MiniMax-M2.7-highspeed": "MiniMax M2.7 High-Speed",
        "MiniMax-M2.5": "MiniMax M2.5 (204K context)",
        "MiniMax-M2.5-highspeed": "MiniMax M2.5 High-Speed (204K context)",
    },
    temperature_min=0.0,
    temperature_max=1.0,
)

PROVIDER_PRESETS: Dict[str, ProviderPreset] = {
    "openai": OPENAI_PRESET,
    "minimax": MINIMAX_PRESET,
}


def get_provider_preset(name: str) -> ProviderPreset:
    """Return a :class:`ProviderPreset` by name (case-insensitive).

    Raises ``ValueError`` for unknown providers.
    """
    key = name.lower()
    if key not in PROVIDER_PRESETS:
        available = ", ".join(sorted(PROVIDER_PRESETS))
        raise ValueError(
            f"Unknown provider '{name}'. Available providers: {available}"
        )
    return PROVIDER_PRESETS[key]


def detect_provider() -> str:
    """Auto-detect provider from available environment variables.

    Returns ``"minimax"`` if ``MINIMAX_API_KEY`` is set (and ``OPENAI_API_KEY``
    is not), otherwise ``"openai"``.
    """
    if os.getenv("MINIMAX_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        return "minimax"
    return "openai"


def clamp_temperature(
    temperature: Optional[float], preset: ProviderPreset
) -> Optional[float]:
    """Clamp *temperature* to the provider's valid range, or return ``None``."""
    if temperature is None:
        return None
    return max(preset.temperature_min, min(temperature, preset.temperature_max))


def strip_think_tags(text: str) -> str:
    """Remove ``<think>…</think>`` blocks from LLM output.

    Some MiniMax models emit reasoning traces wrapped in ``<think>`` tags that
    should be stripped before returning results to the caller.
    """
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def create_provider_client(
    provider: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple[OpenAI, ProviderPreset]:
    """Create an :class:`OpenAI` client configured for the given *provider*.

    Parameters
    ----------
    provider:
        Provider name (``"openai"`` or ``"minimax"``).  When ``None`` the
        provider is auto-detected from environment variables.
    api_key:
        Explicit API key.  Falls back to the provider's ``api_key_env``.
    base_url:
        Override the provider's default base URL.

    Returns
    -------
    tuple[OpenAI, ProviderPreset]
        A configured client and the resolved preset.
    """
    if provider is None:
        provider = detect_provider()

    preset = get_provider_preset(provider)
    resolved_key = api_key or os.getenv(preset.api_key_env, "")
    resolved_url = base_url or preset.base_url

    if not resolved_key:
        raise ValueError(
            f"No API key found for provider '{preset.name}'. "
            f"Set the {preset.api_key_env} environment variable or pass --api-key."
        )

    client = OpenAI(api_key=resolved_key, base_url=resolved_url)
    return client, preset
