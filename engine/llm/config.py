"""LLM Provider Factory — creates configured provider instances.

Provides separate LLM instances for Writer (MiniMax) and Reviewer (OpenRouter/Gemini).
"""

from __future__ import annotations

from engine.llm.openai_compat import OpenAICompatProvider


# ── Reviewer LLM Config (OpenRouter / Gemini Flash Lite) ────────────

REVIEWER_LLM_CONFIG = {
    "provider_class": "OpenAICompatProvider",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "REDACTED_OPENROUTER_KEY",
    "model": "google/gemini-3.1-flash-lite-preview",
    "timeout": 180,
    "temperature": 0.15,
}


def get_reviewer_llm() -> OpenAICompatProvider:
    """Create and return the Reviewer's dedicated LLM provider (OpenRouter/Gemini).

    Uses a SEPARATE model from the Writer — this is intentional for
    independent quality assessment (different model = less bias).

    Returns:
        OpenAICompatProvider configured for Gemini Flash Lite via OpenRouter.
    """
    cfg = REVIEWER_LLM_CONFIG
    return OpenAICompatProvider(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        timeout=cfg["timeout"],
        retries=3,
    )
