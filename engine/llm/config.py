"""LLM Provider Factory — creates configured provider instances.

Provides separate LLM instances for Writer (OpenRouter/Gemini Flash Lite)
and Reviewer (OpenRouter/Gemini Pro).
"""

from __future__ import annotations

from engine.llm.openai_compat import OpenAICompatProvider


# ── Writer LLM Config (OpenRouter / Gemini Flash Lite) ──────────────

WRITER_LLM_CONFIG = {
    "provider_class": "OpenAICompatProvider",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "REDACTED_OPENROUTER_KEY",
    "model": "google/gemini-2.5-flash",
    "timeout": 300,
    "temperature": 0.3,
}


def get_writer_llm() -> OpenAICompatProvider:
    """Create and return the Writer's dedicated LLM provider (OpenRouter/Gemini 2.5 Flash).

    Uses a fast, non-reasoning model for reliable JSON output in content generation.
    Reasoning models (3.1 Pro) break JSON parsing — Flash is safer for Writer.

    Returns:
        OpenAICompatProvider configured for Gemini 2.5 Flash via OpenRouter.
    """
    cfg = WRITER_LLM_CONFIG
    return OpenAICompatProvider(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        timeout=cfg["timeout"],
        retries=3,
    )


# ── Reviewer LLM Config (OpenRouter / Gemini Pro) ──────────────────

REVIEWER_LLM_CONFIG = {
    "provider_class": "OpenAICompatProvider",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": "REDACTED_OPENROUTER_KEY",
    "model": "google/gemini-3.1-pro-preview",
    "timeout": 300,
    "temperature": 0.15,
}


def get_reviewer_llm() -> OpenAICompatProvider:
    """Create and return the Reviewer's dedicated LLM provider (OpenRouter/Gemini Pro).

    Uses a SEPARATE, higher-capability model from the Writer — this is
    intentional for independent quality assessment (different model = less bias).

    Returns:
        OpenAICompatProvider configured for Gemini Pro via OpenRouter.
    """
    cfg = REVIEWER_LLM_CONFIG
    return OpenAICompatProvider(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        timeout=cfg["timeout"],
        retries=3,
    )
