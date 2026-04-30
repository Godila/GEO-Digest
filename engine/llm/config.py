"""LLM Provider Factory — creates configured provider instances.

Writer:  Gemini 3 Flash (reasoning, cheap) — bulk article writing.
Reviewer: Gemini 3.1 Pro (reasoning, best) — quality review + rewrite.
Reader:  MiniMax M2.7 (via Anthropic API) — evidence extraction.
"""

from engine.llm.openai_compat import OpenAICompatProvider

# ── OpenRouter (Gemini) ──
OR_KEY = "REDACTED_OPENROUTER_KEY"
OR_BASE = "https://openrouter.ai/api/v1"


def get_writer_llm() -> OpenAICompatProvider:
    """Writer LLM: Gemini 3 Flash — fast, cheap, reasoning-capable."""
    return OpenAICompatProvider(
        api_key=OR_KEY,
        base_url=OR_BASE,
        model="google/gemini-3-flash-preview",
        timeout=300,
        reasoning_effort="low",      # Minimal thinking → max content tokens
        use_json_mode=True,
        use_response_healing=True,
    )


def get_reviewer_llm() -> OpenAICompatProvider:
    """Reviewer LLM: Gemini 3.1 Pro — best quality review."""
    return OpenAICompatProvider(
        api_key=OR_KEY,
        base_url=OR_BASE,
        model="google/gemini-3.1-pro-preview",
        timeout=300,
        reasoning_effort="medium",   # More thinking for nuanced review
        use_json_mode=True,
        use_response_healing=True,
    )
