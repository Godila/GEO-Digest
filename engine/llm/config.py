"""LLM Provider Factory — creates configured provider instances.

Writer:  Gemini 3 Flash (reasoning, cheap) — bulk article writing.
Editor:  Gemini 3 Flash (reasoning, medium) — critical decisions: source selection, proposals.
Reviewer: Gemini 3.1 Pro (reasoning, best) — quality review + rewrite.
Reader:  MiniMax M2.7 (via Anthropic API) — evidence extraction.
Scout:  MiniMax M2.7 — search + classify.

All keys and models configured via environment variables (see .env.example).
"""
import os
from engine.llm.openai_compat import OpenAICompatProvider

# ── OpenRouter (Gemini) — from env vars ──
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ── Model names — configurable via env, sensible defaults ──
WRITER_MODEL = os.environ.get("WRITER_MODEL", "google/gemini-3-flash-preview")
EDITOR_MODEL = os.environ.get("EDITOR_MODEL", "google/gemini-3-flash-preview")
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "google/gemini-3.1-pro-preview")


def get_writer_llm() -> OpenAICompatProvider:
    """Writer LLM: Gemini 3 Flash — fast, cheap, reasoning-capable."""
    return OpenAICompatProvider(
        api_key=OR_KEY,
        base_url=OR_BASE,
        model=WRITER_MODEL,
        timeout=300,
        reasoning_effort="low",      # Minimal thinking → max content tokens
        use_json_mode=True,
        use_response_healing=True,
    )


def get_editor_llm() -> OpenAICompatProvider:
    """Editor LLM: Gemini 3 Flash with deeper reasoning.

    Editor makes critical decisions:
    - Which sources to select (determines article depth)
    - How many DOI per proposal (was 3-4, should be 15-30)
    - Article structure and scope
    Needs reasoning_effort="medium" for nuanced source evaluation.
    """
    return OpenAICompatProvider(
        api_key=OR_KEY,
        base_url=OR_BASE,
        model=EDITOR_MODEL,
        timeout=300,
        reasoning_effort="medium",   # Deeper thinking for source decisions
        use_json_mode=False,         # Editor returns text + JSON, not pure JSON
        use_response_healing=True,   # Auto-fix JSON in proposals
    )


def get_reviewer_llm() -> OpenAICompatProvider:
    """Reviewer LLM: Gemini 3.1 Pro — best quality review."""
    return OpenAICompatProvider(
        api_key=OR_KEY,
        base_url=OR_BASE,
        model=REVIEWER_MODEL,
        timeout=300,
        reasoning_effort="medium",   # More thinking for nuanced review
        use_json_mode=True,
        use_response_healing=True,
    )
