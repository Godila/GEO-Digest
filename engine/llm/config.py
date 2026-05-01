"""LLM Provider Factory — creates configured provider instances.

Writer:  DeepSeek V4 Pro (reasoning, 384K output) — deep article writing.
Editor:  DeepSeek V4 Flash (reasoning, cheap) — critical decisions: source selection, proposals.
Reviewer: Gemini 3.1 Pro (reasoning, best) — quality review + rewrite.
Reader:  DeepSeek V4 Flash (reasoning, 1M context) — evidence extraction from PDFs.
Scout:  MiniMax M2.7 — search + classify.

All keys and models configured via environment variables (see .env.example).
"""
import os
from engine.llm.openai_compat import OpenAICompatProvider

# ── OpenRouter — from env vars ──
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OR_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ── Model names — configurable via env, sensible defaults ──
WRITER_MODEL = os.environ.get("WRITER_MODEL", "deepseek/deepseek-v4-pro")
EDITOR_MODEL = os.environ.get("EDITOR_MODEL", "deepseek/deepseek-v4-flash")
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "google/gemini-3.1-pro-preview")
READER_MODEL = os.environ.get("READER_MODEL", "deepseek/deepseek-v4-flash")


def get_writer_llm() -> OpenAICompatProvider:
    """Writer LLM: DeepSeek V4 Pro — 384K output, reasoning, deep writing."""
    return OpenAICompatProvider(
        api_key=OR_KEY,
        base_url=OR_BASE,
        model=WRITER_MODEL,
        timeout=600,
        reasoning_effort="high",     # Maximum reasoning for deep analysis
        use_json_mode=True,
        use_response_healing=True,
    )


def get_editor_llm() -> OpenAICompatProvider:
    """Editor LLM: DeepSeek V4 Flash — fast, cheap, reasoning-capable.

    Editor makes critical decisions:
    - Which sources to select (determines article depth)
    - How many DOI per proposal (was 3-4, should be 15-30)
    - Article structure and scope
    """
    return OpenAICompatProvider(
        api_key=OR_KEY,
        base_url=OR_BASE,
        model=EDITOR_MODEL,
        timeout=300,
        reasoning_effort="medium",   # Balanced for source decisions
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


def get_reader_llm() -> OpenAICompatProvider:
    """Reader LLM: DeepSeek V4 Flash — fast, 1M context, reasoning.

    Reader extracts structured evidence from PDFs (60-120K chars each).
    DeepSeek V4 Flash handles long context without timeout, unlike MiniMax.
    """
    return OpenAICompatProvider(
        api_key=OR_KEY,
        base_url=OR_BASE,
        model=READER_MODEL,
        timeout=300,
        reasoning_effort="low",      # Evidence extraction — straightforward
        use_json_mode=True,
        use_response_healing=True,
    )
