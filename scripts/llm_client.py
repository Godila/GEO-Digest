#!/usr/bin/env python3
"""
llm_client.py — Unified MiniMax M2.7 LLM client for Geo-Digest.

Replaces two separate LLM implementations:
  - digest.py: GLM-5V-turbo via Z.AI (OpenAI-compatible)
  - build_graph.py: MiniMax-M2.7 via Anthropic-compatible API

Now everything goes through MiniMax M2.7 only.
GLM is freed up for Hermes usage.

Usage:
  from llm_client import call_minimax, call_llm

  # Anthropic-style (messages list) — for graph semantic edges
  response = call_minimax(
      messages=[{"role": "user", "content": "..."}],
      system="You are...",
      max_tokens=500
  )

  # Simple prompt style — for article enrichment (replaces old GLM call)
  response = call_llm(prompt="...", max_tokens=1024)
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
ENV_FILE = BASE / ".env"


def load_env() -> dict:
    """Load key=value pairs from .env file."""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def _get_config() -> dict:
    """Read MiniMax config from .env."""
    env = load_env()
    return {
        "api_key": env.get("MINIMAX_API_KEY", ""),
        "base_url": env.get("MINIMAX_BASE_URL", "https://api.minimax.io/anthropic").rstrip("/"),
        "model": env.get("MINIMAX_MODEL", "MiniMax-M2.7"),
    }


def call_minimax(
    messages: list[dict],
    system: str = "",
    max_tokens: int = 500,
    timeout: int = 60,
    retries: int = 3,
) -> str:
    """
    Call MiniMax M2.7 via Anthropic-compatible API.

    MiniMax-M2.7 is a reasoning model — it uses tokens for internal thinking
    before generating the response. We add a buffer to max_tokens.

    Args:
        messages: list of {"role": "user"|"assistant", "content": str}
        system: system prompt
        max_tokens: max response tokens (buffer added for thinking)
        timeout: request timeout in seconds
        retries: number of retries on transient errors

    Returns:
        Response text string

    Raises:
        Exception on API error after all retries exhausted
    """
    cfg = _get_config()

    if not cfg["api_key"]:
        raise ValueError("MINIMAX_API_KEY not set in .env")

    # Reasoning model: needs extra tokens for internal thinking
    # M2.7 uses ~100-500 tokens for thinking, then generates content
    effective_max = max(max_tokens + 512, 2048)

    url = f"{cfg['base_url']}/v1/messages"
    payload = json.dumps({
        "model": cfg["model"],
        "max_tokens": effective_max,
        "system": system,
        "messages": messages,
        "disable_thinking": True,  # M2.7: skip extended reasoning, get direct response
    }).encode()

    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "x-api-key": cfg["api_key"],
        "anthropic-version": "2023-06-01",
    })

    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())

            # Extract text from content blocks
            # M2.7 returns "thinking" blocks (reasoning) and optionally "text" blocks
            text_parts = []
            thinking_parts = []
            for block in data.get("content", []):
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "thinking":
                    thinking_parts.append(block.get("thinking", ""))

            # Prefer explicit text output; fall back to thinking content
            result = "\n".join(text_parts) if text_parts else "\n".join(thinking_parts)

            if result:
                usage = data.get("usage", {})
                print(f"  [MiniMax] OK ({len(result)} chars, tokens: {usage})", file=sys.stderr)
                return result
            else:
                last_error = "Empty response"
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue

        except urllib.error.HTTPError as e:
            body = e.read().decode()[:500] if hasattr(e, "read") else ""
            last_error = f"HTTP {e.code}: {body}"
            # Retry on rate limit / server error
            if e.code in (429, 502, 503, 504) and attempt < retries - 1:
                wait = (attempt + 1) * 10
                print(f"  [MiniMax] {last_error}, retrying in {wait}s... ({attempt+1}/{retries})", file=sys.stderr)
                time.sleep(wait)
                continue

        except urllib.error.URLError as e:
            last_error = f"Connection error: {e}"
            if attempt < retries - 1:
                wait = (attempt + 1) * 5
                print(f"  [MiniMax] {last_error}, retrying in {wait}s... ({attempt+1}/{retries})", file=sys.stderr)
                time.sleep(wait)
                continue

    raise Exception(f"MiniMax failed after {retries} attempts: {last_error}")


def call_llm(prompt: str, system: str = "", max_tokens: int = 1024, timeout: int = 120) -> str:
    """
    Convenience wrapper: single prompt string → MiniMax call.

    Replaces the old GLM-based call_llm() in digest.py.
    Converts a simple prompt into the Anthropic messages format.

    Args:
        prompt: user prompt text (single string)
        system: optional system prompt
        max_tokens: max response tokens
        timeout: request timeout in seconds

    Returns:
        Response text string, or "" on failure
    """
    default_system = (
        "Ты — научный редактор-переводчик для российских геологов и экологов. "
        "Пиши простым, понятным языком. Избегай канцеляризмов. "
        "Отвечай только на русском языке."
    )
    try:
        return call_minimax(
            messages=[{"role": "user", "content": prompt}],
            system=system or default_system,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=3,
        )
    except Exception as e:
        print(f"  [LLM] Error: {e}", file=sys.stderr)
        return ""
