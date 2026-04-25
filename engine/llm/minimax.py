"""
MiniMax M2.7 LLM provider.

Wraps the existing scripts/llm_client.py for use through the
LLMProvider interface. Keeps backward compatibility.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

from engine.config import get_config
from engine.llm.base import LLMProvider


class MiniMaxProvider(LLMProvider):
    """
    MiniMax M2.7 via Anthropic-compatible API.
    
    Uses the same endpoint and auth as existing llm_client.py,
    but exposes it through the standard LLMProvider interface.
    """
    
    def __init__(self, **overrides):
        cfg = get_config()
        self._api_key = overrides.get("api_key") or cfg.get_api_key(cfg.llm.api_key_env)
        self._base_url = overrides.get("base_url") or cfg.llm.base_url
        self._model = overrides.get("model") or cfg.llm.model
        self._max_tokens_default = cfg.llm.max_tokens
        self._timeout = cfg.llm.timeout
        self._retries = cfg.llm.retries
        self._disable_thinking = cfg.llm.disable_thinking
    
    @property
    def name(self) -> str:
        return "MiniMax-M2.7"
    
    @property
    def model(self) -> str:
        return self._model
    
    def _get_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
    
    def _build_payload(
        self,
        messages: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 0,
        temperature: float = 0.3,
    ) -> bytes:
        effective_max = max(max_tokens or self._max_tokens_default + 512, 2048)
        
        payload = {
            "model": self._model,
            "max_tokens": effective_max,
            "messages": messages or [{"role": "user", "content": ""}],
            "disable_thinking": self._disable_thinking,
        }
        if system:
            payload["system"] = system
        
        return json.dumps(payload).encode()
    
    def complete(
        self,
        prompt: str = "",
        messages: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        **kwargs,
    ) -> str:
        """Call MiniMax M2.7, return response text."""
        
        if not self._api_key:
            raise ValueError("MINIMAX_API_KEY not set")
        
        # Build message from prompt if no messages provided
        msgs = messages
        if not msgs and prompt:
            msgs = [{"role": "user", "content": prompt}]
        elif prompt and msgs:
            # Append prompt to last user message or add new
            if msgs[-1].get("role") == "user":
                msgs[-1]["content"] += "\n" + prompt
            else:
                msgs.append({"role": "user", "content": prompt})
        
        url = f"{self._base_url.rstrip('/')}/v1/messages"
        payload = self._build_payload(msgs, system, max_tokens, temperature)
        
        req = urllib.request.Request(url, data=payload, headers=self._get_headers())
        
        last_error = None
        for attempt in range(self._retries):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read())
                
                # Extract text from content blocks
                text_parts = []
                for block in data.get("content", []):
                    btype = block.get("type", "")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                
                result = "\n".join(text_parts)
                if result:
                    usage = data.get("usage", {})
                    print(f"  [MiniMax] OK ({len(result)} chars, tokens: {usage})", file=sys.stderr)
                    return result
                else:
                    last_error = "Empty response"
                    
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:500] if hasattr(e, "read") else ""
                last_error = f"HTTP {e.code}: {body}"
                if e.code in (429, 502, 503, 504) and attempt < self._retries - 1:
                    wait = (attempt + 1) * 10
                    print(f"  [MiniMax] {last_error}, retrying in {wait}s... ({attempt+1}/{self._retries})", file=sys.stderr)
                    time.sleep(wait)
                    continue
                    
            except urllib.error.URLError as e:
                last_error = f"Connection error: {e}"
                if attempt < self._retries - 1:
                    wait = (attempt + 1) * 5
                    print(f"  [MiniMax] {last_error}, retrying in {wait}s... ({attempt+1}/{self._retries})", file=sys.stderr)
                    time.sleep(wait)
                    continue
        
        raise Exception(f"MiniMax failed after {self._retries} attempts: {last_error}")
    
    def complete_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        **kwargs,
    ) -> dict:
        """Call MiniMax, parse JSON response with retry."""
        # Wrap prompt to encourage JSON output
        json_system = (
            system + "\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation, "
            "no code blocks. Just raw JSON that can be parsed with json.loads()."
        )
        
        for attempt in range(3):
            try:
                text = self.complete(
                    prompt=prompt,
                    system=json_system,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                # Strip markdown code fences if present
                text = text.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                
                return json.loads(text)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < 2:
                    print(f"  [MiniMax] JSON parse error ({e}), retrying...", file=sys.stderr)
                    continue
                raise ValueError(
                    f"Failed to parse JSON after 3 attempts. Last text: {text[:500]}"
                ) from e
