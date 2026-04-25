"""
OpenAI-compatible LLM provider.

Works with any API that follows the OpenAI chat completions format:
  - OpenAI (GPT-4, GPT-4o, etc.)
  - Anthropic Claude (via proxy)
  - DeepSeek
  - Together AI
  - Groq
  - Ollama (local)
  - Any OpenAI-compatible endpoint

Used primarily for the Reviewer agent (different model from writer).
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


class OpenAICompatProvider(LLMProvider):
    """
    OpenAI-compatible chat completions provider.
    
    Endpoint: {base_url}/chat/completions
    Auth: Bearer token in Authorization header
    """
    
    def __init__(self, **overrides):
        # Default: use reviewer config (this is primary use case)
        cfg = get_config()
        
        self._api_key = overrides.get("api_key") or cfg.get_api_key(
            overrides.get("api_key_env", cfg.reviewer.api_key_env)
        )
        self._base_url = (
            overrides.get("base_url")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self._model = overrides.get("model") or cfg.reviewer.model or "gpt-4o"
        self._max_tokens_default = 8192
        self._timeout = overrides.get("timeout", 180)
        self._retries = overrides.get("retries", 3)
    
    @property
    def name(self) -> str:
        return f"OpenAI-Compat/{self._model}"
    
    @property
    def model(self) -> str:
        return self._model
    
    def _get_headers(self) -> dict:
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
    
    def _build_payload(
        self,
        messages: list[dict],
        system: str = "",
        max_tokens: int = 0,
        temperature: float = 0.3,
    ) -> dict:
        effective_max = max(max_tokens or self._max_tokens_default, 256)
        
        payload = {
            "model": self._model,
            "max_tokens": effective_max,
            "temperature": temperature,
            "messages": messages,
        }
        
        if system:
            # Prepend as system message
            payload["messages"] = [
                {"role": "system", "content": system},
                *payload["messages"],
            ]
        
        return payload
    
    def complete(
        self,
        prompt: str = "",
        messages: list[dict] | None = None,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        **kwargs,
    ) -> str:
        """Call OpenAI-compatible endpoint."""
        
        msgs = messages or []
        if prompt:
            if msgs and msgs[-1].get("role") == "user":
                msgs[-1]["content"] += "\n" + prompt
            else:
                msgs.append({"role": "user", "content": prompt})
        
        url = f"{self._base_url}/chat/completions"
        payload = self._build_payload(msgs, system, max_tokens, temperature)
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=self._get_headers(),
        )
        
        last_error = None
        for attempt in range(self._retries):
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    data = json.loads(resp.read())
                
                # Standard OpenAI format
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    usage = data.get("usage", {})
                    print(f"  [{self.name}] OK ({len(text)} chars, tokens: {usage})", file=sys.stderr)
                    return text
                
                last_error = f"No choices in response: {json.dumps(data)[:200]}"
                
            except urllib.error.HTTPError as e:
                body = e.read().decode()[:500] if hasattr(e, "read") else ""
                last_error = f"HTTP {e.code}: {body}"
                if e.code in (429, 502, 503, 504) and attempt < self._retries - 1:
                    wait = (attempt + 1) * 10
                    print(f"  [{self.name}] {last_error}, retrying in {wait}s... ({attempt+1}/{self._retries})", file=sys.stderr)
                    time.sleep(wait)
                    continue
                    
            except urllib.error.URLError as e:
                last_error = f"Connection error: {e}"
                if attempt < self._retries - 1:
                    wait = (attempt + 1) * 5
                    print(f"  [{self.name}] {last_error}, retrying in {wait}s...", file=sys.stderr)
                    time.sleep(wait)
                    continue
        
        raise Exception(f"{self.name} failed after {self._retries} attempts: {last_error}")
    
    def complete_json(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 4096,
        **kwargs,
    ) -> dict:
        """Call with JSON output expectation."""
        json_system = (
            system + "\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. No markdown fences, no explanation. "
            "Raw JSON only that can be parsed with json.loads()."
        )
        
        for attempt in range(3):
            try:
                text = self.complete(prompt=prompt, system=json_system, max_tokens=max_tokens, **kwargs)
                text = text.strip()
                if text.startswith("```"):
                    lines = text.split("\n")
                    text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
                return json.loads(text)
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < 2:
                    continue
                raise ValueError(f"JSON parse failed after 3 attempts: {text[:500]}") from e
