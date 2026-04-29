
"""MiniMax LLM Provider — wraps existing llm_client.py logic."""
from __future__ import annotations
import json, time, urllib.error, urllib.request
from engine.llm.base import LLMProvider


class MiniMaxProvider(LLMProvider):
    DEFAULT_BASE_URL = "https://api.minimax.chat/anthropic"
    DEFAULT_MODEL = "MiniMax-M2.7"

    def __init__(self, api_key: str = "", base_url: str = "", model: str = "",
                 disable_thinking=True, timeout=180, retries=3, **kwargs):
        super().__init__(model or self.DEFAULT_MODEL, **kwargs)
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.disable_thinking = disable_thinking
        self.timeout = timeout
        self.retries = retries

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def _build_payload(self, prompt, system="", temp=0.3, max_tok=4096):
        messages = []
        if system:
            messages.append({"role": "user", "content": system})
            messages.append({"role": "assistant", "content": "Понятно."})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": temp,
        }
        if self.disable_thinking:
            payload["stop_sequences"] = ["</thinking>"]
        return payload

    def complete(self, prompt, system="", temperature=0.3, max_tokens=4096):
        url = f"{self.base_url}/v1/messages"
        payload = self._build_payload(prompt, system, temperature, max_tokens)
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode())
                content_blocks = result.get("content", [])
                text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
                raw = "\n".join(text_parts).strip()
                if raw.startswith("<thinking>"):
                    idx = raw.find("</thinking>")
                    raw = raw[idx + len("</thinking>"):].strip() if idx >= 0 else ""
                return raw
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"MiniMax API error after {self.retries+1} attempts: {last_err}")

    def health_check(self):
        try:
            resp = self.complete("ping", max_tokens=5)
            return True
        except Exception:
            return False

    # ── Tool-Use Completion ─────────────────────────────────────

    def tool_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        temperature: float = 0.25,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Tool-use completion via Anthropic Messages API.

        Sends messages with optional tool definitions to MiniMax.
        Supports multi-turn conversations with tool_use / tool_result blocks.

        Args:
            messages: Conversation history in Anthropic format.
                User messages may contain tool_result content blocks.
                Assistant messages may contain tool_use content blocks.
            tools: List of tool definitions for function calling.
            system: System prompt string.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            dict with keys: content, stop_reason, usage, model
        """
        url = f"{self.base_url}/v1/messages"
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if system:
            payload["system"] = system
        if self.disable_thinking:
            payload["stop_sequences"] = ["</thinking>"]

        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode())
                return self._parse_tool_response(result)
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"MiniMax API error after {self.retries + 1} attempts: {last_err}")

    def _parse_tool_response(self, result: dict) -> dict:
        """
        Parse raw Anthropic API response into standardised dict.

        Handles both text-only responses and tool_use responses.
        Strips <thinking> tags from text content when present.
        """
        content_blocks = result.get("content", [])
        usage = result.get("usage", {})
        stop_reason = result.get("stop_reason", "end_turn")

        # Strip thinking tags from text blocks
        for block in content_blocks:
            if block.get("type") == "text" and "text" in block:
                raw = block["text"]
                if raw.startswith("<thinking>"):
                    idx = raw.find("</thinking>")
                    block["text"] = raw[idx + len("</thinking>"):].strip() if idx >= 0 else ""

        return {
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
            "model": result.get("model", self.model),
        }
