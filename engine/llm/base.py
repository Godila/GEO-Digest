
"""LLM Provider — abstract interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract LLM provider. All providers must implement complete()."""

    def __init__(self, model: str = "", **kwargs):
        self.model = model
        self._kwargs = kwargs

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Send prompt to LLM, return response text."""
        ...

    def complete_json(self, prompt: str, system: str = "", **kwargs) -> str:
        """Complete with JSON output hint."""
        system = (system + "\n\nRespond ONLY with valid JSON. No markdown fences.").strip()
        return self.complete(prompt, system=system, **kwargs)

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the LLM endpoint is reachable."""
        ...

    # ── Tool-Use Interface ──────────────────────────────────────

    def tool_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        temperature: float = 0.25,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Tool-use completion (function calling).

        Sends messages + tool definitions to LLM and returns structured response.

        Args:
            messages: Conversation history in Anthropic format.
                Each message: {"role": "user"|"assistant", "content": ...}
                Assistant content may contain tool_use blocks.
                User content may contain tool_result blocks.
            tools: List of tool definitions in Anthropic format:
                [{"name": "...", "description": "...", "input_schema": {...}}]
            system: System prompt string.
            temperature: Sampling temperature (default 0.25 for deterministic tool use).
            max_tokens: Max tokens in response.

        Returns:
            dict with keys:
                "content": list[dict] — content blocks (text + tool_use)
                "stop_reason": str — "end_turn" | "tool_use" | "max_tokens"
                "usage": dict — {"input_tokens": int, "output_tokens": int}
                "model": str — model name used
                "error": str (optional) — on failure

        Raises:
            NotImplementedError: If subclass does not override this method.
            RuntimeError: On API errors after retries exhausted.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tool use. "
            f"Override tool_complete() to enable function calling."
        )
