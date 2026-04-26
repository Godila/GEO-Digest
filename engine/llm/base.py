
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
