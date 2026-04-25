"""
LLM Providers — Abstract interface + implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


# ── Abstract Base ───────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Abstract interface for LLM providers.
    
    All agents use this interface — they don't know which
    specific LLM is behind it (MiniMax, OpenAI, Claude, local).
    """
    
    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        **kwargs,
    ) -> str:
        """
        Send a prompt, get response text.
        
        Args:
            prompt: User message (single string)
            system: System / instruction prompt
            max_tokens: Max tokens in response
            temperature: Sampling temperature (0=deterministic, 1=creative)
            **kwargs: Provider-specific options
        
        Returns:
            Response text string
        
        Raises:
            Exception on API error (after retries)
        """
        ...
    
    @abstractmethod
    def complete_messages(
        self,
        messages: list[dict[str, str]],
        *,
        system: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        **kwargs,
    ) -> str:
        """
        Send messages list (chat format), get response.
        
        Args:
            messages: [{"role": "user"|"assistant", "content": "..."}]
            system: System prompt
            max_tokens, temperature: same as complete()
        
        Returns:
            Response text string
        """
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider name: minimax, openai, anthropic, etc."""
        ...
    
    # Convenience wrappers
    
    def complete_json(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 4096,
        **kwargs,
    ) -> dict | list:
        """Complete and parse JSON from response."""
        import json
        text = self.complete(
            prompt + "\n\nRespond with valid JSON only.",
            system=system,
            max_tokens=max_tokens,
            **kwargs,
        )
        # Try to extract JSON from markdown code blocks or raw text
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() in ("```", "```json") else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Could not parse JSON from LLM response (first 200 chars): {text[:200]}")
