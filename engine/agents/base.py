"""Base Agent -- Abstract class all agents inherit from.

Every agent (Scout, Reader, Writer, Reviewer) follows this pattern:
  1. __init__(config, llm, storage)
  2. run(**kwargs) -> AgentResult
  3. _build_prompt(**kwargs) -> str
  4. _parse_response(text: str) -> Typed result
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from engine.schemas import AgentResult
from engine.llm.base import LLMProvider
from engine.storage.base import StorageBackend


class BaseAgent(ABC):
    """
    Abstract base for all pipeline agents.

    Lifecycle:
        agent = ScoutAgent(config, llm, storage)
        result = agent.run(topic="ML geology")
        # result.success = True/False
        # result.data = ScoutResult (typed)

    Subclass must implement:
        - name: str property
        - run(**kwargs) -> AgentResult
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        storage: StorageBackend | None = None,
        **kwargs,
    ):
        from engine.config import get_config

        self._cfg = get_config()
        self._llm = llm
        self._storage = storage

    @property
    def llm(self) -> LLMProvider:
        if self._llm is None:
            from engine.llm import get_llm
            self._llm = get_llm()
        return self._llm

    @property
    def storage(self) -> StorageBackend:
        if self._storage is None:
            from engine.storage import get_storage
            self._storage = get_storage()
        return self._storage

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent identifier used in logs and results."""
        ...

    @abstractmethod
    def run(self, **kwargs) -> AgentResult:
        """
        Execute the agent's main logic.

        Args:
            **kwargs: Agent-specific parameters.
                Scout: topic, max_articles, ...
                Reader: group_or_dois, full_text, ...
                Writer: draft, style, language, user_comment, ...
                Reviewer: article_text, source_articles, strictness, ...

        Returns:
            AgentResult with typed .data field
        """
        ...

    def _generate_id(self) -> str:
        return f"{self.name}_{uuid.uuid4().hex[:8]}"

    def _timed_run(self, fn, *args, **kwargs) -> tuple[Any, float]:
        """Run function with timing."""
        start = time.time()
        try:
            result = fn(*args, **kwargs)
            return result, time.time() - start
        except Exception as e:
            raise e

    def _log(self, msg: str):
        print(f"  [{self.name}] {msg}", file=__import__("sys").stderr)


# ── Convenience Mixins ─────────────────────────────────────────


class LLMCallMixin:
    """Mixin for agents that primarily call LLM with structured prompts."""

    def call_llm(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 0,
        temperature: float = 0.3,
        parse_json: bool = False,
    ) -> Any:
        """
        Call LLM and optionally parse JSON response.

        Returns:
            str if parse_json=False
            dict/list if parse_json=True
        """
        if parse_json:
            return self.llm.complete_json(prompt, system=system, max_tokens=max_tokens or 4096)
        return self.llm.complete(prompt, system=system, max_tokens=max_tokens or 4096, temperature=temperature)
