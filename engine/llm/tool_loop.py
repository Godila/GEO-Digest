"""Tool-Use Loop — multi-turn LLM↔tools orchestration engine.

This is the core "engine" that drives the Editor Agent:
  1. Sends user message + tools to LLM
  2. If LLM calls tools → executes them → sends results back
  3. If LLM responds with text → returns final answer
  4. Repeats until end_turn or max_rounds

Key classes:
  - ToolCall: A single tool invocation from LLM
  - ToolUseResult: Final result of a complete tool-use loop run
  - ToolUseLoop: The orchestrator that manages the multi-turn loop

Usage:
    loop = ToolUseLoop(llm_provider, tool_registry)
    result = loop.run(
        "Проанализируй тему Arctic permafrost emissions",
        system_prompt=EDITOR_SYSTEM_PROMPT,
    )
    print(result.content)          # Final text from LLM
    print(result.tool_calls_made)  # History of all tool calls
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A single tool call extracted from LLM response."""
    id: str
    name: str
    arguments: dict


@dataclass
class ToolUseResult:
    """Result of a complete tool-use loop execution."""
    content: str = ""
    tool_calls_made: list[dict] = field(default_factory=list)
    total_rounds: int = 0
    stop_reason: str = "unknown"
    usage: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return self.stop_reason == "end_turn"

    @property
    def hit_max_rounds(self) -> bool:
        return self.stop_reason == "max_rounds_exceeded"

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "tool_calls_made": self.tool_calls_made,
            "total_rounds": self.total_rounds,
            "stop_reason": self.stop_reason,
            "usage": self.usage,
            "warnings": self.warnings,
            "is_complete": self.is_complete,
            "hit_max_rounds": self.hit_max_rounds,
        }


# ── ToolUseLoop ───────────────────────────────────────────────────

class ToolUseLoop:
    """
    Orchestrates multi-turn tool-use dialogue between LLM and tools.

    The loop:
      1. Send messages + tool schemas to LLM via tool_complete()
      2. Parse response:
         - end_turn   → extract text, return result
         - tool_use   → execute tools, append results, continue
         - max_tokens → warn, prompt to continue
      3. Repeat until end_turn or max_rounds reached

    Safety limits:
      - MAX_ROUNDS (default 8): prevents infinite loops
      - Timeout per round: configurable via constructor
    """

    DEFAULT_MAX_ROUNDS = 8
    DEFAULT_TIMEOUT_SECONDS = 120

    def __init__(
        self,
        llm,
        tools,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        """
        Args:
            llm: LLMProvider instance with tool_complete() method
            tools: ToolRegistry instance with registered tools
            max_rounds: Maximum number of LLM rounds before forced stop
            timeout_seconds: Timeout for each individual round
        """
        self.llm = llm
        self.tools = tools
        self.max_rounds = max_rounds
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: float = 0.25,
        max_tokens: int = 4096,
        initial_messages: list | None = None,
    ) -> ToolUseResult:
        """
        Execute the tool-use loop.

        Args:
            user_message: User's message/question for the LLM
            system_prompt: System instructions for the LLM
            temperature: Generation temperature (0-1)
            max_tokens: Max tokens per LLM response
            initial_messages: Pre-existing message history (for resume)

        Returns:
            ToolUseResult with final content and full call history
        """
        # Build initial message list
        messages = list(initial_messages) if initial_messages else []
        messages.append({"role": "user", "content": user_message})

        # Get tool schemas for this session
        schemas = self.tools.get_schemas()

        # Tracking
        tool_history: list[dict] = []
        total_usage: dict[str, int] = {}
        warnings: list[str] = []

        logger.info(
            "ToolUseLoop started: max_rounds=%d, tools=%d",
            self.max_rounds, len(schemas),
        )

        for round_num in range(1, self.max_rounds + 1):
            logger.debug("ToolUseLoop round %d/%d", round_num, self.max_rounds)

            # Call LLM
            try:
                start_time = time.monotonic()
                response = self.llm.tool_complete(
                    messages=messages,
                    tools=schemas,
                    system=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                elapsed = time.monotonic() - start_time

                if elapsed > self.timeout_seconds * 0.8:
                    warnings.append(
                        f"Round {round_num}: slow response ({elapsed:.1f}s)"
                    )

            except Exception as e:
                warnings.append(f"Round {round_num}: LLM error: {e}")
                logger.error("ToolUseLoop round %d error: %s", round_num, e)
                break

            # Accumulate token usage
            if "usage" in response and isinstance(response["usage"], dict):
                for k, v in response["usage"].items():
                    try:
                        total_usage[k] = total_usage.get(k, 0) + int(v)
                    except (TypeError, ValueError):
                        pass

            # Extract content blocks
            content_blocks = response.get("content", [])
            if isinstance(content_blocks, str):
                # Some providers return raw string — wrap it
                content_blocks = [{"type": "text", "text": content_blocks}]

            # Append assistant response to history
            messages.append({
                "role": "assistant",
                "content": content_blocks,
            })

            # Determine stop reason
            stop_reason = response.get("stop_reason", "end_turn")

            # ── Branch on stop_reason ──
            if stop_reason == "end_turn":
                text = self._extract_text(content_blocks)
                logger.info(
                    "ToolUseLoop finished after %d rounds (end_turn)", round_num
                )
                return ToolUseResult(
                    content=text,
                    tool_calls_made=tool_history,
                    total_rounds=round_num,
                    stop_reason="end_turn",
                    usage=total_usage,
                    warnings=warnings,
                )

            elif stop_reason == "tool_use":
                # Extract and execute tool calls
                tool_calls = self._extract_tool_calls(content_blocks)

                if not tool_calls:
                    warnings.append(
                        f"Round {round_num}: tool_use but no tool_calls found"
                    )
                    break

                # Build tool_result blocks for the next user message
                tool_result_blocks = []

                for tc in tool_calls:
                    exec_start = time.monotonic()
                    result = self.tools.execute(tc.name, tc.arguments)
                    exec_elapsed = (time.monotonic() - exec_start) * 1000

                    # Parse result for logging
                    try:
                        result_data = (
                            result.data if hasattr(result, 'data')
                            else json.loads(str(result))
                            if isinstance(result, str)
                            else {"raw": str(result)}
                        )
                    except (json.JSONDecodeError, TypeError):
                        result_data = {"raw": str(result)}

                    tool_history.append({
                        "round": round_num,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result": result_data,
                        "success": getattr(result, 'success', True),
                        "elapsed_ms": int(exec_elapsed),
                    })

                    # Convert to Anthropic format for LLM context
                    if hasattr(result, 'to_content_block'):
                        block = result.to_content_block(tc.id)
                    else:
                        block = {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": json.dumps(
                                result_data, ensure_ascii=False, default=str
                            )[:8000],  # Truncate huge results
                        }
                    tool_result_blocks.append(block)

                # Append tool results as user message
                messages.append({
                    "role": "user",
                    "content": tool_result_blocks,
                })
                logger.debug(
                    "Round %d: executed %d tools", round_num, len(tool_calls)
                )

            elif stop_reason == "max_tokens":
                warnings.append(
                    f"Round {round_num}: hit max_tokens limit"
                )
                messages.append({
                    "role": "user",
                    "content": "Continue. "
                              "If you have enough information, provide your final answer.",
                })

            else:
                warnings.append(
                    f"Round {round_num}: unexpected stop_reason={stop_reason}"
                )
                logger.warning(
                    "Unexpected stop_reason=%s at round %d", stop_reason, round_num
                )
                break

        # ── Max rounds exceeded or early exit ──
        final_text = ""
        if messages and len(messages) >= 2:
            last_asst = messages[-1]
            if last_asst.get("role") == "assistant":
                final_text = self._extract_text(last_asst.get("content", []))

        if not any("max rounds" in w.lower() for w in warnings):
            warnings.append(
                f"Max rounds ({self.max_rounds}) reached without end_turn"
            )

        logger.warning(
            "ToolUseLoop stopped after %d rounds (max reached or error)",
            min(round_num, self.max_rounds),
        )

        return ToolUseResult(
            content=final_text,
            tool_calls_made=tool_history,
            total_rounds=min(round_num, self.max_rounds),
            stop_reason="max_rounds_exceeded",
            usage=total_usage,
            warnings=warnings,
        )

    # ── Static helpers ───────────────────────────────────────────

    @staticmethod
    def _extract_text(content_blocks: list) -> str:
        """Extract plain text from content blocks, ignoring thinking/tool_use."""
        if not content_blocks:
            return ""

        texts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)

        return "\n".join(texts).strip()

    @staticmethod
    def _extract_tool_calls(content_blocks: list) -> list[ToolCall]:
        """Extract tool_use calls from content blocks."""
        calls = []
        if not content_blocks:
            return calls

        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append(ToolCall(
                    id=block.get("id", f"call_{uuid.uuid4().hex[:6]}"),
                    name=block.get("name", ""),
                    arguments=block.get("input") if isinstance(block.get("input"), dict) else {},
                ))

        return calls
