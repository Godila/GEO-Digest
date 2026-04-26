"""Unit tests for ToolUseLoop — multi-turn LLM↔tools orchestration.

Tests cover:
  - Single-turn: LLM responds immediately (end_turn)
  - Multi-turn: LLM calls tools, gets results, then responds
  - Max rounds limit enforcement
  - Usage accumulation across rounds
  - Initial messages for resume
  - _extract_text: text extraction from content blocks
  - _extract_tool_calls: tool call parsing
  - Edge cases: empty responses, errors, max_tokens

Uses mock LLM and mock tool registry.
"""

import json
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.llm.tool_loop import ToolCall, ToolUseResult, ToolUseLoop
from engine.tools.base import ToolRegistry, ToolResult


# ── Helpers ───────────────────────────────────────────────────────

def text_block(text: str) -> dict:
    """Create a text content block."""
    return {"type": "text", "text": text}


def tool_use_block(name: str, input_data: dict, call_id: str = None) -> dict:
    """Create a tool_use content block."""
    return {
        "type": "tool_use",
        "id": call_id or f"call_{name}",
        "name": name,
        "input": input_data,
    }


def thinking_block(content: str) -> dict:
    """Create a thinking content block."""
    return {"type": "thinking", "thinking": content}


def make_llm_response(
    content_blocks: list | str,
    stop_reason: str = "end_turn",
    usage: dict | None = None,
) -> dict:
    """Build a standard LLM response dict."""
    if isinstance(content_blocks, str):
        content_blocks = [text_block(content_blocks)]
    return {
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 100, "output_tokens": 50},
    }


class MockLLM:
    """Mock LLM that returns pre-defined responses in sequence."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.calls = []

    def tool_complete(self, messages=None, tools=None, system="",
                      temperature=0.25, max_tokens=4096) -> dict:
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "system": system,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        if not self.responses:
            return make_llm_response("No more responses", stop_reason="end_turn")
        return self.responses.pop(0)


class MockToolHandler:
    """Simple mock tool that returns a JSON string."""

    def __init__(self, return_value: dict | str | None = None):
        self.return_value = return_value or {"result": "ok", "count": 42}
        self.calls = []

    def __call__(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        if isinstance(self.return_value, ToolResult):
            return self.return_value
        if isinstance(self.return_value, dict):
            return ToolResult.ok(data=self.return_value)
        return ToolResult.ok(content=str(self.return_value))


def make_mock_registry(tools: dict[str, Any] | None = None) -> ToolRegistry:
    """Create a ToolRegistry with mock handlers."""
    registry = ToolRegistry()
    tools = tools or {
        "search_articles": MockToolHandler({"total_found": 5, "returned": 5, "articles": []}),
        "get_article_detail": MockToolHandler({"found": True, "doi": "10.x/y"}),
        "validate_doi": MockToolHandler({"valid": True, "doi": "10.x/y"}),
        "cluster_by_subtopic": MockToolHandler({"total_relevant": 10, "clusters": []}),
        "find_similar_existing": MockToolHandler({"existing_count": 0}),
        "count_storage_stats": MockToolHandler({"total_articles": 100}),
        "explore_domain": MockToolHandler({"total_in_scope": 100}),
    }
    for name, handler in tools.items():
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"] if name != "count_storage_stats" else [],
        }
        registry.register(
            name=name,
            handler=handler,
            schema=schema,
            description=f"Mock {name}",
        )
    return registry


# ── Test: Single Turn ─────────────────────────────────────────────

class TestSingleTurnEndTurn(unittest.TestCase):
    """LLM responds immediately without calling tools."""

    def test_immediate_end_turn(self):
        """LLM сразу отвечает текстом."""
        llm = MockLLM([make_llm_response(
            [text_block("Вот анализ темы: найдено 15 статей.")],
            stop_reason="end_turn",
            usage={"input_tokens": 50, "output_tokens": 30},
        )])
        registry = make_mock_registry()
        loop = ToolUseLoop(llm, registry)

        result = loop.run("Проанализируй тему Arctic")

        self.assertEqual(result.total_rounds, 1)
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertIn("анализ", result.content.lower())
        self.assertEqual(len(result.tool_calls_made), 0)
        self.assertTrue(result.is_complete)

    def test_empty_content(self):
        """Пустой контент не падает."""
        llm = MockLLM([make_llm_response("", stop_reason="end_turn")])
        loop = ToolUseLoop(llm, make_mock_registry())
        result = loop.run("test")
        self.assertEqual(result.content, "")
        self.assertTrue(result.is_complete)

    def test_string_content_auto_wrapped(self):
        """Строковый контент оборачивается в блок."""
        llm = MockLLM([{"content": "plain string", "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 5}}])
        loop = ToolUseLoop(llm, make_mock_registry())
        result = loop.run("test")
        self.assertEqual(result.content, "plain string")


# ── Test: Multi-Turn with Tools ───────────────────────────────────

class TestMultiTurnWithTools(unittest.TestCase):
    """LLM calls tools, gets results, then responds."""

    def test_two_rounds_one_tool(self):
        """Один вызов tool + финальный ответ."""
        handler = MockToolHandler({"clusters": [{"theme": "methane", "count": 12}]})
        llm = MockLLM([
            # Round 1: LLM вызывает tool
            make_llm_response(
                [tool_use_block("cluster_by_subtopic", {"topic": "methane"})],
                stop_reason="tool_use",
                usage={"input_tokens": 200, "output_tokens": 30},
            ),
            # Round 2: LLM видит результат и отвечает
            make_llm_response(
                [text_block("Найдено 3 кластера по теме methane. Главный — emissions.")],
                stop_reason="end_turn",
                usage={"input_tokens": 300, "output_tokens": 80},
            ),
        ])
        registry = make_mock_registry({"cluster_by_subtopic": handler})
        loop = ToolUseLoop(llm, registry)

        result = loop.run("Проанализируй methane статьи")

        self.assertEqual(result.total_rounds, 2)
        self.assertEqual(len(result.tool_calls_made), 1)
        self.assertEqual(result.tool_calls_made[0]["name"], "cluster_by_subtopic")
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertTrue(result.is_complete)
        self.assertIn("кластер", result.content.lower())

    def test_three_rounds_two_tools(self):
        """Два последовательных вызова tools."""
        stats_handler = MockToolHandler({"total_articles": 150})
        cluster_handler = MockToolHandler({"total_relevant": 20})

        llm = MockLLM([
            make_llm_response(
                [tool_use_block("count_storage_stats", {})],
                stop_reason="tool_use",
            ),
            make_llm_response(
                [tool_use_block("cluster_by_subtopic", {"topic": "climate"})],
                stop_reason="tool_use",
            ),
            make_llm_response(
                [text_block("Анализ завершён. 150 статей, 20 релевантных.")],
                stop_reason="end_turn",
            ),
        ])
        registry = make_mock_registry({
            "count_storage_stats": stats_handler,
            "cluster_by_subtopic": cluster_handler,
        })
        loop = ToolUseLoop(llm, registry)

        result = loop.run("Дай обзор")

        self.assertEqual(result.total_rounds, 3)
        self.assertEqual(len(result.tool_calls_made), 2)
        self.assertEqual(result.tool_calls_made[0]["name"], "count_storage_stats")
        self.assertEqual(result.tool_calls_made[1]["name"], "cluster_by_subtopic")

    def test_multiple_tools_in_one_round(self):
        """Несколько tools за один round."""
        llm = MockLLM([
            make_llm_response(
                [
                    tool_use_block("count_storage_stats", {}, "call_1"),
                    tool_use_block("validate_doi", {"doi": "10.x/y"}, "call_2"),
                ],
                stop_reason="tool_use",
            ),
            make_llm_response(
                [text_block("Всё проверено.")],
                stop_reason="end_turn",
            ),
        ])
        loop = ToolUseLoop(llm, make_mock_registry())

        result = loop.run("Проверь данные")

        self.assertEqual(result.total_rounds, 2)
        self.assertEqual(len(result.tool_calls_made), 2)


# ── Test: Max Rounds Limit ────────────────────────────────────────

class TestMaxRoundsLimit(unittest.TestCase):
    """Enforcement of maximum rounds limit."""

    def test_stops_at_max_rounds(self):
        """Остановка после достижения MAX_ROUNDS."""
        llm = MockLLM([
            make_llm_response(
                [tool_use_block("search_articles", {"query": "test"})],
                stop_reason="tool_use",
            )
            for _ in range(10)  # Больше чем max_rounds
        ])
        loop = ToolUseLoop(llm, make_mock_registry(), max_rounds=3)

        result = loop.run("Бесконечный поиск")

        self.assertEqual(result.total_rounds, 3)
        self.assertTrue(result.hit_max_rounds)
        self.assertTrue(any("max rounds" in w.lower() for w in result.warnings))

    def test_default_max_rounds_is_8(self):
        """По умолчанию MAX_ROUNDS=8."""
        self.assertEqual(ToolUseLoop.DEFAULT_MAX_ROUNDS, 8)


# ── Test: Usage Accumulation ──────────────────────────────────────

class TestUsageAccumulation(unittest.TestCase):
    """Token usage is accumulated across rounds."""

    def test_usage_sums_across_rounds(self):
        """Usage суммируется."""
        llm = MockLLM([
            make_llm_response(
                [tool_use_block("x", {})],
                stop_reason="tool_use",
                usage={"input_tokens": 100, "output_tokens": 20},
            ),
            make_llm_response(
                [text_block("done")],
                stop_reason="end_turn",
                usage={"input_tokens": 200, "output_tokens": 50},
            ),
        ])
        loop = ToolUseLoop(llm, make_mock_registry())

        result = loop.run("test")

        self.assertEqual(result.usage["input_tokens"], 300)   # 100+200
        self.assertEqual(result.usage["output_tokens"], 70)    # 20+50


# ── Test: Initial Messages for Resume ─────────────────────────────

class TestInitialMessagesResume(unittest.TestCase):
    """Support for resuming with existing message history."""

    def test_resume_with_history(self):
        """Можно продолжить с существующей историей."""
        prev_messages = [
            {"role": "user", "content": "Первый вопрос"},
            {"role": "assistant", "content": [text_block("Ответ")]},
        ]
        llm = MockLLM([make_llm_response(
            [text_block("Продолжение ответа")],
            stop_reason="end_turn",
        )])
        loop = ToolUseLoop(llm, make_mock_registry())

        result = loop.run("Продолжи?", initial_messages=prev_messages)

        self.assertIn("Продолжение", result.content)
        # Check that LLM received the history + new message
        sent_messages = llm.calls[0]["messages"]
        # Note: Python list reference — after LLM call, loop appends assistant response
        # to the same list, so we see 4 entries. First 3 are what was sent to LLM.
        self.assertGreaterEqual(len(sent_messages), 3)
        # The new user message should be at index 2 (after 2 prev messages)
        self.assertEqual(sent_messages[2]["content"], "Продолжи?")


# ── Test: Error Handling ──────────────────────────────────────────

class TestErrorHandling(unittest.TestCase):
    """Graceful handling of errors."""

    def test_llm_error_breaks_loop(self):
        """Ошибка LLM прерывает цикл с предупреждением."""
        class BrokenLLM:
            def tool_complete(self, **kwargs):
                raise ConnectionError("API unreachable")

        loop = ToolUseLoop(BrokenLLM(), make_mock_registry(), max_rounds=5)
        result = loop.run("test")

        self.assertGreater(len(result.warnings), 0)
        self.assertIn("error", result.warnings[0].lower())

    def test_unknown_tool_returns_error(self):
        """Неизвестный tool → ошибка в истории но цикл продолжается."""
        llm = MockLLM([
            make_llm_response(
                [tool_use_block("nonexistent_tool", {"q": "x"})],
                stop_reason="tool_use",
            ),
            make_llm_response(
                [text_block("Понял, tool недоступен.")],
                stop_reason="end_turn",
            ),
        ])
        loop = ToolUseLoop(llm, make_mock_registry())

        result = loop.run("test")

        # Should have recorded the failed tool call
        self.assertEqual(len(result.tool_calls_made), 1)
        self.assertFalse(result.tool_calls_made[0].get("success", True))

    def test_max_tokens_continues(self):
        """max_tokens → предупреждение + попытка продолжить."""
        llm = MockLLM([
            make_llm_response(
                [text_block("Начало длинного...")],
                stop_reason="max_tokens",
            ),
            make_llm_response(
                [text_block("...ответа")],
                stop_reason="end_turn",
            ),
        ])
        loop = ToolUseLoop(llm, make_mock_registry())

        result = loop.run("test")

        self.assertEqual(result.total_rounds, 2)
        self.assertTrue(any("max_tokens" in w.lower() for w in result.warnings))

    def test_unexpected_stop_reason(self):
        """Неизвестный stop_reason → остановка."""
        llm = MockLLM([make_llm_response(
            [text_block("response")],
            stop_reason="something_weird",
        )])
        loop = ToolUseLoop(llm, make_mock_registry())

        result = loop.run("test")

        self.assertTrue(any("unexpected" in w.lower() for w in result.warnings))


# ── Test: _extract_text ───────────────────────────────────────────

class TestExtractText(unittest.TestCase):
    """Text extraction from content blocks."""

    def test_only_text_blocks(self):
        blocks = [text_block("hello"), text_block("world")]
        self.assertEqual(ToolUseLoop._extract_text(blocks), "hello\nworld")

    def test_ignores_tool_use_and_thinking(self):
        blocks = [
            thinking_block("hmm..."),
            tool_use_block("search", {"q": "x"}),
            text_block("real answer"),
        ]
        self.assertEqual(ToolUseLoop._extract_text(blocks), "real answer")

    def test_empty_list(self):
        self.assertEqual(ToolUseLoop._extract_text([]), "")

    def test_string_blocks_preserved(self):
        blocks = ["raw string", text_block("typed")]
        self.assertEqual(ToolUseLoop._extract_text(blocks), "raw string\ntyped")


# ── Test: _extract_tool_calls ─────────────────────────────────────

class TestExtractToolCalls(unittest.TestCase):
    """Tool call extraction from content blocks."""

    def test_single_call(self):
        blocks = [tool_use_block("search", {"q": "permafrost"}, "c1")]
        calls = ToolUseLoop._extract_tool_calls(blocks)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "search")
        self.assertEqual(calls[0].arguments, {"q": "permafrost"})
        self.assertEqual(calls[0].id, "c1")

    def test_mixed_content(self):
        blocks = [
            thinking_block("..."),
            tool_use_block("validate", {"doi": "10.x"}, "c2"),
            text_block("checking"),
        ]
        calls = ToolUseLoop._extract_tool_calls(blocks)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "validate")

    def test_no_calls(self):
        blocks = [text_block("just text"), thinking_block("thoughts")]
        calls = ToolUseLoop._extract_tool_calls(blocks)
        self.assertEqual(len(calls), 0)

    def test_empty_input(self):
        calls = ToolUseLoop._extract_tool_calls([])
        self.assertEqual(calls, [])

    def test_generates_id_if_missing(self):
        blocks = [tool_use_block("x", {}, None)]
        calls = ToolUseLoop._extract_tool_calls(blocks)
        self.assertTrue(calls[0].id.startswith("call_"))


# ── Test: ToolUseResult serialization ─────────────────────────────

class TestToolUseResultSerialization(unittest.TestCase):
    """ToolUseResult.to_dict() works correctly."""

    def test_complete_result(self):
        r = ToolUseResult(
            content="answer",
            total_rounds=2,
            stop_reason="end_turn",
            usage={"in": 100, "out": 50},
        )
        d = r.to_dict()
        self.assertTrue(d["is_complete"])
        self.assertFalse(d["hit_max_rounds"])
        self.assertEqual(d["stop_reason"], "end_turn")

    def test_max_rounds_result(self):
        r = ToolUseResult(stop_reason="max_rounds_exceeded")
        d = r.to_dict()
        self.assertFalse(d["is_complete"])
        self.assertTrue(d["hit_max_rounds"])

    def test_warnings_included(self):
        r = ToolUseResult(warnings=["warning 1", "warning 2"])
        d = r.to_dict()
        self.assertEqual(len(d["warnings"]), 2)


if __name__ == "__main__":
    unittest.main()
