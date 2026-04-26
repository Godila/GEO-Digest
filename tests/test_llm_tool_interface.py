"""Unit tests for LLM tool_complete() interface.

Tests cover:
  - Base class raises NotImplementedError
  - MiniMax provider: tool_use response parsing, end_turn parsing
  - MiniMax provider: _parse_tool_response with thinking tags
  - OpenAI compat: message format conversion (Anthropic → OpenAI)
  - OpenAI compat: tool format conversion
  - OpenAI compat: response parsing back to standard format

All tests use mocking — no real API calls.
"""

import json
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.llm.base import LLMProvider
from engine.llm.minimax import MiniMaxProvider
from engine.llm.openai_compat import OpenAICompatProvider


class TestBaseToolCompleteInterface(unittest.TestCase):
    """Test that the abstract base class correctly rejects tool_complete calls."""

    def _make_concrete_provider(self):
        """Create a concrete LLMProvider subclass for testing."""
        class ConcreteProvider(LLMProvider):
            def complete(self, prompt, **kw): return ""
            def health_check(self): return True
        return ConcreteProvider(model="test-model")

    def test_base_raises_not_implemented(self):
        """Базовый класс должен поднимать NotImplementedError."""
        provider = self._make_concrete_provider()
        with self.assertRaises(NotImplementedError) as ctx:
            provider.tool_complete([], [])
        self.assertIn("does not support tool use", str(ctx.exception))

    def test_base_error_message_contains_class_name(self):
        """Сообщение об ошибке должно содержать имя класса."""
        provider = self._make_concrete_provider()
        try:
            provider.tool_complete([], [])
        except NotImplementedError as e:
            self.assertIn("ConcreteProvider", str(e))

    def test_base_accepts_optional_params(self):
        """tool_complete должен принимать все параметры без ошибки до вызова."""
        provider = self._make_concrete_provider()
        # Should not raise on construction of arguments
        try:
            provider.tool_complete(
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"name": "test", "description": "t", "input_schema": {"type": "object"}}],
                system="You are helpful.",
                temperature=0.5,
                max_tokens=1024,
            )
        except NotImplementedError:
            pass  # Expected — we only check it accepts params


class TestMiniMaxToolCompleteParsing(unittest.TestCase):
    """Test MiniMax tool_complete response parsing (no real API calls)."""

    def setUp(self):
        self.provider = MiniMaxProvider(
            api_key="test-key",
            model="MiniMax-M2.5",
            retries=1,
        )

    def test_parse_tool_use_response(self):
        """Корректный парсинг ответа с tool_use блоком."""
        raw_response = {
            "id": "msg_001",
            "type": "message",
            "role": "assistant",
            "model": "MiniMax-M2.5",
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "Let me search for articles."},
                {
                    "type": "tool_use",
                    "id": "toolu_01A2B3C4D5E6",
                    "name": "search_articles",
                    "input": {"query": "permafrost carbon", "limit": 10},
                },
            ],
            "usage": {"input_tokens": 245, "output_tokens": 87},
        }
        result = self.provider._parse_tool_response(raw_response)

        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertEqual(len(result["content"]), 2)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertEqual(result["content"][1]["type"], "tool_use")
        self.assertEqual(result["content"][1]["name"], "search_articles")
        self.assertEqual(result["usage"]["input_tokens"], 245)
        self.assertEqual(result["usage"]["output_tokens"], 87)
        self.assertEqual(result["model"], "MiniMax-M2.5")

    def test_parse_end_turn_response(self):
        """Корректный парсинг end_turn ответа (только текст)."""
        raw_response = {
            "id": "msg_002",
            "type": "message",
            "role": "assistant",
            "model": "MiniMax-M2.7",
            "stop_reason": "end_turn",
            "content": [
                {"type": "text", "text": "Based on my analysis, there are 54 relevant articles."}
            ],
            "usage": {"input_tokens": 512, "output_tokens": 156},
        }
        result = self.provider._parse_tool_response(raw_response)

        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertIn("54 relevant articles", result["content"][0]["text"])

    def test_parse_strips_thinking_tags(self):
        """Удаление <thinking> тегов из текстового контента."""
        raw_response = {
            "id": "msg_003",
            "stop_reason": "end_turn",
            "content": [
                {
                    "type": "text",
                    "text": "<thinking>I need to analyze this carefully.\nStep 1: search\nStep 2: count</thinking>\nBased on my analysis, I found 42 articles about permafrost."
                }
            ],
            "usage": {"input_tokens": 300, "output_tokens": 95},
        }
        result = self.provider._parse_tool_response(raw_response)

        text = result["content"][0]["text"]
        self.assertNotIn("<thinking>", text)
        self.assertIn("42 articles about permafrost", text)

    def test_parse_empty_content(self):
        """Парсинг пустого content не падает."""
        raw_response = {
            "stop_reason": "end_turn",
            "content": [],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = self.provider._parse_tool_response(raw_response)
        self.assertEqual(result["content"], [])
        self.assertEqual(result["stop_reason"], "end_turn")

    def test_parse_default_values_for_missing_fields(self):
        """Значения по умолчанию для отсутствующих полей."""
        raw_response = {}  # Completely empty
        result = self.provider._parse_tool_response(raw_response)
        self.assertEqual(result["content"], [])
        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(result["usage"]["input_tokens"], 0)
        self.assertEqual(result["model"], "MiniMax-M2.5")

    def test_parse_multiple_tool_use_blocks(self):
        """Парсинг нескольких tool_use блоков в одном ответе."""
        raw_response = {
            "stop_reason": "tool_use",
            "content": [
                {"type": "text", "text": "I'll search and then get details."},
                {"type": "tool_use", "id": "tu_001", "name": "search_articles", "input": {"query": "arctic"}},
                {"type": "tool_use", "id": "tu_002", "name": "count_storage_stats", "input": {}},
            ],
            "usage": {"input_tokens": 200, "output_tokens": 120},
        }
        result = self.provider._parse_tool_response(raw_response)

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 2)
        self.assertEqual(tool_blocks[0]["name"], "search_articles")
        self.assertEqual(tool_blocks[1]["name"], "count_storage_stats")


class TestMiniMaxToolCompleteWithMockHTTP(unittest.TestCase):
    """Test MiniMax tool_complete with mocked HTTP calls."""

    def setUp(self):
        self.provider = MiniMaxProvider(
            api_key="test-key-12345",
            model="MiniMax-M2.5",
            timeout=10,
            retries=1,
        )

    @patch("engine.llm.minimax.urllib.request.urlopen")
    def test_tool_complete_sends_tools_in_payload(self, mock_urlopen):
        """tool_complete должен отправлять tools в payload API."""
        # Setup mock response
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Done."}],
            "usage": {"input_tokens": 100, "output_tokens": 20},
            "model": "MiniMax-M2.5",
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        tools = [{
            "name": "get_count",
            "description": "Get article count",
            "input_schema": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        }]
        messages = [{"role": "user", "content": "How many Arctic articles?"}]

        result = self.provider.tool_complete(messages, tools)

        self.assertEqual(result["stop_reason"], "end_turn")
        # Verify request was made
        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        body = json.loads(request_obj.data)
        self.assertIn("tools", body)
        self.assertEqual(len(body["tools"]), 1)
        self.assertEqual(body["tools"][0]["name"], "get_count")

    @patch("engine.llm.minimax.urllib.request.urlopen")
    def test_tool_complete_retry_on_failure(self, mock_urlopen):
        """Retry при ошибке работает корректно."""
        provider = MiniMaxProvider(
            api_key="test",
            model="MiniMax-M2.5",
            timeout=1,
            retries=2,
        )

        # First two calls fail, third succeeds
        mock_success = MagicMock()
        mock_success.read.return_value = json.dumps({
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "Success after retry"}],
            "usage": {"input_tokens": 50, "output_tokens": 10},
            "model": "MiniMax-M2.5",
        }).encode()
        mock_success.__enter__ = MagicMock(return_value=mock_success)
        mock_success.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [
            Exception("Timeout"),
            Exception("Connection error"),
            mock_success,
        ]

        result = provider.tool_complete([{"role": "user", "content": "test"}])
        self.assertEqual(result["content"][0]["text"], "Success after retry")
        self.assertEqual(mock_urlopen.call_count, 3)


class TestOpenAICompatFormatConversion(unittest.TestCase):
    """Test OpenAI-compatible provider format conversions."""

    def test_convert_simple_text_message(self):
        """Конвертация простого текстового сообщения."""
        oai = OpenAICompatProvider._convert_messages([
            {"role": "user", "content": "Hello"}
        ])
        self.assertEqual(len(oai), 1)
        self.assertEqual(oai[0]["role"], "user")
        self.assertEqual(oai[0]["content"], "Hello")

    def test_convert_with_system_prompt(self):
        """System prompt добавляется как первое сообщение."""
        oai = OpenAICompatProvider._convert_messages(
            [{"role": "user", "content": "Hi"}],
            system="You are a helpful assistant.",
        )
        self.assertEqual(oai[0]["role"], "system")
        self.assertEqual(oai[0]["content"], "You are a helpful assistant.")
        self.assertEqual(oai[1]["role"], "user")

    def test_convert_tool_use_to_openai_format(self):
        """Конвертация tool_use блока в OpenAI tool_calls формат."""
        oai = OpenAICompatProvider._convert_messages([{
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me search."},
                {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "search_articles",
                    "input": {"query": "permafrost"},
                },
            ],
        }])
        self.assertEqual(len(oai), 1)
        self.assertEqual(oai[0]["role"], "assistant")
        self.assertIn("tool_calls", oai[0])
        self.assertEqual(len(oai[0]["tool_calls"]), 1)
        self.assertEqual(oai[0]["tool_calls"][0]["function"]["name"], "search_articles")

    def test_convert_tool_result_block(self):
        """Конвертация tool_result блока в текст user-сообщения."""
        oai = OpenAICompatProvider._convert_messages([{
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_abc123",
                    "content": "Found 42 articles",
                    "is_error": False,
                }
            ],
        }])
        self.assertEqual(len(oai), 1)
        self.assertIn("Found 42 articles", oai[0]["content"])
        self.assertIn("toolu_abc123", oai[0]["content"])

    def test_convert_anthropic_tools_to_openai_format(self):
        """Конвертация Anthropic tool schemas в OpenAI function calling формат."""
        anthropic_tools = [{
            "name": "search_articles",
            "description": "Search storage for articles",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        }]
        oai_tools = OpenAICompatProvider._convert_tools(anthropic_tools)

        self.assertEqual(len(oai_tools), 1)
        self.assertEqual(oai_tools[0]["type"], "function")
        fn = oai_tools[0]["function"]
        self.assertEqual(fn["name"], "search_articles")
        self.assertEqual(fn["description"], "Search storage for articles")
        self.assertIn("query", fn["parameters"]["properties"])
        self.assertEqual(fn["parameters"]["properties"]["query"]["type"], "string")

    def test_parse_oai_tool_call_response(self):
        """Парсинг OpenAI ответа с tool_calls обратно в стандартный формат."""
        oai_response = {
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "I'll search for that.",
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "search_articles",
                            'arguments': '{"query": "permafrost", "limit": 5}',
                        },
                    }],
                },
            }],
            "usage": {"prompt_tokens": 200, "completion_tokens": 85},
        }

        provider = OpenAICompatProvider(api_key="test")
        result = provider._parse_oai_tool_response(oai_response)

        self.assertEqual(result["stop_reason"], "tool_use")
        self.assertTrue(any(b["type"] == "text" for b in result["content"]))
        self.assertTrue(any(b["type"] == "tool_use" for b in result["content"]))
        tool_block = [b for b in result["content"] if b["type"] == "tool_use"][0]
        self.assertEqual(tool_block["name"], "search_articles")
        self.assertEqual(tool_block["input"]["query"], "permafrost")

    def test_parse_oai_stop_response(self):
        """Парсинг обычного текстового ответа OpenAI."""
        oai_response = {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "There are 54 articles about permafrost in our database.",
                },
            }],
            "usage": {"prompt_tokens": 150, "completion_tokens": 42},
        }

        provider = OpenAICompatProvider(api_key="test")
        result = provider._parse_oai_tool_response(oai_response)

        self.assertEqual(result["stop_reason"], "end_turn")
        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertIn("54 articles", result["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
