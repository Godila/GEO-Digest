"""OpenAI-Compatible LLM Provider with OpenRouter optimizations.

Supports:
- response_format: {"type": "json_object"} for guaranteed JSON
- reasoning.effort: control thinking token budget for reasoning models
- plugins: response-healing for automatic JSON repair
- Auto-detects OpenRouter base URL and enables optimizations
"""
from __future__ import annotations
import json, time, urllib.error, urllib.request
from engine.llm.base import LLMProvider

class OpenAICompatProvider(LLMProvider):
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, api_key="", base_url="", model="gpt-4o",
                 timeout=300, retries=3, reasoning_effort="low",
                 use_json_mode=True, use_response_healing=True, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.retries = retries
        self.reasoning_effort = reasoning_effort
        self.use_json_mode = use_json_mode
        self.use_response_healing = use_response_healing

    @property
    def is_openrouter(self):
        return "openrouter.ai" in self.base_url

    @property
    def is_reasoning_model(self):
        """Detect reasoning/thinking models."""
        name = self.model.lower()
        return any(x in name for x in [
            "gemini-3", "gemini-2.5", "o1", "o3", "o4",
            "deepseek-r1", "claude-3.7", "claude-4",
            "glm-5",
        ])

    def _headers(self):
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}

    def _build_payload(self, prompt, system, temperature, max_tokens,
                       force_json=False):
        """Build API payload with OpenRouter-specific optimizations."""
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if not self.is_openrouter:
            return payload

        # ── OpenRouter-specific optimizations ──

        # 1. Reasoning effort — reduce thinking tokens for reasoning models
        #    so more of max_tokens budget goes to actual content
        if self.is_reasoning_model and self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        # 2. JSON mode — force valid JSON output
        if force_json and self.use_json_mode:
            payload["response_format"] = {"type": "json_object"}

        # 3. Response healing — auto-repair malformed JSON
        if (force_json or self.use_response_healing) and self.use_response_healing:
            payload.setdefault("plugins", [])
            payload["plugins"].append({"id": "response-healing"})

        return payload

    def complete(self, prompt, system="", temperature=0.3, max_tokens=4096):
        """Standard text completion."""
        payload = self._build_payload(prompt, system, temperature, max_tokens)
        return self._request(payload)

    def _request(self, payload):
        """Execute API request with retries."""
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI API error: {last_err}")

    def health_check(self):
        try:
            self.complete("ping", max_tokens=5); return True
        except Exception:
            return False

    # ── JSON Completion (with OpenRouter optimizations) ──────

    def complete_json(self, prompt: str, system: str = "", temperature: float = 0.3, max_tokens: int = 4096, **kwargs) -> str | dict | list:
        """Complete with guaranteed JSON output.

        When using OpenRouter:
        1. Sends response_format: {type: "json_object"} for guaranteed JSON
        2. Enables response-healing plugin for auto-repair
        3. Uses reasoning.effort to control thinking budget

        Falls back to manual JSON extraction for non-OpenRouter providers.
        """
        import re

        if self.is_openrouter and self.use_json_mode:
            # OpenRouter path: guaranteed JSON + healing
            payload = self._build_payload(
                prompt, system, temperature, max_tokens, force_json=True,
            )
            raw_text = self._request(payload)

            if isinstance(raw_text, dict) or isinstance(raw_text, list):
                return raw_text

            if not isinstance(raw_text, str):
                return raw_text

            # response_format guarantees valid JSON, but just in case:
            text = raw_text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # Markdown fence extraction (shouldn't be needed with json_object mode)
            fence_match = re.search(
                r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL,
            )
            if fence_match:
                try:
                    return json.loads(fence_match.group(1).strip())
                except json.JSONDecodeError:
                    pass

            # Brute-force JSON extraction
            for opener, closer in [("{", "}"), ("[", "]")]:
                start = text.find(opener)
                if start == -1:
                    continue
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == opener:
                        depth += 1
                    elif text[i] == closer:
                        depth -= 1
                        if depth == 0:
                            candidate = text[start:i + 1]
                            try:
                                return json.loads(candidate)
                            except json.JSONDecodeError:
                                fixed = self._repair_truncated_json(candidate, opener, closer)
                                if fixed is not None:
                                    return fixed
                            break

            return raw_text
        else:
            # Non-OpenRouter path: manual JSON extraction
            raw_text = self.complete(
                prompt, system=system, temperature=temperature, max_tokens=max_tokens,
            )
            return self._extract_json(raw_text)

    def _extract_json(self, raw_text):
        """Extract JSON from raw LLM text output."""
        import re

        if not isinstance(raw_text, str):
            return raw_text

        text = raw_text.strip()

        # Strip markdown code fences
        fence_match = re.search(
            r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL,
        )
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for opener, closer in [("{", "}"), ("[", "]")]:
            start = text.find(opener)
            if start == -1:
                continue
            depth = 0
            for i in range(start, len(text)):
                if text[i] == opener:
                    depth += 1
                elif text[i] == closer:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            fixed = self._repair_truncated_json(candidate, opener, closer)
                            if fixed is not None:
                                return fixed
                        break

        return raw_text

    @staticmethod
    def _repair_truncated_json(text: str, opener: str, closer: str):
        """Attempt to repair truncated JSON by closing open structures."""
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        if open_braces < 0 or open_brackets < 0:
            return None

        repaired = text
        if repaired.endswith(",") or repaired.endswith(":"):
            repaired = repaired.rstrip(",:")
        if repaired.endswith('"') and repaired.count('"') % 2 != 0:
            repaired += '"'

        repaired += "]" * max(0, open_brackets)
        repaired += "}" * max(0, open_braces)

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None

    # ── Tool-Use Completion ─────────────────────────────────────

    def tool_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        temperature: float = 0.25,
        max_tokens: int = 4096,
    ) -> dict:
        """Tool-use completion via OpenAI chat/completions API."""
        url = f"{self.base_url}/chat/completions"
        oai_messages = self._convert_messages(messages, system)
        oai_tools = None
        if tools:
            oai_tools = self._convert_tools(tools)

        payload = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if oai_tools:
            payload["tools"] = oai_tools
            payload["tool_choice"] = "auto"

        if self.is_openrouter and self.is_reasoning_model and self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers(), method="POST")
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    result = json.loads(resp.read().decode())
                return self._parse_oai_tool_response(result)
            except Exception as e:
                last_err = e
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI API error after {self.retries + 1} attempts: {last_err}")

    # ── Format Conversion Helpers ───────────────────────────────

    @staticmethod
    def _convert_messages(anthropic_msgs: list[dict], system: str = "") -> list[dict]:
        """Convert Anthropic-format messages to OpenAI chat format."""
        oai = []
        if system:
            oai.append({"role": "system", "content": system})

        for msg in anthropic_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if isinstance(content, str):
                oai.append({"role": role, "content": content})
            elif isinstance(content, list):
                parts = []
                has_tool_calls = False
                tool_calls_list = []

                for block in content:
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "tool_use":
                        has_tool_calls = True
                        tool_calls_list.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                    elif btype == "tool_result":
                        result_text = block.get("content", "")
                        is_error = block.get("is_error", False)
                        prefix = "[Error] " if is_error else ""
                        parts.append(f"Tool {block.get('tool_use_id', '')} result: {prefix}{result_text}")

                text_content = "\n".join(parts)

                if role == "assistant" and has_tool_calls:
                    oai.append({
                        "role": "assistant",
                        "content": text_content or None,
                        "tool_calls": tool_calls_list,
                    })
                else:
                    oai.append({"role": role, "content": text_content})

        return oai

    @staticmethod
    def _convert_tools(anthropic_tools: list[dict]) -> list[dict]:
        """Convert Anthropic tool definitions to OpenAI function calling format."""
        oai_tools = []
        for t in anthropic_tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return oai_tools

    def _parse_oai_tool_response(self, result: dict) -> dict:
        """Parse OpenAI response back to our standard format."""
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        usage = result.get("usage", {})
        finish_reason = choice.get("finish_reason", "stop")

        stop_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "end_turn",
        }
        stop_reason = stop_map.get(finish_reason, "end_turn")

        content_blocks = []
        text_content = msg.get("content", "")
        if text_content:
            content_blocks.append({"type": "text", "text": text_content})

        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            args_str = fn.get("arguments", "{}")
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"call_{len(content_blocks)}"),
                "name": fn.get("name", ""),
                "input": args,
            })

        return {
            "content": content_blocks,
            "stop_reason": stop_reason,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            "model": result.get("model", self.model),
        }
