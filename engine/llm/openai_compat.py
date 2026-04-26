"""OpenAI-Compatible LLM Provider (for Reviewer Agent)."""
from __future__ import annotations
import json, time, urllib.error, urllib.request
from engine.llm.base import LLMProvider

class OpenAICompatProvider(LLMProvider):
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    
    def __init__(self, api_key="", base_url="", model="gpt-4o", timeout=180, retries=3, **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.timeout = timeout
        self.retries = retries
    
    def _headers(self):
        return {"Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}
    
    def complete(self, prompt, system="", temperature=0.3, max_tokens=4096):
        url = f"{self.base_url}/chat/completions"
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": msgs,
                   "temperature": temperature, "max_tokens": max_tokens}
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
        Tool-use completion via OpenAI chat/completions API.

        Converts between Anthropic-format messages (internal) and
        OpenAI chat/completions format with function calling.

        Args:
            messages: Conversation history in Anthropic format.
                Converted to OpenAI format internally.
            tools: List of tool definitions (Anthropic format, converted).
            system: System prompt string.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Returns:
            dict with keys: content, stop_reason, usage, model
            (Same format as MiniMax.tool_complete for consistency)
        """
        url = f"{self.base_url}/chat/completions"

        # Convert Anthropic → OpenAI message format
        oai_messages = self._convert_messages(messages, system)

        # Convert Anthropic → OpenAI tools format
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
                # Handle content blocks (text, tool_use, tool_result)
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
                        # tool_result goes into user message as context
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

        # Map finish reasons
        stop_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
            "content_filter": "end_turn",
        }
        stop_reason = stop_map.get(finish_reason, "end_turn")

        # Build content blocks
        content_blocks = []
        text_content = msg.get("content", "")
        if text_content:
            content_blocks.append({"type": "text", "text": text_content})

        # Convert tool_calls to tool_use blocks
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
