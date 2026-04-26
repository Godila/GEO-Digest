"""Tool Registry — Base classes for LLM function calling tools.

Core abstractions:
  - ToolResult: Standardised return from tool execution
  - BaseTool: Abstract base for defining a tool with schema + handler
  - ToolRegistry: Container that registers, discovers, and executes tools

Usage:
    registry = ToolRegistry()

    @registry.tool(
        name="search_articles",
        description="Search articles in storage",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    )
    def search_articles(query: str, limit: int = 10) -> ToolResult:
        articles = storage.search(query)
        return ToolResult.ok(data=[a.to_dict() for a in articles[:limit]])

    # Get all schemas for LLM
    schemas = registry.get_schemas()  # → list[dict] for API

    # Execute by name (from LLM response)
    result = registry.execute("search_articles", {"query": "permafrost"})
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ── ToolResult ──────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Standardised result from tool execution.

    Used to return structured data back to the LLM after tool execution.
    The LLM sees the content string as the tool_result block.
    """
    success: bool = True
    content: str = ""
    data: Any = None
    error_msg: str = ""
    execution_time_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def ok(cls, content: str = "", data: Any = None, **meta) -> "ToolResult":
        """Create successful result."""
        return cls(success=True, content=content, data=data, **meta)

    @classmethod
    def fail(cls, message: str, data: Any = None, **meta) -> "ToolResult":
        """Create error result."""
        return cls(success=False, content="", error_msg=message, data=data, **meta)

    def to_content_block(self, tool_use_id: str) -> dict:
        """
        Convert to Anthropic tool_result content block.

        Returns dict suitable for appending to user message content list:
            {"type": "tool_result", "tool_use_id": "...", "content": "..."}
        """
        if self.success:
            output = self.content or self._serialize_data()
        else:
            output = f"Error: {self.error_msg}"

        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": output,
            "is_error": not self.success,
        }

    def _serialize_data(self) -> str:
        """Serialise data field to JSON string for LLM consumption."""
        if self.data is None:
            return "(no data)"
        if isinstance(self.data, str):
            return self.data
        try:
            return json.dumps(self.data, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(self.data)

    def to_dict(self) -> dict:
        """Serialise full result to dict (for logging/debugging)."""
        return {
            "success": self.success,
            "content": self.content,
            "data": self.data,
            "error": self.error_msg,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "metadata": self.metadata,
        }


# ── BaseTool ───────────────────────────────────────────────────

class BaseTool:
    """Base class for defining an LLM-callable tool.

    Subclass this to create typed tools with explicit schema:

        class SearchArticlesTool(BaseTool):
            name = "search_articles"
            description = "Search articles in storage"
            input_schema = {...}

            def execute(self, query: str, limit: int = 10) -> ToolResult:
                ...
    """

    name: str = ""
    description: str = ""
    input_schema: dict = {}

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with validated parameters. Override in subclass."""
        raise NotImplementedError(f"{self.__class__.__name__}.execute() not implemented")

    def get_schema(self) -> dict:
        """Return tool definition in Anthropic format for LLM API."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


# ── ToolRegistry ────────────────────────────────────────────────

class ToolRegistry:
    """
    Registry for LLM function calling tools.

    Supports two registration styles:
      1. Decorator-based: @registry.tool(name=..., schema=...)
      2. Class-based:   registry.register_class(SearchArticlesTool)

    Provides:
      - get_schemas(): All tool definitions for LLM API payload
      - execute(name, params): Run tool by name, returns ToolResult
      - get(name): Get tool handler by name
      - list_tools(): List of registered tool names
    """

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._schemas: dict[str, dict] = {}
        self._metadata: dict[str, dict] = {}

    def tool(
        self,
        name: str,
        description: str = "",
        input_schema: dict | None = None,
        **metadata,
    ):
        """
        Decorator to register a function as an LLM tool.

        Usage:
            @registry.tool(
                name="echo",
                description="Echo input",
                input_schema={
                    "type": "object",
                    "properties": {"msg": {"type": "string"}},
                    "required": ["msg"],
                },
            )
            def echo(msg: str) -> ToolResult:
                return ToolResult.ok(content=msg)
        """
        def decorator(fn):
            self._tools[name] = fn
            self._schemas[name] = {
                "name": name,
                "description": description or fn.__doc__ or "",
                "input_schema": input_schema or {"type": "object", "properties": {}},
            }
            self._metadata[name] = metadata
            return fn
        return decorator

    def register_class(self, tool_cls: type[BaseTool]) -> BaseTool:
        """Register a BaseTool subclass instance."""
        instance = tool_cls()
        self._tools[instance.name] = instance.execute
        self._schemas[instance.name] = instance.get_schema()
        return instance

    def register(self, name: str, handler: Callable, schema: dict | None = None, **meta):
        """Manual registration of a tool function."""
        self._tools[name] = handler
        self._schemas[name] = {
            "name": name,
            "description": meta.get("description", ""),
            "input_schema": schema or {"type": "object", "properties": {}},
        }
        self._metadata[name] = meta

    def get(self, name: str) -> Callable | None:
        """Get tool handler by name. Returns None if not found."""
        return self._tools.get(name)

    def get_schema(self, name: str) -> dict | None:
        """Get single tool schema by name."""
        return self._schemas.get(name)

    def get_schemas(self) -> list[dict]:
        """Get ALL tool definitions in Anthropic format (for LLM API)."""
        return list(self._schemas.values())

    def execute(self, name: str, params: dict) -> ToolResult:
        """
        Execute a registered tool by name with given parameters.

        Args:
            name: Tool name (as returned by LLM in tool_use block).
            params: Input parameters dict (from LLM's tool_use.input).

        Returns:
            ToolResult with success/error status and content.

        Raises:
            KeyError: If tool name is not registered.
            Exception: If tool handler raises (wrapped in error result).
        """
        handler = self._tools.get(name)
        if handler is None:
            available = ", ".join(sorted(self._tools.keys())) or "(none)"
            return ToolResult.fail(
                f"Unknown tool '{name}'. Available tools: {available}"
            )

        start = time.monotonic()
        try:
            result = handler(**params)
            elapsed = (time.monotonic() - start) * 1000

            # Wrap raw dicts/strings in ToolResult if needed
            if not isinstance(result, ToolResult):
                if isinstance(result, dict):
                    result = ToolResult.ok(data=result)
                elif isinstance(result, str):
                    result = ToolResult.ok(content=result)
                elif result is None:
                    result = ToolResult.ok(content="(completed)")
                else:
                    result = ToolResult.ok(data=result)

            result.execution_time_ms = elapsed
            return result

        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult.fail(
                message=f"Tool '{name}' raised {type(e).__name__}: {e}",
                execution_time_ms=elapsed,
            )

    def list_tools(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"
