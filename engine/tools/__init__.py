"""Tool Registry — LLM function calling tools for the Editor Agent.

This module provides:
  - ToolRegistry: register, discover, and execute tools by name
  - BaseTool: base class for defining tools with JSON Schema
  - ToolResult: standardised return type from tool execution

Tools are registered at startup and exposed to the LLM via Anthropic
tool_use API format. Each tool has a name, description, and input_schema
(JSON Schema) that the LLM uses to decide when and how to call it.
"""

from engine.tools.base import ToolRegistry, BaseTool, ToolResult

__all__ = ["ToolRegistry", "BaseTool", "ToolResult"]
