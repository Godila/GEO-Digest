"""Unit tests for ToolRegistry, BaseTool, and ToolResult.

Tests cover:
  - ToolResult: ok/error creation, serialisation, content block conversion
  - BaseTool: abstract interface
  - ToolRegistry: decorator registration, class-based registration,
                  execution, error handling, schema listing
"""

import json
import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.tools.base import ToolRegistry, BaseTool, ToolResult


class TestToolResult(unittest.TestCase):
    """Test ToolResult creation and serialisation."""

    def test_ok_creation(self):
        """Создание успешного результата."""
        result = ToolResult.ok(content="Found 42 articles")
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Found 42 articles")
        self.assertEqual(result.error_msg, "")

    def test_error_creation(self):
        """Создание результата с ошибкой."""
        result = ToolResult.fail(message="Storage not found")
        self.assertFalse(result.success)
        self.assertEqual(result.error_msg, "Storage not found")

    def test_ok_with_data(self):
        """ToolResult.ok с данными."""
        data = {"count": 42, "topic": "permafrost"}
        result = ToolResult.ok(data=data)
        self.assertTrue(result.success)
        self.assertEqual(result.data["count"], 42)

    def test_to_content_block_success(self):
        """Конвертация в tool_result блок (успех)."""
        result = ToolResult.ok(content="42 articles found")
        block = result.to_content_block("toolu_abc123")

        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "toolu_abc123")
        self.assertEqual(block["content"], "42 articles found")
        self.assertFalse(block["is_error"])

    def test_content_block_error(self):
        """Конвертация в tool_result блок (ошибка)."""
        result = ToolResult.fail(message="Connection failed")
        block = result.to_content_block("toolu_xyz789")

        self.assertTrue(block["is_error"])
        self.assertIn("Error: Connection failed", block["content"])

    def test_serialize_dict_data(self):
        """Сериализация dict данных в JSON строку."""
        result = ToolResult.ok(data={"articles": [1, 2, 3], "total": 3})
        output = result._serialize_data()
        parsed = json.loads(output)
        self.assertEqual(parsed["total"], 3)

    def test_serialize_string_data(self):
        """Строковые данные возвращаются как есть."""
        result = ToolResult.ok(data="plain text output")
        self.assertEqual(result._serialize_data(), "plain text output")

    def test_serialize_none_data(self):
        """None данные → '(no data)'."""
        result = ToolResult.ok(data=None)
        self.assertEqual(result._serialize_data(), "(no data)")

    def test_to_dict_roundtrip(self):
        """to_dict() содержит все поля."""
        result = ToolResult.ok(
            content="test",
            data={"key": "val"},
            metadata={"source": "test"},
        )
        d = result.to_dict()
        self.assertTrue(d["success"])
        self.assertEqual(d["content"], "test")
        self.assertEqual(d["data"]["key"], "val")


class TestBaseTool(unittest.TestCase):
    """Test BaseTool abstract class."""

    def test_default_attributes(self):
        """Базовые атрибуты по умолчанию."""
        tool = BaseTool()
        self.assertEqual(tool.name, "")
        self.assertEqual(tool.description, "")
        self.assertEqual(tool.input_schema, {})

    def test_execute_raises_not_implemented(self):
        """execute() должен поднимать NotImplementedError."""
        tool = BaseTool()
        with self.assertRaises(NotImplementedError):
            tool.execute()

    def test_get_schema_returns_definition(self):
        """get_schema() возвращает Anthropic-формат определение."""
        tool = BaseTool()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.input_schema = {"type": "object", "properties": {}}

        schema = tool.get_schema()
        self.assertEqual(schema["name"], "test_tool")
        self.assertEqual(schema["description"], "A test tool")
        self.assertIn("input_schema", schema)


class TestToolRegistryDecoratorRegistration(unittest.TestCase):
    """Test ToolRegistry decorator-based registration."""

    def setUp(self):
        self.registry = ToolRegistry()

    def test_register_and_retrieve(self):
        """Регистрация и получение инструмента."""
        @self.registry.tool(
            name="echo",
            description="Echo input back",
            input_schema={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        )
        def echo(msg: str) -> ToolResult:
            return ToolResult.ok(content=msg)

        handler = self.registry.get("echo")
        self.assertIsNotNone(handler)

    def test_decorator_preserves_function(self):
        """Декоратор не меняет функцию."""
        @self.registry.tool(name="identity", description="Identity", input_schema={})
        def identity(x): return x

        # Function should still be callable normally
        self.assertEqual(identity(42), 42)

    def test_register_multiple_tools(self):
        """Регистрация нескольких инструментов."""
        for i in range(5):
            self.registry.tool(
                name=f"tool_{i}",
                description=f"Tool number {i}",
                input_schema={"type": "object", "properties": {}},
            )(lambda x, i=i: ToolResult.ok(content=str(i)))

        self.assertEqual(len(self.registry), 5)
        self.assertIn("tool_0", self.registry)
        self.assertIn("tool_4", self.registry)

    def test_list_tools_sorted(self):
        """list_tools() возвращает отсортированный список."""
        self.registry.tool(name="zebra", description="", input_schema={})(lambda: None)
        self.registry.tool(name="alpha", description="", input_schema={})(lambda: None)
        self.registry.tool(name="middle", description="", input_schema={})(lambda: None)

        tools = self.registry.list_tools()
        self.assertEqual(tools, ["alpha", "middle", "zebra"])


class TestToolRegistryExecution(unittest.TestCase):
    """Test ToolRegistry execute functionality."""

    def setUp(self):
        self.registry = ToolRegistry()

        @self.registry.tool(
            name="echo",
            description="Echo message",
            input_schema={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        )
        def echo(msg: str) -> ToolResult:
            return ToolResult.ok(content=msg)

        @self.registry.tool(
            name="double",
            description="Double a number",
            input_schema={
                "type": "object",
                "properties": {"n": {"type": "number"}},
                "required": ["n"],
            },
        )
        def double(n: int) -> ToolResult:
            return ToolResult.ok(data={"result": n * 2})

        @self.registry.tool(
            name="fail_tool",
            description="A tool that always fails",
            input_schema={"type": "object", "properties": {}},
        )
        def fail_tool() -> ToolResult:
            raise ValueError("Intentional failure for testing")

    def test_execute_calls_handler_with_params(self):
        """execute вызывает handler с переданными параметрами."""
        result = self.registry.execute("echo", {"msg": "hello world"})
        self.assertTrue(result.success)
        self.assertEqual(result.content, "hello world")

    def test_execute_returns_tool_result_for_raw_dict(self):
        """execute оборачивает raw dict в ToolResult."""
        result = self.registry.execute("double", {"n": 21})
        self.assertTrue(result.success)
        self.assertEqual(result.data["result"], 42)

    def test_execute_unknown_tool_returns_error(self):
        """Неизвестный инструмент возвращает ошибку."""
        result = self.registry.execute("nonexistent_tool", {})
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.error_msg)
        self.assertIn("nonexistent_tool", result.error_msg)

    def test_execute_exception_wrapped_in_error(self):
        """Исключение в handler оборачивается в error ToolResult."""
        result = self.registry.execute("fail_tool", {})
        self.assertFalse(result.success)
        self.assertIn("ValueError", result.error_msg)
        self.assertIn("Intentional failure", result.error_msg)

    def test_execute_records_execution_time(self):
        """execute записывает время выполнения."""
        result = self.registry.execute("echo", {"msg": "timing test"})
        self.assertGreater(result.execution_time_ms, 0)

    def test_execute_raw_string_wrapped(self):
        """Строка из handler оборачивается в ToolResult."""
        @self.registry.tool(
            name="str_returner",
            description="Returns string",
            input_schema={"type": "object", "properties": {}},
        )
        def str_fn(): return "raw string"

        result = self.registry.execute("str_returner", {})
        self.assertTrue(result.success)
        self.assertEqual(result.content, "raw string")

    def test_execute_none_wrapped(self):
        """None из handler оборачивается в ToolResult."""
        @self.registry.tool(
            name="void_fn",
            description="Returns nothing",
            input_schema={"type": "object", "properties": {}},
        )
        def void_fn(): return None

        result = self.registry.execute("void_fn", {})
        self.assertTrue(result.success)
        self.assertEqual(result.content, "(completed)")


class TestToolRegistrySchema(unittest.TestCase):
    """Test schema generation from registry."""

    def setUp(self):
        self.registry = ToolRegistry()

        self.registry.tool(
            name="search",
            description="Search articles",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        )(lambda **kw: ToolResult.ok())

        self.registry.tool(
            name="count",
            description="Count articles",
            input_schema={
                "type": "object",
                "properties": {},
            },
        )(lambda **kw: ToolResult.ok())

    def test_get_schemas_returns_all(self):
        """get_schemas() возвращает все инструменты."""
        schemas = self.registry.get_schemas()
        self.assertEqual(len(schemas), 2)

    def test_get_schemas_has_correct_format(self):
        """Схемы в Anthropic формате (name, description, input_schema)."""
        schemas = self.registry.get_schemas()
        for s in schemas:
            self.assertIn("name", s)
            self.assertIn("description", s)
            self.assertIn("input_schema", s)

    def test_get_single_schema(self):
        """get_schema() возвращает схему одного инструмента."""
        schema = self.registry.get_schema("search")
        self.assertIsNotNone(schema)
        self.assertEqual(schema["name"], "search")

    def test_get_single_schema_missing(self):
        """get_schema() для несуществующего → None."""
        schema = self.registry.get_schema("nonexistent")
        self.assertIsNone(schema)


class TestToolRegistryClassBasedRegistration(unittest.TestCase):
    """Test class-based tool registration via BaseTool subclass."""

    def setUp(self):
        self.registry = ToolRegistry()

    def test_register_class_instance(self):
        """Регистрация класса-наследника BaseTool."""

        class GreetTool(BaseTool):
            name = "greet"
            description = "Greet someone"
            input_schema = {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }

            def execute(self, name: str) -> ToolResult:
                return ToolResult.ok(content=f"Hello, {name}!")

        instance = self.registry.register_class(GreetTool)
        self.assertIsInstance(instance, GreetTool)
        self.assertIn("greet", self.registry)

        result = self.registry.execute("greet", {"name": "World"})
        self.assertEqual(result.content, "Hello, World!")


class TestToolRegistryManualRegistration(unittest.TestCase):
    """Test manual (non-decorator) registration."""

    def setUp(self):
        self.registry = ToolRegistry()

    def test_manual_register_and_execute(self):
        """Ручная регистрация и выполнение."""
        def add(a: int, b: int) -> ToolResult:
            return ToolResult.ok(data={"sum": a + b})

        self.registry.register(
            name="add",
            handler=add,
            schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
            description="Add two numbers",
        )

        result = self.registry.execute("add", {"a": 3, "b": 4})
        self.assertTrue(result.success)
        self.assertEqual(result.data["sum"], 7)

    def test_contains_operator(self):
        """Оператор 'in' работает."""
        self.registry.register("test", lambda: ToolResult.ok(), {}, description="test")
        self.assertIn("test", self.registry)
        self.assertNotIn("missing", self.registry)


if __name__ == "__main__":
    unittest.main()
