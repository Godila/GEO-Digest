# Стадия S0: Инфраструктура и подготовка

## Цель
Создать базовую структуру для tool-use архитектуры, обновить зависимости, 
подготовить тестовое окружение.

## Что делаем

### 0.1 Обновляем LLM Base Interface
Файл: `engine/llm/base.py`

Добавить метод `tool_complete()` в абстрактный класс:
```python
def tool_complete(
    self,
    messages: list[dict],
    tools: list[dict],
    system: str = "",
    temperature: float = 0.25,
    max_tokens: int = 4096,
) -> dict:
    """
    Tool-use completion.
    Returns dict with keys: content, stop_reason, usage, model
    """
    raise NotImplementedError(f"{self.__class__.__name__} does not support tool use")
```

### 0.2 Добавляем tool_complete в MiniMax Provider
Файл: `engine/llm/minimax.py`

Реализовать через Anthropic Messages API:
- Отправлять `tools` в payload
- Парсить `content` blocks (type=text + type=tool_use)
- Возвращать `stop_reason` ("end_turn" | "tool_use")
- Retry logic (уже есть, переиспользовать)

### 0.3 Добавляем tool_complete в OpenAI Compat Provider
Файл: `engine/llm/openai_compat.py`

Реализовать через OpenAI chat/completions с `tools`/`function_calling`.

### 0.4 Создаём директорию для tools
```
engine/tools/
├── __init__.py
├── base.py          # ToolRegistry, BaseTool
├── storage_tools.py # search_storage, get_article_detail, validate_doi, etc.
└── scout_tools.py   # propose_scout_query (будет в S3)
```

### 0.5 Конфигурация мульти-моделей
Файл: `engine/config.py` (дополнить)

```yaml
llm:
  provider: minimax
  models:
    fast: MiniMax-M2.5       # для анализа, валидации
    strong: MiniMax-M2.7     # для генерации, writing
    strict: MiniMax-M2.5     # для review (temp=0.05)
  fallback_chain:
    planning: [strong, fast]
    analysis: [fast]
    validation: [strict]
    writing: [strong, fast]
    review: [strict]
```

### 0.6 Тестовая fixtures
Файлы: `tests/fixtures/`
- `sample_articles.jsonl` — 20 статей для unit-тестов
- `sample_tool_response.json` — пример ответа MiniMax с tool_use

## Acceptance Criteria S0

- [ ] `LLMProvider.tool_complete()` поднимает NotImplementedError
- [ ] `MiniMaxProvider.tool_complete()` успешно вызывает API с tools
- [ ] `MiniMaxProvider.tool_complete()` корректно парсит tool_use response
- [ ] `MiniMaxProvider.tool_complete()` корректно парсит end_turn response
- [ ] `OpenAICompatProvider.tool_complete()` реализован (можно заглушка если нет ключа)
- [ ] `ToolRegistry` может регистрировать и находить tools по имени
- [ ] Все существующие тесты проходят (`pytest tests/`)

## Тесты S0

### Unit тесты

**test_llm_tool_interface.py**
```python
class TestToolCompleteInterface:
    def test_base_raises_not_implemented():
        """Базовый класс должен поднимать ошибку"""
        provider = LLMProvider(model="test")
        with pytest.raises(NotImplementedError):
            provider.tool_complete([], [])

    def test_minimax_tool_use_response():
        """MiniMax возвращает stop_reason=tool_use когда модель хочет вызвать tool"""
        # Mock HTTP response with tool_use content block
        result = minimax.tool_complete(messages, tools)
        assert result["stop_reason"] == "tool_use"
        assert any(b["type"] == "tool_use" for b in result["content"])

    def test_minimax_end_turn_response():
        """MiniMax возвращает end_turn когда модель закончила"""
        result = minimax.tool_complete(messages, [])
        assert result["stop_reason"] == "end_turn"

    def test_minimax_multi_turn_with_tool_result():
        """MiniMax корректно обрабатывает tool_result в истории"""
        messages = [
            user_msg,
            assistant_msg_with_tool_call,
            {"role": "user", "content": [tool_result_block]}
        ]
        result = minimax.tool_complete(messages, tools)
        assert "content" in result

    def test_minimax_tool_retry_on_timeout():
        """Retry при timeout работает"""
        # Mock: первый вызов timeout, второй — успех
        result = minimax.tool_complete(messages, tools)
        assert result is not None

    def test_openai_compat_tool_interface():
        """OpenAI compat имеет tool_complete (может быть заглушка)"""
        if OPENAI_KEY_AVAILABLE:
            result = openai_provider.tool_complete(messages, tools)
            assert "stop_reason" in result
        else:
            with pytest.raises(NotImplementedError):
                openai_provider.tool_complete(messages, tools)
```

**test_tool_registry.py**
```python
class TestToolRegistry:
    def test_register_and_retrieve():
        reg = ToolRegistry()
        reg.register("search", search_fn)
        assert reg.get("search") is not None

    def test_unknown_tool_returns_error():
        reg = ToolRegistry()
        result = reg.execute("nonexistent", {})
        assert "error" in result

    def test_execute_calls_handler():
        reg = ToolRegistry()
        reg.register("echo", lambda x: x)
        result = reg.execute("echo", {"msg": "hello"})
        assert result == {"msg": "hello"}

    def test_list_tools_schema():
        reg = ToolRegistry()
        reg.register("search", search_fn, schema=search_schema)
        schemas = reg.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "search"
```

### Интеграционный тест

**test_minimax_tool_use_integration.py**
```python
@mark.integration  # только если есть API ключ
class TestMiniMaxToolUseIntegration:
    def test_real_tool_call_and_response():
        """Настоящий вызов MiniMax API с tools"""
        llm = MiniMaxProvider(api_key=REAL_KEY)
        tools = [{
            "name": "get_count",
            "description": "Get count",
            "input_schema": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"]
            }
        }]
        result = llm.tool_complete(
            messages=[{"role": "user", "content": "How many articles about Arctic?"}],
            tools=tools,
            max_tokens=256
        )
        assert result["stop_reason"] in ("tool_use", "end_turn")
        assert "usage" in result
        assert result["usage"]["input_tokens"] > 0

    def test_real_tool_use_multi_turn():
        """Полный цикл: tool_use → tool_result → final response"""
        llm = MiniMaxProvider(api_key=REAL_KEY)
        tools = [get_count_tool]

        # Turn 1
        r1 = llm.tool_complete([user_msg], tools, max_tokens=256)
        assert r1["stop_reason"] == "tool_use"

        # Turn 2 — отправляем результат
        tool_id = extract_tool_id(r1["content"])
        messages = [
            user_msg,
            build_assistant_msg(r1["content"]),
            build_tool_result_msg(tool_id, "Found 54 articles")
        ]
        r2 = llm.tool_complete(messages, tools, max_tokens=512)
        assert r2["stop_reason"] == "end_turn"
        text = extract_text(r2["content"])
        assert len(text) > 10  # осмысленный ответ
```

## Файлы стадии S0

| Файл | Действие | Строк |
|------|----------|-------|
| `engine/llm/base.py` | Изменить | +15 |
| `engine/llm/minimax.py` | Изменить | +60 |
| `engine/llm/openai_compat.py` | Изменить | +40 |
| `engine/tools/__init__.py` | Создать | 10 |
| `engine/tools/base.py` | Создать | ~80 |
| `tests/test_llm_tool_interface.py` | Создать | ~120 |
| `tests/test_tool_registry.py` | Создать | ~80 |
| `tests/test_minimax_tool_use_integration.py` | Создать | ~100 |
| `tests/fixtures/sample_articles.jsonl` | Создать | ~100 |

**Итого:** ~605 строк новых/изменённых файлов
