# Стадия S2: Tool-Use Loop (LLM Layer)

## Цель
Реализовать tool-use loop — механизм который позволяет LLM вызывать tools,
получать результаты и продолжать работу. Это "двигатель" для Editor Agent.

## Архитектура

```
ToolUseLoop
├── messages: list[dict]       # история диалога
├── tools: ToolRegistry        # доступные tools
├── llm: LLMProvider           # модель (MiniMax / OpenAI)
│
├── run(user_message) → result
│   ├── loop (max_rounds=8):
│   │   ├── llm.tool_complete(messages, tools)
│   │   ├── if stop_reason == "end_turn" → return final
│   │   ├── if stop_reason == "tool_use":
│   │   │   ├── extract tool_calls from response
│   │   │   ├── execute each tool via ToolRegistry
│   │   │   ├── append tool_results to messages
│   │   │   └── continue loop
│   │   └── if stop_reason == "max_tokens" → warning + return partial
│   └── return ToolUseResult(content, rounds, warnings)
│
└── _build_system_prompt(phase) → str
```

## Что делаем

### 2.1 ToolUseLoop
Файл: `engine/llm/tool_loop.py`

```python
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict  # распарсенный JSON

@dataclass  
class ToolUseResult:
    content: str                    # финальный текстовый ответ
    tool_calls_made: list[dict]     # история всех tool calls [{name, args, result, round}]
    total_rounds: int
    stop_reason: str
    usage: dict                     # {input_tokens, output_tokens} суммарно
    warnings: list[str]

class ToolUseLoop:
    """Оркестрирует multi-turn tool-use диалог между LLM и tools."""
    
    MAX_ROUNDS = 8
    
    def __init__(self, llm: LLMProvider, tools: ToolRegistry,
                 max_rounds: int = 8):
        self.llm = llm
        self.tools = tools
        self.max_rounds = max_rounds
    
    def run(
        self,
        user_message: str,
        system_prompt: str = "",
        temperature: float = 0.25,
        max_tokens: int = 4096,
        initial_messages: list[dict] | None = None,
    ) -> ToolUseResult:
        """
        Запускает tool-use loop.
        
        Args:
            user_message: сообщение от пользователя
            system_prompt: системный промпт (инструкции для LLM)
            temperature: температура генерации
            max_tokens: максимальных токенов на ответ
            initial_messages: предварительная история (для resume)
        
        Returns:
            ToolUseResult с финальным ответом и историей tool calls
        """
        messages = list(initial_messages) if initial_messages else []
        messages.append({"role": "user", "content": user_message})
        
        schemas = self.tools.get_schemas()
        tool_history = []  # лог всех tool calls
        total_usage = {"input_tokens": 0, "output_tokens": 0}
        warnings = []
        
        for round_num in range(1, self.max_rounds + 1):
            # Вызов LLM
            try:
                response = self.llm.tool_complete(
                    messages=messages,
                    tools=schemas,
                    system=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                warnings.append(f"Round {round_num}: LLM error: {e}")
                break
            
            # Accumulate usage
            if "usage" in response:
                for k, v in response["usage"].items():
                    total_usage[k] = total_usage.get(k, 0) + v
            
            # Добавляем ответ ассистента в историю
            messages.append({
                "role": "assistant",
                "content": response["content"]
            })
            
            stop_reason = response.get("stop_reason", "end_turn")
            
            if stop_reason == "end_turn":
                # LLM закончила — извлекаем текст
                text = self._extract_text(response["content"])
                return ToolUseResult(
                    content=text,
                    tool_calls_made=tool_history,
                    total_rounds=round_num,
                    stop_reason="end_turn",
                    usage=total_usage,
                    warnings=warnings,
                )
            
            elif stop_reason == "tool_use":
                # Извлекаем tool calls из ответа
                tool_calls = self._extract_tool_calls(response["content"])
                
                if not tool_calls:
                    warnings.append(f"Round {round_num}: tool_use but no tool_calls found")
                    break
                
                # Выполняем каждый tool
                tool_results = []
                for tc in tool_calls:
                    start = time.time()
                    result_raw = self.tools.execute(tc.name, tc.arguments)
                    elapsed = time.time() - start
                    
                    try:
                        result_data = json.loads(result_raw)
                    except json.JSONDecodeError:
                        result_data = {"raw": result_raw}
                    
                    tool_history.append({
                        "round": round_num,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result": result_data,
                        "elapsed_ms": int(elapsed * 1000),
                    })
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result_raw,
                    })
                
                # Добавляем результаты в историю
                messages.append({"role": "user", "content": tool_results})
                
            elif stop_reason == "max_tokens":
                warnings.append(f"Round {round_num}: hit max_tokens limit")
                # Пробуем продолжить с пустым user msg
                messages.append({"role": "user", 
                               "content": "Continue. If you have enough information, provide your final answer."})
                
            else:
                warnings.append(f"Round {round_num}: unexpected stop_reason={stop_reason}")
                break
        
        # Достигли лимита rounds — возвращаем что есть
        text = self._extract_text(messages[-1].get("content", ""))
        warnings.append(f"Max rounds ({self.max_rounds}) reached")
        return ToolUseResult(
            content=text,
            tool_calls_made=tool_history,
            total_rounds=self.max_rounds,
            stop_reason="max_rounds_exceeded",
            usage=total_usage,
            warnings=warnings,
        )
    
    def _extract_text(self, content_blocks: list) -> str:
        """Извлекает текст из content blocks."""
        texts = [b["text"] for b in content_blocks if b.get("type") == "text"]
        return "\n".join(texts).strip()
    
    def _extract_tool_calls(self, content_blocks: list) -> list[ToolCall]:
        """Извлекает tool calls из content blocks."""
        calls = []
        for b in content_blocks:
            if b.get("type") == "tool_use":
                calls.append(ToolCall(
                    id=b.get("id", f"call_{len(calls)}"),
                    name=b.get("name", ""),
                    arguments=b.get("input", {}),
                ))
        return calls
```

### 2.2 System Prompts для Editor
Файл: `engine/prompts/editor_prompts.py`

```python
EDITOR_SYSTEM_PROMPT = """Ты — научный редактор гео-экологического digest'а.

Твоя задача: проанализировать тему на основе данных в хранилище статей
и предложить варианты статей для написания.

У тебя есть инструменты (tools) для работы с базой данных статей.
Используй их чтобы получить ФАКТИЧЕСКИЕ данные перед тем как делать выводы.

ПРАВИЛА:
1. Всегда проверяй факты через tools перед тем как утверждать что-то
2. Каждое утверждение о количестве/статистике должно быть основано на данных из tools
3. Если информации недостаточно — скажи "недостаточно данных"
4. Не выдумывай DOI — используй только те которые вернул validate_doi или search_storage
5. Предлагай статьи которые ПОЛЕЗНЫ и НОВЫЕ (не дублируют уже существующие)

ФОРМАТ ОТВЕТА (когда закончишь собирать информацию):
Предоставь JSON-массив предложений:
[
  {
    "title": "Заголовок статьи",
    "thesis": "Тезис (2-3 предложения)",
    "target_audience": "researchers | general_public | policy_makers",
    "confidence": 0.0-1.0,
    "sources_available": N,
    "sources_needed": N,
    "key_references": ["DOI:10.xxx/..."],
    "gap_filled": "какую informational gap закрывает эта статья"
  }
]
"""

ANALYSIS_SYSTEM_PROMPT = """{editor_base}

ТЕКУЩАЯ ЗАДАЧА: Анализ покрытия темы в хранилище.

Шаги:
1. Используй count_by_cluster чтобы понять какие подтемы есть
2. Используй get_time_range чтобы понять временное покрытие
3. Используй search_storage с разными запросами чтобы оценить глубину
4. Используй check_existing_articles чтобы не дублировать
5. Сформируй картину: что хорошо покрыто, чего не хватает
"""

PROPOSAL_SYSTEM_PROMPT = """{editor_base}

ТЕКУЩАЯ ЗАДАЧА: Предложить варианты статей.

На основе анализа хранилища предложи 3-5 вариантов статей.
Каждый вариант должен:
- Быть основан на РЕАЛЬНЫХ данных из хранилища
- Заполнять информационный пробел (gap)
- Иметь достаточно источников (минимум 5)
- Не дублировать существующие статьи
"""
```

### 2.3 Парсинг ответов LLM
Файл: `engine/llm/response_parser.py`

```python
def parse_proposals_from_text(text: str) -> list[dict]:
    """Извлекает ArticleProposal[] из текстового ответа LLM.
    
    Обрабатывает:
    - Чистый JSON массив
    - JSON в markdown code fences (```json ... ```)
    - Текст с JSON-подобными блоками
    """
    # Strategy 1: direct JSON parse
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return [_validate_proposal(p) for p in data]
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: extract JSON from markdown fences
    import re
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, list):
                return [_validate_proposal(p) for p in data]
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: find JSON array in text
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [_validate_proposal(p) for p in data]
        except json.JSONDecodeError:
            pass
    
    # Fallback: return empty (не пытаться угадать!)
    return []

def _validate_proposal(p: dict) -> dict:
    """Валидирует и нормализует proposal."""
    required = {"title", "thesis"}
    for field in required:
        if field not in p or not p[field]:
            p[field] = f"[не указано: {field}]"
    p.setdefault("confidence", 0.5)
    p.setdefault("sources_available", 0)
    p.setdefault("sources_needed", 0)
    p.setdefault("key_references", [])
    p.setdefault("target_audience", "general_public")
    return p
```

## Acceptance Criteria S2

- [ ] `ToolUseLoop.run()` выполняет 1+ rounds при необходимости
- [ ] `ToolUseLoop.run()` останавливается на `end_turn`
- [ ] `ToolUseLoop.run()` вызывает tools при `stop_reason=tool_use`
- [ ] `ToolUseLoop.run()` отправляет `tool_result` обратно в LLM
- [ ] `ToolUseLoop.run()` останавливается после `MAX_ROUNDS` с предупреждением
- [ ] `ToolUseLoop.run()` накапливает `usage` по всем round'ам
- [ ] `ToolUseLoop.run()` возвращает историю всех tool calls
- [ ] `_extract_tool_calls()` корректно парсит Anthropic format
- [ ] `_extract_text()` игнорирует thinking/tool_use блоки
- [ ] `parse_proposals_from_text()` парсит чистый JSON
- [ ] `parse_proposals_from_text()` парсит JSON в markdown fences
- [ ] `parse_proposals_from_text()` возвращает [] на невалидном вводе (не падает)
- [ ] System prompts содержат инструкции по использованию tools

## Тесты S2

### Unit тесты

**test_tool_loop.py**
```python
class TestToolUseLoop:
    @fixture
    def loop_with_mock_llm():
        """Создаёт loop с mock LLM который возвращает предопределённые ответы."""
        ...
    
    def test_single_turn_end_turn(loop):
        """LLM сразу отвечает без tools."""
        llm = MockLLM(responses=[{
            "content": [{"type": "text", "text": '{"proposals":[]}'}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50}
        }])
        loop = ToolUseLoop(llm, mock_registry)
        result = loop.run("Hello")
        assert result.total_rounds == 1
        assert result.stop_reason == "end_turn"
        assert len(result.tool_calls_made) == 0
    
    def test_two_turns_one_tool_call(loop):
        """LLM вызывает tool, получает результат, отвечает."""
        llm = MockLLM(responses=[
            # Round 1: LLM хочет вызвать tool
            {
                "content": [{"type": "tool_use", "id": "call_1",
                            "name": "count_by_cluster",
                            "input": {"topic": "methane"}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 200, "output_tokens": 30}
            },
            # Round 2: LLM видит результат и отвечает
            {
                "content": [{"type": "text", "text": "Found 3 clusters..."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 300, "output_tokens": 80}
            }
        ])
        registry = MockRegistry(return_value='{"clusters":[{"theme":"A","count":10}]}')
        loop = ToolUseLoop(llm, registry)
        result = loop.run("Analyze methane articles")
        
        assert result.total_rounds == 2
        assert len(result.tool_calls_made) == 1
        assert result.tool_calls_made[0]["name"] == "count_by_cluster"
        assert result.stop_reason == "end_turn"
    
    def test_three_rounds_multiple_tools(loop):
        """LLM последовательно вызывает несколько tools."""
        llm = MockLLM(responses=[
            {"content": [tool_use("count_by_cluster", {})],
             "stop_reason": "tool_use", "usage": default_usage},
            {"content": [tool_use("get_time_range", {})],
             "stop_reason": "tool_use", "usage": default_usage},
            {"content": [text("Analysis complete")],
             "stop_reason": "end_turn", "usage": default_usage},
        ])
        result = loop.run("Analyze")
        assert result.total_rounds == 3
        assert len(result.tool_calls_made) == 2
    
    def test_max_rounds_limit(loop):
        """Остановка после MAX_ROUNDS."""
        llm = MockLLM(responses=[
            {"content": [tool_use("search", {})],
             "stop_reason": "tool_use", "usage": default_usage}
        ] * 9)  # Больше чем MAX_ROUNDS
        loop = ToolUseLoop(llm, mock_registry, max_rounds=3)
        result = loop.run("Search")
        assert result.total_rounds == 3  # остановился на лимите
        assert any("max rounds" in w.lower() for w in result.warnings)
    
    def test_accumulated_usage(loop):
        """Usage суммируется по всем round'ам."""
        llm = MockLLM(responses=[
            {"stop_reason": "tool_use", "usage": {"input_tokens": 100, "output_tokens": 20},
             "content": [tool_use("x", {})]},
            {"stop_reason": "end_turn", "usage": {"input_tokens": 200, "output_tokens": 50},
             "content": [text("done")]},
        ])
        result = loop.run("test")
        assert result.usage["input_tokens"] == 300  # 100 + 200
        assert result.usage["output_tokens"] == 70   # 20 + 50
    
    def test_initial_messages_for_resume(loop):
        """Можно передать начальную историю (для resume)."""
        prev_messages = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": [text("Answer")]},
            {"role": "user", "content": [tool_result_msg("call_1", "data")]},
        ]
        llm = MockLLM(responses=[
            {"stop_reason": "end_turn", "usage": default_usage,
             "content": [text("Follow-up answer")]}
        ])
        result = loop.run("Follow up?", initial_messages=prev_messages)
        assert "Follow-up answer" in result.content

class TestExtractToolCalls:
    def test_single_tool_call():
        blocks = [{"type": "tool_use", "id": "c1", "name": "search", "input": {"q": "x"}}]
        calls = ToolUseLoop._extract_tool_calls(None, blocks)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments == {"q": "x"}
    
    def test_mixed_content():
        blocks = [
            {"type": "thinking", "thinking": "..."},
            {"type": "tool_use", "id": "c1", "name": "validate", "input": {"doi": "10.x"}},
            {"type": "text", "text": "checking..."},
        ]
        calls = ToolUseLoop._extract_tool_calls(None, blocks)
        assert len(calls) == 1
        assert calls[0].name == "validate"

class TestExtractText:
    def test_only_text():
        blocks = [{"type": "text", "text": "hello world"}]
        assert ToolUseLoop._extract_text(None, blocks) == "hello world"
    
    def test_ignores_tool_use_and_thinking():
        blocks = [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "tool_use", "id": "c1", "name": "x", "input": {}},
            {"type": "text", "text": "real answer"},
        ]
        assert ToolUseLoop._extract_text(None, blocks) == "real answer"
    
    def test_empty_content():
        assert ToolUseLoop._extract_text(None, []) == ""

class TestParseProposals:
    def test_clean_json():
        text = '[{"title":"A","thesis":"B"}]'
        props = parse_proposals_from_text(text)
        assert len(props) == 1
        assert props[0]["title"] == "A"
    
    def test_markdown_fences():
        text = 'Here are proposals:\n```json\n[{"title":"A"}]\n```\nDone.'
        props = parse_proposals_from_text(text)
        assert len(props) == 1
    
    def test_json_in_text():
        text = 'I think [{"title":"A","thesis":"B"}] is good'
        props = parse_proposals_from_text(text)
        assert len(props) == 1
    
    def test_invalid_returns_empty():
        props = parse_proposals_from_text("just plain text no json")
        assert props == []
    
    def test_validates_required_fields():
        text = '[{"title":"A"}]'  # missing thesis
        props = parse_proposals_from_text(text)
        assert props[0]["thesis"] == "[не указано: thesis]"
    
    def test_sets_defaults():
        text = '[{"title":"A","thesis":"B"}]'
        props = parse_proposals_from_text(text)
        assert props[0]["confidence"] == 0.5
        assert props[0]["sources_available"] == 0

### Интеграционный тест

**test_tool_loop_integration.py**
```python
@mark.integration
class TestToolLoopRealLLM:
    """Настоящий MiniMax с реальными storage tools."""
    
    def test_real_editor_analysis_loop():
        """Полный цикл: LLM анализирует хранилище через tools."""
        storage = JsonlStorage(data_dir="/app/data")
        tools = create_storage_tools(storage)
        llm = MiniMaxProvider(api_key=REAL_KEY, model="MiniMax-M2.5")
        loop = ToolUseLoop(llm, tools, max_rounds=6)
        
        result = loop.run(
            user_message="Проанализируй тему Arctic permafrost methane emissions. "
                       "Какие есть кластеры статей? Какие годы покрыты? "
                       "Чего не хватает?",
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            temperature=0.1,
        )
        
        assert result.stop_reason == "end_turn"
        assert result.total_rounds >= 2  # минимум 1 tool call + финал
        assert len(result.tool_calls_made) >= 1  # хотя бы один tool был вызван
        assert len(result.content) > 50  # осмысленный ответ
        
        # Проверяем что были вызваны правильные tools
        called_names = [tc["name"] for tc in result.tool_calls_made]
        assert any(n in called_names for n in 
                  ["count_by_cluster", "get_time_range", "search_storage"])
    
    def test_real_proposal_generation():
        """Генерация предложений статей."""
        storage = JsonlStorage(data_dir="/app/data")
        tools = create_storage_tools(storage)
        llm = MiniMaxProvider(api_key=REAL_KEY, model="MiniMax-M2.7")
        loop = ToolUseLoop(llm, tools, max_rounds=8)
        
        result = loop.run(
            user_message="Предложи 3 варианта статей на основе анализа.",
            system_prompt=PROPOSAL_SYSTEM_PROMPT,
            temperature=0.25,
        )
        
        assert result.stop_reason == "end_turn"
        proposals = parse_proposals_from_text(result.content)
        assert len(proposals) >= 1  # хотя бы одно предложение
        if proposals:
            assert "title" in proposals[0]
            assert "thesis" in proposals[0]
```

## Файлы стадии S2

| Файл | Действие | Строк |
|------|----------|-------|
| `engine/llm/tool_loop.py` | Создать | ~200 |
| `engine/llm/response_parser.py` | Создать | ~100 |
| `engine/prompts/editor_prompts.py` | Создать | ~80 |
| `tests/test_tool_loop.py` | Создать | ~300 |
| `tests/test_response_parser.py` | Создать | ~100 |
| `tests/test_tool_loop_integration.py` | Создать | ~120 |

**Итого:** ~900 строк
