# Tool-Use Editor Agent — Сводный план

## Обзор

Полная реализация tool-use архитектуры для GEO-Digest в 8 стадиях (S0-S7).
Каждая стадия имеет чёткие acceptance criteria, unit + integration тесты,
и может разрабатываться независимо (с учётом зависимостей).

## Dependency Graph

```
S0 (Инфраструктура)
├── S1 (Tool Executor) ──┐
├── S2 (Tool-Use Loop) ──┤
│                        ├── S3 (Editor Agent Core) ──┬── S4 (API Endpoints) ──┬── S5 (UI)
│                        │                            │                         │
│                        │                            ├── S6 (Orchestrator v2) ──┘
│                        │                            │
│                        │                            └────────────────────────┤
│                        │                                                   │
│                        └──────────────────────────────────────────────────┘
│                                                                          │
│                                                                          └── S7 (E2E + Финал)
```

**Параллельные потоки:**
- Поток A: S0 → S1 → S3 → S6 (backend core)
- Поток B: S0 → S2 → S3 → S4 → S5 (full stack)
- Поток C: S7 (зависит от всех)

## Quick Start по стадиям

### Начать с S0 (базовая инфраструктура)
```bash
cd ~/.hermes/geo_digest
# Читаем план:
cat plans/tool-use-editor/S0-infrastructure.md
# Реализуем:
# 1. Добавить tool_complete() в base.py
# 2. Реализовать в minimax.py (Anthropic /v1/messages API)
# 3. Создать engine/tools/base.py (ToolRegistry)
# 4. Тесты: test_llm_tool_interface.py, test_tool_registry.py
# 5. Запуск: pytest tests/test_llm_tool_interface.py tests/test_tool_registry.py -v
```

### После S0 — параллельно S1 и S2
```bash
# S1: Storage Tools
cat plans/tool-use-editor/S1-tool-executor.md
# Реализуем storage_tools.py с 7 tools
# Тесты: test_storage_tools.py, test_schema_validation.py

# S2: Tool-Use Loop  
cat plans/tool-use-editor/S2-tool-loop.md
# Реализуем tool_loop.py + response_parser.py + prompts
# Тесты: test_tool_loop.py, test_response_parser.py
```

### После S1+S2 — S3 (Editor Agent)
```bash
cat plans/tool-use-editor/S3-editor-agent.md
# Реализуем editor.py с 5 фазами + checkpoint'ами
# Тесты: test_editor_agent.py, test_editor_integration.py
```

### После S3 — параллельно S4 и S6
```bash
# S4: API Endpoints
cat plans/tool-use-editor/S4-api-endpoints.md
# 8 endpoints в worker/server.py + proxy в dashboard/app.py

# S6: Orchestrator v2
cat plans/tool-use-editor/S6-orchestrator-refactor.md
# State machine вместо linear pipeline
```

### После S4+S6 — S5 (UI)
```bash
cat plans/tool-use-editor/S5-ui-editor.md
# Переработка вкладки Агентов в index.html
# Карточки proposals, detail panel, progress bar
```

### Финал — S7
```bash
cat plans/tool-use-editor/S7-e2e-finalize.md
# E2E тесты, дебаггинг, коммит, push
```

## Ключевые решения (уже принятые)

| Решение | Значение | Почему |
|---------|----------|--------|
| LLM провайдер | MiniMax M2.5/M2.7 | Дешёвый, подтверждённо поддерживает tool use |
| Tool Use API | Anthropic `/v1/messages` | Подтверждён работает на всех моделях |
| Модель для анализа | M2.5 (fast, 2.1с/5K tok) | Анализ не требует креативности |
| Модель для генерации | M2.7 (stronger, 6.4с/5K tok) | Генерация требует качества |
| Max tool rounds | 8 | Баланс между глубиной и стоимостью |
| Checkpoint формат | JSON файлы | Простой, читаемый, git-friendly |
| Storage backend | JsonlStorage (уже есть) | Не меняем то что работает |
| UI фреймворк | Vanilla JS (уже есть) | Не добавляем React/Vue |

## Риски и митигации

| Риск | Вероятность | Влияние | Митигация |
|------|------------|---------|-----------|
| MiniMax tool use нестабилен | Низкая | Высокое | Retry logic уже есть; fallback к prompt-only |
| LLM генерирует невалидный JSON | Средняя | Среднее | parse_proposals_from_text с 3 стратегиями + fallback |
| Tool loop бесконечный цикл | Низкая | Среднее | MAX_ROUNDS=8 + warning |
| Memory leak в Docker | Низкая | Среднее | Мониторинг через docker stats |
| UI сложнее чем ожидалось | Средняя | Низкое | MVP сначала, потом улучшения |
| Пользователь не понимает tool-use UX | Средняя | Низкое | Progressive disclosure: показываем только нужное |

## Метрики успеха

**Количественные:**
- Время анализа темы: < 60 сек (M2.5)
- Время генерации предложений: < 120 сек (M2.7)
- Полный цикл (анализ→выбор→статья): < 5 мин
- % proposals с confidence > 0.5: > 70%
- % DOI валидных: > 95% (vs ~60% в prompt-only)
- Unit test coverage: > 80%
- E2E pass rate: 100%

**Качественные:**
- Нет галлюцинированных DOI (основное улучшение vs старой архитектуры)
- Каждое утверждение в proposal основано на данных из базы
- Пользователь видит process (какие tools вызывались)
- Можно возобновить после перезапуска
- Итеративная доработка с пользователем

## Файловая структура результата

```
engine/
├── llm/
│   ├── base.py              # +tool_complete()
│   ├── minimax.py           # +Anthropic /v1/messages tool use
│   ├── openai_compat.py     # +tool_complete()
│   └── tool_loop.py         # NEW: ToolUseLoop class
│   └── response_parser.py   # NEW: parse_proposals_from_text()
├── tools/
│   ├── __init__.py          # NEW
│   ├── base.py              # NEW: ToolRegistry, BaseTool
│   └── storage_tools.py     # NEW: 7 storage tools
├── agents/
│   ├── editor.py            # NEW: EditorAgent (core)
│   ├── scout.py             # unchanged
│   ├── reader.py            # unchanged
│   ├── writer.py            # unchanged
│   └── reviewer.py          # unchanged
├── orchestrator_v2.py       # NEW: state machine
├── prompts/
│   └── editor_prompts.py    # NEW: system prompts for editor
└── schemas.py               # +EditorState, ArticleProposal, etc.

worker/
└── server.py                # +8 editor/pipeline endpoints

dashboard/
├── app.py                   # +editor proxy routes
└── templates/
    └── index.html           # rewritten agents tab

tests/
├── test_llm_tool_interface.py      # S0
├── test_tool_registry.py           # S0
├── test_minimax_tool_use_integration.py  # S0
├── test_storage_tools.py           # S1
├── test_schema_validation.py       # S1
├── test_tool_loop.py               # S2
├── test_response_parser.py         # S2
├── test_tool_loop_integration.py   # S2
├── test_editor_agent.py            # S3
├── test_editor_integration.py      # S3
├── test_editor_api.py              # S4
├── test_editor_api_integration.py  # S4
├── test_orchestrator_v2.py         # S6
├── test_e2e_full_pipeline.py       # S7
├── test_e2e_browser.py             # S7
└── fixtures/
    └── sample_articles.jsonl       # S0-S1

plans/
└── tool-use-editor/
    ├── README.md                   # Этот файл
    ├── S0-infrastructure.md
    ├── S1-tool-executor.md
    ├── S2-tool-loop.md
    ├── S3-editor-agent.md
    ├── S4-api-endpoints.md
    ├── S5-ui-editor.md
    ├── S6-orchestrator-refactor.md
    └── S7-e2e-finalize.md
```
