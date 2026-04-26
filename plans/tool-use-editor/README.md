# Tool-Use Editor Agent — План реализации

## Цель
Переделать архитектуру GEO-Digest с linear pipeline (Scout→Reader→Writer→Reviewer) на **Tool-Use Editor Agent** — автономного агента который:
1. Сам анализирует хранилище через tool calls (не скармливаем всё в prompt)
2. Сам решает какие данные ему нужны
3. Предлагает варианты статей пользователю на основе реальных фактов из базы
4. Поддерживает итеративную доработку с пользователем
5. Делегирует Writer/Reviewer только на финальном этапе

## Почему Tool-Use
- **Минимизация галлюцинаций**: LLM не генерирует факты — запрашивает их через tools
- **Эффективное контекстное окно**: LLM получает только нужные данные, не 50K tokens сырья
- **Отладочность**: каждый tool call виден в логах
- **MiniMax подтверждённо поддерживает**: Anthropic-compatible API с `tools` + `tool_use` + `tool_result`

## Текущий стек
- MiniMax M2.5 / M2.7 (Anthropic-compatible API) ✅ tool use поддержан
- JsonlStorage (181 статья, ~50K tokens сырых данных)
- Flask dashboard + FastAPI worker
- Docker deployment

## Архитектура до / после

```
ДО (prompt-only):                    ПОСЛЕ (tool-use):
┌──────────────┐                     ┌──────────────────────┐
│ User → topic  │                     │ User → topic/domain   │
│      ↓        │                     │         ↓             │
│ Scout (тупой) │                     │ Editor Agent          │
│      ↓        │                     │  ├─ tool: search_storage
│ [выбрать]     │                     │  ├─ tool: count_by_cluster
│      ↓        │                     │  ├─ tool: get_time_range
│ Reader        │                     │  ├─ tool: validate_doi
│      ↓        │                     │  └─ tool: check_existing
│ [одобрить]    │                     │         ↓             │
│ Writer        │                     │ Proposals → User picks │
│      ↓        │                     │         ↓             │
│ Reviewer      │                     │ Develop (iterative)   │
│      ↓        │                     │         ↓             │
│ DONE          │                     │ Writer + Reviewer     │
└──────────────┘                     └──────────────────────┘
```

## Стадии реализации

| Стадия | Что | Файлов | Сложность | Зависит от |
|--------|-----|--------|-----------|------------|
| S0 | Инфраструктура | ~3 | Лёгкая | Nothing |
| S1 | Tool Executor | ~4 | Средняя | S0 |
| S2 | LLM Tool Use слой | ~3 | Средняя | S0 |
| S3 | Editor Agent (core) | ~5 | Высокая | S1, S2 |
| S4 | API Endpoints | ~3 | Средняя | S3 |
| S5 | UI (вкладка Редактор) | ~6 | Высокая | S4 |
| S6 | Orchestrator refactor | ~3 | Средняя | S3 |
| S7 | E2E + дебаггинг | ~4 | Средняя | S5, S6 |

**Итого:** ~31 файл, ~7-10 дней работы при полной загрузке

## Критерии успеха каждой стадии
Каждая стадия имеет чёткие acceptance criteria и автоматические тесты.
Переход к следующей стадии ТОЛЬКО после прохождения всех тестов текущей.
