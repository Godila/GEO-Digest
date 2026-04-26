# Стадия S8: Интеграция Графа Знаний в Editor Agent

## Цель

Включить граф знаний (Cytoscape knowledge graph с LLM-semantic edges) в tool-use pipeline
Editor Agent, чтобы при анализе тем агент видел **структуру связей** между статьями,
а не только плоский список. Ключевой use-case: **кросс-тематический поиск мостов**
между областями знаний.

## Почему это важно

```
БЕЗ графа:                    С графом:
Editor.search("permafrost")   Editor.search("permafrost")
  → [статьи A,B,C,D]            → [статьи A,B,C,D]
  → flat list, sorted by score     ↓
                                   Editor.graph_cross_topic(
                                     "permafrost", "landslide")
                                   → [статья X — мост!]
                                   → Editor.graph_hubs("permafrost")
                                   → [статья Y — hub, degree=19]
                                   
Результат:                       Результат:
  proposals на основе            proposals с учётом
  текстового поиска              структурной роли статей
  (может пропустить мосты)       (видит скрытые связи)
```

## Предусловия (уже выполнены)

- [x] S0-S7: Tool-Use Editor Agent полностью реализован (263 теста)
- [x] Phase 1: Баги графа исправлены (A1-A5)
- [x] Phase 2: Graph Analytics (PageRank, Betweenness, Louvain) написаны

## Dependency Graph S8

```
S8.1 Graph Tools (engine/tools/graph_tools.py)
  ├── Читает graph_data.json через StorageBackend.load_graph()
  ├── 7 tools: neighbors, path, hubs, clusters, cross_topic, centrality, stats
  └── Регистрируется в ToolRegistry
        │
        ▼
S8.2 Editor Agent Integration
  ├── GraphTools добавляются к StorageTools в ToolRegistry
  ├── EditorAgent автоматически получает graph tools в своём tool-use loop
  └── System prompt обновляется с описанием graph capabilities
        │
        ▼
S8.3 Orchestrator v2 Update
  ├── run_editing_phase() вызывает graph_cross_topic() после search
  ├── Proposals обогащаются graph_centrality данными
  └── develop phase использует graph_neighbors для глубины
        │
        ▼
S8.4 UI + API + Тесты + Деплой
  ├── Dashboard proxy для graph endpoints (если нужно)
  ├── Тесты: test_graph_tools.py (~25 тестов)
  ├── Тесты: test_graph_integration.py (~10 тестов)
  ├── Docker rebuild + push
  └── Коммит на main
```

## S8.1: Graph Tools

### Файл: `engine/tools/graph_tools.py`

```python
"""Graph Tools — LLM-callable functions for knowledge graph queries.

Позволяют Editor Agent запрашивать структурную информацию из графа знаний:
  - Какие статьи связаны с данной?
  - Какой кратчайший путь между двумя работами?
  - Какие статьи — ключевые hubs/мосты?
  - Какие сообщества (кластеры) есть в данных?
  - Какие статьи связывают две темы? (killer feature)

Каждый tool читает graph_data.json через storage.load_graph().
"""

GRAPH_NEIGHBORS_SCHEMA = {
    "type": "object",
    "properties": {
        "doi": {"type": "string", "description": "DOI статьи для поиска соседей"},
        "depth": {"type": "integer", "default": 1,
                  "description": "Глубина поиска (1=прямые соседи, 2=соседи соседей)"},
    },
    "required": ["doi"],
}

GRAPH_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "doi_a": {"type": "string", "description": "DOI первой статьи"},
        "doi_b": {"type": "string", "description": "DOI второй статьи"},
        "max_depth": {"type": "integer", "default": 4,
                     "description": "Максимальная длина пути"},
    },
    "required": ["doi_a", "doi_b"],
}

GRAPH_HUBS_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_filter": {"type": "string", "description":
          "Фильтр по теме (опционально). Пусто = все хабы."},
        "min_degree": {"type": "integer", "default": 5,
                       "description": "Минимальная степень узла"},
        "limit": {"type": "integer", "default": 10,
                  "description": "Максимум результатов"},
    },
}

GRAPH_CLUSTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "min_size": {"type": "integer", "default": 2,
                     "description": "Минимальный размер кластера"},
    },
}

GRAPH_CROSS_TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_a": {"type": "string", "description": "Первая тема (англ или рус)"},
        "topic_b": {"type": "string", "description": "Вторая тема (англ или рус)"},
        "limit": {"type": "integer", "default": 10,
                  "description": "Максимум мост-статей"},
    },
    "required": ["topic_a", "topic_b"],
}

GRAPH_CENTRALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "doi": {"type": "string", "description": "DOI статьи"},
    },
    "required": ["doi"],
}

GRAPH_STATS_SCHEMA = {
    "type": "object",
    "properties": {},
}
```

### 7 Tools:

| # | Name | Описание | Возвращает |
|---|------|----------|------------|
| 1 | `graph_neighbors` | Найти связанные статьи вокруг данной | Список статей + типы рёбер |
| 2 | `graph_path` | Кратчайший путь между двумя работами | Путь (hops) + отношения |
| 3 | `graph_hubs` | Ключевые работы (высокий PageRank/degree) | Top-N hubs |
| 4 | `graph_clusters` | Сообщества/кластеры (Louvain) | Список кластеров + участники |
| 5 | `graph_cross_topic` | Мосты между двумя темами ⭐ | Статьи на стыке тем |
| 6 | `graph_centrality` | Важность узла в сети | PR + betweenness + community |
| 7 | `graph_stats` | Сводка по графу | Размер, плотность, метрики |

### Регистрация:

```python
# В editor.py или при создании ToolRegistry:
from engine.tools.graph_tools import create_graph_tools

storage = get_storage()
registry = create_storage_tools(storage)
graph_registry = create_graph_tools(storage)
# Merge into main registry
for name in graph_registry.list_tools():
    registry.register(
        name=name,
        handler=graph_registry.get(name),
        schema=graph_registry.get_schema(name)["input_schema"],
        description=graph_registry.get_schema(name)["description"],
    )
```

## S8.2: Editor Agent Integration

### Изменения в `engine/agents/editor.py`:

1. **Импорт и регистрация GraphTools** в `_build_tool_registry()`
2. **Обновление system prompt** — добавить описание graph tools
3. **Enrichment proposals** — после генерации, дополнить proposal'ы centrality данными

### Обновление prompts (`engine/prompts/editor_prompts.py`):

Добавить в доступные инструменты:
```
AVAILABLE TOOLS (14 total):
  
Storage Tools (7):
  search_articles, get_article_detail, validate_doi, find_similar_existing,
  cluster_by_subtopic, count_storage_stats, explore_domain
  
Graph Tools (7):
  graph_neighbors — найти статьи, связанные с данной (по семантическим рёбрам)
  graph_path — найти путь связи между двумя работами
  graph_hubs — найти наиболее влиятельные работы в области
  graph_clusters — получить автоматические сообщества статей
  graph_cross_topic — найти статьи на стыке двух тем (МОЩНЫЙ!)
  graph_centrality — узнать насколько статья важна в сети
  graph_stats — общая статистика графа знаний
```

## S8.3: Orchestrator v2 Update

### Изменения в `engine/orchestrator_v2.py`:

`run_editing_phase()` — после `editor.run()`:
```python
# Если граф доступен, обогащаем proposals graph data
if self._has_graph():
    for prop in job.editor_result.get("proposals", []):
        for ref in prop.get("key_references", []):
            doi = ref.replace("DOI:", "").strip()
            cent = self._graph_centrality(doi)
            if cent:
                ref_data = {"doi": doi, **cent}
                prop.setdefault("graph_context", []).append(ref_data)
    
    # Находим кросс-тематические мосты
    bridges = self._find_cross_topic_bridges(job.topic)
    if bridges:
        job.editor_result["cross_topic_bridges"] = bridges
```

## S8.4: Тесты + Деплой

### Тестовые файлы:

**tests/test_graph_tools.py** (~25 тестов):
- TestGraphNeighbors: depth=1, depth=2, nonexistent DOI, empty graph
- TestGraphPath: direct connection, indirect path, no path, same node
- TestGraphHubs: top N, topic filter, min_degree filter, empty
- TestGraphClusters: basic clusters, min_size filter, article-only
- TestGraphCrossTopic: bridge detection, same topic, nonexistent topics
- TestGraphCentrality: hub node, peripheral node, nonexistent
- TestGraphStats: with/without graph, metadata correctness

**tests/test_graph_integration.py** (~10 тестов):
- TestEditorWithGraphTools: editor.run() вызывает graph tools
- TestGraphToolRegistration: все 7 tools зарегистрированы
- TestOrchestratorGraphEnrichment: proposals обогащены centrality
- TestCrossTopicDiscovery: мосты найдены между темами

### Acceptance Criteria S8:

- [ ] Все 7 graph tools работают через ToolRegistry.execute()
- [ ] graph_neighbors возвращает BFS подграф с правильной глубиной
- [ ] graph_path находит кратчайший путь (BFS)
- [ ] graph_hubs возвращает топ по PageRank/degree
- [ ] graph_clusters возвращает Louvain сообщества
- [ ] graph_cross_topic находит статьи на стыке двух тем
- [ ] graph_centrality возвращает PR + betweenness + community ID
- [ ] graph_stats возвращает корректную metadata
- [ ] EditorAgent имеет доступ ко всем graph tools
- [ ] Orchestrator обогащает proposals graph данными
- [ ] Все тесты проходят (цель: +35 тестов к существующим 263)
- [ ] Docker rebuild успешен, UI работает
- [ ] Коммит + push на main

## Файлы стадии S8

| Файл | Действие | Строк |
|------|----------|-------|
| `scripts/graph_analytics.py` | NEW (Phase 2, уже создан) | ~400 |
| `scripts/build_graph.py` | MODIFY (Phase 1+2 fixes, уже изменён) | +100 |
| `engine/tools/graph_tools.py` | NEW (S8.1) | ~350 |
| `engine/agents/editor.py` | MODIFY (S8.2: регистрация tools) | +20 |
| `engine/prompts/editor_prompts.py` | MODIFY (S8.2: обновить prompt) | +15 |
| `engine/orchestrator_v2.py` | MODIFY (S8.3: enrichment) | +40 |
| `worker/dal.py` | MODIFY (Phase 1 A2 fix, уже изменён) | +5 |
| `dashboard/templates/index.html` | MODIFY (Phase 1 A5 fix, уже изменён) | +10 |
| `tests/test_graph_tools.py` | NEW (S8.4) | ~400 |
| `tests/test_graph_integration.py` | NEW (S8.4) | ~200 |

**Итого:** ~1500 строк нового кода + модификации
