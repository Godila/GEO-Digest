# Стадия S1: Tool Executor (Storage Tools)

## Цель
Реализовать все tools которые Editor Agent будет использовать для работы с хранилищем.
Каждый tool — это детерминистичная функция (без LLM!), которая читает из JsonlStorage
и возвращает структурированные данные.

## Архитектура

```
ToolExecutor
├── _tool_search_storage(query, year_from, source, max_results)
│   → BM25-like keyword search → top N статей с DOI/title/abstract/year/source
│
├── _tool_get_article_detail(doi)
│   → Полная инфо по DOI: title, abstract, year, journal, authors, keywords, url
│
├── _tool_validate_doi(doi)
│   → {valid: True/False, doi, title?}
│
├── _tool_count_by_cluster(topic, top_n)
│   → Кластеризация по ключевым словам заголовков
│   → {total_relevant, cluster_count, clusters: [{theme, count}]}
│
├── _tool_check_existing_articles(title_idea)
│   → Поиск похожих уже написанных статей в /app/data/articles/
│   → {existing_count, articles: [{file, similarity, preview}]}
│
├── _tool_get_time_range(query)
│   → {min_year, max_year, total, by_year: {year: count}}
│
└── _tool_propose_scout_query(gap_description, already_have_dois)
    → Формирует оптимальный поисковый запрос для Scout
    → {suggested_query, rationale, estimated_sources}
```

## Что делаем

### 1.1 Base Tool Registry
Файл: `engine/tools/base.py`

```python
class ToolRegistry:
    """Регистрирует и выполняет tools."""
    
    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
    
    def register(self, name: str, handler: Callable, schema: dict) -> None:
        self._tools[name] = ToolDef(name=name, handler=handler, schema=schema)
    
    def get_schemas(self) -> list[dict]:
        return [t.schema for t in self._tools.values()]
    
    def execute(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            # Валидация args против schema (опционально, basic)
            result = tool.handler(**args)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": name})
    
    def list_tools(self) -> list[str]:
        return list(self._tools.keys())
```

### 1.2 Storage Tools
Файл: `engine/tools/storage_tools.py`

```python
class StorageTools:
    """Все tools для работы с JsonlStorage."""
    
    def __init__(self, storage: JsonlStorage):
        self.storage = storage
        # Кэш статей (lazy load)
        self._articles: list[Article] | None = None
    
    def _load_articles(self) -> list[Article]:
        if self._articles is None:
            self._articles = self.storage.load_all_articles()
        return self._articles
    
    # ── search_storage ──
    def search_storage(self, query: str, year_from: int = None,
                       source: str = None, max_results: int = 20) -> dict:
        """BM25-like keyword search."""
        articles = self._load_articles()
        query_words = self._tokenize(query)
        scored = []
        for a in articles:
            text = f"{a.title} {a.abstract or ''}".lower()
            score = sum(w in text for w in query_words)
            if score > 0 and (not year_from or (a.year and a.year >= year_from)):
                if not source or getattr(a, 'source', '') == source:
                    scored.append((score, a))
        scored.sort(key=lambda x: x[0], reverse=True)
        return {
            "total_found": len(scored),
            "returned": len(scored[:max_results]),
            "articles": [self._article_to_dict(a) for _, a in scored[:max_results]]
        }
    
    # ── get_article_detail ──
    def get_article_detail(self, doi: str) -> dict:
        """Полная информация о статье."""
        for a in self._load_articles():
            if a.doi == doi:
                d = self._article_to_dict(a)
                d["found"] = True
                return d
        return {"found": False, "doi": doi}
    
    # ── validate_doi ──
    def validate_doi(self, doi: str) -> dict:
        for a in self._load_articles():
            if a.doi == doi:
                return {"valid": True, "doi": doi, "title": a.title}
        return {"valid": False, "doi": doi}
    
    # ── count_by_cluster ──
    def count_by_cluster(self, topic: str, top_n: int = 10) -> dict:
        """Простая кластеризация по ключевым словам."""
        ...
    
    # ── check_existing_articles ──
    def check_existing_articles(self, title_idea: str) -> dict:
        """Проверка на дубликаты с уже написанными статьями."""
        ...
    
    # ── get_time_range ──
    def get_time_range(self, query: str = None) -> dict:
        """Временное распределение."""
        ...
    
    # Helpers
    def _article_to_dict(self, a: Article) -> dict: ...
    def _tokenize(self, text: str) -> set[str]: ...

def create_storage_tools(storage: JsonlStorage) -> ToolRegistry:
    """Создаёт registry со всеми storage tools."""
    tools = StorageTools(storage)
    reg = ToolRegistry()
    reg.register("search_storage", tools.search_storage, SEARCH_STORAGE_SCHEMA)
    reg.register("get_article_detail", tools.get_article_detail, ARTICLE_DETAIL_SCHEMA)
    reg.register("validate_doi", tools.validate_doi, VALIDATE_DOI_SCHEMA)
    reg.register("count_by_cluster", tools.count_by_cluster, COUNT_CLUSTER_SCHEMA)
    reg.register("check_existing_articles", tools.check_existing_articles, CHECK_EXISTING_SCHEMA)
    reg.register("get_time_range", tools.get_time_range, TIME_RANGE_SCHEMA)
    return reg
```

### 1.3 JSON Schemas для всех tools
Каждый tool имеет Anthropic-compatible input_schema.

## Acceptance Criteria S1

- [ ] `search_storage` находит статьи по ключевым словам (проверка на fixture'ах)
- [ ] `search_storage` фильтрует по year_from и source
- [ ] `search_storage` ограничивает max_results
- [ ] `get_article_detail` возвращает полную инфо по существующему DOI
- [ ] `get_article_detail` возвращает `{found: false}` для несуществующего DOI
- [ ] `validate_doi` возвращает `{valid: true}` для реального DOI
- [ ] `validate_doi` возвращает `{valid: false}` для фейкового DOI
- [ ] `count_by_cluster` группирует статьи по тематикам (минимум 2 кластера на fixture'е)
- [ ] `check_existing_articles` находит совпадения если есть файлы
- [ ] `get_time_range` возвращает корректный min/max год
- [ ] `get_time_range` возвращает by_year распределение
- [ ] `ToolRegistry.execute()` обрабатывает неизвестный tool (возвращает error)
- [ ] `ToolRegistry.execute()` обрабатывает ошибку в handler (возвращает error)
- [ ] Все tools возвращают валидный JSON (json.dumps не падает)

## Тесты S1

### Unit тесты

**test_storage_tools.py**
```python
class TestSearchStorage:
    @fixture
    def storage(tmp_path):  # создаём временный JSONL с fixture данными
        ...
    
    def test_basic_search_finds_matches(storage):
        r = search_storage("permafrost methane")
        assert r["total_found"] > 0
        assert r["returned"] > 0
        assert all("doi" in a for a in r["articles"])
    
    def test_search_no_matches(storage):
        r = search_storage("quantum entanglement dark matter")
        assert r["total_found"] == 0
        assert r["returned"] == 0
    
    def test_search_year_filter(storage):
        r = search_storage("methane", year_from=2023)
        for a in r["articles"]:
            assert a["year"] >= 2023
    
    def test_search_source_filter(storage):
        r = search_storage("methane", source="openalex")
        for a in r["articles"]:
            assert a["source"] == "openalex"
    
    def test_search_max_results(storage):
        r = search_storage("methane", max_results=3)
        assert r["returned"] <= 3
    
    def test_search_ranking_relevant_first(storage):
        r = search_storage("permafrost carbon feedback")
        if r["total_found"] >= 2:
            # Первая статья должна быть релевантнее (больше keyword matches)
            assert len(r["articles"][0]["title"]) > 0

class TestGetArticleDetail:
    def test_existing_doi(storage):
        # Берём DOI из fixture
        doi = FIXTURE_DOIS[0]
        r = get_article_detail(doi)
        assert r["found"] is True
        assert r["doi"] == doi
        assert "title" in r
        assert "abstract" in r
    
    def test_nonexistent_doi(storage):
        r = get_article_detail("10.fake/fake123")
        assert r["found"] is False

class TestValidateDoi:
    def test_valid_doi(storage):
        r = validate_doi(FIXTURE_DOIS[0])
        assert r["valid"] is True
        assert "title" in r
    
    def test_invalid_doi(storage):
        r = validate_doi("10.nonexistent/abc")
        assert r["valid"] is False

class TestCountByCluster:
    def test_returns_clusters(storage):
        r = count_by_cluster("Arctic methane")
        assert r["cluster_count"] > 0
        assert r["total_relevant"] > 0
        assert len(r["clusters"]) > 0
        for c in r["clusters"]:
            assert "theme" in c
            assert "count" in c
            assert c["count"] > 0
    
    def test_top_n_limits(storage):
        r = count_by_cluster("methane", top_n=2)
        assert len(r["clusters"]) <= 2

class TestGetTimeRange:
    def test_basic_range(storage):
        r = get_time_range()
        assert "min_year" in r
        assert "max_year" in r
        assert r["max_year"] >= r["min_year"]
        assert r["total"] > 0
        assert len(r["by_year"]) > 0

class TestCheckExistingArticles:
    def test_no_existing(tmp_path):
        # Пустая директория articles
        r = check_existing_articles("test")
        assert r["existing_count"] == 0
    
    def test_finds_similar(tmp_path):
        # Создаём файл с похожим содержанием
        (tmp_path / "articles").mkdir()
        (tmp_path / "articles" / "arctic_methane.md").write_text(
            "Article about Arctic permafrost methane emissions..."
        )
        r = check_existing_articles("Arctic methane overview")
        assert r["existing_count"] >= 1

class TestToolRegistry:
    def test_execute_success(registry_with_storage):
        r = registry.execute("validate_doi", {"doi": FIXTURE_DOIS[0]})
        data = json.loads(r)
        assert data["valid"] is True
    
    def test_execute_unknown_tool(registry):
        r = registry.execute("nonexistent", {})
        data = json.loads(r)
        assert "error" in data
    
    def test_execute_invalid_args(registry):
        r = registry.execute("validate_doi", {})  # missing required 'doi'
        data = json.loads(r)
        # Должен вернуть error (missing required param)
        assert "error" in data or data.get("valid") is False

class TestSchemaValidation:
    def test_all_tools_have_valid_schema():
        reg = create_storage_tools(mock_storage)
        schemas = reg.get_schemas()
        assert len(schemas) == 6  # 6 storage tools
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "input_schema" in s
            assert "type" in s["input_schema"]
            assert "properties" in s["input_schema"]
            assert "required" in s["input_schema"]

### Интеграционный тест

**test_storage_tools_integration.py**
```python
@mark.integration
class TestStorageToolsRealData:
    """Тесты на реальных данных из Docker контейнера."""
    
    def test_real_storage_search():
        storage = JsonlStorage(data_dir="/app/data")
        tools = StorageTools(storage)
        r = tools.search_storage("permafrost methane", max_results=5)
        assert r["total_found"] > 0
        assert r["returned"] > 0
    
    def test_real_storage_count_clusters():
        storage = JsonlStorage(data_dir="/app/data")
        tools = StorageTools(storage)
        r = tools.count_by_cluster("Arctic")
        assert r["cluster_count"] >= 3  # минимум 3 кластера по Arctic
    
    def test_real_time_range():
        storage = JsonlStorage(data_dir="/app/data")
        tools = StorageTools(storage)
        r = tools.get_time_range()
        assert 2019 <= r["min_year"] <= 2025
        assert r["max_year"] >= 2023
```

## Файлы стадии S1

| Файл | Действие | Строк |
|------|----------|-------|
| `engine/tools/base.py` | Создать | ~80 |
| `engine/tools/storage_tools.py` | Создать | ~280 |
| `tests/test_storage_tools.py` | Создать | ~300 |
| `tests/test_tool_registry.py` | Создать | ~80 |
| `tests/test_schema_validation.py` | Создать | ~40 |
| `tests/fixtures/sample_articles.jsonl` | Создать | ~100 |

**Итого:** ~880 строк
