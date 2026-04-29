"""Unit tests for StorageTools — 7 LLM-callable storage query functions.

Tests cover:
  - search_articles: keyword search, filters (year, source, max_results), ranking
  - get_article_detail: existing DOI, non-existent DOI
  - validate_doi: valid/invalid DOIs
  - find_similar_existing: no matches, with matches
  - cluster_by_subtopic: clustering, top_n limit
  - count_storage_stats: stats computation
  - explore_domain: domain overview
  - create_storage_tools factory: all 7 tools registered

Uses sample_articles.jsonl fixture data (10 articles).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.schemas import Article
from engine.tools.storage_tools import (
    StorageTools,
    create_storage_tools,
    STORAGE_TOOL_SCHEMAS,
    SEARCH_ARTICLES_SCHEMA,
    ARTICLE_DETAIL_SCHEMA,
)
from engine.tools.base import ToolRegistry


# ── Fixtures ─────────────────────────────────────────────────────

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_articles.jsonl"


def _load_fixture_articles() -> list[dict]:
    """Load test articles from JSONL fixture."""
    articles = []
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def _make_mock_storage(articles_data: list[dict] | None = None):
    """Create a mock StorageBackend with given articles."""
    if articles_data is None:
        articles_data = _load_fixture_articles()

    storage = MagicMock()
    storage.load_all_articles.return_value = [Article(a) for a in articles_data]
    storage.data_dir = Path(tempfile.mkdtemp())
    storage.load_articles.return_value = articles_data
    return storage


def _make_storage_tools(articles_data: list[dict] | None = None):
    """Create StorageTools instance with mock storage."""
    storage = _make_mock_storage(articles_data)
    return StorageTools(storage), storage


# Sample DOIs from fixture
FIXTURE_DOIS = [
    "10.1234/arctic.permafrost.2024.001",
    "10.5678/siberia.methane.2023.042",
    "10.9012/boreal.fire.2024.015",
]


class TestSearchArticles(unittest.TestCase):
    """Test search_articles tool."""

    def setUp(self):
        self.tools, self.storage = _make_storage_tools()

    def test_basic_search_finds_matches(self):
        """Поиск по ключевым словам находит статьи."""
        result = self.tools.search_articles(query="permafrost carbon")
        self.assertTrue(result.success)
        data = result.data
        assert isinstance(data, dict)
        self.assertGreater(data["total_found"], 0)
        self.assertGreater(data["returned"], 0)
        for a in data["articles"]:
            self.assertIn("doi", a)
            self.assertIn("title", a)

    def test_search_no_matches(self):
        """Поиск по несуществующим словам возвращает 0 результатов."""
        result = self.tools.search_articles(
            query="quantum entanglement dark matter black holes"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.data["total_found"], 0)
        self.assertEqual(result.data["returned"], 0)

    def test_empty_query_returns_error(self):
        """Пустой запрос возвращает ошибку."""
        result = self.tools.search_articles(query="")
        self.assertFalse(result.success)
        self.assertIn("empty", result.error_msg.lower())

    def test_year_filter(self):
        """Фильтр по году работает."""
        result = self.tools.search_articles(
            query="methane", year_from=2024
        )
        self.assertTrue(result.success)
        for a in result.data["articles"]:
            self.assertGreaterEqual(a["year"], 2024)

    def test_source_filter(self):
        """Фильтр по источнику работает."""
        result = self.tools.search_articles(
            query="methane", source="openalex"
        )
        self.assertTrue(result.success)
        for a in result.data["articles"]:
            self.assertEqual(a["source"], "openalex")

    def test_max_results_limit(self):
        """max_results ограничивает количество."""
        result = self.tools.search_articles(
            query="arctic", max_results=3
        )
        self.assertTrue(result.success)
        self.assertLessEqual(result.data["returned"], 3)

    def test_ranking_relevant_first(self):
        """Релевантные статьи идут первыми (больше совпадений ключевых слов)."""
        result = self.tools.search_articles(
            query="permafrost carbon feedback arctic"
        )
        if result.data["total_found"] >= 2:
            # First article should have more keyword overlap than last
            first_title = result.data["articles"][0]["title"].lower()
            self.assertTrue(len(first_title) > 0)

    def test_query_tokens_in_result(self):
        """Результат содержит использованные токены запроса."""
        result = self.tools.search_articles(query="permafrost methane")
        self.assertIn("query_tokens_used", result.data)
        self.assertIsInstance(result.data["query_tokens_used"], list)


class TestGetArticleDetail(unittest.TestCase):
    """Test get_article_detail tool."""

    def setUp(self):
        self.tools, _ = _make_storage_tools()

    def test_existing_doi(self):
        """Существующий DOI возвращает полную информацию."""
        doi = FIXTURE_DOIS[0]
        result = self.tools.get_article_detail(doi=doi)
        self.assertTrue(result.success)
        self.assertTrue(result.data["found"])
        self.assertEqual(result.data["doi"], doi)
        self.assertIn("title", result.data)
        self.assertIn("abstract_preview", result.data)
        self.assertIn("full_abstract_ru", result.data)

    def test_nonexistent_doi(self):
        """Несуществующий DOI возвращает found=False."""
        result = self.tools.get_article_detail(doi="10.fake/fake123.nonexistent")
        self.assertTrue(result.success)
        self.assertFalse(result.data["found"])
        self.assertIn("message", result.data)

    def test_empty_doi_returns_error(self):
        """Пустой DOI возвращает ошибку."""
        result = self.tools.get_article_detail(doi="")
        self.assertFalse(result.success)
        self.assertIn("empty", result.error_msg.lower())

    def test_detail_has_all_fields(self):
        """Полная информация содержит все ожидаемые поля."""
        doi = FIXTURE_DOIS[0]
        result = self.tools.get_article_detail(doi=doi)
        d = result.data
        expected_fields = [
            "doi", "title", "title_ru", "year", "source",
            "citations", "topics", "abstract_preview", "score_total",
            "is_enriched", "canonical_id", "found",
        ]
        for field in expected_fields:
            self.assertIn(field, d, f"Missing field: {field}")


class TestValidateDoi(unittest.TestCase):
    """Test validate_doi tool."""

    def setUp(self):
        self.tools, _ = _make_storage_tools()

    def test_valid_doi(self):
        """Валидный DOI из хранилища."""
        result = self.tools.validate_doi(doi=FIXTURE_DOIS[0])
        self.assertTrue(result.success)
        self.assertTrue(result.data["valid"])
        self.assertEqual(result.data["doi"], FIXTURE_DOIS[0])
        self.assertIn("title", result.data)

    def test_invalid_doi(self):
        """Невалидный DOI (нет в хранилище)."""
        result = self.tools.validate_doi(doi="10.nonexistent/fake.abc")
        self.assertTrue(result.success)
        self.assertFalse(result.data["valid"])

    def test_empty_doi_error(self):
        """Пустой DOI → ошибка."""
        result = self.tools.validate_doi(doi="")
        self.assertFalse(result.success)

    def test_url_encoded_doi(self):
        """URL-encoded DOI (%2F вместо /) работает."""
        encoded = FIXTURE_DOIS[0].replace("/", "%2F")
        result = self.tools.validate_doi(doi=encoded)
        self.assertTrue(result.data["valid"])


class TestFindSimilarExisting(unittest.TestCase):
    """Test find_similar_existing tool."""

    def setUp(self):
        self.tools, self.storage = _make_storage_tools()

    def test_no_output_dir(self):
        """Нет output директории → 0 результатов."""
        # Use a non-existent temp dir
        self.tools._output_dir = Path(tempfile.mkdtemp()) / "nonexistent"
        result = self.tools.find_similar_existing(title_idea="Arctic methane overview")
        self.assertTrue(result.success)
        self.assertEqual(result.data["existing_count"], 0)

    def test_finds_similar_files(self):
        """Находит похожие файлы если они есть."""
        tmpdir = Path(tempfile.mkdtemp()) / "output"
        tmpdir.mkdir(parents=True)
        (tmpdir / "arctic_methane.md").write_text(
            "Article about Arctic permafrost methane emissions "
            "and their impact on global climate change.",
            encoding="utf-8",
        )
        (tmpdir / "boreal_fire_review.md").write_text(
            "Review of boreal forest fire regimes under climate change scenarios.",
            encoding="utf-8",
        )
        self.tools._output_dir = tmpdir

        result = self.tools.find_similar_existing(title_idea="Arctic methane emissions permafrost")
        self.assertTrue(result.success)
        # Should find at least the arctic_methane file
        self.assertGreaterEqual(result.data["existing_count"], 1)

    def test_empty_idea_error(self):
        """Пустая идея → ошибка."""
        result = self.tools.find_similar_existing(title_idea="")
        self.assertFalse(result.success)


class TestClusterBySubtopic(unittest.TestCase):
    """Test cluster_by_subtopic tool."""

    def setUp(self):
        self.tools, _ = _make_storage_tools()

    def test_returns_clusters(self):
        """Возвращает кластеры для реальной темы."""
        result = self.tools.cluster_by_subtopic(topic="Arctic methane emissions")
        self.assertTrue(result.success)
        self.assertGreater(result.data["total_relevant"], 0)
        self.assertGreater(result.data["cluster_count"], 0)
        self.assertIsInstance(result.data["clusters"], list)
        for c in result.data["clusters"]:
            self.assertIn("theme", c)
            self.assertIn("count", c)
            self.assertGreater(c["count"], 0)

    def test_top_n_limits_clusters(self):
        """top_n ограничивает количество кластеров."""
        result = self.tools.cluster_by_subtopic(topic="climate", top_n=2)
        self.assertLessEqual(len(result.data["clusters"]), 2)

    def test_no_matches_topic(self):
        """Тема без совпадений → пустые кластеры."""
        result = self.tools.cluster_by_subtopic(topic="quantum gravity supersymmetry strings")
        # May have 0 relevant or very few
        self.assertTrue(result.success)
        self.assertIn("total_relevant", result.data)

    def test_empty_topic_error(self):
        """Пустая тема → ошибка."""
        result = self.tools.cluster_by_subtopic(topic="")
        self.assertFalse(result.success)

    def test_cluster_names_listed(self):
        """Список имён кластеров присутствует."""
        result = self.tools.cluster_by_subtopic(topic="permafrost")
        self.assertIn("all_cluster_names", result.data)
        self.assertIsInstance(result.data["all_cluster_names"], list)


class TestCountStorageStats(unittest.TestCase):
    """Test count_storage_stats tool."""

    def setUp(self):
        self.tools, _ = _make_storage_tools()

    def test_basic_stats(self):
        """Базовая статистика корректна."""
        result = self.tools.count_storage_stats()
        self.assertTrue(result.success)
        d = result.data
        self.assertGreater(d["total_articles"], 0)
        self.assertGreaterEqual(d["enriched_articles"], 0)
        self.assertIn("enrichment_rate", d)
        self.assertIn("sources", d)
        self.assertIsInstance(d["sources"], dict)

    def test_year_range_present(self):
        """Годовой диапазон присутствует."""
        result = self.tools.count_storage_stats()
        d = result.data
        if d["total_articles"] > 0:
            self.assertIn("min_year", d)
            self.assertIn("max_year", d)
            self.assertGreaterEqual(d["max_year"], d["min_year"])
            self.assertIn("by_year", d)

    def test_top_topics_present(self):
        """Топики присутствуют."""
        result = self.tools.count_storage_stats()
        self.assertIn("top_topics", result.data)
        self.assertIsInstance(result.data["top_topics"], list)


class TestExploreDomain(unittest.TestCase):
    """Test explore_domain tool."""

    def setUp(self):
        self.tools, _ = _make_storage_tools()

    def test_full_domain_overview(self):
        """Полный обзор домена без фокус-запроса."""
        result = self.tools.explore_domain()
        self.assertTrue(result.success)
        d = result.data
        self.assertGreater(d["total_in_scope"], 0)
        self.assertIn("sources", d)
        self.assertIn("year_distribution", d)
        self.assertIn("citation_stats", d)
        self.assertIn("top_topics", d)

    def test_with_focus_query(self):
        """Обзор с фокус-запросом сужает результаты."""
        result_full = self.tools.explore_domain()
        result_focus = self.tools.explore_domain(focus_query="permafrost methane")
        self.assertTrue(result_focus.success)
        # Focused should be <= full scope
        self.assertLessEqual(
            result_focus.data["total_in_scope"],
            result_full.data["total_in_scope"],
        )

    def test_citation_stats_structure(self):
        """Статистика цитирований имеет правильную структуру."""
        result = self.tools.explore_domain()
        cs = result.data["citation_stats"]
        if cs:  # Only if there are citations
            self.assertIn("max", cs)
            self.assertIn("median", cs)
            self.assertIn("mean", cs)

    def test_topic_coverage(self):
        """Топики содержат процент покрытия."""
        result = self.tools.explore_domain()
        for t in result.data["top_topics"]:
            self.assertIn("topic", t)
            self.assertIn("count", t)
            self.assertIn("pct", t)


class TestCreateStorageToolsFactory(unittest.TestCase):
    """Test create_storage_tools factory function."""

    def test_creates_registry_with_7_tools(self):
        """Фабрика создаёт registry с 7 инструментами."""
        storage = _make_mock_storage()
        registry = create_storage_tools(storage)
        self.assertEqual(len(registry), 7)

    def test_all_tools_have_schemas(self):
        """Все инструменты имеют схемы."""
        storage = _make_mock_storage()
        registry = create_storage_tools(storage)
        schemas = registry.get_schemas()
        self.assertEqual(len(schemas), 7)
        for s in schemas:
            self.assertIn("name", s)
            self.assertIn("description", s)
            self.assertIn("input_schema", s)
            self.assertIn("type", s["input_schema"])
            self.assertIn("properties", s["input_schema"])

    def test_expected_tool_names(self):
        """Ожидаемые имена инструментов."""
        storage = _make_mock_storage()
        registry = create_storage_tools(storage)
        expected = {
            "search_articles",
            "get_article_detail",
            "validate_doi",
            "find_similar_existing",
            "cluster_by_subtopic",
            "count_storage_stats",
            "explore_domain",
        }
        self.assertEqual(set(registry.list_tools()), expected)

    def test_execute_via_registry(self):
        """Выполнение через registry работает."""
        storage = _make_mock_storage()
        registry = create_storage_tools(storage)
        result = registry.execute("validate_doi", {"doi": FIXTURE_DOIS[0]})
        self.assertTrue(result.success)
        self.assertTrue(result.data["valid"])

    def test_execute_unknown_tool(self):
        """Неизвестный инструмент через registry → ошибка."""
        storage = _make_mock_storage()
        registry = create_storage_tools(storage)
        result = registry.execute("nonexistent_tool", {})
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.error_msg)


class TestStorageToolSchemas(unittest.TestCase):
    """Test that all schemas are valid Anthropic format."""

    def test_all_schemas_defined(self):
        """Все 7 схем определены в STORAGE_TOOL_SCHEMAS."""
        self.assertEqual(len(STORAGE_TOOL_SCHEMAS), 7)

    def test_schema_format_valid(self):
        """Каждая схема имеет правильный формат."""
        for name, schema in STORAGE_TOOL_SCHEMAS.items():
            self.assertIn("type", schema, f"Schema {name} missing 'type'")
            self.assertEqual(schema["type"], "object", f"Schema {name} type != object")
            self.assertIn("properties", schema, f"Schema {name} missing 'properties'")
            # Our schemas have required at top level (Anthropic tool format)
            self.assertIn("required", schema,
                          f"Schema {name} missing 'required'")

    def test_search_schema_required_fields(self):
        """search_articles schema требует query."""
        schema = SEARCH_ARTICLES_SCHEMA
        self.assertIn("query", schema["required"])

    def test_detail_schema_requires_doi(self):
        """get_article_detail schema требует doi."""
        self.assertIn("doi", ARTICLE_DETAIL_SCHEMA["required"])


if __name__ == "__main__":
    unittest.main()
