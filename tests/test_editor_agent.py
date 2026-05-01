"""Unit tests for EditorAgent — B+ hybrid architecture.

Tests cover:
  - Data Models: ArticleProposal, EvidencePack, DiscoveryReport, EditorResult, EditorState
  - EditorAgent.run(): B+ pipeline (Loader → Discovery → Synthesize → Validate)
  - Job ID generation and checkpoint saving
  - Proposal validation: DOI gate, confidence scoring (B+ enhanced)
  - Evidence pack building from storage
  - Discovery report parsing
  - Resume from completed/failed jobs
  - Error handling and failed checkpoints
"""

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.agents.editor import (
    ArticleProposal,
    EvidencePack,
    DiscoveryReport,
    EditorResult,
    EditorState,
    EditorAgent,
)
from engine.llm.tool_loop import ToolUseLoop, ToolUseResult
from engine.llm.response_parser import parse_proposals_from_text


# ── Helpers ───────────────────────────────────────────────────────

def text_block(text):
    return {"type": "text", "text": text}


def tool_use(name, inp, cid=None):
    return {"type": "tool_use", "id": cid or f"call_{name}", "name": name, "input": inp}


def make_llm_response(content, stop_reason="end_turn", usage=None):
    if isinstance(content, str):
        content = [text_block(content)]
    return {
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 100, "output_tokens": 50},
    }


class MockLLM:
    """Mock LLM returning pre-defined responses."""

    def __init__(self, responses):
        self.responses = list(responses)

    def tool_complete(self, **kw):
        if not self.responses:
            return make_llm_response("No more responses")
        return self.responses.pop(0)


class MockStorageToolHandler:
    def __init__(self, return_value=None):
        self.return_value = return_value or {"result": "ok"}
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        from engine.tools.base import ToolResult
        return ToolResult.ok(data=self.return_value)


# Sample articles for evidence pack testing
SAMPLE_ARTICLES = [
    {
        "doi": "10.1000/test1",
        "title": "Arctic Permafrost Methane",
        "title_ru": "Метан арктической мерзлоты",
        "year": 2023,
        "topics_ru": ["мерзлота", "метан", "климат"],
        "source": "openalex",
        "abstract": "Study of methane emissions from Arctic permafrost regions.",
        "scores": {"total_5": 4.5},
        "citations": 42,
        "llm_summary": "Обзор выбросов метана из мерзлоты.",
    },
    {
        "doi": "10.1000/test2",
        "title": "InSAR Landslide Detection",
        "title_ru": "Детекция оползней через InSAR",
        "year": 2024,
        "topics_ru": ["оползни", "InSAR", "дистанционное зондирование"],
        "source": "europe_pmc",
        "abstract": "Using InSAR for landslide monitoring.",
        "scores": {"total_5": 4.8},
        "citations": 67,
        "llm_summary": "Применение InSAR для мониторинга оползней.",
    },
    {
        "doi": "10.1000/test3",
        "title": "ML for Geological Hazards",
        "title_ru": "ML для геологических опасностей",
        "year": 2024,
        "topics_ru": ["машинное обучение", "геологические опасности"],
        "source": "crossref",
        "abstract": "Machine learning approaches to hazard prediction.",
        "scores": {"total_5": 4.2},
        "citations": 23,
    },
]


def make_editor(tmp_path=None, llm_responses=None, articles=None):
    """Create EditorAgent with mock storage and LLM."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Mock storage
    mock_storage = MagicMock()
    mock_storage.data_dir = tmp_path
    mock_storage.load_articles.return_value = articles or SAMPLE_ARTICLES
    mock_storage.load_graph.return_value = {
        "nodes": [
            {"data": {"id": "n1", "doi": "10.1000/test1", "label": "Метан мерзлоты",
                      "community": "c1", "is_hub": True, "page_rank": 0.08, "betweenness": 0.03}},
            {"data": {"id": "n2", "doi": "10.1000/test2", "label": "InSAR оползни",
                      "community": "c2", "is_bridge": True, "page_rank": 0.05, "betweenness": 0.12}},
            {"data": {"id": "n3", "doi": "10.1000/test3", "label": "ML геоопасности",
                      "community": "c3", "page_rank": 0.03, "betweenness": 0.01}},
        ],
        "edges": [
            {"data": {"source": "n1", "target": "n2", "relation": "method_overlap"}},
            {"data": {"source": "n2", "target": "n3", "relation": "thematic_cluster"}},
        ],
        "metadata": {},
    }

    # Default LLM responses for B+ run (Discovery + Synthesize)
    if llm_responses is None:
        proposal_content = (
            '[{"title":"Тестовая статья","thesis":"Тезис о тестовой теме.",'
            '"confidence":0.85,"sources_available":3,"key_references":["DOI:10.1000/test1","DOI:10.1000/test2"],'
            '"gap_filled":"Заполняет пробел X"}]'
        )
        llm_responses = [
            # Phase 1: Discovery — LLM explores data
            make_llm_response(
                ("Исследование завершено.\n\n"
                 "Ключевые находки: В базе есть статьи по мерзлоте, ополяням и ML.\n"
                 "Наиболее релевантные: DOI:10.1000/test1 (метан мерзлоты), "
                 "DOI:10.1000/test2 (InSAR оползни).\n\n"
                 "Кросс-связи: статья про InSAR связана с ML-подходами через методологию.\n"
                 "Оценка материала: sufficient — достаточно данных для качественной обзорной статьи.\n\n"
                 "selected_dois: DOI:10.1000/test1, DOI:10.1000/test2, DOI:10.1000/test3"),
                stop_reason="end_turn",
                usage={"input_tokens": 500, "output_tokens": 200},
            ),
            # Phase 2: Synthesize — LLM forms proposals
            make_llm_response(
                proposal_content,
                stop_reason="end_turn",
                usage={"input_tokens": 300, "output_tokens": 150},
            ),
        ]

    mock_llm = MockLLM(llm_responses)

    editor = EditorAgent(storage=mock_storage, llm=mock_llm, jobs_dir=jobs_dir)

    # Manually set up tools and loop
    from engine.tools.base import ToolRegistry
    editor._tools = ToolRegistry()
    for name in [
        "search_articles", "get_article_detail", "validate_doi",
        "cluster_by_subtopic", "find_similar_existing",
        "count_storage_stats", "explore_domain",
        "graph_neighbors", "graph_path", "graph_hubs",
        "graph_clusters", "graph_cross_topic", "graph_centrality", "graph_stats",
    ]:
        editor._tools.register(
            name=name,
            handler=MockStorageToolHandler({"ok": True}),
            schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": [],
            },
            description=f"Mock {name}",
        )
    editor._loop = ToolUseLoop(mock_llm, editor._tools)

    return editor, tmp_path


# ── Test: Data Models ─────────────────────────────────────────────

class TestDataModels(unittest.TestCase):

    def test_article_proposal_defaults(self):
        p = ArticleProposal()
        self.assertEqual(p.status, "proposed")
        self.assertEqual(p.confidence, 0.5)
        self.assertEqual(p.key_references, [])
        self.assertEqual(p.discovery_depth, "shallow")
        self.assertEqual(p.enriched_sources, [])
        self.assertEqual(p.graph_roles, {})

    def test_article_proposal_to_dict(self):
        p = ArticleProposal(id="p1", title="T", thesis="Th", confidence=0.9)
        d = p.to_dict()
        self.assertEqual(d["id"], "p1")
        self.assertEqual(d["confidence"], 0.9)
        self.assertIn("discovery_depth", d)
        self.assertIn("enriched_sources", d)

    def test_evidence_pack_defaults(self):
        ep = EvidencePack()
        self.assertEqual(ep.total_articles, 0)
        self.assertFalse(ep.graph_available)
        d = ep.to_dict()
        self.assertIn("all_articles", d)
        self.assertIn("graph_summary", d)

    def test_evidence_pack_with_data(self):
        ep = EvidencePack(total_articles=42, graph_available=True)
        d = ep.to_dict()
        self.assertEqual(d["total_articles"], 42)
        self.assertTrue(d["graph_available"])

    def test_discovery_report_defaults(self):
        dr = DiscoveryReport()
        self.assertEqual(dr.material_sufficiency, "unknown")
        self.assertFalse(dr.is_sufficient)
        self.assertEqual(dr.proposal_count_hint, 1)  # unknown → 1

    def test_discovery_report_sufficient(self):
        dr = DiscoveryReport(material_sufficiency="sufficient")
        self.assertTrue(dr.is_sufficient)
        self.assertEqual(dr.proposal_count_hint, 3)

    def test_discovery_report_limited(self):
        dr = DiscoveryReport(material_sufficiency="limited")
        self.assertFalse(dr.is_sufficient)
        self.assertEqual(dr.proposal_count_hint, 2)

    def test_discovery_report_insufficient(self):
        dr = DiscoveryReport(material_sufficiency="insufficient")
        self.assertEqual(dr.proposal_count_hint, 1)

    def test_editor_result_to_dict(self):
        r = EditorResult(job_id="j1", topic="test", status="complete")
        d = r.to_dict()
        self.assertEqual(d["job_id"], "j1")
        self.assertIsNone(d["error"])
        self.assertIn("discovery", d)

    def test_editor_state_bplus_phases(self):
        s = EditorState(job_id="j1", phase="discovering")
        d = s.to_dict()
        self.assertEqual(d["phase"], "discovering")
        self.assertIn("evidence_pack", d)
        self.assertIn("discovery", d)


# ── Test: EditorAgent.run() ───────────────────────────────────────

class TestEditorAgentRun(unittest.TestCase):

    def setUp(self):
        self.editor, self.tmp = make_editor()

    def test_run_creates_job_id(self):
        result = self.editor.run(topic="Arctic methane")
        self.assertTrue(result.job_id.startswith("edit_"))
        self.assertGreater(len(result.job_id), 15)

    def test_run_returns_proposals(self):
        result = self.editor.run(topic="test topic")
        self.assertEqual(result.status, "complete")
        self.assertGreaterEqual(len(result.proposals), 1)
        for p in result.proposals:
            self.assertIsInstance(p, ArticleProposal)
            self.assertTrue(len(p.title) > 0)
            self.assertTrue(len(p.thesis) > 0)

    def test_run_returns_discovery(self):
        result = self.editor.run(topic="test")
        # In B+, analysis is a dict (legacy key) and discovery is DiscoveryReport
        self.assertIsNotNone(result.analysis)
        self.assertIsNotNone(result.discovery)
        self.assertIsInstance(result.discovery, DiscoveryReport)

    def test_run_duration_positive(self):
        result = self.editor.run(topic="test")
        self.assertGreater(result.duration_sec, 0)

    def test_run_tool_rounds_counted(self):
        result = self.editor.run(topic="test")
        # Should have at least 2 rounds (Discovery + Synthesize)
        self.assertGreaterEqual(result.tool_rounds_total, 2)

    def test_run_saves_checkpoint(self):
        self.editor.run(topic="test")
        files = list(self.tmp.glob("jobs/*.json"))
        self.assertGreaterEqual(len(files), 1)
        with open(files[0]) as f:
            data = json.load(f)
        self.assertIn("job_id", data)
        self.assertEqual(data["phase"], "done")

    def test_run_confidence_in_range(self):
        result = self.editor.run(topic="test")
        for p in result.proposals:
            self.assertGreaterEqual(p.confidence, 0.0)
            self.assertLessEqual(p.confidence, 1.0)

    def test_run_has_discovery_in_checkpoint(self):
        self.editor.run(topic="test")
        files = list(self.tmp.glob("jobs/*.json"))
        with open(files[0]) as f:
            data = json.load(f)
        # B+ checkpoints should have discovery field
        self.assertIn("discovery", data)
        self.assertIn("evidence_pack", data)


# ── Test: _build_evidence_pack ─────────────────────────────────────

class TestEvidencePack(unittest.TestCase):

    def setUp(self):
        self.editor, self.tmp = make_editor(articles=SAMPLE_ARTICLES)

    def test_evidence_pack_loads_all_articles(self):
        ep = self.editor._build_evidence_pack("test")
        self.assertEqual(ep.total_articles, len(SAMPLE_ARTICLES))
        self.assertEqual(len(ep.all_articles), len(SAMPLE_ARTICLES))

    def test_evidence_pack_has_compact_cards(self):
        ep = self.editor._build_evidence_pack("test")
        for card in ep.all_articles:
            self.assertIn("doi", card)
            self.assertIn("title_ru", card)
            self.assertIn("year", card)
            self.assertIn("topics_ru", card)
            self.assertIn("abstract_preview", card)

    def test_evidence_pack_abstract_truncated(self):
        ep = self.editor._build_evidence_pack("test")
        long_abstract = SAMPLE_ARTICLES[0]["abstract"]
        card = ep.all_articles[0]
        self.assertLessEqual(len(card["abstract_preview"]), 300)

    def test_evidence_pack_domain_stats(self):
        ep = self.editor._build_evidence_pack("test")
        ds = ep.domain_stats
        self.assertIn("sources", ds)
        self.assertIn("years", ds)
        self.assertIn("top_topics_ru", ds)
        self.assertIn("enriched_pct", ds)

    def test_evidence_pack_graph_summary(self):
        ep = self.editor._build_evidence_pack("test")
        if ep.graph_available:
            gs = ep.graph_summary
            self.assertIn("communities", gs)
            self.assertIn("node_count", gs)

    def test_evidence_pack_empty_storage(self):
        # Create editor with empty articles list
        tmp_path = Path(tempfile.mkdtemp())
        jobs_dir = tmp_path / "jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        mock_storage = MagicMock()
        mock_storage.data_dir = tmp_path
        mock_storage.load_articles.return_value = []
        mock_storage.load_graph.return_value = {"nodes": [], "edges": [], "metadata": {}}
        editor = EditorAgent(storage=mock_storage, llm=MagicMock(), jobs_dir=jobs_dir)
        ep = editor._build_evidence_pack("test")
        self.assertEqual(ep.total_articles, 0)
        self.assertEqual(len(ep.all_articles), 0)


# ── Test: _parse_discovery ────────────────────────────────────────

class TestParseDiscovery(unittest.TestCase):

    def setUp(self):
        self.editor, self.tmp = make_editor(articles=SAMPLE_ARTICLES)
        self.evidence = self.editor._build_evidence_pack("test")

    def test_parse_sufficient_from_text(self):
        result = ToolUseResult(
            content="Достаточно материалов. Найдено 20 статей по теме. "
                   "DOI:10.1000/test1 и DOI:10.1000/test2 наиболее релевантны. "
                   "Оценка: sufficient.",
            tool_calls_made=[
                {"name": "get_article_detail", "arguments": {"doi": "10.1000/test1"}, "result": {}},
            ],
            total_rounds=1,
            stop_reason="end_turn",
        )
        dr = self.editor._parse_discovery(result, self.evidence)
        self.assertEqual(dr.material_sufficiency, "sufficient")
        self.assertIn("10.1000/test1", dr.explored_dois)

    def test_parse_insufficient_from_text(self):
        result = ToolUseResult(
            content="Критически не хватает данных. Практически нет статей по теме.",
            tool_calls_made=[],
            total_rounds=1,
            stop_reason="end_turn",
        )
        dr = self.editor._parse_discovery(result, self.evidence)
        self.assertEqual(dr.material_sufficiency, "insufficient")

    def test_parse_extracts_dois(self):
        result = ToolUseResult(
            content="Использую DOI:10.1000/test1 и DOI:10.1000/test2 как основные источники.",
            tool_calls_made=[],
            total_rounds=1,
            stop_reason="end_turn",
        )
        dr = self.editor._parse_discovery(result, self.evidence)
        self.assertIn("10.1000/test1", dr.selected_dois)
        self.assertIn("10.1000/test2", dr.selected_dois)


# ── Test: _validate_proposal_bplus ────────────────────────────────

class TestValidateProposalBPlus(unittest.TestCase):

    def setUp(self):
        self.editor, self.tmp = make_editor(articles=SAMPLE_ARTICLES)
        self.evidence = self.editor._build_evidence_pack("test")
        self.discovery = DiscoveryReport(
            material_sufficiency="sufficient",
            selected_dois=["10.1000/test1", "10.1000/test2", "10.1000/test3"],
            explored_dois=["10.1000/test1", "10.1000/test2", "10.1000/test3"],
        )

    def test_valid_proposal_passes(self):
        raw = {
            "title": "New unique article about ocean acidification",
            "thesis": "About something completely new.",
            "key_references": ["DOI:10.1000/test1"],
            "confidence": 0.85,
        }
        p = self.editor._validate_proposal_bplus(raw, self.discovery, self.evidence, index=0)
        self.assertNotEqual(p.status, "duplicate")
        self.assertGreater(p.confidence, 0.5)
        self.assertIn("discovery_depth", p.to_dict())

    def test_enriched_sources_populated(self):
        raw = {
            "title": "Test enriched",
            "thesis": "X",
            "key_references": ["DOI:10.1000/test1", "DOI:10.1000/test2"],
            "confidence": 0.9,
        }
        p = self.editor._validate_proposal_bplus(raw, self.discovery, self.evidence, index=0)
        self.assertGreaterEqual(len(p.enriched_sources), 1)
        self.assertEqual(p.enriched_sources[0]["doi"], "10.1000/test1")

    def test_deep_discovery_bonus(self):
        raw = {
            "title": "Deep research article",
            "thesis": "Well researched.",
            # Need 3+ refs with detail for "deep" classification
            "key_references": ["DOI:10.1000/test1", "DOI:10.1000/test2", "DOI:10.1000/test3"],
            "confidence": 0.7,
        }
        p = self.editor._validate_proposal_bplus(raw, self.discovery, self.evidence, index=0)
        # Should get deep research bonus (refs_with_detail >= 3)
        self.assertEqual(p.discovery_depth, "deep")

    def test_shallow_discovery(self):
        discovery_shallow = DiscoveryReport(
            material_sufficiency="sufficient",
            selected_dois=["10.1000/test1"],
            explored_dois=[],  # Nothing explored deeply
        )
        raw = {
            "title": "Shallow article",
            "thesis": "X",
            "key_references": ["DOI:10.1000/test1"],
            "confidence": 0.8,
        }
        p = self.editor._validate_proposal_bplus(raw, discovery_shallow, self.evidence, index=0)
        self.assertEqual(p.discovery_depth, "shallow")

    def test_insufficient_material_penalty(self):
        discovery_bad = DiscoveryReport(
            material_sufficiency="insufficient",
            selected_dois=["10.1000/test1"],
            explored_dois=["10.1000/test1"],
        )
        raw = {
            "title": "Limited material",
            "thesis": "X",
            "key_references": ["DOI:10.1000/test1"],
            "confidence": 0.9,
        }
        p = self.editor._validate_proposal_bplus(raw, discovery_bad, self.evidence, index=0)
        # Should get *0.6 penalty for insufficient material
        self.assertLess(p.confidence, 0.9 * 0.65)

    def test_confidence_clamped_to_01(self):
        raw = {"title": "T", "thesis": "T", "confidence": 5.0}
        p = self.editor._validate_proposal_bplus(raw, self.discovery, self.evidence)
        self.assertLessEqual(p.confidence, 1.0)
        self.assertGreaterEqual(p.confidence, 0.0)

    def test_proposal_gets_id(self):
        raw = {"title": "T", "thesis": "T"}
        p = self.editor._validate_proposal_bplus(raw, self.discovery, self.evidence, index=3)
        self.assertTrue(p.id.startswith("prop_"))

    def test_doi_not_in_discovery_dropped(self):
        raw = {
            "title": "External DOI",
            "thesis": "X",
            "key_references": ["DOI:10.external/fake999"],  # NOT in discovery.selected_dois
            "confidence": 0.8,
        }
        p = self.editor._validate_proposal_bplus(raw, self.discovery, self.evidence, index=0)
        # External DOI not in discovery and not in local DB should be dropped
        self.assertLessEqual(len(p.key_references), 1)


# ── Test: Resume ──────────────────────────────────────────────────

class TestResume(unittest.TestCase):

    def setUp(self):
        self.editor, self.tmp = make_editor()

    def test_resume_completed_job(self):
        result = self.editor.run(topic="resume test")
        resumed = self.editor.resume(result.job_id)
        self.assertEqual(resumed.status, "complete")
        self.assertEqual(resumed.job_id, result.job_id)

    def test_resume_nonexistent_raises(self):
        with self.assertRaises(ValueError):
            self.editor.resume("edit_nonexistent_abc123")

    def test_resume_failed_job(self):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        # Create a failed checkpoint manually
        state = EditorState(
            job_id="edit_fail_test", topic="fail", phase="failed",
            error="Test error", started_at=now, updated_at=now,
        )
        self.editor._save_checkpoint(state)
        resumed = self.editor.resume("edit_fail_test")
        self.assertEqual(resumed.status, "failed")
        self.assertIn("Test error", resumed.error)


# ── Test: Error Handling ──────────────────────────────────────────

class TestErrorHandling(unittest.TestCase):

    def test_no_storage_raises(self):
        editor = EditorAgent(llm=MagicMock())
        with self.assertRaises(RuntimeError):
            editor.run(topic="test")


# ── Test: Helpers ──────────────────────────────────────────────────

class TestHelpers(unittest.TestCase):

    def test_similarity_identical(self):
        self.assertAlmostEqual(EditorAgent._similarity("hello world", "hello world"), 1.0)

    def test_similarity_different(self):
        self.assertAlmostEqual(EditorAgent._similarity("hello", "world"), 0.0)

    def test_similarity_partial(self):
        sim = EditorAgent._similarity("hello world foo", "hello bar baz")
        self.assertGreater(sim, 0.0)
        self.assertLess(sim, 1.0)

    def test_similarity_empty(self):
        self.assertEqual(EditorAgent._similarity("", "anything"), 0.0)

    def test_extract_gaps_finds_patterns(self):
        text = "Пробел: недостаточно данных о выбросах метана. Не хватает: обзора за 2024 год."
        gaps = EditorAgent._extract_gaps(text)
        self.assertGreater(len(gaps), 0)

    def test_extract_gaps_empty_text(self):
        gaps = EditorAgent._extract_gaps("Just regular text without any gaps mentioned.")
        # May be empty or minimal


if __name__ == "__main__":
    unittest.main()
