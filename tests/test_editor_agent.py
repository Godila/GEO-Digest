"""Unit tests for EditorAgent — core "brain" of tool-use architecture.

Tests cover:
  - EditorAgent.run(): full pipeline (analysis → proposals → validation)
  - Job ID generation and checkpoint saving
  - Proposal validation: DOI check, duplicate detection, confidence scoring
  - _build_analysis from tool call history
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
    StorageAnalysis,
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


def make_editor(tmp_path=None, llm_responses=None):
    """Create EditorAgent with mock storage and LLM."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    # Mock storage
    mock_storage = MagicMock()
    mock_storage.data_dir = tmp_path

    # Default LLM responses for a successful run
    if llm_responses is None:
        analysis_content = (
            '```json\n[{"title":"Test Article","thesis":"A thesis about test.",'
            '"confidence":0.8,"sources_available":5,"key_references":["DOI:10.real/123"],'
            '"gap_filled":"Fills gap X"}]\n```'
        )
        llm_responses = [
            # Phase 1: Analysis — end_turn immediately (no tools needed for simple test)
            make_llm_response(
                "Анализ завершён. Найдено 50 статей по теме, 3 кластера.",
                stop_reason="end_turn",
                usage={"input_tokens": 200, "output_tokens": 100},
            ),
            # Phase 2: Proposals
            make_llm_response(
                analysis_content,
                stop_reason="end_turn",
                usage={"input_tokens": 300, "output_tokens": 200},
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

    def test_article_proposal_to_dict(self):
        p = ArticleProposal(id="p1", title="T", thesis="Th", confidence=0.9)
        d = p.to_dict()
        self.assertEqual(d["id"], "p1")
        self.assertEqual(d["confidence"], 0.9)

    def test_storage_analysis_to_dict(self):
        a = StorageAnalysis(total_articles=100, year_range=(2020, 2025))
        d = a.to_dict()
        self.assertEqual(d["year_range"], [2020, 2025])

    def test_editor_result_to_dict(self):
        r = EditorResult(job_id="j1", topic="test", status="done")
        d = r.to_dict()
        self.assertEqual(d["job_id"], "j1")
        self.assertIsNone(d["error"])

    def test_editor_state_to_dict(self):
        s = EditorState(job_id="j1", phase="done")
        d = s.to_dict()
        self.assertEqual(d["phase"], "done")


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
        self.assertEqual(result.status, "done")
        self.assertGreaterEqual(len(result.proposals), 1)
        for p in result.proposals:
            self.assertIsInstance(p, ArticleProposal)
            self.assertTrue(len(p.title) > 0)
            self.assertTrue(len(p.thesis) > 0)

    def test_run_returns_analysis(self):
        result = self.editor.run(topic="test")
        self.assertIsNotNone(result.analysis)
        self.assertIsInstance(result.analysis, StorageAnalysis)

    def test_run_duration_positive(self):
        result = self.editor.run(topic="test")
        self.assertGreater(result.duration_sec, 0)

    def test_run_tool_rounds_counted(self):
        result = self.editor.run(topic="test")
        # Should have at least 2 rounds (Phase 1 + Phase 2)
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


# ── Test: Checkpoint / Resume ────────────────────────────────────

class TestEditorResume(unittest.TestCase):

    def setUp(self):
        self.editor, self.tmp = make_editor()

    def test_resume_completed_job(self):
        result = self.editor.run(topic="resume test")
        resumed = self.editor.resume(result.job_id)
        self.assertEqual(resumed.job_id, result.job_id)
        self.assertEqual(resumed.status, "done")
        self.assertEqual(len(resumed.proposals), len(result.proposals))

    def test_resume_nonexistent_raises(self):
        with self.assertRaises(ValueError) as ctx:
            self.editor.resume("nonexistent_job_12345")
        self.assertIn("not found", str(ctx.exception))

    def test_checkpoint_has_all_fields(self):
        self.editor.run(topic="fields test")
        files = list(self.tmp.glob("jobs/*.json"))
        with open(files[0]) as f:
            data = json.load(f)
        required = ["job_id", "topic", "phase", "started_at", "updated_at"]
        for field in required:
            self.assertIn(field, data, f"Missing field: {field}")


# ── Test: Error Handling ─────────────────────────────────────────

class TestEditorErrorHandling(unittest.TestCase):

    def test_error_saves_failed_checkpoint(self):
        editor, tmp = make_editor()

        # Make _format_analysis_context raise to trigger the except block in run()
        original_format = editor._format_analysis_context
        editor._format_analysis_context = lambda analysis: (_ for _ in ()).throw(RuntimeError("Test crash"))

        with self.assertRaises(RuntimeError):
            editor.run(topic="error test")

        files = list(tmp.glob("jobs/*.json"))
        self.assertGreaterEqual(len(files), 1)
        with open(files[-1]) as f:
            data = json.load(f)
        self.assertEqual(data["phase"], "failed")
        self.assertIn("error", data)


# ── Test: _build_analysis ────────────────────────────────────────

class TestBuildAnalysis(unittest.TestCase):

    def test_build_from_tool_history(self):
        loop_result = ToolUseResult(
            content="Анализ завершён. Найдены пробелы в покрытии темы methane.",
            tool_calls_made=[
                {
                    "name": "count_storage_stats",
                    "result": {
                        "total_articles": 181,
                        "enriched_articles": 150,
                        "year_range": {"min_year": 2020, "max_year": 2025},
                        "by_year": {"2023": 42, "2024": 38},
                    },
                },
                {
                    "name": "cluster_by_subtopic",
                    "result": {
                        "total_relevant": 30,
                        "cluster_count": 3,
                        "clusters": [
                            {"theme": "methane emissions", "count": 15},
                            {"theme": "permafrost thaw", "count": 10},
                            {"theme": "carbon feedback", "count": 5},
                        ],
                    },
                },
                {
                    "name": "search_articles",
                    "result": {"total_found": 12, "returned": 10, "articles": []},
                },
            ],
            total_rounds=2,
            stop_reason="end_turn",
            usage={},
            warnings=[],
        )

        analysis = EditorAgent._build_analysis(loop_result)
        self.assertEqual(analysis.total_articles, 181)
        self.assertEqual(len(analysis.clusters), 3)
        self.assertEqual(analysis.clusters[0]["theme"], "methane emissions")
        self.assertEqual(analysis.relevant_count, 12)

    def test_build_empty_history(self):
        loop_result = ToolUseResult(
            content="No data found.",
            tool_calls_made=[],
            total_rounds=1,
            stop_reason="end_turn",
        )
        analysis = EditorAgent._build_analysis(loop_result)
        self.assertEqual(analysis.total_articles, 0)
        self.assertEqual(len(analysis.clusters), 0)


# ── Test: _validate_proposal ─────────────────────────────────────

class TestValidateProposal(unittest.TestCase):

    def setUp(self):
        self.editor, self.tmp = make_editor()
        self.analysis = StorageAnalysis(
            total_articles=100,
            existing_articles=[
                {
                    "file": "arctic_methane.md",
                    "preview": "Arctic Permafrost Methane Emissions Review 2024",
                    "similarity": 0.8,
                },
            ],
        )

    def test_valid_proposal_passes(self):
        raw = {
            "title": "New unique article about ocean acidification",
            "thesis": "About something completely new.",
            "key_references": ["DOI:10.real/123"],
            "confidence": 0.85,
        }
        p = self.editor._validate_proposal(raw, self.analysis, index=0)
        self.assertNotEqual(p.status, "duplicate")
        self.assertGreater(p.confidence, 0.5)

    def test_duplicate_detected(self):
        raw = {
            "title": "Arctic Permafrost Methane Emissions Review",
            "thesis": "About methane in Arctic.",
            "confidence": 0.9,
        }
        p = self.editor._validate_proposal(raw, self.analysis, index=0)
        self.assertEqual(p.status, "duplicate")
        self.assertLess(p.confidence, 0.5)

    def test_low_confidence_for_few_sources(self):
        raw = {
            "title": "Test",
            "thesis": "X",
            "key_references": [],  # No sources!
            "confidence": 0.9,
        }
        p = self.editor._validate_proposal(raw, self.analysis, index=0)
        # Penalty for < 3 sources: * 0.7
        self.assertLess(p.confidence, 0.9)

    def test_confidence_clamped_to_01(self):
        raw = {"title": "T", "thesis": "T", "confidence": 5.0}
        p = self.editor._validate_proposal(raw, self.analysis)
        self.assertLessEqual(p.confidence, 1.0)
        self.assertGreaterEqual(p.confidence, 0.0)

    def test_proposal_gets_id(self):
        raw = {"title": "T", "thesis": "T"}
        p = self.editor._validate_proposal(raw, self.analysis, index=3)
        self.assertTrue(p.id.startswith("prop_"))


# ── Test: Helpers ────────────────────────────────────────────────

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
