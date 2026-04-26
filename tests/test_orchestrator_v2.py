"""Unit tests for Orchestrator v2 — State Machine (S6).

Tests cover:
  - PipelineJob creation and state transitions
  - Full happy path: edit → select → develop → write → review → done
  - Error handling: editor failure, writer failure
  - Review loop: needs_revision sends back to developing
  - Cancel from any state
  - Persistence: save/load job, list jobs with v2 filter
  - Development rounds accumulation
  - Invalid proposal selection
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

from engine.orchestrator_v2 import (
    EditorOrchestrator,
    PipelineState,
    PipelineJob,
    now_iso,
)


# ── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def tmp_jobs_dir(tmp_path):
    d = tmp_path / "jobs"
    d.mkdir()
    return str(d)


@pytest.fixture
def orch(tmp_jobs_dir):
    return EditorOrchestrator(jobs_dir=tmp_jobs_dir)


@pytest.fixture
def mock_editor_result():
    """Мок результата EditorAgent.run()."""
    return {
        "job_id": "edit_test_001",
        "topic": "test topic",
        "phase": "done",
        "proposals": [
            {
                "id": "prop_1",
                "title": "Test Article Title",
                "thesis": "This is a test thesis about important things.",
                "confidence": 0.87,
                "sources_available": 15,
                "target_audience": "researchers",
                "status": "proposed",
                "key_references": ["10.1038/nat1234", "10.1126/sci5678"],
            },
            {
                "id": "prop_2",
                "title": "Duplicate Article",
                "thesis": "Similar to prop 1.",
                "confidence": 0.3,
                "sources_available": 2,
                "status": "duplicate",
                "key_references": [],
            },
        ],
        "analysis": {"total_articles": 100, "gaps": []},
    }


@pytest.fixture
def mock_draft():
    return {
        "content": "Draft content about Arctic methane...",
        "sources_read": 5,
        "summary": "Key findings from references",
    }


@pytest.fixture
def mock_article():
    return {
        "title": "Final Article: Arctic Methane Emissions",
        "content": "Full article text...",
        "word_count": 2500,
        "references_used": ["10.1038/nat1234"],
    }


@pytest.fixture
def mock_review_approve():
    return {"verdict": "approve", "score": 0.9, "comments": "Good article"}


@pytest.fixture
def mock_review_needs_work():
    return {"verdict": "needs_revision", "score": 0.5, "comments": "Missing data on Siberia"}


def _setup_job_at_selecting(orch, mock_editor_result):
    """Helper: создаёт job и доводит до SELECTING."""
    with patch.object(orch.editor, 'run', return_value=mock_editor_result):
        job = orch.create_job(topic="test")
        job = orch.run_editing_phase(job)
    assert job.state == PipelineState.SELECTING
    return job


def _setup_job_at_developing(orch, mock_editor_result):
    """Helper: создаёт job и доводит до DEVELOPING."""
    job = _setup_job_at_selecting(orch, mock_editor_result)
    job = orch.select_proposal(job, "prop_1")
    assert job.state == PipelineState.DEVELOPING
    return job


def _setup_job_at_writing(orch, mock_editor_result, mock_draft):
    """Helper: создаёт job и доводит до WRITING (после develop)."""
    job = _setup_job_at_developing(orch, mock_editor_result)
    with patch.object(orch.reader, 'run', return_value=mock_draft):
        job = orch.develop(job)
    # Меняем состояние вручную для теста write
    job.state = PipelineState.WRITING
    return job


# ── Test: Job Creation ─────────────────────────────────────

class TestCreateJob:
    def test_create_basic(self, orch):
        job = orch.create_job(topic="Arctic methane")
        assert job.job_id.startswith("20")  # YYYYMMDD_
        assert job.topic == "Arctic methane"
        assert job.state == PipelineState.EDITING
        assert job.error is None

    def test_create_with_domain(self, orch):
        job = orch.create_job(topic="test", domain="climate_science")
        assert job.domain == "climate_science"

    def test_create_saves_to_disk(self, orch):
        job = orch.create_job(topic="persist me")
        path = Path(orch.jobs_dir) / f"{job.job_id}.json"
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert data["topic"] == "persist me"
        assert data["state"] == "editing"

    def test_create_generates_unique_ids(self, orch):
        ids = [orch.create_job(topic=f"t{i}").job_id for i in range(5)]
        assert len(set(ids)) == 5  # All unique


class TestEditingPhase:
    def test_editing_calls_editor(self, orch, mock_editor_result):
        with patch.object(orch.editor, 'run', return_value=mock_editor_result) as mock_run:
            job = orch.create_job(topic="test")
            job = orch.run_editing_phase(job)

        mock_run.assert_called_once()
        assert job.state == PipelineState.SELECTING
        assert job.editor_result is not None
        assert len(job.editor_result.get("proposals", [])) == 2

    def test_editing_failure_sets_failed(self, orch):
        with patch.object(orch.editor, 'run', side_effect=RuntimeError("LLM timeout")):
            job = orch.create_job(topic="test")
            job = orch.run_editing_phase(job)

        assert job.state == PipelineState.FAILED
        assert "timeout" in job.error.lower()

    def test_editing_saves_proposals(self, orch, mock_editor_result):
        with patch.object(orch.editor, 'run', return_value=mock_editor_result):
            job = orch.create_job(topic="test")
            job = orch.run_editing_phase(job)

        proposals = job.editor_result.get("proposals", [])
        assert any(p["id"] == "prop_1" for p in proposals)

    def test_editing_updates_timestamp(self, orch, mock_editor_result):
        with patch.object(orch.editor, 'run', return_value=mock_editor_result):
            job = orch.create_job(topic="test")
            t1 = job.updated_at
            import time; time.sleep(0.05)  # tiny delay
            job = orch.run_editing_phase(job)

        assert job.updated_at >= t1


class TestSelectProposal:
    def test_select_valid(self, orch, mock_editor_result):
        job = _setup_job_at_selecting(orch, mock_editor_result)
        job = orch.select_proposal(job, "prop_1")

        assert job.state == PipelineState.DEVELOPING
        assert job.selected_proposal_id == "prop_1"

    def test_select_invalid_raises(self, orch, mock_editor_result):
        job = _setup_job_at_selecting(orch, mock_editor_result)
        with pytest.raises(ValueError, match="not found"):
            orch.select_proposal(job, "nonexistent")

    def test_select_without_editor_result_raises(self, orch):
        job = PipelineJob(job_id="test", topic="t", state=PipelineState.SELECTING)
        with pytest.raises(ValueError, match="not found"):
            orch.select_proposal(job, "any")


class TestDevelop:
    def test_develop_calls_reader(self, orch, mock_editor_result, mock_draft):
        job = _setup_job_at_developing(orch, mock_editor_result)

        with patch.object(orch.reader, 'run', return_value=mock_draft) as mock_read:
            job = orch.develop(job, "first feedback")

        mock_read.assert_called_once()
        assert job.current_draft is not None
        assert job.state == PipelineState.DEVELOPING

    def test_develop_adds_round(self, orch, mock_editor_result, mock_draft):
        job = _setup_job_at_developing(orch, mock_editor_result)

        with patch.object(orch.reader, 'run', return_value=mock_draft):
            job = orch.develop(job, "first feedback")

        assert len(job.development_rounds) == 1
        assert job.development_rounds[0]["user_feedback"] == "first feedback"
        assert job.development_rounds[0]["round"] == 1

    def test_develop_accumulates_rounds(self, orch, mock_editor_result, mock_draft):
        job = _setup_job_at_developing(orch, mock_editor_result)

        with patch.object(orch.reader, 'run', return_value=mock_draft):
            orch.develop(job, "first feedback")
            orch.develop(job, "second feedback")
            orch.develop(job, "third feedback")

        assert len(job.development_rounds) == 3
        assert job.development_rounds[0]["user_feedback"] == "first feedback"
        assert job.development_rounds[1]["user_feedback"] == "second feedback"
        assert job.development_rounds[2]["user_feedback"] == "third feedback"

    def test_develop_no_refs_graceful(self, orch):
        """Develop без key_references не падает."""
        job = PipelineJob(
            job_id="test", topic="t", state=PipelineState.DEVELOPING,
            editor_result={"proposals": [{"id": "p1", "key_references": []}]},
            selected_proposal_id="p1",
        )
        job = orch.develop(job)
        assert job.state == PipelineState.DEVELOPING
        assert len(job.development_rounds) == 1

    def test_develop_reader_error_graceful(self, orch, mock_editor_result):
        job = _setup_job_at_developing(orch, mock_editor_result)

        with patch.object(orch.reader, 'run', side_effect=ConnectionError("network down")):
            job = orch.develop(job)

        assert job.state == PipelineState.DEVELOPING
        assert "error" in (job.current_draft or {})


class TestWrite:
    def test_write_calls_writer(self, orch, mock_editor_result, mock_draft, mock_article):
        job = _setup_job_at_developing(orch, mock_editor_result)
        with patch.object(orch.reader, 'run', return_value=mock_draft):
            job = orch.develop(job)

        with patch.object(orch.writer, 'run', return_value=mock_article) as mock_wr:
            job = orch.write(job)

        mock_wr.assert_called_once()
        assert job.state == PipelineState.REVIEWING
        assert job.final_article is not None

    def test_write_failure_sets_failed(self, orch, mock_editor_result, mock_draft):
        job = _setup_job_at_developing(orch, mock_editor_result)
        with patch.object(orch.reader, 'run', return_value=mock_draft):
            job = orch.develop(job)

        with patch.object(orch.writer, 'run', side_effect=ValueError("Bad draft")):
            job = orch.write(job)

        assert job.state == PipelineState.FAILED
        assert "Bad draft" in job.error

    def test_write_without_selection_raises(self, orch):
        job = PipelineJob(job_id="test", topic="t", state=PipelineState.DEVELOPING)
        with pytest.raises(ValueError, match="no proposal"):
            orch.write(job)


class TestReview:
    def test_review_approve_goes_done(self, orch, mock_editor_result, mock_draft, mock_article, mock_review_approve):
        job = _setup_job_at_writing(orch, mock_editor_result, mock_draft)
        job.final_article = mock_article  # Simulate write completed

        with patch.object(orch.reviewer, 'run', return_value=mock_review_approve):
            job = orch.review(job)

        assert job.state == PipelineState.DONE

    def test_review_needs_revision_goes_back(self, orch, mock_editor_result, mock_draft, mock_article, mock_review_needs_work):
        job = _setup_job_at_writing(orch, mock_editor_result, mock_draft)
        job.final_article = mock_article

        with patch.object(orch.reviewer, 'run', return_value=mock_review_needs_work):
            job = orch.review(job)

        assert job.state == PipelineState.DEVELOPING  # Back!

    def test_review_failure_sets_failed(self, orch, mock_editor_result, mock_draft, mock_article):
        job = _setup_job_at_writing(orch, mock_editor_result, mock_draft)
        job.final_article = mock_article

        with patch.object(orch.reviewer, 'run', side_effect=RuntimeError("Review service down")):
            job = orch.review(job)

        assert job.state == PipelineState.FAILED

    def test_review_no_article_raises(self, orch):
        job = PipelineJob(job_id="test", topic="t", state=PipelineState.REVIEWING)
        with pytest.raises(ValueError, match="No article"):
            orch.review(job)


class TestCancel:
    def test_cancel_from_editing(self, orch):
        job = PipelineJob(job_id="test", topic="t", state=PipelineState.EDITING)
        job = orch.cancel(job)
        assert job.state == PipelineState.CANCELLED

    def test_cancel_from_developing(self, orch):
        job = PipelineJob(job_id="test", topic="t", state=PipelineState.DEVELOPING)
        job = orch.cancel(job)
        assert job.state == PipelineState.CANCELLED

    def test_cancel_from_writing(self, orch):
        job = PipelineJob(job_id="test", topic="t", state=PipelineState.WRITING)
        job = orch.cancel(job)
        assert job.state == PipelineState.CANCELLED

    def test_cancel_from_selecting(self, orch):
        job = PipelineJob(job_id="test", topic="t", state=PipelineState.SELECTING)
        job = orch.cancel(job)
        assert job.state == PipelineState.CANCELLED

    def test_cancel_persists(self, orch):
        job = orch.create_topic="cancel_test" if False else orch.create_job(topic="cancel_test")
        job = orch.cancel(job)
        loaded = orch.load_job(job.job_id)
        assert loaded.state == PipelineState.CANCELLED


class TestPersistence:
    def test_load_job(self, orch):
        original = orch.create_job(topic="persist me")
        original.state = PipelineState.DONE
        orch._save_job(original)  # Explicitly save after state change

        loaded = orch.load_job(original.job_id)
        assert loaded.job_id == original.job_id
        assert loaded.topic == original.topic
        assert loaded.state == PipelineState.DONE

    def test_load_preserves_editor_result(self, orch, mock_editor_result):
        with patch.object(orch.editor, 'run', return_value=mock_editor_result):
            job = orch.create_job(topic="test")
            orch.run_editing_phase(job)

        loaded = orch.load_job(job.job_id)
        assert loaded.editor_result is not None
        assert len(loaded.editor_result.get("proposals", [])) == 2

    def test_load_preserves_development_rounds(self, orch, mock_editor_result, mock_draft):
        job = _setup_job_at_developing(orch, mock_editor_result)
        with patch.object(orch.reader, 'run', return_value=mock_draft):
            orch.develop(job, "feedback")

        loaded = orch.load_job(job.job_id)
        assert len(loaded.development_rounds) == 1

    def test_list_jobs_filters_v2_only(self, orch):
        orch.create_job(topic="v2 job")

        # Создаём вручную v1 job (без state поля с валидным значением)
        v1_path = Path(orch.jobs_dir) / "v1_old_job.json"
        v1_path.write_text('{"job_id":"v1","topic":"old","phase":"done"}')

        jobs = orch.list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_id"].startswith("20")
        assert jobs[0]["topic"] == "v2 job"

    def test_list_jobs_limit(self, orch):
        for i in range(25):
            orch.create_job(topic=f"job_{i}")

        jobs = orch.list_jobs(limit=10)
        assert len(jobs) == 10

    def test_list_jobs_includes_metadata(self, orch, mock_editor_result):
        with patch.object(orch.editor, 'run', return_value=mock_editor_result):
            job = orch.create_job(topic="metadata test")
            orch.run_editing_phase(job)

        jobs = orch.list_jobs()
        assert len(jobs) == 1
        assert "created_at" in jobs[0]
        assert "updated_at" in jobs[0]
        assert "state" in jobs[0]

    def test_load_nonexistent_raises(self, orch):
        with pytest.raises(ValueError, match="not found"):
            orch.load_job("nonexistent_xyz")


class TestHappyPath:
    def test_full_pipeline(self, orch, mock_editor_result, mock_draft, mock_article, mock_review_approve):
        """Полный путь: edit → select → develop → write → review → done."""
        with patch.object(orch.editor, 'run', return_value=mock_editor_result):
            job = orch.create_job(topic="full pipeline test")
            job = orch.run_editing_phase(job)

        assert job.state == PipelineState.SELECTING

        job = orch.select_proposal(job, "prop_1")
        assert job.state == PipelineState.DEVELOPING

        with patch.object(orch.reader, 'run', return_value=mock_draft):
            job = orch.develop(job)
        assert len(job.development_rounds) == 1
        assert job.current_draft is not None

        with patch.object(orch.writer, 'run', return_value=mock_article):
            job = orch.write(job)
        assert job.state == PipelineState.REVIEWING
        assert job.final_article is not None

        with patch.object(orch.reviewer, 'run', return_value=mock_review_approve):
            job = orch.review(job)
        assert job.state == PipelineState.DONE


class TestExtractDois:
    def test_extract_from_strings(self):
        dois = EditorOrchestrator._extract_dois([
            "DOI:10.1038/nat1234",
            "doi:10.1126/sci5678",
            "10.1234/test",
        ])
        assert dois == ["10.1038/nat1234", "10.1126/sci5678", "10.1234/test"]

    def test_extract_from_dicts(self):
        dois = EditorOrchestrator._extract_dois([
            {"doi": "10.1038/nat1234"},
            {"id": "10.1126/sci5678"},
        ])
        assert dois == ["10.1038/nat1234", "10.1126/sci5678"]

    def test_extract_empty(self):
        assert EditorOrchestrator._extract_dois([]) == []
        assert EditorOrchestrator._extract_dois([{}]) == []

    def test_extract_mixed(self):
        dois = EditorOrchestrator._extract_dois([
            "DOI:10.1038/nat1234",
            {"doi": "10.1126/sci5678"},
            "plain text without doi",
        ])
        assert len(dois) == 2
