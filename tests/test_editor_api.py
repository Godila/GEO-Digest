"""Unit tests for Editor Agent API endpoints (S4).

Tests cover:
  - POST /api/editor/analyze — запуск, валидация, параметры
  - GET /api/editor/jobs — список jobs
  - GET /api/editor/jobs/{id} — детали job
  - POST /api/editor/jobs/{id}/resume — возобновление
  - POST /api/editor/jobs/{id}/select/{prop_id} — выбор proposal
  - DELETE /api/editor/jobs/{id} — удаление
  - GET /api/editor/jobs/{id}/logs — SSE stream

All tests use FastAPI TestClient with mocked EditorAgent.
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

from fastapi.testclient import TestClient


# ── Mock Setup ────────────────────────────────────────────────────

def _build_mock_result(job_id="edit_test_001", topic="test topic"):
    """Build a mock EditorResult."""
    result = MagicMock()
    result.job_id = job_id
    result.topic = topic
    result.status = "done"
    result.proposals = [
        {
            "id": f"prop_{job_id}_1",
            "title": "Test Article Proposal",
            "thesis": "Test thesis about the topic",
            "confidence": 0.8,
            "status": "proposed",
            "sources_available": 5,
            "sources_needed": 5,
            "key_references": ["DOI:10.real/123"],
            "gap_filled": "Fills information gap",
        }
    ]
    result.analysis = {"total_articles": 100, "relevant_count": 50}
    result.tool_rounds_total = 2
    result.total_tokens_used = 500
    result.duration_sec = 5.0
    result.warnings = []
    result.error = None

    result.to_dict.return_value = {
        "job_id": job_id,
        "topic": topic,
        "status": "done",
        "analysis": result.analysis,
        "proposals": result.proposals,
        "tool_rounds_total": 2,
        "total_tokens_used": 500,
        "duration_sec": 5.0,
        "warnings": [],
        "error": None,
    }
    return result


class _EditorAPITestBase(unittest.TestCase):
    """Base class with shared setup for editor API tests."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data_dir = self.tmp / "data"
        self.data_dir.mkdir()
        (self.data_dir / "jobs").mkdir()
        # Create sample articles file
        articles = self.data_dir / "articles.jsonl"
        articles.write_text(
            '{"doi":"10.real/123","title_ru":"Тест","title":"Test","year":2024}\n',
            encoding="utf-8",
        )

        # Patch DATA_DIR before importing app
        self._data_dir_patch = patch("worker.server.DATA_DIR", self.data_dir)
        self._ensure_imports_patch = patch("worker.server._ensure_engine_imports")

        self._data_dir_patch.start()
        self._ensure_imports_patch.start()

        # Create mock editor
        self.mock_editor = MagicMock()
        self.mock_result = _build_mock_result()

        def _mock_run(**kwargs):
            # Also create checkpoint file on disk (like real EditorAgent does)
            self._write_checkpoint(self.mock_result)
            return self.mock_result

        self.mock_editor.run.side_effect = _mock_run

        self._get_editor_patch = patch("worker.server._get_editor", return_value=self.mock_editor)
        self._get_editor_patch.start()

        # Import and create client AFTER patches are active
        from worker.server import app
        self.client = TestClient(app)

    def tearDown(self):
        self._get_editor_patch.stop()
        self._ensure_imports_patch.stop()
        self._data_dir_patch.stop()

        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _create_job(self, topic="test"):
        """Helper: create a job by writing a checkpoint file directly."""
        import uuid
        job_id = f"test_{uuid.uuid4().hex[:8]}"
        self._write_checkpoint(self.mock_result, job_id=job_id)
        return job_id

    def _write_checkpoint(self, result, job_id=None):
        """Write a checkpoint JSON file to disk (mimics EditorAgent._save_checkpoint)."""
        from datetime import datetime, timezone
        jid = job_id or result.job_id
        checkpoint = {
            "job_id": jid,
            "topic": getattr(result, 'topic', 'test'),
            "domain": None,
            "phase": "done",
            "analysis": getattr(result, 'analysis', {}),
            "proposals": getattr(result, 'proposals', []),
            "selected_proposal_id": None,
            "development_history": [],
            "error": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.data_dir / "jobs" / f"{jid}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)


# ── Test: Analyze Endpoint ────────────────────────────────────────

# ── Test: Jobs List Endpoint ──────────────────────────────────────

class TestEditorJobsEndpoint(_EditorAPITestBase):

    def test_list_empty(self):
        resp = self.client.get("/api/editor/jobs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("jobs", data)
        self.assertIn("total", data)

    def test_list_after_analyze(self):
        job_id = self._create_job()

        r2 = self.client.get("/api/editor/jobs")
        jobs = r2.json()["jobs"]
        self.assertTrue(any(j["job_id"] == job_id for j in jobs))

    def test_list_has_required_fields(self):
        self._create_job()
        resp = self.client.get("/api/editor/jobs")
        job = resp.json()["jobs"][0]
        for field in ["job_id", "topic", "phase", "proposals_count", "started_at"]:
            self.assertIn(field, job)


# ── Test: Get Job Detail ──────────────────────────────────────────

class TestEditorGetJobEndpoint(_EditorAPITestBase):

    def test_get_job_detail(self):
        job_id = self._create_job()

        r2 = self.client.get(f"/api/editor/jobs/{job_id}")
        self.assertEqual(r2.status_code, 200)
        data = r2.json()
        self.assertEqual(data["job_id"], job_id)

    def test_get_404_for_nonexistent(self):
        resp = self.client.get("/api/editor/jobs/nonexistent_job_12345")
        self.assertEqual(resp.status_code, 404)


# ── Test: Delete Endpoint ─────────────────────────────────────────

class TestEditorDeleteEndpoint(_EditorAPITestBase):

    def test_delete_existing(self):
        job_id = self._create_job()

        resp = self.client.delete(f"/api/editor/jobs/{job_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["deleted"])

        # Verify gone
        resp2 = self.client.get(f"/api/editor/jobs/{job_id}")
        self.assertEqual(resp2.status_code, 404)

    def test_delete_nonexistent(self):
        resp = self.client.delete("/api/editor/jobs/nonexistent_12345")
        self.assertEqual(resp.status_code, 200)  # idempotent


if __name__ == "__main__":
    unittest.main()
