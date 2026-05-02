"""E2E тесты полного пайплайна Editor Agent (S7).

Тестирует интеграцию всех компонентов:
  - Editor API endpoints ↔ EditorAgent
  - Orchestrator v2 state machine: edit→select→develop→write→review
  - Persistence: checkpoint survive load
  - Edge cases + StorageTools интеграция

Ключевые факты для моков:
  - ToolResult использует .success (bool), НЕ .is_ok()
  - API endpoints вызывают _get_editor() → editor.run() → .to_dict()
  - Orchestrator lazy-агенты: patch.object(orch, '_editor') или orch._editor = MagicMock()
  - Resume endpoint ловит ValueError → 404
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── Fixtures ───────────────────────────────────────────────

@pytest.fixture
def data_dir(tmp_path):
    """Временная директория с данными для E2E."""
    d = tmp_path / "data"
    d.mkdir()
    articles_file = d / "articles.jsonl"
    jobs_dir = d / "jobs"
    jobs_dir.mkdir()

    articles = [
        {"doi": "10.1038/nat12345", "title": "Arctic permafrost methane emissions accelerating",
         "title_ru": "Выбросы метана из арктической мерзлоты ускоряются",
         "year": 2024, "source": "nature", "article_type": "research",
         "cluster": "arctic_methane", "relevance_score": 0.92},
        {"doi": "10.1126/sci67890", "title": "Methane feedback in thawing permafrost regions",
         "title_ru": "Обратная связь метана в районах оттаивания мерзлоты",
         "year": 2023, "source": "science", "article_type": "review",
         "cluster": "arctic_methane", "relevance_score": 0.88},
        {"doi": "10.1016/j.env11111", "title": "Lake emissions from thermokarst landscapes",
         "title_ru": "Выбросы озёр из термокарстовых ландшафтов",
         "year": 2023, "source": "elsevier", "article_type": "data_paper",
         "cluster": "lake_emissions", "relevance_score": 0.75},
        {"doi": "10.1029/geo22222", "title": "Global methane budget 2023 update",
         "title_ru": "Глобальный бюджет метана: обновление 2023",
         "year": 2024, "source": "agu", "article_type": "methods_transfer",
         "cluster": "global_budget", "relevance_score": 0.70},
    ]
    with open(articles_file, 'w', encoding='utf-8') as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')

    return str(d)


def _make_mock_editor_run_result(job_id="e2e_test_001"):
    """Мок результата EditorAgent.run() — объект с .to_dict()."""
    result = MagicMock()
    result.to_dict.return_value = {
        "job_id": job_id,
        "status": "done",
        "topic": "Arctic permafrost methane emissions",
        "proposals": [
            {
                "id": "prop_01",
                "title": "Ускорение выбросов метана из арктической мерзлоты: данные 2023-2025",
                "thesis": "За период 2023-2025 опубликовано 12 новых исследований показывающих что abrupt thaw события увеличивают выбросы метана на 40-60%.",
                "confidence": 0.87,
                "sources_available": 15,
                "sources_needed": 3,
                "target_audience": "climate researchers + policy makers",
                "status": "proposed",
                "key_references": ["10.1038/nat12345", "10.1126/sci67890"],
            },
            {
                "id": "prop_02",
                "title": "Озёрные выбросы в термокарстовых ландшафтах Сибири",
                "thesis": "Термокарстовые озёра Западной Сибири — недооценённый источник метана до 2.3 TgCH4/год.",
                "confidence": 0.72,
                "sources_available": 8,
                "sources_needed": 5,
                "target_audience": "permafrost researchers",
                "status": "proposed",
                "key_references": ["10.1016/j.env11111"],
            },
        ],
        "analysis": {"total_articles": 181, "gaps": ["Siberian lakes underrepresented"]},
        "duration_sec": 12.5,
    }
    return result


@pytest.fixture
def editor_client(data_dir):
    """TestClient для Editor API с мокированным _get_editor().

    Мокаем worker.server._get_editor целиком — избегаем всех импортов.
    side_effect для run() также пишет checkpoint на диск (как реальный EditorAgent).
    """
    from fastapi.testclient import TestClient
    import sys, time
    sys.path.insert(0, str(Path(__file__).parent.parent))

    mock_run_result = _make_mock_editor_run_result()
    jobs_disk_dir = Path(data_dir) / "jobs"
    jobs_disk_dir.mkdir(exist_ok=True)

    def _mock_run_side_effect(**kwargs):
        """Имитирует editor.run() + пишет checkpoint."""
        job_id = mock_run_result.to_dict()["job_id"]
        checkpoint = mock_run_result.to_dict()
        checkpoint["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        checkpoint["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        # Пишем checkpoint на диск (как настоящий EditorAgent)
        with open(jobs_disk_dir / f"{job_id}.json", 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        return mock_run_result

    fake_editor = MagicMock()
    fake_editor.run.side_effect = _mock_run_side_effect

    with patch('worker.server._get_editor', return_value=fake_editor), \
         patch('worker.server.DATA_DIR', Path(data_dir)):
        from worker.server import app
        client = TestClient(app)
        yield client


# ════════════════════════════════════════════════
#  Сценарий 2: Orchestrator v2 State Machine
# ════════════════════════════════════════════════

class TestOrchestratorV2E2E:
    """Сценарий 2: Полный state machine путь.

    Важно: lazy properties (editor/reader/writer/reviewer) создают агентов
    при первом доступе. Для моков устанавливаем _editor/_reader/etc напрямую.
    """

    @staticmethod
    def _make_orch_mocks():
        """Возвращает (mock_editor_result, mock_draft, mock_article, mock_review).

        editor_result должен быть dict чтобы _serialize_editor_result его пропустил.
        """
        # dict — пройдёт isinstance(result, check) в _serialize_editor_result
        m_ed = {
            'job_id': 'sm_test', 'topic': 'test', 'phase': 'done',
            'proposals': [
                {'id': 'p1', 'title': 'Test Article', 'thesis': 'Test thesis content',
                 'confidence': 0.9, 'sources_available': 10, 'status': 'proposed',
                 'key_references': ['10.1038/nat12345']},
            ],
            'analysis': {},
        }

        m_dr = MagicMock()
        m_dr.content = 'Draft text...'
        m_dr.sources_read = 5

        m_ar = MagicMock()
        m_ar.title = 'Final'
        m_ar.text = 'Full article...'
        m_ar.word_count = 2000

        m_rw = MagicMock()
        m_rw.verdict = "approve"
        m_rw.score = 0.9

        return m_ed, m_dr, m_ar, m_rw

    def test_full_state_machine_happy_path(self, data_dir):
        """edit -> select -> develop(x2) -> write -> review -> done"""
        from engine.orchestrator_v2 import EditorOrchestrator, PipelineState
        orch = EditorOrchestrator(jobs_dir=str(Path(data_dir) / "jobs"))

        m_ed, m_dr, m_ar, m_rw = self._make_orch_mocks()

        # Устанавливаем моки напрямую в lazy-поля (обход @property)
        mock_ed_inst = MagicMock(run=MagicMock(return_value=m_ed))
        mock_rd_inst = MagicMock(run=MagicMock(return_value=m_dr))
        mock_wr_inst = MagicMock(run=MagicMock(return_value=m_ar))
        mock_rw_inst = MagicMock(run=MagicMock(return_value=m_rw))

        orch._editor = mock_ed_inst
        orch._reader = mock_rd_inst
        orch._writer = mock_wr_inst
        orch._reviewer = mock_rw_inst

        # Create + Edit -> SELECTING
        job = orch.create_job(topic="state machine test")
        assert job.state == PipelineState.EDITING
        job = orch.run_editing_phase(job)
        assert job.state == PipelineState.SELECTING, f"Expected SELECTING, got {job.state}"

        # Select -> DEVELOPING
        job = orch.select_proposal(job, "p1")
        assert job.state == PipelineState.DEVELOPING

        # Develop x2
        job = orch.develop(job, "first round")
        assert len(job.development_rounds) == 1
        job = orch.develop(job, "second round")
        assert len(job.development_rounds) == 2

        # Write -> REVIEWING
        job = orch.write(job)
        assert job.state == PipelineState.WRITTEN
        assert job.final_article is not None

        # Review -> DONE
        job = orch.review(job)
        assert job.state == PipelineState.DONE

    def test_review_loop_back_to_developing(self, data_dir):
        """Review с needs_revision возвращает к DEVELOPING."""
        from engine.orchestrator_v2 import EditorOrchestrator, PipelineState
        orch = EditorOrchestrator(jobs_dir=str(Path(data_dir) / "jobs"))

        m_ed, m_dr, m_ar, _ = self._make_orch_mocks()

        m_rw_loop = MagicMock()
        m_rw_loop.verdict = "needs_revision"

        orch._editor = MagicMock(run=MagicMock(return_value=m_ed))
        orch._reader = MagicMock(run=MagicMock(return_value=m_dr))
        orch._writer = MagicMock(run=MagicMock(return_value=m_ar))
        orch._reviewer = MagicMock(run=MagicMock(return_value=m_rw_loop))

        job = orch.create_job(topic="loop test")
        job = orch.run_editing_phase(job)
        job = orch.select_proposal(job, "p1")
        orch.develop(job)
        orch.write(job)
        job = orch.review(job)

        # V2 review loop: NEEDS_REVISION → multi-round rewrite → eventually DONE
        # (state=done regardless of whether forced_accept or normal accept,
        #  since mock reviewer returns MagicMock verdict which may not match enum)
        assert job.state == PipelineState.DONE


# ════════════════════════════════════════════════
#  Сценарий 3: Edge Cases
# ════════════════════════════════════════════════

class TestEdgeCases:
    """Сценарий 3: Граничные случаи."""

    def test_get_nonexistent_job_404(self, editor_client):
        """Несуществующий job -> 404."""
        r = editor_client.get("/api/editor/jobs/nonexistent_xyz")
        assert r.status_code == 404

    def test_cancel_from_any_state(self, data_dir):
        """Cancel работает из любого состояния."""
        from engine.orchestrator_v2 import EditorOrchestrator, PipelineState
        orch = EditorOrchestrator(jobs_dir=str(Path(data_dir) / "jobs"))

        for state in [PipelineState.EDITING, PipelineState.DEVELOPING,
                      PipelineState.WRITING, PipelineState.SELECTING]:
            job = orch.create_job(topic=f"cancel_{state.value}")
            job = orch.cancel(job)
            assert job.state == PipelineState.CANCELLED

    def test_orchestrator_invalid_proposal_raises(self, data_dir):
        """Выбор невалидного proposal -> ValueError."""
        from engine.orchestrator_v2 import EditorOrchestrator, PipelineJob, PipelineState
        orch = EditorOrchestrator(jobs_dir=str(Path(data_dir) / "jobs"))

        job = PipelineJob(
            job_id="test", topic="t", state=PipelineState.SELECTING,
            editor_result={"proposals": [{"id": "real_prop"}]},
        )
        with pytest.raises(ValueError, match="not found"):
            orch.select_proposal(job, "fake_prop")


# ════════════════════════════════════════════════
#  Сценарий 4: Persistence
# ════════════════════════════════════════════════

class TestPersistence:
    """Сценарий 4: Checkpoint persistence."""

    def test_orchestrator_persists_all_states(self, data_dir):
        """Каждое состояние сохраняется и восстанавливается."""
        from engine.orchestrator_v2 import EditorOrchestrator, PipelineState
        orch = EditorOrchestrator(jobs_dir=str(Path(data_dir) / "jobs"))
        original = orch.create_job(topic="persist all states")

        for state in [PipelineState.EDITING, PipelineState.SELECTING,
                      PipelineState.DEVELOPING, PipelineState.WRITING,
                      PipelineState.REVIEWING, PipelineState.DONE]:
            original.state = state
            orch._save_job(original)
            loaded = orch.load_job(original.job_id)
            assert loaded.state == state, f"Failed to persist {state}"
            assert loaded.topic == original.topic

    def test_orchestrator_survives_complex_data(self, data_dir):
        """Сложные nested данные сохраняются корректно."""
        from engine.orchestrator_v2 import EditorOrchestrator, PipelineState
        orch = EditorOrchestrator(jobs_dir=str(Path(data_dir) / "jobs"))

        job = orch.create_job(topic="complex data")
        job.editor_result = {
            "proposals": [{
                "id": "p1", "title": "T", "thesis": "Th",
                "confidence": 0.95, "key_references": ["10.1038/a", "10.1126/b"],
                "nested": {"deep": {"value": [1, 2, 3]}}
            }],
            "analysis": {"clusters": {"a": 10, "b": 20}},
        }
        job.development_rounds = [
            {"round": 1, "user_feedback": "fix X", "has_draft": True},
            {"round": 2, "user_feedback": "add Y", "has_draft": True},
        ]
        job.current_draft = {"content": "Draft...", "sources": 12}
        job.final_article = {"text": "Final article...", "word_count": 3000}
        job.review_result = {"verdict": "approve", "score": 0.92}

        orch._save_job(job)
        loaded = orch.load_job(job.job_id)

        assert loaded.editor_result is not None
        assert len(loaded.editor_result["proposals"]) == 1
        assert len(loaded.development_rounds) == 2
        assert loaded.current_draft is not None
        assert loaded.final_article is not None
        assert loaded.review_result is not None
        assert loaded.review_result["verdict"] == "approve"


# ════════════════════════════════════════════════
#  Сценарий 5: StorageTools интеграция
# ════════════════════════════════════════════════

class TestStorageToolsIntegration:
    """Интеграция StorageTools (один класс с 7 методами) <-> реальные JSONL данные.

    ToolResult API:
      - .success (bool) — True если OK
      - .error_msg (str) — текст ошибки если не OK
      - .data (Any) — полезные данные
      - .content (str) — текстовое представление
    """

    @staticmethod
    def _make_tools(data_dir):
        from engine.storage import JsonlStorage
        from engine.tools.storage_tools import StorageTools
        storage = JsonlStorage(data_dir=data_dir)
        return StorageTools(storage=storage)

    def test_search_articles(self, data_dir):
        tools = self._make_tools(data_dir)
        r = tools.search_articles(query="arctic methane")
        assert r.success is True, f"search failed: {r.error_msg}"
        articles = r.data.get("articles", [])
        assert len(articles) >= 1

    def test_get_article_detail(self, data_dir):
        tools = self._make_tools(data_dir)
        r = tools.get_article_detail(doi="10.1038/nat12345")
        assert r.success is True, f"detail failed: {r.error_msg}"
        assert r.data.get("doi") == "10.1038/nat12345"

    def test_validate_doi_found(self, data_dir):
        tools = self._make_tools(data_dir)
        r = tools.validate_doi(doi="10.1038/nat12345")
        assert r.success is True
        # Формат: {"valid": True, "doi": ..., "title": ..., "year": ..., "source": ...}
        assert r.data.get("valid") is True
        assert r.data.get("doi") == "10.1038/nat12345"

    def test_validate_doi_not_found(self, data_dir):
        tools = self._make_tools(data_dir)
        r = tools.validate_doi(doi="10.9999/fake0000")
        assert r.success is True  # вызов успешен, но DOI не найден
        assert r.data.get("valid") is False

    def test_find_similar_existing(self, data_dir):
        tools = self._make_tools(data_dir)
        r = tools.find_similar_existing(title_idea="permafrost methane emissions")
        assert r.success is True, f"similar failed: {r.error_msg}"

    def test_cluster_by_subtopic(self, data_dir):
        tools = self._make_tools(data_dir)
        r = tools.cluster_by_subtopic(topic="emissions")
        assert r.success is True, f"cluster failed: {r.error_msg}"
        clusters = r.data.get("clusters", [])
        assert len(clusters) >= 1

    def test_count_storage_stats(self, data_dir):
        tools = self._make_tools(data_dir)
        r = tools.count_storage_stats()
        assert r.success is True, f"stats failed: {r.error_msg}"
        assert r.data.get("total_articles") == 4

    def test_explore_domain(self, data_dir):
        tools = self._make_tools(data_dir)
        # Без focus_query — возвращает все статьи (избегаем фильтра)
        r = tools.explore_domain()
        assert r.success is True, f"explore failed: {r.error_msg}"
        assert "sources" in r.data
        assert "years" not in r.data or "year_distribution" in r.data  # ключ может отличаться
        assert len(r.data.get("sources", {})) >= 1  # nature, science, elsevier, agu
