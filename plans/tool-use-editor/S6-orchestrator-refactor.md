# Стадия S6: Orchestrator Refactor

## Цель
Переделать `engine/orchestrator.py` из linear pipeline в state machine,
который использует EditorAgent как основной entry point, а Reader/Writer/Reviewer
как downstream агентов.

## Архитектура до / после

```
ДО (linear):                    ПОСЛЕ (state machine):
                                  
Scout → [approve] →           EditorAgent
Reader → [approve] →            ↓
Writer → [approve] →          Proposals → User picks
Reviewer → DONE                ↓
                              Selected proposal
                                 ↓
                           ┌────┴────┐
                           ↓         ↓
                        Reader    Writer
                           ↓         ↓
                        [approve]  Reviewer
                           ↓         ↓
                           └────┬────┘
                                ↓
                              DONE
```

## State Machine

```
states:
  idle        → начальное состояние
  editing     → EditorAgent работает (анализ + предложения)
  selecting   → пользователь выбирает предложение
  developing  → итеративная доработка (Reader + user feedback)
  writing     → Writer генерирует финальную статью
  reviewing   → Reviewer проверяет
  done        → успешно завершён
  failed      → ошибка
  cancelled   → отменено пользователем

transitions:
  idle ──run(topic)──→ editing
  editing ──proposals_ready──→ selecting
  selecting ──select(prop)──→ developing
  developing ──user_feedback──→ developing  (loop!)
  developing ──satisfied──→ writing
  writing ──done──→ reviewing
  reviewing ──approved──→ done
  reviewing ──needs_work──→ developing  (back!)
  * ──cancel──→ cancelled
  * ──error──→ failed
```

## Что делаем

### 6.1 New Orchestrator
Файл: `engine/orchestrator_v2.py` (новый, старый оставляем как legacy)

```python
from enum import Enum
from dataclasses import dataclass, field
from engine.agents.editor import EditorAgent, EditorResult
from engine.agents.reader import ReaderAgent
from engine.agents.writer import WriterAgent  
from engine.agents.reviewer import ReviewerAgent

class PipelineState(Enum):
    IDLE = "idle"
    EDITING = "editing"       # Editor работает
    SELECTING = "selecting"   # Ждём выбор пользователя
    DEVELOPING = "developing" # Итеративная доработка
    WRITING = "writing"       # Пишет финальную статью
    REVIEWING = "reviewing"   # Ревью
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class PipelineJob:
    job_id: str
    topic: str
    domain: str | None = None
    state: PipelineState = PipelineState.IDLE
    
    # Editor results
    editor_result: dict | None = None
    selected_proposal_id: str | None = None
    
    # Development (iterative)
    development_rounds: list[dict] = field(default_factory=list)
    current_draft: dict | None = None
    
    # Final output
    final_article: dict | None = None
    review_result: dict | None = None
    
    # Metadata
    created_at: str = ""
    updated_at: str = ""
    error: str | None = None


class EditorOrchestrator:
    """Оркестратор v2: state machine с Editor Agent."""
    
    def __init__(self, jobs_dir: str = "/app/data/jobs"):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        self.editor = EditorAgent()
        self.reader = ReaderAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewerAgent()
    
    def create_job(self, topic: str, domain: str = None,
                   user_comment: str = None) -> PipelineJob:
        """Создаёт новый pipeline job."""
        job = PipelineJob(
            job_id=self._gen_job_id(),
            topic=topic,
            domain=domain,
            state=PipelineState.EDITING,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        self._save_job(job)
        return job
    
    def run_editing_phase(self, job: PipelineJob) -> PipelineJob:
        """Запускает Editor Agent (Phase 1+2)."""
        job.state = PipelineState.EDITING
        self._save_job(job)
        
        try:
            result = self.editor.run(
                topic=job.topic,
                domain=job.domain,
                max_proposals=5,
            )
            job.editor_result = asdict(result)
            job.state = PipelineState.SELECTING
        except Exception as e:
            job.state = PipelineState.FAILED
            job.error = str(e)
        
        job.updated_at = now_iso()
        self._save_job(job)
        return job
    
    def select_proposal(self, job: PipelineJob, prop_id: str) -> PipelineJob:
        """Пользователь выбирает предложение."""
        if not job.editor_result:
            raise ValueError("Editor hasn't completed yet")
        
        proposals = job.editor_result.get("proposals", [])
        selected = next((p for p in proposals if p.get("id") == prop_id), None)
        if not selected:
            raise ValueError(f"Proposal {prop_id} not found")
        
        job.selected_proposal_id = prop_id
        job.state = PipelineState.DEVELOPING
        job.updated_at = now_iso()
        self._save_job(job)
        return job
    
    def develop(self, job: PipelineJob, user_feedback: str = "") -> PipelineJob:
        """Итеративная доработка. Вызывает Reader для глубины."""
        job.state = PipelineState.DEVELOPING
        
        proposal = self._get_selected_proposal(job)
        
        # Читаем источники через Reader
        if proposal and proposal.get("key_references"):
            dois = [r.replace("DOI:", "").strip() 
                   for r in proposal["key_references"]]
            
            try:
                draft = self.reader.run(dois=dois, topic=job.topic)
                job.current_draft = asdict(draft) if hasattr(draft, '__dict__') else draft
            except Exception as e:
                self._log(f"[orch] Reader error: {e}")
                job.current_draft = {"error": str(e), "proposal": proposal}
        
        # Сохраняем round
        job.development_rounds.append({
            "round": len(job.development_rounds) + 1,
            "user_feedback": user_feedback or "(initial)",
            "draft_summary": str(type(job.current_draft)),
            "timestamp": now_iso(),
        })
        
        job.updated_at = now_iso()
        self._save_job(job)
        return job
    
    def write(self, job: PipelineJob) -> PipelineJob:
        """Генерация финальной статьи через Writer."""
        job.state = PipelineState.WRITING
        
        proposal = self._get_selected_proposal(job)
        draft = job.current_draft
        
        try:
            article = self.writer.run(
                topic=proposal.get("title", job.topic),
                thesis=proposal.get("thesis", ""),
                draft_data=draft,
                references=proposal.get("key_references", []),
            )
            job.final_article = asdict(article) if hasattr(article, '__dict__') else article
            job.state = PipelineState.REVIEWING
        except Exception as e:
            job.error = str(e)
            job.state = PipelineState.FAILED
        
        job.updated_at = now_iso()
        self._save_job(job)
        return job
    
    def review(self, job: PipelineJob) -> PipelineJob:
        """Ревью финальной статьи."""
        job.state = PipelineState.REVIEWING
        
        if not job.final_article:
            raise ValueError("No article to review")
        
        try:
            result = self.reviewer.run(
                article=job.final_article,
                references=self._get_selected_proposal(job).get("key_references", []),
            )
            job.review_result = asdict(result) if hasattr(result, '__dict__') else result
            
            verdict = result.verdict if hasattr(result, 'verdict') else result.get('verdict', '')
            if verdict == "approve":
                job.state = PipelineState.DONE
            elif verdict in ("needs_revision", "reject"):
                job.state = PipelineState.DEVELOPING  # Back to development!
            else:
                job.state = PipelineState.DONE  # Default: accept
                
        except Exception as e:
            job.error = str(e)
            job.state = PipelineState.FAILED
        
        job.updated_at = now_iso()
        self._save_job(job)
        return job
    
    def cancel(self, job: PipelineJob) -> PipelineJob:
        job.state = PipelineState.CANCELLED
        job.updated_at = now_iso()
        self._save_job(job)
        return job
    
    def load_job(self, job_id: str) -> PipelineJob:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            raise ValueError(f"Job {job_id} not found")
        with open(path) as f:
            data = json.load(f)
        return PipelineJob(**{k: v for k, v in data.items() 
                             if k in PipelineJob.__dataclass_fields__})
    
    def list_jobs(self) -> list[dict]:
        jobs = []
        for f in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            with open(f) as fh:
                data = json.load(fh)
            # Фильтруем только v2 jobs (имеют state поле)
            if "state" in data:
                jobs.append({
                    "job_id": data.get("job_id", f.stem),
                    "topic": data.get("topic", ""),
                    "state": data.get("state", "unknown"),
                    "selected_proposal": data.get("selected_proposal_id"),
                    "has_final_article": bool(data.get("final_article")),
                    "created_at": data.get("created_at", ""),
                })
        return jobs
    
    # ── Private ──
    
    def _get_selected_proposal(self, job):
        if not job.editor_result or not job.selected_proposal_id:
            return None
        for p in job.editor_result.get("proposals", []):
            if p.get("id") == job.selected_proposal_id:
                return p
        return None
    
    def _save_job(self, job: PipelineJob):
        path = self.jobs_dir / f"{job.job_id}.json"
        with open(path, 'w') as f:
            json.dump(asdict(job), f, ensure_ascii=False, indent=2, default=str)
    
    @staticmethod
    def _gen_job_id() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + \
               secrets.token_hex(4)
```

### 6.2 API Endpoints для State Machine
Добавить в worker/server.py:

```python
_orchestrator_v2 = EditorOrchestrator()

@app.post("/api/pipeline/job")
async def pipeline_create_job(request: Request):
    body = await request.json()
    job = _orchestrator_v2.create_job(
        topic=body.get("topic", ""),
        domain=body.get("domain"),
        user_comment=body.get("comment"),
    )
    # Автоматически запускаем editing phase
    job = _orchestrator_v2.run_editing_phase(job)
    return {"job_id": job.job_id, "state": job.state.value}

@app.post("/api/pipeline/jobs/{id}/select/{prop_id}")
async def pipeline_select(id, prop_id):
    job = _orchestrator_v2.load_job(id)
    job = _orchestrator_v2.select_proposal(job, prop_id)
    return {"state": job.state.value}

@app.post("/api/pipeline/jobs/{id}/develop")
async def pipeline_develop(id, request):
    body = await request.json() or {}
    job = _orchestrator_v2.load_job(id)
    job = _orchestrator_v2.develop(job, body.get("feedback", ""))
    return {"state": job.state.value, "round": len(job.development_rounds)}

@app.post("/api/pipeline/jobs/{id}/write")
async def pipeline_write(id):
    job = _orchestrator_v2.load_job(id)
    job = _orchestrator_v2.write(job)
    return {"state": job.state.value}

@app.post("/api/pipeline/jobs/{id}/review")
async def pipeline_review(id):
    job = _orchestrator_v2.load_job(id)
    job = _orchestrator_v2.review(job)
    return {"state": job.state.value, "verdict": job.review_result.get("verdict") if job.review_result else None}

@app.delete("/api/pipeline/jobs/{id}")
async def pipeline_cancel(id):
    job = _orchestrator_v2.load_job(id)
    job = _orchestrator_v2.cancel(job)
    return {"state": job.state.value}
```

## Acceptance Criteria S6

- [ ] `EditorOrchestrator.create_job()` создаёт PipelineJob с state=EDITING
- [ ] `run_editing_phase()` вызывает EditorAgent и сохраняет proposals
- [ ] `run_editing_phase()` при ошибке → state=FAILED
- [ ] `select_proposal()` меняет state на DEVELOPING
- [ ] `select_proposal()` с невалидным prop_id → ValueError
- [ ] `develop()` вызывает Reader и сохраняет draft
- [ ] `develop()` добавляет round в development_rounds
- [ ] `write()` вызывает Writer и сохраняет final_article
- [ ] `write()` при ошибке → state=FAILED
- [ ] `review()` вызывает Reviewer
- [ ] `review()` с verdict=approve → state=DONE
- [ ] `review()` с verdict=needs_revision → state=DEVELOPING (back!)
- [ ] `cancel()` → state=CANCELLED
- [ ] `load_job()` восстанавливает из JSON файла
- [ ] `list_jobs()` возвращает только v2 jobs (с полем state)
- [ ] Все методы сохраняют checkpoint после изменений
- [ ] Старый orchestrator.py не изменён (backward compat)

## Тесты S6

### Unit тесты

**test_orchestrator_v2.py**
```python
class TestPipelineStateMachine:
    @fixture
    def orch(tmp_path):
        return EditorOrchestrator(jobs_dir=str(tmp_path / "jobs"))
    
    def test_create_job(orch):
        job = orch.create_job(topic="Arctic methane")
        assert job.job_id.startswith("20")
        assert job.topic == "Arctic methane"
        assert job.state == PipelineState.EDITING
        assert Path(orch.jobs_dir / f"{job.job_id}.json").exists()
    
    def test_full_happy_path(orch, mock_agents):
        """Полный путь: edit → select → develop → write → review → done."""
        mock_agents.editor.return_value = mock_editor_result(proposals=[
            mock_proposal(id="p1", confidence=0.9)
        ])
        mock_agents.reader.return_value = mock_draft()
        mock_agents.writer.return_value = mock_article()
        mock_agents.reviewer.return_value = mock_review(verdict="approve")
        
        job = orch.create_job(topic="test")
        job = orch.run_editing_phase(job)
        assert job.state == PipelineState.SELECTING
        
        job = orch.select_proposal(job, "p1")
        assert job.state == PipelineState.DEVELOPING
        
        job = orch.develop(job)
        assert job.state == PipelineState.DEVELOPING
        assert len(job.development_rounds) == 1
        assert job.current_draft is not None
        
        job = orch.write(job)
        assert job.state == PipelineState.REVIEWING
        assert job.final_article is not None
        
        job = orch.review(job)
        assert job.state == PipelineState.DONE
    
    def test_review_sends_back_to_develop(orch, mock_agents):
        mock_agents.reviewer.return_value = mock_review(verdict="needs_revision")
        
        job = _setup_job_at_writing_stage(orch)
        job = orch.review(job)
        assert job.state == PipelineState.DEVELOPING  # Back!
    
    def test_cancel_at_any_state(orch):
        for state in [PipelineState.EDITING, PipelineState.DEVELOPING, 
                      PipelineState.WRITING, PipelineState.SELECTING]:
            job = PipelineJob(job_id="test", topic="t", state=state)
            job = orch.cancel(job)
            assert job.state == PipelineState.CANCELLED
    
    def test_select_invalid_proposal(orch):
        job = PipelineJob(job_id="test", topic="t", 
                          editor_result={"proposals": [{"id": "p1"}]})
        with pytest.raises(ValueError):
            orch.select_proposal(job, "nonexistent")
    
    def test_write_without_selection(orch):
        job = PipelineJob(job_id="test", topic="t", state=PipelineState.DEVELOPING)
        with pytest.raises(ValueError, match="No article"):  # или другая ошибка
            orch.write(job)
    
    def test_develop_accumulates_rounds(orch, mock_agents):
        mock_agents.reader.return_value = mock_draft()
        
        job = _setup_job_at_developing(orch)
        orch.develop(job, "first feedback")
        orch.develop(job, "second feedback")
        
        assert len(job.development_rounds) == 2
        assert job.development_rounds[0]["user_feedback"] == "first feedback"
        assert job.development_rounds[1]["user_feedback"] == "second feedback"
    
    def test_load_job_persists_state(orch):
        original = orch.create_job(topic="persist me")
        original.state = PipelineState.DONE
        
        loaded = orch.load_job(original.job_id)
        assert loaded.job_id == original.job_id
        assert loaded.state == PipelineState.DONE
        assert loaded.topic == original.topic
    
    def test_list_jobs_filters_v2(orch):
        orch.create_job(topic="v2 job")  # has state field
        
        # Создаём вручную v1 job (без state)
        v1_path = orch.jobs_dir / "v1_old_job.json"
        v1_path.write_text('{"job_id":"v1","topic":"old","phase":"done"}')
        
        jobs = orch.list_jobs()
        assert len(jobs) == 1  # Only v2
        assert jobs[0]["job_id"].startswith("20")

class TestErrorHandling:
    def test_editor_failure(orch, mock_agents):
        mock_agents.editor.side_effect = RuntimeError("LLM timeout")
        
        job = orch.create_job(topic="test")
        job = orch.run_editing_phase(job)
        assert job.state == PipelineState.FAILED
        assert "timeout" in job.error.lower()
    
    def test_writer_failure(orch):
        job = _setup_job_at_developing(orch)
        # Mock writer to fail
        orch.writer.run = Mock(side_effect=ValueError("Bad draft"))
        
        job = orch.write(job)
        assert job.state == PipelineState.FAILED
```

## Файлы стадии S6

| Файл | Действие | Строк |
|------|----------|-------|
| `engine/orchestrator_v2.py` | Создать | ~300 |
| `worker/server.py` | Изменить (+pipeline endpoints) | +120 |
| `dashboard/app.py` | Изменить (+pipeline proxy) | +50 |
| `tests/test_orchestrator_v2.py` | Создать | ~280 |

**Итого:** ~750 строк
