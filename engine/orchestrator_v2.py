"""Orchestrator v2 — State Machine для GEO-Digest pipeline.

Заменяет linear pipeline (Scout→Reader→Writer→Reviewer) на state machine
с Editor Agent как основным entry point.

States:
  idle → editing → selecting → developing → writing → reviewing → done
                                    ↑                      ↓
                              (user feedback)     (needs_revision)
                                    └──────────────────────┘

Transitions:
  idle ──run(topic)────────→ editing
  editing ──proposals_ready─→ selecting
  selecting ──select(prop)─→ developing
  developing ──feedback────→ developing (loop!)
  developing ──satisfied───→ writing
  writing ──done──────────→ reviewing
  reviewing ──approved────→ done
  reviewing ──needs_work──→ developing (back!)
  * ──cancel─────────────→ cancelled
  * ──error──────────────→ failed
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class PipelineState(str, Enum):
    """Состояния pipeline job."""
    IDLE = "idle"
    EDITING = "editing"         # Editor работает (анализ + предложения)
    SELECTING = "selecting"     # Ждём выбор пользователя
    DEVELOPING = "developing"   # Итеративная доработка (Reader + feedback)
    WRITING = "writing"         # Writer генерирует финальную статью
    REVIEWING = "reviewing"     # Reviewer проверяет
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineJob:
    """Pipeline job — один полный цикл от темы до статьи."""
    job_id: str
    topic: str
    domain: Optional[str] = None
    user_comment: Optional[str] = None
    state: PipelineState = PipelineState.IDLE

    # Editor results
    editor_result: Optional[dict] = None
    selected_proposal_id: Optional[str] = None

    # Development (iterative)
    development_rounds: list = field(default_factory=list)
    current_draft: Optional[dict] = None

    # Final output
    final_article: Optional[dict] = None
    review_result: Optional[dict] = None

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    error: Optional[str] = None


class EditorOrchestrator:
    """Оркестратор v2: state machine с Editor Agent.

    Использует EditorAgent как основной entry point.
    Reader/Writer/Reviewer — downstream агенты.
    """

    def __init__(self, jobs_dir: str = "/app/data/jobs"):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        # Lazy imports — агенты инициализируются при первом использовании
        self._editor = None
        self._reader = None
        self._writer = None
        self._reviewer = None

    @property
    def editor(self):
        if self._editor is None:
            from engine.agents.editor import EditorAgent
            self._editor = EditorAgent()
        return self._editor

    @property
    def reader(self):
        if self._reader is None:
            from engine.agents.reader import ReaderAgent
            self._reader = ReaderAgent()
        return self._reader

    @property
    def writer(self):
        if self._writer is None:
            from engine.agents.writer import WriterAgent
            self._writer = WriterAgent()
        return self._writer

    @property
    def reviewer(self):
        if self._reviewer is None:
            from engine.agents.reviewer import ReviewerAgent
            self._reviewer = ReviewerAgent()
        return self._reviewer

    # ── Job Lifecycle ──────────────────────────────────────

    def create_job(self, topic: str, domain: str = None,
                   user_comment: str = None) -> PipelineJob:
        """Создаёт новый pipeline job в состоянии EDITING."""
        job = PipelineJob(
            job_id=self._gen_job_id(),
            topic=topic,
            domain=domain,
            user_comment=user_comment,
            state=PipelineState.EDITING,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        self._save_job(job)
        return job

    def run_editing_phase(self, job: PipelineJob) -> PipelineJob:
        """Запускает Editor Agent (анализ + генерация предложений)."""
        job.state = PipelineState.EDITING
        self._save_job(job)

        try:
            result = self.editor.run(
                topic=job.topic,
                domain=job.domain,
                user_instruction=job.user_comment,
                max_proposals=5,
            )
            # Сохраняем результат editor как dict
            job.editor_result = self._serialize_editor_result(result)
            job.state = PipelineState.SELECTING
        except Exception as e:
            job.state = PipelineState.FAILED
            job.error = f"Editor error: {e}"

        job.updated_at = now_iso()
        self._save_job(job)
        return job

    def select_proposal(self, job: PipelineJob, prop_id: str) -> PipelineJob:
        """Пользователь выбирает предложение → переход к DEVELOPING."""
        proposals = (job.editor_result or {}).get("proposals", [])
        selected = next((p for p in proposals if p.get("id") == prop_id), None)
        if not selected:
            raise ValueError(f"Proposal '{prop_id}' not found in {len(proposals)} proposals")

        job.selected_proposal_id = prop_id
        job.state = PipelineState.DEVELOPING
        job.updated_at = now_iso()
        self._save_job(job)
        return job

    def develop(self, job: PipelineJob, user_feedback: str = "") -> PipelineJob:
        """Итеративная доработка: читаем источники через Reader.

        Можно вызывать многократно — каждый вызов добавляет round.
        """
        job.state = PipelineState.DEVELOPING
        proposal = self._get_selected_proposal(job)

        draft_data = None
        if proposal and proposal.get("key_references"):
            dois = self._extract_dois(proposal["key_references"])
            try:
                draft = self.reader.run(dois=dois, topic=job.topic)
                draft_data = self._serialize_draft(draft)
                job.current_draft = draft_data
            except Exception as e:
                job.current_draft = {"error": str(e), "proposal_summary": proposal.get("thesis", "")[:200]}
        else:
            job.current_draft = {"note": "No key references to read"}

        # Сохраняем round
        job.development_rounds.append({
            "round": len(job.development_rounds) + 1,
            "user_feedback": user_feedback or "(initial)",
            "has_draft": job.current_draft is not None,
            "timestamp": now_iso(),
        })

        job.updated_at = now_iso()
        self._save_job(job)
        return job

    def write(self, job: PipelineJob) -> PipelineJob:
        """Генерация финальной статьи через Writer."""
        job.state = PipelineState.WRITING
        proposal = self._get_selected_proposal(job)

        if not proposal:
            raise ValueError("Cannot write: no proposal selected")

        try:
            article = self.writer.run(
                topic=proposal.get("title", job.topic),
                thesis=proposal.get("thesis", ""),
                draft_data=job.current_draft,
                references=proposal.get("key_references", []),
            )
            job.final_article = self._serialize_article(article)
            job.state = PipelineState.REVIEWING
        except Exception as e:
            job.error = f"Writer error: {e}"
            job.state = PipelineState.FAILED

        job.updated_at = now_iso()
        self._save_job(job)
        return job

    def review(self, job: PipelineJob) -> PipelineJob:
        """Ревью финальной статьи через Reviewer."""
        job.state = PipelineState.REVIEWING

        if not job.final_article:
            raise ValueError("No article to review")

        try:
            result = self.reviewer.run(
                article=job.final_article,
                references=self._get_selected_proposal(job).get("key_references", []) if self._get_selected_proposal(job) else [],
            )
            job.review_result = self._serialize_review(result)

            verdict = getattr(result, 'verdict', None) or (result or {}).get('verdict', '')
            if verdict == "approve":
                job.state = PipelineState.DONE
            elif verdict in ("needs_revision", "reject"):
                job.state = PipelineState.DEVELOPING  # Back to development!
            else:
                job.state = PipelineState.DONE  # Default: accept
        except Exception as e:
            job.error = f"Reviewer error: {e}"
            job.state = PipelineState.FAILED

        job.updated_at = now_iso()
        self._save_job(job)
        return job

    def cancel(self, job: PipelineJob) -> PipelineJob:
        """Отмена job из любого состояния."""
        job.state = PipelineState.CANCELLED
        job.updated_at = now_iso()
        self._save_job(job)
        return job

    # ── Persistence ────────────────────────────────────────

    def load_job(self, job_id: str) -> PipelineJob:
        """Загружает job из JSON файла."""
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            raise ValueError(f"Job '{job_id}' not found")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Восстанавливаем enum значения
        if isinstance(data.get("state"), str):
            try:
                data["state"] = PipelineState(data["state"])
            except ValueError:
                data["state"] = PipelineState.IDLE
        return PipelineJob(**{k: v for k, v in data.items()
                              if k in PipelineJob.__dataclass_fields__})

    def list_jobs(self, limit: int = 20) -> list[dict]:
        """Список v2 jobs (только те что имеют поле state)."""
        jobs = []
        for f in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            if len(jobs) >= limit:
                break
            try:
                with open(f, encoding="utf-8") as fh:
                    data = json.load(fh)
                # Фильтруем только v2 jobs (имеют state поле с правильным значением)
                if "state" in data and data["state"] in (s.value for s in PipelineState):
                    jobs.append({
                        "job_id": data.get("job_id", f.stem),
                        "topic": data.get("topic", ""),
                        "state": data.get("state", "unknown"),
                        "selected_proposal": data.get("selected_proposal_id"),
                        "proposals_count": len((data.get("editor_result") or {}).get("proposals", [])),
                        "has_final_article": bool(data.get("final_article")),
                        "development_rounds": len(data.get("development_rounds", [])),
                        "created_at": data.get("created_at", ""),
                        "updated_at": data.get("updated_at", ""),
                        "error": data.get("error"),
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        return jobs

    # ── Private Helpers ────────────────────────────────────

    def _get_selected_proposal(self, job: PipelineJob) -> Optional[dict]:
        if not job.editor_result or not job.selected_proposal_id:
            return None
        for p in job.editor_result.get("proposals", []):
            if p.get("id") == job.selected_proposal_id:
                return p
        return None

    def _save_job(self, job: PipelineJob):
        path = self.jobs_dir / f"{job.job_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(job), f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _gen_job_id() -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)

    @staticmethod
    def _serialize_editor_result(result) -> dict:
        """Конвертирует EditorResult в dict."""
        if hasattr(result, '__dataclass_fields__'):
            return asdict(result)
        if isinstance(result, dict):
            return result
        return {"raw": str(result)}

    @staticmethod
    def _serialize_draft(draft) -> dict:
        if hasattr(draft, '__dataclass_fields__'):
            return asdict(draft)
        if isinstance(draft, dict):
            return draft
        return {"content": str(draft)}

    @staticmethod
    def _serialize_article(article) -> dict:
        if hasattr(article, '__dataclass_fields__'):
            return asdict(article)
        if isinstance(article, dict):
            return article
        return {"text": str(article)}

    @staticmethod
    def _serialize_review(review) -> dict:
        if hasattr(review, '__dataclass_fields__'):
            return asdict(review)
        if isinstance(review, dict):
            return review
        return {"verdict": str(review)}

    @staticmethod
    def _extract_dois(references: list) -> list[str]:
        """Извлекает DOI из списка ссылок."""
        import re
        doi_pattern = re.compile(r'^10\.\d{4,}/\S+')
        dois = []
        for ref in references:
            if isinstance(ref, str):
                # Убираем префиксы DOI:/doi:
                clean = ref.replace("DOI:", "").replace("doi:", "").strip()
                if doi_pattern.match(clean):
                    dois.append(clean)
            elif isinstance(ref, dict):
                doi = ref.get("doi", ref.get("id", ""))
                if doi and doi_pattern.match(doi.strip()):
                    dois.append(doi.strip())
        return dois
