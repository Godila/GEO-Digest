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
import logging
import secrets
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from engine.schemas import ReviewVerdict

logger = logging.getLogger(__name__)


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
    """Pipeline job — один полный цикл от темы до статьи.

    V2 fields (Proactive Reviewer):
      - review_history: list of all review rounds (for multi-pass revision)
      - revision_instructions: current instructions from Reviewer → Writer
      - total_review_rounds: how many review rounds were executed
      - forced_accept: whether the article was auto-accepted after max rounds
    """
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

    # V2: Multi-round review support
    review_history: list = field(default_factory=list)
    revision_instructions: str = ""
    total_review_rounds: int = 0
    forced_accept: bool = False

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    error: Optional[str] = None


class EditorOrchestrator:
    """Оркестратор v2: state machine с Editor Agent.

    Использует EditorAgent как основной entry point.
    Reader/Writer/Reviewer — downstream агенты.
    """

    def __init__(self, jobs_dir: str = "/app/data/jobs",
                 storage=None, llm_provider=None):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        # Dependencies for agent creation — passed from server.py
        self._storage = storage
        self._llm_provider = llm_provider

        # Lazy imports — агенты инициализируются при первом использовании
        self._editor = None
        self._reader = None
        self._writer = None
        self._reviewer = None

    @property
    def editor(self):
        if self._editor is None:
            from engine.agents.editor import EditorAgent
            self._editor = EditorAgent(
                storage=self._storage,
                llm=self._llm_provider,
                jobs_dir=str(self.jobs_dir),
            )
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
        """Запускает Editor Agent (анализ + генерация предложений).

        После генерации proposals обогащает их данными из графа знаний
        (centrality, bridges, hubs) если граф доступен.
        """
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

            # S8.3: Graph enrichment — дополнить proposals graph данными
            self._enrich_with_graph(job)

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

    # ── Constants for Review Loop ──────────────────────────────────

    MAX_REVISION_ROUNDS = 3

    def review(self, job: PipelineJob) -> PipelineJob:
        """Ревью финальной статьи через Reviewer (v2: multi-round revision loop).

        V2 Logic:
          1. Load reviewer LLM separately from writer LLM
          2. Run up to MAX_REVISION_ROUNDS (3) review cycles
          3. After each review:
             - ACCEPT or ACCEPT_WITH_MINOR → DONE
             - NEEDS_REVISION and round < MAX → send to Writer for rewrite
             - Round == MAX → forced accept (with flag)
          4. Each round's result is stored in job.review_history
          5. Bug fix: verdict comparison uses ReviewVerdict enum values, not "approve"
        """
        from engine.agents.article_patterns import REVISION_CONFIG

        job.state = PipelineState.REVIEWING

        if not job.final_article:
            logger.warning(f"[orch] No article yet for review (job {job.job_id}, state={job.state})")
            # Write phase may still be running — don't fail, just return
            job.error = "Review called before article was written"
            job.updated_at = now_iso()
            self._save_job(job)
            return job

        max_rounds = REVISION_CONFIG.get("max_rounds", self.MAX_REVISION_ROUNDS)
        previous_reviews = []
        forced_accept = False

        try:
            for round_num in range(1, max_rounds + 1):
                logger.info(f"[orch] Review round {round_num}/{max_rounds} for job {job.job_id}")

                # Run reviewer for this round
                result = self.reviewer.run(
                    article=job.final_article,
                    references=self._get_selected_proposal(job).get("key_references", [])
                              if self._get_selected_proposal(job) else [],
                    round_number=round_num,
                    previous_reviews=previous_reviews,
                )

                if not result.success:
                    job.error = f"Reviewer error in round {round_num}: {result.error}"
                    job.state = PipelineState.FAILED
                    job.updated_at = now_iso()
                    self._save_job(job)
                    return job

                reviewed = result.data  # ReviewedDraft

                # Store in history
                review_dict = reviewed.to_dict() if hasattr(reviewed, 'to_dict') else \
                    self._serialize_review(reviewed)
                job.review_history.append(review_dict)
                job.review_result = review_dict
                job.total_review_rounds = round_num
                job.updated_at = now_iso()
                self._save_job(job)

                # Determine action based on verdict
                verdict = reviewed.verdict if hasattr(reviewed, 'verdict') else None
                verdict_value = verdict.value if verdict else ""

                # BUG FIX: was comparing to "approve" string — now uses proper enum values
                if verdict_value in (ReviewVerdict.ACCEPT.value, ReviewVerdict.ACCEPT_WITH_MINOR.value):
                    # Article passed!
                    job.state = PipelineState.DONE
                    job.forced_accept = False
                    logger.info(
                        f"[orch] Job {job.job_id} ACCEPTED after round {round_num} "
                        f"(verdict={verdict_value}, score={reviewed.overall_score:.2f})"
                    )
                    break

                elif verdict_value == ReviewVerdict.NEEDS_REVISION.value:
                    if round_num < max_rounds:
                        # Send edits back to Writer for revision
                        logger.info(
                            f"[orch] Round {round_num}: NEEDS_REVISION "
                            f"(score={reviewed.overall_score:.2f}) → sending to Writer"
                        )

                        # Build revision instructions using reviewer's helper
                        if hasattr(self.reviewer, '_build_revision_instructions'):
                            job.revision_instructions = \
                                self.reviewer._build_revision_instructions(reviewed)
                        elif hasattr(reviewed, 'revision_instructions'):
                            job.revision_instructions = reviewed.revision_instructions
                        else:
                            job.revision_instructions = (
                                f"Revise article based on review feedback.\n"
                                f"Issues: {reviewed.issues}\n"
                                f"Edits: {len(reviewed.edits)} changes required."
                            )

                        # Transition back to WRITING state for rewrite
                        job.state = PipelineState.WRITING
                        job.updated_at = now_iso()
                        self._save_job(job)

                        # Rewrite the article with revision instructions
                        job = self._rewrite_article(job)

                        # Update previous_reviews for next round context
                        previous_reviews.append(reviewed)

                    else:
                        # Max rounds reached — forced accept
                        forced_accept = True
                        job.forced_accept = True
                        job.state = PipelineState.DONE
                        logger.info(
                            f"[orch] Job {job.job_id}: FORCED ACCEPT after {max_rounds} rounds "
                            f"(score={reviewed.overall_score:.2f})"
                        )
                        break

                elif verdict_value == ReviewVerdict.REJECT.value:
                    # Proactive critic: REJECT = needs major revision, not failure
                    # Send back to Writer unless at max rounds
                    if round_num < max_rounds:
                        job.revision_instructions = self._format_revision_edits(reviewed)
                        logger.info(
                            f"[orch] Round {round_num}: REJECT "
                            f"(score={reviewed.overall_score:.2f}) → sending to Writer for rewrite"
                        )
                        job = self._rewrite_article(job)
                        # _rewrite_article sets state to REVIEWING
                        # Loop continues to next round
                    else:
                        # Max rounds reached — forced accept with warning
                        reviewed.forced_accept = True
                        job.forced_accept = True
                        job.review_result = self._serialize_review(reviewed)
                        job.state = PipelineState.DONE
                        logger.info(
                            f"[orch] Job {job.job_id}: FORCED ACCEPT after REJECT "
                            f"(score={reviewed.overall_score:.2f}, {max_rounds} rounds)"
                        )

                else:
                    # Unknown verdict — treat as accept (conservative)
                    job.state = PipelineState.DONE
                    logger.info(
                        f"[orch] Unknown verdict '{verdict_value}' → accepting "
                        f"(round {round_num})"
                    )
                    break

        except Exception as e:
            job.error = f"Reviewer error: {e}"
            job.state = PipelineState.FAILED

        job.updated_at = now_iso()
        self._save_job(job)
        return job

    def _rewrite_article(self, job: PipelineJob) -> PipelineJob:
        """Send article back to Writer with revision instructions.

        Called during the review loop when verdict is NEEDS_REVISION.
        Updates job.final_article with the rewritten version.
        """
        proposal = self._get_selected_proposal(job)
        if not proposal:
            logger.info("[orch] Cannot rewrite: no proposal selected, keeping original")
            return job

        try:
            article = self.writer.run(
                topic=proposal.get("title", job.topic),
                thesis=proposal.get("thesis", ""),
                draft_data=job.current_draft,
                references=proposal.get("key_references", []),
                revision_instructions=job.revision_instructions,  # Pass review feedback
            )
            job.final_article = self._serialize_article(article)
            job.state = PipelineState.REVIEWING  # Back to reviewing after rewrite
            logger.info(f"[orch] Article rewritten, returning to REVIEW state")
        except Exception as e:
            logger.info(f"[och] Rewrite failed: {e}, keeping current version")
            # Don't fail the whole job — keep existing article
            job.state = PipelineState.REVIEWING

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

    def _enrich_with_graph(self, job: PipelineJob) -> None:
        """Обогащает proposals данными из графа знаний.

        Для каждого key_reference в каждом proposal добавляет:
        - page_rank, betweenness, community
        - hub/bridge статус

        Также ищет кросс-тематические мосты для темы job.
        """
        try:
            from engine.tools.graph_tools import GraphTools
            gt = GraphTools()

            proposals = (job.editor_result or {}).get("proposals", [])
            for prop in proposals:
                graph_context = []
                dois = self._extract_dois(prop.get("key_references", []))
                for doi in dois:
                    cr = gt.graph_centrality(doi)
                    if cr.success and cr.data:
                        graph_context.append({
                            "doi": doi,
                            "page_rank": round(cr.data.get("page_rank", 0), 4),
                            "betweenness": round(cr.data.get("betweenness", 0), 4),
                            "role": cr.data.get("role", "unknown"),
                            "degree": cr.data.get("degree", 0),
                            "is_hub": cr.data.get("is_hub", False),
                            "is_bridge": cr.data.get("is_bridge", False),
                        })
                if graph_context:
                    prop["graph_context"] = graph_context

            # Cross-topic bridges discovery
            if job.topic:
                # Extract potential sub-topics from the topic string
                words = job.topic.split()
                if len(words) >= 2:
                    bridges_result = gt.graph_cross_topic(words[0], words[-1])
                    if bridges_result.success and bridges_result.data:
                        job.editor_result["cross_topic_bridges"] = [
                            {"label": b["label"], "bridge_score": b["bridge_score"],
                             "direct_mention": b["direct_mention"]}
                            for b in bridges_result.data.get("bridges", [])[:5]
                        ]

        except Exception as e:
            logger.warning(f"[orch] Graph enrichment skipped: {e}")

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

    def _format_revision_edits(self, reviewed) -> str:
        """Convert review edits into writer-friendly revision instructions."""
        parts = [f"## Ревизия (round {reviewed.round_number}, score={reviewed.overall_score:.2f})\n"]
        
        if reviewed.verdict:
            parts.append(f"**Вердикт:** {reviewed.verdict.value}\n")
        
        # Critical issues first
        for edit in (reviewed.edits or []):
            sev = getattr(edit, 'severity', '?')
            desc = getattr(edit, 'description', str(edit))
            location = getattr(edit, 'location', '')
            loc_str = f" [{location}]" if location else ""
            parts.append(f"- **[{sev.upper()}]{loc_str}** {desc}")
        
        # Suggestions
        for s in (reviewed.improvement_suggestions or []):
            parts.append(f"- 💡 {s}")
        
        # Fact check failures
        for fc in (reviewed.fact_checks or []):
            if not getattr(fc, 'is_supported', True):
                claim = getattr(fc, 'claim', str(fc))[:100]
                parts.append(f"- ⚠️ Факт-чек: \"{claim}\" — не подтверждено источниками")
        
        return '\n'.join(parts)

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
