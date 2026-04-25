"""
Schemas — Pydantic models for the entire Agent Engine.

All data structures used by agents, storage, and API pass through here.
Central schema definition = single source of truth.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ── Enums ──────────────────────────────────────────────────────

class GroupType(str, Enum):
    """Тип группы статей (классификация scout'а)."""
    REPLICATION = "replication"       # Методическая статья с потенциалом повторения
    REVIEW = "review"                 # Обзорная статья
    DATA_PAPER = "data_paper"         # Статья про датасет


class JobStage(str, Enum):
    """Этап пайплайна."""
    CREATED = "created"
    SCOUTING = "scouting"
    SCOUT_DONE = "scout_done"
    READING = "reading"
    READ_DONE = "read_done"
    WRITING = "writing"
    WRITE_DONE = "write_done"
    REVIEWING = "reviewing"
    REVIEW_DONE = "review_done"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArticleStyle(str, Enum):
    """Формат выходной статьи."""
    REVIEW = "review"                 # Обзорная статья
    REPLICATION = "replication"      # Репликация метода
    DATA_PAPER_STYLE = "data_paper"  # Data paper
    SHORT_COMM = "short_comm"        # Short communication


class ApprovalAction(str, Enum):
    """Действие пользователя на approval gate."""
    APPROVE = "approve"
    SKIP = "skip"
    REJECT = "reject"
    REVISE = "revise"


class Severity(str, Enum):
    """Уровень проблемы в reviewer'е."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class ReviewVerdict(str, Enum):
    """Вердикт reviewer'а."""
    ACCEPT = "ACCEPT"
    ACCEPT_WITH_MINOR = "ACCEPT_WITH_MINOR"
    NEEDS_REVISION = "NEEDS_REVISION"
    REJECT = "REJECT"


# Alias for backward compatibility — JobStatus == JobStage
JobStatus = JobStage


# ── Core: Article ─────────────────────────────────────────────

class Article(dict):
    """
    Статья из базы данных.
    
    Наследуем dict для совместимости с существующим JSONL форматом.
    Все поля optional — данные приходят из разных источников
    с разной полнотой.
    
    Использование:
        art = Article(**data_from_jsonl)
        art.doi → "10.3390/rs15071857"
        art.canonical_id → "doi:10.3390/rs15071857"
        art.scores.total_5 → 4.5
    """
    
    def __getattr__(self, key: str) -> Any:
        """Доступ к полям как атрибутам + safe defaults."""
        if key.startswith("_"):
            raise AttributeError(key)
        val = self.get(key)
        if val is None:
            return "" if key in ("title", "doi", "abstract", "journal", "authors",
                                  "source", "url", "oa_url", "llm_summary") else (
                [] if isinstance(val, list) else (
                {} if isinstance(val, dict) else val))
        # Nested dict access (e.g., scores.total_5)
        if "." in key and isinstance(val, dict):
            parts = key.split(".", 1)
            return val.get(parts[1]) if parts[0] in self else val
        return val
    
    @property
    def canonical_id(self) -> str:
        """Уникальный ID: doi:... или hash:..."""
        doi = (self.get("doi") or "").strip()
        if doi:
            return f"doi:{doi.lower()}"
        from engine.utils import title_hash
        h = title_hash(self.get("title", ""), str(self.get("year", "")))
        return f"hash:{h}"
    
    @property
    def display_title(self) -> str:
        """Заголовок для отображения (ru > en)."""
        return self.get("title_ru") or self.get("title") or "Без названия"
    
    @property
    def is_enriched(self) -> bool:
        """Есть ли LLM enrichment."""
        return bool(self.get("llm_summary"))
    
    @property
    def score_total(self) -> float:
        """Общий балл 0-5."""
        scores = self.get("scores", {})
        if not scores:
            return self.get("_total_score", 0) or 0
        return scores.get("total_5", scores.get("total", 0) * 5)
    
    def to_public_dict(self) -> dict:
        """Словарь для API response (без внутренних полей)."""
        public_keys = {
            "doi", "title", "title_ru", "abstract", "abstract_ru",
            "year", "authors", "journal", "citations", "source",
            "article_type", "is_oa", "oa_url", "topics", "topics_ru",
            "llm_summary", "score_explanations", "scores",
            "_topic_key", "_topic_name_ru",
        }
        return {k: v for k, v in self.items() if k in public_keys}


# ── Scout output: ArticleGroup ──────────────────────────────────

class DataRequirements:
    """Что нужно для репликации эксперимента."""
    input_data: str = ""
    data_format: str = ""
    volume_estimate: str = ""
    acquisition: str = ""
    preprocessing: list[str] = []
    labels_available: bool = False
    split_strategy: str = ""

    def to_dict(self) -> dict:
        return {
            "input_data": self.input_data,
            "data_format": self.data_format,
            "volume_estimate": self.volume_estimate,
            "acquisition": self.acquisition,
            "preprocessing": self.preprocessing,
            "labels_available": self.labels_available,
            "split_strategy": self.split_strategy,
        }


class InfrastructureNeeds:
    """Требования к инфраструктуре."""
    hardware: str = ""
    software: list[str] = []
    compute_time: str = ""
    storage: str = ""
    expertise: list[str] = []

    def to_dict(self) -> dict:
        return {
            "hardware": self.hardware,
            "software": self.software,
            "compute_time": self.compute_time,
            "storage": self.storage,
            "expertise": self.expertise,
        }


class ArticleGroup:
    """
    Группа статей с общим потенциалом.
    
    Результат работы ScoutAgent — не просто список, а классифицированная
    группа с рекомендацией по использованию.
    """
    def __init__(
        self,
        group_type: GroupType | str = GroupType.REVIEW,
        title_suggestion: str = "",
        confidence: float = 0.0,
        articles: list[Article | dict] | None = None,
        rationale: str = "",
        data_requirements: DataRequirements | dict | None = None,
        infrastructure_needs: InfrastructureNeeds | dict | None = None,
        proposed_contribution: str = "",
        estimated_effort: str = "",
        tags: list[str] | None = None,
    ):
        self.group_type = GroupType(group_type) if isinstance(group_type, str) else group_type
        self.title_suggestion = title_suggestion
        self.confidence = confidence
        self.articles = [a if isinstance(a, Article) else Article(a) for a in (articles or [])]
        self.rationale = rationale
        self.data_requirements = (
            data_requirements if isinstance(data_requirements, DataRequirements)
            else DataRequirements(**(data_requirements or {}))
        )
        self.infrastructure_needs = (
            infrastructure_needs if isinstance(infrastructure_needs, InfrastructureNeeds)
            else InfrastructureNeeds(**(infrastructure_needs or {}))
        )
        self.proposed_contribution = proposed_contribution
        self.estimated_effort = estimated_effort
        self.tags = tags or []

    @property
    def article_count(self) -> int:
        return len(self.articles)

    @property
    def article_dois(self) -> list[str]:
        return [a.get("doi", "") for a in self.articles if a.get("doi")]

    def to_dict(self) -> dict:
        return {
            "group_type": self.group_type.value,
            "title_suggestion": self.title_suggestion,
            "confidence": self.confidence,
            "articles": [a.to_public_dict() for a in self.articles],
            "article_count": self.article_count,
            "rationale": self.rationale,
            "data_requirements": self.data_requirements.to_dict(),
            "infrastructure_needs": self.infrastructure_needs.to_dict(),
            "proposed_contribution": self.proposed_contribution,
            "estimated_effort": self.estimated_effort,
            "tags": self.tags,
        }


# ── Reader output: StructuredDraft (полуфабрикат) ───────────────

class StructuredDraft:
    """
    Полуфабрикат статьи — результат глубокого чтения ReaderAgent.
    
    Это ключевой объект системы: именно он идёт на approval,
    затем в WriterAgent, и становится черновиком статьи.
    """
    def __init__(
        self,
        draft_id: str = "",
        group_type: GroupType | str = GroupType.REVIEW,
        source_articles: list[str] | None = None,
        # Общее
        title_suggestion: str = "",
        abstract_suggestion: str = "",
        keywords: list[str] | None = None,
        gap_identified: str = "",
        proposed_contribution: str = "",
        confidence: float = 0.0,
        estimated_effort: str = "",
        # Для replication
        methods_summary: str = "",
        architecture: str = "",
        data_requirements: DataRequirements | dict | None = None,
        infrastructure_needs: InfrastructureNeeds | dict | None = None,
        code_availability: str = "",
        metrics: dict | None = None,
        baseline_comparison: str = "",
        reproducibility_score: float = 0.0,
        # Для review
        scope: str = "",
        articles_covered: int = 0,
        methodology: str = "",
        trends_identified: list[str] | None = None,
        # Для data paper
        dataset_description: str = "",
        access_method: str = "",
        format_: str = "",
        size_gb: float = 0.0,
        coverage: str = "",
        usage_examples: list[str] | None = None,
        # Raw LLM output (for debugging)
        raw_output: str = "",
        # Metadata
        created_at: str = "",
        agent_version: str = "",
    ):
        self.draft_id = draft_id
        self.group_type = GroupType(group_type) if isinstance(group_type, str) else group_type
        self.source_articles = source_articles or []
        # Общее
        self.title_suggestion = title_suggestion
        self.abstract_suggestion = abstract_suggestion
        self.keywords = keywords or []
        self.gap_identified = gap_identified
        self.proposed_contribution = proposed_contribution
        self.confidence = confidence
        self.estimated_effort = estimated_effort
        # Replication
        self.methods_summary = methods_summary
        self.architecture = architecture
        self.data_requirements = (
            data_requirements if isinstance(data_requirements, DataRequirements)
            else DataRequirements(**(data_requirements or {}))
        )
        self.infrastructure_needs = (
            infrastructure_needs if isinstance(infrastructure_needs, InfrastructureNeeds)
            else InfrastructureNeeds(**(infrastructure_needs or {}))
        )
        self.code_availability = code_availability
        self.metrics = metrics or {}
        self.baseline_comparison = baseline_comparison
        self.reproducibility_score = reproducibility_score
        # Review
        self.scope = scope
        self.articles_covered = articles_covered
        self.methodology = methodology
        self.trends_identified = trends_identified or []
        # Data paper
        self.dataset_description = dataset_description
        self.access_method = access_method
        self.format_ = format_
        self.size_gb = size_gb
        self.coverage = coverage
        self.usage_examples = usage_examples or []
        # Meta
        self.raw_output = raw_output
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.agent_version = agent_version

    def to_dict(self) -> dict:
        d = {
            "draft_id": self.draft_id,
            "group_type": self.group_type.value,
            "source_articles": self.source_articles,
            "title_suggestion": self.title_suggestion,
            "abstract_suggestion": self.abstract_suggestion,
            "keywords": self.keywords,
            "gap_identified": self.gap_identified,
            "proposed_contribution": self.proposed_contribution,
            "confidence": self.confidence,
            "estimated_effort": self.estimated_effort,
            # Replication fields
            "methods_summary": self.methods_summary,
            "architecture": self.architecture,
            "data_requirements": self.data_requirements.to_dict(),
            "infrastructure_needs": self.infrastructure_needs.to_dict(),
            "code_availability": self.code_availability,
            "metrics": self.metrics,
            "baseline_comparison": self.baseline_comparison,
            "reproducibility_score": self.reproducibility_score,
            # Review fields
            "scope": self.scope,
            "articles_covered": self.articles_covered,
            "methodology": self.methodology,
            "trends_identified": self.trends_identified,
            # Data paper fields
            "dataset_description": self.dataset_description,
            "access_method": self.access_method,
            "format": self.format_,
            "size_gb": self.size_gb,
            "coverage": self.coverage,
            "usage_examples": self.usage_examples,
            # Meta
            "created_at": self.created_at,
            "agent_version": self.agent_version,
        }
        return d


# ── Writer output: ArticleDraft ─────────────────────────────────

class ArticleDraft:
    """Черновик статьи — результат WriterAgent."""
    def __init__(
        self,
        draft_id: str = "",
        content: str = "",           # Markdown текст
        style: ArticleStyle | str = ArticleStyle.REVIEW,
        language: str = "ru",
        references: list[dict] | None = None,  # [{doi, citation, index}, ...]
        word_count: int = 0,
        source_draft_id: str = "",   # ID входного StructuredDraft
        user_comment: str = "",
        created_at: str = "",
    ):
        self.draft_id = draft_id
        self.content = content
        self.style = ArticleStyle(style) if isinstance(style, str) else style
        self.language = language
        self.references = references or []
        self.word_count = word_count or len(content.split())
        self.source_draft_id = source_draft_id
        self.user_comment = user_comment
        self.created_at = created_at or datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "draft_id": self.draft_id,
            "content": self.content,
            "style": self.style.value,
            "language": self.language,
            "references": self.references,
            "word_count": self.word_count,
            "source_draft_id": self.source_draft_id,
            "user_comment": self.user_comment,
            "created_at": self.created_at,
        }


# ── Reviewer output: ReviewedDraft ──────────────────────────────

class Edit:
    """Одно исправление reviewer'а."""
    def __init__(self, location: str = "", original: str = "", suggested: str = "",
                 reason: str = "", severity: Severity | str = Severity.MINOR):
        self.location = location
        self.original = original
        self.suggested = suggested
        self.reason = reason
        self.severity = Severity(severity) if isinstance(severity, str) else severity

    def to_dict(self) -> dict:
        return {
            "location": self.location,
            "original": self.original,
            "suggested": self.suggested,
            "reason": self.reason,
            "severity": self.severity.value,
        }


class FactCheck:
    """Проверка факта против источника."""
    def __init__(self, claim: str = "", source_doi: str = "", verified: bool = False,
                 correct_text: str = "", note: str = ""):
        self.claim = claim
        self.source_doi = source_doi
        self.verified = verified
        self.correct_text = correct_text
        self.note = note

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "source_doi": self.source_doi,
            "verified": self.verified,
            "correct_text": self.correct_text,
            "note": self.note,
        }


class Issue:
    """Проблема найденная reviewer'ом."""
    def __init__(self, category: str = "", severity: Severity | str = Severity.MINOR,
                 location: str = "", description: str = "", suggestion: str = ""):
        self.category = category
        self.severity = Severity(severity) if isinstance(severity, str) else severity
        self.location = location
        self.description = description
        self.suggestion = suggestion

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "severity": self.severity.value,
            "location": self.location,
            "description": self.description,
            "suggestion": self.suggestion,
        }


VERDICT_ACCEPT = "ACCEPT"
VERDICT_ACCEPT_MINOR = "ACCEPT_WITH_MINOR"
VERDICT_NEEDS_REVISION = "NEEDS_REVISION"
VERDICT_REJECT = "REJECT"


class ReviewedDraft:
    """Результат проверки ReviewerAgent."""
    def __init__(
        self,
        original_text: str = "",
        revised_text: str = "",
        edits: list[Edit] | None = None,
        issues: list[Issue] | None = None,
        fact_checks: list[FactCheck] | None = None,
        verdict: str = VERDICT_NEEDS_REVISION,
        overall_score: float = 0.0,
        severity_counts: dict | None = None,
        reviewer_model: str = "",
        created_at: str = "",
    ):
        self.original_text = original_text
        self.revised_text = revised_text
        self.edits = edits or []
        self.issues = issues or []
        self.fact_checks = fact_checks or []
        self.verdict = verdict
        self.overall_score = overall_score
        self.severity_counts = severity_counts or {}
        self.reviewer_model = reviewer_model
        self.created_at = created_at or datetime.utcnow().isoformat()

    @property
    def critical_count(self) -> int:
        return self.severity_counts.get("critical", sum(1 for i in self.issues if i.severity == Severity.CRITICAL))

    @property
    def major_count(self) -> int:
        return self.severity_counts.get("major", sum(1 for i in self.issues if i.severity == Severity.MAJOR))

    @property
    def minor_count(self) -> int:
        return self.severity_counts.get("minor", sum(1 for i in self.issues if i.severity == Severity.MINOR))

    def to_dict(self) -> dict:
        return {
            "original_text": self.original_text[:500] + "..." if len(self.original_text) > 500 else self.original_text,
            "revised_text": self.revised_text[:500] + "..." if len(self.revised_text) > 500 else self.revised_text,
            "edits": [e.to_dict() for e in self.edits],
            "issues": [i.to_dict() for i in self.issues],
            "fact_checks": [fc.to_dict() for fc in self.fact_checks],
            "verdict": self.verdict,
            "overall_score": self.overall_score,
            "severity_counts": self.severity_counts,
            "critical_count": self.critical_count,
            "major_count": self.major_count,
            "minor_count": self.minor_count,
            "reviewer_model": self.reviewer_model,
            "created_at": self.created_at,
        }


# ── Agent result (generic) ──────────────────────────────────────

class AgentResult:
    """Результат работы любого агента."""
    def __init__(
        self,
        agent_name: str = "",
        success: bool = False,
        data: Any = None,
        error: str = "",
        duration_seconds: float = 0.0,
        metadata: dict | None = None,
    ):
        self.agent_name = agent_name
        self.success = success
        self.data = data          # Typed: ScoutResult | StructuredDraft | ArticleDraft | ReviewedDraft
        self.error = error
        self.duration_seconds = duration_seconds
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        d = {
            "agent_name": self.agent_name,
            "success": self.success,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 1),
            "metadata": self.metadata,
        }
        # Serialize data based on type
        if hasattr(self.data, "to_dict"):
            d["data"] = self.data.to_dict()
        elif isinstance(self.data, list):
            d["data"] = [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in self.data
            ]
        elif isinstance(self.data, dict):
            d["data"] = self.data
        else:
            d["data"] = str(self.data)
        return d


# ── Scout-specific result ───────────────────────────────────────

class ScoutResult:
    """Результат ScoutAgent."""
    def __init__(
        self,
        groups: list[ArticleGroup] | None = None,
        total_found: int = 0,
        after_dedup: int = 0,
        sources_used: list[str] | None = None,
        query: str = "",
        time_range: str = "",
    ):
        self.groups = groups or []
        self.total_found = total_found
        self.after_dedup = after_dedup
        self.sources_used = sources_used or []
        self.query = query
        self.time_range = time_range

    @property
    def best_group(self) -> ArticleGroup | None:
        """Группа с наивысшим confidence."""
        if not self.groups:
            return None
        return max(self.groups, key=lambda g: g.confidence)

    def to_dict(self) -> dict:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "group_count": len(self.groups),
            "total_found": self.total_found,
            "after_dedup": self.after_dedup,
            "sources_used": self.sources_used,
            "query": self.query,
            "time_range": self.time_range,
            "best_group": self.best_group.to_dict() if self.best_group else None,
        }


# ── Job state (orchestrator) ────────────────────────────────────

class JobState:
    """Состояние задачи пайплайна."""
    def __init__(
        self,
        job_id: str = "",
        status: JobStage | str = JobStage.CREATED,
        pipeline: str = "full",           # full / scout_only / read_write / write_only
        input_data: dict | None = None,
        results: dict | None = None,
        approval_history: list[dict] | None = None,
        created_at: str = "",
        updated_at: str = "",
        error: str = "",
    ):
        self.job_id = job_id
        self.status = JobStage(status) if isinstance(status, str) else status
        self.pipeline = pipeline
        self.input_data = input_data or {}
        self.results = results or {
            "scout": None,     # ScoutResult
            "read": None,      # StructuredDraft (or list for group)
            "write": None,     # ArticleDraft
            "review": None,    # ReviewedDraft
        }
        self.approval_history = approval_history or []
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at
        self.error = error

    @property
    def is_running(self) -> bool:
        return self.status in (
            JobStage.SCOUTING, JobStage.READING,
            JobStage.WRITING, JobStage.REVIEWING,
        )

    @property
    def is_paused(self) -> bool:
        return self.status in (
            JobStage.SCOUT_DONE, JobStage.READ_DONE,
            JobStage.WRITE_DONE,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            JobStage.COMPLETE, JobStage.FAILED,
            JobStage.CANCELLED,
        )

    def add_approval(self, stage: str, action: str, detail: str = ""):
        self.approval_history.append({
            "stage": stage,
            "action": action,
            "detail": detail,
            "at": datetime.utcnow().isoformat(),
        })
        self.touch()

    def touch(self):
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        results_serialized = {}
        for k, v in self.results.items():
            if v is None:
                results_serialized[k] = None
            elif hasattr(v, "to_dict"):
                results_serialized[k] = v.to_dict()
            else:
                results_serialized[k] = v

        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "pipeline": self.pipeline,
            "input_data": self.input_data,
            "results": results_serialized,
            "approval_history": self.approval_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "is_terminal": self.is_terminal,
        }

    def to_file(self, path: str):
        """Сохранить в JSON файл."""
        import json
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))


# ── Import Path for type hints ─────────────────────────────────
from pathlib import Path
