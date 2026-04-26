
"""Schemas — All data structures for the Agent Engine."""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any

class GroupType(str, Enum):
    REPLICATION = "replication"
    REVIEW = "review"
    DATA_PAPER = "data_paper"

class JobStatus(str, Enum):
    CREATED = "created"
    SCOUTING = "scouting"
    SCOUT_DONE = "scout_done"
    READING = "reading"
    READ_DONE = "read_done"
    WRITING = "writing"
    WRITE_DONE = "write_done"
    REVIEWING = "reviewing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Severity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"

class ReviewVerdict(str, Enum):
    ACCEPT = "ACCEPT"
    ACCEPT_WITH_MINOR = "ACCEPT_WITH_MINOR"
    NEEDS_REVISION = "NEEDS_REVISION"
    REJECT = "REJECT"

class Article:
    def __init__(self, data): self._data = data
    def __getattr__(self, name):
        if name.startswith("_"): raise AttributeError(name)
        return self._data.get(name)
    def __getitem__(self, key): return self._data.get(key)
    def get(self, key, default=None): return self._data.get(key, default)
    @property
    def data(self): return self._data
    @property
    def id(self): return self._data.get("canonical_id") or self._data.get("_id") or ""
    @property
    def display_title(self): return self._data.get("title_ru") or self._data.get("title") or "Untitled"
    @property
    def total_score(self):
        s = self._data.get("scores", {})
        return s.get("total_5", s.get("total", 0) * 5)
    @property
    def is_enriched(self): return bool(self._data.get("llm_summary"))
    @property
    def doi(self): return self._data.get("doi", "")
    def to_dict(self): return dict(self._data)

class DataRequirements:
    def __init__(self, input_data="", data_format="", volume_estimate="",
                 acquisition="", preprocessing=None, labels_available=False,
                 split_strategy="", geographic_context=""):
        self.input_data = input_data; self.data_format = data_format
        self.volume_estimate = volume_estimate; self.acquisition = acquisition
        self.preprocessing = preprocessing or []; self.labels_available = labels_available
        self.split_strategy = split_strategy; self.geographic_context = geographic_context
    def to_dict(self):
        return {"input_data": self.input_data, "data_format": self.data_format,
                "volume_estimate": self.volume_estimate, "acquisition": self.acquisition,
                "preprocessing": self.preprocessing, "labels_available": self.labels_available,
                "split_strategy": self.split_strategy, "geographic_context": self.geographic_context}

class InfrastructureNeeds:
    def __init__(self, hardware="", software=None, compute_time="", storage="", expertise=None):
        self.hardware = hardware; self.software = software or []
        self.compute_time = compute_time; self.storage = storage; self.expertise = expertise or []
    def to_dict(self):
        return {"hardware": self.hardware, "software": self.software,
                "compute_time": self.compute_time, "storage": self.storage,
                "expertise": self.expertise}

class StructuredDraft:
    def __init__(self, draft_id="", group_type=GroupType.REVIEW, source_articles=None,
                 title_suggestion="", abstract_suggestion="", keywords=None,
                 gap_identified="", proposed_contribution="", confidence=0.0,
                 estimated_effort="", methods_summary="", architecture="",
                 data_requirements=None, infrastructure_needs=None,
                 code_availability="", metrics=None, baseline_comparison="",
                 reproducibility_score=0.0, scope="", articles_covered=0,
                 methodology="", trends_identified=None,
                 dataset_description="", access_method="", format_="",
                 size_gb=0.0, coverage="", usage_examples=None,
                 created_at="", raw_llm_output=""):
        self.draft_id = draft_id
        self.group_type = GroupType(group_type) if isinstance(group_type, str) else group_type
        self.source_articles = source_articles or []
        self.title_suggestion = title_suggestion; self.abstract_suggestion = abstract_suggestion
        self.keywords = keywords or []; self.gap_identified = gap_identified
        self.proposed_contribution = proposed_contribution; self.confidence = confidence
        self.estimated_effort = estimated_effort; self.methods_summary = methods_summary
        self.architecture = architecture; self.data_requirements = data_requirements
        self.infrastructure_needs = infrastructure_needs; self.code_availability = code_availability
        self.metrics = metrics or {}; self.baseline_comparison = baseline_comparison
        self.reproducibility_score = reproducibility_score; self.scope = scope
        self.articles_covered = articles_covered; self.methodology = methodology
        self.trends_identified = trends_identified or []
        self.dataset_description = dataset_description; self.access_method = access_method
        self.format_ = format_; self.size_gb = size_gb; self.coverage = coverage
        self.usage_examples = usage_examples or []
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.raw_llm_output = raw_llm_output
    @property
    def is_replication(self): return self.group_type == GroupType.REPLICATION
    @property
    def is_review(self): return self.group_type == GroupType.REVIEW
    def to_dict(self):
        d = {"draft_id": self.draft_id, "group_type": self.group_type.value,
             "source_articles": self.source_articles, "title_suggestion": self.title_suggestion,
             "abstract_suggestion": self.abstract_suggestion, "keywords": self.keywords,
             "gap_identified": self.gap_identified, "proposed_contribution": self.proposed_contribution,
             "confidence": self.confidence, "estimated_effort": self.estimated_effort,
             "methods_summary": self.methods_summary, "architecture": self.architecture,
             "code_availability": self.code_availability, "metrics": self.metrics,
             "baseline_comparison": self.baseline_comparison, "reproducibility_score": self.reproducibility_score,
             "scope": self.scope, "articles_covered": self.articles_covered,
             "methodology": self.methodology, "trends_identified": self.trends_identified,
             "dataset_description": self.dataset_description, "access_method": self.access_method,
             "format": self.format_, "size_gb": self.size_gb, "coverage": self.coverage,
             "usage_examples": self.usage_examples, "created_at": self.created_at}
        if self.data_requirements: d["data_requirements"] = self.data_requirements.to_dict()
        if self.infrastructure_needs: d["infrastructure_needs"] = self.infrastructure_needs.to_dict()
        return d
    @classmethod
    def from_dict(cls, data):
        dr = data.get("data_requirements"); infra = data.get("infrastructure_needs")
        return cls(draft_id=data.get("draft_id",""), group_type=data.get("group_type","review"),
            source_articles=data.get("source_articles",[]), title_suggestion=data.get("title_suggestion",""),
            abstract_suggestion=data.get("abstract_suggestion",""), keywords=data.get("keywords",[]),
            gap_identified=data.get("gap_identified",""), proposed_contribution=data.get("proposed_contribution",""),
            confidence=data.get("confidence",0.0), estimated_effort=data.get("estimated_effort",""),
            methods_summary=data.get("methods_summary",""), architecture=data.get("architecture",""),
            data_requirements=DataRequirements(**dr) if dr else None,
            infrastructure_needs=InfrastructureNeeds(**infra) if infra else None,
            code_availability=data.get("code_availability",""), metrics=data.get("metrics"),
            baseline_comparison=data.get("baseline_comparison",""),
            reproducibility_score=data.get("reproducibility_score",0.0),
            scope=data.get("scope",""), articles_covered=data.get("articles_covered",0),
            methodology=data.get("methodology",""), trends_identified=data.get("trends_identified",[]),
            dataset_description=data.get("dataset_description",""), access_method=data.get("access_method",""),
            format_=data.get("format",""), size_gb=data.get("size_gb",0.0),
            coverage=data.get("coverage",""), usage_examples=data.get("usage_examples",[]),
            created_at=data.get("created_at",""), raw_llm_output=data.get("raw_llm_output",""))

class GroupDraft:
    def __init__(self, group_id="", group_type=GroupType.REVIEW, title_suggestion="",
                 source_dois=None, individual_drafts=None, aggregated=None, rationale=""):
        self.group_id = group_id
        self.group_type = GroupType(group_type) if isinstance(group_type, str) else group_type
        self.title_suggestion = title_suggestion; self.source_dois = source_dois or []
        self.individual_drafts = individual_drafts or []; self.aggregated = aggregated; self.rationale = rationale
    def to_dict(self):
        return {"group_id": self.group_id, "group_type": self.group_type.value,
                "title_suggestion": self.title_suggestion, "source_dois": self.source_dois,
                "individual_drafts": [d.to_dict() for d in self.individual_drafts],
                "aggregated": self.aggregated.to_dict() if self.aggregated else None,
                "rationale": self.rationale}
    @classmethod
    def from_dict(cls, data):
        ind = [StructuredDraft.from_dict(d) for d in data.get("individual_drafts",[])]
        agg = data.get("aggregated")
        return cls(group_id=data.get("group_id",""), group_type=data.get("group_type","review"),
            title_suggestion=data.get("title_suggestion",""), source_dois=data.get("source_dois",[]),
            individual_drafts=ind, aggregated=StructuredDraft.from_dict(agg) if agg else None,
            rationale=data.get("rationale",""))

class ArticleGroup:
    def __init__(self, group_id="", group_type=GroupType.REVIEW, title_suggestion="",
                 confidence=0.0, articles=None, rationale="", keywords=None, estimated_articles_count=0):
        self.group_id = group_id
        self.group_type = GroupType(group_type) if isinstance(group_type, str) else group_type
        self.title_suggestion = title_suggestion; self.confidence = confidence
        self.articles = articles or []; self.rationale = rationale; self.keywords = keywords or []
        self.estimated_articles_count = estimated_articles_count or len(self.articles)
    @property
    def article_ids(self):
        return [a.id if isinstance(a, Article) else (a.get("canonical_id") or a.get("_id") or "")
                for a in self.articles]
    def to_dict(self):
        arts = [a.to_dict() if isinstance(a, Article) else a for a in self.articles]
        return {"group_id": self.group_id, "group_type": self.group_type.value,
                "title_suggestion": self.title_suggestion, "confidence": self.confidence,
                "articles": arts, "rationale": self.rationale, "keywords": self.keywords,
                "estimated_articles_count": self.estimated_articles_count}
    @classmethod
    def from_dict(cls, data):
        arts = [Article(a) if isinstance(a, dict) else a for a in data.get("articles",[])]
        return cls(group_id=data.get("group_id",""), group_type=data.get("group_type","review"),
            title_suggestion=data.get("title_suggestion",""), confidence=data.get("confidence",0.0),
            articles=arts, rationale=data.get("rationale",""), keywords=data.get("keywords",[]),
            estimated_articles_count=data.get("estimated_articles_count",0))

class ScoutResult:
    def __init__(self, groups=None, total_found=0, after_dedup=0, topic="", query_time_sec=0.0):
        self.groups = groups or []; self.total_found = total_found; self.after_dedup = after_dedup
        self.topic = topic; self.query_time_sec = query_time_sec
    @property
    def best_group(self):
        return max(self.groups, key=lambda g: g.confidence) if self.groups else None
    def to_dict(self):
        return {"groups": [g.to_dict() for g in self.groups], "total_found": self.total_found,
                "after_dedup": self.after_dedup, "topic": self.topic, "query_time_sec": self.query_time_sec}

class WrittenArticle:
    def __init__(self, text="", title="", format_="markdown", language="ru", word_count=0,
                 references=None, sections=None, needs_research_markers=None, metadata=None):
        self.text = text; self.title = title; self.format_ = format_; self.language = language
        self.word_count = word_count; self.references = references or []; self.sections = sections or []
        self.needs_research_markers = needs_research_markers or []; self.metadata = metadata or {}
    def to_dict(self):
        return {"text": self.text, "title": self.title, "format": self.format_, "language": self.language,
                "word_count": self.word_count, "references": self.references, "sections": self.sections,
                "needs_research_markers": self.needs_research_markers, "metadata": self.metadata}

class Edit:
    def __init__(self, location="", severity=Severity.MINOR, original="", suggested="", reason="", category=""):
        self.location = location
        self.severity = Severity(severity) if isinstance(severity, str) else severity
        self.original = original; self.suggested = suggested; self.reason = reason; self.category = category
    def to_dict(self):
        return {"location": self.location, "severity": self.severity.value, "original": self.original,
                "suggested": self.suggested, "reason": self.reason, "category": self.category}

class FactCheck:
    def __init__(self, claim="", source_doi="", verified=False, actual_text="", verdict=""):
        self.claim = claim; self.source_doi = source_doi; self.verified = verified
        self.actual_text = actual_text; self.verdict = verdict
    def to_dict(self):
        return {"claim": self.claim, "source_doi": self.source_doi, "verified": self.verified,
                "actual_text": self.actual_text, "verdict": self.verdict}

class ReviewedDraft:
    def __init__(self, original_text="", revised_text="", edits=None, issues=None,
                 fact_checks=None, severity_counts=None, verdict=ReviewVerdict.ACCEPT_WITH_MINOR,
                 overall_score=0.0, reviewer_model="", summary=""):
        self.original_text = original_text; self.revised_text = revised_text
        self.edits = edits or []; self.issues = issues or []; self.fact_checks = fact_checks or []
        self.severity_counts = severity_counts or {}
        self.verdict = ReviewVerdict(verdict) if isinstance(verdict, str) else verdict
        self.overall_score = overall_score; self.reviewer_model = reviewer_model; self.summary = summary
    @property
    def critical_issues(self): return self.severity_counts.get("critical", 0)
    @property
    def major_issues(self): return self.severity_counts.get("major", 0)
    @property
    def minor_issues(self): return self.severity_counts.get("minor", 0)
    def to_dict(self):
        return {"original_text": self.original_text[:500] + ("..." if len(self.original_text)>500 else ""),
                "revised_text": self.revised_text[:500] + ("..." if len(self.revised_text)>500 else ""),
                "edits": [e.to_dict() for e in self.edits], "issues": [i.to_dict() for i in self.issues],
                "fact_checks": [f.to_dict() for f in self.fact_checks],
                "severity_counts": self.severity_counts, "verdict": self.verdict.value,
                "overall_score": self.overall_score, "reviewer_model": self.reviewer_model, "summary": self.summary}

class AgentResult:
    def __init__(self, agent_name="", success=False, data=None, error="", duration_sec=0.0, tokens_used=0, metadata=None):
        self.agent_name = agent_name; self.success = success; self.data = data; self.error = error
        self.duration_sec = duration_sec; self.tokens_used = tokens_used; self.metadata = metadata or {}
    def to_dict(self):
        d = {"agent_name": self.agent_name, "success": self.success, "error": self.error,
             "duration_sec": self.duration_sec, "tokens_used": self.tokens_used, "metadata": self.metadata}
        if isinstance(self.data, (ScoutResult, GroupDraft, WrittenArticle, ReviewedDraft)):
            d["data"] = self.data.to_dict()
        elif hasattr(self.data, "to_dict"): d["data"] = self.data.to_dict()
        else: d["data"] = self.data
        return d

class JobState:
    def __init__(self, job_id="", status=JobStatus.CREATED, input_topic="", user_comment="",
                 pipeline="full", results=None, approval_history=None, created_at="",
                 updated_at="", selected_group_index=-1, error=""):
        self.job_id = job_id
        self.status = JobStatus(status) if isinstance(status, str) else status
        self.input_topic = input_topic; self.user_comment = user_comment; self.pipeline = pipeline
        self.results = results or {}; self.approval_history = approval_history or []
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at
        self.selected_group_index = selected_group_index; self.error = error
    @property
    def is_running(self):
        return self.status in (JobStatus.SCOUTING, JobStatus.READING, JobStatus.WRITING, JobStatus.REVIEWING)
    @property
    def is_paused(self):
        return self.status in (JobStatus.SCOUT_DONE, JobStatus.READ_DONE, JobStatus.WRITE_DONE)
    @property
    def is_terminal(self):
        return self.status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED)
    def touch(self): self.updated_at = datetime.utcnow().isoformat()
    def add_approval(self, stage, action, detail=""):
        self.approval_history.append({"stage": stage, "action": action, "detail": detail, "at": datetime.utcnow().isoformat()})
        self.touch()
    def set_result(self, stage, data):
        self.results[stage] = data.to_dict() if hasattr(data, "to_dict") else data
        self.touch()
    def get_result(self, stage, cls=None):
        raw = self.results.get(stage)
        if raw is None: return None
        if cls and hasattr(cls, "from_dict"): return cls.from_dict(raw)
        return raw
    def to_dict(self):
        return {"job_id": self.job_id, "status": self.status.value,
                "input": {"topic": self.input_topic, "comment": self.user_comment, "pipeline": self.pipeline},
                "results": self.results, "approval_history": self.approval_history,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "selected_group_index": self.selected_group_index, "error": self.error}
    @classmethod
    def from_dict(cls, data):
        inp = data.get("input", {})
        return cls(job_id=data.get("job_id",""), status=data.get("status","created"),
            input_topic=inp.get("topic",""), user_comment=inp.get("comment",""),
            pipeline=inp.get("pipeline","full"), results=data.get("results",{}),
            approval_history=data.get("approval_history",[]), created_at=data.get("created_at",""),
            updated_at=data.get("updated_at",""), selected_group_index=data.get("selected_group_index",-1),
            error=data.get("error",""))
