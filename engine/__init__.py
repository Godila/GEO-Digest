"""
GEO-Digest Agent Engine
=======================

Standalone pipeline for:
  Scout → Read → [APPROVAL] → Write → Review

Usage:
    from engine import Orchestrator, ScoutAgent, ReaderAgent, WriterAgent
    
    job = Orchestrator.run(topic="ML for earthquake prediction")
    print(job.result)
"""

__version__ = "1.0.0"

from engine.config import Config as EngineConfig
from engine.schemas import (
    Article, ArticleGroup, StructuredDraft, GroupDraft,
    AgentResult, JobState, JobStatus,
    DataRequirements, InfrastructureNeeds,
    GroupType, Severity, ReviewVerdict,
    WrittenArticle, ReviewedDraft, ScoutResult,
)
from engine.agents.base import BaseAgent
from engine.llm.base import LLMProvider
from engine.storage.base import StorageBackend
from engine.orchestrator import Orchestrator

__all__ = [
    "__version__",
    "EngineConfig",
    "Article", "ArticleGroup", "StructuredDraft", "GroupDraft",
    "AgentResult", "JobState", "JobStatus",
    "DataRequirements", "InfrastructureNeeds",
    "GroupType", "Severity", "ReviewVerdict",
    "WrittenArticle", "ReviewedDraft", "ScoutResult",
    "BaseAgent", "LLMProvider", "StorageBackend",
    "Orchestrator",
]
