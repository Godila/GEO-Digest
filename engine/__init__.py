"""GEO-Digest Agent Engine."""
__version__ = "1.0.0"

from engine.config import EngineConfig, get_config
from engine.schemas import (
    Article,
    ArticleGroup,
    StructuredDraft,
    GroupDraft,
    ReviewedDraft,
    WrittenArticle,
    AgentResult,
    JobState,
    DataRequirements,
    InfrastructureNeeds,
    GroupType,
    Severity,
    ReviewVerdict,
    ScoutResult,
)
from engine.agents.base import BaseAgent
from engine.llm.base import LLMProvider
from engine.storage.base import StorageBackend

__all__ = [
    "__version__",
    "EngineConfig",
    "get_config",
    "Article",
    "ArticleGroup",
    "StructuredDraft",
    "GroupDraft",
    "ReviewedDraft",
    "WrittenArticle",
    "AgentResult",
    "JobState",
    "DataRequirements",
    "InfrastructureNeeds",
    "GroupType",
    "Severity",
    "ReviewVerdict",
    "ScoutResult",
    "BaseAgent",
    "LLMProvider",
    "StorageBackend",
]
