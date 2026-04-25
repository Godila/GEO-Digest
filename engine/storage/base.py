"""
Storage backend — abstract interface.

Engine can work with JSONL (current), SQLite (future), or any other
storage through this abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from engine.schemas import Article, StructuredDraft, ArticleDraft


class StorageBackend(ABC):
    """
    Abstract storage for articles and drafts.
    
    All data operations go through here. Agents never touch files directly.
    """
    
    # ── Articles ──
    
    @abstractmethod
    def load_all_articles(self) -> list[Article]:
        """Load all articles from database."""
        ...
    
    @abstractmethod
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        """Get single article by canonical ID or DOI."""
        ...
    
    @abstractmethod
    def search_articles(
        self,
        query: str = "",
        topic: str = "",
        source: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Article], int]:
        """Search articles with filters. Returns (articles, total_count)."""
        ...
    
    @abstractmethod
    def save_article(self, article: Article | dict) -> None:
        """Save/append a single article."""
        ...
    
    @abstractmethod
    def count_articles(self) -> int:
        """Total number of articles."""
        ...
    
    # ── Graph ──
    
    @abstractmethod
    def load_graph(self) -> dict:
        """Load graph data {nodes: [], edges: [], metadata?: {}}."""
        ...
    
    @abstractmethod
    def save_graph(self, graph_data: dict) -> None:
        """Save graph data."""
        ...
    
    # ── Drafts (agent outputs) ──
    
    @abstractmethod
    def save_draft(self, draft: StructuredDraft | ArticleDraft, job_id: str = "") -> Path:
        """Save a draft to output directory. Returns path."""
        ...
    
    @abstractmethod
    def load_draft(self, draft_id: str) -> Optional[StructuredDraft | ArticleDraft]:
        """Load a draft by ID."""
        ...
    
    @abstractmethod
    def list_drafts(self, job_id: str = "") -> list[dict]:
        """List available drafts. Optionally filter by job_id."""
        ...
    
    # ── Job state ──
    
    @abstractmethod
    def save_job_state(self, state_dict: dict, job_id: str) -> None:
        """Persist job state to disk."""
        ...
    
    @abstractmethod
    def load_job_state(self, job_id: str) -> Optional[dict]:
        """Load job state from disk."""
        ...
    
    @abstractmethod
    def list_jobs(self, active_only: bool = False) -> list[dict]:
        """List jobs. If active_only, only non-terminal states."""
        ...
    
    # ── Health / Info ──
    
    @abstractmethod
    def get_stats(self) -> dict:
        """Return storage statistics."""
        ...


def get_storage(data_dir: Optional[Path] = None) -> StorageBackend:
    """Factory: get storage backend instance."""
    from engine.config import get_config
    from engine.storage.jsonl_backend import JsonlStorage
    
    cfg = get_config()
    return JsonlStorage(data_dir=data_dir or cfg.data_dir)
