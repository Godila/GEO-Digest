"""Storage Backend — abstract interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

class StorageBackend(ABC):
    def __init__(self, data_dir: str = "/app/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Core CRUD ──

    @abstractmethod
    def load_articles(self) -> list: ...

    @abstractmethod
    def save_articles(self, articles: list): ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def get_article_by_doi(self, doi: str): ...

    @abstractmethod
    def add_article(self, article: dict): ...

    # ── Search & query (used by AgentTools) ──

    def search_articles(
        self,
        query: str = "",
        topic: str = "",
        source: str = "",
        limit: int = 50,
    ) -> tuple[list, int]:
        """
        Search articles by text query, topic tag, or source name.
        Returns (articles_list, total_count).
        Default implementation: loads all + filters in-memory.
        Override for indexed backends.
        """
        from engine.schemas import Article

        articles = [Article(a) for a in self.load_articles()]
        total = len(articles)

        if topic:
            articles = [
                a for a in articles
                if topic in (a.get("topics") or [])
                or topic in (a.get("topics_ru") or [])
            ]

        if source:
            articles = [a for a in articles if a.get("source") == source]

        if query:
            q_lower = query.lower()
            articles = [
                a for a in articles
                if q_lower in (a.display_title or "").lower()
                or q_lower in (a.get("abstract", "") or "").lower()
                or q_lower in (a.get("abstract_ru", "") or "").lower()
            ]

        return articles[:limit], total

    def get_article_by_id(self, identifier: str):
        """Find article by DOI, canonical_id, or _id."""
        from engine.schemas import Article

        for a in self.load_articles():
            art = Article(a)
            if identifier in (
                art.get("doi", ""),
                art.canonical_id,
                art.get("_id", ""),
            ):
                return art
        return None

    def load_all_articles(self):
        """Load all articles as Article objects."""
        from engine.schemas import Article
        return [Article(a) for a in self.load_articles()]

    # ── Graph ──

    @abstractmethod
    def load_graph(self) -> dict: ...

    @abstractmethod
    def save_graph(self, graph_data: dict): ...

    # ── Dedup tracking ──

    @abstractmethod
    def seen_dois(self) -> set: ...

    @abstractmethod
    def add_seen_doi(self, doi: str): ...

    # ── Stats ──

    def get_stats(self) -> dict:
        """Return storage statistics."""
        arts = self.load_articles()
        enriched = sum(1 for a in arts if a.get("is_enriched"))
        return {
            "total": len(arts),
            "enriched": enriched,
            "seen_dois": len(self.seen_dois()),
            "data_dir": str(self.data_dir),
        }
