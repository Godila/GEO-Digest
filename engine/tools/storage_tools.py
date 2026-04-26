"""Storage Tools — LLM-callable functions for JsonlStorage access.

These are deterministic (no LLM) tools that the Editor Agent calls via
function calling to query article storage, validate DOIs, cluster results,
check for existing articles, and explore data distribution.

Each tool returns a dict that gets serialised to JSON for the LLM.
All tools use ToolResult from base.py for consistent error handling.

Tools:
  1. search_articles     — BM25-like keyword search across all articles
  2. get_article_detail   — Full info by DOI
  3. validate_doi         — Check if DOI exists in storage
  4. find_similar_existing — Find similar already-written articles
  5. cluster_by_subtopic  — Group articles by keyword clusters
  6. count_storage_stats  — Quick stats about storage contents
  7. explore_domain      — Domain overview: sources, years, topics

Usage:
    from engine.tools.storage_tools import create_storage_tools, STORAGE_TOOL_SCHEMAS
    from engine.storage import get_storage

    registry = create_storage_tools(get_storage())
    schemas = registry.get_schemas()  # → list[dict] for LLM API
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from engine.schemas import Article
from engine.tools.base import ToolRegistry, ToolResult


# ── JSON Schemas (Anthropic format) ──────────────────────────────

SEARCH_ARTICLES_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query — keywords to find in titles/abstracts (e.g., 'permafrost carbon feedback')",
        },
        "year_from": {
            "type": "integer",
            "description": "Minimum publication year (inclusive). Omit for no filter.",
        },
        "source": {
            "type": "string",
            "description": "Filter by source name: 'crossref', 'openalex', 'semantic_scholar', 'europe_pmc'. Omit for all.",
            "enum": ["crossref", "openalex", "semantic_scholar", "europe_pmc"],
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default 20, max 100).",
            "default": 20,
        },
    },
    "required": ["query"],
}

ARTICLE_DETAIL_SCHEMA = {
    "type": "object",
    "properties": {
        "doi": {
            "type": "string",
            "description": "DOI of the article to retrieve full details for.",
        }
    },
    "required": ["doi"],
}

VALIDATE_DOI_SCHEMA = {
    "type": "object",
    "properties": {
        "doi": {
            "type": "string",
            "description": "DOI string to validate against storage.",
        }
    },
    "required": ["doi"],
}

FIND_SIMILAR_EXISTING_SCHEMA = {
    "type": "object",
    "properties": {
        "title_idea": {
            "type": "string",
            "description": "Proposed article title or topic idea to check for duplicates.",
        }
    },
    "required": ["title_idea"],
}

CLUSTER_BY_SUBTOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "Topic to cluster articles by (e.g., 'Arctic methane emissions').",
        },
        "top_n": {
            "type": "integer",
            "description": "Number of top clusters to return (default 10, max 20).",
            "default": 10,
        }
    },
    "required": ["topic"],
}

COUNT_STORAGE_STATS_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "description": "Returns statistics about the current article storage.",
}

EXPLORE_DOMAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "focus_query": {
            "type": "string",
            "description": "Optional focus query to narrow domain exploration. Omit for full overview.",
        }
    },
    "required": [],
    "description": "Explores the domain: sources distribution, year range, topic coverage.",
}


# ── StorageTools Class ───────────────────────────────────────────

class StorageTools:
    """
    Collection of storage-query tools for the Editor Agent.

    Each method is a deterministic function that reads from JsonlStorage
    and returns structured data. Methods are registered as LLM-callable
    tools via create_storage_tools().
    """

    def __init__(self, storage):
        """
        Args:
            storage: StorageBackend instance (typically JsonlStorage).
        """
        self.storage = storage
        self._articles_cache: list[Article] | None = None
        self._output_dir: Path | None = None

    def _load_articles(self) -> list[Article]:
        """Lazy-load all articles from storage (cached)."""
        if self._articles_cache is None:
            self._articles_cache = self.storage.load_all_articles()
        return self._articles_cache

    def _invalidate_cache(self):
        """Force reload on next access."""
        self._articles_cache = None

    def _get_output_dir(self) -> Path:
        """Get output directory for existing articles check."""
        if self._output_dir is None:
            self._output_dir = getattr(self.storage, 'data_dir', Path("/app/data")) / "output"
        return self._output_dir

    # ── Helper methods ────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple tokenisation: lowercase, split on non-alphanumeric, filter stopwords."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "of", "in", "to", "for",
            "with", "on", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "out", "off", "over",
            "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "and", "but", "or", "if", "this",
            "that", "these", "those", "it", "its", "we", "our", "you", "your",
            "i", "me", "my", "he", "him", "his", "she", "her", "they", "them",
            "their", "what", "which", "who", "whom", "about", "against", "up",
            "и", "в", "на", "не", "по", "с", "за", "к", "из", "от", "для", "о",
            "об", "а", "но", "или", "что", "это", "как", "так", "его", "ее", "её",
            "она", "он", "они", "мы", "вы", "мне", "ему", "ей", "им", "их",
            "все", "всех", "всё", "весь", "под", "над", "перед", "между",
        }
        tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ]{3,}", text.lower())
        return {t for t in tokens if t not in stopwords}

    @staticmethod
    def _article_to_compact_dict(a: Article) -> dict:
        """Convert Article to compact dict suitable for LLM consumption."""
        title_ru = a.get("title_ru") or a.get("title") or ""
        abstract_ru = a.get("abstract_ru") or a.get("abstract") or ""
        topics_ru = a.get("topics_ru") or a.get("topics") or []
        llm_summary = a.get("llm_summary", "")

        return {
            "doi": a.doi,
            "title": a.display_title,
            "title_ru": title_ru,
            "year": a.get("year"),
            "source": a.get("source"),
            "article_type": a.get("article_type"),
            "citations": a.get("citations"),
            "topics": topics_ru[:8] if isinstance(topics_ru, list) else [],
            "abstract_preview": abstract_ru[:500],
            "llm_summary": llm_summary[:800] if llm_summary else "",
            "score_total": round(a.total_score, 2) if a.total_score else 0,
            "is_enriched": a.is_enriched,
        }

    @staticmethod
    def _article_to_full_dict(a: Article) -> dict:
        """Convert Article to full detail dict."""
        d = StorageTools._article_to_compact_dict(a)
        d.update({
            "canonical_id": a.id,
            "_id": a.get("_id"),
            "url": a.get("url"),
            "oa_url": a.get("oa_url"),
            "keywords": a.get("keywords", []),
            "external_ids": a.get("external_ids", {}),
            "full_abstract_ru": a.get("abstract_ru") or a.get("abstract") or "",
            "full_abstract_en": a.get("abstract") or "",
            "scores": a.get("scores", {}),
        })
        return d

    # ── Tool 1: search_articles ───────────────────────────────────

    def search_articles(
        self,
        query: str = "",
        year_from: int | None = None,
        source: str = "",
        max_results: int = 20,
    ) -> ToolResult:
        """
        Search articles in storage using BM25-like keyword scoring.

        Ranks by number of matching query keywords in title + abstract.
        Supports filtering by minimum year and source.
        """
        if not query.strip():
            return ToolResult.fail(message="Query cannot be empty")

        articles = self._load_articles()
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return ToolResult.fail(
                message=f"No searchable tokens in query '{query}' "
                        "(too short or all stopwords)"
            )

        scored = []
        for a in articles:
            # Build searchable text from BOTH languages
            title_ru = (a.get("title_ru") or "").lower()
            title_en = (a.get("title") or "").lower()
            abstract_ru = (a.get("abstract_ru") or "").lower()
            abstract_en = (a.get("abstract") or "").lower()
            searchable = f"{title_ru} {title_en} {abstract_ru} {abstract_en}"

            # Count token matches (simple BM25 proxy)
            score = sum(1 for t in query_tokens if t in searchable)

            if score == 0:
                continue

            # Apply filters
            year = a.get("year")
            if year_from and year:
                try:
                    if int(year) < year_from:
                        continue
                except (ValueError, TypeError):
                    pass

            art_source = a.get("source", "")
            if source and art_source != source.lower():
                continue

            scored.append((score, a))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        limited = scored[:min(max_results, 100)]

        return ToolResult.ok(data={
            "total_found": len(scored),
            "returned": len(limited),
            "query_tokens_used": list(query_tokens),
            "articles": [self._article_to_compact_dict(a) for _, a in limited],
        })

    # ── Tool 2: get_article_detail ────────────────────────────────

    def get_article_detail(self, doi: str) -> ToolResult:
        """Get full details of an article by its DOI."""
        if not doi or not doi.strip():
            return ToolResult.fail(message="DOI cannot be empty")

        articles = self._load_articles()
        decoded_doi = doi.replace("%2F", "/")

        for a in articles:
            if a.doi == decoded_doi or a.id == decoded_doi:
                return ToolResult.ok(data={
                    **self._article_to_full_dict(a),
                    "found": True,
                })

        return ToolResult.ok(data={
            "found": False,
            "doi": doi,
            "message": f"Article with DOI '{doi}' not found in storage",
        })

    # ── Tool 3: validate_doi ─────────────────────────────────────

    def validate_doi(self, doi: str) -> ToolResult:
        """Validate whether a DOI exists in local storage."""
        if not doi or not doi.strip():
            return ToolResult.fail(message="DOI cannot be empty")

        articles = self._load_articles()
        decoded_doi = doi.replace("%2F", "/")

        for a in articles:
            if a.doi == decoded_doi or a.id == decoded_doi:
                return ToolResult.ok(data={
                    "valid": True,
                    "doi": decoded_doi,
                    "title": a.display_title,
                    "year": a.get("year"),
                    "source": a.get("source"),
                })

        return ToolResult.ok(data={
            "valid": False,
            "doi": doi,
            "message": "DOI not found in local storage",
        })

    # ── Tool 4: find_similar_existing ────────────────────────────

    def find_similar_existing(self, title_idea: str) -> ToolResult:
        """
        Find existing written articles that may overlap with proposed title.

        Checks output directory for .md files and compares keyword overlap.
        """
        if not title_idea or not title_idea.strip():
            return ToolResult.fail(message="Title idea cannot be empty")

        output_dir = self._get_output_dir()
        idea_tokens = self._tokenize(title_idea)

        if not output_dir.exists():
            return ToolResult.ok(data={
                "existing_count": 0,
                "matches": [],
                "message": "No output directory found",
            })

        matches = []
        for fpath in output_dir.rglob("*.md"):
            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore")[:3000]
                file_tokens = self._tokenize(content)
                if not file_tokens:
                    continue

                # Jaccard similarity on tokens
                intersection = len(idea_tokens & file_tokens)
                union = len(idea_tokens | file_tokens)
                similarity = intersection / union if union > 0 else 0

                if similarity > 0.05:  # Minimum 5% overlap to report
                    preview = content[:200].replace("\n", " ")
                    matches.append({
                        "file": str(fpath.relative_to(output_dir)),
                        "similarity": round(similarity, 3),
                        "shared_keywords": sorted(intersection and (idea_tokens & file_tokens) or []),
                        "preview": preview,
                    })
            except Exception:
                continue

        matches.sort(key=lambda m: m["similarity"], reverse=True)

        return ToolResult.ok(data={
            "existing_count": len(matches),
            "matches": matches[:10],  # Top 10 matches
            "search_tokens": list(idea_tokens),
        })

    # ── Tool 5: cluster_by_subtopic ──────────────────────────────

    def cluster_by_subtopic(self, topic: str, top_n: int = 10) -> ToolResult:
        """
        Cluster articles related to a topic into subtopic groups.

        Uses simple keyword extraction from titles + topics field to group
        articles into thematic clusters.
        """
        if not topic or not topic.strip():
            return ToolResult.fail(message="Topic cannot be empty")

        articles = self._load_articles()
        topic_tokens = self._tokenize(topic)

        # Score relevance of each article to the topic
        relevant = []
        for a in articles:
            title_ru = (a.get("title_ru") or "").lower()
            title_en = (a.get("title") or "").lower()
            abstract_ru = (a.get("abstract_ru") or "").lower()
            abstract_en = (a.get("abstract") or "").lower()
            searchable = f"{title_ru} {title_en} {abstract_ru} {abstract_en}"
            score = sum(1 for t in topic_tokens if t in searchable)

            if score > 0:
                relevant.append((score, a))

        if not relevant:
            return ToolResult.ok(data={
                "total_relevant": 0,
                "cluster_count": 0,
                "clusters": [],
                "message": f"No articles match topic '{topic}'",
            })

        # Cluster by extracting sub-topic keywords from each article's topics field
        clusters: dict[str, list] = defaultdict(list)
        for score, a in relevant:
            article_topics = a.get("topics_ru") or a.get("topics") or []
            if not article_topics:
                clusters["other"].append(a)
            else:
                # Assign to first matching topic group
                assigned = False
                for t in article_topics:
                    t_lower = str(t).lower()
                    # Check if this topic relates to our search topic
                    topic_overlap = any(
                        tok in t_lower for tok in topic_tokens
                    )
                    if topic_overlap or any(
                        tok in t_lower.split()[:3] for tok in topic_tokens
                    ):
                        clusters[t].append(a)
                        assigned = True
                        break
                if not assigned:
                    clusters["other"].append(a)

        # Sort clusters by size, take top_n
        sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))
        top_clusters = sorted_clusters[:top_n]

        cluster_data = [
            {
                "theme": theme,
                "count": len(arts),
                "sample_dois": [a.doi for a in arts[:5] if a.doi],
                "sample_titles": [
                    (a.get("title_ru") or a.get("title") or "")[:80]
                    for a in arts[:3]
                ],
            }
            for theme, arts in top_clusters
        ]

        return ToolResult.ok(data={
            "total_relevant": len(relevant),
            "cluster_count": len(cluster_data),
            "clusters": cluster_data,
            "all_cluster_names": [c["theme"] for c in cluster_data],
        })

    # ── Tool 6: count_storage_stats ──────────────────────────────

    def count_storage_stats(self) -> ToolResult:
        """Return quick statistics about the article storage."""
        articles = self._load_articles()

        total = len(articles)
        enriched = sum(1 for a in articles if a.is_enriched)

        # Source breakdown
        source_counts = Counter(a.get("source", "unknown") for a in articles)

        # Year range
        years = []
        for a in articles:
            y = a.get("year")
            if y:
                try:
                    years.append(int(y))
                except (ValueError, TypeError):
                    pass

        year_range = {}
        if years:
            year_range = {
                "min_year": min(years),
                "max_year": max(years),
                "by_year": dict(Counter(years).most_common()),
            }

        # Topic coverage
        all_topics = []
        for a in articles:
            topics = a.get("topics_ru") or a.get("topics") or []
            all_topics.extend(topics)

        top_topics = Counter(all_topics).most_common(15)

        return ToolResult.ok(data={
            "total_articles": total,
            "enriched_articles": enriched,
            "enrichment_rate": round(enriched / total * 100, 1) if total > 0 else 0,
            "sources": dict(source_counts.most_common()),
            **year_range,
            "top_topics": [{"topic": t, "count": c} for t, c in top_topics],
        })

    # ── Tool 7: explore_domain ────────────────────────────────────

    def explore_domain(self, focus_query: str = "") -> ToolResult:
        """
        Provide a domain overview: sources, temporal distribution, topic map.

        Optional focus_query narrows the analysis to matching articles only.
        """
        articles = self._load_articles()

        # Filter by focus query if provided
        if focus_query and focus_query.strip():
            query_tokens = self._tokenize(focus_query)
            filtered = []
            for a in articles:
                text = (
                    (a.get("title_ru") or a.get("title") or "") + " " +
                    (a.get("abstract_ru") or a.get("abstract") or "")
                ).lower()
                if any(t in text for t in query_tokens):
                    filtered.append(a)
            articles = filtered

        if not articles:
            return ToolResult.ok(data={
                "total_in_scope": 0,
                "message": f"No articles match focus query '{focus_query}'" if focus_query else "Storage is empty",
            })

        # Comprehensive stats
        source_dist = Counter(a.get("source", "unknown") for a in articles)
        type_dist = Counter(a.get("article_type", "unknown") for a in articles)

        years = []
        for a in articles:
            y = a.get("year")
            if y:
                try:
                    years.append(int(y))
                except (ValueError, TypeError):
                    pass

        year_dist = dict(Counter(years).most_common()) if years else {}

        # Citation stats
        citations = []
        for a in articles:
            c = a.get("citations")
            if c:
                try:
                    citations.append(int(c))
                except (ValueError, TypeError):
                    pass

        citation_stats = {}
        if citations:
            citations.sort(reverse=True)
            citation_stats = {
                "max": citations[0] if citations else 0,
                "median": citations[len(citations) // 2] if citations else 0,
                "mean": round(sum(citations) / len(citations), 1),
                "highly_cited_count": sum(1 for c in citations if c >= 50),
            }

        # Topics
        all_topics = []
        for a in articles:
            topics = a.get("topics_ru") or a.get("topics") or []
            all_topics.extend(topics)
        topic_dist = Counter(all_topics).most_common(20)

        return ToolResult.ok(data={
            "total_in_scope": len(articles),
            "focus_query": focus_query or "(none — full scope)",
            "sources": {k: v for k, v in source_dist.items()},
            "article_types": {k: v for k, v in type_dist.items()},
            "year_distribution": year_dist,
            "citation_stats": citation_stats,
            "top_topics": [{"topic": t, "count": c, "pct": round(c / len(articles) * 100, 1)}
                           for t, c in topic_dist],
            "has_llm_summaries": sum(1 for a in articles if a.is_enriched),
            "enrichment_pct": round(sum(1 for a in articles if a.is_enriched) / len(articles) * 100, 1),
        })


# ── Factory Function ─────────────────────────────────────────────

def create_storage_tools(storage) -> ToolRegistry:
    """
    Create a ToolRegistry pre-loaded with all storage tools.

    Args:
        storage: StorageBackend instance (JsonlStorage).

    Returns:
        ToolRegistry with 7 registered tools ready for LLM function calling.
    """
    tools = StorageTools(storage)
    registry = ToolRegistry()

    registry.register(
        name="search_articles",
        handler=tools.search_articles,
        schema=SEARCH_ARTICLES_SCHEMA,
        description="Search articles in storage by keywords. Returns ranked list with DOI, title, abstract preview, score.",
    )

    registry.register(
        name="get_article_detail",
        handler=tools.get_article_detail,
        schema=ARTICLE_DETAIL_SCHEMA,
        description="Get complete information about a single article by DOI including full abstract, authors, keywords.",
    )

    registry.register(
        name="validate_doi",
        handler=tools.validate_doi,
        schema=VALIDATE_DOI_SCHEMA,
        description="Check if a DOI exists in local storage. Returns validity status and basic info if found.",
    )

    registry.register(
        name="find_similar_existing",
        handler=tools.find_similar_existing,
        schema=FIND_SIMILAR_EXISTING_SCHEMA,
        description="Find existing written articles that may overlap with a proposed title/topic idea.",
    )

    registry.register(
        name="cluster_by_subtopic",
        handler=tools.cluster_by_subtopic,
        schema=CLUSTER_BY_SUBTOPIC_SCHEMA,
        description="Group articles related to a topic into thematic sub-clusters based on their topic tags.",
    )

    registry.register(
        name="count_storage_stats",
        handler=tools.count_storage_stats,
        schema=COUNT_STORAGE_STATS_SCHEMA,
        description="Get quick statistics about storage: total articles, enrichment rate, sources, year range, top topics.",
    )

    registry.register(
        name="explore_domain",
        handler=tools.explore_domain,
        schema=EXPLORE_DOMAIN_SCHEMA,
        description="Explore the research domain: sources distribution, temporal trends, citation stats, topic coverage map.",
    )

    return registry


# Convenience: all schemas in one place
STORAGE_TOOL_SCHEMAS = {
    "search_articles": SEARCH_ARTICLES_SCHEMA,
    "get_article_detail": ARTICLE_DETAIL_SCHEMA,
    "validate_doi": VALIDATE_DOI_SCHEMA,
    "find_similar_existing": FIND_SIMILAR_EXISTING_SCHEMA,
    "cluster_by_subtopic": CLUSTER_BY_SUBTOPIC_SCHEMA,
    "count_storage_stats": COUNT_STORAGE_STATS_SCHEMA,
    "explore_domain": EXPLORE_DOMAIN_SCHEMA,
}
