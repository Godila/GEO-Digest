"""GEO-Digest Sources — unified search interface.

Usage:
    from sources import get_active_sources, search_all_sources, merge_results

    sources = get_active_sources(config)
    results = search_all_sources(sources, queries, config)
"""

from .base import SourceSearcher, merge_articles
from .openalex import OpenAlexSource
from .semantic_scholar import SemanticScholarSource
from .arxiv import ArxivSource
from .core_ac import CoreAcSource
from .doaj import DoajSource
from .crossref import CrossRefSource

# Registry of all available sources
_ALL_SOURCES: list[type[SourceSearcher]] = [
    OpenAlexSource,
    SemanticScholarSource,
    CoreAcSource,
    DoajSource,
    ArxivSource,
    CrossRefSource,
]


def get_active_sources() -> list[SourceSearcher]:
    """Instantiate all sources that are available (have API keys etc.).

    Returns sources sorted by priority (lower first).
    """
    instances = []
    for cls in _ALL_SOURCES:
        inst = cls()
        if inst.is_available():
            instances.append(inst)
    return sorted(instances, key=lambda s: s.priority())


def search_all_sources(
    sources: list[SourceSearcher],
    query: str,
    min_year: int = 2023,
) -> list[dict]:
    """
    Run query across all active sources.
    Returns raw results (dedup/merge happens later).
    """
    import time as _time

    all_results = []
    for src in sources:
        try:
            results = src.search(query, min_year=min_year)
            if results:
                all_results.extend(results)
            # Rate limit delay after each source
            _time.sleep(src.rate_limit())
        except Exception as e:
            print(f"  [{src.name()} error] {e}", file=__import__("sys").stderr)

    return all_results


def dedup_and_merge(articles: list[dict], seen_dois: set) -> tuple[list[dict], dict]:
    """
    Deduplicate articles by DOI (or title+year hash).
    Field-level merge when same DOI from multiple sources.
    Returns (unique_articles, doi_index).
    """
    from .base import title_hash

    unique: dict[str, dict] = {}

    for art in articles:
        doi = art.get("doi", "")
        id_key = doi if doi else title_hash(
            art.get("title", ""), str(art.get("year", ""))
        )

        # Skip already-seen DOIs
        if id_key in seen_dois:
            continue

        if id_key in unique:
            # Field-level merge — keep best data from each source
            unique[id_key] = merge_articles(unique[id_key], art)
        else:
            unique[id_key] = dict(art)
            unique[id_key]["_sources"] = [art.get("source", "")]

    result_list = list(unique.values())
    return result_list, unique
