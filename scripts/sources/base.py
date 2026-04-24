"""
GEO-Digest Sources Module — Base classes and unified interface.

Every source implements SourceSearcher ABC:
  - name()          — human-readable identifier
  - search(query)   — returns list[dict] with standardised fields
  - priority()      — execution order (lower = first)
  - weight()        — scoring multiplier (1.0 = baseline)
  - rate_limit()    — seconds delay between requests
  - is_available()  — check API key / prerequisites at runtime

Standard article dict fields (all sources MUST return these):
  source, doi, title, year, authors, institutions, journal,
  abstract, citations, references, topics, is_oa, oa_url, url
Plus optional: pub_types, pmcid, has_fulltext
"""

from abc import ABC, abstractmethod
from typing import Optional


class SourceSearcher(ABC):
    """Abstract base class for all academic search sources."""

    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and article records."""
        ...

    @abstractmethod
    def search(self, query: str, **kwargs) -> list[dict]:
        """
        Search for articles matching query.
        Returns list of standardised article dicts.
        Raises on transient errors (caller handles retry/skip).
        """
        ...

    def priority(self) -> int:
        """Execution order (lower = called first). Default: 50."""
        return 50

    def weight(self) -> float:
        """Scoring bonus multiplier. Default: 1.0 (no bonus/penalty)."""
        return 1.0

    def rate_limit(self) -> float:
        """Seconds to wait after each request. Default: 1.0."""
        return 1.0

    def is_available(self) -> bool:
        """Check if this source can run (API key present, etc.). Default: True."""
        return True


# ── Field-level merge helpers ────────────────────────────────

# Priority order per field: [ (source_name, priority), ... ]
# Higher priority value wins when merging duplicates.
FIELD_PRIORITY = {
    "abstract": {
        "semantic_scholar": 5,
        "core_ac": 4,
        "openalex": 3,
        "doaj": 2,
        "crossref": 1,
        "arxiv": 3,
    },
    "oa_url": {
        "unpaywall": 6,
        "core_ac": 4,
        "doaj": 3,
        "crossref": 2,
        "openalex": 2,
        "europe_pmc": 3,
        "arxiv": 5,
    },
    "citations": {
        "semantic_scholar": 5,
        "openalex": 4,
        "crossref": 3,
    },
    "topics": {
        "openalex": 5,
        "semantic_scholar": 4,
        "core_ac": 2,
    },
    "journal": {
        "crossref": 4,
        "openalex": 3,
        "doaj": 4,
        "core_ac": 2,
    },
    "authors": {
        "semantic_scholar": 3,
        "openalex": 3,
        "crossref": 3,
        "doaj": 3,
        "core_ac": 2,
    },
}


def title_hash(title: str, year: str = "") -> str:
    """Generate stable ID from title + year for dedup without DOI."""
    import hashlib
    canonical = f"{(title or '').strip().lower()}|{year}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def best_value(existing_val, new_val, field: str, new_source: str, existing_source: str = ""):
    """
    Pick the better of two values for a given field using FIELD_PRIORITY.
    Rules:
      - Non-empty beats empty
      - If both non-empty: higher priority source wins
      - Longer abstract wins if same priority
    """
    # Handle None / empty
    existing_empty = existing_val is None or existing_val == "" or existing_val == []
    new_empty = new_val is None or new_val == "" or new_val == []

    if new_empty:
        return existing_val
    if existing_empty:
        return new_val

    # Both have values — check priority
    prios = FIELD_PRIORITY.get(field, {})
    new_prio = prios.get(new_source, 0)
    exist_prio = prios.get(existing_source, 0)

    if new_prio > exist_prio:
        return new_val
    elif exist_prio > new_prio:
        return existing_val
    else:
        # Same priority: prefer longer for text fields
        if isinstance(new_val, str) and isinstance(existing_val, str):
            return new_val if len(new_val) > len(existing_val) else existing_val
        return new_val  # tie-breaker: newer data


def merge_articles(existing: dict, new: dict) -> dict:
    """
    Field-level merge of two article dicts (same DOI).
    Returns merged dict without mutating inputs.
    """
    merged = dict(existing)
    new_source = new.get("source", "")
    existing_source = existing.get("source", "")

    # Fields to merge with priority logic
    merge_fields = [
        "abstract", "oa_url", "citations", "references",
        "topics", "journal", "authors", "institutions",
    ]

    for field in merge_fields:
        merged[field] = best_value(
            existing.get(field), new.get(field),
            field, new_source, existing_source
        )

    # Preserve non-empty flags
    if not merged.get("is_oa") and new.get("is_oa"):
        merged["is_oa"] = True
    if not merged.get("url") and new.get("url"):
        merged["url"] = new["url"]
    if not merged.get("doi") and new.get("doi"):
        merged["doi"] = new["doi"]

    # Track which sources contributed
    sources_seen = set(merged.get("_sources", []))
    sources_seen.add(new_source)
    if existing_source:
        sources_seen.add(existing_source)
    merged["_sources"] = sorted(sources_seen)

    return merged
