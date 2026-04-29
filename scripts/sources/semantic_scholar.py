"""Semantic Scholar source — requires API key for stable use.

Without key: shared pool (5000 req/5min, frequent 429).
With key:    1 req/s stable.

Set SEMANTIC_SCHOLAR_API_KEY in .env to enable.
"""

import json
import os
import urllib.request
import urllib.parse

from .base import SourceSearcher


def _get_api_key() -> str:
    """Load S2 API key from env or .env file."""
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if key:
        return key
    # Fallback: read from .env
    env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", ".env")
    if os.path.exists(env_file):
        for line in open(env_file).readlines():
            line = line.strip()
            if line.startswith("SEMANTIC_SCHOLAR_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


class SemanticScholarSource(SourceSearcher):
    """Semantic Scholar API — best abstracts + citation data."""

    def name(self) -> str:
        return "semantic_scholar"

    def priority(self) -> int:
        return 1

    def weight(self) -> float:
        return 1.10  # best metadata quality

    def rate_limit(self) -> float:
        # With key: 1 req/s → 1s delay. Without: conservative.
        return 2.5 if _get_api_key() else 3.0

    def is_available(self) -> bool:
        """Only available if API key is set."""
        return bool(_get_api_key())

    def search(self, query: str, limit: int = 25, min_year: int = 2023, **kwargs) -> list[dict]:
        api_key = _get_api_key()
        if not api_key:
            return []  # Skip silently when no key

        params = {
            "query": query,
            "limit": limit,
            "year": f"{min_year}-2026",
            "fields": "externalIds,title,year,authors,abstract,journal,"
                      "citationCount,referenceCount,openAccessPaper,"
                      "fieldsOfStudy,publicationTypes,venue,isOpenAccess",
        }
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "GeoDigest/1.0"}
        if api_key:
            headers["x-api-key"] = api_key

        results = []
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            for p in data.get("data", []):
                ext_ids = p.get("externalIds", {}) or {}
                authors = [a.get("name", "") for a in p.get("authors", [])]
                fos = p.get("fieldsOfStudy", []) or []
                pub_types = p.get("publicationTypes", []) or []
                oa = p.get("openAccessPaper") or {}

                results.append({
                    "source": "semantic_scholar",
                    "doi": ext_ids.get("DOI", ""),
                    "title": p.get("title", ""),
                    "year": p.get("year"),
                    "authors": "; ".join(authors[:6]),
                    "institutions": [],
                    "journal": p.get("venue", ""),
                    "abstract": p.get("abstract", "") or "",
                    "citations": p.get("citationCount", 0),
                    "references": p.get("referenceCount", 0),
                    "topics": fos,
                    "is_oa": p.get("isOpenAccess", False),
                    "oa_url": oa.get("url", ""),
                    "url": "",
                    "pub_types": pub_types,
                })
        return results
