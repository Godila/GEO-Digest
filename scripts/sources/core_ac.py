"""CORE.ac.uk source — 200M+ OA articles, free API.

Excellent for open-access content that OpenAlex may miss.
Free tier: 60 req/min, no key required (key gives higher limits).
"""

import json
import urllib.request
import urllib.parse

from .base import SourceSearcher


class CoreAcSource(SourceSearcher):
    """CORE.ac.uk API — largest open access aggregator."""

    def name(self) -> str:
        return "core_ac"

    def priority(self) -> int:
        return 2

    def weight(self) -> float:
        return 1.05  # Verified OA content

    def rate_limit(self) -> float:
        return 1.2  # Free tier: 60 req/min → ~1s between requests

    def search(self, query: str, limit: int = 20, min_year: int = 2023, **kwargs) -> list[dict]:
        params = {
            "q": query,
            "limit": limit,
            "offset": 0,
            # Filter for recent + full text available
            "filters": f"year>={min_year},type:article",
            "sort": "relevance",
        }
        url = f"https://api.core.ac.uk/v3/search/works?{urllib.parse.urlencode(params)}"
        results = []
        req = urllib.request.Request(url, headers={
            "User-Agent": "GeoDigest/1.0",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            for w in data.get("results", []):
                if not w or not isinstance(w, dict):
                    continue

                # Authors — CORE has nested structure
                raw_authors = w.get("authors", []) or []
                authors_list = []
                if isinstance(raw_authors, list):
                    for a in raw_authors[:6]:
                        if isinstance(a, str):
                            authors_list.append(a)
                        elif isinstance(a, dict):
                            name = a.get("name", "")
                            if name:
                                authors_list.append(name)

                # Year
                year = w.get("publishedDate") or w.get("year") or ""
                if isinstance(year, str):
                    year = year[:4]
                try:
                    year = int(year) if year else None
                except (ValueError, TypeError):
                    year = None

                # Journal / publisher
                journal = w.get("publisher", "") or ""
                if not journal:
                    jinfo = w.get("journalInfo", {}) or {}
                    if isinstance(jinfo, dict):
                        journal = jinfo.get("title", "") or ""

                # DOI
                doi = w.get("doi", "") or ""
                if not doi:
                    identifiers = w.get("identifiers", []) or []
                    for ident in identifiers:
                        if isinstance(ident, dict) and ident.get("doi"):
                            doi = ident["doi"]
                            break

                # Download URL (OA PDF)
                download_url = w.get("downloadUrl", "") or ""
                oa_url = w.get("oaLink", "") or download_url

                results.append({
                    "source": "core_ac",
                    "doi": doi,
                    "title": w.get("title", ""),
                    "year": year,
                    "authors": "; ".join(authors_list),
                    "institutions": [],
                    "journal": journal,
                    "abstract": w.get("abstract", "") or "",
                    "citations": 0,  # CORE doesn't provide citation counts
                    "references": 0,
                    "topics": [],  # No topic classification in CORE free tier
                    "is_oa": bool(oa_url),
                    "oa_url": oa_url,
                    "url": "",
                })
        return results
