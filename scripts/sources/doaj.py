"""DOAJ source — Directory of Open Access Journals.

19K+ peer-reviewed OA journals. High quality filter —
only journals that pass DOAJ's editorial review.
Free API: 10 req/sec.
Note: DOAJ blocks Python urllib, using requests library instead.
"""

import json
import time

import requests

from .base import SourceSearcher


class DoajSource(SourceSearcher):
    """DOAJ API v1 search."""

    def name(self) -> str:
        return "doaj"

    def priority(self) -> int:
        return 3

    def weight(self) -> float:
        return 1.08  # Peer-reviewed OA bonus

    def rate_limit(self) -> float:
        return 0.5  # Very generous: 10 req/sec allowed

    def search(self, query: str, page_size: int = 15, min_year: int = 2023, **kwargs) -> list[dict]:
        # DOAJ API v1: query is in the PATH, not a query param
        # Correct: /api/v1/search/articles/{query}?pageSize=5
        # Wrong:   /api/v1/search/articles?query={query}  → 404
        import urllib.parse
        params = {
            "pageSize": page_size,
            "sort": "relevance:desc",
        }
        safe_query = urllib.parse.quote(query)
        url = f"https://doaj.org/api/v1/search/articles/{safe_query}"
        results = []
        resp = requests.get(url, params=params, timeout=30, headers={
            "User-Agent": "GeoDigest/1.0 (https://github.com/Godila/GEO-Digest)",
            "Accept": "application/json",
        })
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("results", []):
            if not item or not isinstance(item, dict):
                continue
            bibjson = item.get("bibjson", {}) or {}

            # Title
            title_raw = bibjson.get("title", "")
            if isinstance(title_raw, list):
                title = title_raw[0] if title_raw else ""
            else:
                title = str(title_raw)

            # Authors
            author_list = bibjson.get("author", []) or []
            authors_list = []
            for a in author_list[:6]:
                if isinstance(a, str):
                    authors_list.append(a)
                elif isinstance(a, dict):
                    name = a.get("name", "")
                    if name:
                        authors_list.append(name)

            # Year from date string like "2024-03-15"
            year_str = ""
            for date_field in ["journal", "created_date"]:
                d = bibjson.get(date_field, {})
                if isinstance(d, dict):
                    year_str = d.get("year", "") or ""
                    if not year_str:
                        month = d.get("month", "")
                        if month:
                            year_str = str(month)[:4]
                    if year_str:
                        break
            try:
                year = int(year_str) if year_str else None
            except (ValueError, TypeError):
                year = None

            # Journal
            journal_obj = bibjson.get("journal", {}) or {}
            journal = journal_obj.get("title", "")
            if isinstance(journal, list):
                journal = journal[0] if journal else ""

            # Abstract / keywords
            abstract = bibjson.get("abstract", "") or ""
            keywords = bibjson.get("keyword", []) or []

            # DOI and identifiers
            doi = ""
            identifiers = {}
            for ident in (bibjson.get("identifier") or []):
                if isinstance(ident, dict):
                    id_type = ident.get("id_type", "").lower()
                    id_val = ident.get("id", "")
                    if id_type == "doi":
                        doi = id_val.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
                    identifiers[id_type] = id_val
                elif isinstance(ident, str) and ident.startswith("http"):
                    if "doi.org" in ident:
                        doi = ident.replace("https://doi.org/", "")

            # Links — find PDF/fulltext URL
            oa_url = ""
            links = bibjson.get("link") or []
            for link in links:
                if isinstance(link, dict):
                    content_type = link.get("content_type", "")
                    url_val = link.get("url", "")
                    if "pdf" in (content_type or "").lower() or "fulltext" in (url_val or "").lower():
                        oa_url = url_val
                        break
                    if not oa_url and url_val:
                        oa_url = url_val  # fallback: first link

            results.append({
                "source": "doaj",
                "doi": doi,
                "title": title.strip(),
                "year": year,
                "authors": "; ".join(authors_list),
                "institutions": [],
                "journal": journal.strip() if isinstance(journal, str) else "",
                "abstract": abstract.strip(),
                "citations": 0,
                "references": 0,
                "topics": keywords + [journal] if isinstance(journal, str) else keywords,
                "is_oa": bool(oa_url),
                "oa_url": oa_url,
                "url": doi,
                "license": bibjson.get("license", ""),
            })

        time.sleep(self.rate_limit())
        return results
