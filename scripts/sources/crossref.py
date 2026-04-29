"""CrossRef — enrich-only source.

NOT used for searching (duplicates OpenAlex).
Used for: ISSN/DOI lookup, journal metadata, PDF URL enrichment.
"""

import json
import re
import urllib.request
import urllib.parse

from .base import SourceSearcher


class CrossRefSource(SourceSearcher):
    """CrossRef API — 140M+ DOI records. Enrich-only mode."""

    def name(self) -> str:
        return "crossref"

    def priority(self) -> int:
        return 99  # Not used in search pipeline

    def weight(self) -> float:
        return 1.0  # Baseline

    def rate_limit(self) -> float:
        return 0.5

    def is_available(self) -> bool:
        """Available but not used for search."""
        return False  # Disabled from search pipeline

    def search(self, query: str, rows: int = 15, min_year: int = 2023, **kwargs) -> list[dict]:
        """
        Search CrossRef. Only called explicitly for enrichment.
        NOT part of the main search pipeline.
        """
        date_from = f"from-pub-date:{min_year}-01-01"
        params = {
            "query": query,
            "filter": f"{date_from},type:journal-article",
            "rows": rows,
            "sort": "published",
            "order": "desc",
            "select": "DOI,title,abstract,author,published-print,published-online,"
                      "container-title,URL,is-referenced-by-count,"
                      "link,subject",
        }
        url = f"https://api.crossref.org/works?{urllib.parse.urlencode(params)}"
        results = []
        req = urllib.request.Request(url, headers={
            "User-Agent": "GeoDigest/1.0 (mailto:research@example.com)"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            for item in data.get("message", {}).get("items", []):
                raw_title = item.get("title", [""])[0] if item.get("title") else ""

                # Abstract — JATS XML tags, strip them
                abstract_raw = item.get("abstract", "") or ""
                abstract = re.sub(r"<[^>]+>", "", abstract_raw).strip()

                authors_list = item.get("author", []) or []
                authors = "; ".join(
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in authors_list[:6]
                )

                pub_date = (
                    item.get("published-print") or
                    item.get("published-online") or {}
                )
                date_parts = pub_date.get("date-parts", [[None]])[0]
                year = date_parts[0] if date_parts else None

                journal = (item.get("container-title") or [""])[0]

                cited = item.get("is-referenced-by-count", 0)
                refs = len(item.get("reference", []) or [])

                links = item.get("link", []) or []
                pdf_url = ""
                html_url = item.get("URL", "")
                for link in links:
                    content_type = link.get("content-type", "")
                    if "pdf" in content_type:
                        pdf_url = link.get("URL", "")

                subjects = []
                for s in (item.get("subject") or []):
                    if isinstance(s, str):
                        subjects.append(s)

                results.append({
                    "source": "crossref",
                    "doi": (item.get("DOI") or "").replace("https://doi.org/", ""),
                    "title": raw_title.strip(),
                    "year": year,
                    "authors": authors,
                    "institutions": [],
                    "journal": journal,
                    "abstract": abstract,
                    "citations": cited,
                    "references": refs,
                    "topics": subjects,
                    "is_oa": bool(pdf_url),
                    "oa_url": pdf_url or html_url,
                    "url": html_url,
                })
        return results

    def lookup_doi(self, doi: str) -> dict | None:
        """Enrich: get metadata for a specific DOI."""
        if not doi:
            return None
        try:
            url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "GeoDigest/1.0 (mailto:research@example.com)"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read()).get("message", {})
                if not data:
                    return None
                raw_title = data.get("title", [""])[0] if data.get("title") else ""
                journal = (data.get("container-title") or [""])[0]
                links = data.get("link", []) or []
                pdf_url = ""
                for link in links:
                    if "pdf" in link.get("content-type", ""):
                        pdf_url = link.get("URL", "")
                return {
                    "journal": journal,
                    "oa_url": pdf_url,
                    "citations": data.get("is-referenced-by-count", 0),
                    "references": len(data.get("reference", []) or []),
                }
        except Exception:
            return None
