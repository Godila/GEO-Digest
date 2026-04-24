"""OpenAlex source — free, no key required. 250M+ works."""

import json
import urllib.request
import urllib.parse

from .base import SourceSearcher


def reconstruct_abstract(inv_idx: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index."""
    if not inv_idx:
        return ""
    positions = {}
    for word, idxs in inv_idx.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[k] for k in sorted(positions))


class OpenAlexSource(SourceSearcher):
    """OpenAlex API — broadest free academic index."""

    def name(self) -> str:
        return "openalex"

    def priority(self) -> int:
        return 0  # PRIMARY — fastest and broadest

    def weight(self) -> float:
        return 1.05

    def rate_limit(self) -> float:
        return 2.0  # polite pool: 1 req/s recommended; we use 2s to be safe

    def search(self, query: str, per_page: int = 25, min_year: int = 2023, **kwargs) -> list[dict]:
        filters = [f"publication_year:>{min_year}", "type:article"]
        params = {
            "search": query,
            "filter": ",".join(filters),
            "sort": "cited_by_count:desc",
            "per_page": per_page,
            "select": "doi,title,publication_year,authorships,institutions,"
                      "primary_location,cited_by_count,open_access,topics,type,"
                      "abstract_inverted_index",
        }
        url = f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}"
        results = []
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GeoDigest/1.0 (mailto:research@example.com)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            for w in data.get("results", []):
                if not w or not isinstance(w, dict):
                    continue

                # Guard: skip malformed entries
                authorships = [a for a in w.get("authorships", []) if a and isinstance(a, dict)]
                authors = [a.get("author", {}).get("display_name", "") for a in authorships]
                institutions_raw = []
                for a in authorships:
                    for inst in a.get("institutions", []):
                        if inst and isinstance(inst, dict):
                            institutions_raw.append(inst.get("display_name", ""))
                topics_list = [t for t in w.get("topics", []) if t and isinstance(t, dict)]
                topics = [t.get("display_name", "") for t in topics_list]
                oa = w.get("open_access", {}) or {}

                # Reconstruct abstract from inverted index
                inv_idx = w.get("abstract_inverted_index")
                abstract = reconstruct_abstract(inv_idx) if inv_idx else ""

                # Fix: title may be array of chars or words
                raw_title = w.get("title", [])
                if isinstance(raw_title, list):
                    if raw_title and len(str(raw_title[0])) <= 2:
                        title_clean = "".join(str(p) for p in raw_title)
                    else:
                        title_clean = " ".join(str(p) for p in raw_title)
                else:
                    title_clean = str(raw_title)
                title_clean = title_clean.replace("  ", " ").strip()

                results.append({
                    "source": "openalex",
                    "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                    "title": title_clean,
                    "year": w.get("publication_year"),
                    "authors": "; ".join(authors[:6]),
                    "institutions": list(set(institutions_raw))[:5],
                    "journal": ((w.get("primary_location") or {}).get("source") or {}).get("display_name", ""),
                    "abstract": abstract,
                    "citations": w.get("cited_by_count", 0),
                    "references": w.get("referenced_works_count", 0),
                    "topics": topics,
                    "is_oa": oa.get("oa_url") is not None,
                    "oa_url": oa.get("oa_url", ""),
                    "url": w.get("id", ""),
                })
        return results
