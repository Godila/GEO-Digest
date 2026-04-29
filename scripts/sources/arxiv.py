"""arXiv source — preprints + ML methods in geophysics.

Domain-restricted: only searches within physics.geo-ph, cs.LG,
eess.GV categories. Best for ML/deep learning methods papers.
"""

import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse

from .base import SourceSearcher


class ArxivSource(SourceSearcher):
    """arXiv API — preprints, strong for ML/geophysics crossover."""

    def name(self) -> str:
        return "arxiv"

    def priority(self) -> int:
        return 4  # After main sources — niche

    def weight(self) -> float:
        return 0.95  # Slight penalty: preprint, not peer-reviewed

    def rate_limit(self) -> float:
        return 2.5  # arXiv recommends 3s; 2.5s is safe for burst

    # Categories we search within — expanded for broader ML/geophysics coverage
    CATEGORIES = [
        "physics.geo-ph",       # Geophysics
        "cond-mat.stat-mech",   # Statistical mechanics
        "cs.LG",                # Machine Learning
        "stat.ML",              # Statistics / Machine Learning (NEW)
        "cs.CV",                # Computer Vision — satellite/remote sensing (NEW)
        "eess.GV",              # Geoscience & Remote Sensing
        "eess.SP",              # Signal Processing — seismic/DAS (NEW)
    ]

    def _build_query(self, query: str) -> str:
        """Wrap query in category restriction."""
        cats = " OR ".join(f"cat:{c}" for c in self.CATEGORIES)
        return f"({cats}) AND all:{query}"

    def search(self, query: str, max_results: int = 15, **kwargs) -> list[dict]:
        params = {
            "search_query": self._build_query(query),
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"http://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
        results = []
        req = urllib.request.Request(url, headers={"User-Agent": "GeoDigest/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            root = ET.fromstring(resp.read())
            ns = {"atom": "http://www.w3.org/2005/Atom",
                  "arxiv": "http://arxiv.org/schemas/atom"}
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                published = entry.find("atom:published", ns).text[:4]
                authors = [a.find("atom:name", ns).text
                           for a in entry.findall("atom:author", ns)]
                cats = [c.get("term", "") for c in entry.findall("atom:category", ns)]
                arxiv_id = entry.find("atom:id", ns).text
                links = entry.findall("atom:link", ns)
                pdf_url = ""
                for link in links:
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")

                results.append({
                    "source": "arxiv",
                    "doi": arxiv_id.replace("http://arxiv.org/abs/", "arXiv:"),
                    "title": title,
                    "year": int(published) if published.isdigit() else 2024,
                    "authors": "; ".join(authors[:6]),
                    "institutions": [],
                    "journal": "arXiv preprint",
                    "abstract": summary,
                    "citations": 0,
                    "references": 0,
                    "topics": cats,
                    "is_oa": True,
                    "oa_url": pdf_url,
                    "url": arxiv_id,
                })
        return results
