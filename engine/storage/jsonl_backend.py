"""JSONL Storage Backend."""
from __future__ import annotations
import json
from pathlib import Path
from engine.storage.base import StorageBackend
from engine.schemas import Article

class JsonlStorage(StorageBackend):
    def __init__(self, data_dir="/app/data", articles_file="articles.jsonl",
                 graph_file="graph_data.json", seen_dois_file="seen_dois.txt"):
        super().__init__(data_dir)
        self.articles_path = self.data_dir / articles_file
        self.graph_path = self.data_dir / graph_file
        self.seen_path = self.data_dir / seen_dois_file

    def load_articles(self) -> list:
        if not self.articles_path.exists():
            return []
        arts = []
        with open(self.articles_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        arts.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return arts

    def save_articles(self, articles: list):
        with open(self.articles_path, "w", encoding="utf-8") as f:
            for a in articles:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")

    def count(self) -> int:
        if not self.articles_path.exists():
            return 0
        with open(self.articles_path) as f:
            return sum(1 for l in f if l.strip())

    def get_article_by_doi(self, doi: str):
        for a in self.load_articles():
            if a.get("doi") == doi:
                return Article(a)
        return None

    def add_article(self, article: dict):
        arts = self.load_articles()
        arts.append(article)
        self.save_articles(arts)

    def load_graph(self) -> dict:
        if not self.graph_path.exists():
            return {"nodes": [], "edges": [], "metadata": {}}
        with open(self.graph_path, encoding="utf-8") as f:
            return json.load(f)

    def save_graph(self, graph_data: dict):
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

    def seen_dois(self) -> set:
        if not self.seen_path.exists():
            return set()
        return set(l.strip() for l in open(self.seen_path).readlines() if l.strip())

    def add_seen_doi(self, doi: str):
        with open(self.seen_path, "a") as f:
            f.write(doi + "\n")

    # ── Optimized overrides ──

    def add_articles_batch(self, articles: list[dict], skip_seen: bool = True) -> int:
        """
        Bulk-add articles, skipping already-seen DOIs.
        Returns count of actually added articles.
        """
        if not articles:
            return 0

        seen = self.seen_dois() if skip_seen else set()
        existing = {a.get("doi") for a in self.load_articles() if a.get("doi")}
        existing.update(seen)

        # Deduplicate within batch itself (keep first occurrence)
        batch_seen: set[str] = set()
        new_arts = []
        for a in articles:
            doi = a.get("doi")
            if doi and (doi in existing or doi in batch_seen):
                continue
            if doi:
                batch_seen.add(doi)
            new_arts.append(a)

        if not new_arts:
            return 0

        all_arts = self.load_articles()
        all_arts.extend(new_arts)
        self.save_articles(all_arts)

        # Mark DOIs as seen
        for a in new_arts:
            if a.get("doi"):
                self.add_seen_doi(a["doi"])

        return len(new_arts)

    def search_articles(
        self,
        query: str = "",
        topic: str = "",
        source: str = "",
        limit: int = 50,
    ) -> tuple[list, int]:
        """Optimized search for JSONL — single pass through file."""
        from engine.schemas import Article

        arts_raw = self.load_articles()
        total = len(arts_raw)

        results = []
        for a in arts_raw:
            art = Article(a)

            if source and art.get("source") != source:
                continue
            if topic:
                topics = art.get("topics") or []
                topics_ru = art.get("topics_ru") or []
                if topic not in topics and topic not in topics_ru:
                    continue
            if query:
                q_lower = query.lower()
                title = (art.display_title or "").lower()
                abstract = (art.get("abstract", "") or "").lower()
                abstract_ru = (art.get("abstract_ru", "") or "").lower()
                if q_lower not in title and q_lower not in abstract and q_lower not in abstract_ru:
                    continue

            results.append(art)
            if len(results) >= limit:
                break

        return results, total
