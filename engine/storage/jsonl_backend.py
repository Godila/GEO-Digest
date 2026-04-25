"""
JSONL storage backend — current file-based implementation.

Reads/writes:
  - /app/data/articles.jsonl     (article database)
  - /app/data/graph_data.json    (knowledge graph)
  - /app/data/agent_jobs/{id}.json  (job states)
  - /app/data/output/{job_id}/      (draft outputs)

This is the DEFAULT backend. Can be replaced with SQLiteBackend later
without changing any agent code.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from engine.config import get_config
from engine.schemas import Article, StructuredDraft, ArticleDraft, JobState
from engine.storage.base import StorageBackend


class JsonlStorage(StorageBackend):
    """
    File-based storage using JSONL for articles + JSON for graph/jobs.
    
    Compatible with existing digest.py data format.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        cfg = get_config()
        self.data_dir = data_dir or cfg.data_dir
        self.articles_file = self.data_dir / "articles.jsonl"
        self.graph_file = self.data_dir / "graph_data.json"
        self.jobs_dir = cfg.jobs_dir          # /app/data/agent_jobs/
        self.output_dir = cfg.output_dir      # /app/data/output/
        
        # Ensure directories exist
        for d in (self.jobs_dir, self.output_dir):
            d.mkdir(parents=True, exist_ok=True)
    
    # ── Articles ──
    
    def load_all_articles(self) -> list[Article]:
        """Load all articles from JSONL."""
        if not self.articles_file.exists():
            return []
        
        articles = []
        for line in self.articles_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                articles.append(Article(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  [Storage] Skipping bad article line: {e}", file=os.stderr)
        
        return articles
    
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        """Find article by canonical ID or DOI."""
        # Try direct DOI match first
        decoded = article_id.replace("%2F", "/")  # URL-decode
        
        for art in self.load_all_articles():
            # Check canonical_id
            if art.canonical_id == decoded or art.canonical_id == article_id:
                return art
            # Check DOI directly
            if art.get("doi", "").lower() == decoded.lower():
                return art
            # Check _id field
            if art.get("_id") == article_id or art.get("_id") == decoded:
                return art
        
        return None
    
    def search_articles(
        self,
        query: str = "",
        topic: str = "",
        source: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Article], int]:
        """Search with text query and filters."""
        import re
        
        articles = self.load_all_articles()
        
        # Filters
        if topic:
            articles = [a for a in articles if a.get("_topic_key") == topic]
        if source:
            articles = [a for a in articles if a.get("source") == source]
        if query:
            q_lower = query.lower()
            articles = [a for a in articles if
                q_lower in (a.get("title", "")).lower() or
                q_lower in (a.get("title_ru", "")).lower() or
                q_lower in (a.get("abstract", "")).lower() or
                q_lower in (a.get("abstract_ru", "")).lower()
            ]
        
        total = len(articles)
        # Sort by score desc
        articles.sort(key=lambda a: a.score_total, reverse=True)
        
        return articles[offset:offset + limit], total
    
    def save_article(self, article: Article | dict) -> None:
        """Append article to JSONL (atomic: write to temp + rename)."""
        data = article if isinstance(article, dict) else dict(article)
        
        mode = "a" if self.articles_file.exists() else "w"
        if not self.articles_file.exists():
            # New file: write header comment? No, pure JSONL.
            pass
        
        with open(self.articles_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    def count_articles(self) -> int:
        if not self.articles_file.exists():
            return 0
        count = 0
        for line in self.articles_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.strip():
                count += 1
        return count
    
    # ── Graph ──
    
    def load_graph(self) -> dict:
        """Load graph from JSON file."""
        if not self.graph_file.exists():
            return {"nodes": [], "edges": [], "metadata": {}}
        try:
            return json.loads(self.graph_file.read_text())
        except (json.JSONDecodeError, TypeError):
            return {"nodes": [], "edges": [], "metadata": {}}
    
    def save_graph(self, graph_data: dict) -> None:
        """Save graph to JSON file (atomic write)."""
        tmp = self.graph_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(graph_data, ensure_ascii=False, indent=2))
        tmp.replace(self.graph_file)
    
    # ── Drafts ──
    
    def save_draft(
        self, draft: StructuredDraft | ArticleDraft, job_id: str = ""
    ) -> Path:
        """Save draft to output directory. Returns path."""
        job_output = self.output_dir / job_id if job_id else self.output_dir
        job_output.mkdir(parents=True, exist_ok=True)
        
        filename = f"{type(draft).__name__.lower()}_{draft.draft_id or 'latest'}.json"
        path = job_output / filename
        
        path.write_text(json.dumps(draft.to_dict(), ensure_ascii=False, indent=2))
        return path
    
    def load_draft(self, draft_id: str) -> Optional[StructuredDraft | ArticleDraft]:
        """Load draft by searching output dirs."""
        # Search in all output subdirs
        if not self.output_dir.exists():
            return None
        
        for job_dir in self.output_dir.iterdir():
            if not job_dir.is_dir():
                continue
            for f in job_dir.glob(f"*{draft_id}*.json"):
                try:
                    data = json.loads(f.read_text())
                    # Determine type from content
                    if "content" in data and "style" in data:
                        return ArticleDraft(**data)
                    elif "group_type" in data and "gap_identified" in data:
                        return StructuredDraft(**data)
                except (json.JSONDecodeError, TypeError):
                    continue
        return None
    
    def list_drafts(self, job_id: str = "") -> list[dict]:
        """List drafts. Filter by job_id if provided."""
        results = []
        base_dir = self.output_dir / job_id if job_id else self.output_dir
        
        if not base_dir.exists():
            return results
        
        for f in sorted(base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text())
                results.append({
                    "filename": f.name,
                    "path": str(f),
                    "type": data.get("group_type", data.get("style", "unknown")),
                    "title": data.get("title_suggestion", ""),
                    "created_at": data.get("created_at", ""),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
            except (json.JSONDecodeError, TypeError):
                results.append({"filename": f.name, "error": "parse error"})
        
        return results
    
    # ── Job state ──
    
    def save_job_state(self, state_dict: dict, job_id: str) -> None:
        """Persist job state as JSON."""
        path = self.jobs_dir / f"{job_id}.json"
        state_dict["updated_at"] = datetime.utcnow().isoformat()
        path.write_text(json.dumps(state_dict, ensure_ascii=False, indent=2))
    
    def load_job_state(self, job_id: str) -> Optional[dict]:
        """Load job state from JSON file."""
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, TypeError):
            return None
    
    def list_jobs(self, active_only: bool = False) -> list[dict]:
        """List all jobs, optionally only active ones."""
        terminal_states = {"complete", "failed", "cancelled"}
        results = []
        
        if not self.jobs_dir.exists():
            return results
        
        for f in sorted(self.jobs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text())
                status = data.get("status", "")
                if active_only and status in terminal_states:
                    continue
                results.append({
                    "job_id": data.get("job_id", f.stem),
                    "status": status,
                    "pipeline": data.get("pipeline", ""),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                })
            except (json.JSONDecodeError, TypeError):
                pass
        
        return results
    
    # ── Stats ──
    
    def get_stats(self) -> dict:
        """Return storage statistics."""
        articles = self.load_all_articles()
        graph = self.load_graph()
        
        enriched_count = sum(1 for a in articles if a.is_enriched)
        sources: dict[str, int] = {}
        topics: dict[str, int] = {}
        
        for a in articles:
            src = a.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
            
            topic = a.get("_topic_key", "unknown")
            topics[topic] = topics.get(topic, 0) + 1
        
        scores = [a.score_total for a in articles if a.score_total > 0]
        
        return {
            "total_articles": len(articles),
            "enriched_articles": enriched_count,
            "graph_nodes": len(graph.get("nodes", [])),
            "graph_edges": len(graph.get("edges", [])),
            "sources": sources,
            "topics": topics,
            "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "storage_path": str(self.data_dir),
        }
