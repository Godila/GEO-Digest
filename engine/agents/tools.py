"""
Agent Tools — shared utilities used by multiple agents.

These are NOT LLM tools (function calling). These are Python helper functions
that agents call during their run() method:
  - search_articles() — find articles in storage
  - load_article() — get full article by ID/DOI
  - download_pdf() — fetch full text
  - format_article_for_llm() — convert article to prompt-friendly text
  - extract_citations() — parse references from text
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from engine.config import get_config
from engine.schemas import Article, StructuredDraft, ArticleGroup, GroupType
from engine.storage.base import StorageBackend


class AgentTools:
    """
    Shared toolkit for all agents.
    
    Instantiated once per agent run. Provides access to storage,
    search, PDF downloading, and formatting utilities.
    """
    
    def __init__(self, storage: StorageBackend | None = None):
        self.storage = storage or self._get_storage()
        self._pdf_cache_dir = get_config().data_dir / "pdf_cache"
        self._pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_storage(self):
        from engine.storage import get_storage
        return get_storage()
    
    # ── Search ──

    def search(
        self,
        query: str = "",
        topic: str = "",
        source: str = "",
        limit: int = 50,
        min_score: float = 0.0,
    ) -> list[Article]:
        """Search articles in storage with filters."""
        articles, _ = self.storage.search_articles(
            query=query, topic=topic, source=source, limit=limit * 3  # oversample for filtering
        )

        if min_score > 0:
            articles = [a for a in articles if a.score_total >= min_score]

        return articles[:limit]

    def search_fresh(
        self,
        query: str,
        limit: int = 50,
        min_year: int = 2023,
        save_to_storage: bool = True,
    ) -> list[Article]:
        """
        Perform fresh search across all academic sources (OpenAlex, SemanticScholar, etc.).

        This is the bridge between the Agent Engine and the GEO-Digest sources module.
        Results are deduplicated against seen DOIs and optionally saved to storage.

        Args:
            query: Search query string (e.g., "permafrost carbon feedback Arctic")
            limit: Max results to return
            min_year: Minimum publication year (default 2023)
            save_to_storage: If True, persist new articles to JSONL storage

        Returns:
            List of Article objects found across all active sources.
        """
        import sys
        from pathlib import Path

        # Resolve path to sources module relative to project root
        _project_root = Path(__file__).resolve().parent.parent.parent  # engine/ → project root
        _sources_path = str(_project_root / "scripts")

        if _sources_path not in sys.path:
            sys.path.insert(0, _sources_path)

        try:
            from sources import get_active_sources, search_all_sources, dedup_and_merge
        except ImportError as e:
            print(f"  [Sources] Import error: {e}", file=sys.stderr)
            print(f"  [Sources] Tried path: {_sources_path}", file=sys.stderr)
            return []

        # Get available sources
        sources_list = get_active_sources()
        if not sources_list:
            print("  [Sources] No active sources available (check API keys)", file=sys.stderr)
            return []

        print(f"  [Sources] Searching with {len(sources_list)} source(s): {', '.join(s.name() for s in sources_list)}", file=sys.stderr)

        # Run search
        raw_results = search_all_sources(sources_list, query, min_year=min_year)
        print(f"  [Sources] Raw results: {len(raw_results)} articles", file=sys.stderr)

        if not raw_results:
            return []

        # Deduplicate against seen DOIs
        seen = self.storage.seen_dois()
        unique_articles, doi_index = dedup_and_merge(raw_results, seen)
        print(f"  [Sources] After dedup: {len(unique_articles)} unique articles", file=sys.stderr)

        # Convert to Article objects
        articles = []
        for art_dict in unique_articles:
            try:
                article = Article(art_dict)
                articles.append(article)
            except Exception as e:
                print(f"  [Sources] Skip malformed article: {e}", file=sys.stderr)

        # Sort by citations (best first) then year (newest first)
        articles.sort(key=lambda a: (
            -(int(a.get("citations") or 0)),
            -(int(a.get("year") or 0)),
        ))

        # Save new articles to storage
        if save_to_storage and articles:
            added = self.storage.add_articles_batch([a.data for a in articles])
            if added:
                print(f"  [Sources] Saved {added} new articles to storage", file=sys.stderr)

        return articles[:limit]
    
    def search_by_doi(self, doi: str) -> Optional[Article]:
        """Find article by DOI."""
        if not doi:
            return None
        decoded = doi.replace("%2F", "/")
        return self.storage.get_article_by_id(decoded)
    
    def get_top_articles(
        self,
        limit: int = 25,
        topic: str = "",
        enriched_only: bool = True,
    ) -> list[Article]:
        """Get top-scoring articles, optionally filtered by topic."""
        articles, _ = self.storage.search_articles(topic=topic, limit=limit * 2)
        
        if enriched_only:
            articles = [a for a in articles if a.is_enriched]
        
        articles.sort(key=lambda a: a.score_total, reverse=True)
        return articles[:limit]
    
    def get_all_enriched(self) -> list[Article]:
        """Get all enriched articles sorted by score."""
        articles = self.storage.load_all_articles()
        enriched = [a for a in articles if a.is_enriched]
        enriched.sort(key=lambda a: a.score_total, reverse=True)
        return enriched
    
    # ── Single article ──
    
    def load_article(self, identifier: str) -> Optional[Article]:
        """Load article by DOI, canonical_id, or _id."""
        return self.storage.get_article_by_id(identifier)
    
    # ── Formatting for LLM ──
    
    def format_article_summary(self, article: Article, include_llm: bool = True) -> str:
        """
        Format single article as compact text block for LLM prompt.
        
        Output example:
          [1] Smith et al. (2024). Title here.
              Journal Name. Score: 4.5/5
              Abstract: Lorem ipsum...
              
              LLM Summary: Key findings...
        """
        from engine.utils import format_citation
        
        lines = []
        idx = getattr(article, "_idx", "")
        prefix = f"[{idx}] " if idx else ""
        
        citation = format_citation(article)
        lines.append(f"{prefix}{citation}")
        lines.append(f"  DOI: {article.get('doi', 'N/A')}")
        lines.append(f"  Source: {article.get('source', '?')} | Type: {article.get('article_type', '?')}")
        lines.append(f"  Year: {article.get('year', '?')} | Citations: {article.get('citations', '?')}")
        lines.append(f"  Score: {article.score_total:.1f}/5")
        
        abstract = article.get("abstract_ru") or article.get("abstract", "")
        if abstract:
            lines.append(f"\n  Abstract: {abstract[:500]}")
        
        if include_llm and article.is_enriched:
            llm = article.get("llm_summary", "")
            if llm:
                # Truncate very long summaries
                llm_text = llm if len(llm) < 800 else llm[:800] + "..."
                lines.append(f"\n  AI Summary: {llm_text}")
        
        topics = article.get("topics_ru") or article.get("topics", [])
        if topics:
            lines.append(f"  Topics: {', '.join(topics)}")
        
        return "\n".join(lines)
    
    def format_articles_batch(
        self,
        articles: list[Article],
        max_per_topic: int = 10,
        include_llm: bool = True,
    ) -> str:
        """
        Format batch of articles as structured text for LLM.
        
        Groups by topic, numbered sequentially.
        """
        if not articles:
            return "(No articles found)"
        
        # Group by topic
        groups: dict[str, list[Article]] = {}
        for art in articles:
            key = art.get("_topic_key", art.get("topics", ["other"])[0] if art.get("topics") else "other")
            if key not in groups:
                groups[key] = []
            if len(groups[key]) < max_per_topic:
                groups[key].append(art)
        
        blocks = []
        total = 0
        
        for topic_key, arts in sorted(groups.items()):
            topic_name = arts[0].get("_topic_name_ru", topic_key) if arts else topic_key
            blocks.append(f"\n=== Тема: {topic_name} ({len(arts)} статей) ===")
            
            for i, art in enumerate(arts):
                total += 1
                art._idx = total
                blocks.append(self.format_article_summary(art, include_llm=include_llm))
                blocks.append("")  # blank line between
        
        header = f"Всего статей: {total}\n{'='*50}"
        return header + "\n".join(blocks)
    
    def format_structured_draft(self, draft: StructuredDraft) -> str:
        """Format StructuredDraft as readable text for approval display."""
        lines = [
            f"═══ ПОЛУФАБРИКАТ СТАТЬИ ═══",
            f"",
            f"ID: {draft.draft_id}",
            f"Тип: {draft.group_type.value}",
            f"Заголовок: {draft.title_suggestion}",
            f"Confidence: {draft.confidence:.0%}",
            f"Усилия: {draft.estimated_effort}",
            f"",
            f"── Источники ({len(draft.source_articles)}) ──",
        ]
        
        for i, doi in enumerate(draft.source_articles, 1):
            art = self.load_article(doi)
            if art:
                lines.append(f"  {i}. {art.display_title} ({doi})")
            else:
                lines.append(f"  {i}. {doi}")
        
        if draft.gap_identified:
            lines.extend([f"", f"── Исследовательский gap ──", draft.gap_identified])
        
        if draft.proposed_contribution:
            lines.extend([f"", f"── Предлагаемый вклад ──", draft.proposed_contribution])
        
        if draft.methods_summary:
            lines.extend([f"", f"── Методы ──", draft.methods_summary])
        
        dr = draft.data_requirements
        if any([dr.input_data, dr.data_format, dr.volume_estimate]):
            lines.extend([
                f"",
                f"── Требования к данным ──",
                f"  Входные данные: {dr.input_data}",
                f"  Формат: {dr.data_format}",
                f"  Объём: {dr.volume_estimate}",
                f"  Способ получения: {dr.acquisition}",
            ])
        
        infra = draft.infrastructure_needs
        if any([infra.hardware, infra.software]):
            lines.extend([
                f"",
                f"── Инфраструктура ──",
                f"  Железо: {infra.hardware}",
                f"  ПО: {', '.join(infra.software) if infra.software else '-'}",
                f"  Время вычислений: {infra.compute_time}",
            ])
        
        if draft.code_availability:
            lines.extend([f"", f"── Код ──", draft.code_availability])
        
        if draft.keywords:
            lines.extend([f"", f"Ключевые слова: {', '.join(draft.keywords)}"])
        
        return "\n".join(lines)
    
    # ── PDF handling ──
    
    def download_pdf(self, article: Article, timeout: int = 30) -> Optional[Path]:
        """
        Download PDF for article.
        
        Tries in order:
          1. oa_url from Unpaywall
          2. Direct URL
          3. arXiv URL (if arXiv ID present)
        
        Returns path to cached PDF file, or None.
        """
        cfg = get_config()
        
        urls_to_try = []
        
        # 1. Open Access URL (best quality)
        oa_url = article.get("oa_url", "")
        if oa_url and ("pdf" in oa_url.lower() or "arxiv" in oa_url.lower() or "doi" in oa_url.lower()):
            urls_to_try.append(("OA", oa_url))
        
        # 2. Regular URL
        url = article.get("url", "")
        if url and url != oa_url:
            urls_to_try.append(("URL", url))
        
        # 3. arXiv
        doi = article.get("doi", "")
        if "10.48550" in doi or "arxiv" in doi.lower():
            arxiv_id = doi.split("/")[-1] if "/" in doi else doi
            urls_to_try.append(("arXiv", f"https://arxiv.org/pdf/{arxiv_id}.pdf"))
        
        # Also check external_ids
        ext_ids = article.get("external_ids", {})
        if isinstance(ext_ids, dict):
            arxiv_id = ext_ids.get("ArXiv", "")
            if arxiv_id:
                urls_to_try.append(("arXiv-id", f"https://arxiv.org/pdf/{arxiv_id}.pdf"))
        
        for label, url in urls_to_try:
            path = self._try_download_pdf(url, article.canonical_id, timeout)
            if path:
                print(f"  [PDF] Downloaded via {label}: {path.name}", file=sys.stderr)
                return path
        
        print(f"  [PDF] Not available for {article.display_title}", file=sys.stderr)
        return None
    
    def _try_download_pdf(self, url: str, cache_key: str, timeout: int = 30) -> Optional[Path]:
        """Attempt to download PDF from URL. Returns path or None."""
        import hashlib
        
        safe_name = hashlib.md5(cache_key.encode()).hexdigest()[:12]
        cache_path = self._pdf_cache_dir / f"{safe_name}.pdf"
        
        # Return cached version if exists and non-empty
        if cache_path.exists() and cache_path.stat().st_size > 1000:
            return cache_path
        
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (GEO-Digest Research Agent; contact@geo-digest.org)",
                "Accept": "application/pdf,*/*",
            })
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                
            # Validate it's actually a PDF
            if len(data) < 1000 or not data.startswith(b"%PDF"):
                return None
            
            cache_path.write_bytes(data)
            return cache_path
            
        except Exception as e:
            print(f"  [PDF] Failed {url[:60]}...: {e}", file=sys.stderr)
            return None
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file.
        
        Tries PyMuPDF first, falls back to pdfplumber, then pdftotext CLI.
        """
        # Try PyMuPDF (fitz)
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            texts = []
            for page in doc:
                t = page.get_text()
                if t.strip():
                    texts.append(t)
            doc.close()
            result = "\n\n".join(texts)
            if len(result.strip()) > 200:
                return result
        except ImportError:
            pass
        except Exception as e:
            print(f"  [PDF] PyMuPDF error: {e}", file=sys.stderr)
        
        # Try pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(pdf_path)) as pdf:
                texts = [p.extract_text() or "" for p in pdf.pages]
            result = "\n\n".join(texts)
            if len(result.strip()) > 200:
                return result
        except ImportError:
            pass
        except Exception as e:
            print(f"  [PDF] pdfplumber error: {e}", file=sys.stderr)
        
        # Try pdftotext CLI
        try:
            import subprocess
            result = subprocess.run(
                ["pdftotext", "-layout", str(pdf_path), "-"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and len(result.stdout.strip()) > 200:
                return result.stdout
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  [PDF] pdftotext error: {e}", file=sys.stderr)
        
        return ""
    
    # ── Stats ──
    
    def get_stats(self) -> dict:
        """Delegate to storage stats."""
        return self.storage.get_stats()
