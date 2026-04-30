"""Reader Agent -- PDF -> StructuredDraft. (Sprint 3)

Logika:
1. Poluchaet ArticleGroup ot Scout'a (ili spisok DOI)
2. Skachivaet PDF dlya kazhdoy stat'i
3. Izvlekaet tekst iz PDF
4. Otpravlyaet LLM polnyy tekst + metadata
5. Vozvrashchaet StructuredDraft s strukturirovannym analizom

Dlya raznykh GroupType raznye polya:
- REPLICATION: methods, data_reqs, infra, code, reproducibility, gaps
- REVIEW: methodology, trends, scope, baseline, gaps
- DATA_PAPER: dataset_desc, access, format, size, coverage
"""

from __future__ import annotations
import os

from engine.agents.base import BaseAgent, LLMCallMixin
from engine.agents.tools import AgentTools
from engine.schemas import (
    Article, ArticleGroup, GroupType,
    DataRequirements, InfrastructureNeeds,
    StructuredDraft, AgentResult,
)

# ── Prompty ────────────────────────────────────────────────

READER_SYSTEM_PROMPT = """Ty -- nauchnyy analitik geologo-ekologicheskikh issledovaniy.
Analiziruesh polnyy tekst stat'i i sozdaesh' strukturirovannyy chernovik.

Otchay v JSON tolko etu strukturu (ne dobavlyay drugie polya):
{
  "methods_summary": "kratkoe opisanie metodologii (2-3 predlozheniya)",
  "gap_identified": "kakoy issledovatel'skiy gap vydelen (1-2 predlozheniya)",
  "proposed_contribution": "chto mozhno napisat' na etoy osnove (2-3 predlozheniya)",
  "confidence": 0.0-1.0,
  "keywords": ["tag1", "tag2", "tag3"],
  "title_suggestion": "predlozheniye zagolovka dlya budushchey stat'i",
  "abstract_suggestion": "predlozheniye annotatsii (3-4 predlozheniya)"
}

Dopolnitel'nye polya zavisyat ot tipa gruppy:
- Dlya REPLICATION dobav': data_requirements, infrastructure_needs,
  code_availability, reproducibility_score, metrics, baseline_comparison
- Dlya REVIEW dobav': methodology, trends_identified, scope
- Dlya DATA_PAPER dobav': dataset_description, access_method, format_,
  size_gb, coverage, usage_examples"""

READER_ARTICLE_PROMPT = """Article: {title}
Authors: {authors}
DOI: {doi}
Abstract: {abstract}

Full text (first {max_chars} chars):
{full_text}

Analyze this article for a {group_type} paper.
{extra_instructions}"""


RICH_READER_SYSTEM_PROMPT = """Ты — научный аналитик геоэкологических исследований.
Проводишь ДЕТАЛЬНЫЙ анализ статей для написания новой научной работы.

Извлеки из каждой статьи:
1. КЛЮЧЕВЫЕ ФАКТЫ с цифрами: точные значения, проценты, размеры выборок, p-values
2. МЕТОДОЛОГИЮ подробно: какие данные, период, территория, модели, ПО
3. РЕЗУЛЬТАТЫ с числами: корреляции, тренды, статистика
4. ПРОТИВОРЕЧИЯ между работами: где авторы расходятся
5. GAP в знаниях — отдельно для каждого с контекстом
6. ЦИТАТЫ: 3-5 ключевых утверждений verbatim для использования в статье

Верни JSON:
{
  "key_facts": [{"claim": "...", "source_doi": "...", "evidence": "..."}],
  "methods_detail": [{"method": "...", "data_source": "...", "period": "...", "tools": "..."}],
  "results_with_numbers": [{"finding": "...", "value": "...", "comparison": "..."}],
  "contradictions": [{"claim_a": "...", "author_a": "...", "claim_b": "...", "author_b": "..."}],
  "gaps": [{"gap": "...", "context": "...", "who_noted": "..."}],
  "verbatim_quotes": [{"quote": "...", "source_doi": "...", "section": "..."}],
  "cross_connections": [{"articles": ["DOI1", "DOI2"], "connection": "..."}]
}"""

EVIDENCE_EXTRACTOR_SYSTEM_PROMPT = """Ты — научный ассистент, извлекающий структурированные evidence из научных статей.
Твоя задача — извлечь конкретные утверждения с точными цитатами для использования в новой научной статье.

Для каждого evidence укажи:
1. Точную цитату (verbatim) из текста — на языке оригинала
2. Тип утверждения (claim_type):
   - method_result: результат применения метода (точность, F1, RMSE, и т.д.)
   - finding: научное открытие, наблюдение, закономерность
   - limitation: ограничение метода, данных, подхода
   - comparison: сравнение с другими методами/подходами
   - gap: нерешённая проблема, пробел в знаниях
   - recommendation: рекомендация для будущих исследований
3. Контекст: в какой секции статьи это найдено
4. Ключевые слова для сопоставления с секциями новой статьи

ВАЖНО: Цитаты должны быть ТОЧНЫМИ — verbatim из текста, не пересказ. Минимум 10 evidence на статью.

Верни JSON:
{
  "evidence": [
    {
      "quote": "точная цитата из текста",
      "claim_type": "method_result|finding|limitation|comparison|gap|recommendation",
      "section": "Introduction|Methodology|Results|Discussion|Conclusion",
      "page": 0,
      "keywords": ["keyword1", "keyword2"],
      "context": "краткое пояснение что означает эта цитата"
    }
  ],
  "summary": "Краткое описание вклада этой статьи в 1-2 предложениях",
  "methodology_summary": "Описание методологии",
  "key_numbers": ["96.2% accuracy", "500 samples", "2020-2023 period"]
}"""


class ReaderAgent(BaseAgent, LLMCallMixin):
    """Chitaet PDF stat'i i sozdaet StructuredDraft."""

    @property
    def name(self) -> str:
        return "reader"

    def run(
        self,
        group: ArticleGroup | None = None,
        dois: list[str] | None = None,
        full_text: bool = True,
        max_pdf_size_mb: float = 50,
        **kwargs,
    ) -> AgentResult:
        """
        Zapustit' chteniye i analiz.

        Args:
            group: ArticleGroup ot Scout'a
            dois: spisok DOI (alternativa group)
            full_text: skachivat' polnyy PDF ili tol'ko abstract
            max_pdf_size_mb: maks. razmer PDF dlya skachivaniya

        Returns:
            AgentResult s StructuredDraft v .data
        """
        self._log(f"Chteniye nachato: group={group.group_id if group else 'None'}, dois={len(dois) if dois else 0}")

        try:
            # 1. Opredelyaem spisok statey
            articles = self._resolve_articles(group, dois)
            if not articles:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error="Ne ukazany stat'i dlya chteniya (ni group, ni dois)",
                )

            self._log(f"Statey dlya analiza: {len(articles)}")

            # 2. Skachivayem i izvlekayem tekst
            extracted = self._extract_all_texts(articles, full_text, max_pdf_size_mb)

            # 3. LLM analiz
            draft = self._analyze_with_llm(group, articles, extracted)

            # Build rich context for Writer — detailed analysis with facts, quotes, contradictions
            try:
                draft.rich_context = self._build_rich_context(
                    extracted,
                    group.group_type if group else GroupType.REVIEW
                )
            except Exception as e:
                self._log(f"Rich context build warning: {e}")
                draft.rich_context = ""

            # Extract structured evidence blocks for evidence-grounded writing
            try:
                draft.evidence_blocks = self._extract_evidence_blocks(
                    extracted,
                    group.group_type if group else GroupType.REVIEW
                )
                self._log(f"Evidence blocks: {len(draft.evidence_blocks)} sources, "
                          f"{sum(len(eb.get('quotes', [])) for eb in draft.evidence_blocks)} quotes extracted")
            except Exception as e:
                self._log(f"Evidence extraction warning: {e}")
                draft.evidence_blocks = []

            return AgentResult(
                agent_name=self.name,
                success=True,
                data=draft,
            )

        except Exception as e:
            self._log(f"Oshibka: {e}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    def _resolve_articles(
        self, group: ArticleGroup | None, dois: list[str] | None
    ) -> list[Article]:
        """Poluchit' spisok Article iz group ili doi."""
        if group and group.articles:
            return group.articles

        if dois:
            tools = AgentTools(self.storage)
            articles = []
            missing_dois = []
            needs_pdf = []  # Articles in storage but without PDF URL
            for doi in dois:
                art = tools.search_by_doi(doi)
                if art:
                    articles.append(art)
                    # Check if article has no PDF URL — try to enrich via Unpaywall
                    if not getattr(art, 'pdf_url', None):
                        needs_pdf.append((doi, art))
                else:
                    missing_dois.append(doi)
            
            # Enrich articles without PDF URL via Unpaywall API
            if needs_pdf:
                self._log(f"  {len(needs_pdf)} DOI in storage without PDF, enriching via Unpaywall...")
                for doi, art in needs_pdf:
                    self._enrich_pdf_url(doi, art)
            
            # Fallback: create Article stubs for DOIs not in storage
            if missing_dois:
                self._log(f"  {len(missing_dois)} DOI not in storage, resolving via API...")
                for doi in missing_dois:
                    art = self._resolve_doi_from_api(doi)
                    if art:
                        articles.append(art)
                    else:
                        self._log(f"  Could not resolve DOI: {doi}")
            
            self._log(f"  Resolved: {len(articles)}/{len(dois)} articles")
            return articles

        return []
    
    def _enrich_pdf_url(self, doi: str, art: Article) -> None:
        """Enrich Article in-place with PDF URL from Unpaywall."""
        import urllib.parse, json
        try:
            email = os.environ.get("UNPAYWALL_EMAIL", "geo-digest@research.bot")
            url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "GEO-Digest/1.0 (mailto:geo-digest@research.bot)"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            
            pdf_url = (data.get("best_oa_location") or {}).get("url_for_pdf", "")
            if not pdf_url:
                pdf_url = (data.get("first_oa_location") or {}).get("url_for_pdf", "")
            
            if pdf_url:
                # Set on the underlying dict (Article wraps a dict)
                if hasattr(art, '_data'):
                    art._data['pdf_url'] = pdf_url
                self._log(f"    PDF enriched for {doi}: {pdf_url[:60]}")
            else:
                self._log(f"    No OA PDF for {doi}")
        except Exception as e:
            self._log(f"    Unpaywall failed for {doi}: {e}")
    
    def _resolve_doi_from_api(self, doi: str) -> Optional[Article]:
        """Create Article stub from DOI using Unpaywall + Crossref APIs."""
        import urllib.parse, json
        
        title = ""
        abstract = ""
        year = None
        authors = []
        pdf_url = ""
        
        # 1. Unpaywall for OA URL + metadata
        try:
            email = os.environ.get("UNPAYWALL_EMAIL", "geo-digest@research.bot")
            url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "GEO-Digest/1.0 (mailto:geo-digest@research.bot)"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            title = data.get("title", "")
            year = data.get("year")
            pdf_url = (data.get("best_oa_location") or {}).get("url_for_pdf", "")
            if not pdf_url:
                pdf_url = (data.get("first_oa_location") or {}).get("url_for_pdf", "")
            for a in (data.get("z_authors") or [])[:10]:
                authors.append(a.get("given", "") + " " + a.get("family", ""))
        except Exception:
            pass
        
        # 2. Crossref for abstract if missing
        if not title or not abstract:
            try:
                url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
                req = urllib.request.Request(url, headers={
                    "User-Agent": "GEO-Digest/1.0 (mailto:geo-digest@research.bot)"
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    cr = json.loads(resp.read()).get("message", {})
                if not title:
                    title = cr.get("title", [""])[0]
                abstract = cr.get("abstract", "")
                if not year:
                    dp = cr.get("published-print") or cr.get("published-online") or {}
                    year = (dp.get("date-parts") or [[None]])[0][0]
            except Exception:
                pass
        
        if not title:
            return None
        
        return Article({
            "doi": doi,
            "title": title,
            "abstract": abstract[:2000] if abstract else "",
            "year": year or 2024,
            "authors": authors[:10],
            "source": "api_resolved",
            "oa_url": pdf_url,
            "enrichment_status": "basic",
        })

    def _extract_all_texts(
        self,
        articles: list[Article],
        full_text: bool,
        max_pdf_size_mb: float,
    ) -> dict[str, dict]:
        """Skachat' PDF i izvlech' tekst dlya kazhdoy stat'i.

        Returns:
            {doi_or_index: {"article": Article, "text": str, "source": "pdf|abstract|none"}}
        """
        tools = AgentTools(self.storage)
        result = {}

        for i, art in enumerate(articles):
            key = art.doi or f"article_{i}"
            self._log(f"  Obrabotka [{i+1}/{len(articles)}]: {art.title[:60]}")

            text = ""
            source = "none"

            if full_text:
                # Probuem skachat' PDF
                pdf_path = tools.download_pdf(art, timeout=30)
                if pdf_path:
                    text = tools.extract_text_from_pdf(pdf_path)
                    if text:
                        source = "pdf"
                        self._log(f"    PDF: {len(text)} simvolov")

            # Fallback na abstract
            if not text and art.abstract:
                text = art.abstract
                source = "abstract"
                self._log(f"    Abstract: {len(text)} simvolov")

            result[key] = {
                "article": art,
                "text": text,
                "source": source,
            }

        return result

    def _analyze_with_llm(
        self,
        group: ArticleGroup | None,
        articles: list[Article],
        extracted: dict[str, dict],
    ) -> StructuredDraft:
        """Otpravit' izvlechennyy tekst v LLM i poluchit' StructuredDraft."""
        group_type = group.group_type if group else GroupType.REVIEW

        # Sobiraem kontent vseh statey
        parts = []
        for key, data in extracted.items():
            art = data["article"]
            text = data["text"]
            source = data["source"]

            # Ogranichivaem dlinu teksta (LLM context limit)
            max_chars = 15000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... [obrezano]"

            extra = self._get_type_instructions(group_type)
            part = READER_ARTICLE_PROMPT.format(
                title=art.title,
                authors=art.authors or "Unknown",
                doi=art.doi or "N/A",
                abstract=(art.abstract or "")[:500],
                full_text=text,
                max_chars=max_chars,
                group_type=group_type.value,
                extra_instructions=extra,
            )
            parts.append(part)

        # Dlya odnoy stat'i — pryamoy zapros, dlya neskol'kikh — agregatsiya
        if len(parts) == 1:
            raw = self.call_llm(prompt=parts[0], system=READER_SYSTEM_PROMPT,
                                max_tokens=4096, parse_json=True)
            return self._parse_draft(raw, group_type, articles, parts[0])
        else:
            # Mnogo statey: analiziruem po odel'nosti, potom agregiruem
            return self._analyze_multiple(parts, group_type, articles)

    def _get_type_instructions(self, group_type: GroupType) -> str:
        """Dopolnitel'nye instruktsii dlya konkretnogo tipa."""
        instructions = {
            GroupType.REPLICATION: (
                "Focus on: exact methods used, data sources, software/hardware, "
                "hyperparameters, evaluation metrics. Identify what's needed "
                "to replicate this on different regions/data."
            ),
            GroupType.REVIEW: (
                "Focus on: research landscape, methodologies compared, trends, "
                "open questions, gaps in current literature, positioning "
                "for a new review article."
            ),
            GroupType.DATA_PAPER: (
                "Focus on: dataset structure, access method, size, format, "
                "coverage (spatial/temporal), usage examples, limitations, "
                "potential applications."
            ),
        }
        return instructions.get(group_type, "")

    def _analyze_multiple(
        self,
        parts: list[str],
        group_type: GroupType,
        articles: list[Article],
    ) -> StructuredDraft:
        """Analiziruet neskol'ko statey i agregiruet v odin draft."""
        all_raw = []
        for i, part in enumerate(parts):
            self._log(f"  LLM analiz stat'i {i+1}/{len(parts)}")
            raw = self.call_llm(prompt=part, system=READER_SYSTEM_PROMPT,
                                max_tokens=4096, parse_json=True)
            all_raw.append(raw)

        # Agregatsiya: berem pervuyu kak osnovu, dobavlyayem ostal'nye
        base = all_raw[0] if all_raw else {}
        draft = self._parse_draft(base, group_type, articles, parts[0])

        # Dobavlayem trendy/metody iz ostatka
        if group_type == GroupType.REVIEW:
            trends = set(draft.trends_identified or [])
            for raw in all_raw[1:]:
                if isinstance(raw, dict):
                    for t in raw.get("trends_identified", []):
                        trends.add(t)
            draft.trends_identified = list(trends)

        draft.articles_covered = len(articles)
        return draft

    def _extract_evidence_blocks(
        self, extracted: dict, group_type: 'GroupType'
    ) -> list[dict]:
        """Extract structured evidence blocks with verbatim quotes for evidence-grounded writing.

        For each source PDF, calls LLM to extract typed claims with exact quotes,
        page numbers, and section context. Returns list of evidence blocks.
        """
        evidence_blocks = []
        for key, data in extracted.items():
            art = data["article"]
            text = data["text"]
            source = data["source"]

            # Only extract from PDF content (not abstract-only)
            if source != "pdf" or not text or len(text) < 500:
                # Fallback: create minimal evidence block from abstract
                if art.abstract:
                    evidence_blocks.append({
                        "source": f"{art.authors or 'Unknown'}, {art.year or 'n.d.'}",
                        "doi": art.doi or "",
                        "title": art.title,
                        "summary": art.abstract[:300],
                        "methodology_summary": "",
                        "key_numbers": [],
                        "quotes": [],
                    })
                continue

            # Truncate text for LLM (keep up to 15K chars for evidence extraction)
            text_for_llm = text[:15000] if len(text) > 15000 else text

            prompt = f"""Статья: {art.title}
Авторы: {art.authors or 'N/A'}
DOI: {art.doi or 'N/A'}
Год: {art.year or 'N/A'}

Текст статьи:
{text_for_llm}

Тип целевой статьи: {group_type.value}

Извлеки ВСЕ значимые evidence из этой статьи. Минимум 10."""

            raw = self.call_llm(
                prompt=prompt,
                system=EVIDENCE_EXTRACTOR_SYSTEM_PROMPT,
                max_tokens=6000,
                parse_json=True,
            )

            if not isinstance(raw, dict):
                # JSON parse failed — try to extract quotes from raw text
                self._log(f"Evidence JSON parse failed for '{art.title[:50]}', extracting from raw text")
                raw_lines = str(raw)
                # Minimal fallback: create one block with raw text as summary
                evidence_blocks.append({
                    "source": f"{art.authors or 'Unknown'}, {art.year or 'n.d.'}",
                    "doi": art.doi or "",
                    "title": art.title,
                    "summary": raw_lines[:500],
                    "methodology_summary": "",
                    "key_numbers": [],
                    "quotes": [],
                })
                continue

            # Transform to evidence block format
            quotes = []
            for ev in raw.get("evidence", []):
                if not isinstance(ev, dict):
                    continue
                quotes.append({
                    "text": ev.get("quote", ""),
                    "claim_type": ev.get("claim_type", "finding"),
                    "section": ev.get("section", ""),
                    "page": ev.get("page", 0),
                    "keywords": ev.get("keywords", []),
                    "context": ev.get("context", ""),
                })

            block = {
                "source": f"{art.authors or 'Unknown'}, {art.year or 'n.d.'}",
                "doi": art.doi or "",
                "title": art.title,
                "summary": raw.get("summary", ""),
                "methodology_summary": raw.get("methodology_summary", ""),
                "key_numbers": raw.get("key_numbers", []),
                "quotes": quotes,
            }
            evidence_blocks.append(block)

        return evidence_blocks

    def _build_rich_context(self, extracted: dict, group_type: 'GroupType') -> str:
        """Build rich context string for Writer from detailed article analysis."""
        all_analyses = []
        for key, data in extracted.items():
            art = data["article"]
            text = data["text"]
            if not text or text == data["article"].abstract:
                # Skip if only abstract available — not enough for rich analysis
                if data["source"] != "pdf":
                    continue

            prompt_text = f"""Статья: {art.title}
Авторы: {art.authors or 'N/A'}
DOI: {art.doi or 'N/A'}
Год: {art.year or 'N/A'}

Текст статьи:
{text[:12000]}

Проведи детальный анализ для написания {group_type.value} статьи."""

            raw = self.call_llm(
                prompt=prompt_text,
                system=RICH_READER_SYSTEM_PROMPT,
                max_tokens=4096,
                parse_json=True,
            )

            if isinstance(raw, dict):
                analysis = self._format_rich_analysis(art, raw)
                all_analyses.append(analysis)

        return "\n\n---\n\n".join(all_analyses) if all_analyses else ""

    def _format_rich_analysis(self, art, raw: dict) -> str:
        """Format rich analysis dict into readable string for Writer context."""
        parts = [f"## {art.title} ({art.doi or 'no DOI'})"]

        if raw.get("key_facts"):
            parts.append("### Ключевые факты:")
            for f in raw["key_facts"][:8]:
                claim = f.get("claim", "")
                evidence = f.get("evidence", "")
                parts.append(f"  • {claim}" + (f" — {evidence}" if evidence else ""))

        if raw.get("methods_detail"):
            parts.append("### Методология:")
            for m in raw["methods_detail"][:5]:
                parts.append(f"  • {m.get('method', '')}: данные={m.get('data_source', '')}, период={m.get('period', '')}, инструменты={m.get('tools', '')}")

        if raw.get("results_with_numbers"):
            parts.append("### Результаты с цифрами:")
            for r in raw["results_with_numbers"][:8]:
                parts.append(f"  • {r.get('finding', '')}: {r.get('value', '')}" + (f" (сравнение: {r.get('comparison', '')})" if r.get('comparison') else ""))

        if raw.get("contradictions"):
            parts.append("### Противоречия:")
            for c in raw["contradictions"][:5]:
                parts.append(f"  • {c.get('author_a', '?')}: {c.get('claim_a', '')} ↔ {c.get('author_b', '?')}: {c.get('claim_b', '')}")

        if raw.get("gaps"):
            parts.append("### Пробелы в знаниях:")
            for g in raw["gaps"][:5]:
                parts.append(f"  • {g.get('gap', '')} (контекст: {g.get('context', '')})")

        if raw.get("verbatim_quotes"):
            parts.append("### Ключевые цитаты для использования:")
            for q in raw["verbatim_quotes"][:6]:
                parts.append(f"  «{q.get('quote', '')}» — {q.get('source_doi', '')}")

        if raw.get("cross_connections"):
            parts.append("### Связи с другими статьями:")
            for cc in raw["cross_connections"][:5]:
                parts.append(f"  • {', '.join(cc.get('articles', []))}: {cc.get('connection', '')}")

        return "\n".join(parts)

    def _parse_draft(
        self,
        raw: dict | list | str,
        group_type: GroupType,
        articles: list[Article],
        raw_prompt: str,
    ) -> StructuredDraft:
        """Parsim LLM-otvet v StructuredDraft."""
        if not isinstance(raw, dict):
            raw = {}

        # Bazovye polya
        draft_id = f"draft_{group_type.value}_{hash(raw_prompt) % 10000:04x}"

        # Data requirements (tol'ko dlya REPLICATION)
        data_req = None
        infra = None
        if group_type == GroupType.REPLICATION:
            dr = raw.get("data_requirements", {})
            if dr and isinstance(dr, dict):
                data_req = DataRequirements(
                    data_types=dr.get("data_types", []),
                    spatial_coverage=dr.get("spatial_coverage", ""),
                    temporal_coverage=dr.get("temporal_coverage", ""),
                    sample_size=dr.get("sample_size"),
                    format_notes=dr.get("format_notes", ""),
                )
            inf = raw.get("infrastructure_needs", {})
            if inf and isinstance(inf, dict):
                infra = InfrastructureNeeds(
                    software=inf.get("software", []),
                    hardware=inf.get("hardware", ""),
                    compute_hours=inf.get("compute_hours"),
                    expertise=inf.get("expertise", []),
                    cost_estimate=inf.get("cost_estimate", ""),
                )

        draft = StructuredDraft(
            draft_id=draft_id,
            group_type=group_type,
            source_articles=[a.doi for a in articles if a.doi],
            title_suggestion=raw.get("title_suggestion", ""),
            abstract_suggestion=raw.get("abstract_suggestion", ""),
            keywords=raw.get("keywords", []),
            gap_identified=raw.get("gap_identified", ""),
            proposed_contribution=raw.get("proposed_contribution", ""),
            confidence=float(raw.get("confidence", 0.5)),
            methods_summary=raw.get("methods_summary", ""),
            architecture=raw.get("architecture", ""),
            data_requirements=data_req,
            infrastructure_needs=infra,
            code_availability=raw.get("code_availability", ""),
            metrics=raw.get("metrics", {}),
            baseline_comparison=raw.get("baseline_comparison", ""),
            reproducibility_score=float(raw.get("reproducibility_score", 0.0)),
            scope=raw.get("scope", ""),
            articles_covered=len(articles),
            methodology=raw.get("methodology", ""),
            trends_identified=raw.get("trends_identified", []),
            dataset_description=raw.get("dataset_description", ""),
            access_method=raw.get("access_method", ""),
            format_=raw.get("format_", ""),
            size_gb=float(raw.get("size_gb", 0.0)),
            coverage=raw.get("coverage", ""),
            usage_examples=raw.get("usage_examples", []),
            raw_llm_output=str(raw),
        )
        return draft

    def estimate_cost(self, group: ArticleGroup | None = None,
                      num_articles: int = 3) -> dict:
        """Otsenit' stoimost' (tokeny)."""
        avg_article_tokens = 8000  # ~5-8k tokenov na stat'yu
        output_per_article = 2000
        total_input = 500 + (num_articles * avg_article_tokens)
        total_output = num_articles * output_per_article
        return {
            "estimated_tokens": total_input + total_output,
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "num_articles": num_articles,
        }

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        """Validirovat' vhodnye parametry."""
        group = kwargs.get("group")
        dois = kwargs.get("dois")
        if not group and not dois:
            return False, "ukazhi 'group' (ArticleGroup) ili 'dois' (spisok DOI)"
        if dois and not isinstance(dois, (list, tuple)):
            return False, "'dois' dolzhen byt' spiskom"
        return True, ""
