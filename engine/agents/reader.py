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
            for doi in dois:
                art = tools.search_by_doi(doi)
                if art:
                    articles.append(art)
            return articles

        return []

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
