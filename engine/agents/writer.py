"""Writer Agent -- StructuredDraft -> WrittenArticle. (Sprint 5)

Logika:
1. Poluchaet StructuredDraft ot Reader'a
2. Zagruzhaet polnyy tekst istochnikov (esli nuzhno)
3. Generiruyet stat'yu v Markdown/LaTeX cherez LLM
4. Vozvrashchaet WrittenArticle s tekstim, sekcjami, referencami

Formaty:
- Markdown (po umolchaniyu) — dlya blogov, Telegram, Ghost
- LaTeX — dlya akademicheskikh publikatsiy
"""

from __future__ import annotations

from engine.agents.base import BaseAgent, LLMCallMixin
from engine.agents.tools import AgentTools
from engine.schemas import (
    GroupType, WrittenArticle, AgentResult,
    StructuredDraft,
)

# ── Prompty ────────────────────────────────────────────────

WRITER_SYSTEM_PROMPT = """Ty -- opytnyy nauchnyy pisatel' v oblasti geologii i ekologii.
Pishesh' stat'i na osnove strukturirovannogo chernovika.

Pravila:
1. Pisat' na yazyke ukazannom v zaprose (ru/en)
2. Soblydat' strukturu: titul -> annotatsiya -> vvedeniye -> metody -> rezultaty -> obuzhdeniye -> zaklyucheniye -> literaturu
3. Vse utverzhdeniya dolzhny byt' obosnovany (ssylki naistochniki)
4. Dlya REPLICATION: podrobno opisat' metodologiyu, dannye, infrastrukturu
5. Dlya REVIEW: sravnitel'nyy analiz, trendy, gapy
6. Format: {format}

Otchay tolko stat'yu (bez obolochki JSON):
{{"title": "...", "sections": [{{"heading": "...", "content": "..."}}], "references": ["..."], "word_count": N}}"""

WRITER_PROMPT = """NAPISAT' STAT'YU

Topic/Title suggestion: {title_suggestion}
Abstract suggestion: {abstract_suggestion}
Type: {group_type}
Language: {language}
User comment: {user_comment}

=== STRUCTURED DRAFT ===
Gap identified: {gap_identified}
Proposed contribution: {proposed_contribution}
Methods summary: {methods_summary}
Confidence: {confidence}
Keywords: {keywords}

{type_specific}

=== SOURCE ARTICLES ===
{source_articles_info}

Write a complete {group_type} article in {language} ({format} format).
{extra_style_instructions}"""


class WriterAgent(BaseAgent, LLMCallMixin):
    """Pishet stat'yu na osnove StructuredDraft."""

    @property
    def name(self) -> str:
        return "writer"

    def run(
        self,
        draft: StructuredDraft | None = None,
        style: str = "academic",
        language: str = "ru",
        format_: str = "markdown",
        user_comment: str = "",
        **kwargs,
    ) -> AgentResult:
        """
        Zapisat' stat'yu.

        Args:
            draft: StructuredDraft ot Reader'a
            style: academic | blog | popular
            language: ru | en
            format_: markdown | latex
            user_comment: dopolnitel'nye ukazaniya polzovatelya

        Returns:
            AgentResult s WrittenArticle v .data
        """
        self._log(f"Pisaniye nachato: type={draft.group_type.value if draft else '?'}, lang={language}, fmt={format_}")

        if not draft:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="Ne peredan draft (StructuredDraft)",
            )

        try:
            # 1. Sobiraem info ob istochnikakh
            tools = AgentTools(self.storage)
            source_info = self._build_source_info(draft, tools)

            # 2. Formiruem prompt
            prompt = self._build_prompt(draft, source_info, style, language, format_, user_comment)

            # 3. Generiruyem stat'yu
            raw_article = self.call_llm(
                prompt=prompt,
                system=WRITER_SYSTEM_PROMPT.format(format=format_),
                max_tokens=8192,
                parse_json=True,
                temperature=0.4,  # nemnogo kreativnosti dlya pisatelya
            )

            # 4. Parsim v WrittenArticle
            article = self._parse_written(raw_article, draft, format_, language)

            return AgentResult(
                agent_name=self.name,
                success=True,
                data=article,
            )

        except Exception as e:
            self._log(f"Oshibka: {e}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    def _build_source_info(self, draft: StructuredDraft, tools: AgentTools) -> str:
        """Sobrat' kratkuju informaciju ob istochnikakh."""
        parts = []
        for i, doi in enumerate(draft.source_articles[:10], 1):
            art = tools.search_by_doi(doi)
            if art:
                part = f"{i}. {art.title}"
                if art.authors:
                    part += f" ({art.authors})"
                if art.abstract:
                    part += f"\n   {art.abstract[:200]}"
                parts.append(part)
            else:
                parts.append(f"{i}. DOI: {doi} (ne naydena v storage)")

        if not parts:
            return "(No source articles specified)"
        return "\n\n".join(parts)

    def _build_prompt(
        self,
        draft: StructuredDraft,
        source_info: str,
        style: str,
        language: str,
        format_: str,
        user_comment: str,
    ) -> str:
        """Sobrat' polnyy prompt dlya pisatelya."""
        # Tip-spetsifichnaya informacija
        type_specific = ""
        if draft.group_type == GroupType.REPLICATION:
            type_specific = f"""Replication details:
- Methods: {draft.methods_summary}
- Data requirements: {draft.data_requirements}
- Infrastructure: {draft.infrastructure_needs}
- Code availability: {draft.code_availability}
- Reproducibility score: {draft.reproducibility_score}
- Metrics: {draft.metrics}
- Baseline comparison: {draft.baseline_comparison}"""
        elif draft.group_type == GroupType.REVIEW:
            type_specific = f"""Review details:
- Methodology: {draft.methodology}
- Trends: {draft.trends_identified}
- Scope: {draft.scope}
- Articles covered: {draft.articles_covered}"""
        elif draft.group_type == GroupType.DATA_PAPER:
            type_specific = f"""Dataset details:
- Description: {draft.dataset_description}
- Access: {draft.access_method}
- Format: {draft.format_}
- Size: {draft.size_gb} GB
- Coverage: {draft.coverage}
- Usage examples: {draft.usage_examples}"""

        # Style instruktsii
        style_map = {
            "academic": "Use formal academic style with proper citations.",
            "blog": "Write engaging blog post style, accessible to educated non-specialists.",
            "popular": "Popular science style, minimal jargon, vivid explanations.",
        }
        extra_style = style_map.get(style, "")

        return WRITER_PROMPT.format(
            title_suggestion=draft.title_suggestion or "(to be generated)",
            abstract_suggestion=draft.abstract_suggestion or "(to be generated)",
            group_type=draft.group_type.value,
            language=language,
            user_comment=user_comment or "(none)",
            gap_identified=draft.gap_identified,
            proposed_contribution=draft.proposed_contribution,
            methods_summary=draft.methods_summary or "(not specified)",
            confidence=draft.confidence,
            keywords=", ".join(draft.keywords) if draft.keywords else "(none)",
            type_specific=type_specific,
            source_articles_info=source_info,
            format=format_,
            extra_style_instructions=extra_style,
        )

    def _parse_written(
        self,
        raw: dict | str,
        draft: StructuredDraft,
        format_: str,
        language: str,
    ) -> WrittenArticle:
        """Parsim LLM-otvet v WrittenArticle."""
        if isinstance(raw, str):
            # LLM vernul tekst mimo JSON — sobiraem v odnu stat'yu
            return WrittenArticle(
                text=raw,
                title=draft.title_suggestion or "Generated Article",
                format_=format_,
                language=language,
                word_count=len(raw.split()),
                metadata={"source_draft_id": draft.draft_id},
            )

        if not isinstance(raw, dict):
            raw = {}

        # Sobiraem sektsii
        sections = raw.get("sections", [])
        if isinstance(sections, list):
            sections = [
                {"heading": s.get("heading", ""), "content": s.get("content", "")}
                if isinstance(s, dict) else {"heading": "", "content": str(s)}
                for s in sections
            ]
        else:
            sections = []

        # Skladyvaem polnyy tekst iz sekciy
        text_parts = []
        title = raw.get("title", draft.title_suggestion or "Generated Article")
        text_parts.append(f"# {title}\n")
        for sec in sections:
            if sec["heading"]:
                text_parts.append(f"## {sec['heading']}\n")
            text_parts.append(sec["content"] + "\n")
        full_text = "\n".join(text_parts)

        references = raw.get("references", [])
        if isinstance(references, list):
            refs_text = "\n".join(f"- {r}" for r in references)
            full_text += f"\n## References\n{refs_text}\n"

        word_count = len(full_text.split())

        return WrittenArticle(
            text=full_text,
            title=title,
            format_=format_,
            language=language,
            word_count=word_count,
            references=references if isinstance(references, list) else [],
            sections=sections,
            metadata={
                "source_draft_id": draft.draft_id,
                "group_type": draft.group_type.value,
                "llm_raw_keys": list(raw.keys()),
            },
        )

    def estimate_cost(self, draft: StructuredDraft | None = None,
                      num_sources: int = 5) -> dict:
        """Otsenit' stoimost'."""
        input_tokens = 3000 + (num_sources * 500)
        output_tokens = 4096  # dlinnaya stat'ya
        return {
            "estimated_tokens": input_tokens + output_tokens,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
        }

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        """Validirovat'."""
        draft = kwargs.get("draft")
        if not draft:
            return False, "obhyazatel'en parametr 'draft' (StructuredDraft)"
        return True, ""
