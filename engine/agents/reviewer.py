"""Reviewer Agent -- fact-check + style review s drugoy model'yu. (Sprint 6)

Logika:
1. Poluchaet WrittenArticle ot Writer'a
2. Ispol'zuyet OTDEL'NUYU model' (ReviewerConfig: OpenAI-compatible)
3. Fact-check: proveryaet utverzhdeniya protiv istochnikov
4. Style review: grammatika, struktura, yasnost'
5. Vozvrashchaet ReviewedDraft s verdiktom i spiskom popravok

Verdikty:
- ACCEPT — stat'ya otlichnaya, mozhno publikovat'
- ACCEPT_WITH_MINOR — melkie pravki, ne kritichno
- NEEDS_REVISION — nuzhna pererabotka
- REJECT — ne podkhodit, nuzhno pisat' zanovo
"""

from __future__ import annotations

from engine.agents.base import BaseAgent, LLMCallMixin
from engine.schemas import (
    WrittenArticle, ReviewedDraft, Edit, FactCheck,
    Severity, ReviewVerdict, AgentResult,
    GroupType,
)

# ── Prompty ────────────────────────────────────────────────

REVIEWER_SYSTEM_PROMPT = """Ty -- strogiy nauchnyy rezent (reviewer) po geologii i ekologii.
Proveryaesh' stat'i na tochnost', kachestvo i sootvetstviye standartam.

Tvoja zadacha:
1. FACT-CHECK: Proverit' kazhdoe utverzhdeniye (est' li dokazatel'stva?)
2. STYLE CHECK: Grammatika, stilistika, yasnost', struktura
3. STRUCTURE: Pravil'nost' nauchnoy struktury stat'i
4. CITATIONS: Nalichiye ssylok na istochniki

Otchay v JSON:
{
  "verdict": "ACCEPT|ACCEPT_WITH_MINOR|NEEDS_REVISION|REJECT",
  "overall_score": 0.0-1.0,
  "summary": "obshchij otzyv (2-3 predlozheniya)",
  "edits": [
    {"location": "section/paragraph", "severity": "critical|major|minor",
     "original": "tekst", "suggested": "ispravlennyy tekst",
     "reason": "pochemu", "category": "fact|style|structure|citation"}
  ],
  "fact_checks": [
    {"claim": "utverzhdeniye", "source_doi": "doi_istochnika",
     "verified": true/false, "actual_text": "chto v istochnike",
     "verdict": "confirmed|contradicted|not_found"}
  ],
  "issues": ["spisok problem otdel'no"],
  "severity_counts": {"critical": N, "major": N, "minor": N}
}"""

REVIEWER_PROMPT = """PROVERIT' STAT'YU

Title: {title}
Format: {format}
Language: {language}
Word count: {word_count}

=== FULL ARTICLE TEXT ===
{article_text}

=== SOURCE REFERENCES ===
{references}

Strictness level: {strictness}
(1=liberal, 5=extremely strict)

Review this article thoroughly. Check every claim against sources.
Be specific about what needs fixing."""


class ReviewerAgent(BaseAgent, LLMCallMixin):
    """Rezentuet stat'yu s pomoshch'yu drugoy modeli."""

    @property
    def name(self) -> str:
        return "reviewer"

    def run(
        self,
        article: WrittenArticle | None = None,
        source_articles: list | None = None,
        strictness: int = 3,
        **kwargs,
    ) -> AgentResult:
        """
        Zapustit' rezenzirovanie.

        Args:
            article: WrittenArticle ot Writer'a
            source_articles: spisok statey-dlya fact-checka
            strictness: 1-5 (1=legkiy, 5=strogij)

        Returns:
            AgentResult s ReviewedDraft v .data
        """
        self._log(f"Rezenzirovanie: \"{article.title if article else '?'}\", strict={strictness}")

        if not article:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="Ne peredana stat' (WrittenArticle)",
            )

        try:
            # 1. Formiruem prompt
            refs_text = self._format_references(source_articles or [])
            prompt = REVIEWER_PROMPT.format(
                title=article.title,
                format=article.format_,
                language=article.language,
                word_count=article.word_count,
                article_text=article.text[:25000],  # limit dlya context
                references=refs_text,
                strictness=strictness,
            )

            # 2. Vyzyvaem reviewer model' (otdel'naya ot writer!)
            raw_review = self.call_llm(
                prompt=prompt,
                system=REVIEWER_SYSTEM_PROMPT,
                max_tokens=4096,
                parse_json=True,
                temperature=0.2,  # nizkaja temperatura dlya konservativnosti
            )

            # 3. Parsim v ReviewedDraft
            reviewed = self._parse_review(raw_review, article)

            return AgentResult(
                agent_name=self.name,
                success=True,
                data=reviewed,
            )

        except Exception as e:
            self._log(f"Oshibka: {e}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    def _format_references(self, source_articles: list) -> str:
        """Formatiruyem istochniki dlya prompta."""
        if not source_articles:
            return "(No source articles provided for fact-checking)"
        parts = []
        for i, art in enumerate(source_articles, 1):
            if hasattr(art, 'title'):
                parts.append(f"{i}. {art.title}")
                if hasattr(art, 'doi') and art.doi:
                    parts.append(f"   DOI: {art.doi}")
                if hasattr(art, 'abstract') and art.abstract:
                    parts.append(f"   {art.art.abstract[:300]}")
            elif isinstance(art, dict):
                parts.append(f"{i}. {art.get('title', '?')}")
                if art.get('doi'):
                    parts.append(f"   DOI: {art['doi']}")
            elif isinstance(art, str):
                parts.append(f"{i}. DOI: {art}")
        return "\n".join(parts)

    def _parse_review(
        self,
        raw: dict | list | str,
        original: WrittenArticle,
    ) -> ReviewedDraft:
        """Parsim LLM-otzet v ReviewedDraft."""
        if not isinstance(raw, dict):
            raw = {}

        # Verdict
        verdict_str = raw.get("verdict", "NEEDS_REVISION").upper()
        try:
            verdict = ReviewVerdict[verdict_str]
        except KeyError:
            verdict = ReviewVerdict.NEEDS_REVISION

        # Edits
        edits = []
        for e in raw.get("edits", []):
            if isinstance(e, dict):
                sev_str = e.get("severity", "minor").upper()
                try:
                    severity = Severity[sev_str]
                except KeyError:
                    severity = Severity.MINOR
                edits.append(Edit(
                    location=e.get("location", ""),
                    severity=severity,
                    original=e.get("original", ""),
                    suggested=e.get("suggested", ""),
                    reason=e.get("reason", ""),
                    category=e.get("category", ""),
                ))

        # Fact checks
        fact_checks = []
        for fc in raw.get("fact_checks", []):
            if isinstance(fc, dict):
                fact_checks.append(FactCheck(
                    claim=fc.get("claim", ""),
                    source_doi=fc.get("source_doi", ""),
                    verified=bool(fc.get("verified", False)),
                    actual_text=fc.get("actual_text", ""),
                    verdict=fc.get("verdict", ""),
                ))

        # Issues
        issues = raw.get("issues", [])
        if isinstance(issues, list):
            issues = [str(i) for i in issues]

        # Severity counts
        severity_counts = raw.get("severity_counts", {})
        if not isinstance(severity_counts, dict):
            severity_counts = {}

        # Score
        score = float(raw.get("overall_score", 0.5))

        return ReviewedDraft(
            original_text=original.text,
            revised_text="",  # zapolnyaetsya pri primenenii editov
            edits=edits,
            issues=issues,
            fact_checks=fact_checks,
            severity_counts=severity_counts,
            verdict=verdict,
            overall_score=score,
            reviewer_model=self._cfg.reviewer.model if self._cfg else "unknown",
            summary=raw.get("summary", ""),
        )

    def estimate_cost(self, word_count: int = 3000) -> dict:
        """Otsenit' stoimost' review."""
        input_tokens = 1000 + word_count * 1.5  # ~1.5 tokena na slovo
        output_tokens = 2048
        return {
            "estimated_tokens": int(input_tokens + output_tokens),
            "estimated_input_tokens": int(input_tokens),
            "estimated_output_tokens": output_tokens,
        }

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        """Validirovat'."""
        article = kwargs.get("article")
        if not article:
            return False, "obhyazatel'en parametr 'article' (WrittenArticle)"
        strictness = kwargs.get("strictness", 3)
        if not 1 <= strictness <= 5:
            return False, "strictness dolzhen byt' 1-5"
        return True, ""
