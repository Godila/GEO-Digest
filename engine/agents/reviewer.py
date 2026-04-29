"""Reviewer Agent v2 — Proactive Reviewer with Rubric Engine + Iterative Revision Loop.

Sprint 7 upgrade:
- Uses SEPARATE LLM provider (OpenRouter / Gemini Flash Lite) — not self.llm
- Integrates article_patterns rubric (format_rubric_prompt, REVIEW_CRITERIA, TONE_RULES)
- Multi-round revision support (round_number, previous_reviews context)
- Extended JSON output schema: round_number, score_by_category, improvement_suggestions
- _build_revision_instructions() for Writer feedback loop

Logika:
1. Poluchaet WrittenArticle ot Writer'a
2. Ispol'zuyet OTDEL'NUYU model' (get_reviewer_llm() → OpenRouter/Gemini)
3. Fact-check: proveryaet utverzhdeniya protiv istochnikov
4. Style review: grammatika, struktura, yasnost' po RUBRIKE
5. Vozvrashchaet ReviewedDraft s verdiktom i spiskom popravok
6. Mozhno zapyuskat' neskol'ko raz (round 1, 2, 3) s istoriey predydushchikh

Verdikty:
- ACCEPT — stat'ya otlichnaya, mozhno publikovat'
- ACCEPT_WITH_MINOR — melkie pravki, ne kritichno
- NEEDS_REVISION — nuzhna pererabotka
- REJECT — ne podkhodit, nuzhno pisat' zanovo
"""

from __future__ import annotations

from engine.agents.base import BaseAgent, LLMCallMixin
from engine.agents.article_patterns import (
    format_rubric_prompt,
    REVIEW_CRITERIA,
    TONE_RULES,
    REVISION_CONFIG,
    get_article_type,
    get_criteria_for_type,
)
from engine.schemas import (
    WrittenArticle, ReviewedDraft, Edit, FactCheck,
    Severity, ReviewVerdict, AgentResult,
)
from engine.llm.config import get_reviewer_llm, REVIEWER_LLM_CONFIG


# ── System Prompt v2 (rubric-enhanced) ─────────────────────────────

REVIEWER_SYSTEM_PROMPT_V2 = """Ty -- strogiy nauchnyy rezent (reviewer) po geologii i ekologii.
Proveryaesh' stat'i na tochnost', kachestvo i sootvetstviye standartam russkikh nauchnykh zhurnalov.

Tvoja zadacha:
1. FACT-CHECK: Proverit' kazhdoe utverzhdeniye (est' li dokazatel'stva?)
2. STYLE CHECK: Grammatika, stilistika, yasnost', struktura
3. STRUCTURE: Pravil'nost' nauchnoy struktury stat'i (IMRaD)
4. CITATIONS: Nalichiye ssylok na istochniki
5. RUBRIC ASSESSMENT: Otsenit' po kategoriyam (structure, content_quality, scientific_rigor, style_and_language, formatting)

{rubric_section}

Otchay v JSON (STRICTLY valid JSON, no markdown fences):
{{
  "verdict": "ACCEPT|ACCEPT_WITH_MINOR|NEEDS_REVISION|REJECT",
  "overall_score": 0.0-1.0,
  "summary": "obshchij otzyv (2-3 predlozheniya)",
  "round_number": N,
  "score_by_category": {{
    "structure": 0.0-1.0,
    "content_quality": 0.0-1.0,
    "scientific_rigor": 0.0-1.0,
    "style_and_language": 0.0-1.0,
    "formatting": 0.0-1.0
  }},
  "improvement_suggestions": ["konkretnaya rekomendatsiya 1", "rekomendatsiya 2"],
  "edits": [
    {{"location": "section/paragraph", "severity": "critical|major|minor",
     "original": "tekst", "suggested": "ispravlennyy tekst",
     "reason": "pochemu", "category": "fact|style|structure|citation"}}
  ],
  "fact_checks": [
    {{"claim": "utverzhdeniye", "source_doi": "doi_istochnika",
     "verified": true/false, "actual_text": "chto v istochnike",
     "verdict": "confirmed|contradicted|not_found"}}
  ],
  "issues": ["spisok problem otdel'no"],
  "severity_counts": {{"critical": N, "major": N, "minor": N}},
  "revision_instructions": "Konkretnye instruktsii dlya Writer'a: chto i kak ispravit'. Format: spisok deystviy s ukazaniem mest."
}}"""


# ── User Prompt Template ────────────────────────────────────────────

REVIEWER_PROMPT_V2 = """PROVERIT' STAT'YU (Round {round_number} of {max_rounds})

Title: {title}
Article Type: {article_type}
Format: {format}
Language: {language}
Word count: {word_count}

=== FULL ARTICLE TEXT ===
{article_text}

=== SOURCE REFERENCES ===
{references}

=== PREVIOUS REVIEW HISTORY ===
{previous_reviews_context}

Strictness level: {strictness}
(1=liberal, 5=extremely strict)

Review this article thoroughly against the rubric criteria.
Check every claim against sources.
Be specific about what needs fixing.

If this is round 2+, focus on whether issues from previous rounds were addressed."""


class ReviewerAgent(BaseAgent, LLMCallMixin):
    """Rezentuet stat'yu s pomoshch'yu drugoy modeli (OpenRouter / Gemini Flash Lite).

    V2 features:
    - Rubric-based evaluation (article_patterns integration)
    - Multi-round revision loop support
    - Separate reviewer LLM (not shared with writer)
    - Category-level scoring
    - Revision instructions generation for Writer
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Reviewer uses its OWN LLM provider — not self.llm (which is writer's model)
        self._reviewer_llm = None

    @property
    def reviewer_llm(self):
        """Lazy-load the dedicated reviewer LLM (OpenRouter/Gemini)."""
        if self._reviewer_llm is None:
            self._reviewer_llm = get_reviewer_llm()
        return self._reviewer_llm

    @property
    def name(self) -> str:
        return "reviewer"

    def run(
        self,
        article: WrittenArticle | dict | None = None,
        source_articles: list | None = None,
        strictness: int = 3,
        round_number: int = 1,
        previous_reviews: list | None = None,
        article_type: str = "",
        **kwargs,
    ) -> AgentResult:
        """
        Zapustit' rezenzirovanie (v2 — s podderzhkoj iteratsij).

        Args:
            article: WrittenArticle ot Writer'a (ili dict s polyami text/title/...)
            source_articles: spisok statey-dlya fact-checka
            strictness: 1-5 (1=legkiy, 5=strogij). Auto-adjusted by round.
            round_number: nomer raunda revizii (1-based). Vliyaet na strictness.
            previous_reviews: istoriya predydushchikh raundov (list[ReviewedDraft])
            article_type: tip stat'i iz get_article_type() ("original_research", "review", etc.)

        Returns:
            AgentResult s ReviewedDraft v .data
        """
        # Normalize article input
        if isinstance(article, dict):
            article = WrittenArticle(
                text=article.get("text", ""),
                title=article.get("title", ""),
                format_=article.get("format", article.get("format_", "markdown")),
                language=article.get("language", "ru"),
                word_count=article.get("word_count", len(article.get("text", "").split())),
                references=article.get("references", []),
            )

        if not article or not getattr(article, 'text', None):
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="Ne peredana stat'ya (WrittenArticle) ili pustoj text",
            )

        # Auto-detect article type if not provided
        if not article_type:
            article_type = get_article_type(article.text or "", article.title or "")

        # Adjust strictness by round (progressively more lenient)
        effective_strictness = self._get_round_strictness(round_number, strictness)

        self._log(
            f"Rezenzirovanie Round {round_number}: "
            f'"{article.title or "?"}" '
            f"type={article_type}, strict={effective_strictness}"
        )

        try:
            # 1. Build system prompt with rubric
            rubric_text = format_rubric_prompt()
            system_prompt = REVIEWER_SYSTEM_PROMPT_V2.format(rubric_section=rubric_text)

            # 2. Build user prompt with context
            refs_text = self._format_references(source_articles or [])
            prev_context = self._format_previous_reviews(previous_reviews or [])

            max_rounds = REVISION_CONFIG.get("max_rounds", 3)
            prompt = REVIEWER_PROMPT_V2.format(
                round_number=round_number,
                max_rounds=max_rounds,
                title=article.title or "",
                article_type=article_type,
                format_=getattr(article, 'format_', 'markdown'),
                format=getattr(article, 'format_', 'markdown'),
                language=getattr(article, 'language', 'ru'),
                word_count=getattr(article, 'word_count', len((article.text or "").split())),
                article_text=(article.text or "")[:25000],  # limit dlya context
                references=refs_text,
                previous_reviews_context=prev_context,
                strictness=effective_strictness,
            )

            # 3. Call REVIEWER model (separate from writer!) — with timeout
            raw_review = self._run_with_timeout(
                self.reviewer_llm.complete_json,
                120,
                prompt=prompt,
                system=system_prompt,
                max_tokens=4096,
                temperature=REVIEWER_LLM_CONFIG["temperature"],
            )

            # 4. Parse into ReviewedDraft (v2 extended)
            reviewed = self._parse_review_v2(raw_review, article, round_number, article_type)

            self._log(
                f"Round {round_number} complete: verdict={reviewed.verdict.value}, "
                f"score={reviewed.overall_score:.2f}, "
                f"issues={reviewed.severity_counts}"
            )

            return AgentResult(
                agent_name=self.name,
                success=True,
                data=reviewed,
            )

        except Exception as e:
            self._log(f"Oshibka review round {round_number}: {e}")
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
            )

    # ── Round-aware strictness ──────────────────────────────────────

    def _get_round_strictness(self, round_number: int, base_strictness: int) -> int:
        """Adjust strictness based on revision round.

        Round 1: strictest (catch everything)
        Round 2: moderate (focus on remaining issues)
        Round 3: liberal (avoid infinite loops)
        """
        round_key = f"round_{round_number}_strictness"
        config_value = REVISION_CONFIG.get(round_key)
        if config_value is not None:
            return config_value
        # Fallback: decrease by 1 each round, min 1
        return max(1, base_strictness - (round_number - 1))

    # ── Previous reviews formatting ─────────────────────────────────

    def _format_previous_reviews(self, previous_reviews: list) -> str:
        """Format previous review rounds as context for the LLM."""
        if not previous_reviews:
            return "(This is the first review round — no previous history)"

        parts = []
        for i, rev in enumerate(previous_reviews, 1):
            if isinstance(rev, ReviewedDraft):
                parts.append(
                    f"--- Round {getattr(rev, 'round_number', i)} ---\n"
                    f"Verdict: {rev.verdict.value}\n"
                    f"Score: {rev.overall_score:.2f}\n"
                    f"Issues ({rev.severity_counts}): {rev.issues[:5]}\n"
                    f"Summary: {rev.summary[:200]}"
                )
            elif isinstance(rev, dict):
                parts.append(
                    f"--- Round {rev.get('round_number', i)} ---\n"
                    f"Verdict: {rev.get('verdict', '?')}\n"
                    f"Score: {rev.get('overall_score', 0)}\n"
                    f"Issues: {str(rev.get('issues', ''))[:200]}"
                )
            else:
                parts.append(f"--- Round {i} ---\n{str(rev)[:300]}")

        return "\n\n".join(parts)

    # ── Reference formatting ────────────────────────────────────────

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
                    abs_text = art.abstract[:300] if hasattr(art.abstract, '__len__') else str(art.abstract)[:300]
                    parts.append(f"   {abs_text}")
            elif isinstance(art, dict):
                parts.append(f"{i}. {art.get('title', '?')}")
                if art.get('doi'):
                    parts.append(f"   DOI: {art['doi']}")
            elif isinstance(art, str):
                parts.append(f"{i}. DOI: {art}")
        return "\n".join(parts)

    # ── V2 Parser (extended schema) ─────────────────────────────────

    def _parse_review_v2(
        self,
        raw: dict | list | str,
        original: WrittenArticle,
        round_number: int,
        article_type: str,
    ) -> ReviewedDraft:
        """Parsim LLM-otvet v ReviewedDraft (v2 s dop polyami)."""
        if not isinstance(raw, dict):
            try:
                import json
                raw = json.loads(raw) if isinstance(raw, str) else {}
            except (json.JSONDecodeError, TypeError):
                raw = {}

        # Verdict
        verdict_str = raw.get("verdict", "NEEDS_REVISION").upper()
        try:
            verdict = ReviewVerdict[verdict_str]
        except KeyError:
            # Auto-map common variants
            if "accept" in verdict_str and "minor" in verdict_str:
                verdict = ReviewVerdict.ACCEPT_WITH_MINOR
            elif "accept" in verdict_str:
                verdict = ReviewVerdict.ACCEPT
            elif "reject" in verdict_str:
                verdict = ReviewVerdict.REJECT
            else:
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

        # Issues — convert to Edit objects (not plain strings!)
        issues = []
        for i in raw.get("issues", []):
            if isinstance(i, dict):
                sev_str = i.get("seriousness", i.get("severity", "minor")).upper()
                try:
                    severity = Severity[sev_str]
                except KeyError:
                    severity = Severity.MINOR
                issues.append(Edit(
                    location=i.get("section", i.get("location", "")),
                    severity=severity,
                    original="",
                    suggested="",
                    reason=str(i.get("description", i.get("desc", "")))[:300],
                    category=i.get("category", "content"),
                ))
            elif isinstance(i, str):
                # Plain string issue — wrap as minor edit
                issues.append(Edit(
                    location="general",
                    severity=Severity.MINOR,
                    original="", suggested="", reason=i[:300],
                    category="general",
                ))
            else:
                issues.append(Edit(
                    location="general", severity=Severity.MINOR,
                    original="", suggested="", reason=str(i)[:300],
                    category="general",
                ))

        # Severity counts
        severity_counts = raw.get("severity_counts", {})
        if not isinstance(severity_counts, dict):
            severity_counts = {}

        # Score — robust parsing with multiple fallback strategies
        raw_score = raw.get("overall_score")
        score = self._parse_score_value(raw_score)

        # Fallback: estimate from severity counts if score looks wrong
        if score <= 0.01 and (edits or fact_checks):
            score = self._estimate_score_from_issues(edits, fact_checks)

        # V2 new fields
        score_by_category = raw.get("score_by_category", {})
        if not isinstance(score_by_category, dict):
            score_by_category = {}

        improvement_suggestions = raw.get("improvement_suggestions", [])
        if not isinstance(improvement_suggestions, list):
            improvement_suggestions = [str(improvement_suggestions)]

        revision_instructions = raw.get("revision_instructions", "")

        return ReviewedDraft(
            original_text=original.text or "",
            revised_text="",  # zapolnyaetsya pri primenenii editov
            edits=edits,
            issues=issues,
            fact_checks=fact_checks,
            severity_counts=severity_counts,
            verdict=verdict,
            overall_score=score,
            reviewer_model=REVIEWER_LLM_CONFIG["model"],
            summary=raw.get("summary", ""),
            # V2 extended fields:
            round_number=round_number,
            score_by_category=score_by_category,
            improvement_suggestions=improvement_suggestions,
            revision_instructions=revision_instructions,
            article_type=article_type,
        )

    # ── Revision Instructions Builder ───────────────────────────────

    def _build_revision_instructions(self, reviewed: ReviewedDraft) -> str:
        """Build structured revision instructions for Writer.

        Returns a JSON string containing a list of edit dicts with keys:
          - section: which part of the article
          - description: what needs fixing
          - severity: critical / major / minor
          - action: what to do (исправить / добавить / удалить / переписать)

        The orchestrator's _parse_revision_instructions() will parse this
        JSON list and pass it directly to Writer.rewrite_article().
        """
        import json as _json

        # Normalize: accept both ReviewedDraft and plain dict
        if isinstance(reviewed, dict):
            edits = reviewed.get('edits', [])
            suggestions = reviewed.get('improvement_suggestions', [])
            fact_checks = reviewed.get('fact_checks', [])
        else:
            edits = getattr(reviewed, 'edits', [])
            suggestions = getattr(reviewed, 'improvement_suggestions', [])
            fact_checks = getattr(reviewed, 'fact_checks', [])

        structured_edits = []

        for e in edits:
            if isinstance(e, dict):
                sev = str(e.get('severity', 'minor')).lower()
                loc = e.get('location', '')
                reason = e.get('reason', e.get('description', ''))
                suggested = e.get('suggested', '')
            else:
                # Edit dataclass
                sev_val = getattr(e, 'severity', Severity.MINOR)
                sev = sev_val.value if hasattr(sev_val, 'value') else str(sev_val).lower()
                loc = getattr(e, 'location', '')
                reason = getattr(e, 'reason', getattr(e, 'description', ''))
                suggested = getattr(e, 'suggested', '')

            # Determine action verb from category / content
            action = "исправить"
            if suggested and len(suggested) > 20:
                action = "переписать"
            if isinstance(e, dict):
                cat = e.get('category', '')
            else:
                cat = getattr(e, 'category', '')
            if cat in ('structure',):
                action = "переструктурировать"
            elif cat in ('citation',):
                action = "добавить ссылку"

            structured_edits.append({
                "section": loc,
                "description": str(reason)[:300],
                "severity": sev,
                "action": action,
            })

        # Add failed fact-checks as critical edits
        for fc in fact_checks:
            if isinstance(fc, dict):
                verified = fc.get('verified', True)
            else:
                verified = getattr(fc, 'verified', True)
            if not verified:
                claim = fc.get('claim', str(fc)[:120]) if isinstance(fc, dict) else getattr(fc, 'claim', str(fc))[:120]
                structured_edits.append({
                    "section": "general",
                    "description": f"Факт-чек: \"{claim}\" — не подтверждено источниками",
                    "severity": "critical",
                    "action": "исправить",
                })

        # Add improvement suggestions as minor edits
        for s in suggestions:
            if isinstance(s, str) and s.strip():
                structured_edits.append({
                    "section": "general",
                    "description": str(s)[:300],
                    "severity": "minor",
                    "action": "улучшить",
                })

        if not structured_edits:
            return "[]"

        return _json.dumps(structured_edits, ensure_ascii=False)

    # ── Score parsing helpers ───────────────────────────────────────

    def _parse_score_value(self, raw_score) -> float:
        """Parse overall_score from LLM output with robust fallbacks.

        Gemini Flash Lite may return: 0, "0", "0.45", None, "N/A", etc.
        """
        if raw_score is None:
            return 0.5  # neutral default
        if isinstance(raw_score, (int, float)):
            s = float(raw_score)
            return max(0.0, min(1.0, s)) if s > 0 else 0.3  # 0 → suspicious, use low default
        if isinstance(raw_score, str):
            cleaned = raw_score.strip().strip('"').strip("'")
            # Try direct parse
            try:
                s = float(cleaned)
                if s >= 0 and s <= 1:
                    return s
                # Score out of 100?
                if 0 < s <= 100:
                    return s / 100.0
            except ValueError:
                pass
            # Textual hints
            lower = cleaned.lower()
            if any(w in lower for w in ("excellent", "perfect", "отлично")):
                return 0.95
            if any(w in lower for w in ("good", "хорошо", "accept")):
                return 0.75
            if any(w in lower for w in ("fair", "средне", "minor")):
                return 0.55
            if any(w in lower for w in ("poor", "плохо", "reject", "needs")):
                return 0.35
            if any(w in lower for w in ("bad", "critical")):
                return 0.15
        return 0.4  # unknown = slightly below average

    def _estimate_score_from_issues(self, edits: list, fact_checks: list) -> float:
        """Estimate score from issue severity when LLM score is unreliable.

        Formula: start at 1.0, deduct per issue type.
        """
        score = 1.0
        critical = sum(1 for e in edits if getattr(e, 'severity', '') == 'CRITICAL')
        major = sum(1 for e in edits if getattr(e, 'severity', '') == 'MAJOR')
        minor = sum(1 for e in edits if getattr(e, 'severity', '') == 'MINOR')
        failed_facts = sum(1 for fc in fact_checks if not getattr(fc, 'verified', True))

        score -= critical * 0.25   # each critical = -25%
        score -= major * 0.12      # each major = -12%
        score -= minor * 0.03      # each minor = -3%
        score -= failed_facts * 0.15  # each failed fact-check = -15%

        return max(0.05, round(score, 2))

    # ── Cost estimation ─────────────────────────────────────────────

    def estimate_cost(self, word_count: int = 3000) -> dict:
        """Otsenit' stoimost' review."""
        input_tokens = 1500 + word_count * 1.5  # ~1.5 tokena na slovo (+ rubric overhead)
        output_tokens = 3000  # Bigger output for v2 schema
        return {
            "estimated_tokens": int(input_tokens + output_tokens),
            "estimated_input_tokens": int(input_tokens),
            "estimated_output_tokens": output_tokens,
        }

    # ── Validation ─────────────────────────────────────────────────

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        """Validirovat'."""
        article = kwargs.get("article")
        if not article:
            return False, "obyazatel'en parametr 'article' (WrittenArticle)"
        strictness = kwargs.get("strictness", 3)
        if not 1 <= strictness <= 5:
            return False, "strictness dolzhen byt' 1-5"
        round_number = kwargs.get("round_number", 1)
        if round_number < 1:
            return False, "round_number dolzhen byt' >= 1"
        return True, ""
