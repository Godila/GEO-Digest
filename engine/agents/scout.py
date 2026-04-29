"""Scout Agent — search, score, and group articles by publication potential.

Pipeline v2 integration:
1. Receives topic from orchestrator
2. Searches articles in storage and/or fresh APIs
3. Scores each article using engine.scoring (6 criteria)
4. Filters by relevance threshold
5. Sends top-N candidates to LLM for classification
6. Returns ScoutResult with ranked groups + score metadata
"""

from __future__ import annotations

import logging

from engine.agents.base import BaseAgent, LLMCallMixin
from engine.agents.tools import AgentTools
from engine.schemas import (
    Article, ArticleGroup, GroupType, Severity,
    DataRequirements, InfrastructureNeeds,
    ScoutResult, AgentResult,
)

logger = logging.getLogger(__name__)

# ── Prompts ────────────────────────────────────────────────

SCOUT_SYSTEM_PROMPT = """Ты — научный редактор гео-экологического дайджеста.
Анализируешь статьи и определяешь их потенциал для написания новых работ.

Классифицируй каждую статью по типу:
- REPLICATION: метод можно воспроизвести на других данных/регионах
- REVIEW: группа статей для обзорной статьи
- DATA_PAPER: датасет как самостоятельный ресурс

Ответ строго в JSON формате:
{
  "articles": [
    {
      "doi": "...",
      "group_type": "REPLICATION|REVIEW|DATA_PAPER",
      "confidence": 0.0-1.0,
      "rationale": "краткое обоснование",
      "tags": ["tag1", "tag2"],
      "data_requirements": {...},
      "infrastructure_needs": {...}
    }
  ]
}"""

SCOUT_CLASSIFY_PROMPT = """Topic: {topic}
Count: {count}

Articles (pre-filtered by relevance scoring):
{articles_text}

Для каждой статьи определи:
1. group_type — какой потенциал видишь
2. confidence — уверенность (0-1)
3. rationale — почему так решил (1-2 предложения)
4. tags — 3-5 ключевых тегов

Для REPLICATION укажи:
- data_requirements: какие данные нужны для воспроизведения
- infrastructure_needs: ПО, железо, вычислительные ресурсы"""


class ScoutAgent(BaseAgent, LLMCallMixin):
    """Finds and groups articles by publication potential.

    v2: Enriches candidates with engine.scoring before LLM classification.
    """

    @property
    def name(self) -> str:
        return "scout"

    def run(
        self,
        topic: str = "",
        max_articles: int = 20,
        mode: str = "storage",
        min_confidence: float = 0.5,
        min_score_5: float = 2.5,
        topic_query_text: str = "",
        **kwargs,
    ) -> AgentResult:
        """Run scouting pipeline.

        Args:
            topic: Search topic
            max_articles: Max articles to return
            mode: "storage", "fresh", or "mixed"
            min_confidence: Min LLM classification confidence (0-1)
            min_score_5: Min scoring engine score (0-5 scale, default 2.5)
            topic_query_text: Query text for relevance scoring boost
        """
        self._log(f"Scout started: topic={topic}, mode={mode}, max={max_articles}, min_score={min_score_5}")

        try:
            # Phase 1: Collect + Score
            articles, scored_count = self._collect_and_score(
                topic, max_articles, mode, min_score_5, topic_query_text
            )
            if not articles:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error="Не найдено статей по запросу (после скоринга)",
                )

            self._log(f"After scoring filter: {len(articles)} (scored: {scored_count})")

            # Phase 2: LLM Classification
            groups = self._classify_articles(topic, articles)

            high_conf = [g for g in groups if g.confidence >= min_confidence]
            high_conf.sort(key=lambda g: g.confidence, reverse=True)
            self._log(f"Groups after confidence filter: {len(high_conf)}")

            result = ScoutResult(
                topic=topic,
                total_found=scored_count,
                after_dedup=len(articles),
                groups=high_conf,
            )
            return AgentResult(
                agent_name=self.name, success=True, data=result,
            )
        except Exception as e:
            self._log(f"Error: {e}")
            return AgentResult(
                agent_name=self.name, success=False, error=str(e),
            )

    def _collect_and_score(
        self,
        topic: str,
        max_articles: int,
        mode: str,
        min_score_5: float,
        topic_query_text: str,
    ) -> tuple[list[Article], int]:
        """Collect articles, apply scoring engine, filter by relevance.

        Returns:
            (filtered_articles, total_scored_count)
        """
        tools = AgentTools(self.storage)
        raw_articles = []
        seen_dois = set()

        # Collect from storage / fresh search
        if mode in ("storage", "mixed"):
            # Split long topic into keywords for substring search
            # storage.search uses exact substring match, so long phrases return 0
            keywords = self._extract_keywords(topic)
            for kw in keywords:
                stored = tools.search(kw, limit=max_articles * 2)
                for a in stored:
                    doi = a.get("doi", "") or a.get("id", "")
                    if doi not in seen_dois:
                        seen_dois.add(doi)
                        raw_articles.append(a)
            self._log(f"From storage: {len(raw_articles)} (via {len(keywords)} keywords)")

        if mode in ("fresh", "mixed") and len(raw_articles) < max_articles * 3:
            remaining = max_articles * 3 - len(raw_articles)
            self._log(f"Fresh search: need {remaining} more")
            try:
                fresh = tools.search_fresh(
                    query=topic,
                    limit=remaining,
                    save_to_storage=True,
                )
                raw_articles.extend(fresh)
                self._log(f"Fresh results: {len(fresh)}")
            except Exception as e:
                self._log(f"Fresh search failed: {e}")

        if not raw_articles:
            return [], 0

        # Score with engine.scoring
        total_scored = len(raw_articles)
        try:
            from engine.scoring import score_articles_batch
            scored_dicts = score_articles_batch(
                articles=[a.to_dict() if hasattr(a, 'to_dict') else dict(a.data) for a in raw_articles],
                topic_query_text=topic_query_text or topic,
                min_score_5=min_score_5,
            )
            # Convert back to Article objects with enriched data
            filtered = [Article(d) for d in scored_dicts[:max_articles]]
            self._log(f"Scoring: {total_scored} → {len(scored_dicts)} passed → {len(filtered)} returned")
            return filtered, total_scored
        except ImportError:
            # Fallback: no scoring module — return raw
            self._log("engine.scoring not available, skipping scoring filter")
            return raw_articles[:max_articles], total_scored
        except Exception as e:
            logger.warning(f"[scout] Scoring failed: {e}, using unfiltered results")
            return raw_articles[:max_articles], total_scored

    # ── Stopwords for keyword extraction ──────────────────────
    _STOPWORDS = frozenset(
        "a an the for and of in on to with from by using based application applications "
        "approach approaches method methods new novel study review analysis overview "
        "towards its their or between through which while during both these those "
        "this that is are was were be been being have has had do does did will would "
        "could should may might shall can".split()
    )

    def _extract_keywords(self, topic: str) -> list[str]:
        """Split a long topic into searchable 1-3 word keywords.

        storage.search_articles uses exact substring match on title/abstract.
        Long phrases like 'deep learning for seismic interpretation' match nothing.
        We split into meaningful bigrams/unigrams, filtering stopwords.
        """
        words = topic.lower().replace(",", " ").replace(";", " ").split()
        # Filter stopwords
        meaningful = [w for w in words if w not in self._STOPWORDS and len(w) > 2]
        if not meaningful:
            return [topic]  # fallback: use full topic

        keywords = []
        # Bigrams first (better precision)
        for i in range(len(meaningful) - 1):
            bigram = f"{meaningful[i]} {meaningful[i + 1]}"
            keywords.append(bigram)
        # Then individual words
        keywords.extend(meaningful)

        # Deduplicate, limit to top 6
        seen = set()
        unique = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique[:6]

    def _collect_candidates(
        self, topic: str, max_articles: int, mode: str
    ) -> list[Article]:
        """Legacy method — collect without scoring. Used by old pipeline v1."""
        tools = AgentTools(self.storage)
        articles = []

        if mode in ("storage", "mixed"):
            stored = tools.search(topic, limit=max_articles)
            articles.extend(stored)
            self._log(f"From storage: {len(stored)}")

        if mode in ("fresh", "mixed") and len(articles) < max_articles:
            remaining = max_articles - len(articles)
            self._log(f"Fresh search: need {remaining} more")
            fresh = tools.search_fresh(
                query=topic,
                limit=remaining,
                save_to_storage=True,
            )
            articles.extend(fresh)
            self._log(f"Fresh results: {len(fresh)}")

        return articles[:max_articles]

    def _classify_articles(
        self, topic: str, articles: list[Article]
    ) -> list[ArticleGroup]:
        """Send articles to LLM for classification."""
        if not articles:
            return []

        parts = []
        for i, art in enumerate(articles, 1):
            part = f"{i}. {art.title}"
            if art.authors:
                part += f" | Authors: {art.authors}"
            if art.abstract:
                abstract = art.abstract[:300] + ("..." if len(art.abstract) > 300 else "")
                part += f"\n   Abstract: {abstract}"
            if art.doi:
                part += f"\n   DOI: {art.doi}"
            # Add score info if available
            scores = art.get("scores", {})
            if scores:
                total_5 = scores.get("total_5", 0)
                art_type = art.get("article_type", "")
                part += f"\n   Score: {total_5:.1f}/5.0 | Type: {art_type}"
            parts.append(part)

        prompt = SCOUT_CLASSIFY_PROMPT.format(
            topic=topic,
            count=len(articles),
            articles_text="\n\n".join(parts),
        )

        raw_response = self.call_llm(
            prompt=prompt,
            system=SCOUT_SYSTEM_PROMPT,
            max_tokens=8192,
            parse_json=True,
        )
        return self._parse_classification(raw_response, articles)

    def _parse_classification(
        self, raw: dict | list | str, original_articles: list[Article]
    ) -> list[ArticleGroup]:
        """Parse LLM response into ArticleGroup list."""
        import json as _json
        import re as _re

        groups = []

        # Try to extract JSON if raw is string
        parsed = raw
        if isinstance(raw, str):
            text = raw.strip()

            # Remove markdown code fences
            for fence_pattern in (r'```json\s*\n?', r'```\s*\n?', r'```\s*$'):
                text = _re.sub(fence_pattern, '', text)
            text = text.strip()

            # Try direct parse first
            try:
                parsed = _json.loads(text)
            except _json.JSONDecodeError:
                # Try to find JSON object with articles key
                match = _re.search(r'\{[^{}]*"articles"\s*:\s*\[[\s\S]*\][^{}]*\}', text)
                if match:
                    try:
                        parsed = _json.loads(match.group())
                    except _json.JSONDecodeError:
                        pass

                # Fallback: find any JSON object
                if isinstance(parsed, str):
                    start = text.rfind('{')
                    if start >= 0:
                        for end in range(len(text), start, -1):
                            try:
                                candidate = text[start:end]
                                parsed = _json.loads(candidate)
                                break
                            except (_json.JSONDecodeError, ValueError):
                                continue

        if not isinstance(parsed, (dict, list)):
            self._log(f"Unclear format: {type(parsed)}")
            return self._fallback_groups(original_articles)

        # Format 1: { "groups": [...] }
        if isinstance(parsed, dict) and "groups" in parsed:
            return self._parse_group_format(parsed.get("groups", []), original_articles)

        # Format 2: { "articles": [...] }
        items = parsed.get("articles", []) if isinstance(parsed, dict) else parsed
        if not items:
            self._log("Empty LLM response, using fallback")
            return self._fallback_groups(original_articles)

        doi_map = {a.doi: a for a in original_articles if a.doi}

        for item in items:
            if not isinstance(item, dict):
                continue

            doi = item.get("doi", "")
            type_str = item.get("group_type", "REVIEW").upper()
            confidence = float(item.get("confidence", 0.3))
            rationale = item.get("rationale", "")
            tags = item.get("tags", [])

            try:
                group_type = GroupType[type_str]
            except KeyError:
                group_type = GroupType.REVIEW

            source_article = doi_map.get(doi)
            groups.append(ArticleGroup(
                group_id=f"group_{len(groups):03d}",
                group_type=group_type,
                articles=[source_article] if source_article else [],
                confidence=confidence,
                rationale=rationale,
                keywords=tags,
            ))

        if not groups:
            return self._fallback_groups(original_articles)
        return groups

    def _parse_group_format(self, groups_data, original_articles):
        """Parse LLM response with { groups: [...] } structure."""
        groups = []
        doi_map = {a.doi: a for a in original_articles if a.doi}

        for gdata in groups_data:
            if not isinstance(gdata, dict):
                continue
            g_articles = []
            for adata in gdata.get("articles", []):
                if isinstance(adata, dict):
                    doi = adata.get("doi", "")
                    art = doi_map.get(doi)
                    if art:
                        g_articles.append(art)
            groups.append(ArticleGroup(
                group_id=gdata.get("group_id", f"group_{len(groups):03d}"),
                group_type=gdata.get("group_type", "review"),
                title_suggestion=gdata.get("title_suggestion", ""),
                confidence=float(gdata.get("confidence", 0.5)),
                articles=g_articles,
                rationale=gdata.get("rationale", ""),
                keywords=gdata.get("keywords", gdata.get("tags", [])),
            ))

        # If LLM returned groups but no articles matched by DOI,
        # distribute original articles across groups
        total_arts = sum(len(g.articles) for g in groups)
        if total_arts == 0 and original_articles:
            self._log(f"LLM returned groups without articles, distributing {len(original_articles)}")
            for i, art in enumerate(original_articles):
                target_group = groups[i % len(groups)] if groups else None
                if target_group:
                    target_group.articles.append(art)

        if not groups:
            return self._fallback_groups(original_articles)
        return groups

    def _fallback_groups(self, original_articles):
        """Create default group when LLM parsing fails."""
        if not original_articles:
            return []
        return [ArticleGroup(
            group_id="group_000",
            group_type=GroupType.REVIEW,
            articles=original_articles[:10],
            confidence=0.3,
            rationale="Default group (LLM did not classify)",
            keywords=["unclassified"],
        )]

    def estimate_cost(self, topic: str, max_articles: int = 20) -> dict:
        """Estimate cost (tokens)."""
        est_input = 500 + (max_articles * 150)
        est_output = 200 + (max_articles * 80)
        return {
            "estimated_tokens": est_input + est_output,
            "estimated_input_tokens": est_input,
            "estimated_output_tokens": est_output,
            "max_articles": max_articles,
        }

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        """Validate input parameters."""
        topic = kwargs.get("topic", "")
        if not topic or len(topic.strip()) < 3:
            return False, "topic должен быть от 3 символов"
        return True, ""
