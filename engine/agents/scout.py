"""Scout Agent -- poisk i gruppirovka statey po potentsialu. (Sprint 2)

Logika:
1. Poluchaet topic ot polzovatelya / orchestrator'a
2. Ishchet stat'i v storage ili zapuskaet svezhiy poisk
3. Otpravlyaet top-N kandidatov v LLM dlya klassifikatsii
4. Gruppiruet po tipu potentsiala: REPLICATION / REVIEW / DATA_PAPER
5. Vozvrashchaet ScoutResult s ranzhirovannymi gruppami
"""

from __future__ import annotations

from engine.agents.base import BaseAgent, LLMCallMixin
from engine.agents.tools import AgentTools
from engine.schemas import (
    Article, ArticleGroup, GroupType, Severity,
    DataRequirements, InfrastructureNeeds,
    ScoutResult, AgentResult,
)

# ── Prompty ────────────────────────────────────────────────

SCOUT_SYSTEM_PROMPT = """Ty -- nauchnyy redaktor geo-ekologicheskogo digest'a.
Analiziruesh stat'i i opredelyaesh ikh potentsial dlya pisaniya novykh rabot.

Klassifitsiruy kazhduyu stat'yu po tipu:
- REPLICATION: metod mozhno vosproizvesti na drugikh dannykh/regionakh
- REVIEW: gruppa statey dlya obzornoy stat'i
- DATA_PAPER: dataset kak samostoyatel'nyy resurs

Otchay strogo v JSON format:
{
  "articles": [
    {
      "doi": "...",
      "group_type": "REPLICATION|REVIEW|DATA_PAPER",
      "confidence": 0.0-1.0,
      "rationale": "kratkoe obosnovanie",
      "tags": ["tag1", "tag2"],
      "data_requirements": {...},
      "infrastructure_needs": {...}
    }
  ]
}"""

SCOUT_CLASSIFY_PROMPT = """Topic: {topic}
Count: {count}

Articles:
{articles_text}

Dlya kazhdoy stat'i opredi:
1. group_type -- kakoy potentsial vidish'
2. confidence -- uverennost' (0-1)
3. rationale -- pochemu tak reshil (1-2 predlozheniya)
4. tags -- 3-5 klyuchevykh tegov

Dlya REPLICATION ukazhi:
- data_requirements: kakie dannye nuzhny dlya vosproizvedeniya
- infrastructure_needs: PO, zhelezo, vychislitel'nye resursy"""


class ScoutAgent(BaseAgent, LLMCallMixin):
    """Nakhodit i gruppiruet stat'i po potentsialu publikatsii."""

    @property
    def name(self) -> str:
        return "scout"

    def run(
        self,
        topic: str = "",
        max_articles: int = 20,
        mode: str = "storage",
        min_confidence: float = 0.5,
        **kwargs,
    ) -> AgentResult:
        """Zapustit' skauting."""
        self._log(f"Skauting nachat: topic={topic}, mode={mode}, max={max_articles}")

        try:
            articles = self._collect_candidates(topic, max_articles, mode)
            if not articles:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error="Ne naydeno statey po zaprosu",
                )

            self._log(f"Kandidatov: {len(articles)}")
            groups = self._classify_articles(topic, articles)

            high_conf = [g for g in groups if g.confidence >= min_confidence]
            high_conf.sort(key=lambda g: g.confidence, reverse=True)
            self._log(f"Grup posle fil'tra: {len(high_conf)}")

            result = ScoutResult(
                query=topic,
                total_found=len(articles),
                analyzed=len(articles),
                groups=high_conf,
                group_count=len(high_conf),
            )
            return AgentResult(
                agent_name=self.name, success=True, data=result,
            )
        except Exception as e:
            self._log(f"Oshibka: {e}")
            return AgentResult(
                agent_name=self.name, success=False, error=str(e),
            )

    def _collect_candidates(
        self, topic: str, max_articles: int, mode: str
    ) -> list[Article]:
        """Sobrat' stat'i iz storage i/ili svezhego poiska."""
        tools = AgentTools(self.storage)
        articles = []

        if mode in ("storage", "mixed"):
            stored = tools.search(topic, limit=max_articles)
            articles.extend(stored)
            self._log(f"Iz storage: {len(stored)}")

        if mode in ("fresh", "mixed") and len(articles) < max_articles:
            remaining = max_articles - len(articles)
            self._log(f"Svezhiy poisk: nuzhno eshche {remaining} (TODO)")

        return articles[:max_articles]

    def _classify_articles(
        self, topic: str, articles: list[Article]
    ) -> list[ArticleGroup]:
        """Otpravit' stat'i v LLM dlya klassifikatsii."""
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
            parts.append(part)

        prompt = SCOUT_CLASSIFY_PROMPT.format(
            topic=topic,
            count=len(articles),
            articles_text="\n\n".join(parts),
        )

        raw_response = self.call_llm(
            prompt=prompt,
            system=SCOUT_SYSTEM_PROMPT,
            max_tokens=4096,
            parse_json=True,
        )
        return self._parse_classification(raw_response, articles)

    def _parse_classification(
        self, raw: dict | list | str, original_articles: list[Article]
    ) -> list[ArticleGroup]:
        """Parsim LLM-otvet v spisok ArticleGroup."""
        groups = []

        if isinstance(raw, dict):
            items = raw.get("articles", [])
        elif isinstance(raw, list):
            items = raw
        else:
            self._log(f"Neponyatnyy format: {type(raw)}")
            return groups

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

            data_req = None
            infra = None
            if group_type == GroupType.REPLICATION and "data_requirements" in item:
                dr = item["data_requirements"]
                data_req = DataRequirements(
                    data_types=dr.get("data_types", []),
                    spatial_coverage=dr.get("spatial_coverage", ""),
                    temporal_coverage=dr.get("temporal_coverage", ""),
                    sample_size=dr.get("sample_size"),
                    format_notes=dr.get("format_notes", ""),
                )
                if "infrastructure_needs" in item:
                    inf = item["infrastructure_needs"]
                    infra = InfrastructureNeeds(
                        software=inf.get("software", []),
                        hardware=inf.get("hardware", ""),
                        compute_hours=inf.get("compute_hours"),
                        expertise=inf.get("expertise", []),
                        cost_estimate=inf.get("cost_estimate", ""),
                    )

            source_article = doi_map.get(doi)
            groups.append(ArticleGroup(
                group_id=f"group_{len(groups):03d}",
                group_type=group_type,
                articles=[source_article] if source_article else [],
                confidence=confidence,
                rationale=rationale,
                tags=tags,
                data_requirements=data_req,
                infrastructure_needs=infra,
            ))

        if not groups and original_articles:
            groups.append(ArticleGroup(
                group_id="group_000",
                group_type=GroupType.REVIEW,
                articles=original_articles[:10],
                confidence=0.3,
                rationale="Default group (LLM did not classify)",
                tags=["unclassified"],
            ))
        return groups

    def estimate_cost(self, topic: str, max_articles: int = 20) -> dict:
        """Otsenit' stoimost' (tokeny)."""
        est_input = 500 + (max_articles * 150)
        est_output = 200 + (max_articles * 80)
        return {
            "estimated_tokens": est_input + est_output,
            "estimated_input_tokens": est_input,
            "estimated_output_tokens": est_output,
            "max_articles": max_articles,
        }

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        """Validirovat' vhodnye parametry."""
        topic = kwargs.get("topic", "")
        if not topic or len(topic.strip()) < 3:
            return False, "topic dolzhen byt' ot 3 simvolov"
        return True, ""
