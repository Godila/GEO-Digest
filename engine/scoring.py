"""Scoring Bridge — re-usable article scoring for GEO-Digest agents.

Extracted from scripts/digest.py to be usable by Scout, Editor, and other
engine agents without importing the full digest pipeline.

Scoring criteria (6 dimensions):
  1. Methodological transferability (25%)
  2. Geographic analogy (20%)
  3. Thematic relevance (20%)
  4. Publication potential (20%)
  5. Practical output (10%)
  6. Novelty of approach (5%)

Returns weighted total 0-1 (total_5: 0-5 scale for UI).
"""

from __future__ import annotations

from typing import Optional

# ── Keyword Lists ──────────────────────────────────────────

ANALOGOUS_REGIONS_KEYWORDS = [
    "caucasus", "alpine", "swiss alps", "italian alps", "french alps",
    "iranian plateau", "central asia", "kazakhstan", "kyrgyzstan", "uzbekistan",
    "andes", "chilean", "argentinian", "peruvian",
    "mediterranean", "turkey", "greek", "balkan",
    "carpathian", "romanian", "ukrainian mountain",
    "tianshan", "pamir", "hinda kush", "zagros",
    "mountainous region", "fold-thrust belt", "active orogeny",
]

NOVELTY_KEYWORDS_ML = [
    "transformer", "large language model", "foundation model",
    "diffusion", "graph neural", "reinforcement learning",
]

NOVELTY_KEYWORDS_GEO = [
    "novel dataset", "new dataset", "first-of-its-kind", "first ever",
    "pioneering", "breakthrough", "state-of-the-art", "sota",
    "multimodal fusion", "hybrid approach", "cross-domain",
    "nanosatellite", "cubesat", "drone-based", "uav-based",
    "fiber optic", "das", "distributed acoustic sensing",
    "real-time monitoring", "near-real-time", "operational system",
    "open data", "open access dataset", "publicly available data",
]

REPRODUCIBLE_METHODS = [
    "field measurement", "field survey", "monitoring network",
    "soil sampling", "water sampling", "geochemical analysis",
    "seismic station", "accelerometer array", "gnss", "insar",
    "remote sensing", "satellite", "sentinel", "landsat",
    "machine learning", "deep learning", "neural network",
    "random forest", "gradient boosting", "transformer",
    "statistical model", "bayesian", "monte carlo",
    "numerical modeling", "finite element", "finite difference",
    "gis mapping", "spatial analysis",
]

PRACTICAL_OUTPUTS_STRONG = [
    "code available", "open source", "github", "gitlab",
    "software available", "toolkit", "web application",
    "api available", "rest api", "interactive map",
    "benchmark dataset", "validation dataset", "ground truth",
]

PRACTICAL_OUTPUTS_MODERATE = [
    "dataset", "database", "catalog", "framework",
    "algorithm", "methodology", "model output",
]

SOURCE_WEIGHTS = {
    "semantic_scholar": 1.10,
    "doaj":             1.08,
    "openalex":         1.05,
    "core_ac":          1.05,
    "crossref":         1.00,
    "arxiv":            0.95,
    "europe_pmc":       1.00,
}

DEFAULT_SCORING_WEIGHTS = {
    "methodological_transferability": 0.25,
    "geographic_analogy":            0.25,
    "thematic_relevance":            0.20,
    "publication_potential":         0.15,
    "practical_output":              0.10,
    "novelty_of_approach":           0.05,
}


# ── Helper ─────────────────────────────────────────────────

def _find_matches(keywords: list[str], text: str) -> list[str]:
    """Return keywords found in text."""
    return [kw for kw in keywords if kw in text]


# ── Main Scoring Function ──────────────────────────────────

def score_article(
    article: dict,
    weights: Optional[dict] = None,
    topic_query_text: str = "",
) -> dict:
    """
    Score an article on 6 criteria.

    Args:
        article: Article dict with title, abstract, topics, etc.
        weights: Optional weight overrides. Defaults to DEFAULT_SCORING_WEIGHTS.
        topic_query_text: Optional concatenated query text for relevance boost.

    Returns:
        Article dict with added keys:
          - scores: {transferability, geographic_analogy, thematic_relevance,
                     publication_potential, practical_output, novelty,
                     total, total_5}
          - score_explanations: {criterion: str}
          - article_type: str
    """
    weights = weights or DEFAULT_SCORING_WEIGHTS

    text_parts = [
        article.get("title", ""),
        article.get("abstract", ""),
        " ".join(article.get("topics", [])) if isinstance(article.get("topics"), list) else str(article.get("topics", "")),
        " ".join(article.get("institutions", [])) if isinstance(article.get("institutions"), list) else str(article.get("institutions", "")),
        article.get("journal", ""),
    ]
    all_text = " ".join(tp.lower() for tp in text_parts if tp)
    abstract = article.get("abstract", "").lower()
    title_lower = article.get("title", "").lower()

    explanations = {}

    # 1. Methodological transferability (25%)
    trans_score = 0.0
    method_hits = _find_matches(REPRODUCIBLE_METHODS, all_text)
    trans_score = min(1.0, len(method_hits) / 3.0)
    field_bonus = any(kw in all_text for kw in ["field study", "field campaign", "in situ", "on-site"])
    if field_bonus:
        trans_score = min(1.0, trans_score + 0.2)
    if article.get("source") == "openalex" and not article.get("abstract") and not method_hits:
        geo_topics = [t for t in article.get("topics", []) if any(g in t.lower() for g in ["geolog", "seism", "earth", "hydrolog", "climat", "ecolog", "environment", "natural hazard", "remote sens"])]
        if geo_topics:
            trans_score = 0.4
            method_hits = ["geo-scientific topic (no abstract)"]

    trans_parts = []
    if method_hits:
        trans_parts.append(f"Найдено методов: {', '.join(method_hits[:5])}")
    if field_bonus:
        trans_parts.append("+полевые работы (+0.2)")
    if not method_hits and not field_bonus and trans_score > 0:
        trans_parts.append("Гео-топик без абстракта (базовый балл)")
    explanations["transferability"] = "; ".join(trans_parts) if trans_parts else "Методы не идентифицированы"

    # 2. Geographic analogy (20%)
    geo_score = 0.2
    geo_hits = _find_matches(ANALOGOUS_REGIONS_KEYWORDS, all_text)
    direct_russia = any(kw in all_text for kw in ["russia", "caucasus", "post-soviet"])

    is_regional_study = (
        len(geo_hits) > 0 or direct_russia or
        any(kw in title_lower for kw in ["case study", "pilot", "regional"])
    )
    is_global_resource = (
        any(kw in all_text for kw in ["dataset", "database", "global", "worldwide"]) or
        any(kw in title_lower for kw in ["dataset", "database", "catalog"])
    )

    if is_global_resource and not is_regional_study:
        geo_score = 0.5
        if any(kw in all_text for kw in ["multi-region", "cross-border", "continental"]):
            geo_score = 0.7
        if any(kw in all_text for kw in ["global coverage", "worldwide", "earth observation"]):
            geo_score = 0.85
        geo_explain = f"Глобальный ресурс (применимость {geo_score:.0%})"
    else:
        geo_score = min(1.0, len(geo_hits) / 2.0)
        if direct_russia:
            geo_score = min(1.0, geo_score + 0.3)
        geo_parts = []
        if geo_hits:
            geo_parts.append(f"Регионы: {', '.join(geo_hits[:4])}")
        if direct_russia:
            geo_parts.append("+упоминание России/Кавказа")
        if not geo_hits and not direct_russia:
            region_mentioned = [t for t in article.get("topics", []) if any(g in t.lower() for g in ["china", "asia", "europe", "america"])]
            if region_mentioned:
                geo_parts.append(f"Исследует {region_mentioned[0]} (не аналог Югу России)")
            else:
                geo_parts.append("Регион не указан / не аналогичен")
        geo_explain = "; ".join(geo_parts) if geo_parts else "Нет географической привязки"
    explanations["geographic_analogy"] = geo_explain

    # 3. Thematic relevance (20%)
    theme_score = 0.55

    if topic_query_text:
        query_lower = topic_query_text.lower()
        query_in_title = sum(1 for q in query_lower.split() if len(q) > 3 and q in title_lower)
        query_in_abstract = sum(1 for q in query_lower.split() if len(q) > 3 and q in abstract)
        theme_score += min(0.3, (query_in_title * 0.08) + (query_in_abstract * 0.03))

    citations = article.get("citations", 0)
    if citations >= 50:
        theme_score = min(1.0, theme_score + 0.15)
    elif citations >= 20:
        theme_score = min(1.0, theme_score + 0.10)
    elif citations >= 5:
        theme_score = min(1.0, theme_score + 0.05)

    if article.get("source") != "arxiv":
        theme_score = min(1.0, theme_score + 0.10)
    if article.get("is_oa"):
        theme_score = min(1.0, theme_score + 0.05)

    theme_parts = [f"База: {theme_score:.2f}"]
    if citations > 0:
        theme_parts.append(f"+цитирования ({citations})")
    if article.get("source") != "arxiv":
        theme_parts.append("+рецензируемый источник")
    if article.get("is_oa"):
        theme_parts.append("+открытый доступ")
    explanations["thematic_relevance"] = ", ".join(theme_parts)

    # 4. Publication potential (20%)
    pub_score = 0.3
    pub_factors = []

    if any(kw in title_lower for kw in ["review", "survey", "meta-analysis", "systematic review", "overview"]):
        pub_score = max(pub_score, 0.9)
        pub_factors.append("Обзорная статья")
    if any(kw in all_text for kw in ["new method", "novel approach", "proposed framework", "we propose"]):
        pub_score = max(pub_score, 0.8)
        pub_factors.append("Предложен новый метод")
    if "case study" in all_text:
        pub_score = max(pub_score, 0.65)
        pub_factors.append("Case study (воспроизводимо)")
    if any(kw in all_text for kw in ["results show", "we found", "demonstrate", "validate"]):
        pub_score = max(pub_score, 0.55)
        pub_factors.append("Конкретные результаты")

    if any(kw in title_lower for kw in ["review", "survey", "meta-analysis"]):
        article_type = "review"
    elif any(kw in all_text for kw in ["new method", "novel approach", "proposed framework", "deep learning", "transformer"]):
        article_type = "method_transfer"
    else:
        article_type = "reproduction"

    explanations["publication_potential"] = "; ".join(pub_factors) if pub_factors else "Стандартная публикация"

    # 5. Practical output (10%)
    practical_strong = _find_matches(PRACTICAL_OUTPUTS_STRONG, all_text)
    practical_moderate = _find_matches(PRACTICAL_OUTPUTS_MODERATE, all_text)
    practical_score = min(1.0, len(practical_strong) * 0.35 + len(practical_moderate) * 0.15)

    pract_parts = []
    if practical_strong:
        pract_parts.append(f"Артефакты: {', '.join(practical_strong[:4])}")
    if practical_moderate and not practical_strong:
        pract_parts.append(f"Упоминания: {', '.join(practical_moderate[:4])}")
    explanations["practical_output"] = "; ".join(pract_parts) if pract_parts else "Конкретные артефакты не указаны"

    # 6. Novelty of approach (5%)
    ml_novel = _find_matches(NOVELTY_KEYWORDS_ML, all_text)
    geo_novel = _find_matches(NOVELTY_KEYWORDS_GEO, all_text)
    novelty_score = min(1.0, len(ml_novel) * 0.33 + len(geo_novel) * 0.20)

    novel_parts = []
    if ml_novel:
        novel_parts.append(f"ML-новизна: {', '.join(ml_novel[:3])}")
    if geo_novel:
        novel_parts.append(f"Гео-новизна: {', '.join(geo_novel[:3])}")
    explanations["novelty"] = "; ".join(novel_parts) if novel_parts else "Стандартные методы"

    # ── Weighted total ──
    w_trans = weights.get("methodological_transferability", 0.25)
    w_geo = weights.get("geographic_analogy", 0.20)
    w_theme = weights.get("thematic_relevance", 0.20)
    w_pub = weights.get("publication_potential", 0.20)
    w_pract = weights.get("practical_output", 0.10)
    w_novel = weights.get("novelty_of_approach", 0.05)

    total = (trans_score * w_trans + geo_score * w_geo + theme_score * w_theme +
             pub_score * w_pub + practical_score * w_pract + novelty_score * w_novel)

    # Source reliability multiplier
    src = article.get("source", "")
    src_weight = SOURCE_WEIGHTS.get(src, 1.0)

    # Metadata-poverty compensation
    poor_metadata = (
        (src in ("doaj", "core_ac", "arxiv")) and
        article.get("citations", 0) == 0 and
        not article.get("institutions")
    )
    if poor_metadata:
        total += 0.08

    total_adj = min(1.0, total * src_weight)

    article["scores"] = {
        "transferability": round(trans_score, 3),
        "geographic_analogy": round(geo_score, 3),
        "thematic_relevance": round(theme_score, 3),
        "publication_potential": round(pub_score, 3),
        "practical_output": round(practical_score, 3),
        "novelty": round(novelty_score, 3),
        "total": round(total_adj, 3),
        "total_5": round(total_adj * 5, 2),
    }
    article["score_explanations"] = explanations
    article["article_type"] = article_type

    return article


def score_articles_batch(
    articles: list[dict],
    weights: Optional[dict] = None,
    topic_query_text: str = "",
    min_score_5: float = 0.0,
) -> list[dict]:
    """
    Score a batch of articles and optionally filter by minimum score.

    Args:
        articles: List of article dicts.
        weights: Optional weight overrides.
        topic_query_text: For thematic relevance boost.
        min_score_5: Minimum total_5 score to keep (0-5 scale). Default: keep all.

    Returns:
        List of scored article dicts, sorted by total_5 descending.
    """
    scored = []
    for art in articles:
        try:
            scored_art = score_article(dict(art), weights=weights, topic_query_text=topic_query_text)
            total_5 = scored_art.get("scores", {}).get("total_5", 0)
            if total_5 >= min_score_5:
                scored.append(scored_art)
        except Exception:
            continue

    scored.sort(key=lambda a: a.get("scores", {}).get("total_5", 0), reverse=True)
    return scored


def extract_topic_query_text(config: dict) -> str:
    """Extract concatenated query text from config for relevance scoring."""
    topics_cfg = config.get("topics", {})
    if isinstance(topics_cfg, dict):
        query_parts = []
        for tk, tv in topics_cfg.items():
            if isinstance(tv, dict):
                query_parts.extend(tv.get("queries", []))
            elif isinstance(tv, list):
                query_parts.extend(tv)
        return " ".join(query_parts).lower()
    return str(topics_cfg).lower() if topics_cfg else ""
