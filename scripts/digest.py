"""
Geo-Ecology Digest Agent — Main Script
Searches world academic papers relevant for South Russia geology/ecology.
Scores, deduplicates, formats cards, updates knowledge graph.
"""

import json
import os
import sys
import time
import hashlib
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── LLM client (MiniMax M2.7) ───────────────────────────────────
import sys, os as _os
_scripts_dir = _os.path.dirname(_os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from llm_client import call_llm as call_llm

# ── Paths ───────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GEO_DATA_DIR", BASE / "data"))
ARTICLES_DB = DATA_DIR / "articles.jsonl"
SEEN_DOIS = DATA_DIR / "seen_dois.txt"
ARTICLES_DIR = DATA_DIR / "articles"
GRAPHIFY_OUT = BASE / "graphify-out"
CONFIG_PATH = BASE / "config.yaml"

ARTICLES_DIR.mkdir(exist_ok=True)
GRAPHIFY_OUT.mkdir(exist_ok=True)


# ── Config loader (YAML via pyyaml) ────────────────────────────
def load_config():
    """Load config.yaml into a dict using pyyaml."""
    import yaml
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


# ── Deduplication ──────────────────────────────────────────────
def load_seen_dois() -> set:
    """Load set of already-shown DOIs."""
    if SEEN_DOIS.exists():
        return {line.strip() for line in SEEN_DOIS.read_text().splitlines() if line.strip()}
    return set()


def add_seen_dois(dois: list[str]):
    """Append new DOIs to seen list."""
    with open(SEEN_DOIS, "a") as f:
        for doi in dois:
            f.write(doi + "\n")


def title_hash(title: str, year: str) -> str:
    """Fallback ID for articles without DOI."""
    return hashlib.md5(f"{title}|{year}".encode()).hexdigest()[:12]


# ── Articles DB (JSONL) ────────────────────────────────────────
def save_article(record: dict):
    """Append article record to JSONL."""
    record["_saved_at"] = datetime.now(timezone.utc).isoformat()
    with open(ARTICLES_DB, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_articles(tag: str = None, limit: int = 50) -> list[dict]:
    """Load articles from JSONL. Optional filter by tag."""
    results = []
    if ARTICLES_DB.exists():
        with open(ARTICLES_DB, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if tag and tag not in rec.get("tags", []):
                    continue
                results.append(rec)
                if len(results) >= limit:
                    break
    return results


# ── OpenAlex Search ────────────────────────────────────────────
def reconstruct_abstract(inv_index: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index."""
    if not inv_index:
        return ""
    # Build position-indexed word list
    positions = {}
    for word, indexes in inv_index.items():
        for idx in indexes:
            positions[idx] = word
    if not positions:
        return ""
    # Sort by position and join
    return " ".join(positions[k] for k in sorted(positions.keys()))

# ── Unpaywall Lookup ──────────────────────────────────────────
def lookup_unpaywall(doi: str, email: str = None) -> dict:
    """Lookup OA status and PDF URL for a DOI via Unpaywall.
    Free, needs email in User-Agent. 100K/day rate limit."""
    if not doi:
        return {}
    
    # Get email from param -> env -> config -> fallback
    if not email:
        email = os.environ.get("UNPAYWALL_EMAIL", "")
    if not email:
        env_file = BASE / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line.startswith("UNPAYWALL_EMAIL=") and not line.startswith("#"):
                    email = line.split("=", 1)[1].strip().strip("'\"")
                    break
    if not email:
        email = "gkaitmazov@proton.me"  # default
    
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={urllib.parse.quote(email)}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": f"GeoDigest/1.0 (mailto:{email})"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            data = json.loads(raw)
            # Guard: Unpaywall may return null for oa_locations or entire object
            if not data or not isinstance(data, dict):
                return {}
            best_loc = data.get("best_oa_location") or {}
            oa_locations = data.get("oa_locations") or []
            return {
                "is_oa": data.get("is_oa", False),
                "oa_status": data.get("oa_status", "") or "",
                "pdf_url": best_loc.get("url_for_pdf", "") if best_loc else "",
                "html_url": best_loc.get("url_for_html", "") if best_loc else "",
                "license": best_loc.get("license", "") if best_loc else "",
                "host_type": best_loc.get("host_type", "") if best_loc else "",
                "oa_locations_count": len(oa_locations) if isinstance(oa_locations, list) else 0,
            }
    except Exception as e:
        print(f"  [Unpaywall error for {doi[:20]}...] {e}", file=sys.stderr)
        return {}


# ── Relevance Gate ───────────────────────────────────────────────
GEO_KEYWORDS = [
    # Seismology
    "seismic", "earthquake", "aftershock", "fault", "tectonic", "magnitude",
    "hypocenter", "epicenter", "p-wave", "s-wave", "ground motion", "seismogram",
    "radon", "electromagnetic precursor",
    # Geophysics
    "geophysical", "gravity", "magnetic", "resistivity", "velocity model",
    "seismic reflection", "seismic refraction", "well logging", "borehole",
    "mud volcano", "geothermal",
    # Landslide/hazards
    "landslide", "slope stability", "debris flow", "rockfall", "mass movement",
    "hillslope", "soil slip", "creep", "liquefaction",
    # Geoecology / mining
    "heavy metal", "contamination", "soil pollution", "mine tailing",
    "remediation", "phytoremediation", "bioavailability", "geochemical",
    "mining impact", "environmental assessment", "ecological risk",
    # Climate / remote sensing
    "climate trend", "temperature trend", "precipitation change", "aridification",
    "remote sensing", "satellite", "sentinel", "landsat", "inSAR", "insar",
    "ndvi", "land cover", "drought monitoring", "atmospheric monitoring",
    "aerosol", "no2", "pm2.5",
    # Oil & gas
    "hydrocarbon", "petroleum", "reservoir", "permeability", "porosity",
    "well test", "basin modeling", "sedimentary basin", "fluid dynamics",
    "formation damage", "near-wellbore",
    # Mineralogy
    "mineral", "granite", "zircon", "titanite", "apatite", "geochronolog",
    "u-pb dating", "magmatic belt", "provenance",
    # General geo
    "geological", "geomorphology", "hydrology", "watershed", "river basin",
    "mountain region", "orogeny", "crustal", "subsurface", "stratigraph",
    "quaternary", "holocene", "pleistocene",
]


def passes_relevance_gate(article: dict) -> bool:
    """Filter out papers that are clearly off-topic."""
    text = (
        article.get("title", "") + " " +
        article.get("abstract", "") + " " +
        article.get("journal", "") + " " +
        " ".join(article.get("topics", []))
    ).lower()

    hits = sum(1 for kw in GEO_KEYWORDS if kw in text)

    # Need at least 2 geo keywords OR 1 very specific one
    strong_hits = [kw for kw in GEO_KEYWORDS if kw in text and len(kw) > 10]
    return hits >= 2 or len(strong_hits) >= 1


# ── Scoring Engine ─────────────────────────────────────────────
ANALOGOUS_REGIONS_KEYWORDS = [
    "caucasus", "alpine", "swiss alps", "italian alps", "french alps",
    "iranian plateau", "central asia", "kazakhstan", "kyrgyzstan", "uzbekistan",
    "andes", "chilean", "argentinian", "peruvian",
    "mediterranean", "turkey", "greek", "balkan",
    "carpathian", "romanian", "ukrainian mountain",
    "tianshan", "pamir", "hindu kush", "zagros",
    "mountainous region", "fold-thrust belt", "active orogeny",
]

# [FIX E] Расширенные индикаторы новизны: ML + гео-научные
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

# [FIX F] Уточнённые практические артефакты (без слишком общих слов)
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


def score_article(article: dict, config: dict) -> dict:
    """
    Score an article on 6 criteria.
    Returns scores dict + total + explanations (for UI transparency).
    """
    weights = config.get("scoring", {})
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

    # ── Helper: collect matched keywords for explanation ──
    def find_matches(keywords, text):
        return [kw for kw in keywords if kw in text]

    # Store explanations per criterion
    explanations = {}

    # 1. Methodological transferability (25%)
    trans_score = 0.0
    method_hits = find_matches(REPRODUCIBLE_METHODS, all_text)
    trans_score = min(1.0, len(method_hits) / 3.0)
    # Bonus for field methods
    field_bonus = any(kw in all_text for kw in ["field study", "field campaign", "in situ", "on-site"])
    if field_bonus:
        trans_score = min(1.0, trans_score + 0.2)
    # OpenAlex fallback
    if article.get("source") == "openalex" and not article.get("abstract") and not method_hits:
        geo_topics = [t for t in article.get("topics", []) if any(g in t.lower() for g in ["geolog", "seism", "earth", "hydrolog", "climat", "ecolog", "environment", "natural hazard", "remote sens"])]
        if geo_topics:
            trans_score = 0.4
            method_hits = ["geo-scientific topic (no abstract)"]

    # Build explanation
    trans_parts = []
    if method_hits:
        trans_parts.append(f"Найдено методов: {', '.join(method_hits[:5])}")
    if field_bonus:
        trans_parts.append("+полевые работы (+0.2)")
    if not method_hits and not field_bonus and trans_score > 0:
        trans_parts.append(f"Гео-топик без абстракта (базовый балл)")
    explanations["transferability"] = "; ".join(trans_parts) if trans_parts else "Методы не идентифицированы"

    # 2. Geographic analogy (20%) — [FIX C] улучшенная логика
    geo_score = 0.0
    geo_hits = find_matches(ANALOGOUS_REGIONS_KEYWORDS, all_text)
    direct_russia = any(kw in all_text for kw in ["russia", "caucasus", "post-soviet"])

    # [FIX C] Определяем тип статьи: региональное исследование или глобальный метод/датасет
    is_regional_study = (
        len(geo_hits) > 0 or direct_russia or
        any(kw in title_lower for kw in ["case study", "pilot", "regional"])
    )
    is_global_resource = (
        any(kw in all_text for kw in ["dataset", "database", "global", "worldwide"]) or
        any(kw in title_lower for kw in ["dataset", "database", "catalog"])
    )

    if is_global_resource and not is_regional_study:
        # Глобальный ресурс → заменяем geo_analogy на global_applicability
        geo_score = 0.5  # базовый: применимо глобально
        if any(kw in all_text for kw in ["multi-region", "cross-border", "continental"]):
            geo_score = 0.7
        if any(kw in all_text for kw in ["global coverage", "worldwide", "earth observation"]):
            geo_score = 0.85
        geo_explain = f"Глобальный ресурс (применимость {geo_score:.0%})"
    else:
        # Региональное исследование — стандартная логика
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

    # 3. Thematic relevance (20%) — [FIX D] расширенный диапазон
    # Базовый score зависит от качества совпадения топика
    theme_score = 0.4  # [FIX D] было 0.7 → теперь 0.4

    # Бонус за точное совпадение ключевых слов из запроса
    topic_query_text = " ".join(config.get("topics", [])).lower() if isinstance(config.get("topics", list), list) else ""
    if topic_query_text:
        query_in_title = sum(1 for q in topic_query_text.split() if len(q) > 3 and q in title_lower)
        query_in_abstract = sum(1 for q in topic_query_text.split() if len(q) > 3 and q in abstract)
        theme_score += min(0.3, (query_in_title * 0.08) + (query_in_abstract * 0.03))

    # Citation bonus
    citations = article.get("citations", 0)
    if citations >= 50:
        theme_score = min(1.0, theme_score + 0.15)
    elif citations >= 20:
        theme_score = min(1.0, theme_score + 0.10)
    elif citations >= 5:
        theme_score = min(1.0, theme_score + 0.05)

    # Peer-reviewed bonus
    if article.get("source") != "arxiv":
        theme_score = min(1.0, theme_score + 0.10)

    # OA bonus
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

    # 4. Publication potential (20%) — [FIX F] накопительная логика вместо каскада
    pub_score = 0.3  # базовый
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

    # Determine article type
    if any(kw in title_lower for kw in ["review", "survey", "meta-analysis"]):
        article_type = "review"
    elif any(kw in all_text for kw in ["new method", "novel approach", "proposed framework", "deep learning", "transformer"]):
        article_type = "method_transfer"
    else:
        article_type = "reproduction"

    explanations["publication_potential"] = "; ".join(pub_factors) if pub_factors else "Стандартная публикация"

    # 5. Practical output (10%) — [FIX F] двухуровневые ключевые слова
    practical_strong = find_matches(PRACTICAL_OUTPUTS_STRONG, all_text)
    practical_moderate = find_matches(PRACTICAL_OUTPUTS_MODERATE, all_text)

    strong_count = len(practical_strong)
    moderate_count = len(practical_moderate)
    practical_score = min(1.0, strong_count * 0.35 + moderate_count * 0.15)

    pract_parts = []
    if practical_strong:
        pract_parts.append(f"Артефакты: {', '.join(practical_strong[:4])}")
    if practical_moderate and not practical_strong:
        pract_parts.append(f"Упоминания: {', '.join(practical_moderate[:4])}")
    explanations["practical_output"] = "; ".join(pract_parts) if pract_parts else "Конкретные артефакты не указаны"

    # 6. Novelty of approach (5%) — [FIX E] ML + гео-научные индикаторы
    ml_novel = find_matches(NOVELTY_KEYWORDS_ML, all_text)
    geo_novel = find_matches(NOVELTY_KEYWORDS_GEO, all_text)

    novelty_score = min(1.0, len(ml_novel) * 0.33 + len(geo_novel) * 0.20)

    novel_parts = []
    if ml_novel:
        novel_parts.append(f"ML-новизна: {', '.join(ml_novel[:3])}")
    if geo_novel:
        novel_parts.append(f"Гео-новизна: {', '.join(geo_novel[:3])}")
    explanations["novelty"] = "; ".join(novel_parts) if novel_parts else "Стандартные методы"

    # Weighted total — [FIX A] правильный подсчёт, без дублирования в JS
    w_trans = weights.get("methodological_transferability", 0.25)
    w_geo = weights.get("geographic_analogy", 0.20)
    w_theme = weights.get("thematic_relevance", 0.20)
    w_pub = weights.get("publication_potential", 0.20)
    w_pract = weights.get("practical_output", 0.10)
    w_novel = weights.get("novelty_of_approach", 0.05)

    total = (trans_score * w_trans + geo_score * w_geo + theme_score * w_theme +
             pub_score * w_pub + practical_score * w_pract + novelty_score * w_novel)

    # Source reliability multiplier (from source weight config)
    src = article.get("source", "")
    SOURCE_WEIGHTS = {
        "semantic_scholar": 1.10,
        "doaj":             1.08,
        "openalex":         1.05,
        "core_ac":          1.05,
        "crossref":         1.00,
        "arxiv":            0.95,
        "europe_pmc":       1.00,
    }
    src_weight = SOURCE_WEIGHTS.get(src, 1.0)
    total_adj = min(1.0, total * src_weight)  # cap at 1.0

    article["scores"] = {
        "transferability": round(trans_score, 3),
        "geographic_analogy": round(geo_score, 3),
        "thematic_relevance": round(theme_score, 3),
        "publication_potential": round(pub_score, 3),
        "practical_output": round(practical_score, 3),
        "novelty": round(novelty_score, 3),
        "total": round(total_adj, 3),  # взвешенная сумма 0-1 (с source weight)
        "total_5": round(total_adj * 5, 2),  # шкала 0-5 для UI [FIX A]
    }
    # [FIX B] Обоснования для каждого критерия
    article["score_explanations"] = explanations
    article["article_type"] = article_type

    return article


# ── LLM Enrichment Module ─────────────────────────────────────
def find_similar_articles(article: dict, db_articles: list[dict], top_n: int = 2) -> list[dict]:
    """Find similar articles from database based on topic overlap."""
    art_topics = set(article.get("topics", []))
    art_topic_key = article.get("_topic_key", "")
    art_words = set(article.get("title", "").lower().split())
    
    scored = []
    for db_art in db_articles:
        doi_db = db_art.get("doi", "")
        doi_art = article.get("doi", "")
        if doi_db and doi_db == doi_art:
            continue  # skip self
        
        score = 0.0
        # Same topic bonus
        if db_art.get("_topic_key") == art_topic_key:
            score += 0.5
        # Topic overlap
        db_topics = set(db_art.get("topics", []))
        if art_topics and db_topics:
            overlap = len(art_topics & db_topics) / max(len(art_topics | db_topics), 1)
            score += overlap * 0.3
        # Title word overlap
        db_words = set(db_art.get("title", "").lower().split())
        word_overlap = len(art_words & db_words) / max(len(art_words | db_words), 1)
        score += word_overlap * 0.2
        
        if score > 0.1:
            scored.append((score, db_art))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_n]]


import re


# ── Regex patterns for LLM response parsing ──────────────────
# Tolerant to variations: "===ЗАГОЛОВОК_RU===", "=== ЗАГОЛОВОК_RU ===", etc.
# Each pattern uses non-greedy capture + lookahead to stop at next === marker
_SECTION_PATTERNS = {
    "title_ru":      re.compile(r"===\s*ЗАГОЛОВОК_RU\s*===?\s*\n?(.*?)(?=\s*(?:===|$))", re.DOTALL),
    "abstract_ru":   re.compile(r"===\s*АННОТАЦИЯ_RU\s*===?\s*\n?(.*?)(?=\s*(?:===|$))", re.DOTALL),
    "topics_ru":     re.compile(r"===\s*ТЕМЫ_RU\s*===?\s*\n?(.*?)(?=\s*(?:===|📝|🔬|🇷🇺|💡)|$)", re.DOTALL),
    "overview":      re.compile(r"===\s*ОБЗОР\s*===?\s*\n?(.*)", re.DOTALL),  # last section — greedy OK
}

# Validation thresholds
_VALIDATE = {
    "title_ru":      {"min_len": 10,  "max_len": 500, "label": "заголовок"},
    "abstract_ru":   {"min_len": 30,  "max_len": 2000, "label": "аннотация"},
    "overview":      {"min_len": 50,  "max_len": 3000, "label": "обзор"},
}


def _extract_fallback_title(response: str) -> str:
    """
    Try to extract a Russian title from unstructured LLM response.
    
    Patterns tried (in order):
    1. First line that looks like a title (Russian, 10-200 chars, no emoji prefix)
    2. Text between start and first === or 📝 marker
    """
    if not response:
        return ""
    
    lines = response.strip().split("\n")
    
    # Pattern 1: first substantive Russian line (before any section marker)
    for line in lines:
        stripped = line.strip()
        # Skip empty, markers, emoji-prefixed, too short/long
        if not stripped:
            continue
        if stripped.startswith("===") or stripped.startswith("📝") or \
           stripped.startswith("🔬") or stripped.startswith("🇷🇺") or \
           stripped.startswith("💡") or stripped.startswith("-"):
            continue
        if len(stripped) < 10 or len(stripped) > 300:
            continue
        # Must contain Cyrillic (Russian text)
        if any('\u0400' <= c <= '\u04FF' for c in stripped):
            return stripped
    
    return ""


def _translate_title_fallback(title_en: str) -> str:
    """
    Fallback: call MiniMax to translate just the article title.
    Used when main enrichment didn't produce title_ru.
    """
    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key or not title_en:
        return ""
    
    prompt = f"""Переведи название научной статьи на русский язык (1 предложение, научно):

{title_en}

Ответ только переводом, без пояснений:"""
    
    try:
        data = json.dumps({
            "model": "MiniMax-Text-01",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.3,
        }).encode()
        
        req = urllib.request.Request(
            "https://api.minimax.chat/v1/text/chatcompletion_v2",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            translated = (result.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
            
            if translated and len(translated) >= 10:
                print(f"  [LLM] ✅ Fallback title: {translated[:60]}...", file=sys.stderr)
                return translated
    except Exception as e:
        print(f"  [LLM] ⚠ Fallback title translation failed: {e}", file=sys.stderr)
    
    return ""


def _parse_llm_response(response: str) -> dict:
    """
    Parse structured LLM response into sections using regex.
    
    Returns dict with keys: title_ru, abstract_ru, topics_ru, overview
    Each value is a cleaned string (or list for topics_ru).
    Empty string if section not found.
    """
    result = {"title_ru": "", "abstract_ru": "", "topics_ru": [], "overview": ""}
    
    # Extract raw text for each section
    raw = {}
    for key, pat in _SECTION_PATTERNS.items():
        m = pat.search(response)
        if m:
            raw[key] = m.group(1).strip()
        else:
            raw[key] = ""
    
    result["title_ru"] = raw.get("title_ru", "")
    result["abstract_ru"] = raw.get("abstract_ru", "")
    result["overview"] = raw.get("overview", "")

    # Fallback: if no structured title, try to extract from raw response
    # Some LLM responses skip ===ЗАГОЛОВОК_RU=== but still have Russian text
    if not result["title_ru"] and response:
        result["title_ru"] = _extract_fallback_title(response)

    # Parse topics: split by comma or newline/bullet
    topics_raw = raw.get("topics_ru", "")
    if topics_raw:
        # Try comma first, then newline/bullet as fallback
        if "," in topics_raw:
            parts = [t.strip().lstrip("-•* ").strip() for t in topics_raw.split(",")]
        else:
            parts = [t.strip().lstrip("-•* ").strip() for t in topics_raw.split("\n") if t.strip()]
        result["topics_ru"] = [t for t in parts if t and len(t) > 1]
    
    return result


def _validate_sections(parsed: dict, article_title: str) -> None:
    """Log warnings for suspiciously short/long sections."""
    for key, cfg in _VALIDATE.items():
        val = parsed.get(key, "")
        length = len(val)
        label = cfg["label"]
        
        if val and length < cfg["min_len"]:
            print(f"  [LLM] ⚠ {label} подозрительно короткий ({length} < {cfg['min_len']}): "
                  f"{article_title[:40]}...", file=sys.stderr)
        elif val and length > cfg["max_len"]:
            print(f"  [LLM] ⚠ {label} очень длинный ({length} > {cfg['max_len']}), "
                  f"обрезаем: {article_title[:40]}...", file=sys.stderr)
    
    # Check topics count
    topics = parsed.get("topics_ru", [])
    if len(topics) == 0 and parsed.get("overview"):
        print(f"  [LLM] ⚠ темы не распарсены (0 шт): {article_title[:40]}...",
              file=sys.stderr)


def enrich_article(article: dict, db_articles: list[dict]) -> dict:
    """Enrich a single article with LLM-generated summary and analysis."""
    title = article.get("title", "N/A")
    abstract = article.get("abstract", "")[:2000]
    authors = article.get("authors", "N/A")
    journal = article.get("journal", "N/A")
    year = article.get("year", "N/A")
    topic_name = article.get("_topic_name_ru", "")
    scores = article.get("scores", {})
    
    # Find similar articles from DB
    similar = find_similar_articles(article, db_articles, top_n=2)
    similar_text = ""
    if similar:
        similar_lines = []
        for s in similar:
            similar_lines.append(f"- {s.get('title', 'N/A')[:80]} ({s.get('journal', '')}, {s.get('year', '')})")
        similar_text = "\nПохожие статьи из нашей базы:\n" + "\n".join(similar_lines)
    
    prompt = f"""Проанализируй эту научную статью и напиши краткий обзор для российского исследователя.

НАЗВАНИЕ: {title}
АВТОРЫ: {authors}
ЖУРНАЛ: {journal} ({year})
ТЕМА: {topic_name}
АБСТРАКТ: {abstract}
КЛЮЧЕВЫЕ ТЕМЫ: {', '.join(article.get('topics', [])[:5])}

ОЦЕНКИ СКОРИНГА:
- Переносимость метода: {scores.get('transferability', 0):.2f}
- Географическая аналогия: {scores.get('geographic_analogy', 0):.2f}
- Тематическая релевантность: {scores.get('thematic_relevance', 0):.2f}
- Потенциал публикации: {scores.get('publication_potential', 0):.2f}
- Практический выход: {scores.get('practical_output', 0):.2f}
{similar_text}

Напиши ответ в таком формате (строго по секциям):

===ЗАГОЛОВОК_RU===
[переведи название статьи на русский язык, научно, 1 предложение]

===АННОТАЦИЯ_RU===
[краткий пересказ аннотации на русском, 2-3 предложения]

===ТЕМЫ_RU===
[переведи ключевые темы на русский через запятую, 3-5 тем]

===ОБЗОР===
📝 О ЧЁМ ЭТА СТАТЬЯ (2-3 предложения простым языком):
[объясни суть работы так, чтобы понял студент-геолог]

🔬 ГЛАВНЫЕ ВЫВОДЫ:
- [первый ключевой вывод]
- [второй ключевой вывод]
- [третий ключевой вывод]

🇷🇺 ПОЧЕМУ ЭТО ИНТЕРЕСНО ДЛЯ ЮГА РОССИИ:
[объясни конкретно: какие регионы/объекты Юга РФ подходят для применения]

💡 ЧТО С ЭТИМ МОЖНО ДЕЛАТЬ:
[выбери одно: ВОСПРОИЗВЕДЕНИЕ / ОБЗОР / МЕТОДОЛОГИЧЕСКИЙ ТРАНСФЕР] и объясни план действий"""

    # Блок 2.1: увеличен max_tokens до 1536 — хватает на все 4 секции
    response = call_llm(prompt, max_tokens=1536)

    # Парсим структурированный ответ
    enriched = article.copy()

    # Блок 1.2: regex-парсинг вместо строкового split/replace
    parsed = _parse_llm_response(response or "")
    
    # Блок 1.3: валидация качества выходных данных
    _validate_sections(parsed, title)

    # Сохраняем русские поля
    title_ru = parsed["title_ru"]
    abstract_ru = parsed["abstract_ru"]
    topics_ru = parsed["topics_ru"]
    overview_text = parsed["overview"]

    if title_ru:
        enriched["title_ru"] = title_ru
    elif not title_ru and article.get("title"):
        # Fallback: translate English title if LLM didn't produce Russian title
        fb_title = _translate_title_fallback(article["title"])
        if fb_title:
            enriched["title_ru"] = fb_title
        else:
            print(f"  [LLM] ⚠ No title_ru (no fallback either): {article['title'][:40]}...",
                  file=sys.stderr)
    if abstract_ru:
        enriched["abstract_ru"] = abstract_ru
    if topics_ru:
        enriched["topics_ru"] = topics_ru
    # Основной саммари — это обзорная часть
    enriched["llm_summary"] = overview_text or response or ""
    enriched["similar_articles"] = [
        {"title": s.get("title", ""), "doi": s.get("doi", ""), "journal": s.get("journal", "")}
        for s in similar
    ]
    
    return enriched


def enrich_articles_batch(articles: list[dict]) -> list[dict]:
    """
    Enrich all selected articles with LLM summaries.
    
    Блок 1.1: каждая статья обёрнута в try/except — одна плохая не убивает батч.
    """
    # Load existing DB for similarity search
    db_articles = load_articles(limit=200)
    
    enriched = []
    total = len(articles)
    failures = 0
    
    for i, art in enumerate(articles):
        art_label = art.get("title", "")[:50]
        print(f"  [LLM] Enriching {i+1}/{total}: {art_label}...")
        
        # Блок 1.1: per-article error handling
        try:
            enriched_art = enrich_article(art, db_articles)
            
            # Проверка что обогащение дало хоть какой-то результат
            has_content = bool(enriched_art.get("llm_summary"))
            if not has_content:
                print(f"  [LLM] ⚠ Статья обогащена без llm_summary (пустой LLM ответ): {art_label}",
                      file=sys.stderr)
            
            enriched.append(enriched_art)
            
        except Exception as e:
            failures += 1
            print(f"  [LLM] ❌ Ошибка обогащения статьи {i+1}/{total}: {art_label}",
                  file=sys.stderr)
            print(f"  [LLM]    {type(e).__name__}: {e}", file=sys.stderr)
            # Добавляем статью как есть, без обогащения — не теряем данные
            fallback = art.copy()
            fallback["_enrich_error"] = str(e)
            enriched.append(fallback)
        
        if i < total - 1:
            time.sleep(5.0)  # Delay to avoid rate limit
    
    if failures:
        print(f"  [LLM] ⚠ Завершено с {failures}/{total} ошибками обогащения",
              file=sys.stderr)
    else:
        print(f"  [LLM] ✅ Все {total} статей успешно обогащены")
    
    return enriched


# ── Card Formatter (Russian output) ────────────────────────────
TYPE_LABELS = {
    "reproduction": "ВОСПРОИЗВЕДЕНИЕ",
    "review": "ОБЗОР",
    "method_transfer": "МЕТОД",
}

REPRODUCIBILITY_LABELS = {
    "high": "высокая",
    "medium": "средняя",
    "low": "низкая",
}


def format_card(article: dict, topic_name_ru: str) -> str:
    """Format a single article as a Russian-language card with LLM enrichment."""
    scores = article.get("scores", {})
    atype = TYPE_LABELS.get(article.get("article_type", "reproduction"), "ВОСПРОИЗВЕДЕНИЕ")
    doi_str = article.get("doi", "")
    doi_display = f"DOI: {doi_str}" if doi_str else ""
    
    # LLM enriched content
    llm_summary = article.get("llm_summary", "")
    similar = article.get("similar_articles", [])
    similar_lines = ""
    if similar:
        for s in similar:
            s_doi = s.get("doi", "")
            s_doi_text = f" (DOI: {s_doi})" if s_doi else ""
            similar_lines += f"  → {s.get('title', '')[:70]}...{s_doi_text}\n"

    # Reproducibility assessment
    trans = scores.get("transferability", 0)
    geo = scores.get("geographic_analogy", 0)
    if trans >= 0.7 and geo >= 0.5:
        reprod = "high"
    elif trans >= 0.4 and geo >= 0.3:
        reprod = "medium"
    else:
        reprod = "low"

    # Publication idea based on type
    atype_en = article.get("article_type", "reproduction")
    if atype_en == "review":
        pub_idea = (
            f"Обзорная статья: сравнение мировых подходов по теме "
            f"«{topic_name_ru}» с добавлением данных по Югу России"
        )
    elif atype_en == "method_transfer":
        pub_idea = (
            f"Методологическая статья: применение метода "
            f"«{article['title'][:80]}...» к материалам Юга России"
        )
    else:
        pub_idea = (
            f"Повторное исследование: воспроизвести методику на "
            f"аналогичном объекте в Южном регионе РФ"
        )

    # OA info from Unpaywall
    oa_status = article.get("oa_status", "")
    pdf_url = article.get("pdf_url", "")
    license_str = article.get("license", "")
    oa_detail = ""
    if article.get("is_oa"):
        oa_parts = [f"OA: {oa_status}" if oa_status else "OA: yes"]
        if license_str:
            oa_parts.append(f"License: {license_str}")
        if pdf_url:
            oa_parts.append(f"PDF: {pdf_url[:60]}...")
        oa_detail = " | ".join(oa_parts)

    # Build card with or without LLM content
    if llm_summary:
        card = f"""
{'='*72}
[{atype}] {article.get('title', 'N/A')}
{'='*72}
Авторы:     {article.get('authors', 'N/A')}
Журнал:    {article.get('journal', 'N/A')} ({article.get('year', 'N/A')})
{doi_display}
Источник:   {article.get('source', '')}
Цитирований: {article.get('citations', 0)} | Откр. доступ: {'✅' if article.get('is_oa') else '❌'}{f' ({oa_detail})' if oa_detail else ''}
{'─'*72}
{llm_summary}
{'─'*72}"""
        if similar_lines:
            card += f"""📚 ПОХОЖИЕ СТАТЬИ ИЗ БАЗЫ:
{similar_lines}{'─'*72}
"""
        card += f"""Скоринг: {scores.get('total', 0):.2f}/1.0
  переносимость={scores.get('transferability', 0):.2f} | гео.аналогия={scores.get('geographic_analogy', 0):.2f}
  тематика={scores.get('thematic_relevance', 0):.2f} | публикация={scores.get('publication_potential', 0):.2f}
{'='*72}
"""
    else:
        # Fallback without LLM (original format)
        oa_line = f"Откр. доступ: {'✅' + (f' ({oa_detail})' if oa_detail else '') if article.get('is_oa') else '❌'}"
        card = f"""
{'='*72}
[{atype}] {article.get('title', 'N/A')}
{'='*72}
Авторы:     {article.get('authors', 'N/A')}
Журнал:    {article.get('journal', 'N/A')} ({article.get('year', 'N/A')})
{doi_display}
Источник:   {article.get('source', '')}
{oa_line}
Тема журнала: {topic_name_ru}
{'─'*72}
О чём:
  {article.get('abstract', 'N/A')[:500]}{'...' if len(article.get('abstract', '')) > 500 else ''}
{'─'*72}
Скоринг:
  Переносимость:      {scores.get('transferability', 0):.2f}
  Географ. аналогия:  {scores.get('geographic_analogy', 0):.2f}
  Тематика:           {scores.get('thematic_relevance', 0):.2f}
  Потенциал публикации:{scores.get('publication_potential', 0):.2f}
  Практич. выход:     {scores.get('practical_output', 0):.2f}
  Новизна:            {scores.get('novelty', 0):.2f}
  ─────────────────────────────
  ИТОГО:              {scores.get('total', 0):.2f}
{'─'*72}
Релевантность для РФ:
  Воспроизводимость: {REPRODUCIBILITY_LABELS[reprod]}
Потенциал публикации:
  → {pub_idea}
{'='*72}
"""
    return card


# ── Digest Builder ─────────────────────────────────────────────
def build_digest(config: dict) -> str:
    """
    Main pipeline: search → dedup → score → select top N → format.
    Returns formatted digest string.
    """
    topics_cfg = config.get("topics", {})
    digest_cfg = config.get("digest", {})
    n_articles = digest_cfg.get("articles_per_run", 4)
    min_year = digest_cfg.get("min_year", 2023)

    print(f"[*] Loading seen DOIs...")
    seen = load_seen_dois()

    # Also load DOIs from existing articles.jsonl (catch orphans from crashed runs)
    existing = load_articles()
    for art in existing:
        doi = art.get("doi", "")
        if doi:
            seen.add(doi)
        else:
            seen.add(title_hash(art.get("title", ""), str(art.get("year", ""))))

    print(f"[*] Already shown: {len(seen)} articles (seen_dois.txt + articles.jsonl)")

    # ── Source health tracking ─────────────────────────────────
    # Skip sources that return rate-limit / server errors consecutively
    _skipped_sources = set()   # sources disabled for this run
    _source_fails = {}         # source -> consecutive fail count
    # ── Search via sources module ─────────────────────────────
    from sources import get_active_sources, search_all_sources, dedup_and_merge

    active_sources = get_active_sources()
    src_names = [s.name() for s in active_sources]
    print(f"[*] Active sources: {', '.join(src_names)}")
    if not src_names:
        print("[!] No sources available! Check API keys in .env", file=sys.stderr)

    all_candidates = []

    for topic_key, topic_cfg in topics_cfg.items():
        name_ru = topic_cfg.get("name_ru", topic_key)
        queries = topic_cfg.get("queries", [])
        print(f"\n[*] Topic: {name_ru} ({len(queries)} queries)")

        for q in queries:
            print(f"  Searching: {q[:60]}...")

            # Run all active sources with built-in rate limiting
            results = search_all_sources(active_sources, q, min_year=min_year)

            if results:
                src_counts: dict[str, int] = {}
                for r in results:
                    sn = r.get("source", "?")
                    src_counts[sn] = src_counts.get(sn, 0) + 1
                for sn, cnt in sorted(src_counts.items()):
                    print(f"    {sn}: {cnt} results")

            for r in results:
                r["_topic_key"] = topic_key
                r["_topic_name_ru"] = name_ru
                r["_query"] = q

            all_candidates.extend(results)

    print(f"\n[*] Total raw results: {len(all_candidates)}")

    # ── Field-level dedup & merge (via sources module) ────────
    unique_articles, unique_index = dedup_and_merge(all_candidates, seen)

    candidates = list(unique_articles)
    print(f"[*] After dedup: {len(candidates)} new articles")

    # ── Unpaywall Enrichment: check OA status for all unique articles ──
    print("[*] Unpaywall OA check...")
    unpaywall_enriched = 0
    for art in candidates:
        doi = art.get("doi", "")
        if doi and not art.get("pdf_url"):
            oa_info = lookup_unpaywall(doi)
            if oa_info:
                if oa_info.get("is_oa") and not art.get("is_oa"):
                    art["is_oa"] = True
                if oa_info.get("pdf_url") and not art.get("oa_url"):
                    art["oa_url"] = oa_info["pdf_url"]
                art["pdf_url"] = oa_info.get("pdf_url", "")
                art["oa_status"] = oa_info.get("oa_status", "")
                art["license"] = oa_info.get("license", "")
                unpaywall_enriched += 1
    print(f"[*] Unpaywall enriched: {unpaywall_enriched}/{len(candidates)} articles")

    # Relevance gate — filter out off-topic papers
    before_gate = len(candidates)
    candidates = [a for a in candidates if passes_relevance_gate(a)]
    print(f"[*] After relevance gate: {len(candidates)} (filtered {before_gate - len(candidates)})")

    # Score each
    print("[*] Scoring...")
    for art in candidates:
        score_article(art, config)

    # Sort by total score descending
    candidates.sort(key=lambda x: x.get("scores", {}).get("total", 0), reverse=True)

    # Select top N, ensure diversity (max 2 per topic)
    selected = []
    topic_counts = {}
    for art in candidates:
        tk = art.get("_topic_key", "")
        if topic_counts.get(tk, 0) >= 2 and len(topics_cfg) > 2:
            continue
        selected.append(art)
        topic_counts[tk] = topic_counts.get(tk, 0) + 1
        if len(selected) >= n_articles:
            break

    print(f"[*] Selected: {len(selected)} articles")

    # LLM Enrichment — generate summaries, analysis, find similar articles
    print("\n[*] LLM Enrichment — generating article summaries...")
    selected = enrich_articles_batch(selected)
    print(f"[*] Enrichment complete for {len(selected)} articles")

    # Format digest
    now = datetime.now(timezone.utc)
    digest_header = f"""
╔══════════════════════════════════════════════════════════════════╗
║          GEO-ECOLOGY DIGEST — Daily Research Briefing            ║
║              Geology & Ecology of Russian South                   ║
╠══════════════════════════════════════════════════════════════════╣
║  Date: {now.strftime('%Y-%m-%d %H:%M UTC')}                                    ║
║  Issue: #{len(seen) // n_articles + 1}                                               ║
║  Total in database: {len(seen)} previously shown                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

    cards = []
    new_dois = []
    for art in selected:
        card = format_card(art, art.get("_topic_name_ru", ""))
        cards.append(card)
        doi = art.get("doi", "")
        if doi:
            new_dois.append(doi)
        else:
            new_dois.append(title_hash(art.get("title", ""), str(art.get("year", ""))))

        # Save to DB (atomic: record DOI in seen_dois immediately)
        save_article(art)
        if doi:
            add_seen_dois([doi])
        else:
            add_seen_dois([title_hash(art.get("title", ""), str(art.get("year", "")))])

        # Save article text for Graphify
        art_text = f"""# {art.get('title', 'N/A')}

**Authors:** {art.get('authors', 'N/A')}
**Journal:** {art.get('journal', 'N/A')} ({art.get('year', 'N/A')})
**DOI:** {art.get('doi', 'N/A')}
**Source:** {art.get('source', 'N/A')}
**Type:** {art.get('article_type', 'N/A')}
**Score:** {art.get('scores', {}).get('total', 0)}
**Topic:** {art.get('_topic_name_ru', 'N/A')}

## Abstract
{art.get('abstract', 'N/A')}

## Topics
{', '.join(art.get('topics', []))}

## Institutions
{', '.join(art.get('institutions', []))}
"""
        safe_title = "".join(c if c.isalnum() or c in "-_ " else "_" for c in art.get("title", "untitled"))[:100]
        art_file = ARTICLES_DIR / f"{safe_title}.md"
        art_file.write_text(art_text, encoding="utf-8")

    # (seen DOIs already updated atomically per-article above)
    # new_dois kept for summary only

    digest_body = "\n".join(cards)

    summary = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Articles found this run: {len(selected)}
New DOIs added to database: {len(new_dois)}
Total tracked articles: {len(seen) + len(new_dois)}

Breakdown by type:
"""

    type_counts = {}
    topic_sel_counts = {}
    for art in selected:
        t = art.get("article_type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
        tk = art.get("_topic_name_ru", "unknown")
        topic_sel_counts[tk] = topic_sel_counts.get(tk, 0) + 1

    for t, cnt in type_counts.items():
        summary += f"  - {TYPE_LABELS.get(t, t)}: {cnt}\n"
    summary += "\nBy topic:\n"
    for t, cnt in topic_sel_counts.items():
        summary += f"  - {t}: {cnt}\n"

    full_digest = digest_header + digest_body + summary

    # Save digest to file
    digest_file = BASE / f"digest_{now.strftime('%Y%m%d')}.md"
    digest_file.write_text(full_digest, encoding="utf-8")
    print(f"[*] Digest saved to: {digest_file}")

    return full_digest


# ── Octopoda Integration ────────────────────────────────────────
def save_to_octopoda(records: list[dict]):
    """Store article metadata in Octopoda persistent memory."""
    try:
        from octopoda import AgentRuntime
        agent = AgentRuntime("geo_digest")
        for i, rec in enumerate(records):
            agent.remember(
                f"article_{i}_{rec.get('doi', 'unknown')[:12]}",
                json.dumps(rec, ensure_ascii=False)
            )
        agent.remember(
            "last_run",
            datetime.now(timezone.utc).isoformat()
        )
        agent.remember(
            "total_tracked",
            str(len(load_seen_dois()))
        )
        print(f"[Octopoda] Saved {len(records)} records")
    except Exception as e:
        print(f"[Octopoda] Error: {e}", file=sys.stderr)


# ── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  GEO-ECOLOGY DIGEST AGENT")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    config = load_config()

    digest = build_digest(config)

    # Save to Octopoda
    selected_articles = load_articles(limit=4)  # last 4 saved
    save_to_octopoda(selected_articles)

    print("\n" + "=" * 60)
    print("  DIGEST COMPLETE")
    print("=" * 60)

    # Print digest to stdout (for cron delivery)
    print(digest)

    # ── Remove lockfile (allow next run) ──
    lock_file = BASE / "digest.lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
            print("[Lock] Removed digest.lock", file=sys.stderr)
        except OSError:
            pass

    return digest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Geo-Ecology Digest Agent")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml (overrides default)")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID from run_manager (for logging)")
    args = parser.parse_args()

    # Override config path if provided
    if args.config:
        CONFIG_PATH = Path(args.config)
        if CONFIG_PATH.exists():
            print(f"[*] Using custom config: {CONFIG_PATH}")

    main()
