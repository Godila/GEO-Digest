"""
GEO-Digest Data Access Layer (DAL)
==================================
Единая точка чтения данных из /app/data/.
Все API endpoints используют этот модуль — нигде больше не парсим файлы напрямую.
"""

import json
import re
from pathlib import Path
from typing import Optional
from collections import deque

DATA_DIR = Path("/app/data")
ARTICLES_DB = DATA_DIR / "articles.jsonl"
GRAPH_DATA = DATA_DIR / "graph_data.json"
RUN_STATS = DATA_DIR / "last_run_stats.json"
ARTICLES_MD_DIR = DATA_DIR / "articles"


# ── Articles ──────────────────────────────────────────────────

def load_all_articles() -> list[dict]:
    """Загрузить все статьи из JSONL."""
    if not ARTICLES_DB.exists():
        return []
    arts = []
    for line in ARTICLES_DB.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line:
            try:
                arts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return arts


def get_article_by_id(article_id: str) -> Optional[dict]:
    """Найти статью по каноническому _id."""
    for a in load_all_articles():
        if a.get("_id") == article_id:
            return a
    # Fallback для старых статей без _id — поиск по doi
    if article_id.startswith("doi:"):
        doi = article_id[4:]
        for a in load_all_articles():
            if (a.get("doi") or "").strip().lower() == doi.lower():
                return a
    # Fallback по hash
    if article_id.startswith("hash:"):
        h = article_id[5:]
        from sources.base import title_hash as _th
        for a in load_all_articles():
            if _th(a.get("title", ""), str(a.get("year", ""))) == h:
                return a
    return None


def get_article_enrichment_md(article: dict) -> Optional[str]:
    """Прочитать .md enrichment файл для статьи."""
    md_path = article.get("_md_path", "")
    if md_path:
        full_path = DATA_DIR / md_path
        if full_path.exists():
            return full_path.read_text(errors="ignore")
    # Fallback: поиск по заголовку
    title = article.get("title", "")
    if title:
        safe = re.sub(r'[/\\:*?"<>|]', "_", title[:80])
        fallback = ARTICLES_MD_DIR / f"{safe}.md"
        if fallback.exists():
            return fallback.read_text(errors="ignore")
    return None


def search_articles(
    query: str = "",
    topic: str = "",
    source: str = "",
    min_score: float = 0,
    max_score: float = 10,
    min_year: int = 0,
    is_oa: Optional[bool] = None,
    sort_by: str = "score_desc",
    limit: int = 20,
    offset: int = 0,
    fields: Optional[list[str]] = None,
) -> dict:
    """
    Поиск и фильтрация статей.

    query: текстовый поиск по title + abstract + llm_summary + title_ru
    topic: фильтр по _topic_key
    source: фильтр по source
    min_score/max_score: фильтр по total_5 score
    is_oa: фильтр по открытому доступу
    sort_by: score_desc | score_asc | date_desc | date_asc | citations_desc | year_desc
    limit/offset: пагинация
    fields: проекция полей (None = все поля, всегда включает _id)

    Возвращает: {"total": N, "count": N, "offset": ..., "limit": ...,
                 "results": [...], "query": {...}}
    """
    arts = load_all_articles()

    # Текстовый поиск
    if query:
        q = query.lower()
        arts = [
            a for a in arts
            if q in (a.get("title") or "").lower()
            or q in (a.get("abstract") or "").lower()
            or q in (a.get("llm_summary") or "").lower()
            or q in (a.get("title_ru") or "").lower()
            or q in (a.get("abstract_ru") or "").lower()
        ]

    # Фильтры
    if topic:
        arts = [a for a in arts if a.get("_topic_key") == topic]
    if source:
        arts = [a for a in arts if a.get("source") == source]
    if min_score > 0:
        arts = [a for a in arts if _get_total_score(a) >= min_score]
    if max_score < 10:
        arts = [a for a in arts if _get_total_score(a) <= max_score]
    if min_year > 0:
        arts = [a for a in arts if (a.get("year") or 0) >= min_year]
    if is_oa is not None:
        arts = [a for a in arts if bool(a.get("is_oa")) == is_oa]

    total = len(arts)

    # Сортировка
    _sort_keys = {
        "score_desc": lambda a: _get_total_score(a),
        "score_asc": lambda a: -_get_total_score(a),
        "date_desc": lambda a: a.get("_saved_at", "") or "",
        "date_asc": lambda a: a.get("_saved_at", "") or "z",
        "citations_desc": lambda a: a.get("citations", 0) or 0,
        "year_desc": lambda a: a.get("year", 0) or 0,
    }
    key_fn = _sort_keys.get(sort_by, _sort_keys["score_desc"])
    arts.sort(key=key_fn, reverse=True)

    # Пагинация
    results = arts[offset:offset + limit]

    # Always enrich with graph IDs (before projection so _graph_id is available)
    _enrich_articles_with_graph_ids(results)

    # Проекция полей
    if fields:
        projected = []
        for a in results:
            p = {f: a[f] for f in fields if f in a}
            p["_id"] = a.get("_id")
            projected.append(p)
        results = projected

    return {
        "total": total,
        "count": len(results),
        "offset": offset,
        "limit": limit,
        "results": results,
        "query": {"q": query, "topic": topic, "source": source,
                  "min_score": min_score, "sort_by": sort_by},
    }


def search_with_ranking(query: str, limit: int = 20, **kwargs) -> dict:
    """Поиск с текстовым ранжированием (бонус за совпадения)."""
    base = search_articles(query=query, limit=limit * 3, **kwargs)
    q = query.lower()

    for art in base["results"]:
        bonus = 0.0
        reasons = []

        title = (art.get("title") or "").lower()
        abstract = (art.get("abstract") or "").lower()
        summary = (art.get("llm_summary") or "").lower()
        title_ru = (art.get("title_ru") or "").lower()
        topics_str = ", ".join(art.get("topics_ru", []) + art.get("topics", [])).lower()

        # Query words scoring
        for word in q.split():
            if len(word) < 2:
                continue
            if word in title:
                bonus += 0.30
                reasons.append(f"'{word}' в заголовке")
            elif word in abstract:
                bonus += 0.10
                reasons.append(f"'{word}' в abstract")
            elif word in summary:
                bonus += 0.20
                reasons.append(f"'{word}' в LLM-summary")
            elif word in title_ru:
                bonus += 0.15
                reasons.append(f"'{word}' в названии (RU)")
            elif word in topics_str:
                bonus += 0.12
                reasons.append(f"'{word}' в топиках")

        art["_search_score"] = round(_get_total_score(art) + bonus, 3)
        art["_rank_reason"] = "; ".join(reasons[:5]) if reasons else ""

    # Re-sort by search score
    base["results"].sort(key=lambda a: a.get("_search_score", 0), reverse=True)
    base["results"] = base["results"][:limit]
    base["count"] = len(base["results"])
    base["query"]["q"] = query
    base["query"]["ranking"] = True

    return base


def _get_total_score(article: dict) -> float:
    """Извлечь total_5 или total score."""
    s = article.get("scores", {})
    return s.get("total_5", s.get("total", 0))


# ── Graph ────────────────────────────────────────────────────

def load_graph() -> dict:
    """Загрузить граф из graph_data.json."""
    if not GRAPH_DATA.exists():
        return {"nodes": [], "edges": [], "metadata": {}}
    return json.loads(GRAPH_DATA.read_text())


def get_node(node_id: str) -> Optional[dict]:
    """Найти узел по ID."""
    g = load_graph()
    for n in g.get("nodes", []):
        if n["data"].get("id") == node_id:
            return n["data"]
    return None


def get_neighbors(node_id: str, depth: int = 1, edge_types: Optional[list] = None) -> dict:
    """Получить подграф вокруг узла (BFS).

    depth: сколько прыжков (1 = прямые соседи)
    edge_types: фильтр типов рёбер (None = все)
    """
    g = load_graph()
    nodes_map = {n["data"]["id"]: n["data"] for n in g.get("nodes", [])}
    edges = g.get("edges", [])

    collected_nodes = set()
    collected_edges = []

    frontier = {node_id}
    for _d in range(depth):
        next_frontier = set()
        for nid in frontier:
            collected_nodes.add(nid)
            for e in edges:
                ed = e["data"]
                src, tgt = ed.get("source"), ed.get("target")
                rel = ed.get("relation", "")

                if edge_types and rel not in edge_types:
                    continue

                if src == nid or tgt == nid:
                    other = tgt if src == nid else src
                    collected_edges.append(ed)
                    collected_nodes.add(other)
                    next_frontier.add(other)
        frontier = next_frontier - collected_nodes

    result_nodes = [nodes_map[nid] for nid in collected_nodes if nid in nodes_map]

    return {
        "center": node_id,
        "depth": depth,
        "node_count": len(result_nodes),
        "edge_count": len(collected_edges),
        "nodes": result_nodes,
        "edges": collected_edges,
    }


def find_path(from_id: str, to_id: str, max_depth: int = 4) -> dict:
    """Найти кратчайший путь между двумя узлами (BFS).

    Возвращает: {"found": bool, "length": int, "hops": [...]}
    Каждый hop: {"from": id, "to": id, "relation": str, "confidence": float}
    """
    g = load_graph()
    adj = {}
    for e in g.get("edges", []):
        ed = e["data"]
        s, t = ed.get("source"), ed.get("target")
        rel = ed.get("relation", "")
        conf = ed.get("confidence", 1.0)
        adj.setdefault(s, []).append((t, rel, conf))
        adj.setdefault(t, []).append((s, rel, conf))

    visited = {from_id}
    queue = deque([(from_id, [])])

    while queue:
        current, path = queue.popleft()
        if current == to_id:
            hops = [
                {"from": fr, "to": to, "relation": r, "confidence": c}
                for fr, to, r, c in path
            ] + [{"from": current, "to": current, "relation": "", "confidence": 1.0}]
            return {"found": True, "length": len(path), "hops": hops}

        if len(path) >= max_depth:
            continue

        for neighbor, rel, conf in adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [(current, neighbor, rel, conf)]))

    return {"found": False, "length": -1, "hops": [], "error": "no path found"}


def get_subgraph(topic: str = "", node_type: str = "", min_score: float = 0) -> dict:
    """Получить подграф с фильтрами.

    topic: фильтр по topic_key/topic_ru
    node_type: фильтр по nodeType (article, topic, etc.)
    min_score: минимальный score для article-узлов
    """
    g = load_graph()
    all_nodes = g.get("nodes", [])
    all_edges = g.get("edges", [])

    valid_ids = set()

    if node_type:
        for n in all_nodes:
            d = n["data"]
            if d.get("nodeType") != node_type:
                continue
            if min_score > 0 and d.get("score", 0) < min_score:
                continue
            valid_ids.add(d.get("id"))

    if topic:
        topic_ids = set()
        for n in all_nodes:
            d = n["data"]
            if d.get("topic_key") == topic or d.get("topic_ru") == topic:
                topic_ids.add(d.get("id"))
        if valid_ids:
            valid_ids &= topic_ids
        else:
            valid_ids = topic_ids
        # Добавляем соседей через рёбра
        expanded = set(valid_ids)
        for e in all_edges:
            ed = e["data"]
            s, t = ed.get("source"), ed.get("target")
            if s in valid_ids or t in valid_ids:
                expanded.add(s)
                expanded.add(t)
        valid_ids = expanded

    if not valid_ids and not topic and not node_type:
        valid_ids = {n["data"].get("id") for n in all_nodes}

    nodes = [n["data"] for n in all_nodes if n["data"].get("id") in valid_ids]
    edges = [
        e["data"] for e in all_edges
        if e["data"].get("source") in valid_ids and e["data"].get("target") in valid_ids
    ]

    return {"node_count": len(nodes), "edge_count": len(edges), "nodes": nodes, "edges": edges}


# ── Stats & Meta ─────────────────────────────────────────────

def get_stats() -> dict:
    """Агрегированная статистика системы."""
    arts = load_all_articles()
    g = load_graph()

    scores = [_get_total_score(a) for a in arts]

    by_topic = {}
    for a in arts:
        tk = a.get("_topic_key", "unknown")
        by_topic.setdefault(tk, {"count": 0, "avg_score": 0})
        by_topic[tk]["count"] += 1
        by_topic[tk]["avg_score"] += _get_total_score(a)
    for v in by_topic.values():
        if v["count"] > 0:
            v["avg_score"] = round(v["avg_score"] / v["count"], 2)

    by_source = {}
    for a in arts:
        src = a.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1

    run_stats = {}
    if RUN_STATS.exists():
        try:
            run_stats = json.loads(RUN_STATS.read_text())
        except Exception:
            pass

    llm_edge_count = sum(
        1 for e in g.get("edges", [])
        if e.get("data", {}).get("llm_generated") is True
    )

    return {
        "article_count": len(arts),
        "node_count": len(g.get("nodes", [])),
        "edge_count": len(g.get("edges", [])),
        "llm_edge_count": llm_edge_count,
        "score_avg": round(sum(scores) / len(scores), 2) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "score_min": min(scores) if scores else 0,
        "by_topic": by_topic,
        "by_source": by_source,
        "oa_count": sum(1 for a in arts if a.get("is_oa")),
        "with_doi": sum(1 for a in arts if a.get("doi")),
        "with_enrichment": sum(1 for a in arts if a.get("_enriched_at")),
        "with_canonical_id": sum(1 for a in arts if a.get("_id")),
        "last_run": run_stats.get("funnel", {}),
        "generated_at": g.get("metadata", {}).get("generated_at"),
    }


def get_topics() -> list[dict]:
    """Список всех топиков с количеством статей."""
    arts = load_all_articles()
    topics = {}
    for a in arts:
        tk = a.get("_topic_key", "")
        tn = a.get("_topic_name_ru", tk)
        if tk and tk not in topics:
            topics[tk] = {"key": tk, "name_ru": tn, "name_en": "", "count": 0}
        if tk in topics:
            topics[tk]["count"] += 1
    return sorted(topics.values(), key=lambda x: -x["count"])


def get_info() -> dict:
    """Метаданные системы для агента."""
    stats = get_stats()
    return {
        "version": "0.5",
        "name": "GEO-Digest Agent API",
        "data_format": "file-based (JSONL + Cytoscape JSON)",
        "article_count": stats["article_count"],
        "node_count": stats["node_count"],
        "edge_count": stats["edge_count"],
        "endpoints": [
            "/api/a/articles",
            "/api/a/article/:id",
            "/api/a/search",
            "/api/a/graph/neighbors",
            "/api/a/graph/path",
            "/api/a/graph/subgraph",
            "/api/a/stats",
            "/api/a/topics",
            "/api/a/info",
            "/api/a/export",
        ],
        "last_updated": stats.get("generated_at"),
        "id_schemes": {
            "article_canonical": "doi:<lowercase-doi> or hash:<sha256-16>",
            "graph_node": "article_<N>, topic_<slug>, source_<name>",
            "note": "Graph endpoints use graph_node IDs. Use /api/a/articles?fields=_graph_id to get mapping.",
        },
    }


# ── ID Mapping (canonical ↔ graph) ──────────────────────────

def _build_graph_id_map() -> dict[str, str]:
    """Build mapping: canonical_id → graph_node_id."""
    g = load_graph()
    arts = load_all_articles()
    art_by_title = {}
    for a in arts:
        t = (a.get("title") or "").strip().lower()
        if t:
            art_by_title[t] = a.get("_id", "")

    # Positional fallback: article order ≈ graph node order
    art_nodes = [n["data"] for n in g.get("nodes", []) if n["data"].get("nodeType") == "article"]
    pos_mapping = {}  # index → canonical_id
    for i, a in enumerate(arts):
        if i < len(art_nodes):
            pos_mapping[a.get("_id", "")] = art_nodes[i].get("id", "")

    mapping = {}  # canonical_id → graph_id
    for n in g.get("nodes", []):
        d = n["data"]
        nid = d.get("id", "")
        label = (d.get("label") or "").strip().lower()
        title = (d.get("title") or "").strip().lower()

        # Try to match by title/label
        for key in [title, label]:
            if key and key in art_by_title:
                cid = art_by_title[key]
                if cid:
                    mapping[cid] = nid
                    break

    # Merge positional fallback (doesn't overwrite exact matches)
    for cid, gid in pos_mapping.items():
        if cid and cid not in mapping:
            mapping[cid] = gid

    return mapping


def resolve_graph_id(canonical_id: str) -> Optional[str]:
    """Resolve canonical article ID to graph node ID."""
    m = _build_graph_id_map()
    return m.get(canonical_id)


def _enrich_articles_with_graph_ids(articles: list[dict]) -> list[dict]:
    """Add _graph_id field to articles based on DOI/title matching with graph nodes."""
    g = load_graph()
    if not g.get("nodes"):
        return articles

    # Build lookup: DOI → graph_node_id  and  label → graph_node_id
    doi_to_gid = {}
    label_to_gid = {}
    for n in g.get("nodes", []):
        d = n["data"]
        nid = d.get("id", "")
        doi = (d.get("doi") or "").strip().lower()
        label = (d.get("label") or "").strip().lower()
        title_en = (d.get("label_en") or "").strip().lower()
        if doi:
            doi_to_gid[doi] = nid
        if label:
            label_to_gid[label] = nid
        if title_en and title_en != label:
            label_to_gid[title_en] = nid

    unmatched = []
    for idx, a in enumerate(articles):
        matched = False
        art_doi = (a.get("doi") or "").strip().lower()

        # Priority 1: DOI match (most reliable with A5 stable IDs)
        if art_doi and art_doi in doi_to_gid:
            a["_graph_id"] = doi_to_gid[art_doi]
            matched = True

        # Priority 2: Title/label match
        if not matched:
            title = (a.get("title_ru") or a.get("title") or "").strip().lower()
            if title in label_to_gid:
                a["_graph_id"] = label_to_gid[title]
                matched = True

        if not a.get("_graph_id"):
            unmatched.append(idx)

    if unmatched:
        print(f"[DAL] {len(unmatched)}/{len(articles)} articles without graph ID match")

    return articles
