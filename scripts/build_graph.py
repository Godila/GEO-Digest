#!/usr/bin/env python3
"""
build_graph.py — Build knowledge graph from articles.jsonl using MiniMax LLM

Reads articles from the database, extracts semantic relationships between
article pairs via MiniMax-M2.7 (Anthropic-compatible API), and outputs:
  - graph_data.json    — structured graph for Cytoscape.js dashboard
  - graphify-out/      — Graphify-compatible outputs (HTML viz)

Usage:
  python3 scripts/build_graph.py              # full rebuild
  python3 scripts/build_graph.py --no-llm     # metadata-only (no API calls)
  python3 scripts/build_graph.py --update     # incremental (new articles only)
"""

import json
import os
import sys
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations

# ── LLM client (MiniMax M2.7) ───────────────────────────────────
import sys, os as _os
_scripts_dir = _os.path.dirname(_os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
from llm_client import call_minimax

# ── Paths ───────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GEO_DATA_DIR", BASE / "data"))
ARTICLES_DB = DATA_DIR / "articles.jsonl"
GRAPH_OUTPUT = DATA_DIR / "graph_data.json"
CORPUS_DIR = BASE / "corpus"
GRAPHIFY_OUT = BASE / "graphify-out"


# ── Stable Article ID generation (A5) ───────────────────────────
def _make_art_id(art: dict, fallback_idx: int = 0) -> str:
    """Generate a stable graph node ID for an article.
    
    Uses DOI when available (stable across reorders), falls back to title hash.
    """
    doi = (art.get("doi") or "").strip()
    if doi:
        safe_doi = doi.lower().replace("/", "_").replace(".", "_")[:40]
        return f"art_{safe_doi}"
    import hashlib
    title_hash = hashlib.md5(
        (art.get("title", "") + str(art.get("year", ""))).encode()
    ).hexdigest()[:12]
    return f"art_{title_hash}"


# ── Load articles ─────────────────────────────────────────────
def load_articles(limit: int = 0) -> list[dict]:
    """Load articles from JSONL database."""
    if not ARTICLES_DB.exists():
        print(f"[ERROR] {ARTICLES_DB} not found")
        return []
    
    articles = []
    with open(ARTICLES_DB, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    
    if limit > 0:
        articles = articles[-limit:]
    
    return articles


# ── Build nodes from articles (metadata-only) ──────────────────
def build_metadata_nodes(articles: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Build graph nodes and edges from article metadata.
    
    Node types:
      - article: main article node (large)
      - topic: thematic cluster node
      - source: data source node
      - type_node: article_type (reproduction/review/methods/transfer)
    
    Edge types:
      - article → topic (about_topic)
      - article → source (from_source)
      - article → type_node (is_type)
    """
    nodes = []
    edges = []
    seen_nodes = set()
    
    for i, art in enumerate(articles):
        art_id = _make_art_id(art, i)
        doi = art.get("doi", f"unknown_{i}")
        scores = art.get("scores", {})
        total_score = sum(scores.values()) if scores else 0

        # Fallback score for articles that skipped LLM enrichment
        # Uses citation count + recency as proxy for importance
        is_fallback = total_score == 0
        if is_fallback:
            cites = art.get("citations", 0)
            year = art.get("year") or 2020
            age = max(0, 2026 - year)
            # Citation factor: log scale, capped at 5 points
            cite_score = min(5, (cites / 100) ** 0.5 if cites > 0 else 0.1)
            # Recency bonus: newer = higher, -0.2 per year
            recency = max(0, 1.0 - age * 0.15)
            total_score = round(cite_score * 1.5 + recency, 2)
        
        # Article node — русская метка для графа
        topic_ru = art.get("_topic_name_ru", "")
        title_ru = art.get("title_ru", "")
        title_short = art.get("title", "Untitled")[:60]
        abstract_ru = art.get("abstract_ru", "")
        topics_ru_list = art.get("topics_ru", [])
        # Приоритет: title_ru > topic_ru > оригинал (обрезанный)
        graph_label = title_ru or (f"{topic_ru}" if topic_ru else title_short)

        art_node = {
            "data": {
                "id": art_id,
                "label": graph_label,
                "label_en": title_short,  # оригинальный заголовок для тултипа
                "title_ru": title_ru,     # русский заголовок если есть
                "nodeType": "article",
                "doi": doi,
                "journal": art.get("journal", ""),
                "year": art.get("year", 0),
                "citations": art.get("citations", 0),
                "topic_ru": topic_ru,
                "topic_key": art.get("_topic_key", ""),
                "article_type": art.get("article_type", ""),
                "is_oa": art.get("is_oa", False),
                "oa_url": art.get("oa_url", "") or "",
                "total_score": round(total_score, 2),
                "is_fallback": is_fallback,  # True = score from citations, not LLM
                "score_transferability": round(scores.get("transferability", 0), 2),
                "score_geographic": round(scores.get("geographic_analogy", 0), 2),
                "score_thematic": round(scores.get("thematic_relevance", 0), 2),
                "score_publication": round(scores.get("publication_potential", 0), 2),
                "abstract": (art.get("abstract") or "")[:500],
                "abstract_ru": abstract_ru[:1000] if abstract_ru else "",   # Блок 2.3: русский аннотация
                "llm_summary": (art.get("llm_summary") or "")[:1000],
                "topics_ru": topics_ru_list,                                # Блок 2.3: русские темы
                "authors": art.get("authors", [])[:5],
                "source": art.get("source", ""),
                "url": art.get("url", "") or "",
                "size": 25,  # default; recalculated in Step 4 from PageRank
            }
        }
        nodes.append(art_node)
        seen_nodes.add(art_id)
        
        # Topic nodes — русские названия (приоритет: topics_ru из LLM > оригинал)
        all_topics = art.get("topics", [])
        all_topics_ru = art.get("topics_ru", [])
        # Если есть русский перевод тем — используем его для меток
        topic_label_map = {}
        if all_topics_ru and len(all_topics_ru) == len(all_topics):
            for t_en, t_ru in zip(all_topics, all_topics_ru):
                topic_label_map[t_en] = t_ru
        
        for idx, topic in enumerate(all_topics):
            # Use hash of full topic string to avoid collisions
            # e.g. "Carbon Storage" and "carbon storage" get different IDs
            import hashlib
            topic_hash = hashlib.md5(topic.strip().lower().encode()).hexdigest()[:12]
            topic_id = f"topic_{topic_hash}"
            # Русская метка: из LLM-перевода или fallback на _topic_name_ru
            ru_label = ""
            if topic in topic_label_map:
                ru_label = topic_label_map[topic]
            elif idx < len(all_topics_ru):
                ru_label = all_topics_ru[idx]
            
            display_label = ru_label or art.get("_topic_name_ru", topic)[:80]
            
            if topic_id not in seen_nodes:
                nodes.append({
                    "data": {
                        "id": topic_id,
                        "label": display_label,
                        "label_en": topic,  # оригинальная тема для тултипа
                        "nodeType": "topic",
                        "size": 15,
                    }
                })
                seen_nodes.add(topic_id)
            
            edges.append({
                "data": {
                    "id": f"{art_id}_topic_{topic_id}",
                    "source": art_id,
                    "target": topic_id,
                    "relation": "about_topic",
                    "confidence": 1.0,
                    "label": "",
                }
            })
        
        # Source node — русская метка
        src = art.get("source", "unknown")
        src_id = f"source_{src}"
        SRC_NAMES = {
            "openalex": "OpenAlex",
            "crossref": "CrossRef",
            "europe_pmc": "Europe PMC",
            "semantic_scholar": "Semantic Scholar",
            "arxiv": "arXiv",
        }
        if src_id not in seen_nodes:
            nodes.append({
                "data": {
                    "id": src_id,
                    "label": SRC_NAMES.get(src, src),
                    "nodeType": "source",
                    "size": 12,
                }
            })
            seen_nodes.add(src_id)
        
        edges.append({
            "data": {
                "id": f"{art_id}_{src_id}",
                "source": art_id,
                "target": src_id,
                "relation": "from_source",
                "confidence": 1.0,
                "label": "",
            }
        })
        
        # Article type node — русские названия
        atype = art.get("article_type", "unknown")
        type_id = f"type_{atype}"
        TYPE_NAMES = {
            "reproduction": "Воспроизведение",
            "review": "Обзор",
            "methods_transfer": "Трансфер методов",
            "method_transfer": "Трансфер методов",
            "unknown": "Другое",
        }
        if type_id not in seen_nodes:
            nodes.append({
                "data": {
                    "id": type_id,
                    "label": TYPE_NAMES.get(atype, atype.replace("_", " ").title()),
                    "nodeType": "article_type",
                    "size": 14,
                }
            })
            seen_nodes.add(type_id)
        
        edges.append({
            "data": {
                "id": f"{art_id}_{type_id}",
                "source": art_id,
                "target": type_id,
                "relation": "is_type",
                "confidence": 1.0,
                "label": "",
            }
        })
    
    return nodes, edges


# ── Extract relations via MiniMax ──────────────────────────────
RELATION_SYSTEM_PROMPT = """You are a research paper relationship analyzer. Given two academic papers about geology, ecology, seismology, remote sensing, or environmental science, identify meaningful connections between them.

Respond ONLY with valid JSON array. Each relation must have:
- "type": one of: method_overlap, shared_region, builds_on, contrasting_approach, same_authors, complementary_data, citation_link, thematic_cluster
- "description": brief explanation (one sentence)
- "strength": float 0.1-1.0 (how strong is this connection)

Example response:
[{"type":"method_overlap","description":"Both use InSAR for ground deformation monitoring","strength":0.8}] 

If no meaningful connection exists, respond: []"""
# ── Topic filter ─────────────────────────────────────────────
def share_topics(art_a: dict, art_b: dict) -> bool:
    """Check if two articles share at least one topic.

    Pairs without common topics are extremely unlikely to have
    meaningful semantic connections — skip LLM call entirely.
    
    CRITICAL FIX (v3): If BOTH articles have NO topics, return False
    to avoid super-hub effect where one untagged article gets
    compared against everything else, generating spurious edges.
    """
    topics_a = set(t.lower().strip() for t in art_a.get("topics", []) if t)
    topics_b = set(t.lower().strip() for t in art_b.get("topics", []) if t)
    # Both empty → no common ground → skip (was: True, caused super-hub)
    if not topics_a and not topics_b:
        return False
    if not topics_a or not topics_b:
        return True  # One has topics, other doesn't → let LLM decide
    return bool(topics_a & topics_b)


# ── Batch relation extraction ────────────────────────────────
BATCH_SYSTEM_PROMPT = """You are a research paper relationship analyzer.
Given MULTIPLE pairs of academic papers about geology, ecology, seismology,
remote sensing, or environmental science, identify meaningful connections.

Respond ONLY with valid JSON object: {"results": [{"pair_idx": N, "relations": [...]}]}
Each relations array contains objects with:
- "type": one of: method_overlap, shared_region, builds_on, contrasting_approach,
  same_authors, complementary_data, citation_link, thematic_cluster
- "description": brief explanation (one sentence)
- "strength": float 0.1-1.0

If no meaningful connection for a pair, use empty relations array: []
Include ALL pair indices from the input (0 to N-1).

Example:
{"results": [{"pair_idx": 0, "relations": [{"type":"method_overlap","description":"Both use InSAR","strength":0.8}]}, {"pair_idx": 1, "relations": []}]}"""

BATCH_SIZE = 8   # pairs per API call
MAX_CONCURRENT = 3  # parallel batch requests


def batch_extract_relations(pairs: list) -> dict:
    """Extract relations for multiple article pairs in ONE API call."""
    if not pairs:
        return {}

    pair_texts = []
    for idx, (art_a, art_b, _, _) in enumerate(pairs):
        pair_texts.append(
            f"Pair {idx}:\n"
            f"Paper A:\n"
            f"  Title: {art_a.get('title', 'N/A')}\n"
            f"  Topics: {', '.join(art_a.get('topics', []))}\n"
            f"  Abstract: {(art_a.get('abstract') or '')[:300]}\n"
            f"  Type: {art_a.get('article_type', 'N/A')}\n\n"
            f"Paper B:\n"
            f"  Title: {art_b.get('title', 'N/A')}\n"
            f"  Topics: {', '.join(art_b.get('topics', []))}\n"
            f"  Abstract: {(art_b.get('abstract') or '')[:300]}\n"
            f"  Type: {art_b.get('article_type', 'N/A')}"
        )

    prompt = (
        "Analyze these paper pairs for meaningful research connections:\n\n"
        + "\n\n---\n\n".join(pair_texts)
    )

    try:
        response = call_minimax(
            messages=[{"role": "user", "content": prompt}],
            system=BATCH_SYSTEM_PROMPT,
            max_tokens=2000,
        )
        return _parse_batch_response(response, len(pairs))
    except Exception as e:
        print(f"  [Batch Error] {e}")
        return {i: [] for i in range(len(pairs))}


def _parse_batch_response(text: str, expected_pairs: int) -> dict:
    """Parse batch LLM response into per-pair relations."""
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        newline = text.find("\n", start)
        if newline >= 0:
            text = text[newline + 1:]
        end = text.rfind("```")
        if end >= 0:
            text = text[:end]
        text = text.strip()

    valid_types = {
        "method_overlap", "shared_region", "builds_on",
        "contrasting_approach", "same_authors", "complementary_data",
        "citation_link", "thematic_cluster",
    }
    try:
        data = json.loads(text)
        results = {}
        for item in data.get("results", []):
            idx = item.get("pair_idx")
            relations = []
            for r in item.get("relations", []):
                if isinstance(r, dict) and r.get("type") in valid_types:
                    relations.append({
                        "type": r["type"],
                        "description": r.get("description", "")[:200],
                        "strength": min(1.0, max(0.1, float(r.get("strength", 0.5)))),
                    })
            results[idx] = relations
        return results
    except (json.JSONDecodeError, AttributeError):
        print(f"  [Batch Parse Warning] Could not parse structured response")
        return {i: [] for i in range(expected_pairs)}


def extract_relation(art_a: dict, art_b: dict) -> list[dict]:
    """Ask MiniMax to find relationships between two articles."""
    prompt = f"""Paper A:
Title: {art_a.get('title', 'N/A')}
Topics: {', '.join(art_a.get('topics', []))}
Abstract: {(art_a.get('abstract') or '')[:400]}
Type: {art_a.get('article_type', 'N/A')}

Paper B:
Title: {art_b.get('title', 'N/A')}
Topics: {', '.join(art_b.get('topics', []))}
Abstract: {(art_b.get('abstract') or '')[:400]}
Type: {art_b.get('article_type', 'N/A')}

What meaningful research connections exist between these two papers?"""

    try:
        response = call_minimax(
            messages=[{"role": "user", "content": prompt}],
            system=RELATION_SYSTEM_PROMPT,
            max_tokens=500,
        )
        
        # Parse JSON from response
        text = response.strip()
        
        # Strip markdown code fences if present (MiniMax wraps in ```json ... ```)
        if "```" in text:
            # Remove opening fence
            start = text.find("```")
            newline = text.find("\n", start)
            if newline >= 0:
                text = text[newline+1:]
            # Remove closing fence
            end = text.rfind("```")
            if end >= 0:
                text = text[:end]
            text = text.strip()
        
        # Try to extract JSON array
        if text.startswith("["):
            relations = json.loads(text)
        else:
            # Find JSON array in response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                relations = json.loads(text[start:end])
            else:
                relations = []
        
        # Validate and clean
        valid_types = {
            "method_overlap", "shared_region", "builds_on",
            "contrasting_approach", "same_authors", "complementary_data",
            "citation_link", "thematic_cluster",
        }
        cleaned = []
        for r in relations:
            if isinstance(r, dict) and r.get("type") in valid_types:
                cleaned.append({
                    "type": r["type"],
                    "description": r.get("description", "")[:200],
                    "strength": min(1.0, max(0.1, float(r.get("strength", 0.5)))),
                })
        return cleaned
    
    except Exception as e:
        print(f"  [LLM Error] {e}")
        return []


def build_llm_edges(articles: list[dict], existing_edges: list[dict],
                     min_strength: float = 0.3,
                     skip_pairs: set = None) -> tuple[list[dict], set]:
    """
    Use MiniMax to find semantic edges between article pairs.

    OPTIMIZED (v2):
    1. Topic filter: skip pairs without common topics (~60-80% reduction)
    2. Batching: 8 pairs per API call (8x fewer network round-trips)
    3. Concurrent: up to 3 parallel batch requests
    """
    n = len(articles)
    if n < 2:
        print("[LLM] Need at least 2 articles for relation extraction")
        return [], set()

    skip_pairs = skip_pairs or set()
    total_pairs = n * (n - 1) // 2
    REL_NAMES = {
        "method_overlap": "Общие методы",
        "shared_region": "Общий регион",
        "builds_on": "Развивает",
        "contrasting_approach": "Контрастный подход",
        "same_authors": "Общие авторы",
        "complementary_data": "Доп. данные",
        "citation_link": "Цитирование",
        "thematic_cluster": "Тематический кластер",
    }

    # Phase 1: Collect candidate pairs with topic filter
    candidates = []
    skipped_incremental = 0
    filtered_no_topic = 0

    for i, j in combinations(range(n), 2):
        art_a, art_b = articles[i], articles[j]
        art_id_a = _make_art_id(art_a, i)
        art_id_b = _make_art_id(art_b, j)
        doi_i = (art_a.get("doi") or f"idx_{i}").strip().lower()
        doi_j = (art_b.get("doi") or f"idx_{j}").strip().lower()
        pair_key = tuple(sorted([doi_i, doi_j]))

        if pair_key in skip_pairs:
            skipped_incremental += 1
            continue
        if not share_topics(art_a, art_b):
            filtered_no_topic += 1
            continue
        candidates.append((i, j, art_a, art_b, art_id_a, art_id_b, pair_key))

    print(f"[LLM] Pairs: {total_pairs} total, "
          f"{skipped_incremental} incremental-skip, "
          f"{filtered_no_topic} topic-filtered, "
          f"{len(candidates)} to analyze")

    if not candidates:
        print("[LLM] No pairs need analysis")
        return [], skip_pairs

    # Phase 2: Batch + concurrent LLM calls
    llm_edges = []
    processed_pairs = set()
    batches = [candidates[i:i+BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]
    total_batches = len(batches)

    print(f"[LLM] {len(candidates)} pairs in {total_batches} batches "
          f"(BATCH_SIZE={BATCH_SIZE}, MAX_CONCURRENT={MAX_CONCURRENT})")

    def process_batch(batch_idx_and_batch):
        batch_idx, batch = batch_idx_and_batch
        pairs_for_api = [(art_a, art_b, art_id_a, art_id_b)
                          for (_, _, art_a, art_b, art_id_a, art_id_b, _) in batch]
        results = batch_extract_relations(pairs_for_api)
        edges = []
        local_processed = set()
        for local_idx, (_, _, _, _, art_id_a, art_id_b, pair_key) in enumerate(batch):
            relations = results.get(local_idx, [])
            for rel in relations:
                if rel["strength"] >= min_strength:
                    edge_id = f"{art_id_a}_{art_id_b}_{rel['type']}"
                    edges.append({
                        "data": {
                            "id": edge_id,
                            "source": art_id_a,
                            "target": art_id_b,
                            "relation": rel["type"],
                            "label": REL_NAMES.get(rel["type"], rel["type"].replace("_", " ").title()),
                            "confidence": rel["strength"],
                            "description": rel["description"],
                            "llm_generated": True,
                        }
                    })
            local_processed.add(pair_key)
        return batch_idx, edges, local_processed

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(process_batch, (bi, b)): bi for bi, b in enumerate(batches)}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            try:
                batch_idx, batch_edges, batch_processed = future.result()
                llm_edges.extend(batch_edges)
                processed_pairs.update(batch_processed)
                print(f"  [Batch {completed}/{total_batches}] "
                      f"+{len(batch_edges)} edges from batch #{batch_idx+1}")
            except Exception as e:
                print(f"  [Batch {completed}/{total_batches}] Error: {e}")

    print(f"[LLM] Done: {len(llm_edges)} semantic edges from {len(candidates)} pairs "
          f"({skipped_incremental} incremental-skip, {filtered_no_topic} topic-filtered)")
    return llm_edges, processed_pairs


# ── Topic co-occurrence edges ─────────────────────────────────
def build_topic_cooccurrence_edges(articles: list[dict]) -> list[dict]:
    """Connect articles that share topics."""
    edges = []
    n = len(articles)

    for i in range(n):
        for j in range(i + 1, n):
            topics_i = set(articles[i].get("topics", []))
            topics_j = set(articles[j].get("topics", []))
            shared = topics_i & topics_j

            if shared:
                strength = len(shared) / max(len(topics_i | topics_j), 1)
                edges.append({
                    "data": {
                        "id": f"cooccur_{_make_art_id(articles[i], i)}_{_make_art_id(articles[j], j)}",
                        "source": _make_art_id(articles[i], i),
                        "target": _make_art_id(articles[j], j),
                        "relation": "shared_topics",
                        "label": f"Общие темы",
                        "confidence": round(strength, 2),
                        "shared_topics": list(shared),
                    }
                })

    return edges


# ── Extract processed pairs from existing graph ──────────────
def _extract_processed_pairs(graph: dict) -> set:
    """
    Extract set of (doi_a, doi_b) pairs that already have LLM edges
    from a previously saved graph.

    Used for incremental updates to avoid re-computing existing pairs.
    """
    pairs = set()
    # We need to reverse-map node IDs back to DOIs
    # Node IDs are like "art_10_xxxx" (DOI-based) or "art_<hash>"
    # Each article node has a "doi" field in data
    node_doi_map = {}
    for node in graph.get("nodes", []):
        data = node.get("data", {})
        if data.get("nodeType") == "article":
            doi = (data.get("doi") or "").strip().lower()
            node_id = data.get("id", "")
            if doi and node_id:
                node_doi_map[node_id] = doi

    for edge in graph.get("edges", []):
        if not edge.get("data", {}).get("llm_generated"):
            continue
        src = edge["data"].get("source", "")
        tgt = edge["data"].get("target", "")
        doi_a = node_doi_map.get(src, "")
        doi_b = node_doi_map.get(tgt, "")
        if doi_a and doi_b:
            pairs.add(tuple(sorted([doi_a, doi_b])))

    # Also check metadata for stored pairs (from previous incremental runs)
    stored = graph.get("metadata", {}).get("processed_llm_pairs", [])
    if stored:
        for p in stored:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pairs.add(tuple(sorted([str(p[0]).lower(), str(p[1]).lower()])))

    return pairs


# ── Main pipeline ─────────────────────────────────────────────
def build_graph(use_llm: bool = True, incremental: bool = False) -> dict:
    """Full graph building pipeline.

    Args:
        use_llm: If True, call MiniMax for semantic edge extraction.
        incremental: If True, only compute LLM edges for NEW article pairs.
                     Metadata + co-occurrence edges are always rebuilt (fast).
                     LLM edges from previous runs are preserved and merged.
    """
    print("=" * 60)
    print("  GEO-DIGEST GRAPH BUILDER")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print(f"  Mode: {'INCREMENTAL' if incremental else 'FULL REBUILD'}")
    print("=" * 60)

    # ── Step markers for worker log polling ──────────────
    print("[GRAPH_STEP] loading", file=sys.stderr, flush=True)

    # Load articles
    articles = load_articles()
    if not articles:
        print("[ERROR] No articles found. Run digest first.")
        print("[GRAPH_STEP] error", file=sys.stderr, flush=True)
        return {}

    print(f"\nLoaded {len(articles)} articles")

    # ── Incremental: extract existing LLM pairs ───────────────
    existing_llm_edges = []
    processed_pairs = set()

    if incremental and GRAPH_OUTPUT.exists():
        existing = json.loads(GRAPH_OUTPUT.read_text())
        old_count = len(existing.get("nodes", []))
        if old_count >= len(articles):
            print(f"[Incremental] No new articles ({old_count} in graph, {len(articles)} in DB)")
            return existing

        new_count = len(articles) - old_count
        print(f"[Incremental] {old_count} existing + {new_count} new articles")

        # Extract already-processed LLM pairs from previous graph
        processed_pairs = _extract_processed_pairs(existing)
        # Also preserve existing LLM edge objects for merging
        existing_llm_edges = [
            e for e in existing.get("edges", [])
            if e.get("data", {}).get("llm_generated")
        ]
        if processed_pairs:
            print(f"[Incremental] Reusing {len(processed_pairs)} existing LLM pairs")
    elif not incremental:
        print("[Full rebuild] All LLM edges will be recomputed from scratch")
    
    # Step 1: Metadata nodes + edges
    print("[GRAPH_STEP] metadata", file=sys.stderr, flush=True)
    print("\n--- Step 1: Building metadata graph ---")
    nodes, meta_edges = build_metadata_nodes(articles)
    print(f"  {len(nodes)} nodes, {len(meta_edges)} metadata edges")
    
    all_edges = list(meta_edges)
    
    # Step 2: Topic co-occurrence edges
    print("[GRAPH_STEP] cooccurrence", file=sys.stderr, flush=True)
    print("\n--- Step 2: Topic co-occurrence ---")
    cooccur_edges = build_topic_cooccurrence_edges(articles)
    all_edges.extend(cooccur_edges)
    print(f"  {len(cooccur_edges)} co-occurrence edges")
    
    # Step 3: LLM-based semantic edges (optional)
    llm_edges = []
    all_processed_pairs = set(processed_pairs)  # start with existing
    if use_llm:
        print("[GRAPH_STEP] llm_semantic", file=sys.stderr, flush=True)
        print("\n--- Step 3: MiniMax semantic extraction ---")
        try:
            new_llm_edges, new_processed = build_llm_edges(
                articles, all_edges,
                skip_pairs=processed_pairs if incremental else None
            )
            llm_edges = new_llm_edges
            all_processed_pairs.update(new_processed)

            # Incremental mode: merge with preserved existing LLM edges
            if incremental and existing_llm_edges:
                # Build set of current node IDs to validate old edges
                current_node_ids = {n["data"]["id"] for n in nodes}
                merged_count = 0
                for old_edge in existing_llm_edges:
                    src = old_edge["data"].get("source", "")
                    tgt = old_edge["data"].get("target", "")
                    # Only keep old edge if both endpoints still exist in current graph
                    if src in current_node_ids and tgt in current_node_ids:
                        llm_edges.append(old_edge)
                        merged_count += 1
                print(f"  [Merge] Kept {merged_count} existing + {len(new_llm_edges)} new LLM edges")

            all_edges.extend(llm_edges)

            # A4: Deduplicate — remove co-occurrence edges when LLM edge exists
            # LLM semantic edges are more informative than simple topic overlap
            if llm_edges:
                llm_pairs = set()
                for e in llm_edges:
                    pair = tuple(sorted([e["data"]["source"], e["data"]["target"]]))
                    llm_pairs.add(pair)
                before_dedup = len(all_edges)
                all_edges = [
                    e for e in all_edges
                    if e["data"].get("relation") != "shared_topics"
                    or tuple(sorted([e["data"]["source"], e["data"]["target"]])) not in llm_pairs
                ]
                removed = before_dedup - len(all_edges)
                if removed:
                    print(f"  [Dedup] Removed {removed} redundant co-occurrence edges (LLM edges exist)")

        except ValueError as e:
            print(f"  [SKIP] {e}")
        except Exception as e:
            print(f"  [ERROR] LLM extraction failed: {e}")

    # Step 4: Graph analytics — PageRank, Betweenness, Louvain communities
    print("[GRAPH_STEP] analytics", file=sys.stderr, flush=True)
    print("\n--- Step 4: Graph analytics ---")
    try:
        from graph_analytics import compute_all_metrics
        metrics = compute_all_metrics(nodes, all_edges)

        # Annotate article nodes with metrics
        pr = metrics.get("page_rank", {})
        bc = metrics.get("betweenness", {})
        comms = metrics.get("communities", {})

        for n in nodes:
            d = n["data"]
            nid = d.get("id", "")
            if d.get("nodeType") == "article":
                d["page_rank"] = round(pr.get(nid, 0), 4)
                d["betweenness"] = round(bc.get(nid, 0), 4)
                d["community"] = comms.get(nid, -1)
                # Mark hubs and bridges
                d["is_hub"] = nid in metrics.get("hub_nodes", [])
                d["is_bridge"] = nid in metrics.get("bridge_nodes", [])
                # Recalculate node size from structural importance
                base_size = 20 + d["page_rank"] * 50  # PageRank: 20-70
                if d["is_hub"]:
                    base_size += 10   # hub → visually prominent
                if d["is_bridge"]:
                    base_size += 5    # bridge → slight boost
                d["size"] = max(15, min(80, base_size))

        # Add analytics to metadata
        analytics_meta = {
            "hub_count": len(metrics.get("hub_nodes", [])),
            "bridge_count": len(metrics.get("bridge_nodes", [])),
            "community_count": len(metrics.get("community_info", [])),
            "communities": metrics.get("community_info", []),
        }
        print(f"  Hubs: {analytics_meta['hub_count']}, "
              f"Bridges: {analytics_meta['bridge_count']}, "
              f"Communities: {analytics_meta['community_count']}")
    except Exception as e:
        print(f"  [WARN] Analytics failed (non-critical): {e}")
        analytics_meta = {"error": str(e)}

    # Assemble final graph
    graph = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "article_count": len(articles),
            "node_count": len(nodes),
            "edge_count": len(all_edges),
            "llm_edge_count": len(llm_edges),
            "use_llm": use_llm,
            "incremental": incremental,
            "processed_llm_pairs": sorted(list(all_processed_pairs)),
            **analytics_meta,  # Phase 2: graph analytics
        },
        "nodes": nodes,
        "edges": all_edges,
    }
    
    # Save outputs
    print("[GRAPH_STEP] saving", file=sys.stderr, flush=True)
    GRAPH_OUTPUT.write_text(json.dumps(graph, indent=2, ensure_ascii=False))
    print(f"\nSaved: {GRAPH_OUTPUT} ({GRAPH_OUTPUT.stat().st_size} bytes)")
    
    # Also save corpus .md files for graphify compatibility
    CORPUS_DIR.mkdir(exist_ok=True)
    for i, art in enumerate(articles):
        doi = (art.get("doi") or "unknown")[:30].replace("/", "_")
        md = f"""# {art.get('title', 'Untitled')}

**DOI:** {art.get('doi', 'N/A')}
**Journal:** {art.get('journal', 'N/A')} ({art.get('year', '?')})
**Authors:** {', '.join(art.get('authors', [])[:5])}
**Source:** {art.get('source', 'N/A')}
**Topics:** {', '.join(art.get('topics', []))}
**Type:** {art.get('article_type', 'N/A')}
**Score:** {sum(art.get('scores', {}).values()):.2f}

## Abstract
{(art.get('abstract') or 'N/A')[:1500]}

## Summary
{(art.get('llm_summary') or 'N/A')[:2000]}
"""
        (CORPUS_DIR / f"article_{i+1:02d}_{_make_art_id(art, i)}.md").write_text(md)
    
    # Summary
    print("\n" + "=" * 60)
    print("  GRAPH COMPLETE")
    print("=" * 60)
    print(f"  Articles:  {len(articles)}")
    print(f"  Nodes:     {len(nodes)}")
    print(f"  Edges:     {len(all_edges)}")
    print(f"    - metadata:    {len(meta_edges)}")
    print(f"    - co-occurrence: {len(cooccur_edges)}")
    print(f"    - LLM semantic: {len(llm_edges)}")

    # Analytics summary
    if 'analytics_meta' in dir() and isinstance(analytics_meta, dict) and 'hub_count' in analytics_meta:
        print(f"\n  --- Graph Analytics (Phase 2) ---")
        print(f"  Hub nodes:     {analytics_meta.get('hub_count', '?')}")
        print(f"  Bridge nodes:  {analytics_meta.get('bridge_count', '?')}")
        print(f"  Communities:   {analytics_meta.get('community_count', '?')}")

    print("[GRAPH_COMPLETE]", file=sys.stderr, flush=True)
    
    return graph


# ── CLI ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build knowledge graph from articles")
    parser.add_argument("--no-llm", action="store_true", help="Skip MiniMax LLM calls (metadata only)")
    parser.add_argument("--update", action="store_true", help="Incremental update (only new articles)")
    args = parser.parse_args()
    
    result = build_graph(use_llm=not args.no_llm, incremental=args.update)
    
    # Exit with error if empty
    if not result:
        sys.exit(1)
