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
import urllib.request
import urllib.error
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
        art_id = f"article_{i+1}"
        doi = art.get("doi", f"unknown_{i}")
        scores = art.get("scores", {})
        total_score = sum(scores.values()) if scores else 0
        
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
                "size": max(20, min(60, total_score * 20)),  # node size by score
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
            topic_id = f"topic_{topic.lower().replace(' ', '_').replace('/', '_')[:50]}"
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
    Use MiniMax to find semantic edges between ALL article pairs.

    Strategy: compare each pair once, only keep relations above min_strength.
    Rate-limit: ~1.7 calls per second to stay within API limits.

    Full graph is built for knowledge base cross-query capability:
    every article connects to every other via LLM-analyzed relations.

    Args:
        skip_pairs: set of (doi_a, doi_b) tuples that already have LLM edges.
                    These pairs will be SKIPPED (incremental mode).

    Returns:
        (llm_edges, processed_pairs) — edges + set of processed DOI pairs
    """
    n = len(articles)
    if n < 2:
        print("[LLM] Need at least 2 articles for relation extraction")
        return [], set()

    skip_pairs = skip_pairs or set()
    llm_edges = []
    processed_pairs = set()   # track what we've done (for next incremental run)
    pair_count = 0
    total_pairs = n * (n - 1) // 2

    # Count how many we'll skip vs compute
    will_compute = 0
    for i in range(n):
        for j in range(i + 1, n):
            doi_i = (articles[i].get("doi") or f"idx_{i}").strip().lower()
            doi_j = (articles[j].get("doi") or f"idx_{j}").strip().lower()
            pair_key = tuple(sorted([doi_i, doi_j]))
            if pair_key not in skip_pairs:
                will_compute += 1

    skipped_count = total_pairs - will_compute
    if skipped_count > 0:
        print(f"[LLM] Incremental: {skipped_count}/{total_pairs} pairs already built, "
              f"computing {will_compute} new...")
    else:
        print(f"[LLM] Analyzing {total_pairs} article pairs for full graph...")

    if will_compute == 0:
        print("[LLM] All pairs already have semantic edges — nothing to do")
        return [], skip_pairs

    for i, j in combinations(range(n), 2):
        art_a, art_b = articles[i], articles[j]
        art_id_a = f"article_{i+1}"
        art_id_b = f"article_{j+1}"

        # Check if this pair was already processed
        doi_i = (art_a.get("doi") or f"idx_{i}").strip().lower()
        doi_j = (art_b.get("doi") or f"idx_{j}").strip().lower()
        pair_key = tuple(sorted([doi_i, doi_j]))

        if pair_key in skip_pairs:
            processed_pairs.add(pair_key)  # mark as present
            continue
        
        pair_count += 1
        print(f"  [{pair_count}/{total_pairs}] {art_id_a} <-> {art_id_b}...", end=" ", flush=True)
        
        relations = extract_relation(art_a, art_b)
        
        # Russian labels for relation types
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

        for rel in relations:
            if rel["strength"] >= min_strength:
                edge_id = f"{art_id_a}_{art_id_b}_{rel['type']}"
                llm_edges.append({
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
                print(f"+{rel['type']}({rel['strength']:.1f})", end="", flush=True)
        
        print()  # newline after each pair

        # Track this pair as processed (success or not — we tried it)
        processed_pairs.add(pair_key)

        # Rate limiting: ~1.7 calls/sec for API safety
        time.sleep(0.6)

    print(f"[LLM] Done: {len(llm_edges)} semantic edges from {pair_count} pairs "
          f"({skipped_count} skipped)")
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
                        "id": f"cooccur_article_{i+1}_article_{j+1}",
                        "source": f"article_{i+1}",
                        "target": f"article_{j+1}",
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
    # Node IDs are like "article_1", "article_2" etc.
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

    # Load articles
    articles = load_articles()
    if not articles:
        print("[ERROR] No articles found. Run digest first.")
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
    print("\n--- Step 1: Building metadata graph ---")
    nodes, meta_edges = build_metadata_nodes(articles)
    print(f"  {len(nodes)} nodes, {len(meta_edges)} metadata edges")
    
    all_edges = list(meta_edges)
    
    # Step 2: Topic co-occurrence edges
    print("\n--- Step 2: Topic co-occurrence ---")
    cooccur_edges = build_topic_cooccurrence_edges(articles)
    all_edges.extend(cooccur_edges)
    print(f"  {len(cooccur_edges)} co-occurrence edges")
    
    # Step 3: LLM-based semantic edges (optional)
    llm_edges = []
    all_processed_pairs = set(processed_pairs)  # start with existing
    if use_llm:
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
        except ValueError as e:
            print(f"  [SKIP] {e}")
        except Exception as e:
            print(f"  [ERROR] LLM extraction failed: {e}")
    
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
        },
        "nodes": nodes,
        "edges": all_edges,
    }
    
    # Save outputs
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
        (CORPUS_DIR / f"article_{i+1:02d}_{doi}.md").write_text(md)
    
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
