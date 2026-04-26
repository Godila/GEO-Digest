"""Graph Tools — LLM-callable functions for knowledge graph queries.

Allows Editor Agent to query structural information from the knowledge graph:
  - Which articles are connected to a given one?
  - What's the shortest path between two works?
  - Which articles are key hubs/bridges?
  - What communities (clusters) exist in the data?
  - Which articles bridge two topics? (killer feature)

Each tool reads graph_data.json via storage.load_graph().

Tools:
  1. graph_neighbors     — Find articles linked to a given article
  2. graph_path          — Shortest path between two articles
  3. graph_hubs          — Most influential articles (high PageRank/degree)
  4. graph_clusters      — Communities (Louvain)
  5. graph_cross_topic   — Articles bridging two topics
  6. graph_centrality    — Node importance in network
  7. graph_stats         — Graph summary

Usage:
    from engine.tools.graph_tools import create_graph_tools, GRAPH_TOOL_SCHEMAS
    from engine.storage import get_storage

    registry = create_graph_tools(get_storage())
    schemas = registry.get_schemas()  # -> list[dict] for LLM API
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from engine.tools.base import ToolRegistry, ToolResult

# Default graph path (same as worker/dal.py)
DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "graph_data.json"


# ── JSON Schemas (Anthropic format) ──────────────────────────────

GRAPH_NEIGHBORS_SCHEMA = {
    "type": "object",
    "properties": {
        "doi": {
            "type": "string",
            "description": "DOI of the article to find neighbors for",
        },
        "depth": {
            "type": "integer",
            "description": "Search depth: 1=direct neighbors only, 2=neighbors of neighbors (default 1)",
            "default": 1,
        },
        "edge_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Filter by edge types (e.g. ['thematic_cluster', 'method_overlap']). Omit for all.",
        },
    },
    "required": ["doi"],
}

GRAPH_PATH_SCHEMA = {
    "type": "object",
    "properties": {
        "doi_a": {
            "type": "string",
            "description": "DOI of the first article",
        },
        "doi_b": {
            "type": "string",
            "description": "DOI of the second article",
        },
        "max_depth": {
            "type": "integer",
            "description": "Maximum path length to search (default 4)",
            "default": 4,
        },
    },
    "required": ["doi_a", "doi_b"],
}

GRAPH_HUBS_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_filter": {
            "type": "string",
            "description": "Filter hubs by topic keyword. Omit for all hubs.",
        },
        "min_degree": {
            "type": "integer",
            "description": "Minimum node degree (connections) to be considered a hub (default 5)",
            "default": 5,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results (default 10)",
            "default": 10,
        },
    },
}

GRAPH_CLUSTERS_SCHEMA = {
    "type": "object",
    "properties": {
        "min_size": {
            "type": "integer",
            "description": "Minimum cluster size to include (default 2)",
            "default": 2,
        },
    },
}

GRAPH_CROSS_TOPIC_SCHEMA = {
    "type": "object",
    "properties": {
        "topic_a": {
            "type": "string",
            "description": "First topic name or keyword (e.g., 'permafrost', 'carbon storage')",
        },
        "topic_b": {
            "type": "string",
            "description": "Second topic name or keyword (e.g., 'landslide', 'methane emission')",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum bridge articles to return (default 10)",
            "default": 10,
        },
    },
    "required": ["topic_a", "topic_b"],
}

GRAPH_CENTRALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "doi": {
            "type": "string",
            "description": "DOI of the article to check centrality for",
        },
    },
    "required": ["doi"],
}

GRAPH_STATS_SCHEMA = {
    "type": "object",
    "properties": {},
}


# ── Internal helpers ─────────────────────────────────────────────

def _load_graph(graph_path: Optional[str] = None) -> dict:
    """Load and return graph data from JSON file."""
    path = Path(graph_path) if graph_path else DEFAULT_GRAPH_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _find_node_by_doi(graph: dict, doi: str) -> Optional[dict]:
    """Find a graph node by DOI (case-insensitive)."""
    doi_lower = doi.lower().strip()
    for node in graph.get("nodes", []):
        nd = node.get("data", {})
        if nd.get("doi", "").lower() == doi_lower:
            return node
    return None


def _node_id(node: dict) -> str:
    """Get node ID from a Cytoscape node."""
    return node.get("data", {}).get("id", "")


def _build_adjacency(graph: dict) -> dict[str, list[tuple[str, dict]]]:
    """Build adjacency list: node_id -> [(neighbor_id, edge_data), ...].

    Returns bidirectional adjacency (undirected graph).
    """
    adj: dict[str, list[tuple[str, dict]]] = {}
    for edge in graph.get("edges", []):
        ed = edge.get("data", {})
        src = ed.get("source", "")
        tgt = ed.get("target", "")
        if not src or not tgt:
            continue
        adj.setdefault(src, []).append((tgt, ed))
        adj.setdefault(tgt, []).append((src, ed))
    return adj


def _get_article_nodes(graph: dict) -> list[dict]:
    """Return only article-type nodes."""
    return [
        n for n in graph.get("nodes", [])
        if n.get("data", {}).get("nodeType") == "article"
    ]


def _format_article_node(node: dict) -> dict:
    """Extract useful fields from an article node for LLM consumption."""
    d = node.get("data", {})
    return {
        "id": d.get("id", ""),
        "doi": d.get("doi", ""),
        "label": d.get("label", d.get("title", ""))[:200],
        "year": d.get("year"),
        "source": d.get("source"),
        "page_rank": d.get("page_rank"),
        "betweenness": d.get("betweenness"),
        "community": d.get("community"),
        "is_hub": d.get("is_hub", False),
        "is_bridge": d.get("is_bridge", False),
    }


# ── Tool Implementations ─────────────────────────────────────────

class GraphTools:
    """Container for graph query tools.

    Each method is a tool that can be registered in ToolRegistry.
    """

    def __init__(self, graph_path: Optional[str] = None):
        self._graph_path = graph_path

    def _graph(self) -> dict:
        return _load_graph(self._graph_path)

    # ── Tool 1: Neighbors ────────────────────────────────────

    def graph_neighbors(self, doi: str, depth: int = 1,
                        edge_types: Optional[list] = None) -> ToolResult:
        """Find articles linked to a given article via semantic edges.

        Returns neighbors at given depth (BFS traversal).
        """
        g = self._graph()
        if not g:
            return ToolResult.fail("Graph not found or empty")

        node = _find_node_by_doi(g, doi)
        if not node:
            return ToolResult.fail(f"Article with DOI '{doi}' not found in graph")

        adj = _build_adjacency(g)
        start = _node_id(node)

        # BFS up to depth
        visited = {start}
        current_level = {start}
        neighbors_found = []

        for d in range(depth):
            next_level = set()
            for nid in current_level:
                for neighbor, edata in adj.get(nid, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_level.add(neighbor)
                        # Only include article nodes in results
                        neighbor_node = None
                        for n in g.get("nodes", []):
                            if _node_id(n) == neighbor:
                                neighbor_node = n
                                break
                        if neighbor_node:
                            nd = neighbor_node.get("data", {})
                            if edge_types and edata.get("relation") not in edge_types:
                                continue
                            neighbors_found.append({
                                **_format_article_node(neighbor_node),
                                "relation": edata.get("relation", ""),
                                "confidence": round(edata.get("confidence", 0), 2),
                                "distance": d + 1,
                            })
            current_level = next_level

        if not neighbors_found:
            return ToolResult.ok(
                content=f"No neighbors found for '{doi}' at depth {depth}",
                data={"neighbors": [], "count": 0},
            )

        return ToolResult.ok(
            content=f"Found {len(neighbors_found)} neighbors for '{doi}' at depth {depth}",
            data={
                "query_doi": doi,
                "depth": depth,
                "count": len(neighbors_found),
                "neighbors": neighbors_found,
            },
        )

    # ── Tool 2: Shortest Path ─────────────────────────────────

    def graph_path(self, doi_a: str, doi_b: str,
                   max_depth: int = 4) -> ToolResult:
        """Find shortest connection path between two articles.

        Uses BFS to find the shortest path through semantic edges.
        Returns the sequence of articles and relationships connecting them.
        """
        g = self._graph()
        if not g:
            return ToolResult.fail("Graph not found or empty")

        node_a = _find_node_by_doi(g, doi_a)
        node_b = _find_node_by_doi(g, doi_b)

        if not node_a:
            return ToolResult.fail(f"Article A with DOI '{doi_a}' not found")
        if not node_b:
            return ToolResult.fail(f"Article B with DOI '{doi_b}' not found")

        start = _node_id(node_a)
        end = _node_id(node_b)

        if start == end:
            return ToolResult.ok(
                content="Same article — distance is 0",
                data={"path": [_format_article_node(node_a)], "hops": 0},
            )

        adj = _build_adjacency(g)

        # BFS from start to end
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth + 1:
                continue

            for neighbor, edata in adj.get(current, []):
                if neighbor == end:
                    # Build result path
                    full_path = []
                    for pid in path + [neighbor]:
                        for n in g.get("nodes", []):
                            if _node_id(n) == pid:
                                full_path.append(_format_article_node(n))
                                break
                    hops = len(full_path) - 1
                    return ToolResult.ok(
                        content=f"Path found: {hops} hop(s) between '{doi_a}' and '{doi_b}'",
                        data={
                            "path": full_path,
                            "hops": hops,
                            "last_edge_relation": edata.get("relation", ""),
                        },
                    )

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return ToolResult.ok(
            content=f"No path found between '{doi_a}' and '{doi_b}' within depth {max_depth}",
            data={"path": [], "hops": -1},
        )

    # ── Tool 3: Hubs ─────────────────────────────────────────

    def graph_hubs(self, topic_filter: str = "", min_degree: int = 5,
                   limit: int = 10) -> ToolResult:
        """Find most influential/hub articles in the graph.

        Hubs are articles with high degree (many connections) and/or
        high PageRank score. These are the key works that connect
        different areas of research.
        """
        g = self._graph()
        if not g:
            return ToolResult.fail("Graph not found or empty")

        adj = _build_adjacency(g)
        articles = _get_article_nodes(g)

        # Compute degree for each article node
        scored = []
        for art in articles:
            aid = _node_id(art)
            d = art.get("data", {})
            degree = len(adj.get(aid, []))

            if degree < min_degree:
                continue

            # Topic filter: check label/title
            if topic_filter:
                label = (d.get("label", "") + " " + d.get("title", "")).lower()
                if topic_filter.lower() not in label:
                    continue

            scored.append({
                **_format_article_node(art),
                "degree": degree,
                "score": round(d.get("page_rank", 0) * 100 + degree, 2),
            })

        # Sort by composite score (PageRank-weighted degree)
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:limit]

        if not top:
            return ToolResult.ok(
                content="No hub articles found matching criteria",
                data={"hubs": [], "count": 0},
            )

        return ToolResult.ok(
            content=f"Found {len(top)} hub articles (min_degree={min_degree})",
            data={
                "hubs": top,
                "count": len(top),
                "criteria": {"min_degree": min_degree, "topic_filter": topic_filter or None},
            },
        )

    # ── Tool 4: Clusters/Communities ──────────────────────────

    def graph_clusters(self, min_size: int = 2) -> ToolResult:
        """Get detected communities/clusters in the knowledge graph.

        Clusters are computed by Louvain algorithm during graph build.
        Each cluster groups topically related articles.
        """
        g = self._graph()
        if not g:
            return ToolResult.fail("Graph not found or empty")

        analytics = g.get("metadata", {}).get("analytics", {})
        communities_raw = analytics.get("communities", {})

        if not communities_raw:
            return ToolResult.ok(
                content="No community detection data in graph. "
                       "Rebuild graph with analytics enabled.",
                data={"clusters": [], "count": 0},
            )

        # Group articles by community ID
        clusters_map: dict[int, list] = {}
        for node in _get_article_nodes(g):
            d = node.get("data", {})
            comm_id = d.get("community")
            if comm_id is None:
                comm_id = -1  # unclustered
            clusters_map.setdefault(comm_id, []).append(_format_article_node(node))

        # Filter by min_size and build output
        clusters = []
        for comm_id in sorted(clusters_map.keys()):
            members = clusters_map[comm_id]
            if len(members) < min_size:
                continue

            # Compute modularity class info from raw data
            mod_class = communities_raw.get(str(comm_id), {})
            clusters.append({
                "community_id": comm_id,
                "size": len(members),
                "modularity_score": round(mod_class.get("modularity_contribution", 0), 4),
                "members": members,
                # Sample topics from members
                "sample_topics": list(set(
                    m["label"][:80] for m in members[:5]
                ))[:5],
            })

        clusters.sort(key=lambda c: c["size"], reverse=True)

        return ToolResult.ok(
            content=f"Found {len(clusters)} clusters (min_size={min_size}), "
                   f"{sum(c['size'] for c in clusters)} articles clustered",
            data={
                "clusters": clusters,
                "count": len(clusters),
                "total_clustered": sum(c["size"] for c in clusters),
            },
        )

    # ── Tool 5: Cross-Topic Bridges ⭐ ─────────────────────────

    def graph_cross_topic(self, topic_a: str, topic_b: str,
                          limit: int = 10) -> ToolResult:
        """Find articles that BRIDGE two topics.

        This is the most powerful tool for cross-theme discovery.
        It finds articles that are connected to BOTH topic areas,
        revealing hidden connections between seemingly separate domains.

        Strategy:
         1. Find articles related to topic_a (by title/topic match)
         2. Find articles related to topic_b
         3. Find their intersection OR find articles that have edges
            to both groups (bridge nodes)
        """
        g = self._graph()
        if not g:
            return ToolResult.fail("Graph not found or empty")

        ta_lower = topic_a.lower().strip()
        tb_lower = topic_b.lower().strip()

        articles = _get_article_nodes(g)
        adj = _build_adjacency(g)

        # Classify articles by topic relevance
        group_a_ids = set()
        group_b_ids = set()

        for art in articles:
            d = art.get("data", {})
            text = (d.get("label", "") + " " +
                    d.get("title", "") + " " +
                    " ".join(d.get("topics", []))).lower()
            aid = _node_id(art)

            if ta_lower in text:
                group_a_ids.add(aid)
            if tb_lower in text:
                group_b_ids.add(aid)

        # Direct intersection: articles mentioning both topics
        direct_bridges = group_a_ids & group_b_ids

        # Structural bridges: articles connected to both groups
        structural_bridges = set()
        for aid in group_a_ids:
            for neighbor, _ in adj.get(aid, []):
                if neighbor in group_b_ids:
                    structural_bridges.add(neighbor)
                # Also check 2-hop: neighbor of neighbor in group B
                for nn, _ in adj.get(neighbor, []):
                    if nn in group_b_ids:
                        structural_bridges.add(aid)
                        break

        # Combine and deduplicate
        all_bridge_ids = (direct_bridges | structural_bridges) - (group_a_ids & group_b_ids)
        # Include direct bridges too
        all_bridge_ids |= direct_bridges

        # Score bridges: prefer those with high centrality and many cross-edges
        bridge_results = []
        for aid in all_bridge_ids:
            for art in articles:
                if _node_id(art) == aid:
                    d = art.get("data", {})
                    # Count cross-connections
                    cross_edges = sum(
                        1 for nid, _ in adj.get(aid, [])
                        if nid in group_a_ids or nid in group_b_ids
                    )
                    is_direct = aid in direct_bridges
                    entry = {
                        **_format_article_node(art),
                        "cross_connections": cross_edges,
                        "direct_mention": is_direct,
                        "structural_bridge": aid in structural_bridges,
                        "bridge_score": round(
                            (d.get("page_rank", 0) * 50 +
                             d.get("betweenness", 0) * 30 +
                             cross_edges * 10 +
                             (100 if is_direct else 0)), 2),
                    }
                    bridge_results.append(entry)
                    break

        bridge_results.sort(key=lambda x: x["bridge_score"], reverse=True)
        top = bridge_results[:limit]

        if not top:
            return ToolResult.ok(
                content=f"No bridge articles found between '{topic_a}' and '{topic_b}'. "
                       f"Try broader topic keywords or check spelling.",
                data={"bridges": [], "group_a_size": len(group_a_ids),
                      "group_b_size": len(group_b_ids)},
            )

        return ToolResult.ok(
            content=f"Found {len(top)} bridge articles connecting "
                   f"'{topic_a}' ({len(group_a_ids)} articles) and "
                   f"'{topic_b}' ({len(group_b_ids)} articles)",
            data={
                "bridges": top,
                "count": len(top),
                "topic_a": topic_a,
                "topic_b": topic_b,
                "group_a_size": len(group_a_ids),
                "group_b_size": len(group_b_ids),
            },
        )

    # ── Tool 6: Centrality ────────────────────────────────────

    def graph_centrality(self, doi: str) -> ToolResult:
        """Get centrality metrics for a specific article.

        Returns PageRank, betweenness, community membership,
        hub/bridge status, and degree.
        """
        g = self._graph()
        if not g:
            return ToolResult.fail("Graph not found or empty")

        node = _find_node_by_doi(g, doi)
        if not node:
            return ToolResult.fail(f"Article with DOI '{doi}' not found in graph")

        d = node.get("data", {})
        aid = _node_id(node)
        adj = _build_adjacency(g)
        degree = len(adj.get(aid, []))

        # Get neighbor info
        neighbors = []
        for nid, edata in adj.get(aid, []):
            for n in g.get("nodes", []):
                if _node_id(n) == nid:
                    neighbors.append({
                        "id": nid,
                        "label": n.get("data", {}).get("label", "")[:100],
                        "relation": edata.get("relation", ""),
                    })
                    break

        result = {
            **_format_article_node(node),
            "degree": degree,
            "neighbor_count": len(neighbors),
            "neighbors_sample": neighbors[:10],
            "role": self._classify_role(d, degree),
        }

        return ToolResult.ok(
            content=f"Centrality for '{d.get('label', doi)[:80]}': "
                   f"PR={d.get('page_rank', 0):.4f}, BTW={d.get('betweenness', 0):.4f}, "
                   f"degree={degree}, role={result['role']}",
            data=result,
        )

    @staticmethod
    def _classify_role(node_data: dict, degree: int) -> str:
        """Classify the structural role of a node."""
        roles = []
        if node_data.get("is_hub"):
            roles.append("hub")
        if node_data.get("is_bridge"):
            roles.append("bridge")
        pr = node_data.get("page_rank", 0)
        btw = node_data.get("betweenness", 0)

        if pr > 0.03 and btw > 0.05:
            roles.append("key_connector")
        elif pr > 0.02:
            roles.append("influential")
        elif btw > 0.03:
            roles.append("connector")
        elif degree <= 1:
            roles.append("peripheral")

        return ", ".join(roles) if roles else "regular"

    # ── Tool 7: Stats ─────────────────────────────────────────

    def graph_stats(self) -> ToolResult:
        """Get summary statistics about the knowledge graph.

        Useful for understanding the overall structure before diving in.
        """
        g = self._graph()
        if not g:
            return ToolResult.fail("Graph not found or empty")

        meta = g.get("metadata", {})
        nodes = g.get("nodes", [])
        edges = g.get("edges", [])
        articles = _get_article_nodes(g)
        adj = _build_adjacency(g)

        # Edge type distribution
        edge_types: dict[str, int] = {}
        llm_count = 0
        cooccur_count = 0
        meta_count = 0
        for e in edges:
            ed = e.get("data", {})
            rel = ed.get("relation", "unknown")
            edge_types[rel] = edge_types.get(rel, 0) + 1
            if ed.get("llm_generated"):
                llm_count += 1
            elif rel == "shared_topics":
                cooccur_count += 1
            else:
                meta_count += 1

        # Degree distribution
        degrees = []
        for art in articles:
            aid = _node_id(art)
            degrees.append(len(adj.get(aid, [])))

        degrees_sorted = sorted(degrees, reverse=True) if degrees else []

        # Analytics summary
        analytics = meta.get("analytics", {})

        result = {
            "article_count": len(articles),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "edge_breakdown": {
                "llm_semantic": llm_count,
                "co_occurrence": cooccur_count,
                "metadata": meta_count,
            },
            "edge_types": edge_types,
            "degree_stats": {
                "max": max(degrees_sorted) if degrees_sorted else 0,
                "min": min(degrees_sorted) if degrees_sorted else 0,
                "mean": round(sum(degrees_sorted) / len(degrees_sorted), 1) if degrees_sorted else 0,
                "top_3_hubs": degrees_sorted[:3],
            },
            "analytics_available": bool(analytics),
            "communities_count": len(analytics.get("communities", {})),
            "generated_at": meta.get("generated_at", "unknown"),
            "isolated_articles": sum(1 for d in degrees if d == 0),
        }

        return ToolResult.ok(
            content=(
                f"Graph: {len(articles)} articles, {len(nodes)} total nodes, "
                f"{len(edges)} edges ({llm_count} LLM semantic, "
                f"{cooccur_count} co-occurrence, {meta_count} metadata). "
                f"{result['isolated_articles']} isolated articles."
            ),
            data=result,
        )


# ── Factory function ────────────────────────────────────────────

def create_graph_tools(graph_path: Optional[str] = None) -> ToolRegistry:
    """Create a ToolRegistry pre-loaded with all 7 graph tools.

    Args:
        graph_path: Optional path to graph_data.json. Defaults to data/graph_data.json.

    Returns:
        ToolRegistry with 7 registered tools ready for EditorAgent use.
    """
    tools = GraphTools(graph_path=graph_path)
    registry = ToolRegistry()

    # Register each method as a tool
    registry.tool(
        name="graph_neighbors",
        description=(
            "Find articles semantically connected to a given article. "
            "Shows which works cite, build on, complement, or contrast with the target. "
            "Use this to explore the research context around a specific paper."
        ),
        input_schema=GRAPH_NEIGHBORS_SCHEMA,
    )(tools.graph_neighbors)

    registry.tool(
        name="graph_path",
        description=(
            "Find the shortest semantic connection path between two articles. "
            "Shows how two seemingly unrelated works are connected through intermediate papers. "
            "Essential for discovering indirect relationships."
        ),
        input_schema=GRAPH_PATH_SCHEMA,
    )(tools.graph_path)

    registry.tool(
        name="graph_hubs",
        description=(
            "Find the most influential/hub articles in the knowledge graph. "
            "Hubs have many connections and high PageRank — they are foundational works "
            "that connect multiple research directions. Use this to identify key references."
        ),
        input_schema=GRAPH_HUBS_SCHEMA,
    )(tools.graph_hubs)

    registry.tool(
        name="graph_clusters",
        description=(
            "Get detected research communities/clusters in the knowledge graph. "
            "Each cluster groups topically related articles identified by Louvain algorithm. "
            "Use this to understand the thematic landscape of available literature."
        ),
        input_schema=GRAPH_CLUSTERS_SCHEMA,
    )(tools.graph_clusters)

    registry.tool(
        name="graph_cross_topic",
        description=(
            "Find articles that BRIDGE two different topics/themes. "
            "This is the MOST POWERFUL tool for cross-theme discovery — it finds works "
            "that sit at the intersection of two research areas, revealing hidden connections. "
            "Example: graph_cross_topic('permafrost', 'landslide') finds papers about "
            "permafrost degradation triggering landslides."
        ),
        input_schema=GRAPH_CROSS_TOPIC_SCHEMA,
    )(tools.graph_cross_topic)

    registry.tool(
        name="graph_centrality",
        description=(
            "Get detailed importance metrics for a specific article in the network: "
            "PageRank (influence), betweenness (bridge importance), community membership, "
            "hub/bridge status, degree, and structural role classification."
        ),
        input_schema=GRAPH_CENTRALITY_SCHEMA,
    )(tools.graph_centrality)

    registry.tool(
        name="graph_stats",
        description=(
            "Get overall statistics about the knowledge graph: size, edge types, "
            "degree distribution, number of communities, isolated articles. "
            "Call this first to understand what data is available."
        ),
        input_schema=GRAPH_STATS_SCHEMA,
    )(tools.graph_stats)

    return registry
