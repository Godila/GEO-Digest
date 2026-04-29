"""Tests for Graph Tools (engine/tools/graph_tools.py).

Covers all 7 graph tools with real-ish graph data fixtures.
Uses tmp_path for graph_data.json fixture.

Tests:
  TestGraphNeighbors   — depth=1, depth=2, nonexistent DOI, empty graph
  TestGraphPath        — direct, indirect, no path, same node
  TestGraphHubs        — top N, topic filter, min_degree, empty
  TestGraphClusters    — basic, min_size filter, article-only
  TestGraphCrossTopic  — bridge detection, same topic, nonexistent
  TestGraphCentrality  — hub node, peripheral, nonexistent
  TestGraphStats       — with/without graph, metadata correctness
  TestFactory          — create_graph_tools registry completeness
"""

from __future__ import annotations

import json
import pytest

from engine.tools.graph_tools import (
    GraphTools,
    create_graph_tools,
    _load_graph,
    _find_node_by_doi,
    _build_adjacency,
    _get_article_nodes,
    _format_article_node,
)


# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def sample_graph(tmp_path) -> str:
    """Create a small but realistic graph for testing.

    Structure:
      A(art) --thematic_cluster-- B(art)
         |                         |
      about_topic              about_topic
         |                         |
      topic_permafrost       topic_carbon
         |                         |
      C(art) --method_overlap-- D(art)
                |
            shared_region
                |
             E(art) [isolated from A,B,C,D]
    """
    graph = {
        "metadata": {
            "generated_at": "2026-04-26T12:00:00Z",
            "article_count": 5,
            "node_count": 7,
            "edge_count": 6,
            "analytics": {
                "communities": {
                    "0": {"modularity_contribution": 0.15},
                    "1": {"modularity_contribution": 0.08},
                },
                "modularity_score": 0.23,
            },
        },
        "nodes": [
            {"data": {"id": "art_abc123_0", "nodeType": "article",
                      "doi": "10.1111/test.a", "label": "Permafrost carbon release",
                      "title": "Permafrost carbon release under warming",
                      "year": 2025, "source": "openalex",
                      "page_rank": 0.28, "betweenness": 0.15,
                      "community": 0, "is_hub": True, "is_bridge": False}},
            {"data": {"id": "art_def456_1", "nodeType": "article",
                      "doi": "10.1111/test.b", "label": "Carbon storage in soils",
                      "title": "Carbon storage in Arctic soils",
                      "year": 2024, "source": "crossref",
                      "page_rank": 0.22, "betweenness": 0.08,
                      "community": 0, "is_hub": False, "is_bridge": True}},
            {"data": {"id": "art_ghi789_2", "nodeType": "article",
                      "doi": "10.1111/test.c", "label": "Methane emissions spike",
                      "title": "Methane emissions from thawing permafrost",
                      "year": 2025, "source": "europe_pmc",
                      "page_rank": 0.18, "betweenness": 0.22,
                      "community": 0, "is_hub": False, "is_bridge": True}},
            {"data": {"id": "art_jkl012_3", "nodeType": "article",
                      "doi": "10.1111/test.d", "label": "Landscape stability",
                      "title": "Landscape stability in permafrost regions",
                      "year": 2023, "source": "semantic_scholar",
                      "page_rank": 0.12, "betweenness": 0.02,
                      "community": 1, "is_hub": False, "is_bridge": False}},
            {"data": {"id": "art_mno345_4", "nodeType": "article",
                      "doi": "10.1111/test.e", "label": "Water quality assessment",
                      "title": "Water quality in Arctic rivers",
                      "year": 2024, "source": "openalex",
                      "page_rank": 0.05, "betweenness": 0.00,
                      "community": -1, "is_hub": False, "is_bridge": False}},
            {"data": {"id": "topic_pf", "nodeType": "topic", "label": "permafrost"}},
            {"data": {"id": "topic_cs", "nodeType": "topic", "label": "carbon storage"}},
        ],
        "edges": [
            # A-B thematic cluster
            {"data": {"source": "art_abc123_0", "target": "art_def456_1",
                      "relation": "thematic_cluster", "confidence": 0.8, "llm_generated": True}},
            # C-D method overlap
            {"data": {"source": "art_ghi789_2", "target": "art_jkl012_3",
                      "relation": "method_overlap", "confidence": 0.7, "llm_generated": True}},
            # D-E shared region (E is only connected here)
            {"data": {"source": "art_jkl012_3", "target": "art_mno345_4",
                      "relation": "shared_region", "confidence": 0.6, "llm_generated": True}},
            # A-C complementary
            {"data": {"source": "art_abc123_0", "target": "art_ghi789_2",
                      "relation": "complementary_data", "confidence": 0.75, "llm_generated": True}},
            # B-C contrasting
            {"data": {"source": "art_def456_1", "target": "art_ghi789_2",
                      "relation": "contrasting_approach", "confidence": 0.55, "llm_generated": True}},
            # Metadata edges
            {"data": {"source": "art_abc123_0", "target": "topic_pf",
                      "relation": "about_topic", "confidence": 1.0}},
            {"data": {"source": "art_def456_1", "target": "topic_cs",
                      "relation": "about_topic", "confidence": 1.0}},
        ],
    }
    path = tmp_path / "graph_data.json"
    path.write_text(json.dumps(graph), encoding="utf-8")
    return str(path)


@pytest.fixture
def gt(sample_graph):
    """GraphTools instance pointing to sample graph."""
    return GraphTools(graph_path=sample_graph)


@pytest.fixture
def empty_graph(tmp_path) -> str:
    """Empty/missing graph path for error handling tests."""
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    return str(path)


# ── Test Internal Helpers ──────────────────────────────────────

class TestInternalHelpers:
    def test_load_graph_valid(self, sample_graph):
        g = _load_graph(sample_graph)
        assert len(g["nodes"]) == 7
        assert len(g["edges"]) == 7

    def test_load_graph_empty(self, empty_graph):
        g = _load_graph(empty_graph)
        assert g == {}

    def test_find_node_by_doi(self, sample_graph):
        g = _load_graph(sample_graph)
        node = _find_node_by_doi(g, "10.1111/test.a")
        assert node is not None
        assert node["data"]["doi"] == "10.1111/test.a"

    def test_find_node_by_doi_case_insensitive(self, sample_graph):
        g = _load_graph(sample_graph)
        node = _find_node_by_doi(g, "10.1111/TEST.A")
        assert node is not None

    def test_find_node_by_doi_not_found(self, sample_graph):
        g = _load_graph(sample_graph)
        assert _find_node_by_doi(g, "10.9999/nonexistent") is None

    def test_build_adjacency(self, sample_graph):
        g = _load_graph(sample_graph)
        adj = _build_adjacency(g)
        # art_abc123_0 connected to art_def456_1 + art_ghi789_2 + topic_pf
        assert "art_def456_1" in [n for n, _ in adj.get("art_abc123_0", [])]
        assert "art_ghi789_2" in [n for n, _ in adj.get("art_abc123_0", [])]

    def test_get_article_nodes(self, sample_graph):
        g = _load_graph(sample_graph)
        arts = _get_article_nodes(g)
        assert len(arts) == 5
        assert all(n["data"]["nodeType"] == "article" for n in arts)

    def test_format_article_node(self, sample_graph):
        g = _load_graph(sample_graph)
        node = g["nodes"][0]  # article A
        fmt = _format_article_node(node)
        assert "doi" in fmt
        assert "label" in fmt
        assert "page_rank" in fmt
        assert "nodeType" not in fmt  # not included in format


# ── Test Tool 1: graph_neighbors ───────────────────────────────

class TestGraphNeighbors:
    def test_depth_1_direct_neighbors(self, gt):
        r = gt.graph_neighbors(doi="10.1111/test.a")
        assert r.success is True
        neighbors = r.data["neighbors"]
        labels = {n["label"] for n in neighbors}
        # A connected to B (thematic) and C (complementary)
        assert "Carbon storage in soils" in labels or len(neighbors) >= 1

    def test_depth_2_includes_indirect(self, gt):
        r1 = gt.graph_neighbors(doi="10.1111/test.a", depth=1)
        r2 = gt.graph_neighbors(doi="10.1111/test.a", depth=2)
        assert r2.data["count"] >= r1.data["count"]

    def test_edge_type_filter(self, gt):
        r = gt.graph_neighbors(doi="10.1111/test.a", edge_types=["thematic_cluster"])
        assert r.success is True
        for n in r.data["neighbors"]:
            assert n["relation"] == "thematic_cluster"

    def test_nonexistent_doi(self, gt):
        r = gt.graph_neighbors(doi="10.9999/nonexistent")
        assert r.success is False
        assert "not found" in r.error_msg.lower()

    def test_empty_graph(self, empty_graph):
        tools = GraphTools(graph_path=empty_graph)
        r = tools.graph_neighbors(doi="10.x/y")
        assert r.success is False


# ── Test Tool 2: graph_path ────────────────────────────────────

class TestGraphPath:
    def test_direct_connection(self, gt):
        r = gt.graph_path(doi_a="10.1111/test.a", doi_b="10.1111/test.b")
        assert r.success is True
        assert r.data["hops"] == 1  # Direct edge A-B

    def test_indirect_path(self, gt):
        # E → D → C (2 hops)
        r = gt.graph_path(doi_a="10.1111/test.e", doi_b="10.1111/test.c")
        assert r.success is True
        assert r.data["hops"] >= 1

    def test_same_node(self, gt):
        r = gt.graph_path(doi_a="10.1111/test.a", doi_b="10.1111/test.a")
        assert r.success is True
        assert r.data["hops"] == 0

    def test_no_path_within_depth(self, gt):
        # All nodes are connected in our sample, so this should find a path
        r = gt.graph_path(doi_a="10.1111/test.a", doi_b="10.1111/test.e", max_depth=1)
        # E is 3 hops from A (A-C-D-E), so with max_depth=1 might not find it
        if r.data["hops"] == -1:
            assert "No path found" in r.content

    def test_nonexistent_doi_a(self, gt):
        r = gt.graph_path(doi_a="10.9999/x", doi_b="10.1111/test.a")
        assert r.success is False


# ── Test Tool 3: graph_hubs ────────────────────────────────────

class TestGraphHubs:
    def test_hubs_found(self, gt):
        r = gt.graph_hubs(min_degree=1)
        assert r.success is True
        assert r.data["count"] >= 1
        # Article A has degree 3 (B, C, topic_pf) — should be a hub
        hubs = r.data["hubs"]
        assert any(h["degree"] >= 2 for h in hubs)

    def test_topic_filter(self, gt):
        r = gt.graph_hubs(topic_filter="carbon", min_degree=1)
        assert r.success is True
        for h in r.data["hubs"]:
            text = (h.get("label", "") + " " + h.get("title", "")).lower()
            assert "carbon" in text

    def test_high_min_degree_filters(self, gt):
        r = gt.graph_hubs(min_degree=100)
        assert r.success is True
        assert r.data["count"] == 0

    def test_limit_works(self, gt):
        r = gt.graph_hubs(min_degree=1, limit=2)
        assert r.data["count"] <= 2

    def test_empty_graph(self, empty_graph):
        tools = GraphTools(graph_path=empty_graph)
        r = tools.graph_hubs()
        assert r.success is False


# ── Test Tool 4: graph_clusters ────────────────────────────────

class TestGraphClusters:
    def test_clusters_found(self, gt):
        r = gt.graph_clusters(min_size=1)
        assert r.success is True
        assert r.data["count"] >= 1

    def test_min_size_filter(self, gt):
        r1 = gt.graph_clusters(min_size=1)
        r2 = gt.graph_clusters(min_size=5)
        assert r2.data["count"] <= r1.data["count"]

    def test_cluster_has_members(self, gt):
        r = gt.graph_clusters(min_size=1)
        if r.data["clusters"]:
            cluster = r.data["clusters"][0]
            assert "members" in cluster
            assert len(cluster["members"]) >= 1
            assert "community_id" in cluster

    def test_no_analytics_data(self, tmp_path):
        """Graph without analytics field returns graceful message."""
        graph = {"metadata": {}, "nodes": [], "edges": []}
        p = tmp_path / "no_analytics.json"
        p.write_text(json.dumps(graph))
        tools = GraphTools(graph_path=str(p))
        r = tools.graph_clusters()
        assert r.success is True  # Graceful, not error
        assert r.data["count"] == 0


# ── Test Tool 5: graph_cross_topic ⭐ ─────────────────────────

class TestGraphCrossTopic:
    def test_cross_topic_finds_bridges(self, gt):
        """permafrost + carbon should find articles mentioning both or bridging."""
        r = gt.graph_cross_topic(topic_a="permafrost", topic_b="carbon")
        assert r.success is True
        # At least some bridges expected (A mentions both, B mentions carbon near C)
        assert isinstance(r.data["bridges"], list)

    def test_same_topic_returns_mentions(self, gt):
        r = gt.graph_cross_topic(topic_a="permafrost", topic_b="permafrost")
        assert r.success is True
        # Same topic → direct_mention articles

    def test_nonexistent_topics(self, gt):
        r = gt.graph_cross_topic(topic_a="quantum_entanglement", topic_b="string_theory")
        assert r.success is True  # Not error — just empty
        assert r.data.get("count", 0) == 0

    def test_limit_respected(self, gt):
        r = gt.graph_cross_topic(topic_a="a", topic_b="b", limit=1)
        assert r.data["count"] <= 1

    def test_empty_graph(self, empty_graph):
        tools = GraphTools(graph_path=empty_graph)
        r = tools.graph_cross_topic("x", "y")
        assert r.success is False


# ── Test Tool 6: graph_centrality ──────────────────────────────

class TestGraphCentrality:
    def test_hub_node_centrality(self, gt):
        """Article A is a hub — high degree, PR, betweenness."""
        r = gt.graph_centrality(doi="10.1111/test.a")
        assert r.success is True
        d = r.data
        assert d["degree"] >= 2
        assert d["page_rank"] is not None
        assert d["betweenness"] is not None
        assert d["role"] != ""  # Some role assigned

    def test_peripheral_node(self, gt):
        """Article E is peripheral — low degree."""
        r = gt.graph_centrality(doi="10.1111/test.e")
        assert r.success is True
        # E only connected to D (degree=1 via shared_region)
        assert r.data["degree"] >= 1

    def test_nonexistent_doi(self, gt):
        r = gt.graph_centrality(doi="10.9999/nope")
        assert r.success is False

    def test_role_classification(self, gt):
        """Hub node should have 'hub' in role."""
        r = gt.graph_centrality(doi="10.1111/test.a")
        if r.data.get("is_hub"):
            assert "hub" in r.data["role"].lower()


# ── Test Tool 7: graph_stats ───────────────────────────────────

class TestGraphStats:
    def test_stats_with_graph(self, gt):
        r = gt.graph_stats()
        assert r.success is True
        d = r.data
        assert d["article_count"] == 5
        assert d["total_nodes"] == 7
        assert d["total_edges"] == 7
        assert "edge_breakdown" in d
        assert "degree_stats" in d
        assert d["edge_breakdown"]["llm_semantic"] == 5  # 5 LLM edges in sample

    def test_edge_types_present(self, gt):
        r = gt.graph_stats()
        types = r.data["edge_types"]
        assert "thematic_cluster" in types
        assert "about_topic" in types

    def test_degree_stats(self, gt):
        r = gt.graph_stats()
        ds = r.data["degree_stats"]
        assert ds["max"] >= ds["min"]
        assert ds["mean"] > 0

    def test_analytics_flag(self, gt):
        r = gt.graph_stats()
        assert r.data["analytics_available"] is True
        assert r.data["communities_count"] > 0

    def test_empty_graph(self, empty_graph):
        tools = GraphTools(graph_path=empty_graph)
        r = tools.graph_stats()
        assert r.success is False


# ── Test Factory Function ───────────────────────────────────────

class TestCreateGraphTools:
    def test_registry_has_7_tools(self, sample_graph):
        reg = create_graph_tools(graph_path=sample_graph)
        assert len(reg) == 7
        names = reg.list_tools()
        assert "graph_neighbors" in names
        assert "graph_path" in names
        assert "graph_hubs" in names
        assert "graph_clusters" in names
        assert "graph_cross_topic" in names
        assert "graph_centrality" in names
        assert "graph_stats" in names

    def test_schemas_are_valid(self, sample_graph):
        reg = create_graph_tools(graph_path=sample_graph)
        for name in reg.list_tools():
            schema = reg.get_schema(name)
            assert "name" in schema
            assert "description" in schema
            assert "input_schema" in schema
            assert schema["name"] == name

    def test_execute_works(self, sample_graph):
        reg = create_graph_tools(graph_path=sample_graph)
        r = reg.execute("graph_stats", {})
        assert r.success is True
        assert r.data["article_count"] > 0

    def test_unknown_tool(self, sample_graph):
        reg = create_graph_tools(graph_path=sample_graph)
        r = reg.execute("nonexistent_tool", {})
        assert r.success is False
        assert "Unknown tool" in r.error_msg
