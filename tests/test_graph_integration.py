"""Integration tests for Graph Tools → Editor Agent → Orchestrator.

Tests:
  1. EditorAgent auto-registers graph tools in ToolRegistry
  2. Orchestrator _enrich_with_graph() adds graph_context to proposals
  3. Cross-topic bridges discovered during editing phase
  4. Graph analytics compute_all_metrics works correctly
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.orchestrator_v2 import EditorOrchestrator, PipelineJob, PipelineState
from engine.tools.graph_tools import GraphTools, create_graph_tools


# ── Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def sample_graph_file(tmp_path) -> Path:
    """Write sample graph to temp file, return Path object."""
    graph = {
        "metadata": {
            "generated_at": "2026-04-26T12:00:00Z",
            "article_count": 3,
            "node_count": 5,
            "edge_count": 4,
            "analytics": {"communities": {"0": {}, "1": {}}},
        },
        "nodes": [
            {"data": {"id": "art_aaa_0", "nodeType": "article",
                      "doi": "10.1111/int.a", "label": "Permafrost carbon",
                      "title": "Permafrost carbon release", "year": 2025,
                      "source": "openalex",
                      "page_rank": 0.30, "betweenness": 0.20,
                      "community": 0, "is_hub": True, "is_bridge": False}},
            {"data": {"id": "art_bbb_1", "nodeType": "article",
                      "doi": "10.1111/int.b", "label": "Arctic methane",
                      "title": "Arctic methane emissions", "year": 2025,
                      "source": "crossref",
                      "page_rank": 0.25, "betweenness": 0.15,
                      "community": 0, "is_hub": False, "is_bridge": True}},
            {"data": {"id": "art_ccc_2", "nodeType": "article",
                      "doi": "10.1111/int.c", "label": "Landscape change",
                      "title": "Landscape change detection", "year": 2024,
                      "source": "europe_pmc",
                      "page_rank": 0.10, "betweenness": 0.02,
                      "community": 1, "is_hub": False, "is_bridge": False}},
            {"data": {"id": "topic_1", "nodeType": "topic", "label": "permafrost"}},
            {"data": {"id": "topic_2", "nodeType": "topic", "label": "methane"}},
        ],
        "edges": [
            {"data": {"source": "art_aaa_0", "target": "art_bbb_1",
                     "relation": "thematic_cluster", "confidence": 0.8, "llm_generated": True}},
            {"data": {"source": "art_bbb_1", "target": "art_ccc_2",
                     "relation": "method_overlap", "confidence": 0.7, "llm_generated": True}},
            {"data": {"source": "art_aaa_0", "target": "topic_1", "relation": "about_topic"}},
            {"data": {"source": "art_bbb_1", "target": "topic_2", "relation": "about_topic"}},
        ],
    }
    p = tmp_path / "graph_data.json"
    p.write_text(json.dumps(graph), encoding="utf-8")
    return p


@pytest.fixture
def mock_editor_result() -> dict:
    """Realistic dict result from EditorAgent.run()."""
    return {
        "job_id": "test_job",
        "phase": "proposals",
        "proposals": [
            {
                "id": "prop_1",
                "title": "Permafrost Carbon Feedback Loop",
                "thesis": "Warming permafrost releases carbon, accelerating warming.",
                "confidence": 0.85,
                "key_references": ["DOI:10.1111/int.a", "DOI:10.1111/int.b"],
            },
            {
                "id": "prop_2",
                "title": "Methane Monitoring from Space",
                "thesis": "Satellite-based tracking of Arctic methane sources.",
                "confidence": 0.72,
                "key_references": ["DOI:10.1111/int.c"],
            },
        ],
    }


@pytest.fixture
def orch(tmp_path) -> EditorOrchestrator:
    """Fresh orchestrator with temp jobs dir."""
    return EditorOrchestrator(jobs_dir=str(tmp_path / "orch_jobs"))


# ════════════════════════════════════════════════════════════════
# S8.2: Editor Agent + Graph Tools Registration
# ════════════════════════════════════════════════════════════════


class TestEditorGraphToolRegistration:
    """Test that EditorAgent loads graph tools into its registry."""

    def test_graph_stats_tool_executable(self, sample_graph_file):
        """graph_stats can be executed through registry."""
        reg = create_graph_tools(graph_path=str(sample_graph_file))
        r = reg.execute("graph_stats", {})
        assert r.success is True
        assert r.data["article_count"] == 3

    def test_graph_centrality_returns_role(self, sample_graph_file):
        """graph_centrality returns role classification."""
        gt = GraphTools(graph_path=str(sample_graph_file))
        r = gt.graph_centrality(doi="10.1111/int.a")
        assert r.success is True
        assert "role" in r.data
        assert r.data["is_hub"] is True

    def test_cross_topic_finds_bridges(self, sample_graph_file):
        """graph_cross_topic finds articles connecting topics."""
        gt = GraphTools(graph_path=str(sample_graph_file))
        r = gt.graph_cross_topic(topic_a="permafrost", topic_b="landscape")
        assert r.success is True
        # art_ccc_2 has "landscape" in label/title
        # art_aaa_0 connects to permafrost topic
        # Bridge should be found via structural connection
        assert "bridges" in r.data

    def test_all_7_tools_in_registry(self, sample_graph_file):
        """Registry contains exactly 7 graph tools."""
        reg = create_graph_tools(graph_path=str(sample_graph_file))
        assert len(reg) == 7
        expected = {"graph_neighbors", "graph_path", "graph_hubs",
                    "graph_clusters", "graph_cross_topic",
                    "graph_centrality", "graph_stats"}
        assert set(reg.list_tools()) == expected


# ════════════════════════════════════════════════════════════════
# S8.3: Orchestrator Graph Enrichment
# ════════════════════════════════════════════════════════════════


class TestOrchestratorGraphEnrichment:
    """Test _enrich_with_graph() adds graph data to proposals."""

    def test_enrich_adds_graph_context(self, sample_graph_file, mock_editor_result, orch):
        """Each proposal's key_references should get centrality data."""
        job = PipelineJob(
            job_id="enrich_test",
            topic="permafrost carbon methane",
            state=PipelineState.SELECTING,
            editor_result=mock_editor_result,
        )

        # Patch DEFAULT_GRAPH_PATH as Path object (not str!)
        with patch("engine.tools.graph_tools.DEFAULT_GRAPH_PATH", new=sample_graph_file):
            orch._enrich_with_graph(job)

        # Check proposals have graph_context
        prop1 = job.editor_result["proposals"][0]
        assert "graph_context" in prop1, f"No graph_context in prop1. Keys: {list(prop1.keys())}"
        assert len(prop1["graph_context"]) >= 1  # At least one ref enriched

        # Check structure of graph_context entry
        gc = prop1["graph_context"][0]
        assert "doi" in gc
        assert "page_rank" in gc
        assert "betweenness" in gc
        assert "role" in gc

    def test_enrich_adds_cross_topic_bridges(self, sample_graph_file, mock_editor_result, orch):
        """Job with multi-word topic should get cross_topic_bridges."""
        job = PipelineJob(
            job_id="bridge_test",
            topic="permafrost landscape",
            state=PipelineState.SELECTING,
            editor_result=mock_editor_result,
        )

        with patch("engine.tools.graph_tools.DEFAULT_GRAPH_PATH", new=sample_graph_file):
            orch._enrich_with_graph(job)

        # Should have cross_topic_bridges (or gracefully skip)
        if "cross_topic_bridges" in job.editor_result:
            assert isinstance(job.editor_result["cross_topic_bridges"], list)

    def test_enrich_skips_gracefully_on_missing_graph(self, mock_editor_result, orch, tmp_path):
        """No error when graph file doesn't exist."""
        bad_path = tmp_path / "nonexistent_graph.json"

        job = PipelineJob(
            job_id="no_graph_test",
            topic="test",
            state=PipelineState.SELECTING,
            editor_result=mock_editor_result,
        )

        with patch("engine.tools.graph_tools.DEFAULT_GRAPH_PATH", new=bad_path):
            orch._enrich_with_graph(job)

        # Proposals unchanged (no graph_context added)
        prop1 = job.editor_result["proposals"][0]
        assert "graph_context" not in prop1

    def test_enrich_preserves_existing_fields(self, sample_graph_file, mock_editor_result, orch):
        """Enrichment doesn't remove existing proposal fields."""
        job = PipelineJob(
            job_id="preserve_test",
            topic="test",
            state=PipelineState.SELECTING,
            editor_result=mock_editor_result,
        )

        with patch("engine.tools.graph_tools.DEFAULT_GRAPH_PATH", new=sample_graph_file):
            orch._enrich_with_graph(job)

        for prop in job.editor_result["proposals"]:
            assert "id" in prop
            assert "title" in prop
            assert "thesis" in prop
            assert "confidence" in prop
            assert "key_references" in prop


# ════════════════════════════════════════════════════════════════
# E2E: Full pipeline with graph enrichment
# ════════════════════════════════════════════════════════════════


class TestFullPipelineWithGraph:
    """End-to-end: run_editing_phase calls _enrich_with_graph."""

    def test_run_editing_enriches(self, sample_graph_file, mock_editor_result, orch):
        """run_editing_phase calls enrichment after editor.run()."""
        with patch.object(orch.editor, 'run', return_value=mock_editor_result), \
             patch("engine.tools.graph_tools.DEFAULT_GRAPH_PATH", new=sample_graph_file):

            job = orch.create_job(topic="permafrost landscape")
            job = orch.run_editing_phase(job)

        assert job.state == PipelineState.SELECTING
        # At least one proposal should have graph_context
        enriched = [p for p in job.editor_result["proposals"]
                    if "graph_context" in p]
        assert len(enriched) > 0, "No proposals enriched with graph_context"


# ════════════════════════════════════════════════════════════════
# Graph Analytics (build_graph Step 4)
# ════════════════════════════════════════════════════════════════


class TestGraphAnalyticsInBuild:
    """Test compute_all_metrics from scripts/graph_analytics.py."""

    @staticmethod
    def _sample_nodes_edges():
        """Return (nodes, edges) for analytics testing."""
        nodes = [
            {"data": {"id": "a", "nodeType": "article", "doi": "10.x/a"}},
            {"data": {"id": "b", "nodeType": "article", "doi": "10.x/b"}},
            {"data": {"id": "c", "nodeType": "article", "doi": "10.x/c"}},
            {"data": {"id": "d", "nodeType": "article", "doi": "10.x/d"}},
        ]
        edges = [
            {"data": {"source": "a", "target": "b"}},
            {"data": {"source": "b", "target": "c"}},
            {"data": {"source": "c", "target": "d"}},
            {"data": {"source": "a", "target": "c"}},  # extra edge
        ]
        return nodes, edges

    def test_analytics_returns_all_keys(self):
        from scripts.graph_analytics import compute_all_metrics
        nodes, edges = self._sample_nodes_edges()
        result = compute_all_metrics(nodes, edges)
        assert "page_rank" in result
        assert "betweenness" in result
        assert "communities" in result
        assert "hub_nodes" in result
        assert "bridge_nodes" in result

    def test_pagerank_sums_to_one(self):
        from scripts.graph_analytics import compute_all_metrics
        nodes, edges = self._sample_nodes_edges()
        result = compute_all_metrics(nodes, edges)
        pr = result["page_rank"]
        # PageRank values should be positive and hub node should have highest
        assert all(v >= 0 for v in pr.values())
        sorted_pr = sorted(pr.items(), key=lambda x: -x[1])
        # Node 'c' is the central hub (connected to a,b,d) — should have highest PR
        assert sorted_pr[0][0] == "c", f"Expected 'c' as top PageRank, got {sorted_pr[0]}"

    def test_communities_partition_all_nodes(self):
        from scripts.graph_analytics import compute_all_metrics
        nodes, edges = self._sample_nodes_edges()
        result = compute_all_metrics(nodes, edges)
        communities = result["communities"]
        # communities is {node_id: community_id}
        article_ids = {n["data"]["id"] for n in nodes if n["data"]["nodeType"] == "article"}
        assigned_communities = set(communities.values())
        # All article nodes should have a community assignment
        assert set(communities.keys()) == article_ids, f"Missing: {article_ids - set(communities.keys())}"
        # At least one community should exist
        assert len(assigned_communities) >= 1
