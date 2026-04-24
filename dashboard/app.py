#!/usr/bin/env python3
"""
Geo-Ecology Digest Dashboard — FastAPI + Cytoscape.js

Web dashboard for browsing articles, viewing knowledge graph,
and exploring relationships discovered by MiniMax LLM.

Run:  python3 dashboard/app.py
      or via Docker: docker compose up --build
"""

import json
import os
import sys
import subprocess
import threading
import uuid
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Add scripts dir to path for run_manager import ──────────────
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ── Paths ───────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
ARTICLES_DB = BASE / "articles.jsonl"
GRAPH_DATA = BASE / "graph_data.json"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Geo-Ecology Digest Dashboard",
    description="Knowledge graph & article browser for geo-ecology research",
    version="1.0.0",
)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Background graph build state ───────────────────────────────
_graph_build_state = {
    "job_id": None,
    "status": "idle",       # idle, running, done, error
    "started_at": None,
    "finished_at": None,
    "stdout": "",
    "stderr": "",
}


def _run_graph_build_bg(job_id: str, use_llm: bool, incremental: bool):
    """Run build_graph.py in background thread, update _graph_build_state."""
    global _graph_build_state
    _graph_build_state.update({
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "stdout": "",
        "stderr": "",
    })
    cmd = [sys.executable, str(BASE / "scripts" / "build_graph.py")]
    if not use_llm:
        cmd.append("--no-llm")
    if incremental:
        cmd.append("--update")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(BASE),
            timeout=900,  # 15 min max for LLM edges
        )
        _graph_build_state["stdout"] = result.stdout[-3000:] if result.stdout else ""
        _graph_build_state["stderr"] = result.stderr[-500:] if result.stderr else ""
        _graph_build_state["status"] = "done" if result.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        _graph_build_state["status"] = "error"
        _graph_build_state["stderr"] = "Timeout (900s)"
    except Exception as e:
        _graph_build_state["status"] = "error"
        _graph_build_state["stderr"] = str(e)
    finally:
        _graph_build_state["finished_at"] = datetime.now(timezone.utc).isoformat()


# ── Data loaders ───────────────────────────────────────────────
def load_articles() -> list[dict]:
    """Load all articles from JSONL database."""
    if not ARTICLES_DB.exists():
        return []
    articles = []
    with open(ARTICLES_DB, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def load_graph() -> dict:
    """Load pre-built graph data."""
    if not GRAPH_DATA.exists():
        return {"nodes": [], "edges": [], "metadata": {}}
    return json.loads(GRAPH_DATA.read_text())


# ── API Endpoints ──────────────────────────────────────────────

@app.get("/api/articles")
async def api_articles(
    sort_by: str = Query("total_score", description="Sort field"),
    order: str = Query("desc", description="asc or desc"),
    topic: str = Query("", description="Filter by topic_key"),
    article_type: str = Query("", description="Filter by article_type"),
    source: str = Query("", description="Filter by source"),
    limit: int = Query(50, description="Max results"),
    offset: int = Query(0, description="Offset for pagination"),
):
    """Get articles with sorting and filtering."""
    articles = load_articles()

    # Enrich with computed total_score
    for art in articles:
        scores = art.get("scores", {})
        art["_total_score"] = round(sum(scores.values()), 2) if scores else 0

    # Filters
    if topic:
        articles = [a for a in articles if a.get("_topic_key") == topic]
    if article_type:
        articles = [a for a in articles if a.get("article_type") == article_type]
    if source:
        articles = [a for a in articles if a.get("source") == source]

    # Sort
    reverse = (order == "desc")
    valid_sort_fields = {
        "_total_score", "year", "citations", "_saved_at",
        "score_transferability", "score_geographic",
        "score_thematic", "score_publication",
    }
    sort_field = sort_by if sort_by in valid_sort_fields else "_total_score"
    articles.sort(key=lambda a: a.get(sort_field, 0), reverse=reverse)

    # Paginate
    total = len(articles)
    articles = articles[offset : offset + limit]

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "articles": articles,
    }


@app.get("/api/articles/{article_id}")
async def api_article_detail(article_id: int):
    """Get single article by index (1-based)."""
    articles = load_articles()
    if article_id < 1 or article_id > len(articles):
        raise HTTPException(status_code=404, detail="Article not found")
    return articles[article_id - 1]


@app.get("/api/graph")
async def api_graph():
    """Get full graph data for Cytoscape.js."""
    return load_graph()


@app.get("/api/topics")
async def api_topics():
    """Get unique topics with counts."""
    articles = load_articles()
    topics = {}
    for art in articles:
        key = art.get("_topic_key", "unknown")
        name = art.get("_topic_name_ru", key)
        if key not in topics:
            topics[key] = {"key": key, "name_ru": name, "count": 0}
        topics[key]["count"] += 1
    return sorted(topics.values(), key=lambda x: x["count"], reverse=True)


@app.get("/api/stats")
async def api_stats():
    """Dashboard statistics."""
    articles = load_articles()
    graph = load_graph()

    # Compute stats
    total_scores = []
    for art in articles:
        scores = art.get("scores", {})
        total_scores.append(sum(scores.values()) if scores else 0)

    sources = {}
    types = {}
    for art in articles:
        src = art.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
        atype = art.get("article_type", "unknown")
        types[atype] = types.get(atype, 0) + 1

    oa_count = sum(1 for a in articles if a.get("is_oa"))

    return {
        "total_articles": len(articles),
        "total_nodes": len(graph.get("nodes", [])),
        "total_edges": len(graph.get("edges", [])),
        "llm_edges": graph.get("metadata", {}).get("llm_edge_count", 0),
        "open_access_count": oa_count,
        "avg_score": round(sum(total_scores) / len(total_scores), 2) if total_scores else 0,
        "max_score": round(max(total_scores), 2) if total_scores else 0,
        "sources": sources,
        "types": types,
        "topics_count": len(set(a.get("_topic_key") for a in articles)),
        "graph_generated_at": graph.get("metadata", {}).get("generated_at"),
    }


@app.post("/api/graph/rebuild")
async def api_rebuild_graph(use_llm: bool = True, incremental: bool = False):
    """Trigger graph rebuild (async — returns immediately, poll /api/graph/status)."""
    global _graph_build_state
    if _graph_build_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Graph build already in progress")

    job_id = uuid.uuid4().hex[:8]
    thread = threading.Thread(
        target=_run_graph_build_bg,
        args=(job_id, use_llm, incremental),
        daemon=True,
    )
    thread.start()

    return {
        "success": True,
        "job_id": job_id,
        "status": "running",
        "message": f"Graph build started (llm={use_llm}, incremental={incremental})",
    }


@app.get("/api/graph/status")
async def api_graph_status():
    """Poll background graph build status."""
    state = dict(_graph_build_state)
    # If done, also return current graph metadata
    if state["status"] in ("done", "idle"):
        try:
            gd = load_graph().get("metadata", {})
            state["graph_data"] = gd
        except Exception:
            state["graph_data"] = {}
    return state


@app.get("/api/health")
async def api_health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "has_articles": ARTICLES_DB.exists(),
        "has_graph": GRAPH_DATA.exists(),
    }


# ── Digest Run Management API ─────────────────────────────────

@app.post("/api/digest/start")
async def api_digest_start(config_overrides: dict = None):
    """Start a new digest pipeline run in background."""
    try:
        from run_manager import start_digest_run
        result = start_digest_run(config_overrides or {})
        return result
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"run_manager not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/digest/status")
async def api_digest_status():
    """Get current digest run status (for live polling)."""
    try:
        from run_manager import get_status
        return get_status()
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/digest/stop")
async def api_digest_stop():
    """Stop current running digest."""
    try:
        from run_manager import stop_run
        return stop_run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/digest/runs")
async def api_digest_runs(limit: int = Query(20, description="Max runs to return")):
    """List historical digest runs."""
    try:
        from run_manager import list_runs
        return {"runs": list_runs(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/digest/runs/{run_id}")
async def api_digest_run_detail(run_id: str):
    """Get single run details with full log."""
    try:
        from run_manager import get_run
        run = get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Config API ─────────────────────────────────────────────────

@app.get("/api/digest/config")
async def api_config_get():
    """Get current digest configuration."""
    try:
        from run_manager import get_config
        cfg = get_config()
        # Return only UI-relevant parts (no API keys)
        return {
            "digest": cfg.get("digest", {}),
            "scoring": cfg.get("scoring", {}),
            "topics": {k: {"name_ru": v.get("name_ru", k), "queries_count": len(v.get("queries", []))}
                       for k, v in cfg.get("topics", {}).items()},
            "sources": {k: {"enabled": v.get("enabled", True), "priority": v.get("priority", 0)}
                        for k, v in cfg.get("sources", {}).items()},
            "article_types": cfg.get("article_types", {}),
            "analogous_regions": cfg.get("analogous_regions", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/digest/config")
async def api_config_update(updates: dict):
    """Update digest configuration."""
    try:
        from run_manager import update_config
        updated = update_config(updates)
        return {"ok": True, "config": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dedup Stats ────────────────────────────────────────────────

@app.get("/api/digest/dedup")
async def api_dedup_stats():
    """Get deduplication statistics."""
    try:
        from run_manager import get_dedup_stats
        return get_dedup_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Page Routes ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    """Main dashboard page."""
    template = TEMPLATES_DIR / "index.html"
    if template.exists():
        return HTMLResponse(template.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Geo-Digest Dashboard</h1><p>Template not found.</p>")


# ── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "3000"))
    print(f"Starting Geo-Digest Dashboard on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
