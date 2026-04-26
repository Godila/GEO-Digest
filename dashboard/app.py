#!/usr/bin/env python3
"""
Geo-Ecology Digest Dashboard — Pure UI + API Gateway.

Serves static frontend and proxies heavy operations to Worker service.
Reads data from shared /app/data volume (articles.jsonl, graph_data.json).

Architecture:
  Browser → Dashboard (:3000) → Worker (:3001) [compute]
                              ↘ /app/data [shared volume]

Run:  python dashboard/app.py
      or via Docker: docker compose up dashboard
"""

import json
import os
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlencode

import requests
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# ── Paths ───────────────────────────────────────────────────────
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))

ARTICLES_DB = DATA_DIR / "articles.jsonl"
GRAPH_DATA = DATA_DIR / "graph_data.json"
STATUS_FILE = DATA_DIR / "run_status.json"

# ── Worker URL ──────────────────────────────────────────────────
WORKER_URL = os.environ.get("WORKER_URL", "http://localhost:3001")

# ── App ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Geo-Ecology Digest Dashboard",
    description="Knowledge graph & article browser for geo-ecology research",
    version="2.0.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Helper: call worker API ────────────────────────────────────
def _worker_get(path: str, timeout: int = 10) -> dict:
    """GET request to worker, raise on error."""
    try:
        r = requests.get(f"{WORKER_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Worker unavailable")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Worker error: {e}")


def _worker_post(path: str, json_body: dict | None = None, timeout: int = 10) -> dict:
    """POST request to worker."""
    try:
        r = requests.post(f"{WORKER_URL}{path}", json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Worker unavailable")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Worker error: {e}")


# ── Data loaders (fallback: read from shared volume) ────────────
def load_articles() -> list[dict]:
    """Fallback: read articles from shared volume."""
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
    if not GRAPH_DATA.exists():
        return {"nodes": [], "edges": [], "metadata": {}}
    return json.loads(GRAPH_DATA.read_text())


# ══════════════════════════════════════════════════════════════════
#  DATA ENDPOINTS (proxy to Worker /api/a/* or fallback to volume)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/articles", response_class=JSONResponse)
async def api_articles(
    sort_by: str = Query("total_score"),
    order: str = Query("desc"),
    topic: str = Query(""),
    article_type: str = Query(""),
    source: str = Query(""),
    limit: int = Query(50),
    offset: int = Query(0),
):
    """Article list — proxy to worker /api/a/articles, adapt format for frontend."""
    try:
        params = {"limit": str(limit), "offset": str(offset)}
        if source:
            params["source"] = source
        data = _worker_get(f"/api/a/articles?{urlencode(params)}")

        # Adapt worker format {total, count, offset, limit, results, query}
        # to frontend-expected format {total, offset, limit, articles}
        results = data.get("results", [])
        # Compute _total_score for sorting (frontend expects this field)
        for art in results:
            scores = art.get("scores", {})
            art["_total_score"] = round(
                scores.get("total_5", scores.get("total", 0)),
                2,
            ) if scores else 0

        # Apply client-side filters that worker may not support
        if topic:
            results = [a for a in results if a.get("_topic_key") == topic]
        if article_type:
            results = [a for a in results if a.get("article_type") == article_type]

        # Sort (worker returns pre-sorted, but respect frontend sort param)
        reverse = order == "desc"
        valid_fields = {
            "_total_score", "year", "citations", "_saved_at",
            "score_transferability", "score_geographic",
            "score_thematic", "score_publication",
        }
        sort_base = sort_by
        if sort_base.endswith("_asc"):
            reverse = False; sort_base = sort_base[:-4]
        elif sort_base.endswith("_desc"):
            reverse = True; sort_base = sort_base[:-5]
        sort_field = sort_base if sort_base in valid_fields else "_total_score"

        def _sort_key(a):
            val = a.get(sort_field)
            return val if val is not None else 0

        results.sort(key=_sort_key, reverse=reverse)

        total = data.get("total", len(results))
        results = results[:limit]

        return Response(
            content=json.dumps(
                {"total": total, "offset": offset, "limit": limit, "articles": results},
                ensure_ascii=False,
            ),
            media_type="application/json",
        )
    except HTTPException:
        raise
    except Exception:
        # Fallback: read from shared volume (backward compat)
        articles = load_articles()
        for art in articles:
            scores = art.get("scores", {})
            art["_total_score"] = round(
                scores.get("total_5", scores.get("total", sum(scores.values()) * 5 / len(scores) if scores else 0)),
                2,
            ) if scores else 0
        if topic:
            articles = [a for a in articles if a.get("_topic_key") == topic]
        if article_type:
            articles = [a for a in articles if a.get("article_type") == article_type]
        if source:
            articles = [a for a in articles if a.get("source") == source]
        reverse = order == "desc"
        articles.sort(key=lambda a: a.get(sort_by, 0), reverse=reverse)
        total = len(articles)
        articles = articles[offset : offset + limit]
        return Response(
            content=json.dumps({"total": total, "offset": offset, "limit": limit, "articles": articles}, ensure_ascii=False),
            media_type="application/json",
        )


@app.get("/api/articles/{article_id:path}")
async def api_article_detail(article_id: str):
    """Article detail — try worker by canonical ID first, fallback to index."""
    # Try worker's canonical ID lookup
    try:
        import urllib.parse
        decoded_id = urllib.parse.unquote(article_id)
        return _worker_get(f"/api/a/article/{decoded_id}")
    except HTTPException:
        pass
    # Fallback: numeric index into local file
    articles = load_articles()
    try:
        idx = int(article_id)
        if idx < 1 or idx > len(articles):
            raise HTTPException(status_code=404)
        return articles[idx - 1]
    except ValueError:
        raise HTTPException(status_code=404)


@app.get("/api/graph")
async def api_graph():
    """Graph data — return raw {nodes, edges, metadata} for frontend Cytoscape conversion."""
    raw = load_graph()
    # Frontend's initGraph() expects {nodes: [...], edges: [...], metadata?}
    # and converts to Cytoscape {elements: [...]} itself.
    # Just return the raw file contents as-is.
    return raw


@app.get("/api/topics")
async def api_topics():
    """Topics list — proxy to worker /api/a/topics, adapt format."""
    try:
        data = _worker_get("/api/a/topics")
        # Worker returns: {"topics": [{"name": ..., "count": ...}, ...], "count": N}
        # Frontend expects: [{"key": ..., "name_ru": ..., "count": ...}, ...]
        topics_list = []
        for t in data.get("topics", []):
            name = t.get("name", t.get("name_ru", t.get("key", "?")))
            key = t.get("key", name.lower().replace(" ", "_"))
            topics_list.append({
                "key": key,
                "name_ru": name,
                "count": t.get("count", 0),
            })
        return sorted(topics_list, key=lambda x: x["count"], reverse=True)
    except HTTPException:
        raise
    except Exception:
        # Fallback
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
    """Stats — proxy to worker /api/a/stats, adapt field names for frontend."""
    try:
        data = _worker_get("/api/a/stats")
        # Map worker field names → frontend-expected names
        return {
            "total_articles": data.get("article_count", 0),
            "total_nodes": data.get("node_count", 0),
            "total_edges": data.get("edge_count", 0),
            "llm_edges": data.get("llm_edge_count", 0),
            "open_access_count": data.get("oa_count", 0),
            "avg_score": data.get("score_avg", 0),
            "max_score": data.get("score_max", 0),
            "sources": data.get("by_source", {}),
            "types": {},  # worker doesn't have types breakdown
            "topics_count": len(data.get("by_topic", {})),
            "graph_generated_at": data.get("generated_at"),
        }
    except HTTPException:
        raise
    except Exception:
        # Fallback
        articles = load_articles()
        graph = load_graph()
        total_scores = []
        for art in articles:
            scores = art.get("scores", {})
            total_scores.append(scores.get("total_5", scores.get("total", 0)))
        sources = {}
        for art in articles:
            sources[art.get("source", "unknown")] = sources.get(art.get("source", "unknown"), 0) + 1
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
            "types": {},
            "topics_count": len(set(a.get("_topic_key") for a in articles)),
            "graph_generated_at": graph.get("metadata", {}).get("generated_at"),
        }


@app.get("/api/search", response_class=JSONResponse)
async def api_search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20),
    offset: int = Query(0),
):
    """Full-text search — proxy to worker /api/a/search."""
    params = urlencode({"q": q, "limit": str(limit), "offset": str(offset)})
    data = _worker_get(f"/api/a/search?{params}")
    # Worker returns {total, results: [...], query, took_ms}
    # Return in compatible format
    return Response(
        content=json.dumps(data, ensure_ascii=False),
        media_type="application/json",
    )


@app.get("/api/health")
async def api_health():
    # Check worker availability
    worker_ok = False
    try:
        r = requests.get(f"{WORKER_URL}/api/health", timeout=3)
        worker_ok = r.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok",
        "service": "geo-digest-dashboard",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "has_articles": ARTICLES_DB.exists(),
        "has_graph": GRAPH_DATA.exists(),
        "worker_ok": worker_ok,
        "worker_url": WORKER_URL,
    }


# ══════════════════════════════════════════════════════════════════
#  PROXY ENDPOINTS (delegate to Worker)
# ══════════════════════════════════════════════════════════════════

@app.post("/api/digest/start")
async def api_digest_start(request: Request):
    """Start digest pipeline → proxy to worker."""
    # Read config overrides from UI and forward to worker
    body = None
    try:
        body = await request.json()
    except Exception:
        pass
    return _worker_post("/api/digest/run", json_body=body)


@app.get("/api/digest/status")
async def api_digest_status():
    """Poll digest status → proxy to worker + merge file status."""
    worker_status = _worker_get("/api/digest/status")

    # Also read live status file (written by worker's log poller)
    if STATUS_FILE.exists():
        try:
            file_status = json.loads(STATUS_FILE.read_text())
            worker_status["progress"] = file_status.get("progress", {})
            worker_status["log_lines"] = file_status.get("log_lines", [])[-50:]
        except Exception:
            pass

    return worker_status


@app.post("/api/digest/stop")
async def api_digest_stop():
    """Stop digest → proxy to worker."""
    return _worker_post("/api/digest/stop")


@app.get("/api/digest/runs")
async def api_digest_runs(limit: int = Query(20)):
    """List runs → proxy to worker."""
    return _worker_get(f"/api/digest/runs?limit={limit}")


@app.get("/api/digest/runs/{run_id}")
async def api_digest_run_detail(run_id: str):
    """Run detail → proxy to worker."""
    return _worker_get(f"/api/digest/runs/{run_id}")


@app.get("/api/digest/config")
async def api_config_get():
    """Get config → proxy to worker."""
    return _worker_get("/api/digest/config")


@app.post("/api/digest/config")
async def api_config_update(updates: dict):
    """Update config → proxy to worker."""
    return _worker_post("/api/digest/config", None)


@app.get("/api/digest/dedup")
async def api_dedup_stats():
    """Dedup stats → proxy to worker."""
    return _worker_get("/api/digest/dedup")


@app.post("/api/graph/rebuild")
async def api_rebuild_graph(use_llm: bool = True, incremental: bool = False):
    """Trigger graph rebuild → proxy to worker (async)."""
    return _worker_post("/api/graph/rebuild", {"use_llm": use_llm, "incremental": incremental})


@app.get("/api/graph/status")
async def api_graph_status():
    """Poll graph build status → proxy to worker."""
    return _worker_get("/api/graph/status")


# ── Engine Proxy Routes ──────────────────────────────────────────

@app.post("/api/engine/scout")
async def api_engine_scout(request: Request):
    """Scout for new sources → proxy to worker."""
    body = None
    try:
        body = await request.json()
    except Exception:
        pass
    return _worker_post("/api/engine/scout", json_body=body)


@app.get("/api/engine/jobs")
async def api_engine_jobs():
    """List engine jobs → proxy to worker."""
    return _worker_get("/api/engine/jobs")


@app.get("/api/engine/jobs/{job_id}")
async def api_engine_job_detail(job_id: str):
    """Engine job detail → proxy to worker."""
    return _worker_get(f"/api/engine/jobs/{job_id}")


@app.get("/api/engine/status")
async def api_engine_status():
    """Poll engine status → proxy to worker."""
    return _worker_get("/api/engine/status")


# ══════════════════════════════════════════════════════════════════
#  PAGE ROUTE
# ══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    template = TEMPLATES_DIR / "index.html"
    if template.exists():
        return HTMLResponse(template.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Geo-Digest Dashboard</h1><p>Template not found.</p>")


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "3000"))
    print(f"Starting Geo-Digest Dashboard on http://0.0.0.0:{port}")
    print(f"Worker URL: {WORKER_URL}")
    print(f"Data dir: {DATA_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port)
