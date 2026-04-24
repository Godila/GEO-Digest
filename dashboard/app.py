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

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
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


def _worker_post(path: str, params: dict | None = None, timeout: int = 10) -> dict:
    """POST request to worker."""
    try:
        r = requests.post(f"{WORKER_URL}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        raise HTTPException(status_code=503, detail="Worker unavailable")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Worker error: {e}")


# ── Data loaders (read from shared volume) ─────────────────────
def load_articles() -> list[dict]:
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
#  DATA ENDPOINTS (read from shared volume, no worker needed)
# ══════════════════════════════════════════════════════════════════

@app.get("/api/articles")
async def api_articles(
    sort_by: str = Query("total_score"),
    order: str = Query("desc"),
    topic: str = Query(""),
    article_type: str = Query(""),
    source: str = Query(""),
    limit: int = Query(50),
    offset: int = Query(0),
):
    articles = load_articles()

    for art in articles:
        scores = art.get("scores", {})
        art["_total_score"] = round(sum(scores.values()), 2) if scores else 0

    if topic:
        articles = [a for a in articles if a.get("_topic_key") == topic]
    if article_type:
        articles = [a for a in articles if a.get("article_type") == article_type]
    if source:
        articles = [a for a in articles if a.get("source") == source]

    reverse = order == "desc"
    valid_fields = {
        "_total_score", "year", "citations", "_saved_at",
        "score_transferability", "score_geographic",
        "score_thematic", "score_publication",
    }
    sort_field = sort_by if sort_by in valid_fields else "_total_score"
    articles.sort(key=lambda a: a.get(sort_field, 0), reverse=reverse)

    total = len(articles)
    articles = articles[offset : offset + limit]

    return {"total": total, "offset": offset, "limit": limit, "articles": articles}


@app.get("/api/articles/{article_id}")
async def api_article_detail(article_id: int):
    articles = load_articles()
    if article_id < 1 or article_id > len(articles):
        raise HTTPException(status_code=404)
    return articles[article_id - 1]


@app.get("/api/graph")
async def api_graph():
    return load_graph()


@app.get("/api/topics")
async def api_topics():
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
    articles = load_articles()
    graph = load_graph()

    total_scores = []
    for art in articles:
        scores = art.get("scores", {})
        total_scores.append(sum(scores.values()) if scores else 0)

    sources, types = {}, {}
    for art in articles:
        sources[art.get("source", "unknown")] = sources.get(art.get("source", "unknown"), 0) + 1
        types[art.get("article_type", "unknown")] = types.get(art.get("article_type", "unknown"), 0) + 1

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
async def api_digest_start():
    """Start digest pipeline → proxy to worker."""
    return _worker_post("/api/digest/run")


@app.get("/api/digest/status")
async def api_digest_status():
    """Poll digest status → proxy to worker + merge file status."""
    worker_status = _worker_get("/api/digest/status")

    # Also read live status file (written by worker's log poller)
    if STATUS_FILE.exists():
        try:
            file_status = json.loads(STATUS_FILE.read_text())
            worker_status["progress"] = file_status.get("progress", {})
            worker_status["log_lines"] = file_status.get("log_lines", [])[-30:]
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
