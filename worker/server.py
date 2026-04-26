#!/usr/bin/env python3
"""
GEO-DIGEST WORKER — Compute service for digest pipeline & graph building.

Runs as separate container. Dashboard calls this for heavy operations.

API:
  POST /api/digest/run       Start digest pipeline (async)
  GET  /api/digest/status    Poll digest progress
  POST /api/graph/rebuild    Start graph build (async)
  GET  /api/graph/status     Poll graph progress
  GET  /api/health           Health check

Data shared with dashboard via /app/data volume:
  articles.jsonl, graph_data.json, seen_dois.txt, run_status.json
"""

import json
import os
import sys
import subprocess
import threading
import uuid
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from typing import Optional

# ── Add scripts to path for run_manager import ────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ── Paths ───────────────────────────────────────────────────────
# Worker runs scripts from /app/scripts, data lives in /app/data
# In Docker: source code mounted at /app/src (see docker-compose.yml)
_FILE_DIR = Path(__file__).resolve().parent  # /app/worker
WORKER_DIR = _FILE_DIR.parent               # /app (or project root)
SCRIPTS_DIR = WORKER_DIR / "scripts"
DATA_DIR = WORKER_DIR / "data"

# Engine: try Docker mount path first, then local path
_ENGINE_CANDIDATES = [
    WORKER_DIR / "src" / "engine",   # Docker: .:/app/src → /app/src/engine
    _FILE_DIR.parent.parent / "engine",  # Local development
    Path(__file__).resolve().parent.parent / "engine",  # Alternative resolve
]
ENGINE_DIR = None
for _cand in _ENGINE_CANDIDATES:
    if (_cand / "__init__.py").exists() or (_cand / "config.py").exists():
        ENGINE_DIR = _cand
        break
if ENGINE_DIR is None:
    ENGINE_DIR = _ENGINE_CANDIDATES[0]  # fallback

# ── Engine imports ─────────────────────────────────────────────
# Lazy imports: engine code lives in /app/src/engine (Docker mount)
# and may not be available at import time. We resolve + import on first use.
_engine_imports_done = False

def _ensure_engine_imports():
    """Lazily import engine modules. Safe to call multiple times."""
    global _engine_imports_done
    if _engine_imports_done:
        return

    # Re-resolve ENGINE_DIR at call time (volume is mounted by now)
    global ENGINE_DIR
    _candidates = [
        WORKER_DIR / "src" / "engine",   # Docker: .:/app/src → /app/src/engine
        Path(__file__).resolve().parent.parent / "engine",
        WORKER_DIR / "engine",
    ]
    for _c in _candidates:
        if (_c / "config.py").exists():
            ENGINE_DIR = _c
            break

    if str(ENGINE_DIR) not in sys.path:
        # Add parent of engine to path so 'from engine.xxx import' works
        _engine_parent = ENGINE_DIR.parent
        if str(_engine_parent) not in sys.path:
            sys.path.insert(0, str(_engine_parent))

    # Load .env for API keys
    _env_path = WORKER_DIR / ".env"
    if _env_path.exists():
        for _line in _env_path.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                if _k.strip() not in os.environ:
                    os.environ[_k.strip()] = _v.strip()

    # Now import
    global get_config, ScoutAgent, JsonlStorage
    from engine.config import get_config as _gc
    from engine.agents.scout import ScoutAgent as _sa
    from engine.storage.jsonl_backend import JsonlStorage as _js
    get_config = _gc
    ScoutAgent = _sa
    JsonlStorage = _js
    _engine_imports_done = True

# Ensure data dir exists
DATA_DIR.mkdir(exist_ok=True)

# Scripts use these paths — symlink or configure them to DATA_DIR
ARTICLES_DB = DATA_DIR / "articles.jsonl"
GRAPH_DATA = DATA_DIR / "graph_data.json"
STATUS_FILE = DATA_DIR / "run_status.json"
RUNS_DIR = WORKER_DIR / "runs"
CONFIG_PATH = WORKER_DIR / "config.yaml"

RUNS_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="GEO-Digest Worker",
    description="Compute backend: digest pipeline + knowledge graph",
    version="1.0.0",
)


# ── Job State ───────────────────────────────────────────────────
_jobs = {
    "digest": {
        "job_id": None,
        "status": "idle",      # idle, running, done, error
        "started_at": None,
        "finished_at": None,
        "stdout": "",
        "stderr": "",
        "pid": None,
    },
    "graph": {
        "job_id": None,
        "status": "idle",
        "started_at": None,
        "finished_at": None,
        "stdout": "",
        "stderr": "",
        "pid": None,
    },
}

# ── Engine Job State ────────────────────────────────────────────
_engine_jobs: dict[str, dict] = {}  # {job_id: {status, started_at, finished_at, result, error, log_lines}}


def _update_job(job_type: str, **kwargs):
    """Update job state entry (in-memory + persist to file)."""
    _jobs[job_type].update(kwargs)
    # Persist full state to file so it survives container restarts
    try:
        status = dict(_jobs["digest"])
        status["graph"] = dict(_jobs["graph"])
        STATUS_FILE.write_text(json.dumps(status, indent=2, ensure_ascii=False))
    except Exception:
        pass


def _get_job(job_type: str) -> dict:
    """Get copy of job state."""
    return dict(_jobs[job_type])


# ── Digest Pipeline ────────────────────────────────────────────

def _run_digest_bg(job_id: str):
    """Run digest.py as background process, stream output to status file."""
    _update_job("digest", status="running", started_at=datetime.now(timezone.utc).isoformat())

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "digest.py"),
        "--config", str(CONFIG_PATH),
        "--run-id", job_id,
    ]

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "GEO_DATA_DIR": str(DATA_DIR)}

    try:
        log_file = RUNS_DIR / f"run_{job_id}.log"
        with open(log_file, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(WORKER_DIR),
                env=env,
            )
            _update_job("digest", pid=proc.pid)

            # Stream output for live status
            step_keywords = {
                "Loading seen DOIs": ("dedup", "Загрузка базы DOI..."),
                "Total raw results": ("searching", "Поиск завершён"),
                "After dedup": ("dedup", "Дедупликация..."),
                "Unpaywall": ("enriching_oa", "Проверка открытого доступа..."),
                "relevance gate": ("filtering", "Фильтр релевантности..."),
                "Scoring": ("scoring", "Скоринг статей..."),
                "LLM Enrichment": ("enriching", "LLM-обогащение..."),
                "[LLM] Enriching": ("enriching", None),
                "Selected:": ("selecting", "Отбор лучших статей"),
                "DIGEST COMPLETE": ("complete", "Дайджест готов!"),
                "Delivery": ("delivering", "Отправка в Telegram..."),
            }

            current_step = "starting"
            for line_raw in iter(proc.stdout.readline if hasattr(proc.stdout, 'readline') else lambda: "", ""):
                # When stdout is redirected to file, we poll the log instead
                break  # stdout is file, not pipe — use polling below

            # Wait for process, poll log file for progress
            import time
            while proc.poll() is None:
                time.sleep(3)
                _poll_digest_log(log_file, step_keywords)

            returncode = proc.wait()
            _poll_digest_log(log_file, step_keywords)  # final update

            status = "done" if returncode == 0 else "error"
            _update_job("digest",
                status=status,
                finished_at=datetime.now(timezone.utc).isoformat(),
                stdout=_tail_file(log_file, 2000),
            )

            # Save run summary for history (fix: was never called!)
            _save_digest_run_summary(job_id, log_file, status)

    except Exception as e:
        _update_job("digest", status="error", stderr=str(e),
                    finished_at=datetime.now(timezone.utc).isoformat())


def _poll_digest_log(log_file: Path, keywords: dict):
    """Backward-compatible wrapper for digest log polling."""
    _poll_job_log(log_file, keywords, job_type="digest")


def _save_digest_run_summary(job_id: str, log_file: Path, status: str):
    """Parse digest log and save run summary for history display.

    Extracts key metrics from log output (articles found/selected,
    duplicates skipped, sources used, topics searched) and writes
    run_{job_id}.json via run_manager.save_run_summary().
    """
    try:
        from run_manager import save_run_summary
        import re

        log_text = log_file.read_text(errors="ignore") if log_file.exists() else ""

        # Parse metrics from log lines
        articles_found = 0
        articles_selected = 0
        duplicates_skipped = 0
        sources_used = {}
        topics_searched = []

        for line in log_text.splitlines():
            # "Total raw results: N"
            m = re.search(r"Total raw results:\s*(\d+)", line)
            if m:
                articles_found = int(m.group(1))

            # "Selected: N articles"
            m = re.search(r"Selected:\s*(\d+)\s*articles?", line)
            if m:
                articles_selected = int(m.group(1))

            # "After dedup: N (skipped X)"
            m = re.search(r"After dedup:\s*\d+\s*\(skipped\s*(\d+)", line)
            if m:
                duplicates_skipped = int(m.group(1))

            # Source counts: "openalex: 42 results"
            m = re.search(r"\s+(\w[\w_]*):\s*(\d+)\s*results", line)
            if m:
                src_name = m.group(1)
                found = int(m.group(2))
                if src_name not in sources_used:
                    sources_used[src_name] = {"found": 0}
                sources_used[src_name]["found"] += found

            # Topic header: "[*] Topic: Name (N queries)"
            m = re.search(r"Topic:\s*(.+?)\s*\(", line)
            if m and "Topic:" in line:
                topics_searched.append(m.group(1).strip())

        # Calculate duration
        started_at = _get_job("digest").get("started_at")
        finished_at = _get_job("digest").get("finished_at")
        duration_sec = 0
        if started_at and finished_at:
            try:
                t0 = datetime.fromisoformat(started_at)
                t1 = datetime.fromisoformat(finished_at)
                duration_sec = int((t1 - t0).total_seconds())
            except Exception:
                pass

        run_data = {
            "id": job_id,
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration_sec,
            "articles_found": articles_found,
            "articles_selected": articles_selected,
            "duplicates_skipped": duplicates_skipped,
            "sources_used": sources_used,
            "topics_searched": topics_searched,
            "log_lines": len(log_text.splitlines()) if log_text else 0,
        }

        save_run_summary(run_data)
        print(f"[run_history] Saved summary for {job_id}: "
              f"found={articles_found} selected={articles_selected} "
              f"dups={duplicates_skipped} status={status}")

    except Exception as e:
        print(f"[run_history] WARNING: Failed to save run summary: {e}", file=sys.stderr)


def _tail_file(path: Path, max_chars: int = 2000) -> str:
    """Return last N chars of a file."""
    try:
        text = path.read_text(errors="ignore")
        return text[-max_chars:] if len(text) > max_chars else text
    except Exception:
        return ""


# ── Graph Build ─────────────────────────────────────────────────

def _run_graph_bg(job_id: str, use_llm: bool, incremental: bool):
    """Run build_graph.py as background process with live log streaming."""
    _update_job("graph", status="running", started_at=datetime.now(timezone.utc).isoformat())

    cmd = [sys.executable, str(SCRIPTS_DIR / "build_graph.py")]
    if not use_llm:
        cmd.append("--no-llm")
    if incremental:
        cmd.append("--update")

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "GEO_DATA_DIR": str(DATA_DIR)}

    # Graph step detection keywords
    graph_steps = {
        "[GRAPH_STEP] loading":      ("loading",     "Загрузка статей..."),
        "[GRAPH_STEP] metadata":     ("metadata",    "Построение метаданных..."),
        "[GRAPH_STEP] cooccurrence": ("cooccurrence","Топики co-occurrence..."),
        "[GRAPH_STEP] llm_semantic": ("llm_semantic","LLM семантика..."),
        "[GRAPH_STEP] saving":       ("saving",      "Сохранение графа..."),
        "[GRAPH_COMPLETE]":          ("complete",    "Граф построен!"),
    }

    try:
        log_file = RUNS_DIR / f"graph_{job_id}.log"
        with open(log_file, "w") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(WORKER_DIR),
                env=env,
            )
            _update_job("graph", pid=proc.pid)

            # Poll log file for live progress
            import time
            while proc.poll() is None:
                time.sleep(3)
                _poll_job_log(log_file, graph_steps, job_type="graph")

            returncode = proc.wait()
            _poll_job_log(log_file, graph_steps, job_type="graph")  # final update

            _update_job("graph",
                status="done" if returncode == 0 else "error",
                finished_at=datetime.now(timezone.utc).isoformat(),
                stdout=_tail_file(log_file, 3000),
            )

    except Exception as e:
        _update_job("graph", status="error", stderr=str(e),
                    finished_at=datetime.now(timezone.utc).isoformat())


def _poll_job_log(log_file: Path, keywords: dict, job_type: str = "digest"):
    """Read latest lines from log and update status file with progress."""
    try:
        lines = log_file.read_text(errors="ignore").splitlines()
        if not lines:
            return

        # Find last matching keyword
        last_match = None
        last_idx = -1
        for kw, (step, msg) in keywords.items():
            for i, line in enumerate(lines):
                if kw in line and i > last_idx:
                    last_match = (step, msg)
                    last_idx = i

        if last_match:
            step, msg = last_match
            pct_map = {
                "digest": {
                    "starting": 5, "searching": 15, "dedup": 30,
                    "enriching_oa": 40, "filtering": 55, "scoring": 65,
                    "enriching": 80, "selecting": 90, "complete": 100,
                    "building_graph": 95,
                },
                "graph": {
                    "loading": 10, "metadata": 35, "cooccurrence": 50,
                    "llm_semantic": 70, "saving": 95, "complete": 100,
                },
            }
            pct = pct_map.get(job_type, {}).get(step, 50)

            # Smart log extraction for UI:
            # - Take more lines (80 vs 30) for better context
            # - Filter out repetitive noise (Unpaywall 404s for arXiv)
            # - Keep meaningful lines: steps, results, topics, scores, errors
            raw_tail = [l.strip() for l in lines[-80:] if l.strip()]
            log_lines = _filter_log_lines(raw_tail, job_type)

            _update_job(job_type,
                progress={"step": step, "message": msg, "pct": pct},
                log_lines=log_lines[-40:],  # send up to 40 filtered lines
            )
    except Exception:
        pass  # don't crash the poller


def _filter_log_lines(lines: list[str], job_type: str = "digest") -> list[str]:
    """Filter log lines for UI display.

    Removes repetitive noise and keeps meaningful information:
    - Collapses consecutive identical error patterns into summaries
    - Always keeps step markers ([*], [STEP], topic headers)
    - Keeps result counts and key metrics
    - Limits noise to max 3 consecutive same-type errors
    """
    import re

    filtered = []
    consecutive_errors = 0
    prev_error_type = None
    error_summary_count = 0

    # Patterns to classify lines
    STEP_MARKERS = ("[*]", "[STEP]", "[GRAPH_STEP]", "Topic:", "GEO-ECOLOGY",
                    "Started:", "Total raw", "After dedup", "Selected:",
                    "relevance gate", "Scoring", "LLM Enrichment",
                    "[LLM]", "DIGEST COMPLETE", "Delivery", "FUNNEL")
    RESULT_PATTERNS = ("results", "articles", "saved", "scored",
                       "enriched", "selected", "skipped")

    for line in lines:
        # Always keep important markers
        if any(mk in line for mk in STEP_MARKERS):
            filtered.append(line)
            consecutive_errors = 0
            prev_error_type = None
            continue

        # Keep result/summary lines
        if any(rp in line.lower() for rp in RESULT_PATTERNS):
            filtered.append(line)
            consecutive_errors = 0
            prev_error_type = None
            continue

        # Classify error/noise lines
        is_unpaywall_404 = "Unpaywall error" in line and "404" in line
        is_rate_limit = "429" in line or "Too Many Requests" in line
        is_server_error = "500" in line and "Internal Server Error" in line
        is_generic_error = ("error" in line.lower() or "Error" in line
                           or "Traceback" in line)

        if is_unpaywall_404 or is_rate_limit or is_server_error:
            error_type = "unpaywall_404" if is_unpaywall_404 else \
                        "rate_limit" if is_rate_limit else "server_error"

            if error_type == prev_error_type:
                consecutive_errors += 1
                # Only show first 3 of each consecutive error type
                if consecutive_errors <= 3:
                    filtered.append(line)
                elif consecutive_errors == 4:
                    # Replace 4th+ with summary note
                    filtered.append(f"  ... (+{len(lines) - len(filtered)} more {error_type} messages suppressed)")
            else:
                # New error type — reset counter
                prev_error_type = error_type
                consecutive_errors = 1
                filtered.append(line)
        elif is_generic_error:
            # Keep genuine errors (tracebacks, real failures)
            filtered.append(line)
            consecutive_errors = 0
            prev_error_type = None
        else:
            # Regular info line — keep it
            filtered.append(line)
            consecutive_errors = 0
            prev_error_type = None

    return filtered


# ── API Endpoints ──────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "service": "geo-digest-worker",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "has_articles": ARTICLES_DB.exists(),
        "has_graph": GRAPH_DATA.exists(),
        "jobs": {k: {"status": v["status"], "pid": v.get("pid")} for k, v in _jobs.items()},
    }


@app.post("/api/digest/run")
async def digest_run():
    """Start digest pipeline (async, returns immediately)."""
    job = _get_job("digest")
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Digest already running")

    job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    thread = threading.Thread(target=_run_digest_bg, args=(job_id,), daemon=True)
    thread.start()

    _update_job("digest", job_id=job_id)

    return {"success": True, "job_id": job_id, "status": "running"}


@app.get("/api/digest/status")
async def digest_status():
    """Poll digest pipeline status."""
    job = _get_job("digest")
    result = dict(job)

    # Also read live status from file (written by digest log poller)
    if STATUS_FILE.exists():
        try:
            file_status = json.loads(STATUS_FILE.read_text())
            result["progress"] = file_status.get("progress", {})
        except Exception:
            pass

    return result


@app.post("/api/graph/rebuild")
async def graph_rebuild(use_llm: bool = True, incremental: bool = False):
    """Start graph build (async, returns immediately)."""
    job = _get_job("graph")
    if job["status"] == "running":
        raise HTTPException(status_code=409, detail="Graph build already in progress")

    job_id = uuid.uuid4().hex[:8]

    thread = threading.Thread(target=_run_graph_bg, args=(job_id, use_llm, incremental), daemon=True)
    thread.start()

    _update_job("graph", job_id=job_id)

    return {
        "success": True,
        "job_id": job_id,
        "status": "running",
        "message": f"Graph build started (llm={use_llm}, incremental={incremental})",
    }


@app.get("/api/graph/status")
async def graph_status():
    """Poll graph build status."""
    job = _get_job("graph")
    result = dict(job)

    # If done/error, attach graph metadata
    if job["status"] in ("done", "idle"):
        try:
            if GRAPH_DATA.exists():
                g = json.loads(GRAPH_DATA.read_text())
                result["graph_data"] = g.get("metadata", {})
        except Exception:
            pass

    return result


# ── Digest Management Endpoints (via run_manager) ────────────

@app.post("/api/digest/stop")
async def digest_stop():
    """Stop current running digest."""
    try:
        from run_manager import stop_run
        return stop_run()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/digest/runs")
async def digest_runs(limit: int = 20):
    """List historical digest runs."""
    try:
        from run_manager import list_runs
        return {"runs": list_runs(limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/digest/runs/{run_id}")
async def digest_run_detail(run_id: str):
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


@app.get("/api/digest/config")
async def config_get():
    """Get current digest configuration."""
    try:
        from run_manager import get_config
        cfg = get_config()
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
async def config_update(updates: dict):
    """Update digest configuration."""
    try:
        from run_manager import update_config
        updated = update_config(updates)
        return {"ok": True, "config": updated}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/digest/dedup")
async def dedup_stats():
    """Get deduplication statistics."""
    try:
        from run_manager import get_dedup_stats
        return get_dedup_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ════════════════════════════════════════════════════════════
#  AGENT-FACING API (/api/a/*)
#  Compact JSON for AI agents, not for browser UI.
#  All data access goes through DAL (worker/dal.py).
# ════════════════════════════════════════════════════════════

from .dal import (
    search_articles, search_with_ranking, get_article_by_id,
    get_article_enrichment_md, get_neighbors, find_path,
    get_subgraph, get_stats, get_topics, get_info, load_graph,
    load_all_articles, resolve_graph_id,
)


@app.get("/api/a/articles")
async def agent_articles(
    q: str = "",
    topic: str = "",
    source: str = "",
    min_score: float = 0,
    max_score: float = 10,
    min_year: int = 0,
    is_oa: Optional[bool] = None,
    sort_by: str = "score_desc",
    limit: int = 20,
    offset: int = 0,
    fields: str = "",
):
    """Search & filter articles with pagination and field projection.
    
    fields: comma-separated list of fields to include (e.g. "title,score_5,doi").
           Always includes _id. Empty = all fields.
    """
    field_list = [f.strip() for f in fields.split(",") if f.strip()] if fields else None
    result = search_articles(
        query=q, topic=topic, source=source,
        min_score=min_score, max_score=max_score, min_year=min_year,
        is_oa=is_oa, sort_by=sort_by, limit=limit, offset=offset,
        fields=field_list,
    )
    return result


@app.get("/api/a/article/{article_id:path}")
async def agent_article(article_id: str):
    """Get full article by canonical ID + enrichment markdown content."""
    art = get_article_by_id(article_id)
    if not art:
        raise HTTPException(status_code=404, detail=f"Article '{article_id}' not found")

    # Enrich with graph ID
    from .dal import _enrich_articles_with_graph_ids
    _enrich_articles_with_graph_ids([art])

    # Attach enrichment markdown
    md_content = get_article_enrichment_md(art)
    if md_content:
        art["_md_content"] = md_content

    return art


@app.get("/api/a/search")
async def agent_search(
    q: str = "",
    limit: int = 20,
    topic: str = "",
    source: str = "",
    min_score: float = 0,
    fields: str = "",
):
    """Text search with relevance ranking (bonus for title/abstract/topic matches)."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")

    extra = {}
    if topic:
        extra["topic"] = topic
    if source:
        extra["source"] = source
    if min_score > 0:
        extra["min_score"] = min_score
    if fields:
        extra["fields"] = [x.strip() for x in fields.split(",") if x.strip()]

    result = search_with_ranking(query=q, limit=limit, **extra)
    return result


# ── Graph endpoints ────────────────────────────────────────

@app.get("/api/a/graph/neighbors")
async def agent_graph_neighbors(id: str, depth: int = 1, edge_types: str = ""):
    """Get subgraph around a node (BFS).

    id: node ID — accepts both canonical (doi:.../hash:...) and graph (article_1) IDs
    depth: hop count (1 = direct neighbors only)
    edge_types: comma-separated relation types to filter (empty = all)
    """
    # Auto-resolve canonical ID → graph ID
    graph_id = resolve_graph_id(id) or id
    et_list = [t.strip() for t in edge_types.split(",") if t.strip()] if edge_types else None
    result = get_neighbors(node_id=graph_id, depth=depth, edge_types=et_list)
    return result


@app.get("/api/a/graph/path")
async def agent_graph_path(from_id: str, to_id: str, max_depth: int = 4):
    """Find shortest path between two nodes (BFS).

    Accepts both canonical and graph node IDs.
    Returns hops with relations and confidence scores.
    """
    from_g = resolve_graph_id(from_id) or from_id
    to_g = resolve_graph_id(to_id) or to_id
    result = find_path(from_id=from_g, to_id=to_g, max_depth=max_depth)
    if not result["found"]:
        raise HTTPException(status_code=404, detail=result.get("error", "no path found"))
    return result


@app.get("/api/a/graph/subgraph")
async def agent_graph_subgraph(topic: str = "", node_type: str = "", min_score: float = 0):
    """Get subgraph with filters by topic, node type, or minimum score."""
    result = get_subgraph(topic=topic, node_type=node_type, min_score=min_score)
    return result


@app.get("/api/a/graph/nodes")
async def agent_graph_nodes(node_type: str = "", limit: int = 50):
    """List all graph nodes with their IDs and labels.
    
    Use this to discover valid node IDs for neighbors/path endpoints.
    """
    g = load_graph()
    nodes = []
    for n in g.get("nodes", []):
        d = n["data"]
        if node_type and d.get("nodeType") != node_type:
            continue
        nodes.append({
            "id": d.get("id"),
            "type": d.get("nodeType"),
            "label": d.get("label", ""),
            "score": d.get("score"),
        })
    return {"count": len(nodes), "nodes": nodes[:limit]}


# ── Stats & Meta ───────────────────────────────────────────

@app.get("/api/a/stats")
async def agent_stats():
    """Aggregated system statistics for agent consumption."""
    return get_stats()


@app.get("/api/a/topics")
async def agent_topics():
    """List all topics with article counts."""
    return {"topics": get_topics(), "count": len(get_topics())}


@app.get("/api/a/info")
async def agent_info():
    """System metadata: version, data format, available endpoints, counts."""
    return get_info()


# ── Export ─────────────────────────────────────────────────

@app.get("/api/a/export")
async def agent_export(format: str = "compact"):
    """Export data in various formats.

    format=compact:   articles list with essential fields only (for context window)
    format=jsonld:     JSON-LD with @context
    format=graph_kg:   Graph in KG format (not Cytoscape), nodes + edges as objects
    """
    fmt_clean = format.lower().strip()

    if fmt_clean == "compact":
        arts = load_all_articles()
        compact = []
        for a in arts:
            s = a.get("scores", {})
            compact.append({
                "_id": a.get("_id"),
                "title": a.get("title"),
                "title_ru": a.get("title_ru"),
                "year": a.get("year"),
                "authors": a.get("authors"),
                "journal": a.get("journal"),
                "score": round(s.get("total_5", s.get("total", 0)), 2),
                "topics": a.get("topics_ru", []) or a.get("topics", []),
                "abstract": (a.get("abstract") or "")[:300],
                "llm_summary": (a.get("llm_summary") or "")[:200],
                "doi": a.get("doi"),
                "source": a.get("source"),
            })
        return {"format": "compact", "count": len(compact), "articles": compact}

    elif fmt_clean == "jsonld":
        g = load_graph()
        arts = load_all_articles()
        ctx = {
            "@vocab": "http://geo-digest.org/",
            "doi": "http://dx.doi.org/",
        }
        ld_nodes = []
        for n in g.get("nodes", []):
            d = dict(n["data"])
            d["@type"] = d.pop("nodeType", "Node").capitalize()
            ld_nodes.append(d)

        ld_edges = []
        for e in g.get("edges", []):
            d = dict(e["data"])
            ld_edges.append({
                "@id": f"edge:{d.get('source')}->{d.get('target')}",
                "source": d.pop("source", ""),
                "target": d.pop("target", ""),
                "type": d.pop("relation", "related_to"),
                **d,
            })

        return {
            "@context": ctx,
            "articles_count": len(arts),
            "nodes": ld_nodes,
            "edges": ld_edges,
        }

    elif fmt_clean == "graph_kg":
        g = load_graph()
        kg_nodes = []
        type_map = {}  # track types for @type prefix
        for n in g.get("nodes", []):
            d = n["data"]
            ntype = d.get("nodeType", "unknown")
            nid = d.get("id", "")
            prefix = "article" if ntype == "article" else ("topic" if ntype == "topic" else "node")
            kg_nodes.append({
                "id": f"{prefix}:{nid}",
                "type": ntype,
                "label": d.get("label", ""),
                **{k: v for k, v in d.items() if k not in ("id", "label", "nodeType")},
            })

        kg_edges = []
        for e in g.get("edges", []):
            d = e["data"]
            kg_edges.append({
                "source": d.get("source", ""),
                "target": d.get("target", ""),
                "relation": d.get("relation", "related_to"),
                "confidence": d.get("confidence", 1.0),
            })

        return {
            "@context": {"@vocab": "http://geo-digest.org/"},
            "node_count": len(kg_nodes),
            "edge_count": len(kg_edges),
            "nodes": kg_nodes,
            "edges": kg_edges,
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown format '{fmt_clean}'. Supported: compact, jsonld, graph_kg"
        )


# ── Engine API ──────────────────────────────────────────────────

def _serialize_scout_result(result):
    """Serialize ScoutResult groups for JSON response."""
    if result is None:
        return None
    groups = []
    for g in (result.groups or []):
        articles = []
        for a in (g.articles or []):
            if hasattr(a, "_data"):
                d = dict(a._data)
            elif isinstance(a, dict):
                d = a
            else:
                d = {}
            articles.append({
                "title": d.get("title_ru") or d.get("title", ""),
                "doi": d.get("doi", ""),
                "year": d.get("year"),
                "citations": d.get("citations", 0),
            })
        gt = g.group_type.value if hasattr(g.group_type, "value") else str(g.group_type)
        groups.append({
            "group_id": g.group_id,
            "group_type": gt,
            "confidence": g.confidence,
            "keywords": g.keywords or [],
            "articles": articles,
        })
    return {
        "topic": getattr(result, "topic", ""),
        "total_found": getattr(result, "total_found", 0),
        "after_dedup": getattr(result, "after_dedup", 0),
        "groups": groups,
    }


def _run_scout_bg(job_id: str, topic: str, mode: str, max_articles: int):
    """Run ScoutAgent in background thread."""
    job = _engine_jobs[job_id]
    job["status"] = "running"
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    job["log_lines"] = []

    def _log(msg):
        ts = datetime.now(timezone.utc).isoformat()
        line = f"[{ts}] {msg}"
        job["log_lines"].append(line)

    try:
        config = get_config()
        storage = JsonlStorage(data_dir=str(DATA_DIR))
        agent = ScoutAgent(config=config, storage=storage)
        _log(f"ScoutAgent created, running with topic={topic}, mode={mode}, max_articles={max_articles}")
        agent_result = agent.run(topic=topic, mode=mode, max_articles=max_articles)
        if agent_result.success and agent_result.data:
            job["result"] = _serialize_scout_result(agent_result.data)
            job["status"] = "done"
            _log(f"Scout completed successfully: {len(agent_result.data.groups)} groups")
        else:
            job["error"] = agent_result.error or "Scout returned no results"
            job["status"] = "error"
            _log(f"Scout failed: {job['error']}")
    except Exception as e:
        job["error"] = str(e)
        job["status"] = "error"
        _log(f"Scout exception: {e}")
    finally:
        job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/engine/scout")
async def engine_scout(request: Request):
    """Run ScoutAgent with fresh search. Returns immediately with job_id.
    Accepts both JSON body and query params."""
    _ensure_engine_imports()

    # Try JSON body first, then query params
    try:
        body = await request.json()
        topic = body.get("topic", "")
        mode = body.get("mode", "fresh")
        max_articles = body.get("max_articles", 10)
    except Exception:
        topic = request.query_params.get("topic", "")
        mode = request.query_params.get("mode", "fresh")
        try:
            max_articles = int(request.query_params.get("max_articles", "10"))
        except (ValueError, TypeError):
            max_articles = 10

    if not topic or not topic.strip():
        raise HTTPException(status_code=422, detail="topic is required")

    job_id = uuid.uuid4().hex[:12]

    _engine_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
        "log_lines": [],
        "params": {"topic": topic.strip(), "mode": mode, "max_articles": max_articles},
    }

    thread = threading.Thread(
        target=_run_scout_bg,
        args=(job_id, topic.strip(), mode, max_articles),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "running"}


@app.get("/api/engine/jobs")
async def engine_list_jobs():
    """List all engine jobs."""
    _ensure_engine_imports()
    return {
        "jobs": [
            {
                "job_id": jid,
                "status": j["status"],
                "started_at": j.get("started_at"),
                "finished_at": j.get("finished_at"),
                "params": j.get("params", {}),
            }
            for jid, j in _engine_jobs.items()
        ],
        "total": len(_engine_jobs),
    }


@app.get("/api/engine/jobs/{job_id}")
async def engine_get_job(job_id: str):
    """Get engine job status + results."""
    _ensure_engine_imports()
    if job_id not in _engine_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = _engine_jobs[job_id]
    resp = {
        "job_id": job["job_id"],
        "status": job["status"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "params": job.get("params", {}),
        "log_lines": job.get("log_lines", []),
    }
    if job["result"]:
        resp["scout_result"] = job["result"]
    if job["error"]:
        resp["error"] = job["error"]
    return resp


@app.delete("/api/engine/jobs/{job_id}")
async def engine_cancel_job(job_id: str):
    """Cancel a running job."""
    _ensure_engine_imports()
    if job_id not in _engine_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job = _engine_jobs[job_id]
    if job["status"] in ("done", "error", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Job already {job['status']}")
    job["status"] = "cancelled"
    job["finished_at"] = datetime.now(timezone.utc).isoformat()
    ts = datetime.now(timezone.utc).isoformat()
    job["log_lines"].append(f"[{ts}] Job cancelled by user")
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/api/engine/jobs/{job_id}/logs")
async def engine_job_logs_stream(job_id: str):
    """SSE stream of job log lines (real-time)."""
    from fastapi.responses import StreamingResponse
    import asyncio

    _ensure_engine_imports()
    if job_id not in _engine_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    async def event_generator():
        last_idx = 0
        while True:
            job = _engine_jobs.get(job_id)
            if not job:
                yield f"data: {{'done':true}}\n\n"
                break
            lines = job.get("log_lines", [])
            # Send new lines since last check
            while last_idx < len(lines):
                import json as _j
                line = lines[last_idx]
                yield f"data: {_j.dumps({'line': line})}\n\n"
                last_idx += 1
            # Check if job is terminal
            if job["status"] in ("done", "error", "cancelled"):
                yield f"data: {{'done':true,'status':'{job['status']}'}}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/engine/status")
async def engine_status():
    """Engine health + storage stats."""
    _ensure_engine_imports()
    try:
        config = get_config()
        engine_ok = True
        engine_error = None
    except Exception as e:
        engine_ok = False
        engine_error = str(e)
        config = None

    # Storage stats
    storage_stats = {}
    try:
        storage = JsonlStorage(data_dir=str(DATA_DIR))
        all_articles = storage.load_all_articles()
        storage_stats = {
            "article_count": len(all_articles),
            "storage_path": str(DATA_DIR),
            "storage_exists": DATA_DIR.exists(),
        }
    except Exception as e:
        storage_stats = {"error": str(e), "storage_path": str(DATA_DIR)}

    return {
        "engine": {
            "ok": engine_ok,
            "error": engine_error,
            "config_loaded": config is not None,
        },
        "storage": storage_stats,
        "jobs": {
            "active": sum(1 for j in _engine_jobs.values() if j["status"] == "running"),
            "total": len(_engine_jobs),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
