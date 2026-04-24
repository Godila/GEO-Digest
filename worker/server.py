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

from fastapi import FastAPI, HTTPException

# ── Add scripts to path for run_manager import ────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# ── Paths ───────────────────────────────────────────────────────
# Worker runs scripts from /app/scripts, data lives in /app/data
WORKER_DIR = Path(__file__).resolve().parent.parent  # /app or project root
SCRIPTS_DIR = WORKER_DIR / "scripts"
DATA_DIR = WORKER_DIR / "data"

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

            _update_job("digest",
                status="done" if returncode == 0 else "error",
                finished_at=datetime.now(timezone.utc).isoformat(),
                stdout=_tail_file(log_file, 2000),
            )

    except Exception as e:
        _update_job("digest", status="error", stderr=str(e),
                    finished_at=datetime.now(timezone.utc).isoformat())


def _poll_digest_log(log_file: Path, keywords: dict):
    """Backward-compatible wrapper for digest log polling."""
    _poll_job_log(log_file, keywords, job_type="digest")


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

            # Extract last N meaningful lines as log_lines
            log_lines = [l.strip() for l in lines[-30:] if l.strip()]

            _update_job(job_type,
                progress={"step": step, "message": msg, "pct": pct},
                log_lines=log_lines[-15:],
            )
    except Exception:
        pass  # don't crash the poller


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
