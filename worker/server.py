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
ENV_FILE = WORKER_DIR / ".env"

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
    """Update job state entry."""
    _jobs[job_type].update(kwargs)


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

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

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
                "Graph": ("building_graph", "Построение графа знаний..."),
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
    """Read latest lines from digest log and update status file."""
    try:
        lines = log_file.read_text(errors="ignore").splitlines()
        if not lines:
            return

        # Find last meaningful progress line
        current_step = "starting"
        msg = ""
        for line in reversed(lines[-50:]):
            stripped = line.strip()
            for kw, (step, step_msg) in keywords.items():
                if kw in stripped:
                    current_step = step
                    msg = step_msg if step_msg else stripped
                    break
            if current_step != "starting":
                break

        step_order = ["starting", "searching", "dedup", "enriching_oa",
                      "filtering", "scoring", "selecting", "enriching",
                      "building_graph", "delivering", "complete"]
        pct = 0
        if current_step in step_order:
            pct = int((step_order.index(current_step) + 1) / len(step_order) * 100)

        # Write status file (shared with dashboard)
        STATUS_FILE.write_text(json.dumps({
            "status": "running" if _jobs["digest"]["status"] == "running" else _jobs["digest"]["status"],
            "run_id": _jobs["digest"]["job_id"],
            "started_at": _jobs["digest"]["started_at"],
            "progress": {"step": current_step, "message": msg, "pct": pct},
            "log_lines": lines[-50:],
        }, ensure_ascii=False, indent=2))

    except Exception:
        pass


def _tail_file(path: Path, max_chars: int = 2000) -> str:
    """Return last N chars of a file."""
    try:
        text = path.read_text(errors="ignore")
        return text[-max_chars:] if len(text) > max_chars else text
    except Exception:
        return ""


# ── Graph Build ─────────────────────────────────────────────────

def _run_graph_bg(job_id: str, use_llm: bool, incremental: bool):
    """Run build_graph.py as background process."""
    _update_job("graph", status="running", started_at=datetime.now(timezone.utc).isoformat())

    cmd = [sys.executable, str(SCRIPTS_DIR / "build_graph.py")]
    if not use_llm:
        cmd.append("--no-llm")
    if incremental:
        cmd.append("--update")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(WORKER_DIR),
            env=env,
            timeout=900,
        )
        _update_job("graph",
            status="done" if result.returncode == 0 else "error",
            finished_at=datetime.now(timezone.utc).isoformat(),
            stdout=result.stdout[-3000:] if result.stdout else "",
            stderr=result.stderr[-500:] if result.stderr else "",
        )

    except subprocess.TimeoutExpired:
        _update_job("graph", status="error", stderr="Timeout (900s)",
                    finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as e:
        _update_job("graph", status="error", stderr=str(e),
                    finished_at=datetime.now(timezone.utc).isoformat())


# ── Telegram Bot (background thread) ────────────────────────────
_tg_bot_thread = None


def start_tg_bot():
    """Start Telegram bot polling in background thread."""
    global _tg_bot_thread
    if _tg_bot_thread and _tg_bot_thread.is_alive():
        return False

    def _run():
        env = {**os.environ}
        if ENV_FILE.exists():
            load_dotenv_result = _load_env(ENV_FILE)
            if load_dotenv_result:
                env.update(load_dotenv_result)
        subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "tg_bot.py"), "poll"],
            cwd=str(WORKER_DIR),
            env=env,
        )

    _tg_bot_thread = threading.Thread(target=_run, daemon=True)
    _tg_bot_thread.start()
    return True


def _load_env(env_file: Path) -> dict | None:
    """Load .env file into dict."""
    try:
        env_vars = {}
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env_vars[k.strip()] = v.strip().strip('"').strip("'")
        return env_vars
    except Exception:
        return None


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


@app.on_event("startup")
async def on_startup():
    """Start TG bot on worker startup."""
    start_tg_bot()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)
