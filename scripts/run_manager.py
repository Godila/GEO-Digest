#!/usr/bin/env python3
"""
Geo-Digest Run Manager — manages digest pipeline execution with status tracking.

Provides:
  - Background digest execution with live status reporting
  - Run history with summaries
  - Deduplication statistics
  - Configuration management

Status lifecycle: idle → searching → dedup → scoring → enriching → building_graph → complete|error

Usage:
  python3 scripts/run_manager.py --config path/to/config.yaml [--articles 4] [--topics seismology,geophysics_methods]
  (or import as module from app.py)
"""

import json
import os
import sys
import time
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
# Auto-detect base dir: works both natively and in Docker (.:/app)
_file_dir = Path(__file__).resolve().parent
# If run_manager is in scripts/, parent is project root; if in root, use directly
if _file_dir.name == "scripts":
    _BASE = _file_dir.parent
else:
    _BASE = _file_dir

# Fallback to home dir if auto-detection doesn't find config.yaml
if not (_BASE / "config.yaml").exists():
    _BASE = Path(os.path.expanduser("~/.hermes/geo_digest"))

BASE = _BASE
RUNS_DIR = BASE / "runs"
STATUS_FILE = BASE / "run_status.json"
CONFIG_PATH = BASE / "config.yaml"
ARTICLES_DB = BASE / "articles.jsonl"
SEEN_DOIS = BASE / "seen_dois.txt"

RUNS_DIR.mkdir(exist_ok=True)


# ── Status Management ─────────────────────────────────────────
def get_status() -> dict:
    """Read current run status."""
    if STATUS_FILE.exists():
        try:
            return json.loads(STATUS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"status": "idle", "runs": []}


def set_status(update: dict):
    """Update run status (merge)."""
    current = get_status()
    current.update(update)
    # Keep runs history separate
    if "runs" not in current:
        current["runs"] = []
    STATUS_FILE.write_text(json.dumps(current, ensure_ascii=False, indent=2))


def reset_status():
    """Reset to idle state, preserving history."""
    current = get_status()
    runs = current.get("runs", [])
    STATUS_FILE.write_text(json.dumps({"status": "idle", "runs": runs}, ensure_ascii=False, indent=2))


# ── Run History ────────────────────────────────────────────────
def list_runs(limit: int = 20) -> list[dict]:
    """List historical runs with summaries."""
    runs = []
    for f in sorted(RUNS_DIR.glob("run_*.json"), reverse=True)[:limit]:
        try:
            data = json.loads(f.read_text())
            runs.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return runs


def get_run(run_id: str) -> dict | None:
    """Get single run details including log."""
    run_file = RUNS_DIR / f"run_{run_id}.json"
    log_file = RUNS_DIR / f"run_{run_id}.log"
    if not run_file.exists():
        return None
    data = json.loads(run_file.read_text())
    # Attach log content
    if log_file.exists():
        data["log"] = log_file.read_text()
    else:
        data["log"] = ""
    return data


def save_run_summary(run_data: dict):
    """Save completed run summary."""
    run_id = run_data.get("id", "")
    run_file = RUNS_DIR / f"run_{run_id}.json"
    run_file.write_text(json.dumps(run_data, ensure_ascii=False, indent=2))

    # Update runs list in status
    status = get_status()
    runs = status.get("runs", [])
    # Add or update
    existing = next((r for r in runs if r.get("id") == run_id), None)
    summary = {
        "id": run_id,
        "started_at": run_data.get("started_at"),
        "finished_at": run_data.get("finished_at"),
        "status": run_data.get("status", "unknown"),
        "articles_found": run_data.get("articles_found", 0),
        "articles_selected": run_data.get("articles_selected", 0),
        "duplicates_skipped": run_data.get("duplicates_skipped", 0),
        "sources_used": run_data.get("sources_used", {}),
        "topics_searched": run_data.get("topics_searched", []),
        "duration_sec": run_data.get("duration_sec", 0),
        "config": run_data.get("config", {}),
    }
    if existing:
        runs[runs.index(existing)] = summary
    else:
        runs.insert(0, summary)
    # Keep last 50
    runs = runs[:50]
    status["runs"] = runs
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2))


# ── Config ─────────────────────────────────────────────────────
def get_config() -> dict:
    """Load current config.yaml."""
    import yaml
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def update_config(updates: dict) -> dict:
    """Update config.yaml with new values (partial merge)."""
    import yaml
    config = get_config()

    # Deep merge helper
    def deep_merge(base, override):
        result = base.copy()
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(result.get(k), dict):
                result[k] = deep_merge(result[k], v)
            else:
                result[k] = v
        return result

    config = deep_merge(config, updates)

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return config


# ── Dedup Statistics ───────────────────────────────────────────
def get_dedup_stats() -> dict:
    """Get deduplication statistics."""
    seen = set()
    if SEEN_DOIS.exists():
        seen = {line.strip() for line in SEEN_DOIS.read_text().splitlines() if line.strip()}

    articles = []
    if ARTICLES_DB.exists():
        with open(ARTICLES_DB, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        articles.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    db_dois = set()
    db_titles = {}
    for art in articles:
        doi = art.get("doi", "")
        if doi:
            db_dois.add(doi)
        title = art.get("title", "")
        year = str(art.get("year", ""))
        if title:
            key = f"{title}|{year}"
            db_titles[key] = art.get("_saved_at", "")

    # How many DB articles are in seen?
    in_seen = db_dois & seen
    not_in_seen = db_dois - seen

    return {
        "total_seen_dois": len(seen),
        "total_db_articles": len(articles),
        "db_articles_with_doi": len(db_dois),
        "db_articles_in_seen_list": len(in_seen),
        "db_articles_not_in_seen": len(not_in_seen),
        "seen_dois_sample": sorted(list(seen))[:20],
        "db_dois_sample": sorted(list(not_in_seen))[:10],
    }


# ── Digest Runner (background execution) ───────────────────────
_current_process: subprocess.Popen | None = None
_run_thread: threading.Thread | None = None


def start_digest_run(config_overrides: dict = None) -> dict:
    """
    Start a digest pipeline run in background.
    Returns initial status immediately.
    """
    global _current_process, _run_thread

    # Check if already running
    status = get_status()
    if status.get("status") not in ("idle", "complete", "error"):
        return {"ok": False, "error": f"Уже выполняется: {status.get('status')}"}

    # Generate run ID
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = RUNS_DIR / f"run_{run_id}.log"

    # Prepare config overrides as env or temp config
    final_config = get_config()
    if config_overrides:
        import yaml
        def deep_merge(base, override):
            result = base.copy()
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(result.get(k), dict):
                    result[k] = deep_merge(result[k], v)
                else:
                    result[k] = v
            return result
        final_config = deep_merge(final_config, config_overrides)

    # Write temp config with overrides
    temp_config = RUNS_DIR / f"config_{run_id}.yaml"
    with open(temp_config, "w") as f:
        yaml.dump(final_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # Set initial status
    set_status({
        "status": "starting",
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "articles_per_run": final_config.get("digest", {}).get("articles_per_run", 4),
            "min_year": final_config.get("digest", {}).get("min_year", 2023),
            "topics": list(final_config.get("topics", {}).keys()),
            "sources": {k: v.get("enabled", True) for k, v in final_config.get("sources", {}).items()},
            "scoring": final_config.get("scoring", {}),
        },
        "progress": {"step": "", "message": "Инициализация...", "pct": 0},
        "log_lines": [],
    })

    def run_pipeline():
        global _current_process
        try:
            # Open log file for writing
            with open(log_file, "w") as log_f:
                log_f.write(f"[{datetime.now(timezone.utc).isoformat()}] Starting digest pipeline...\n")
                log_f.flush()

                # Build command — use temp config
                cmd = [
                    sys.executable,
                    str(BASE / "scripts" / "digest.py"),
                    "--config", str(temp_config),
                    "--run-id", run_id,
                ]

                _current_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(BASE),
                    bufsize=1,  # Line buffered
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},  # Force unbuffered output from child
                )

                # Stream output to both log file and status
                current_step = "starting"  # init for sticky progress
                step_keywords = {
                    "Loading seen DOIs": ("dedup", "Загрузка базы DOI..."),
                    "Total raw results": ("searching", "Поиск завершён"),
                    "After dedup": ("dedup", "Дедупликация..."),
                    "Unpaywall": ("enriching_oa", "Проверка открытого доступа..."),
                    "relevance gate": ("filtering", "Фильтр релевантности..."),
                    "Scoring": ("scoring", "Скоринг статей..."),
                    "LLM Enrichment": ("enriching", "LLM-обогащение..."),
                    "[LLM] Enriching": ("enriching", None),  # per-article progress — keep step, update msg
                    "Selected:": ("selecting", "Отбор лучших статей"),
                    "DIGEST COMPLETE": ("complete", "Дайджест готов!"),
                    "Graph": ("building_graph", "Построение графа знаний..."),
                    "Delivery": ("delivering", "Отправка в Telegram..."),
                    "skip": ("warning", ""),
                    "Error": ("error", "Ошибка!"),
                }

                for line_raw in iter(_current_process.stdout.readline, ""):
                    line = line_raw.rstrip("\n")
                    log_f.write(line + "\n")
                    log_f.flush()

                    # Detect progress step
                    # Sticky progress: keep last known step instead of resetting to "running"/0%
                    msg = line.strip()
                    matched_step = None
                    matched_msg = ""

                    for keyword, (step_name, step_msg) in step_keywords.items():
                        if keyword in line:
                            matched_step = step_name
                            matched_msg = step_msg if step_msg else msg
                            break

                    if matched_step:
                        current_step = matched_step
                        msg = matched_msg

                    # Estimate percentage based on current (possibly sticky) step
                    step_order = ["starting", "searching", "dedup", "enriching_oa",
                                  "filtering", "scoring", "selecting", "enriching",
                                  "building_graph", "delivering", "complete"]
                    if current_step in step_order:
                        pct = int((step_order.index(current_step) + 1) / len(step_order) * 100)
                    # else: keep previous pct value (sticky)

                    # Update status file (every line — for live log + progress)
                    status = get_status()
                    logs = status.get("log_lines", [])
                    logs.append(line)
                    logs = logs[-50:]

                    set_status({
                        "progress": {"step": current_step, "message": msg, "pct": pct},
                        "log_lines": logs,
                    })

                _current_process.wait()
                returncode = _current_process.returncode

                log_f.write(f"[{datetime.now(timezone.utc).isoformat()}] Process exited with code {returncode}\n")
                log_f.flush()

            # Parse results from digest output
            finished_at = datetime.now(timezone.utc).isoformat()

            # Try to extract stats from log
            run_summary = parse_run_log(log_file, run_id, returncode)
            run_summary["finished_at"] = finished_at
            run_summary["status"] = "complete" if returncode == 0 else "error"
            run_summary["returncode"] = returncode
            run_summary["config"] = status.get("config", {})

            save_run_summary(run_summary)

            # Final status
            set_status({
                "status": "complete" if returncode == 0 else "error",
                "finished_at": finished_at,
                "progress": {
                    "step": "complete" if returncode == 0 else "error",
                    "message": "Готово!" if returncode == 0 else f"Ошибка (код {returncode})",
                    "pct": 100,
                },
                "result": run_summary,
            })

        except Exception as e:
            set_status({
                "status": "error",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "progress": {"step": "error", "message": str(e), "pct": 0},
            })
            with open(log_file, "a") as log_f:
                log_f.write(f"[ERROR] {e}\n")
        finally:
            _current_process = None
            # Cleanup temp config
            if temp_config.exists():
                temp_config.unlink(missing_ok=True)

    _run_thread = threading.Thread(target=run_pipeline, daemon=True)
    _run_thread.start()

    return {"ok": True, "run_id": run_id}


def stop_run() -> dict:
    """Stop current running digest."""
    global _current_process
    if _current_process and _current_process.poll() is None:
        _current_process.terminate()
        set_status({
            "status": "stopped",
            "progress": {"step": "stopped", "message": "Остановлено пользователем", "pct": 0},
        })
        return {"ok": True, "message": "Останавливаем..."}
    return {"ok": False, "error": "Нет активного запуска"}


def parse_run_log(log_path: Path, run_id: str, returncode: int) -> dict:
    """Parse digest log file to extract run statistics."""
    text = log_path.read_text() if log_path.exists() else ""

    stats = {
        "id": run_id,
        "started_at": "",
        "articles_found": 0,
        "articles_selected": 0,
        "duplicates_skipped": 0,
        "raw_results": 0,
        "after_dedup": 0,
        "after_gate": 0,
        "oa_enriched": 0,
        "llm_enriched": 0,
        "graph_nodes": 0,
        "graph_edges": 0,
        "sources_used": {},
        "topics_searched": [],
        "errors": [],
        "warnings": [],
    }

    for line in text.split("\n"):
        # Extract numbers from known patterns
        if "Total raw results:" in line:
            try:
                stats["raw_results"] = int(line.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        elif "After dedup:" in line:
            try:
                stats["after_dedup"] = int(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "After relevance gate:" in line:
            try:
                parts = line.split(":")
                stats["after_gate"] = int(parts[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "Unpaywall enriched:" in line:
            try:
                stats["oa_enriched"] = int(line.split(":")[-1].split("/")[0].strip())
            except (ValueError, IndexError):
                pass
        elif "Selected:" in line and "articles" in line:
            try:
                stats["articles_selected"] = int(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "Enrichment complete" in line:
            try:
                stats["llm_enriched"] = int(line.split("for")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "CrossRef:" in line:
            src = stats["sources_used"].setdefault("crossref", {"found": 0})
            try:
                src["found"] += int(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "OpenAlex:" in line:
            src = stats["sources_used"].setdefault("openalex", {"found": 0})
            try:
                src["found"] += int(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "Europe PMC:" in line:
            src = stats["sources_used"].setdefault("europe_pmc", {"found": 0})
            try:
                src["found"] += int(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "S2:" in line and "skip" not in line:
            src = stats["sources_used"].setdefault("semantic_scholar", {"found": 0})
            try:
                src["found"] += int(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "arXiv:" in line:
            src = stats["sources_used"].setdefault("arxiv", {"found": 0})
            try:
                src["found"] += int(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "Topic:" in line and "(" in line and "queries" not in line.lower():
            topic_name = line.split("Topic:")[-1].strip().split("(")[0].strip()
            if topic_name and topic_name not in stats["topics_searched"]:
                stats["topics_searched"].append(topic_name)
        elif "Already shown:" in line:
            try:
                stats["duplicates_skipped"] = int(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "[ERROR]" in line or "Traceback" in line:
            stats["errors"].append(line.strip())
        elif "[skip]" in line or "Warning" in line:
            stats["warnings"].append(line.strip())

    # Calculate duplicates skipped from raw vs after_dedup
    if stats["raw_results"] > 0 and stats["after_dedup"] >= 0:
        stats["duplicates_skipped"] = max(stats["duplicates_skipped"],
                                           stats["raw_results"] - stats["after_dedup"])
    stats["articles_found"] = stats["after_dedup"]

    # Graph stats
    graph_file = BASE / "graph_data.json"
    if graph_file.exists():
        try:
            gd = json.loads(graph_file.read_text())
            stats["graph_nodes"] = len(gd.get("nodes", []))
            stats["graph_edges"] = len(gd.get("edges", []))
        except (json.JSONDecodeError, OSError):
            pass

    # Duration
    started = stats.get("started_at", "")
    if started:
        try:
            t_start = datetime.fromisoformat(started.replace("Z", "+00:00"))
            duration = (datetime.now(timezone.utc) - t_start).total_seconds()
            stats["duration_sec"] = round(duration, 1)
        except (ValueError, OSError):
            pass

    return stats


# ── CLI entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Geo-Digest Run Manager")
    sub = parser.add_subparsers(dest="command")

    # Start a run
    p_start = sub.add_parser("start", help="Start a digest run")
    p_start.add_argument("--articles", type=int, default=None)
    p_start.add_argument("--topics", type=str, default=None)
    p_start.add_argument("--min-year", type=int, default=None)
    p_start.add_argument("--no-llm", action="store_true")

    # Status
    sub.add_parser("status", help="Show current status")

    # History
    sub.add_parser("history", help="Show run history")

    # Dedup stats
    sub.add_parser("dedup", help="Show deduplication statistics")

    # Stop
    sub.add_parser("stop", help="Stop running digest")

    args = parser.parse_args()

    if args.command == "start":
        overrides = {}
        if args.articles is not None:
            overrides.setdefault("digest", {})["articles_per_run"] = args.articles
        if args.min_year is not None:
            overrides.setdefault("digest", {})["min_year"] = args.min_year
        if args.topics:
            topics_list = args.topics.split(",")
            cfg = get_config()
            all_topics = cfg.get("topics", {})
            filtered = {k: v for k, v in all_topics.items() if k in topics_list}
            if filtered:
                overrides["topics"] = filtered
        result = start_digest_run(overrides)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif args.command == "status":
        s = get_status()
        print(json.dumps(s, ensure_ascii=False, indent=2))

    elif args.command == "history":
        runs = list_runs()
        for r in runs:
            print(f"  {r['id']} | {r.get('status','?')} | "
                  f"найдено:{r.get('articles_found',0)} выбрано:{r.get('articles_selected',0)} | "
                  f"{r.get('started_at','?')}")

    elif args.command == "dedup":
        d = get_dedup_stats()
        print(json.dumps(d, ensure_ascii=False, indent=2))

    elif args.command == "stop":
        result = stop_run()
        print(json.dumps(result, ensure_ascii=False, indent=2))

    else:
        parser.print_help()
