# GEO-Digest Dead/Legacy/Unused Code Report

Generated: 2026-05-02

Methodology: Cross-referenced all endpoints in `worker/server.py` against frontend calls in `dashboard/templates/index.html` (via dashboard proxy in `dashboard/app.py`). Checked all engine files for legacy/deprecated/unused markers.

---

## 1. worker/server.py — DEAD ENDPOINTS (not called by frontend)

### Completely Dead (no dashboard proxy, no frontend call)

| File:Line | Method/Endpoint | Reason |
|-----------|-----------------|--------|
| worker/server.py:801 | `GET /api/a/graph/neighbors` | No dashboard proxy, no frontend call |
| worker/server.py:816 | `GET /api/a/graph/path` | No dashboard proxy, no frontend call |
| worker/server.py:831 | `GET /api/a/graph/subgraph` | No dashboard proxy, no frontend call |
| worker/server.py:838 | `GET /api/a/graph/nodes` | No dashboard proxy, no frontend call |
| worker/server.py:873 | `GET /api/a/info` | No dashboard proxy, no frontend call |
| worker/server.py:881 | `GET /api/a/export` | No dashboard proxy, no frontend call |
| worker/server.py:2027 | `GET /api/pipeline/jobs/{id}/formats` | No dashboard proxy, no frontend call |

### Agent-facing API (dashboard proxies exist but frontend never calls them)

| File:Line | Method/Endpoint | Reason |
|-----------|-----------------|--------|
| worker/server.py:1056 | `POST /api/engine/scout` | Dashboard proxies but frontend uses /pipeline/run |
| worker/server.py:1102 | `GET /api/engine/jobs` | Dashboard proxies but frontend never calls |
| worker/server.py:1121 | `GET /api/engine/jobs/{job_id}` | Dashboard proxies but frontend never calls |
| worker/server.py:1143 | `DELETE /api/engine/jobs/{job_id}` | Dashboard proxies but frontend never calls |
| worker/server.py:1159 | `GET /api/engine/jobs/{job_id}/logs` | Dashboard proxies but frontend never calls |
| worker/server.py:1196 | `GET /api/engine/status` | Dashboard proxies but frontend never calls |

### Legacy v1 (superseded by /pipeline/* endpoints)

| File:Line | Method/Endpoint | Reason |
|-----------|-----------------|--------|
| worker/server.py:1264 | `POST /api/orchestrator/job` | Legacy v1 — frontend uses POST /pipeline/run |
| worker/server.py:1487 | `POST /api/editor/analyze` | Legacy — frontend uses POST /pipeline/run |
| worker/server.py:1646 | `POST /api/editor/jobs/{id}/resume` | Legacy — no frontend call |
| worker/server.py:1660 | `POST /api/editor/jobs/{id}/select/{prop_id}` | Legacy — frontend uses /pipeline/jobs/{id}/select |
| worker/server.py:1702 | `GET /api/editor/jobs/{id}/logs` | Legacy — no frontend call |

### Other Unused

| File:Line | Method/Endpoint | Reason |
|-----------|-----------------|--------|
| worker/server.py:689 | `POST /api/digest/config` | Frontend only calls GET, never POST |
| worker/server.py:772 | `GET /api/a/search` | Dashboard has proxy but frontend never calls |

---

## 2. engine/orchestrator.py — ENTIRE FILE IS LEGACY V1

| File:Line | Class/Method | Reason |
|-----------|-------------|--------|
| engine/orchestrator.py:1-228 | `class Orchestrator` (entire file) | Legacy v1 orchestrator, superseded by engine/orchestrator_v2.py (`EditorOrchestrator`). Only used by `/api/orchestrator/*` legacy endpoints and test_e2e.py. All methods are v1: `create_job`, `start_job`, `approve_group`, `approve_draft`, `request_revision`, `skip_review`, `cancel_job`, `load_state`, `list_jobs`, `_run_scout_async`, `_run_reader_async`, `_run_writer_async` |

---

## 3. engine/agents/scout.py — Legacy Method

| File:Line | Method | Reason |
|-----------|--------|--------|
| engine/agents/scout.py:263 | `_collect_candidates` | Explicitly marked "Legacy method — collect without scoring. Used by old pipeline v1." **Zero callers** in entire codebase. Replaced by `_collect_and_score` (line 155). |

---

## 4. engine/agents/editor.py — Legacy Fields/Params

| File:Line | Item | Reason |
|-----------|------|--------|
| engine/agents/editor.py:127 | `EditorResult.analysis` field | Comment: "Legacy key — holds DiscoveryReport dict" |
| engine/agents/editor.py:229 | `EditorAgent.run(max_proposals=5)` param | Comment: "legacy param, ignored (B+ decides count)" — parameter accepted but never used |

---

## Summary Statistics

- **Total dead/unused endpoints in server.py**: 20
- **Entire legacy file**: 1 (engine/orchestrator.py, 228 lines)
- **Dead methods in agents**: 1 (`_collect_candidates`)
- **Legacy fields/params**: 2 (editor.py)

### Frontend-verified endpoints (ACTIVELY USED by dashboard/templates/index.html):

These are confirmed used and should NOT be removed:
- GET /api/health, POST /api/digest/run, GET /api/digest/status
- POST /api/digest/stop, GET /api/digest/runs, GET /api/digest/runs/{id}
- GET /api/digest/config, GET /api/digest/dedup
- POST /api/graph/rebuild, GET /api/graph/status
- GET /api/a/articles, GET /api/a/stats, GET /api/a/topics
- GET /api/orchestrator/jobs, GET /api/orchestrator/jobs/{id}, DELETE /api/orchestrator/jobs/{id}
- GET /api/editor/jobs, GET /api/editor/jobs/{id}, DELETE /api/editor/jobs/{id}
- POST /api/pipeline/run, GET /api/pipeline/jobs, GET /api/pipeline/jobs/{id}
- POST /api/pipeline/jobs/{id}/select, POST /api/pipeline/jobs/{id}/develop
- POST /api/pipeline/jobs/{id}/write, POST /api/pipeline/jobs/{id}/review
- GET /api/pipeline/jobs/{id}/export, DELETE /api/pipeline/jobs/{id}
