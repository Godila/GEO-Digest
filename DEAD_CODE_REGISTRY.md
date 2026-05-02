# GEO-Digest Dead Code — Registry for Future Cleanup

## worker/server.py — 20 Dead/Unused Endpoints

### Completely dead (no proxy, no frontend — 7):
- L801  GET  /api/a/graph/neighbors — no dashboard proxy
- L816  GET  /api/a/graph/path — no dashboard proxy
- L831  GET  /api/a/graph/subgraph — no dashboard proxy
- L838  GET  /api/a/graph/nodes — no dashboard proxy
- L873  GET  /api/a/info — no dashboard proxy
- L881  GET  /api/a/export — no dashboard proxy
- L2027 GET  /api/pipeline/jobs/{id}/formats — no dashboard proxy

### Agent-facing (dashboard proxies but frontend never calls — 6):
- L1056 POST /api/engine/scout
- L1102 GET  /api/engine/jobs
- L1121 GET  /api/engine/jobs/{id}
- L1143 DELETE /api/engine/jobs/{id}
- L1159 GET  /api/engine/jobs/{id}/logs
- L1196 GET  /api/engine/status

### Legacy v1 (superseded by /pipeline/* — 5):
- L1264 POST /api/orchestrator/job — frontend uses /pipeline/run
- L1487 POST /api/editor/analyze — frontend uses /pipeline/run
- L1646 POST /api/editor/jobs/{id}/resume
- L1660 POST /api/editor/jobs/{id}/select/{prop_id} — frontend uses /pipeline/.../select
- L1702 GET  /api/editor/jobs/{id}/logs

### Other unused — 2:
- L689  POST /api/digest/config — frontend only GETs config
- L772  GET  /api/a/search — dashboard proxies but frontend never calls

## engine/orchestrator.py — ENTIRE FILE (228 lines) — DEAD
- Class Orchestrator (all methods) — legacy v1, superseded by orchestrator_v2.py
- Only used by /api/orchestrator/* legacy endpoints + test_e2e.py

## engine/agents/scout.py — Dead Method
- L263 _collect_candidates — "Legacy method — used by old pipeline v1". Zero callers.

## engine/agents/editor.py — Legacy Fields
- L127 EditorResult.analysis — "Legacy key — holds DiscoveryReport dict"
- L229 EditorAgent.run(max_proposals) — "legacy param, ignored"
