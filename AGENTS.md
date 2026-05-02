# GEO-Digest Project Constitution

## Code Search — MANDATORY TOOLS

### Semantic Search (PRIMARY)
```bash
cd /home/hermeswebui/.hermes/geo_digest && semble search "query" -k 5
```
Use for ALL code exploration: finding features, understanding flow, locating implementations.
DO NOT use grep/search_files for exploratory queries — use semble.

### Structural Search (SECONDARY)
```bash
cd /home/hermeswebui/.hermes/geo_digest && leankg query "ClassName"
cd /home/hermeswebui/.hermes/geo_digest && leankg impact engine/agents/writer.py
```
Use for class hierarchy, function calls, impact analysis.

### Exact String Search (LAST RESORT)
```bash
search_files pattern="_LLM_TIMEOUT_SECONDS" target=content
```
Use ONLY for exact literal matches where semantic search is overkill.

## Architecture

Multi-agent pipeline: Scout → Editor → [SELECT] → Reader → Writer → Reviewer → DONE
- orchestrator_v2.py: state machine, review loop, DOI extraction
- Writer=DeepSeek V4 Pro, Editor/Reader=DeepSeek V4 Flash, Reviewer=Gemini 3.1 Pro
- Docker: worker (3001) + dashboard (3000), `sudo /tmp/docker/docker compose`

## Key Constraints
- Article schema: use `art._data["key"]=val` for dynamic fields
- PipelineJob.final_article (NOT written_article)
- Background threads die silently — use foreground `docker exec`
- `.env` at `/app/.env`, load manually in docker exec scripts
- Deploy: `docker cp` engine files → restart worker
