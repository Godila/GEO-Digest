# Стадия S4: API Endpoints

## Цель
Добавить API endpoints для Editor Agent в worker/server.py + proxy в dashboard/app.py.
Endpoints должны поддерживать: запуск, статус, resume, выбор предложения, финализация.

## Endpoints

```
Worker (:3001)                    Dashboard (:3000 proxy)
────────────────                  ──────────────────────────
POST   /api/editor/analyze        → POST   /api/editor/analyze
GET    /api/editor/jobs           → GET    /api/editor/jobs
GET    /api/editor/jobs/{id}      → GET    /api/editor/jobs/{id}
POST   /api/editor/jobs/{id}/resume → POST /api/editor/jobs/{id}/resume
POST   /api/editor/jobs/{id}/select/{prop_id}  → select proposal
POST   /api/editor/jobs/{id}/develop            → iterative dev
DELETE /api/editor/jobs/{id}      → DELETE /api/editor/jobs/{id}
GET    /api/editor/jobs/{id}/logs → SSE logs stream
```

## Что делаем

### 4.1 Worker Endpoints
Файл: `worker/server.py` (дополнить)

```python
# ── Editor Agent Endpoints ──
_editor_jobs: dict[str, dict] = {}  # in-memory cache (checkpoints on disk)

@app.post("/api/editor/analyze")
async def editor_analyze(request: Request):
    """Запуск Editor Agent: анализ темы + генерация предложений."""
    _ensure_engine_imports()
    body = await request.json()
    
    topic = body.get("topic", "")
    domain = body.get("domain")
    instruction = body.get("user_instruction")
    max_proposals = body.get("max_proposals", 5)
    
    if not topic and not domain:
        raise HTTPException(400, "topic or domain required")
    
    try:
        editor = EditorAgent()
        
        # Запуск в background thread (как Scout)
        import threading
        result_container = {"result": None, "error": None}
        
        def target():
            try:
                result_container["result"] = asdict(
                    editor.run(topic=topic, domain=domain,
                              user_instruction=instruction,
                              max_proposals=max_proposals)
                )
            except Exception as e:
                result_container["error"] = str(e)
        
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=300)  # 5 минут timeout
        
        if thread.is_alive():
            raise HTTPException(504, "Editor analysis timeout (>5 min)")
        
        if result_container["error"]:
            raise HTTPException(500, result_container["error"])
        
        result = result_container["result"]
        _editor_jobs[result["job_id"]] = result
        
        return {
            "job_id": result["job_id"],
            "status": result["status"],
            "proposals_count": len(result.get("proposals", [])),
            "duration_sec": result.get("duration_sec"),
            "message": f"Analysis complete: {len(result.get('proposals', []))} proposals",
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/editor/jobs")
async def editor_list_jobs():
    """Список всех editor jobs (с диска)."""
    _ensure_engine_imports()
    jobs_dir = Path("/app/data/jobs")
    jobs = []
    if jobs_dir.exists():
        for f in sorted(jobs_dir.glob("*.json"), reverse=True):
            with open(f) as fh:
                data = json.load(fh)
            # Фильтруем только editor jobs (имеют proposals поле)
            if "proposals" in data or "analysis" in data:
                jobs.append({
                    "job_id": data.get("job_id", f.stem),
                    "topic": data.get("topic", ""),
                    "phase": data.get("phase", "unknown"),
                    "proposals_count": len(data.get("proposals") or []),
                    "status": data.get("phase", "unknown"),
                    "started_at": data.get("started_at", ""),
                    "updated_at": data.get("updated_at", ""),
                })
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/api/editor/jobs/{job_id}")
async def editor_get_job(job_id: str):
    """Детали editor job с proposals."""
    job_path = Path(f"/app/data/jobs/{job_id}.json")
    if not job_path.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    
    with open(job_path) as f:
        data = json.load(f)
    
    return data


@app.post("/api/editor/jobs/{job_id}/resume")
async def editor_resume_job(job_id: str):
    """Возобновить/перезапустить анализ."""
    _ensure_engine_imports()
    editor = EditorAgent()
    result = editor.resume(job_id)
    return asdict(result)


@app.post("/api/editor/jobs/{job_id}/select/{prop_id}")
async def editor_select_proposal(job_id: str, prop_id: str):
    """Выбрать предложение для дальнейшей разработки."""
    job_path = Path(f"/app/data/jobs/{job_id}.json")
    if not job_path.exists():
        raise HTTPException(404, f"Job {job_id} not found")
    
    with open(job_path) as f:
        data = json.load(f)
    
    # Находим proposal и помечаем selected
    proposals = data.get("proposals", [])
    found = False
    for p in proposals:
        if p.get("id") == prop_id:
            p["status"] = "selected"
            found = True
            break
    
    if not found:
        raise HTTPException(404, f"Proposal {prop_id} not found")
    
    data["selected_proposal_id"] = prop_id
    data["updated_at"] = now_iso()
    
    with open(job_path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    
    return {"status": "selected", "proposal_id": prop_id}


@app.delete("/api/editor/jobs/{job_id}")
async def editor_delete_job(job_id: str):
    """Удалить editor job."""
    job_path = Path(f"/app/data/jobs/{job_id}.json")
    if job_path.exists():
        job_path.unlink()
    if job_id in _editor_jobs:
        del _editor_jobs[job_id]
    return {"deleted": True, "job_id": job_id}
```

### 4.2 Dashboard Proxy
Файл: `dashboard/app.py` (дополнить)

Аналогично orchestrator proxy — generic `_worker_request()` уже есть,
добавляем маршруты.

### 4.3 SSE Logs Endpoint
Для real-time логов работы Editor Agent:

```python
@app.get("/api/editor/jobs/{job_id}/logs")
async def editor_job_logs(job_id: str):
    """SSE stream логов editor job."""
    job_path = Path(f"/app/data/jobs/{job_id}.json")
    if not job_path.exists():
        raise HTTPException(404, "Job not found")
    
    async def generate():
        # Читаем checkpoint и выдаём как события
        with open(job_path) as f:
            data = json.load(f)
        
        yield f"data: {json.dumps({'type': 'status', 'phase': data.get('phase')})}\n\n"
        
        if data.get("analysis"):
            yield f"data: {json.dumps({'type': 'analysis', 'clusters': len(data['analysis'].get('clusters', []))})}\n\n"
        
        if data.get("proposals"):
            yield f"data: {json.dumps({'type': 'proposals', 'count': len(data['proposals'])})}\n\n"
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Acceptance Criteria S4

- [ ] `POST /api/editor/analyze` запускает EditorAgent и возвращает job_id
- [ ] `POST /api/editor/analyze` возвращает proposals_count
- [ ] `POST /api/editor/analyse` с пустым topic+domain → 400
- [ ] `GET /api/editor/jobs` возвращает список jobs с диска
- [ ] `GET /api/editor/jobs/{id}` возвращает полный checkpoint JSON
- [ ] `GET /api/editor/jobs/{id}` для несуществующего → 404
- [ ] `POST /api/editor/jobs/{id}/resume` возобновляет job
- [ ] `POST /api/editor/jobs/{id}/select/{prop_id}` помечает proposal как selected
- [ ] `DELETE /api/editor/jobs/{id}` удаляет файл
- [ ] `GET /api/editor/jobs/{id}/logs` возвращает SSE stream
- [ ] Все endpoint'ы работают через dashboard proxy

## Тесты S4

### Unit тесты

**test_editor_api.py**
```python
class TestEditorAnalyzeEndpoint:
    @fixture
    def client():  # TestClient для FastAPI
        ...
    
    def test_analyze_success(client):
        resp = client.post("/api/editor/analyze", json={
            "topic": "Arctic methane",
            "max_proposals": 3
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "done"
        assert data["proposals_count"] >= 1
    
    def test_analyze_requires_topic(client):
        resp = client.post("/api/editor/analyze", json={})
        assert resp.status_code == 400
    
    def test_analyze_with_domain(client):
        resp = client.post("/api/editor/analyze", json={
            "domain": "climate change",
            "max_proposals": 2
        })
        assert resp.status_code == 200
    
    def test_analyze_with_instruction(client):
        resp = client.post("/api/editor/analyze", json={
            "topic": "methane",
            "user_instruction": "Focus on 2024-2025 studies only"
        })
        assert resp.status_code == 200

class TestEditorJobsEndpoint:
    def test_list_empty(client):
        resp = client.get("/api/editor/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert "total" in data
    
    def test_list_after_analyze(client):
        # Сначала создаём job
        r1 = client.post("/api/editor/analyze", json={"topic": "test"})
        job_id = r1.json()["job_id"]
        
        r2 = client.get("/api/editor/jobs")
        jobs = r2.json()["jobs"]
        assert any(j["job_id"] == job_id for j in jobs)
    
    def test_get_job_detail(client):
        r1 = client.post("/api/editor/analyze", json={"topic": "test"})
        job_id = r1.json()["job_id"]
        
        r2 = client.get(f"/api/editor/jobs/{job_id}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["job_id"] == job_id
        assert "proposals" in data or "analysis" in data
    
    def test_get_404(client):
        resp = client.get("/api/editor/jobs/nonexistent")
        assert resp.status_code == 404

class TestEditorSelectProposal:
    def test_select_proposal(client):
        r1 = client.post("/api/editor/analyze", json={"topic": "test"})
        job_id = r1.json()["job_id"]
        
        # Получаем proposals
        r2 = client.get(f"/api/editor/jobs/{job_id}")
        props = r2.json().get("proposals", [])
        if props:
            prop_id = props[0]["id"]
            
            r3 = client.post(f"/api/editor/jobs/{job_id}/select/{prop_id}")
            assert r3.status_code == 200
            assert r3.json()["status"] == "selected"
            
            # Проверяем что сохранилось
            r4 = client.get(f"/api/editor/jobs/{job_id}")
            assert r4.json()["selected_proposal_id"] == prop_id
    
    def test_select_nonexistent_proposal(client):
        r1 = client.post("/api/editor/analyze", json={"topic": "test"})
        job_id = r1.json()["job_id"]
        
        resp = client.post(f"/api/editor/jobs/{job_id}/select/fake_prop")
        assert resp.status_code == 404

class TestEditorDelete:
    def test_delete_existing(client):
        r1 = client.post("/api/editor/analyze", json={"topic": "test"})
        job_id = r1.json()["job_id"]
        
        resp = client.delete(f"/api/editor/jobs/{job_id}")
        assert resp.status_code == 200
        
        resp2 = client.get(f"/api/editor/jobs/{job_id}")
        assert resp2.status_code == 404

class TestEditorLogsSSE:
    def test_logs_stream(client):
        r1 = client.post("/api/editor/analyze", json={"topic": "test"})
        job_id = r1.json()["job_id"]
        
        resp = client.get(f"/api/editor/jobs/{job_id}/logs")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        
        data = resp.text
        assert "data:" in data
        assert "[DONE]" in data

### Интеграционный тест

**test_editor_api_integration.py**
```python
@mark.integration
class TestEditorAPIRealDocker:
    """Тесты через Docker exec curl."""
    
    BASE = "http://localhost:3001/api/editor"
    
    def test_full_cycle_via_api():
        # 1. Analyze
        r = requests.post(f"{BASE}/analyze", json={
            "topic": "Arctic permafrost methane",
            "max_proposals": 3
        }, timeout=300)
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        
        # 2. List jobs
        r = requests.get(f"{BASE}/jobs")
        assert any(j["job_id"] == job_id for j in r.json()["jobs"])
        
        # 3. Get detail
        r = requests.get(f"{BASE}/jobs/{job_id}")
        data = r.json()
        assert len(data.get("proposals", [])) >= 1
        
        # 4. Select first proposal
        prop_id = data["proposals"][0]["id"]
        r = requests.post(f"{BASE}/jobs/{job_id}/select/{prop_id}")
        assert r.status_code == 200
        
        # 5. Logs
        r = requests.get(f"{BASE}/jobs/{job_id}/logs", timeout=10)
        assert r.status_code == 200
        
        # 6. Cleanup
        r = requests.delete(f"{BASE}/jobs/{job_id}")
        assert r.status_code == 200
    
    def test_dashboard_proxy_works():
        """Проверка что proxy через dashboard работает."""
        r = requests.post("http://localhost:3000/api/editor/analyze", json={
            "topic": "proxy test"
        }, timeout=300)
        assert r.status_code == 200
```

## Файлы стадии S4

| Файл | Действие | Строк |
|------|----------|-------|
| `worker/server.py` | Изменить (+editor endpoints) | +250 |
| `dashboard/app.py` | Изменить (+editor proxy routes) | +80 |
| `tests/test_editor_api.py` | Создать | ~280 |
| `tests/test_editor_api_integration.py` | Создать | ~80 |

**Итого:** ~690 строк
