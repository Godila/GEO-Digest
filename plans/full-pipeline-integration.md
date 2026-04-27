# GEO-Digest: Полный цикл статьи — План интеграции

> **Последнее обновление:** 2026-04-27  
> **Статус:** Phase 0 (багфиксы) → Phase 1 (API) — ready to start  
> **Ветка:** `main` | **Коммит:** `d7a15a8` (select proposal fix)  
> **GitHub:** https://github.com/Godila/GEO-Digest

---

## 1. ЦЕЛЬ

Замкнуть полный цикл написания научной статьи:

```
Тема пользователя
  → Editor Agent (анализ хранилища + генерация предложений)
  → Пользователь выбирает предложение
  → Reader Agent (чтение источников → StructuredDraft)
  → Writer Agent (черновик статьи → WrittenArticle)
  → Reviewer Agent (факт-чек + стилевая проверка)
  → Готовая статья (Markdown/LaTeX)
```

**Human-in-the-loop** на каждом этапе перехода.

---

## 2. ТЕКУЩИЙ СТАТУС КОМПОНЕНТОВ

| Компонент | Файл | Строк | Готовность | Статус |
|-----------|------|-------|------------|--------|
| ToolUseLoop (S0-S7) | `engine/llm/tool_loop.py` | 356 | 95% | ✅ Работает |
| Editor Agent | `engine/agents/editor.py` | 619 | 90% | ✅ Работает через API |
| Storage Tools (7) | `engine/tools/storage_tools.py` | 699 | 90% | ✅ Интегрированы |
| Graph Tools (7) | `engine/tools/graph_tools.py` | 837 | 85% | ⚠️ Silent except |
| Orchestrator v2 | `engine/orchestrator_v2.py` | 443 | 80% | 📝 Написан, НЕ подключён к HTTP |
| Writer Agent | `engine/agents/writer.py` | 296 | 80% | 📝 Готов, не вызывается |
| Reviewer Agent | `engine/agents/reviewer.py` | 259 | 75% | 📝 Готов, не вызывается |
| Reader Agent | `engine/agents/reader.py` | 369 | 60% | ⚠️ Нет PDF download |
| Schemas | `engine/schemas.py` | 324 | 70% | ⚠️ Plain classes |
| Worker API (Editor) | `worker/server.py` | 1719 | 75% | ✅ Editor endpoints работают |
| Dashboard UI | `dashboard/templates/index.html` | 4400 | 40% | ⚠️ 3 stub-функции |
| Dashboard API GW | `dashboard/app.py` | 598 | 70% | ⚠️ Нет /api/pipeline/* |

### Что РАБОТЕТ сейчас:
- POST /api/editor/analyze → EditorAgent.run() → proposals ✅
- Polling фаз (idle→analyzing→proposing→done) ✅
- Список jobs, открытие, удаление ✅
- Выбор proposal (select) ✅
- Граф знаний (Cytoscape.js lazy load) ✅

### Что НЕ работает:
- develop/write/review — toast "будет в S6" ❌
- Orchestrator v2 не имеет HTTP endpoint'ов ❌
- Reader не скачивает PDF ❌

---

## 3. АРХИТЕКТУРА

### 3.1 Целевая архитектура

```
Browser (vanilla JS)
  │
  ▼
Dashboard (:3000) ──proxy──▶ Worker (:3001)
  │  (FastAPI)                │  (FastAPI)
  │                            │
  ▼                            ▼
/api/pipeline/*         EditorOrchestrator (singleton)
                          │
  POST run               create_job()
  GET  jobs              ├── run_editing_phase()
  GET  {id}              │    ├→ EditorAgent.run(topic)
  POST {id}/select       │    │   ├→ ToolUseLoop(ANALYSIS_PROMPT)
  POST {id}/develop      │    │   ├→ ToolUseLoop(PROPOSAL_PROMPT)
  POST {id}/write        │    │   └→ post-validation + graph enrich
  POST {id}/review       │    └→ state = SELECTING
  DELETE {id}            │
                          ├── select_proposal(prop_id)
                          │    └→ state = DEVELOPING
                          │
                          ├── develop(feedback?)
                          │    ├→ ReaderAgent.run(dois=...)
                          │    │   └→ StructuredDraft (или abstract-only fallback)
                          │    └→ state = DEVELOPING (loop) or SATISFIED
                          │
                          ├── write()
                          │    ├→ WriterAgent.run(draft=...)
                          │    │   └→ WrittenArticle (Markdown)
                          │    └→ state = REVIEWING
                          │
                          └── review()
                               ├→ ReviewerAgent.run(article=...)
                               │   └→ ReviewedDraft (verdict + edits + fact_checks)
                               └→ state = DONE or back to DEVELOPING
```

### 3.2 State Machine

```
  idle ──run(topic)────────→ editing
  editing ──proposals_ready─→ selecting     [auto]
  selecting ──select(prop)──→ developing   [user click]
  developing ──feedback────→ developing    [user feedback, LOOP!]
  developing ──satisfied───→ writing       [user click]
  writing ──done──────────→ reviewing      [auto]
  reviewing ──approved────→ done           [auto or user]
  reviewing ──needs_work──→ developing    [auto, back for revision]
  * ──cancel─────────────→ cancelled      [from any state]
  * ──error──────────────→ failed         [from any state]
```

### 3.3 Persistence

```
/app/data/jobs/{job_id}.json
{
  "job_id": "pipe_20260427_120000_abc123",
  "topic": "...",
  "state": "developing",          // PipelineState enum value
  "editor_result": {...},          // from EditorAgent
  "selected_proposal_id": "...",
  "development_rounds": [...],
  "current_draft": {...},          // StructuredDraft from Reader
  "final_article": {...},          // WrittenArticle from Writer
  "review_result": {...},          // ReviewedDraft from Reviewer
  "created_at": "...",
  "updated_at": "..."
}
```

---

## 4. ИЗВЕСТНЫЕ БАГИ (для исправления в Phase 0)

### Критические

| ID | Файл:Строка | Проблема | Фикс |
|----|-------------|----------|------|
| B1 | `orchestrator_v2.py:104` | `EditorAgent()` создан без storage/llm → self.loop=None → RuntimeError | Передать `storage=` и `llm_provider=` при создании |
| B2 | `editor.py:295-298` | Token accounting: `input_tokens + input_tokens` (опечатка!), output ignored | Исправить на input + output |

### Серьёзные

| ID | Файл:Строка | Проблема | Фикс |
|----|-------------|----------|------|
| B3 | `editor.py:173` | `except Exception: pass` — graph tools молча исчезают | Добавить logger.warning + return empty result |
| B4 | `server.py` (multiple) | Race condition: poll читает ← одновременно → select пишет JSON | Добавить threading.Lock() вокруг записи |
| B5 | `editor.py:606-614` | `_similarity`: word-level Jaccard без .lower(), без stemming RU | Добавить .lower(), optional pymorphy2 |

### Умеренные (можно отложить)

| ID | Файл:Строка | Проблема | Приоритет |
|----|-------------|----------|-----------|
| B6 | `graph_tools.py:267-270` | O(n*m) lookup вместо pre-built dict | P2 |
| B7 | `graph_tools.py:548` | cross_topic: логическая ошибка вычитания множеств | P2 |
| B8 | `tool_loop.py:264` | Truncate tool results at 8000 chars silently | P3 |
| B9 | `tool_loop.py:232-234` | Sequential tool execution (no parallelism) | P3 |

---

## 5. ROADMAP ИМПЛЕМЕНТАЦИИ

### Phase 0: Фундаментальные багфиксы (~1 час)

- [ ] **P0-B1**: Починить EditorAgent() init в orchestrator_v2.py
- [ ] **P0-B2**: Исправить token accounting в editor.py
- [ ] **P1-B3**: Заменить silent except на logging в editor.py
- [ ] **P1-B4**: Добавить file_lock для JSON writes в server.py
- [ ] **P2-B5**: Улучшить _similarity (lowercase + optional stemming)

**Acceptance:** `orch.editor.run(topic="test")` не падает с RuntimeError

### Phase 1: Orchestrator v2 HTTP API (~2 часа)

#### 1.1 Worker endpoint'ы (`worker/server.py`)

- [ ] **P1-API1**: `POST /api/pipeline/run` — создать job + запустить editing phase (background thread)
- [ ] **P1-API2**: `GET /api/pipeline/jobs` — список pipeline jobs
- [ ] **P1-API3**: `GET /api/pipeline/jobs/{id}` — детали job (lazy enrichment)
- [ ] **P1-API4**: `POST /api/pipeline/jobs/{id}/select` — выбрать proposal
- [ ] **P1-API5**: `POST /api/pipeline/jobs/{id}/develop` — запустить reader
- [ ] **P1-API6**: `POST /api/pipeline/jobs/{id}/write` — запустить writer
- [ ] **P1-API7**: `POST /api/pipeline/jobs/{id}/review` — запустить reviewer
- [ ] **P1-API8**: `DELETE /api/pipeline/jobs/{id}` — отменить job

#### 1.2 Dashboard прокси (`dashboard/app.py`)

- [ ] **P1-PROXY1**: Зеркалировать все /api/pipeline/* endpoint'ы
- [ ] **P1-PROXY2**: Увеличить timeout для develop/write (могут быть долгими)

#### 1.3 Singleton orchestrator

- [ ] **P1-SINGLETON**: Создать `EditorOrchestrator` singleton в worker/server.py (при старте)
- [ ] **P1-SINGLETON**: Передать storage + llm_provider из конфига

**Acceptance:**
```bash
curl -X POST http://localhost:3000/api/pipeline/run \
  -d '{"topic":"тест"}'
# → {"job_id": "pipe_...", "state": "editing"}

curl http://localhost:3000/api/pipeline/jobs
# → [{job_id, topic, state, ...}]

# После завершения editing:
curl -X POST http://localhost:3000/api/pipeline/jobs/{id}/select \
  -d '{"proposal_id": "prop_..."}'
# → {"state": "developing"}
```

### Phase 2: Reader Agent — PDF Pipeline (~2 часа)

- [ ] **P2-R1**: Unpaywall integration (проверка OA по DOI)
- [ ] **P2-R2**: PDF download (httpx streaming, timeout 30s, max 50MB)
- [ ] **P2-R3**: PyMuPDF text extraction (проверить что есть в Docker image)
- [ ] **P2-R4**: Abstract-only fallback (критично! работать без PDF)
- [ ] **P2-R5**: Batch aggregation read_group() для нескольких DOI
- [ ] **P2-R6**: Тест: ReaderAgent.run(dois=["10.xxxx"]) → StructuredDraft

**Acceptance:** develop фаза возвращает StructuredDraft даже когда PDF недоступен

### Phase 3: UI Интеграция — Полный цикл (~3 часа)

#### 3.1 Замена stub-функций

- [ ] **P3-UI1**: `sendForDevelopment()` → real POST /api/pipeline/{id}/develop
- [ ] **P3-UI2**: `finalizeArticle()` → real POST /api/pipeline/{id}/write
- [ ] **P3-UI3**: `reviewArticle()` → real POST /api/pipeline/{id}/review

#### 3.2 Новые UI компоненты

- [ ] **P3-UI4**: Pipeline Status Bar (editing→selecting→developing→writing→reviewing→done)
- [ ] **P3-UI5**: Draft Panel (StructuredDraft: methods/data/infra/gaps tabs)
- [ ] **P3-UI6**: Article Preview (Markdown render финальной статьи)
- [ ] **P3-UI7**: Review Results (edits table, verdict badge, fact_checks)
- [ ] **P3-UI8**: Кнопка "Скачать статью" (.md download)
- [ ] **P3-UI9**: Feedback форма для iterative development loop

#### 3.3 Polling расширение

- [ ] **P3-UI10**: Расширить polling для pipeline states (не только editor phases)
- [ ] **P3-UI11**: Показывать progress для develop/write/review фаз

**Acceptance:** Полный цикл работает в браузере без перезагрузки страницы

### Phase 4: Reviewer — Отдельный LLM провайдер (~1 час)

- [ ] **P4-REV1**: Проверить openai_compat.py существует и работает
- [ ] **P4-REV2**: Добавить reviewer секцию в config.yaml
- [ ] **P4-REV3**: Для MVP: один MiniMax OK (разные prompts = разное поведение)
- [ ] **P4-REV4**: Логирование reviewer model name в review_result

**Acceptance:** review возвращает ReviewedDraft с verdict != null

### Phase 5: E2E Тест + Полировка (~2 часа)

- [ ] **P5-E2E1**: Полный прогон: тема → proposals → select → develop → write → review → done
- [ ] **P5-E2E2**: Проверка сохранения состояния между перезапусками worker
- [ ] **P5-E2E3**: Race condition тест (одновременный poll + action)
- [ ] **P5-E2E4**: Graceful degradation (нет PDF → abstract-only → статья хуже но есть)
- [ ] **P5-E2E5**: Все кнопки UI работают, ошибки обрабатываются с тостами
- [ ] **P5-E2E6**: Commit + push + обновить этот документ

---

## 6. ТЕХНОЛОГИЧЕСКИЙ СТЕК

| Компонент | Технология | Версия | Оценка |
|-----------|-----------|--------|--------|
| Язык | Python | 3.12+ | ✅ |
| Worker API | FastAPI | latest | ✅ |
| Dashboard API | Flask | latest | ⚠️ Почему 2 фреймворка? |
| Frontend | Vanilla JS | ES2022 | ✅ Для 1 dev OK |
| LLM (primary) | MiniMax M2.7 | Anthropic-compatible | ✅ |
| LLM (reviewer) | MiniMax M2.7 (same) | TODO: separate provider | ⚠️ |
| Graph viz | Cytoscape.js | latest (CDN/lazy) | ✅ |
| Storage | JSONL (articles) + JSON (jobs) | — | ✅ Для <10K статей |
| Containerisation | Docker Compose | v2 | ✅ |
| PDF parsing | PyMuPDF | 1.23+ | ⚠️ Нужно проверить |
| Config | YAML | — | ✅ |

---

## 7. КЛЮЧЕВЫЕ ФАЙЛЫ (карта проекта)

```
/root/.hermes/geo_digest/
├── engine/
│   ├── __init__.py
│   ├── config.py                    # Конфигурация
│   ├── schemas.py                   # Data structures (324 строки)
│   ├── llm/
│   │   ├── base.py                  # LLMProvider ABC
│   │   ├── minimax.py               # MiniMax M2.7 wrapper
│   │   └── openai_compat.py         # OpenAI-compatible (for reviewer)
│   ├── tools/
│   │   ├── base.py                  # ToolResult, BaseTool, ToolRegistry
│   │   ├── storage_tools.py         # 7 tools (search, detail, validate...)
│   │   └── graph_tools.py           # 7 tools (neighbors, path, hubs...)
│   ├── agents/
│   │   ├── base.py                  # BaseAgent ABC + LLMCallMixin
│   │   ├── editor.py                # ★ Editor Agent (ToolUseLoop) — 619 строк
│   │   ├── reader.py                # Reader Agent (PDF→Draft) — 369 строк
│   │   ├── writer.py                # Writer Agent (Draft→Article) — 296 строк
│   │   └── reviewer.py              # Reviewer Agent (fact-check) — 259 строк
│   ├── prompts/
│   │   └── editor_prompts.py        # 4 системных промпта
│   ├── orchestrator.py              # Legacy V1 (Scout→Reader→Writer→Reviewer)
│   ├── orchestrator_v2.py           # ★ V2 State Machine (Editor entry point) — 443 строки
│   └── api.py                       # FastAPI Engine API (порт 3002)
│
├── worker/
│   ├── server.py                    # ★ Worker API (порт 3001) — 1719 строк
│   └── Dockerfile                   # python:3.12-slim
│
├── dashboard/
│   ├── app.py                       # Dashboard Flask gateway (порт 3000) — 598 строк
│   └── templates/
│       └── index.html               # ★ Весь frontend (HTML+CSS+JS) — ~4400 строк
│
├── docker-compose.yml               # dashboard(:3000) + worker(:3001)
├── config.yaml                      # Конфигурация (7 topics, 37 queries)
├── ROADMAP.md                       # Оригинальный roadmap (8 спринтов)
└── plans/
    └── tool-use-editor/
        └── S6-orchestrator-refactor.md  # План S6 (506 строк)
```

---

## 8. КОНФИГУРАЦИЯ И СЕКРЕТЫ

### Переменные окружения
- `MINIMAX_API_KEY` → `/app/.env` (worker container)
- GitHub PAT: `github_pat_11ALH...XhNM` (Godila)

### Docker
- Dashboard memory: 256MB, port 3000
- Worker memory: 2GB, port 3001 (internal)
- Порт 3001 НЕ проброшен наружу (только через dashboard proxy)
- Shared volume: `geo_digest_geo-data` → `/app/data/`

### Push команда
```bash
GIT_TERMINAL_PROMPT=0 git push \
  https://Godila:<PAT>@github.com/Godila/GEO-Digest.git main
```

---

## 9. DECISION LOG

| Дата | Решение | Контекст |
|------|---------|----------|
| 2026-04-27 | Variant D: Graph Isolation (fullscreen tab) | Экономит ~645KB, TTI 3× быстрее |
| 2026-04-27 | Lazy load Cytoscape.js | Грузится только по клику на вкладку "Граф" |
| 2026-04-27 | Orchestrator v2 как основной pipeline | State machine вместо linear V1 |
| 2026-04-27 | Human-in-the-loop на каждом переходе | Approval gates между фазами |
| 2026-04-27 | JSON files persistence (not SQLite) | Personal tool, <100 jobs, OK |
| 2026-04-27 | Vanilla JS (no framework) | 1 developer, 4400 lines manageable |
| 2026-04-27 | Option A: Full Orchestrator v2 integration | Выбрано пользователем |

---

## 10. ССЫЛКИ НА СМЕЖНЫЕ ДОКУМЕНТЫ

- **ROADMAP.md** — Оригинальный план 8 спринтов (устарел частично)
- **plans/tool-use-editor/S6-orchestrator-refactor.md** — Детальный план S6
- **Skill: geo-digest-graph-audit** — Аудит графа знаний
- **Skill: geo-digest-frontend-audit** — Аудит UI/UX
- **Skill: geo-digest-agent-api** — API поверх файлового движка
- **Skill: geo-digest-maintenance** — Руководство по поддержке

---

*Этот документ — Single Source of Truth для разработки полного цикла статьи.*
*Обновлять после каждого значимого изменения.*
