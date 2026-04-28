# 🌍 GEO-Digest — Geo-Ecology Research Intelligence Platform

<p align="center">
  <strong>Мультиагентная платформа для автоматизированного поиска, анализа и написания научных статей по геоэкологии</strong><br>
  <em>Фокус: Южная Россия, Северный Кавказ, Чёрное море, Каспийский регион</em>
</p>

---

## 📖 Оглавление

- [Концепция](#-концепция)
- [Архитектура](#-архитектура)
- [Пайплайны](#-пайплайны)
  - [Digest Pipeline (сбор статей)](#1-digest-pipeline-сбор-статей)
  - [Knowledge Graph (граф знаний)](#2-knowledge-graph-граф-знаний)
  - [Article Pipeline (написание статьи)](#3-article-pipeline--pipeline-написания-статьи)
- [Принципы работы](#-принципы-работы)
- [Критерии оценки](#-критерии-оценки)
- [Кодовая база](#-кодовая-база)
- [Быстрый старт](#-быстрый-старт)
- [API Reference](#api-reference)
- [Технологический стек](#-технологический-стек)
- [Roadmap](#roadmap)
- [Лицензия](#-лицензия)

---

## 🎯 Концепция

**GEO-Digest** — автономная исследовательская платформа, которая:

1. **Собирает** академические статьи из **7 академических источников** (OpenAlex, Semantic Scholar, DOAJ, CoreAC, arXiv, CrossRef, Europe PMC) по тематическим запросам на русском и английском
2. **Дедуплицирует**, **скорит** и **обогащает** их через LLM (резюме на русском языке)
3. **Строит граф знаний** — семантические связи между статьями, выявленные через LLM
4. **Генерирует научно-популярные статьи** через multi-agent pipeline: Editor → Reader → Writer → Reviewer
5. **Доставляет** результаты через Web Dashboard с интерактивным графом знаний

### Ключевая идея

> Исследователь задаёт тему → система находит релевантные статьи → LLM-агенты анализируют, синтезируют и пишут статью → независимый reviewer проверяет качество → пользователь получает готовый материал с графом связей источников.

### Тематические кластеры (7 тем)

| Кластер | Примеры запросов |
|---------|-----------------|
| 🌊 **Каспий/Чёрное море** | Caspian Sea level change, Black Sea ecology |
| 🏔️ **Кавказские экосистемы** | Caucasus biodiversity, mountain ecosystems |
| 💧 **Водные ресурсы** | water quality monitoring, river basins |
| 🌡️ **Изменение климата** | climate change impact Southern Russia |
| 🦠 **Биоразнообразие** | species invasion, endemic species |
| ⛰️ **Геоморфология** | coastal erosion, landslide hazards |
| 🔬 **Методы мониторинга** | remote sensing, GIS, environmental DNA |

---

## 🏗️ Архитектура

### Микросервисы v2

```
┌──────────┐     ┌──────────────┐     ┌─────────────────┐
│  Nginx   │────▶│  Dashboard   │────▶│     Worker      │
│  (:443)  │◀────│  (:3000)     │◀────│     (:3001)     │
│  Caddy   │     │  Pure UI +   │     │  Compute Engine │
└──────────┘     │  API Gateway │     │                 │
                 └──────┬───────┘     └────────┬────────┘
                         │                       │
                 geo-data volume          ┌──────┴──────┐
                 (shared storage)         │             │
                                      digest.py    build_graph.py
                                      (сбор)       (граф)
                                           │             │
                                      sources/      llm_client.py
                                      (7 API)       (MiniMax M2.7)
                                           │             │
                                           └──────┬──────┘
                                                  │
                                          ┌───────┴───────┐
                                          │    engine/     │
                                          │                │
                                          │ orchestrator_v2│ ← State Machine
                                          │  ┌────────────┐│
                                          │  │   agents/  ││
                                          │  │ editor     ││ ← B+ Hybrid
                                          │  │ reader     ││ ← MarkItDown PDF
                                          │  │ writer     ││ ← MiniMax M2.7
                                          │  │ reviewer   ││ ← OpenRouter/Gemini
                                          │  │ scout      ││ ← Группировка
                                          │  └────────────┘│
                                          │  llm/          │ ← Provider layer
                                          │  tools/        │ ← 14 tools (FC)
                                          │  storage/      │ ← JSONL backend
                                          │  prompts/      │ ← System prompts
                                          └────────────────┘
```

### Компоненты

| Компонент | Порт | Роль | Технология |
|-----------|------|------|------------|
| **Dashboard** | 3000 | SPA UI + API Gateway | Flask + Vanilla JS |
| **Worker** | 3001 | Compute backend | FastAPI + Background threads |
| **Nginx/Caddy** | 443 | Reverse proxy, SSL termination | Caddy |

---

## 🔄 Пайплайны

### 1. Digest Pipeline (сбор статей)

**Скрипт:** `scripts/digest.py` (~1380 строк)

```
Topic Queries (7 clusters × lang)
        │
        ▼
  ┌─────────────────────────────────────────┐
  │         SEARCH PHASE (7 sources)         │
  │  OpenAlex · S2 · DOAJ · CoreAC · arXiv  │
  │  CrossRef · Europe PMC                   │
  └────────────────┬────────────────────────┘
                   ▼
  ┌─────────────────────────────────────────┐
  │         DEDUP PHASE                     │
  │  DOI normalization → seen_dois.txt      │
  │  Title hash (SHA256) → fuzzy match      │
  └────────────────┬────────────────────────┘
                   ▼
  ┌─────────────────────────────────────────┐
  │         SCORING PHASE (6 criteria)       │
  │  ┌─────────────────────────────────────┐ │
  │  │ Recency      (weight: 0.20) newer=better │ │
  │  │ Citations   (weight: 0.25) more=better  │ │
  │  │ OpenAccess  (weight: 0.15) OA=bonus     │ │
  │  │ SourceTrust (weight: 0.20) DOAJ > arXiv  │ │
  │  │ TopicMatch  (weight: 0.10) query relevance│ │
  │  │ JournalIF   (weight: 0.10) IF bonus       │ │
  │  └─────────────────────────────────────┘ │
  └────────────────┬────────────────────────┘
                   ▼
  ┌─────────────────────────────────────────┐
  │         LLM ENRICHMENT                  │
  │  GLM / MiniMax → RU summary             │
  │  4 секции: объяснение, находки,         │
  │           региональная релевантность,   │
  │           план действий                 │
  └────────────────┬────────────────────────┘
                   ▼
            articles.jsonl  (DB)
```

#### Источники данных (7 API)

| Источник | Файл | Вес | Особенности |
|----------|-------|-----|-------------|
| **DOAJ** | `sources/doaj.py` | **1.08** | Peer-reviewed OA journals (выший вес!) |
| **Semantic Scholar** | `sources/semantic_scholar.py` | **1.10** | Лучшие abstracts + citation data |
| **Core AC** | `sources/core_ac.py` | **1.05** | Verified OA content |
| **OpenAlex** | `sources/openalex.py` | **1.05** | ~100M OA статей, бесплатный |
| **CrossRef** | `sources/crossref.py` | **1.00** | Enrich-only (DOI lookup) |
| **arXiv** | `sources/arxiv.py` | **0.95** | Preprints (не peer-reviewed) |
| **Europe PMC** | `sources/europe_pmc.py` | **1.05** | Biomedical/биология фокус |

### 2. Knowledge Graph (граф знаний)

**Скрипт:** `scripts/build_graph.py` (~908 строк)

```
articles.jsonl (enriched articles)
        │
        ▼
  ┌───────────────────────────────────────┐
  │   PAIRWISE SEMANTIC ANALYSIS          │
  │   Для каждой пары статей → LLM:       │
  │   "Есть ли содержательная связь?"     │
  │   Тип: citation/methods/topic/data    │
  │   Сила: 0.1–1.0                       │
  └────────────────┬──────────────────────┘
                   ▼
  ┌───────────────────────────────────────┐
  │   GRAPH CONSTRUCTION                 │
  │   Nodes: статьи (doi = id)           │
  │   Edges: семантические связи          │
  │   Layout: Cytoscape.js (cose)        │
  └────────────────┬──────────────────────┘
                   ▼
          graph_data.json (Cytoscape format)
                   │
                   ▼
  ┌───────────────────────────────────────┐
  │   GRAPH ANALYTICS                    │
  │   • PageRank centrality (hubs)       │
  │   • Louvain communities (clusters)   │
  │   • Bridge nodes (межкластерные)     │
  │   • Shortest paths                   │
  └───────────────────────────────────────┘
```

**Формат графа:** Cytoscape.js JSON (`elements: { nodes: [], edges: [] }`)

**Режимы построения:**
- `full` — полный пересчёт всех пар (O(n²))
- `no-llm` — только цитатные связи (без LLM)
- `update` — инкрементальный (только новые статьи)

### 3. Article Pipeline (Pipeline написания статьи)

**Оркестратор:** `engine/orchestrator_v2.py` (~900 строк)

```
┌──────────────────────────────────────────────────────────────────┐
│                    ARTICLE PIPELINE (B+ Hybrid)                   │
│                                                                   │
│  User Topic                                                      │
│      │                                                           │
│      ▼                                                           │
│  ┌─────────┐                                                     │
│  │ EDITOR  │ ◄── B+ Hybrid Architecture                          │
│  │ Agent   │     Phase 0: EvidencePack (Python loader)           │
│  │         │     Phase 1: Discovery (LLM explores, ≤4 rounds)    │
│  │ ~640 loc│     Phase 2: Synthesize (LLM proposes, 1 round)     │
│  │         │     Phase 3: Validate (DOI gate + confidence)       │
│  └────┬────┘                                                     │
│       │ 1–3 proposals                                            │
│       ▼  (user selects)                                         │
│  ┌─────────┐     ┌──────────────────────────────────────┐        │
│  │ READER  │────▶│  StructuredDraft                     │        │
│  │ Agent   │     │  • gap_identified                    │        │
│  │ ~380 loc│     │  • proposed_contribution              │        │
│  │         │     │  • methods_summary                   │        │
│  │ PDF via │     └──────────────────────────────────────┘        │
│  │MarkItDown│                                                    │
│  └────┬────┘                                                     │
│       ▼                                                          │
│  ┌─────────┐     ┌──────────────────────────────────────┐        │
│  │ WRITER  │────▶│  WrittenArticle                     │        │
│  │ Agent   │     │  Markdown/LaTeX format               │        │
│  │ ~300 loc│     │  Academic/blog/popular style         │        │
│  │         │     │  RU or EN language                  │        │
│  └────┬────┘     └──────────────────────────────────────┘        │
│       │                                                          │
│       ▼  ◄──┐                                                   │
│  ┌─────────┤  │  Revision Loop (max 3 rounds)                   │
│  │REVIEWER │──┘                                                   │
│  │ Agent   │     7 категорий оценки:                             │
│  │ ~650 loc│     structure · factual_accuracy · academic_style   │
│  │         │     citations · coherence · tone_match · format     │
│  │OpenRouter│    Verdict: ACCEPT / NEEDS_REVISION / REJECT       │
│  │Gemini   │     REJECT ≠ dead end → triggers rewrite!           │
│  └────┬────┘                                                     │
│       │                                                          │
│       ▼                                                          │
│  ✅ DONE — Article ready with review history                     │
└──────────────────────────────────────────────────────────────────┘
```

#### State Machine (Orchestrator v2)

```
idle → editing → selecting → developing → writing → written → reviewing → done
  │                                                              │
  └────────────────── cancelled / error / failed ────────────────┘

Revision loop:
  writing ←── NEEDS_REVISION / REJECT (round < 3)
  writing ←── forced_accept (round == 3, max reached)
```

---

## ⚙️ Принципы работы

### 1. B+ Hybrid Editor (Phase 5 innovation)

Замена хаотичного tool-use loop на структурированный research→synthesis пайплайн:

| Метрика | OLD (ToolUseLoop ×9) | NEW (B+ Hybrid) |
|---------|---------------------|-----------------|
| LLM раундов | 9 | 2–5 |
| Токены | ~104K | ~25K |
| Время | ~190 sec | 50–90 sec |
| Качество | Хаотичные вызовы tools | Структурированное исследование |

**Dataclasses B+:**
- `EvidencePack` — все статьи в компактном виде + статистика графа + доменная статистика
- `DiscoveryReport` — отчёт исследования: ключевые находки, выбранные DOI, кросс-связи, гипотезы о пробелах
- `ArticleCard` — компактное представление статьи для LLM-контекста
- `GraphSummary` — предвычисленная статистика графа (hubs, bridges, communities)
- `DomainStats` — статистика хранилища (total, sources, years, topics)

### 2. Tool Use (Function Calling)

Editor Agent использует **14 tools** через Anthropic-style function calling:

**Storage Tools (7):**
| Tool | Описание |
|------|----------|
| `search_articles` | Поиск по ключевым словам |
| `get_article_detail` | Детали статьи по DOI |
| `validate_doi` | Валидация формата DOI |
| `find_similar_existing` | Поиск похожих статей |
| `cluster_by_subtopic` | Кластеризация по подтеме |
| `count_storage_stats` | Статистика хранилища |
| `explore_domain` | Обзор доменной области |

**Graph Tools (7):**
| Tool | Описание |
|------|----------|
| `graph_neighbors` | Соседние узлы в графе |
| `graph_path` | Кратчайший путь между статьями |
| `graph_hubs` | PageRank/degree centrality |
| `graph_clusters` | Louvain communities |
| `graph_cross_topic` | Мосты между темами ★ |
| `graph_centrality` | Важность узла |
| `graph_stats` | Общая статистика графа |

### 3. Graceful Fallbacks

Система предпочитает авто-восстановление вместо hard failures:

- Нет статьи для ревью → сначала запустить write
- REJECT от reviewer → перезапись (не failure!)
- Unstructured JSON от LLM → heuristic fallback парсинг
- Max revision rounds → forced accept с предупреждением
- Пустой storage → 1 proposal + warning (не пустой результат)

### 4. Human-in-the-Loop

Пользователь участвует в ключевых точках:
1. **Выбор proposal** — Editor генерирует 1–3 варианта, пользователь выбирает
2. **Approval gates** — между этапами pipeline
3. **Review feedback** — можно отправить на доработку вручную

---

## 📊 Критерии оценки

### Scoring Digest (6 критериев)

| Критерий | Вес | Описание |
|----------|-----|----------|
| **Recency** | 0.20 | Новизна статьи (новее = лучше) |
| **Citations** | 0.25 | Цитируемость (больше = лучше) |
| **Open Access** | 0.15 | Бонус за открытый доступ |
| **Source Trust** | 0.20 | Надёжность источника (DOAJ > arXiv) |
| **Topic Match** | 0.10 | Релевантность запросу |
| **Journal Impact** | 0.10 | Импакт-фактор журнала |

### Scoring Proposals (Editor)

| Критерий | Описание |
|----------|----------|
| **Confidence** | 0.0–1.0, уверенность редактора в качестве proposal |
| **DOI Coverage** | % key_references с валидными DOI |
| **Discovery Depth** | deep (≥60% refs explored) / medium / shallow |
| **Novelty** | Оценка новизны (gap_filled) |
| **Feasibility** | Оценка реализуемости (estimated_sections) |

### Scoring Review (Reviewer v2 — 7 категорий)

| Категория | Описание |
|-----------|----------|
| **Structure** | IMRaD, разделы, логика |
| **Factual Accuracy** | Фактологическая корректность |
| **Academic Style** | Академический стиль языка |
| **Citations** | Корректность ссылок |
| **Coherence** | Связность narrative |
| **Tone Match** | Соответствие паттерну типа статьи |
| **Format Compliance** | Соответствие формату журнала |

**Verdict:** `ACCEPT` / `NEEDS_REVISION` / `REJECT`
**Strictness:** автоматически регулируется (review=5, short_communication=4, original_research=3)

---

## 📁 Кодовая база

### Структура проекта

```
GEO-Digest/
├── config.yaml                      # Главный конфиг (источники, веса, темы)
├── pyproject.toml                   # Пакет geo-digest v1.0.0
├── docker-compose.yml               # 2 сервиса: dashboard + worker
├── Dockerfile.dashboard             # UI контейнер
├── Caddyfile                        # Reverse proxy
├── .env.example                     # Шаблон переменных окружения
├── ROADMAP.md                       # Дорожная карта (8+ спринтов)
│
├── dashboard/                       # Web-интерфейс + API Gateway
│   ├── app.py                       # FastAPI сервер (:3000), proxy → Worker
│   └── templates/index.html         # SPA (~3500 строк): граф, карточки, pipeline UI
│
├── worker/                          # Compute backend
│   ├── server.py                    # Worker API (:3001), 8 endpoints
│   ├── dal.py                       # Data Access Layer (CRUD articles/graph/jobs)
│   └── Dockerfile                   # Python 3.12 + markitdown[pdf]
│
├── engine/                          # ★ Ядро — pip-пакет
│   ├── __init__.py                  # Public API, re-экспорт schemas
│   ├── config.py                    # Загрузчик конфигурации (singleton get_config())
│   ├── schemas.py                   # Все data-модели (~500 строк)
│   ├── cli.py                       # CLI (geo-digest command)
│   ├── api.py                       # REST API для Engine (:3002)
│   ├── orchestrator.py              # Orchestrator v1 (legacy)
│   ├── orchestrator_v2.py           # ★ Orchestrator v2 (state machine + review loop)
│   ├── utils.py                     # Утилиты (hash, truncate, duration)
│   │
│   ├── agents/                      # Multi-agent pipeline
│   │   ├── base.py                  # ABC BaseAgent + LLMCallMixin
│   │   ├── scout.py                 # Scout — поиск и группировка статей
│   │   ├── editor.py                # ★ Editor — B+ Hybrid (EvidencePack → Discovery → Synthesize → Validate)
│   │   ├── reader.py                # Reader — PDF → StructuredDraft (MarkItDown)
│   │   ├── writer.py                # Writer — StructuredDraft → WrittenArticle
│   │   ├── reviewer.py              # ★ Reviewer v2 — rubric engine, 7 categories, round-aware
│   │   ├── tools.py                 # Shared utilities (PDF cache, URL fetch)
│   │   └── article_patterns.py      # Паттерны типов статей + REVISION_CONFIG
│   │
│   ├── llm/                         # LLM провайдеры
│   │   ├── base.py                  # ABC LLMProvider
│   │   ├── minimax.py               # MiniMax M2.7 провайдер
│   │   ├── openai_compat.py         # OpenAI-compatible (для OpenRouter/Gemini)
│   │   ├── config.py                # Конфигурация провайдеров
│   │   ├── response_parser.py       # Парсер LLM ответов (JSON extraction)
│   │   └── tool_loop.py             # ★ ToolUseLoop — цикл function calling
│   │
│   ├── prompts/                     # System prompts
│   │   └── editor_prompts.py        # DISCOVERY + SYNTHESIZE prompts (RU)
│   │
│   ├── tools/                       # LLM-callable tools (function calling)
│   │   ├── base.py                  # Tool Registry framework (@registry.tool())
│   │   ├── storage_tools.py         # 7 Storage Tools (~700 строк)
│   │   └── graph_tools.py           # 7 Graph Tools (~838 строк)
│   │
│   └── storage/                     # Backend хранения
│       ├── base.py                  # ABC StorageBackend
│       └── jsonl_backend.py         # JSONL реализация
│
├── scripts/                         # Скрипты сбора данных
│   ├── digest.py                    # ★ Главный digest-пайплайн (~1380 строк)
│   ├── build_graph.py               # Построение графа знаний (~908 строк)
│   ├── run_manager.py               # Менеджер запусков (background + status)
│   ├── llm_client.py                # Unified LLM клиент (MiniMax M2.7)
│   ├── setup-server.sh              # Setup-скрипт деплоя
│   ├── migrate_add_ids.py           # Migration: canonical_id
│   ├── graph_analytics.py           # Аналитика графа (centrality, communities)
│   └── sources/                     # 7 академических API клиентов
│       ├── __init__.py              # Auto-discovery источников
│       ├── base.py                  # ABC SourceSearcher
│       ├── openalex.py              # OpenAlex API
│       ├── semantic_scholar.py      # Semantic Scholar API
│       ├── core_ac.py               # Core AC API
│       ├── doaj.py                  # DOAJ API
│       ├── arxiv.py                 # arXiv API
│       ├── crossref.py              # CrossRef API
│       └── europe_pmc.py            # Europe PMC API
│
├── tests/                           # Тесты (~5000 строк)
│   ├── test_e2e.py                  # E2E базовый поток
│   ├── test_e2e_full_pipeline.py    # Полный E2E pipeline
│   ├── test_editor_agent.py         # Editor Agent (45 тестов, 8 классов)
│   ├── test_editor_api.py           # REST API editor endpoints
│   ├── test_graph_integration.py    # Интеграционные тесты графа
│   ├── test_graph_tools.py          # 7 Graph Tools
│   ├── test_llm_tool_interface.py   # ToolUseLoop + parser
│   ├── test_orchestrator_v2.py      # State machine v2
│   ├── test_response_parser.py      # Парсер LLM ответов
│   ├── test_storage_tools.py        # 7 Storage Tools
│   ├── test_tool_loop.py            # ToolUseLoop цикл
│   ├── test_tool_registry.py        # ToolRegistry
│   └── fixtures/                    # Тестовые данные
│       ├── sample_articles.jsonl    # 15+ тестовых статей
│       └── sample_tool_response.json
│
└── plans/                           # Планы разработки
    ├── full-pipeline-integration.md  # SSOT план интеграции
    └── tool-use-editor/              # Детальные планы S0-S8
        ├── OVERVIEW.md
        ├── README.md
        ├── S0-infrastructure.md
        ├── S1-tool-executor.md
        ├── S2-tool-loop.md
        ├── S3-editor-agent.md
        ├── S4-api-endpoints.md
        ├── S5-ui-editor.md
        ├── S6-orchestrator-refactor.md
        ├── S7-e2e-finalize.md
        └── S8-graph-integration.md
```

### Статистика кода

| Компонент | Файлов | ~Строк | Роль |
|-----------|--------|--------|------|
| **engine/** | 24 | ~4,800 | Ядро: агенты, LLM, tools, storage |
| **worker/** | 3 | ~2,560 | Compute API, DAL |
| **dashboard/** | 2 | ~4,150 | SPA frontend + gateway |
| **scripts/** | 14 | ~4,200 | Сбор данных, 7 источников |
| **tests/** | 14 | ~5,000 | Тесты (327 passing) |
| **plans/** | 10 | ~1,700 | Документация разработки |
| **Config/Docker** | 10 | ~400 | Инфраструктура |
| **ИТОГО** | **~77** | **~22,800** | |

### Ключевые файлы (звёздочкой выделены наиболее архитектурно значимые)

| Файл | Роль | Почему важен |
|------|------|--------------|
| `config.yaml` | Конфигурация | Single source of truth для всего проекта |
| `engine/orchestrator_v2.py` | State machine | Сердце pipeline — состояния, переходы, review loop |
| `engine/agents/editor.py` | Editor Agent | B+ Hybrid — главная инновация проекта |
| `engine/agents/reviewer.py` | Reviewer v2 | Независимый критик с rubric engine |
| `engine/llm/tool_loop.py` | ToolUseLoop | Function calling engine |
| `engine/tools/storage_tools.py` | Storage Tools | 7 tools для LLM-доступа к данным |
| `engine/tools/graph_tools.py` | Graph Tools | 7 tools для LLM-доступа к графу |
| `scripts/digest.py` | Digest pipeline | Сбор статей из 7 источников |
| `scripts/build_graph.py` | Graph builder | Построение графа знаний |
| `worker/server.py` | Worker API | 8 HTTP endpoints |
| `dashboard/templates/index.html` | Frontend | Весь UI в одном файле |

---

## 🚀 Быстрый старт

### Требования

- **Docker & Docker Compose**
- **Python ≥ 3.10** (для локальной разработки)
- **API ключи:** `MINIMAX_API_KEY`, `OPENROUTER_API_KEY` (опционально)

### 1. Клонирование и настройка

```bash
git clone https://github.com/Godila/GEO-Digest.git
cd GEO-Digest

# Копировать шаблон переменных окружения
cp .env.example .env
# Отредактировать .env — вставить свои API ключи
nano .env
```

### 2. Запуск через Docker Compose (рекомендуется)

```bash
# Сборка и запуск обоих контейнеров
docker compose build --no-cache worker dashboard
docker compose up -d

# Проверить статус
docker compose ps
# Должно показать: dashboard (healthy, port 3000), worker (healthy, port 3001)
```

### 3. Локальный запуск (для разработки)

```bash
# Установить зависимости
pip install -e ".[dev,pdf]"

# Запустить Worker (backend)
cd worker && uvicorn server:app --port 3001 --reload &

# Запустить Dashboard (frontend)
cd dashboard && python app.py &

# Открыть http://localhost:3000
```

### 4. Запуск digest-пайплайна

```bash
# Через CLI
geo-digest run --topic "Caspian Sea level change"

# Через API
curl -X POST http://localhost:3000/api/digest/run \
  -H "Content-Type: application/json" \
  -d '{"topics": ["caspian sea"]}'
```

### 5. Запуск article pipeline

```bash
# Создать job на написание статьи
curl -X POST http://localhost:3000/api/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Изменение уровня Каспийского моря"}'

# → {"job_id": "20260428_143052_a1b2c3", "state": "editing"}

# Получить список jobs
curl -s http://localhost:3000/api/pipeline/jobs

# Выбрать proposal
curl -X POST http://localhost:3000/api/pipeline/jobs/{JOB_ID}/select \
  -H "Content-Type: application/json" \
  -d '{"proposal_id": "prop_..."}'

# Запустить чтение PDF (develop)
curl -X POST http://localhost:3000/api/pipeline/jobs/{JOB_ID}/develop

# Написать статью (write)
curl -X POST http://localhost:3000/api/pipeline/jobs/{JOB_ID}/write

# Отправить на ревю
curl -X POST http://localhost:3000/api/pipeline/jobs/{JOB_ID}/review
```

---

## 📡 API Reference

### Digest API (Worker :3001, proxied through Dashboard :3000)

| Method | Endpoint | Описание |
|--------|----------|----------|
| `POST` | `/api/digest/run` | Запустить digest-пайплайн |
| `GET` | `/api/digest/status/{run_id}` | Статус запуска |
| `GET` | `/api/digest/history` | История запусков |
| `GET` | `/api/digest/runs` | Список запусков |

### Graph API

| Method | Endpoint | Описание |
|--------|----------|----------|
| `POST` | `/api/graph/rebuild` | Перестроить граф |
| `GET` | `/api/graph/status` | Статус графа |
| `GET` | `/api/graph/data` | Данные графа (JSON) |
| `GET` | `/api/graph/stats` | Статистика графа |

### Pipeline API (Article Generation)

| Method | Endpoint | Описание |
|--------|----------|----------|
| `POST` | `/api/pipeline/run` | Создать job (запуск Editor) |
| `GET` | `/api/pipeline/jobs` | Список всех jobs |
| `GET` | `/api/pipeline/jobs/{id}` | Детали job |
| `POST` | `/api/pipeline/jobs/{id}/select` | Выбрать proposal |
| `POST` | `/api/pipeline/jobs/{id}/develop` | Запустить Reader (PDF) |
| `POST` | `/api/pipeline/jobs/{id}/write` | Запустить Writer |
| `POST` | `/api/pipeline/jobs/{id}/review` | Запустить Reviewer |
| `DELETE` | `/api/pipeline/jobs/{id}` | Отменить job |

### Articles API

| Method | Endpoint | Описание |
|--------|----------|----------|
| `GET` | `/api/articles` | Все статьи (пагинация) |
| `GET` | `/api/articles/{doi}` | Статья по DOI |
| `GET` | `/api/articles/search?q=` | Поиск по ключевым словам |

### Health

| Method | Endpoint | Описание |
|--------|----------|----------|
| `GET` | `/api/health` | Health check (Worker) |

---

## 🔧 Технологический стек

### Core
- **Python 3.12** — основной язык
- **FastAPI** — Worker API (async-ready)
- **Flask** — Dashboard Gateway
- **Vanilla JS** — Frontend (self-contained HTML)

### LLM / AI
- **MiniMax M2.7** — primary LLM (Editor, Writer, Enrichment)
- **OpenRouter + Gemini 3.1 Flash Lite** — Reviewer (independent critic)
- **ToolUseLoop (Anthropic-style)** — function calling engine
- **MarkItDown 0.1.x** — PDF/DOCX/PPTX текстовая экстракция

### Data & Storage
- **JSONL** — основное хранилище статей (`articles.jsonl`)
- **JSON** — граф (`graph_data.json`), jobs, metadata
- **Cytoscape.js** — визуализация графа знаний

### Infrastructure
- **Docker & Docker Compose** — контейнеризация
- **Caddy** — reverse proxy, SSL, static caching
- **Nginx** — альтернатива Caddy

### Academic Sources (API)
- OpenAlex, Semantic Scholar, DOAJ, CoreAC, arXiv, CrossRef, Europe PMC
- **Unpaywall** — Open Access enrichment (DOI → OA URL)

### Development
- **pytest** — тестирование (327 tests passing)
- **ruff** — linting
- **pydantic** — валидация данных

---

## 🗺️ Roadmap

### Выполненные фазы

| Фаза | Описание | Статус |
|------|----------|--------|
| **P0** | Bugfixes (critical fixes) | ✅ Complete |
| **P1** | HTTP API (8 endpoints) | ✅ Complete |
| **P2** | Reader PDF (MarkItDown integration) | ✅ Complete |
| **P3** | UI Integration (pipeline modal, status bar) | ✅ Complete |
| **P4** | Proactive Reviewer v2 (rubric engine, 7 cats) | ✅ Complete |
| **P5** | B+ Hybrid Editor (EvidencePack→Discovery→Synthesize) | ✅ Complete |
| **P5b** | Bug #34 + UI Feedback (WRITTEN state) | ✅ Complete |

### Планируемые фазы

| Фаза | Описание | Статус |
|------|----------|--------|
| **P6** | Full browser E2E test | 🔲 Planned |
| **P7** | Graph tools activation in Discovery phase | 🔲 Planned |
| **P8** | Multi-language support (EN articles) | 🔲 Planned |
| **P9** | Scheduled digests (cron auto-run) | 🔲 Planned |
| **P10** | Export formats (DOCX, PDF, LaTeX) | 🔲 Planned |

Подробности: [`ROADMAP.md`](ROADMAP.md) и [`plans/full-pipeline-integration.md`](plans/full-pipeline-integration.md)

---

## 📐 Принципы дизайна

1. **Graceful Degradation** — любой компонент может недоступен, система продолжит работу с ограниченной функциональностью
2. **Independent Critic** — Reviewer использует отдельный LLM (Gemini) для объективной оценки
3. **Human-in-the-Loop** — ключевые решения принимает пользователь (выбор proposal, approval)
4. **Tool-First Architecture** — LLM управляется через structured tools, не free-form промпты
5. **File-based Persistence** — JSONL + JSON, без БД (просто бэкап, migrate, debug)
6. **Observable** — каждый этап логируется, статус доступен через API в real-time
7. **Cost-effective** — B+ Hybrid сократил токены с 104K до 25K на editing phase

---

## 🤝 Contributing

1. Fork проект
2. Create feature branch (`git checkout -b feature/amazing`)
3. Write tests (`pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing`)
6. Open Pull Request

### Тесты

```bash
# Запустить все тесты
pytest tests/ -v

# Только Editor Agent
pytest tests/test_editor_agent.py -v

# Только Graph Tools
pytest tests/test_graph_tools.py -v
```

---

## 📄 Лицензия

MIT License — см. [`LICENSE`](LICENSE) файл.

---

<p align="center">
  <strong>GEO-Digest</strong> — Research Intelligence for Geo-Ecology<br>
  <sub>Built with ❤️ for the Southern Russia research community</sub>
</p>
