# GEO-Digest Agent Engine — Roadmap

**Цель:** GEO-Digest → Полуфабрикат статьи → Approval → Черновик → Review → Готовая статья  
**Стек:** Python 3.11+, FastAPI, MiniMax (primary) + OpenAI-compatible (reviewer), PyMuPDF/Grobid (PDF)  
**Принцип:** Engine = standalone pip-пакет. Hermes = один из фронтендов (только для разработки).

---

## Sprint 1: Фундамент Engine

**Цель:** Создать каркас, который можно импортировать и запускать anywhere.

### 1.1 Структура проекта + Base классы

**Файлы:**
```
engine/
├── __init__.py              # Публичный API: from geo_digest import ...
├── config.py                # ConfigLoader (yaml/env/defaults)
├── schemas.py               # Pydantic модели (Article, Draft, Job, etc.)
├── llm/
│   ├── __init__.py
│   ├── base.py              # ABC LLMProvider: complete(prompt) → str
│   ├── minimax.py           # MiniMaxHTTPProvider (существующий клиент)
│   └── openai_compat.py     # OpenAI-compatible (для reviewer/других)
├── storage/
│   ├── __init__.py
│   ├── base.py              # ABC StorageBackend
│   └── jsonl_backend.py     # Текущий articles.jsonl + graph_data.json
├── agents/
│   ├── __init__.py
│   ├── base.py              # ABC BaseAgent + AgentResult
│   ├── registry.py          # Регистрация агентов по имени
│   └── tools.py             # Общие инструменты (search, fetch, parse)
└── orchestrator.py          # State machine для пайплайнов
```

**Acceptance:**
- [ ] `from engine import Orchestrator, ScoutAgent` — работает
- [ ] `LLMProvider.complete("Hello")` → строка (MiniMax)
- [ ] `StorageBackend.load_articles()` → list[Article]
- [ ] `BaseAgent.run(**kwargs) -> AgentResult` — интерфейс определён

### 1.2 Миграция существующего кода в engine

**Что переносим:**
- `scripts/llm_client.py` → `engine/llm/minimax.py` (обёртка)
- `worker/dal.py` → частично в `engine/storage/jsonl_backend.py`
- `scripts/sources/*` → реюзаем как есть (import)

**Acceptance:**
- [ ] `engine.llm.minimax.MiniMaxClient` работает с существующим MINIMAX_API_KEY
- [ ] `engine.storage.JsonlStorage` читает текущий `/app/data/articles.jsonl`
- [ ] Существующий `digest.py` НЕ ломается (backward compat)

---

## Sprint 2: Scout Agent — умный поиск и группировка

**Цель:** Не просто "найти статьи", а "найти потенциал для статьи".

### 2.1 Расширенный поиск

**Новые источники (поверх существующих openalex/doaj/arxiv/core_ac):**
- **Semantic Scholar** — citation count, citation context, related papers
- **CrossRef** — метаданные, real DOI resolution, funder info

**Файл:** `engine/agents/scout.py`

**Вход:**
```python
ScoutInput(
    topic="машинное обучение для прогноза землетрясений",
    time_range="2023-2025",
    min_citations=5,
    max_results=50,
    languages=["en"],
)
```

**Выход (новый!):**
```python
ScoutResult(
    groups=[
        ArticleGroup(
            group_type="replication",       # или "review" / "data_paper"
            title_suggestion="ML-методы для сейсмической инверсии",
            confidence=0.85,
            articles=[Article(...), ...],
            rationale="DispFormer + SafeNet используют схожие подходы...",
        ),
        ArticleGroup(
            group_type="review",
            title_suggestion="Обзор DL в геофизике 2023-2025",
            confidence=0.78,
            articles=[...],
            rationale="4 обзорные статьи покрывают 300+ работ...",
        ),
    ],
    total_found=2542,
    after_dedup=1600,
)
```

**Логика группировки (ключевая фича):**
```
1. Кластеризация по:
   - Общим темам (topic_key)
   - Общим методам (из title/abstract NLP extraction)
   - Citation overlap (кто цитирует кого)

2. Классификация группы:
   - REPLICATION: 1-3 методические статьи с кодом/данными
   - REVIEW: 2+ обзора + много первичных исследований
   - DATA_PAPER: статьи про датасеты

3. Оценка потенциала (confidence):
   - Наличие OA PDF (+0.2)
   - Наличие кода на GitHub (+0.15)
   - Свежесть (+0.1 за год)
   - Citation momentum (+0.1)
   - Gap identified (+0.15)
```

**Acceptance:**
- [ ] `scout.run(topic="...")` → 3-8 групп статей
- [ ] Каждая группа имеет type, confidence, rationale
- [ ] Группа "replication" содержит data_requirements (что нужно для повтора)
- [ ] Semantic Scholar интегрирован (или заглушка с TODO)

### 2.2 REST endpoint для scout

**Endpoint:** `POST /api/agent/scout`
**Возвращает:** job_id → polling → результат

**Acceptance:**
- [ ] `curl -X POST /api/agent/scot -d '{"topic": "..."}'` → job_id
- [ ] `GET /api/agent/{job}/status` → running/done + progress %
- [ ] `GET /api/agent/{job}/result` → JSON со ScoutResult

---

## Sprint 3: Reader Agent — глубокое чтение

**Цель:** Превратить "список статей" в "структурированные полуфабрикаты".

### 3.1 PDF pipeline

**Файл:** `engine/agents/reader.py`

**Pipeline на статью:**
```
DOI → Unpaywall (OA URL?) 
    → YES: download PDF
    → NO:  fallback to Grobid / abstract only
    
PDF → PyMuPDF extract text
    → (опц.) Grobid → structured XML (tables, refs, formulas)
    
Text + Metadata → LLM extraction prompt
    → StructuredDraft (JSON schema)
```

**StructuredDraft schema (ключевой объект!):**
```python
@dataclass
class StructuredDraft:
    draft_id: str
    group_type: Literal["replication", "review", "data_paper"]
    source_articles: list[str]  # DOIs
    
    # --- Общее для всех типов ---
    title_suggestion: str
    abstract_suggestion: str
    keywords: list[str]
    gap_identified: str              # "чего не хватает в поле"
    proposed_contribution: str        # "что можно добавить"
    confidence: float                 # 0..1
    estimated_effort: str             # "2 недели / месяц"
    
    # --- Для replication ---
    methods_summary: str              # что делают
    architecture: str | None          # ML архитектура (если есть)
    data_requirements: DataReqs | None
    infrastructure_needs: InfraNeeds | None
    code_availability: str | None     # GitHub link / нет кода
    metrics: dict[str, Any] | None    # accuracy, F1, RMSE...
    baseline_comparison: str | None    # vs SOTA
    reproducibility_score: float      # 0..1
    
    # --- Для review ---
    scope: str | None                 # что охватывает обзор
    articles_covered: int             # сколько работ анализирует
    methodology: str | None           # как отбирались работы
    trends_identified: list[str]      # какие тренды видны
    
    # --- Для data_paper ---
    dataset_description: str | None
    access_method: str | None         # URL / request / FTP
    format: str | None                # NetCDF / GeoTIFF / CSV
    size_gb: float | None
    coverage: str | None              # geographic / temporal
    usage_examples: list[str] | None


@dataclass
class DataReqs:
    input_data: str                   # "seismic reflection cubes, 3D volumes"
    data_format: str                  # "SEG-Y"
    volume_estimate: str              # "~50GB для типичного survey"
    acquisition: str                  # "2D/3D seismic survey"
    preprocessing: list[str]          # ["noise reduction", "NMO correction"]
    labels_available: bool            # есть ли ground truth
    split_strategy: str               # "80/10/10 by survey"


@dataclass
class InfraNeeds:
    hardware: str                     # "GPU 8GB+ VRAM"
    software: list[str]               # ["PyTorch 2.0+", "Madagascar (Seismic Unix)"]
    compute_time_estimate: str        # "2-3 дня training"
    storage: str                      # "~100GB для данных + модели"
    expertise_required: list[str]     # ["geophysics basics", "PyTorch", "signal processing"]
```

**LLM prompt для extraction (пример):**
```
Ты — научный аналитик. Проанализируй статью и извлеки:

1. ЧТО ДЕЛАЮТ (метод): 2-3 предложения
2. ДАННЫЕ НА ВХОДЕ: что нужно, формат, объём
3. ИНФРАСТРУКТУРА: GPU, ПО, время обучения
4. РЕЗУЛЬТАТЫ: метрики, сравнение с baseline
5. КОД И ДАННЫЕ: ссылки на GitHub/датасет
6. ВОЗМОЖНОСТЬ РЕПЛИКАЦИИ: оценка 0-5, почему
7. GAP: чего не хватает, что можно улучшить
8. МОЙ ВКЛАД: если бы я применял это к данным Чёрного моря, 
   что бы я сделал (конкретно)

Ответ строго в формате JSON по схеме.
```

**Acceptance:**
- [ ] `reader.read(doi="10.xxxx")` → StructuredDraft
- [ ] Для ML-статей: заполнены architecture, metrics, code_availability
- [ ] Для обзорных статей: заполнены scope, trends_identified
- [ ] data_requirements содержит конкретику (не общие фразы)
- [ ] Работает без PDF (fallback на abstract + metadata)

### 3.2 Batch чтение (группа статей)

**Метод:** `reader.read_group(group: ArticleGroup) -> GroupDraft`

Читает каждую статью в группе, затем **агрегирует** через LLM:
```
"Вот данные по N статьям. Синтезируй общий полуфабрикат:
 - Какие методы пересекаются?
 - Какие данные нужны для всех?
 - Какая общая gap?
 - Предложи структуру будущей статьи."
```

**Acceptance:**
- [ ] `reader.read_group(group)` → агрегированный GroupDraft
- [ ] Агрегация выявляет пересечения между статьями

---

## Sprint 4: Approval Gate + State Machine

**Цель:** Точка принятия решения. Human-in-the-loop.

### 4.1 Job state machine

**Файл:** `engine/orchestrator.py`

```
Состояния задачи:
  CREATED → SCOUTING → SCOUT_DONE ⏸️
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                APPROVED   SKIPPED  REJECTED
                    │                            
                    ▼                            
                READING → READ_DONE ⏸️
                                  │
                        ┌─────────┼─────────┐
                        ▼         ▼         ▼
                    APPROVED   SKIPPED  REJECTED
                        │
                        ▼
                    WRITING → WRITE_DONE ⏸️
                                        │
                                  ┌─────┴─────┐
                                  ▼           ▼
                              APPROVE     REVISE
                                  │           │
                                  ▼           │
                               REVIEWING ←───┘
                                  │
                                  ▼
                              REVIEW_DONE ✅
```

**Хранение состояния:**
```python
# /app/data/agent_jobs/{job_id}.json
{
    "job_id": "agent_20260425_130000",
    "status": "scout_done",           # current state
    "created_at": "2026-04-25T13:00:00Z",
    "updated_at": "2026-04-25T13:15:00Z",
    
    "input": {
        "topic": "ML для прогноза землетрясений",
        "user_comment": "",
    },
    
    "results": {
        "scout": { ... },            # ScoutResult
        "read": null,                # GroupDraft (when done)
        "write": null,               # ArticleDraft (when done)
        "review": null,              # ReviewedDraft (when done)
    },
    
    "approval_history": [
        {"stage": "scout", "action": "approve", "group_index": 0, "at": "..."},
    ],
}
```

### 4.2 REST API для approval

**Endpoints:**
```
POST /api/agent/run              → создать задачу (полный пайплайн)
POST /api/agent/run?pipeline=scout → только scout
GET  /api/agent/{job_id}          → состояние задачи
GET  /api/agent/{job_id}/result/{stage} → результат этапа
POST /api/agent/{job_id}/approve → утвердить текущий этап
                                 Body: {"group_index": 0, "comment": "акцент на Black Sea"}
POST /api/agent/{job_id}/skip    → пропустить этап
POST /api/agent/{job_id}/reject  → отклонить (откат или стоп)
POST /api/agent/{job_id}/revise  → отправить на доработку
                                   Body: {"feedback": "больше деталей по методу"}
GET  /api/agent/jobs              → список всех задач
GET  /api/agent/jobs/active       → активные задачи
```

**Acceptance:**
- [ ] POST /api/agent/run → job создан, статус = scouting
- [ ] Scout завершён → статус автоматически = scout_done (пауза!)
- [ ] POST approve → переходит к reading
- [ ] POST approve (после read) → writing
- [ ] POST approve (после write) → reviewing
- [ ] Review done → статус = complete
- [ ] Можно skip любой этап
- [ ] Можно reject → задача останавливается
- [ ] User comment пробрасывается в следующий агент

---

## Sprint 5: Writer Agent

**Цель:** Превратить утверждённый draft в черновик статьи.

### 5.1 Шаблоны статей

**Файл:** `engine/agents/writer.py`

**Поддерживаемые форматы вывода:**

| Тип статьи | Структура | Когда использовать |
|-----------|----------|-------------------|
| **Review article** | Intro → Taxonomy → By-method → Trends → Gaps → Future work | Группа типа "review" |
| **Method replication** | Intro → Original method → My adaptation → Expected results → Risks | Группа типа "replication" |
| **Data paper** | Intro → Dataset description → Methods → Results → Availability | Группа типа "data_paper" |
| **Short communication** | Intro → Key finding → Implications → 1 figure | Быстрая публикация |

**Шаблон (пример для replication):**
```markdown
# {title}

## Abstract
{auto-generated from draft}

## 1. Introduction
- Почему эта тема важна (из scout rationale)
- Что сделали оригинальные авторы (из methods_summary)
- Что хочу сделать я (из proposed_contribution + user_comment)

## 2. Background & Related Work
- 2.1 {method_category} overview
- 2.2 Original approach: {architecture summary}
- 2.3 Applications in geosciences

## 3. Methodology
- 3.1 Original method (детально, из reader output)
- 3.2 Proposed modifications
- 3.3 Data description ({data_requirements + user region context})
- 3.4 Implementation plan

## 4. Expected Results & Evaluation
- 4.1 Metrics (из baseline_comparison)
- 4.2 Comparison with original
- 4.3 Risk assessment

## 5. Discussion
- 5.1 Limitations (из gaps_identified)
- 5.2 Future work
- 5.3 Broader impact

## References
[1] {first_article_citation}
[2] {second_article_citation}
...

---
*Generated by GEO-Digest Agent Engine*
*Source articles: N papers reviewed*
*Confidence: {confidence}*
```

### 5.2 LLM Prompt для writer

```
Ты — научный писатель. Написать черновик статьи по СЛЕДУЮЩИМ данным:

== ТИП СТАТЬИ: {group_type} ==
== ЗАГОЛОВОК: {title_suggestion} ==
== ПОЛЬЗОВАТЕЛЬСКИЙ КОММЕНТАРИЙ: {user_comment} ==

== ИСТОЧНИКИ СТАТЕЙ (уже прочитаны):
{structured_drafts_from_reader}

== ТРЕБОВАНИЯ:
- Научный стиль, без воды
- Конкретные цифры и факты из источников
- Цитировать как [1], [2]...
- Если чего-то нет в источниках — написать [NEED RESEARCH]
- Длина: ~4000 слов для review, ~2500 для replication
- Язык: русский (научный) или английский (на выбор)

== ВЫВОД: Markdown текст статьи
```

**Acceptance:**
- [ ] `writer.write(draft, style="review", lang="ru")` → Markdown строка
- [ ] `writer.write(draft, style="replication", lang="en")` → English Markdown
- [ ] Библиография формируется автоматически из DOIs
- [ ] [NEED RESEARCH] маркирует места где не хватает данных
- [ ] User_comment учитывается в тексте ("акцент на Чёрном море")

---

## Sprint 6: Reviewer Agent

**Цель:** Другая модель проверяет и исправляет черновик.

### 6.1 Второй LLM провайдер

**Файл:** `engine/llm/openai_compat.py`

```python
# Поддерживает любой OpenAI-compatible API:
# - OpenAI GPT-4
# - Anthropic Claude (через proxy)
# - DeepSeek
# - Ollama (локально)
# - Together AI
# - Groq

config.yaml:
  reviewer:
    provider: openai_compat
    base_url: "https://api.openai.com/v1"  # или другой
    model: "gpt-4o"
    api_key_env: "OPENAI_API_KEY"
```

**Почему другая модель:**
- MiniMax пишет → GPT-4/Claude проверяет (разные "взгляды")
- Разные модели делают разные ошибки → пересечение = качество
- Можно сравнить два review'ра

### 6.2 Логика reviewer'а

**Файл:** `engine/agents/reviewer.py`

**Что проверяет:**

| Категория | Что ищет | Пример исправления |
|-----------|---------|-------------------|
| **Нейросленг** | Клише ИИ-текстов | "отметим, что важно отметить" → "следует отметить" |
| **Факты** | Соответствие источникам | "статья X говорит Y" → проверить |
| **Галлюцинации** | Выдуманные цитаты/факты | Удалить или пометить [VERIFY] |
| **Научный стиль** | Пассивный залог, точность | "мы сделали" → "было проведено" |
| **Структура** | IMRaD compliance | Переставить секции если надо |
| **Логика** | Выводы следуют из данных | "Therefore" — есть ли связь? |
| **Цитирования** | Правильность ссылок | Проверить что [1] реально говорит это |
| **Язык** | Грамматика, терминология | Исправить ошибки |

**Выход:**
```python
ReviewedDraft(
    original_text: str,
    revised_text: str,
    edits: list[Edit],             # что изменилось
    issues: list[Issue],           # найденные проблемы
    severity_counts: dict,         # {"critical": 0, "major": 3, "minor": 12}
    fact_checks: list[FactCheck],  # проверки фактов
    verdict: str,                   # "ACCEPT / ACCEPT_WITH_MINOR / NEEDS_REVISION / REJECT",
    overall_score: float,           # 0..1
)
```

**Prompt:**
```
Ты — строгий scientific reviewer. Проверь черновик статьи.

Критерии:
1. ФАКТЫ: каждая претензия к источнику должна быть верна
2. ЯЗЫК: научный стиль, без клише ИИ-генерации  
3. ЛОГИКА: выводы следуют из предпосылок
4. СТРУКТУРА: IMRaD compliance
5. ЦИТИРОВАНИЕ: корректность [N] ссылок

Для каждой проблемы укажи:
- location: "section 3.2, paragraph 2"
- severity: critical/major/minor
- original: "..."
- suggested: "..."
- reason: "why"

Выведи: исправленный текст + список правок.
```

**Acceptance:**
- [ ] `reviewer.review(markdown_text)` → ReviewedDraft
- [ ] Использует отдельный LLM (не MiniMax writer'а)
- [ ] Выделяет нейросленг конкретными примерами
- [ ] Проверяет факты против source_articles
- [ ] verdict + overall_score помогают принять решение

---

## Sprint 7: CLI + Packaging

**Цель:** `pip install geo-digest` → работает везде.

### 7.1 CLI entry point

**Файл:** `cli.py` + `pyproject.toml`

```bash
# Полный пайплайн (с approval pauses)
$ geo-digest run --topic "ML землетрясения" --lang ru
> Scout found 5 groups. Opening approval...
> [0] Replication: DispFormer seismic inversion (conf: 0.87)
> [1] Review: DL in geophysics 2023-2025 (conf: 0.82)
> [2] Data: TRIMS LST dataset China (conf: 0.71)
> Select group [0-2 or skip]: 0
> Comment (optional): применить к данным Чёрного моря
> Reading 3 articles...
> Writing draft...
> Reviewing with gpt-4o...
> Done! Output: output/agent_xxx/article.md

# Только поиск
$ geo-digest scout --query "landslide susceptibility mapping" --since 2024

# Только чтение
$ geo-digest read --doi 10.3390/rs15071857 --full

# Только написание (из сохранённого draft)
$ geo-digest write --draft output/agent_xxx/draft.json --style replication

# Только ревью
$ geo-digest review --input article.md --model gpt-4o

# Статус задач
$ geo-digest jobs --active
$ geo-digest show agent_xxx

# Конфигурация
$ geo-digest init          # создать .env + config.yaml
$ geo-digest config show   # показать текущую конфигурацию
```

### 7.2 pyproject.toml

```toml
[project]
name = "geo-digest"
version = "1.0.0"
description = "Agent-powered research digest → article drafting pipeline"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.104",
    "uvicorn>=0.24",
    "httpx>=0.25",
    "pydantic>=2.5",
    "pymupdf>=1.23",        # PDF parsing
    "pyyaml>=6.0",
    "rich>=13.0",           # CLI beauty
    "click>=8.1",           # CLI framework
]

[project.scripts]
geo-digest = "cli:main"

[tool.setuptools.packages.find]
include = ["engine*", "worker*", "scripts*"]
```

**Acceptance:**
- [ ] `pip install -e .` устанавливает пакет
- [ ] `geo-digest --help` работает
- [ ] `geo-digest run --topic "..."` запускает полный цикл
- [ ] Approval работает интерактивно в терминале
- [ ] Вывод сохраняется в `output/{job_id}/`

---

## Sprint 8: Интеграция + E2E тест + Docker

**Цель:** Всё работает вместе, деплоится одной командой.

### 8.1 E2E тест (реальный прогон)

**Сценарий:**
```
1. geo-digest run --topic "coastal erosion Black Sea" --lang ru
2. Scout находит 4 группы (из существующих 71 статьи + новые)
3. Выбираем группу [1] (review type)
4. Reader читает 5 статей → structured draft
5. Writer генерирует черновик (~3000 слов)
6. Reviewer (если настроен) проверяет
7. Итог: output/{job}/article.md
```

**Acceptance:**
- [ ] Полный цикл от темы до markdown-файла
- [ ] Черновик содержит библиографию
- [ ] Черновик имеет структуру IMRaD
- [ ] Нет явных галлюцинаций (проверка 3 случайных цитат)
- [ ] Время выполнения < 15 минут (для 5-10 статей)

### 8.2 Docker обновление

**docker-compose.yml изменения:**
```yaml
services:
  worker:
    # Уже есть FastAPI :3001
    # Добавляются endpoints: /api/agent/*
    environment:
      - REVIEWER_PROVIDER=openai_compat
      - REVIEWER_MODEL=gpt-4o
      - OPENAI_API_KEY=${OPENAI_API_KEY}   # опционально
  
  # CLI можно запускать внутри контейнера:
  # docker compose exec worker geo-digest run --topic "..."
```

**Acceptance:**
- [ ] `docker compose up --build` поднимает всё
- [ ] `/api/agent/run` доступен извне (через порт или только внутренний)
- [ ] Dashboard показывает agent jobs (новая вкладка?)
- [ ] Логи пайплайнов доступны в `/app/runs/`

---

## Приоритет и зависимости

```
Sprint 1 (Foundation)
  └─ должен быть первым, всё остальное зависит от него
  └─ ~3-4 часа работы

Sprint 2 (Scout) ─┬─ можно параллельно с Sprint 3
  └─ зависит от Sprint 1
  └─ ~3 часа

Sprint 3 (Reader) ─┤
  └─ зависит от Sprint 1
  └─ ~4 часа (самый сложный, PDF pipeline)

Sprint 4 (Approval Gate)
  └─ зависит от Sprint 2 + 3 (нужны их результаты)
  └─ ~2 часа

Sprint 5 (Writer)
  └─ зависит от Sprint 3 (нужен StructuredDraft)
  └─ ~2-3 часа

Sprint 6 (Reviewer)
  └─ зависит от Sprint 5 (нужен черновик)
  └─ ~2-3 часа

Sprint 7 (CLI + Package)
  └─ зависит от Sprint 5 (минимально рабочий пайплайн)
  └─ ~2 часа

Sprint 8 (E2E + Docker)
  └─ зависит от всего остального
  └─ ~2 часа
```

**Итого: ~20-24 часа чистой работы.** При нашей скорости — 2-3 сессии.

## Что НЕ входит в roadmap (будущее):

- Phase 1: SQLite миграция (можно сделать позже, JSONL работает)
- Web UI для внешних пользователей (не нужно, personal tool)
- Email/Telegram delivery (уже есть в digest.py)
- Multi-user / SaaS (anti-goal)
- Автоматическая публикация (human always in loop)

---

*Последнее обновление: 2026-04-25*
*Статус: Planning → Ready to start Sprint 1*
