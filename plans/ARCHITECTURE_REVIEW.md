# 🏛️ ARCHITECTURE REVIEW — GEO-Digest

**Дата:** 2026-04-28  
**Ревьюер:** Architect Agent (Hermes)  
**Версия кода:** post-Writer Quality Overhaul (Stages 1–4)

---

## 📊 Оценка: 7.5 / 10

| Компонент | Оценка | Комментарий |
|-----------|--------|-------------|
| **Архитектура** | 9/10 | B+ Hybrid, state machine, tool-first — зрелая архитектура |
| **Digest Pipeline** | 9/10 | 7 источников, scoring, dedup, LLM enrichment — production-ready |
| **Knowledge Graph** | 8/10 | Cytoscape.js, analytics, интеграция с tools — хорошо |
| **Editor Agent** | 9/10 | EvidencePack→Discovery→Synthesize→Validate — инновационно |
| **Reader Agent** | 7/10 | rich_context добавлен, но НЕ вызывается в pipeline |
| **Writer Agent** | 6/10 | Хорошие промпты, но JSON→text баг + Reader bypass |
| **Reviewer Agent** | 7/10 | 7 категорий rubric engine, но лёгкая модель (Gemini Flash Lite) |
| **Оркестрация** | 6/10 | State machine работает, но Reader не интегрирован в write flow |
| **Тесты** | 8/10 | 332/336 pass, хорошее покрытие агентов |
| **UI/Dashboard** | 7/10 | Функционален, но group_type dropdown не подключён |
| **Инфраструктура** | 8/10 | Docker, API, proxy — хорошо для личного проекта |
| **Документация** | 9/10 | README 400+ строк, ROADMAP, API reference — отлично |

---

## ✅ Что работает отлично

### 1. B+ Hybrid Editor — главная инновация
Сокращение 104K→25K токенов, 190s→50-90s — это реальный engineering win.
EvidencePack предзагружает контекст, LLM только принимает решения.

### 2. Digest Pipeline — production quality
7 академических API, scoring с 6 критериями, dedup через DOI + title hash,
Unpaywall enrichment — полноценный data pipeline.

### 3. Tool-First Architecture
14 tools (7 storage + 7 graph) через function calling — это правильный подход.
LLM не угадывает, а вызывает tools для получения точных данных.

### 4. Article Quality (когда Writer получает данные)
Сгенерированная статья «Транзиентная деформация земной коры»:
- 1773 слова, 7 разделов
- Таблицы с метриками (RMSE, AUC-ROC)
- Конкретные цифры и ссылки
- Структура IMRaD
- Обсуждение ограничений

Это **значительно лучше** чем было до Writer Quality Overhaul.

---

## 🐛 КРИТИЧЕСКИЕ БАГИ (блокируют 10/10)

### Баг #1: `call_llm(parse_json=True)` возвращает STRING, не dict
**Файл:** `engine/agents/base.py:123-124`  
**Симптом:** `complete_json()` добавляет «Respond ONLY with valid JSON» к system prompt и вызывает `self.complete()`, который возвращает **сырую строку**. `call_llm()` **НЕ вызывает `json.loads()`** на результате.

**Следствие:** Writer `_parse_written()` получает **строку** (JSON-текст), а не dict → идёт по ветке `isinstance(raw, str)` → `WrittenArticle.text = raw_json_string` вместо собранного markdown.

**В stored article:** `text` = JSON-объект как строка, `sections=[]`, `references=[]`.

**Fix:**
```python
# engine/agents/base.py
def call_llm(self, prompt, system="", max_tokens=0, temperature=0.3, parse_json=False):
    if parse_json:
        raw = self.llm.complete_json(prompt, system=system, max_tokens=max_tokens or 4096)
        import json
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw  # fallback to string
    return self.llm.complete(prompt, system=system, max_tokens=max_tokens or 4096, temperature=temperature)
```

**Приоритет:** 🔴 CRITICAL — без этого Writer не может корректно разобрать JSON от LLM

---

### Баг #2: Reader НЕ вызывается в article pipeline
**Файл:** `engine/orchestrator_v2.py:251-293`  
**Симптом:** Pipeline flow = `edit → select → write → review`. Метод `develop()` (который вызывает Reader) **пропускается** — он вызывается только через отдельный API endpoint.

**Следствие:** Writer получает **synthetic StructuredDraft** без `rich_context`. Используется fallback: abstract[:800] от 5 источников. Это как писать диссертацию по аннотациям вместо полных текстов.

**Fix (вариант A — авто-вызов Reader в write):**
```python
# В методе write() перед созданием synthetic draft:
if needs_synthetic and proposal.get("key_references"):
    try:
        dois = self._extract_dois(proposal["key_references"])
        result = self.reader.run(dois=dois, topic=job.topic)
        draft_data = result.data if hasattr(result, 'data') else result
        logger.info(f"[orch] Auto-invoked Reader for {len(dois)} DOIs")
    except Exception as e:
        logger.warning(f"[orch] Reader failed, using synthetic: {e}")
        # fall through to synthetic draft creation
```

**Fix (вариант B — pipeline flow):** Добавить develop как обязательный шаг:
`edit → select → develop → write → review`

**Приоритет:** 🔴 CRITICAL — без Reader статья пишется по аннотациям

---

### Баг #3: group_type не передаётся через pipeline API
**Файл:** `worker/server.py` → `orchestrator_v2.py`  
**Симптом:** UI dropdown `pipeline-group-type` отправляет group_type, но он **не доходит** до `PipelineJob` и не передаётся в `write()`.

**Следствие:** Writer использует эвристику (3 источника → data_paper) вместо типа, выбранного пользователем.

**Fix:** Передавать group_type из request в PipelineJob и в `_resolve_group_type()`.

**Приоритет:** 🟡 HIGH — пользовательский выбор игнорируется

---

## ⚠️ ВАЖНЫЕ УЛУЧШЕНИЯ (для перехода 7.5 → 9/10)

### 4. Reviewer использует слишком лёгкую модель
**Текущее:** `google/gemini-3.1-flash-lite-preview` — это самая лёгкая модель Gemini.  
**Проблема:** Reviewer принял статью за 1 раунд с score 0.85. Нет должной строгости.

**Fix:** Использовать `gemini-2.5-flash` или `gemini-2.5-pro` для review. Или хотя бы увеличить strictness level.

### 5. Editor ToolUseLoop лимит 4 раунда
**Текущее:** `ToolUseLoop stopped after 4 rounds` — Editor нашёл только 5 статей.  
**Проблема:** Для review статьи нужно 20-30 источников, для research — 10-15.

**Fix:** Увеличить лимит до 6-8 раундов для Editor Discovery phase. Или добавить adaptive stopping (когда LLM не находит новых статей 2 раунда подряд).

### 6. Writer промпт теряет часть информации
**Текущее:** Writer получает StructuredDraft с DOIs, но без PDF-контента.  
**Проблема:** `_build_source_info()` fallback показывает только abstract[:800].

**Fix (после исправления Бага #2):** rich_context от Reader даст Writer полноценный анализ каждого источника.

---

## 💡 АРХИТЕКТУРНЫЕ РЕКОМЕНДАЦИИ (для 10/10)

### A. Multi-Pass Writer (Draft → Expand → Polish)
Сейчас Writer делает одну генерацию. Для длинных статей (review: 4000-7000 слов) это недостаточно.

**Предложение:**
1. **Draft pass** — Skeleton с ключевыми тезисами (используя Writer промпт)
2. **Expand pass** — Для каждой секции генерируется полный текст отдельно
3. **Polish pass** — Финальная шлифовка (связность, переходы, ссылки)

Это позволит писать статьи по 5000+ слов без потери качества.

### B. Reader → Writer Data Contract
Сейчас Reader и Writer слабо связаны через StructuredDraft. Нужен явный contract:

```python
class WriterContext:
    """Полный контекст для Writer."""
    source_analyses: list[SourceAnalysis]  # от Reader
    gap_analysis: str                       # от Editor proposal
    key_findings: list[str]                 # из EvidencePack
    methodology_patterns: list[str]         # из graph tools
    regional_relevance: str                 # из config topics
```

### C. Регистрация LLM ответов для debugging
Сейчас нет логирования промптов и ответов LLM. Для итерации на качестве это критично.

```python
# В base.py call_llm:
logger.debug(f"[LLM] prompt_len={len(prompt)}, system_len={len(system)}, max_tokens={max_tokens}")
logger.debug(f"[LLM] response_len={len(raw)}, is_json={parse_json}")
```

### D. Human-in-the-Loop для quality gates
Сейчас пользователь выбирает proposal, но не может:
- Уточнить фокус статьи перед написанием
- Дать feedback на draft перед review
- Выбрать какие источники включить/исключить

Добавить шаги: `edit → select → **user_focus** → develop → **user_approve_sources** → write → review → **user_final**`

---

## 🗺️ ROADMAP: 7.5 → 10/10

| Приоритет | Задача | Effort | Impact |
|-----------|--------|--------|--------|
| 🔴 P0 | Fix `call_llm(parse_json=True)` — добавить json.loads() | 5 min | Writer перестаёт хранить JSON как текст |
| 🔴 P0 | Авто-вызов Reader в write() | 30 min | Writer получает rich_context |
| 🟡 P1 | Передача group_type через pipeline | 15 min | Пользовательский выбор работает |
| 🟡 P1 | Усиление Reviewer модели | 10 min | Более строгий и полезный review |
| 🟢 P2 | Multi-Pass Writer (draft→expand→polish) | 2-3 часа | Статьи по 5000+ слов |
| 🟢 P2 | LLM prompt/response logging | 30 min | Видимость для итерации |
| 🔵 P3 | WriterContext data contract | 1 час | Чистая архитектура Reader→Writer |
| 🔵 P3 | Adaptive ToolUseLoop stopping | 1 час | Больше источников для Editor |
| ⚪ P4 | Multi-step human-in-the-loop | 2-3 часа | Полный контроль качества |

---

## 🏁 SUMMARY

**GEO-Digest — это впечатляющий проект** с продуманной архитектурой, инновационным B+ Hybrid Editor, и качественным Digest Pipeline. Кодовая база ~22,800 строк организована логично, с хорошим разделением ответственности.

**Главный затык** — Writer Quality Overhaul был реализован на 80%:
- ✅ Промпты переписаны (writer_prompts.py)
- ✅ Dynamic max_tokens добавлен
- ✅ Rich context спроектирован
- ❌ Но Reader не вызывается в pipeline → rich_context пустой
- ❌ `call_llm(parse_json=True)` не парсит JSON → статья хранится как JSON-строка
- ❌ group_type не доходит от UI до Writer

**После исправления 3 критических багов (P0 + P1) — оценка поднимется до 8.5/10.**  
**После Multi-Pass Writer — до 9/10.**  
**С полным data contract + enhanced human-in-the-loop — 10/10.**
