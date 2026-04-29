# Writer Quality Overhaul — Implementation Plan

> **Для Hermes:** Используй subagent-driven-development skill для реализации.

**Цель:** Превратить Writer из генератора коротких сухих текстов в полноценного автора научных статей с глубиной, фактами и структурой.

**Архитектура:** 3 корневых фикса (информационный поток, лимиты токенов, подключение article_patterns) + 3 поддерживающих фикса (bug fixes, транслит, UI dropdown).

**Tech Stack:** Python 3.12, MiniMax M2.7 API, OpenRouter/Gemini (reviewer)

**Базовый путь:** `/home/hermeswebui/.hermes/geo_digest/`

---

## STAGE 1: Rich Data Flow — Reader → Writer (Корень 1)

Цель: Writer получает развёрнутый контекст вместо 200-символьных обрубков.

### Task 1.1: Добавить поле `rich_context` в StructuredDraft

**Файл:** `engine/schemas.py` (~строка 83, класс StructuredDraft)

**Что сделать:**
- Добавить параметр `rich_context=""` в `__init__`
- Сохранять как `self.rich_context`
- Включить в `to_dict()` и `to_json()`

```python
# В __init__ добавить:
rich_context="",
# ...
self.rich_context = rich_context

# В to_dict() добавить:
"rich_context": self.rich_context,
```

### Task 1.2: Создать Rich Analysis Prompt для Reader

**Файл:** `engine/agents/reader.py` (новый промпт после строки 47)

**Что сделать:** Добавить `RICH_READER_SYSTEM_PROMPT` — промпт для глубокого анализа, который извлекает НЕ суммари, а конкретные факты:

```python
RICH_READER_SYSTEM_PROMPT = """Ты — научный аналитик геоэкологических исследований.
Проводишь ДЕТАЛЬНЫЙ анализ статей для написания новой научной работы.

Извлеки из каждой статьи:
1. КЛЮЧЕВЫЕ ФАКТЫ с цифрами: точные значения, проценты, размеры выборок, p-values
2. МЕТОДОЛОГИЮ подробно: какие данные, период, территория, модели, ПО
3. РЕЗУЛЬТАТЫ с числами: корреляции, тренды, статистика
4. ПРОТИВОРЕЧИЯ между работами: где авторы расходятся
5. GAP в знаниях — отдельно для каждого с контекстом
6. ЦИТАТЫ: 3-5 ключевых утверждений verbatim для использования в статье

Верни JSON:
{
  "key_facts": [{"claim": "...", "source_doi": "...", "evidence": "..."}],
  "methods_detail": [{"method": "...", "data_source": "...", "period": "...", "tools": "..."}],
  "results_with_numbers": [{"finding": "...", "value": "...", "comparison": "..."}],
  "contradictions": [{"claim_a": "...", "author_a": "...", "claim_b": "...", "author_b": "..."}],
  "gaps": [{"gap": "...", "context": "...", "who_noted": "..."}],
  "verbatim_quotes": [{"quote": "...", "source_doi": "...", "section": "..."}],
  "cross_connections": [{"articles": ["DOI1", "DOI2"], "connection": "..."}]
}"""
```

### Task 1.3: Реализовать метод `_build_rich_context` в ReaderAgent

**Файл:** `engine/agents/reader.py` (после `_analyze_multiple`, ~строка 276)

**Что сделать:** Новый метод, который вызывает LLM с RICH_READER_SYSTEM_PROMPT для каждой статьи и агрегирует результат:

```python
def _build_rich_context(self, extracted: dict, group_type: GroupType) -> str:
    """Build rich context string for Writer from detailed article analysis."""
    all_analyses = []
    for key, data in extracted.items():
        art = data["article"]
        text = data["text"]
        if not text:
            continue
        
        prompt = f"""Статья: {art.title}
Авторы: {art.authors or 'N/A'}
DOI: {art.doi or 'N/A'}
Год: {art.year or 'N/A'}

Текст статьи:
{text[:12000]}

Проведи детальный анализ для написания {group_type.value} статьи."""
        
        raw = self.call_llm(
            prompt=prompt,
            system=RICH_READER_SYSTEM_PROMPT,
            max_tokens=4096,
            parse_json=True,
        )
        
        if isinstance(raw, dict):
            analysis = self._format_rich_analysis(art, raw)
            all_analyses.append(analysis)
    
    return "\n\n---\n\n".join(all_analyses) if all_analyses else ""
```

### Task 1.4: Подключить rich_context в pipeline Reader → Writer

**Файл:** `engine/agents/reader.py` (метод `run`, ~строка 106)

**Что сделать:** После `_analyze_with_llm()` добавить вызов `_build_rich_context()` и записать в draft:

```python
# В методе run(), после строки draft = self._analyze_with_llm(...):
draft.rich_context = self._build_rich_context(extracted, 
    group.group_type if group else GroupType.REVIEW)
```

### Task 1.5: Writer использует rich_context вместо 200-символьных обрубков

**Файл:** `engine/agents/writer.py`, метод `_build_source_info` (строка 135)

**Что сделать:** Если draft.rich_context есть — использовать его. Если нет — fallback к текущему:

```python
def _build_source_info(self, draft: StructuredDraft, tools: AgentTools) -> str:
    # Если есть rich_context от Reader — используем его
    if draft.rich_context:
        return f"=== ДЕТАЛЬНЫЙ АНАЛИЗ ИСТОЧНИКОВ ===\n\n{draft.rich_context}"
    
    # Fallback: старая логика (title + 200 chars abstract)
    parts = []
    for i, doi in enumerate(draft.source_articles[:10], 1):
        art = tools.search_by_doi(doi)
        if art:
            part = f"{i}. {art.title}"
            if art.authors:
                part += f" ({art.authors})"
            if art.abstract:
                # УВЕЛИЧЕНО: было 200, стало 800 символов
                part += f"\n   {art.abstract[:800]}"
            if art.llm_summary:
                part += f"\n   LLM-резюме: {art.llm_summary[:500]}"
            parts.append(part)
        else:
            parts.append(f"{i}. DOI: {doi} (не найдена)")
    
    return "\n\n".join(parts) if parts else "(No source articles specified)"
```

---

## STAGE 2: Writer Prompt Overhaul + article_patterns (Корень 3)

Цель: Writer знает, что такое хорошая статья, следует структуре и стилю.

### Task 2.1: Создать `engine/prompts/writer_prompts.py`

**Новый файл.** Генерация динамических промптов для Writer на основе article_patterns.py.

```python
"""Writer Prompts — динамические промпты на основе article_patterns.py."""
from engine.agents.article_patterns import (
    EXPECTED_SECTIONS, TONE_RULES, SECTION_TRANSITIONS,
    ARTICLE_TYPES,
)
from engine.schemas import GroupType


def build_writer_system_prompt(group_type: GroupType, format_: str, language: str) -> str:
    """Dynamic system prompt with full structural guidance."""
    type_key = _group_type_to_article_type(group_type)
    type_info = ARTICLE_TYPES.get(type_key, ARTICLE_TYPES["original_research"])
    sections_guidance = _format_sections_for_type(type_info)
    tone = _format_tone_rules()
    transitions = _format_transitions()
    
    lang_label = "русском" if language == "ru" else "English"
    
    return f"""Ты — опытный научный писатель в области геоэкологии, геологии и экологических наук.
Пишешь полноценные научные статьи на {lang_label} языке в формате {format_}.

== ТИП СТАТЬИ ==
{type_info['label']}
Ключевые критерии качества: {', '.join(type_info['key_criteria'])}

== ОБЯЗАТЕЛЬНАЯ СТРУКТУРА СЕКЦИЙ ==
{sections_guidance}

== СТИЛИСТИЧЕСКИЕ ПРАВИЛА ==
{tone}

== ФРАЗЫ ДЛЯ ПЕРЕХОДОВ МЕЖДУ СЕКЦИЯМИ ==
{transitions}

== ТРЕБОВАНИЯ К ЦИТИРОВАНИЮ ==
- КАЖДОЕ утверждение подкреплено ссылкой: [Автор и др., год] или (Author et al., Year)
- Для review: минимум 20-30 источников
- Для original research: минимум 10-15 источников  
- Баланс: классические работы + свежие (последние 5 лет)
- DOI указан для каждого источника если доступен

== ФОРМАТ ВЫВОДА ==
Верни JSON (без markdown-обёрток):
{{"title": "...", "sections": [{{"heading": "...", "content": "..."}}], "references": ["..."], "word_count": N}}

ВАЖНО:
- content каждой секции должен быть ПОЛНЫМ текстом, не наброском
- Каждый абзац содержит конкретные факты, цифры, ссылки
- НЕ пиши общих фраз без содержания — каждое предложение несёт информацию"""


def build_target_word_count(group_type: GroupType) -> dict:
    """Return target word count range for article type."""
    type_key = _group_type_to_article_type(group_type)
    limits = {
        "review": {"min": 4000, "max": 7000, "target": 5500},
        "original_research": {"min": 3000, "max": 5000, "target": 4000},
        "short_communication": {"min": 1000, "max": 2000, "target": 1500},
        "data_paper": {"min": 1500, "max": 3000, "target": 2000},
    }
    return limits.get(type_key, limits["original_research"])


def build_length_instruction(group_type: GroupType, language: str) -> str:
    """Build explicit length instruction for the prompt."""
    wc = build_target_word_count(group_type)
    lang_label = "слов" if language == "ru" else "words"
    return f"""== ЦЕЛЕВОЙ ОБЪЁМ ==
Минимум: {wc['min']} {lang_label}
Целевой: {wc['target']} {lang_label}
Максимум: {wc['max']} {lang_label}

Каждая секция должна быть содержательной:
- Введение: 3-5 абзацев (контекст → проблема → цель → новизна)
- Методы: 3-5 абзацев (данные → алгоритмы → параметры → верификация)
- Результаты: 4-8 абзацев (конкретные цифры, сравнения, таблицы)
- Обсуждение: 3-6 абзацев (интерпретация → сравнение → ограничения → перспективы)
- Заключение: 1-2 абзаца (резюме + значимость)"""


def _group_type_to_article_type(group_type: GroupType) -> str:
    mapping = {
        GroupType.REVIEW: "review",
        GroupType.REPLICATION: "original_research",
        GroupType.DATA_PAPER: "data_paper",
    }
    return mapping.get(group_type, "original_research")


def _format_sections_for_type(type_info: dict) -> str:
    type_sections = type_info.get("sections", [])
    result = []
    for sec_def in EXPECTED_SECTIONS:
        if sec_def["id"] in type_sections or sec_def.get("required"):
            line = f"• {sec_def['label']}"
            if "word_range" in sec_def:
                line += f" ({sec_def['word_range'][0]}-{sec_def['word_range'][1]} слов)"
            if "structure" in sec_def:
                line += ":\n  " + "\n  ".join(f"→ {s}" for s in sec_def["structure"])
            if "tone_markers" in sec_def:
                markers = "; ".join(sec_def["tone_markers"][:4])
                line += f"\n  Типичные фразы: {markers}"
            if "critical_rule" in sec_def:
                line += f"\n  ⚠ {sec_def['critical_rule']}"
            result.append(line)
    return "\n\n".join(result)


def _format_tone_rules() -> str:
    forbidden = "\n".join(f"  ✗ {r}" for r in TONE_RULES["forbidden"])
    required = "\n".join(f"  ✓ {r}" for r in TONE_RULES["required"])
    return f"""Регистр: {TONE_RULES['register']}
Лицо: {TONE_RULES['person']}

ЗАПРЕЩЕНО:
{forbidden}

ОБЯЗАТЕЛЬНО:
{required}"""


def _format_transitions() -> str:
    lines = []
    for key, phrases in SECTION_TRANSITIONS.items():
        lines.append(f"{key}:")
        for p in phrases:
            lines.append(f"  • {p}")
    return "\n".join(lines)
```

### Task 2.2: Переписать WriterAgent на использование новых промптов

**Файл:** `engine/agents/writer.py`

**Что сделать:**
1. Удалить старые `WRITER_SYSTEM_PROMPT` и `WRITER_PROMPT` (строки 25-60)
2. Импортировать `from engine.prompts.writer_prompts import build_writer_system_prompt, build_target_word_count, build_length_instruction`
3. Переписать метод `run()` для использования динамических промптов
4. Переписать `_build_prompt()` с подключением length_instruction

```python
# Новый run() — ключевые изменения:
def run(self, draft=None, style="academic", language="ru", format_="markdown",
        user_comment="", **kwargs):
    # ... (валидация draft)
    
    tools = AgentTools(self.storage)
    source_info = self._build_source_info(draft, tools)
    
    # Dynamic system prompt from article_patterns
    system_prompt = build_writer_system_prompt(draft.group_type, format_, language)
    
    # Build user prompt with length instructions
    prompt = self._build_prompt(draft, source_info, style, language, format_, user_comment)
    
    # Token limit per article type
    target = build_target_word_count(draft.group_type)
    max_tokens = min(target["target"] * 2, 16384)  # ~2 tokens per word for RU
    
    raw_article = self.call_llm(
        prompt=prompt,
        system=system_prompt,
        max_tokens=max_tokens,
        parse_json=True,
        temperature=0.4,
    )
    # ... (parsing)
```

### Task 2.3: Переписать _build_prompt() с length instruction

**Файл:** `engine/agents/writer.py`, метод `_build_prompt`

```python
def _build_prompt(self, draft, source_info, style, language, format_, user_comment):
    from engine.prompts.writer_prompts import build_length_instruction
    
    length_instruction = build_length_instruction(draft.group_type, language)
    
    # Type-specific details (оставляем как есть, но на нормальном языке)
    type_specific = self._build_type_specific(draft)
    
    # Style instructions — расширенные
    style_instructions = {
        "academic": "Строгий академический стиль. Безличные конструкции или 'мы'. Терминологическая точность. Каждое утверждение с доказательством.",
        "blog": "Доступный научно-популярный стиль. Живой язык, но без потери точности. Метафоры для сложных концепций.",
        "popular": "Научно-популярный стиль. Минимум жаргона. Яркие объяснения. Образные сравнения.",
    }
    
    return f"""НАПИСАТЬ СТАТЬЮ

Заголовок: {draft.title_suggestion or '(сгенерировать)'}
Аннотация-набросок: {draft.abstract_suggestion or '(сгенерировать)'}
Тип: {draft.group_type.value}
Язык: {language}
Комментарий пользователя: {user_comment or '(нет)'}

== СТРУКТУРИРОВАННЫЙ ЧЕРНОВИК ==
Исследовательский пробел: {draft.gap_identified or '(не определён)'}
Предлагаемый вклад: {draft.proposed_contribution or '(не определён)'}
Методология: {draft.methods_summary or '(не указана)'}
Уверенность: {draft.confidence}
Ключевые слова: {', '.join(draft.keywords) if draft.keywords else '(нет)'}

{type_specific}

{length_instruction}

== ИСТОЧНИКИ ==
{source_info}

Напиши полную, содержательную статью типа {draft.group_type.value} на {language}.
{style_instructions.get(style, style_instructions['academic'])}"""
```

---

## STAGE 3: Token Limits per Type + Bug Fixes (Корень 2 + support)

### Task 3.1: Добавить WRITER_TOKEN_LIMITS в writer.py

**Файл:** `engine/agents/writer.py` (после импортов)

```python
WRITER_TOKEN_LIMITS = {
    GroupType.REVIEW: 16384,          # ~6000-7000 слов на русском
    GroupType.REPLICATION: 12288,     # ~4000-5000 слов
    GroupType.DATA_PAPER: 8192,       # ~2500-3000 слов
}
```

Использовать в `run()`:
```python
max_tokens = WRITER_TOKEN_LIMITS.get(draft.group_type, 8192)
```

### Task 3.2: Fix — synthetic draft fallback использует GroupType из proposal

**Файл:** `engine/orchestrator_v2.py`, строки 282 и 559

**Что сделать:** Вместо `GroupType.DATA_PAPER` использовать GroupType из proposal или job:

```python
# Было:
group_type=GroupType.DATA_PAPER,

# Стало:
group_type=_resolve_group_type(proposal),
```

Добавить helper:
```python
def _resolve_group_type(self, proposal: dict) -> GroupType:
    """Determine GroupType from proposal metadata."""
    type_str = proposal.get("article_type", "").lower()
    if "review" in type_str:
        return GroupType.REVIEW
    if "replication" in type_str:
        return GroupType.REPLICATION
    # Default based on key_references count
    refs = proposal.get("key_references", [])
    if len(refs) >= 8:
        return GroupType.REVIEW
    return GroupType.DATA_PAPER
```

### Task 3.3: Fix — _rewrite_article сохраняет оригинальный GroupType

**Файл:** `engine/orchestrator_v2.py`, строка 559

Тот же фикс: использовать GroupType из job.proposal, не DATA_PAPER.

---

## STAGE 4: UI — Dropdown типа статьи

### Task 4.1: Добавить group_type в API создания job

**Файл:** `worker/server.py` — endpoint POST `/api/pipeline/run`

Принимать опциональное поле `group_type` в body:
```json
{"topic": "...", "group_type": "review"}
```

Передавать в orchestrator.

**Файл:** `engine/orchestrator_v2.py` — метод `create_job()` — принимать и сохранять group_type.

### Task 4.2: Добавить dropdown в Dashboard UI

**Файл:** `dashboard/templates/index.html` — в модальное окно создания pipeline job

Добавить select:
```html
<select id="pipeline-group-type">
  <option value="review">Обзорная статья (15-25 стр.)</option>
  <option value="replication">Оригинальное исследование (8-12 стр.)</option>
  <option value="data_paper">Статья-данные (5-8 стр.)</option>
</select>
```

### Task 4.3: Добавить textarea для комментария

**Файл:** `dashboard/templates/index.html`

```html
<textarea id="pipeline-user-comment" placeholder="Дополнительные указания: фокус, аудитория, акценты..."></textarea>
```

---

## STAGE 5: Верификация и тесты

### Task 5.1: Обновить test_editor_agent.py

Проверить что RichReader возвращает non-empty rich_context.

### Task 5.2: Обновить test_orchestrator_v2.py

Проверить что synthetic draft использует правильный GroupType.

### Task 5.3: Запустить полный E2E

```bash
cd /home/hermeswebui/.hermes/geo_digest
python -m pytest tests/ -v --tb=short
```

---

## Порядок выполнения

| Стадия | Зависимости | Ожидаемый эффект |
|--------|-------------|-----------------|
| Stage 1 (Rich Data Flow) | Нет | Writer видит факты, цифры, цитаты |
| Stage 2 (Prompt Overhaul) | Stage 1 | Writer знает структуру и стиль |
| Stage 3 (Tokens + Bugs) | Нет | Writer может писать длинные статьи |
| Stage 4 (UI) | Stage 3 | Пользователь выбирает тип |
| Stage 5 (Tests) | Все | Верификация |

Stage 1, 2, 3 — можно делать последовательно, каждый даёт измеримый прирост.
Stage 4 — параллельно с 2-3.
Stage 5 — после всех.
