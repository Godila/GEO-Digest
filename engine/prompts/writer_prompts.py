"""Writer Prompts — Multi-Pass промпты на основе article_patterns.py.

Калибровка по журналу «Геология и геофизика Юга России» (Scopus, ВАК).
Средний объём реальной статьи: 4407 слов. Плотность цитирования: 8.6/1000 слов.

Multi-Pass стратегия:
  Pass 1: Outline (детальный план с тезисами каждого абзаца)
  Pass 2: Expand (полный текст каждой секции отдельно)
  Pass 3: Polish (шлифовка связности, цитирования, переходов)
"""
from engine.agents.article_patterns import (
    EXPECTED_SECTIONS, TONE_RULES, SECTION_TRANSITIONS,
    ARTICLE_TYPES, MULTI_PASS_CONFIG, FORMATTING_RULES,
)
from engine.schemas import GroupType


# ── Helper functions for prompt formatting ──────────────────────

def _format_rules_list(rules: list[str]) -> str:
    """Format a list of rhetorical rules as bullet points."""
    return "\n".join(f"  • {r}" for r in rules)


def _chr10join(items: list[str]) -> str:
    """Join items with newline + indent for nested display."""
    return "\n  ".join(items)


# ═══════════════════════════════════════════════════════════════
#  PASS 1: OUTLINE — детальный план статьи
# ═══════════════════════════════════════════════════════════════

def build_outline_system_prompt(group_type: GroupType, language: str) -> str:
    """System prompt for Pass 1: creating detailed article outline."""
    type_key = _group_type_to_article_type(group_type)
    type_info = ARTICLE_TYPES.get(type_key, ARTICLE_TYPES["original_research"])
    sections_guidance = _format_sections_for_type(type_info)
    lang_label = "русском" if language == "ru" else "English"
    target_wc = type_info.get("target_words", 4500)

    return f"""Ты — опытный научный писатель в области геоэкологии и геонаук.
Создаёшь ДЕТАЛЬНЫЙ план научной статьи на {lang_label} языке.

ЦЕЛЕВОЙ ОБЪЁМ статьи: {target_wc} слов.
Целевое количество ссылок: {type_info.get('target_refs', [30, 50])[0]}-{type_info.get('target_refs', [30, 50])[1]}

Тип статьи: {type_info['label']}

== ОБЯЗАТЕЛЬНАЯ СТРУКТУРА ==
{sections_guidance}

== ЗАДАЧА ==
Создай детальный план статьи. Для КАЖДОЙ секции укажи:
1. Заголовок секции
2. Количество абзацев
3. Для КАЖДОГО абзаца — ключевой тезис (что именно будет сказано)
4. Какие конкретные цифры/факты будут приведены
5. Какие ссылки на литературу будут использованы

ФОРМАТ ВЫВОДА (JSON, без markdown):
{{"title": "...", "outline": [{{"section": "...", "paragraphs": [{{"thesis": "...", "key_facts": ["..."], "references": ["..."]}}]}}], "total_target_words": {target_wc}}}"""


def build_outline_user_prompt(draft, reader_context: str = "") -> str:
    """User prompt for outline generation."""
    topic = getattr(draft, 'title', '') or getattr(draft, 'topic', '') or 'Геоэкологическое исследование'
    proposal_text = ""
    
    if hasattr(draft, 'proposal') and draft.proposal:
        proposal_text = f"\n\nПРЕДЛОЖЕНИЕ ПО СТАТЬЕ:\n{draft.proposal}"
    
    if hasattr(draft, 'key_references') and draft.key_references:
        refs_text = "\n".join(
            f"- {r}" if isinstance(r, str) else f"- {r.get('title', r.get('doi', str(r)))}"
            for r in draft.key_references[:20]
        )
        proposal_text += f"\n\nДОСТУПНЫЕ ИСТОЧНИКИ ({len(draft.key_references)} шт.):\n{refs_text}"

    context_block = ""
    if reader_context:
        context_block = f"\n\nБОГАТЫЙ КОНТЕКСТ ИЗ ИСТОЧНИКОВ:\n{reader_context[:10000]}"

    return f"""Создай детальный план статьи на тему: {topic}

ТРЕБОВАНИЯ:
- Каждый абзац должен содержать минимум 1 конкретную цифру
- Обсуждение должно занимать 25-35% объёма
- План должен включать минимум {len(getattr(draft, 'key_references', [])) or 25} ссылок
- Каждый тезис должен быть подкреплён фактом из источников
{proposal_text}{context_block}"""


# ═══════════════════════════════════════════════════════════════
#  PASS 2: EXPAND — полный текст по секциям
# ═══════════════════════════════════════════════════════════════

def build_expand_system_prompt(group_type: GroupType, language: str) -> str:
    """System prompt for Pass 2: expanding each section."""
    type_key = _group_type_to_article_type(group_type)
    type_info = ARTICLE_TYPES.get(type_key, ARTICLE_TYPES["original_research"])
    sections_guidance = _format_sections_for_type(type_info)
    tone = _format_tone_rules()
    transitions = _format_transitions()
    lang_label = "русском" if language == "ru" else "English"
    
    return f"""Ты — опытный научный писатель в области геоэкологии и геонаук.
Пишешь полноценную научную статью на {lang_label} языке.

== ТИП СТАТЬИ ==
{type_info['label']}

== ОБЯЗАТЕЛЬНАЯ СТРУКТУРА СЕКЦИЙ ==
{sections_guidance}

== СТИЛИСТИЧЕСКИЕ ПРАВИЛА ==
{tone}

== ФРАЗЫ ДЛЯ ПЕРЕХОДОВ МЕЖДУ СЕКЦИЯМИ ==
{transitions}

== ТРЕБОВАНИЯ К ЦИТИРОВАНИЮ ==
- КАЖДОЕ утверждение подкреплено ссылкой: [Автор и др., год] или (Author et al., Year)
- Минимум {type_info.get('target_refs', [30, 50])[0]} источников
- Плотность: 8-10 ссылок на 1000 слов
- Баланс: классические работы + свежие (последние 5 лет, не менее 40%)
- DOI указан для каждого источника если доступен

== КРИТИЧЕСКИЕ ПРАВИЛА ==
1. КАЖДЫЙ абзац — 4-8 предложений, не 1-2
2. В КАЖДОМ абзаце — минимум 1 конкретное число (магнитуда, концентрация, координаты, %)
3. ОБСУЖДЕНИЕ — не пересказ результатов, а ИНТЕРПРЕТАЦИЯ + сравнение + механизм + ограничения
4. Никаких общих фраз — 'значения увеличились' → 'значения увеличились на 23% (с 1.2±0.1 до 1.5±0.2 мг/кг)'
5. Каждый факт контекстуализирован: помещён в рамку существующих исследований

== ФОРМАТ ВЫВОДА ==
Верни JSON (без markdown-обёрток ```json ```):
{{"title": "...", "sections": [{{"heading": "...", "content": "..."}}], "references": ["..."], "word_count": N}}"""


def build_expand_user_prompt(outline: str, reader_context: str = "") -> str:
    """User prompt for expanding sections based on outline."""
    context_block = ""
    if reader_context:
        context_block = f"""

ИСПОЛЬЗУЙ ЭТОТ КОНТЕКСТ ИЗ ИСТОЧНИКОВ (конкретные цифры, методы, выводы):
{reader_context[:8000]}

ВАЖНО: Извлекай из контекста конкретные числа, названия методов, имена авторов, годы публикации.
Каждое число из контекста — кандидат для включения в статью."""

    return f"""Напиши ПОЛНЫЙ текст статьи по этому плану:

{outline}

ТРЕБОВАНИЯ К ОБЪЁМУ:
- Введение: 400-700 слов (3-6 абзацев)
- Методы: 800-1200 слов (4-8 абзацев)
- Результаты: 900-1500 слов (5-10 абзацев)
- Обсуждение: 1000-1500 слов (5-10 абзацев) — САМАЯ ДЛИННАЯ секция
- Выводы: 200-400 слов (1-2 абзаца)
{context_block}

ПИШИ ПОЛНЫЙ ТЕКСТ, НЕ НАБРОСОК. Каждый абзац 4-8 предложений."""


# ═══════════════════════════════════════════════════════════════
#  PASS 2b: SECTION-BY-SECTION EXPAND — одна секция за раз
# ═══════════════════════════════════════════════════════════════

# Мэппинг: какая часть rich_context нужна каждой секции
SECTION_CONTEXT_KEYWORDS = {
    "введение": ["gap", "пробел", "актуальн", "значим", "цель", "задач", "обзор", "истори", "контекст", "trend"],
    "литератур": ["обзор", "литератур", "сравнен", "ранее", "известн", "существующ", "previous", "review"],
    "метод": ["метод", "подход", "алгоритм", "модель", "методик", "датчик", "аппарат", "параметр", "method", "dataset", "данн"],
    "результат": ["результат", "значени", "концентрац", "магнитуд", "координат", "скор", "точност", "показат", "обнаруж", "найден", "result", "found", "%", "±"],
    "обсужден": ["обсужден", "интерпретаци", "сравнен", "противореч", "соглас", "ограничен", "limitat", "discuss", "mechanism", "механизм"],
    "вывод": ["вывод", "заключен", "значим", "рекомендац", "перспектив", "conclusion", "future"],
    "conclusion": ["вывод", "заключен", "значим", "рекомендац", "перспектив"],
    "introduction": ["gap", "пробел", "актуальн", "цель", "задач"],
    "method": ["метод", "подход", "алгоритм", "модель", "датчик"],
    "result": ["результат", "значени", "концентрац", "скор", "точност"],
    "discussion": ["обсужден", "интерпретаци", "сравнен", "ограничен"],
    "review": ["обзор", "литератур", "сравнен", "ранее"],
}

# Целевой объём секций (слова) и max_tokens для LLM
SECTION_TARGETS = {
    "введение":        {"words": (400, 700),  "tokens": 6000},
    "обзор литературы": {"words": (800, 1500), "tokens": 12000},
    "методы":           {"words": (800, 1200), "tokens": 9000},
    "результаты":       {"words": (900, 1500), "tokens": 12000},
    "обсуждение":       {"words": (1000, 1500), "tokens": 12000},
    "выводы":           {"words": (200, 400),   "tokens": 4500},
    "список литературы": {"words": (100, 300),  "tokens": 3000},
    # English fallback
    "introduction":    {"words": (400, 700),  "tokens": 6000},
    "literature":      {"words": (800, 1500), "tokens": 12000},
    "methods":         {"words": (800, 1200), "tokens": 9000},
    "results":         {"words": (900, 1500), "tokens": 12000},
    "discussion":      {"words": (1000, 1500), "tokens": 12000},
    "conclusion":      {"words": (200, 400),   "tokens": 4500},
    "references":      {"words": (100, 300),   "tokens": 3000},
}


def extract_section_context(section_heading: str, rich_context: str) -> str:
    """Extract the relevant portion of rich_context for a specific section.
    
    Instead of truncating rich_context to 6K/8K, we pick paragraphs that
    contain keywords relevant to this section's topic.
    """
    if not rich_context:
        return ""
    
    heading_lower = section_heading.lower()
    keywords = []
    for key, kws in SECTION_CONTEXT_KEYWORDS.items():
        if key in heading_lower:
            keywords = kws
            break
    
    # If no keywords matched, return full context (up to 15K chars)
    if not keywords:
        return rich_context[:15000]
    
    # Split context into paragraphs and score each by keyword matches
    paragraphs = rich_context.split("\n\n")
    scored = []
    for p in paragraphs:
        p_lower = p.lower()
        score = sum(1 for kw in keywords if kw in p_lower)
        scored.append((score, p))
    
    # Take paragraphs with matches first, then fill with remaining
    matched = [p for score, p in sorted(scored, reverse=True) if score > 0]
    remaining = [p for score, p in scored if score == 0]
    
    result_paragraphs = matched + remaining
    result = "\n\n".join(result_paragraphs)
    
    # Cap at 12000 chars per section (generous)
    return result[:12000]


def get_section_target(section_heading: str) -> dict:
    """Get target word count and max_tokens for a section."""
    heading_lower = section_heading.lower()
    for key, target in SECTION_TARGETS.items():
        if key in heading_lower:
            return target
    # Default for unknown sections
    return {"words": (500, 1000), "tokens": 9000}


def build_section_expand_system_prompt(group_type: GroupType, language: str, 
                                        section_heading: str, format_: str = "markdown") -> str:
    """System prompt for expanding a single section."""
    from engine.agents.article_patterns import get_rhetorical_rules, CARS_MOVES, EVIDENCE_CHAINING_EXAMPLE

    type_key = _group_type_to_article_type(group_type)
    type_info = ARTICLE_TYPES.get(type_key, ARTICLE_TYPES["original_research"])
    lang_label = "русском" if language == "ru" else "English"
    target = get_section_target(section_heading)
    tone = _format_tone_rules()
    
    # Formatting rules (markdown or latex)
    format_key = format_ if format_ in FORMATTING_RULES else "markdown"
    fmt = FORMATTING_RULES[format_key]
    format_rules = f"""
== ПРАВИЛА ФОРМАТИРОВАНИЯ ({format_key.upper()}) ==
- Формулы: {fmt['equations']}
- Таблицы: {fmt['tables']}
- Рисунки: {fmt['figures']}
- Графики/диаграммы: {fmt['matplotlib']}
- Цитирование: {fmt['citations']}
"""

    # Rhetorical rules specific to this section type
    rhetoric = get_rhetorical_rules(section_heading)
    rhetoric_block = f"""
== РИТОРИЧЕСКАЯ СТРУКТУРА ЭТОЙ СЕКЦИИ ({section_heading}) ==
Паттерн: {rhetoric['pattern']}
Правила:
{_format_rules_list(rhetoric['rules'])}
"""

    # CARS model for Introduction
    cars_block = ""
    if "introduction" in section_heading.lower():
        cars_block = f"""
== CARS МОДЕЛЬ (обязательно для Introduction) ==
{CARS_MOVES['move_1_territory']['label']}:
  {_chr10join(CARS_MOVES['move_1_territory']['steps'])}
{CARS_MOVES['move_2_niche']['label']}:
  {_chr10join(CARS_MOVES['move_2_niche']['steps'])}
{CARS_MOVES['move_3_occupy']['label']}:
  {_chr10join(CARS_MOVES['move_3_occupy']['steps'])}
"""

    # Evidence chaining for Results/Discussion
    evidence_block = ""
    if any(kw in section_heading.lower() for kw in ("result", "discussion", "обсужден", "результат")):
        evidence_block = f"""
== EVIDENCE CHAINING (обязательно) ==
{EVIDENCE_CHAINING_EXAMPLE}

КАЖДЫЙ абзац должен содержать минимум 2 ссылки на источники.
Используй схему: Claim → [Author, Year] → Supporting [Author, Year] → Counter [Author, Year] → Implication
"""
    
    return f"""Ты — опытный научный писатель в области геоэкологии и геонаук.
Пишешь ОДНУ секцию '{section_heading}' научной статьи на {lang_label} языке.

== ТИП СТАТЬИ ==
{type_info['label']}

== ЦЕЛЕВОЙ ОБЪЁМ ЭТОЙ СЕКЦИИ ==
{target['words'][0]}-{target['words'][1]} слов, примерно {target['words'][0]//5}-{target['words'][1]//5} абзацев.
КАЖДЫЙ абзац — 4-8 предложений. Не 1-2, не 15.

{rhetoric_block}{cars_block}{evidence_block}
== СТИЛИСТИЧЕСКИЕ ПРАВИЛА ==
{tone}
{format_rules}
== КРИТИЧЕСКИЕ ПРАВИЛА ==
1. КАЖДЫЙ абзац — минимум 1 конкретное число (магнитуда, концентрация, координаты, %, ±)
2. КАЖДОЕ утверждение подкреплено ссылкой: [Автор и др., год] или (Author et al., Year)
3. Никаких общих фраз: 'значения увеличились' → 'значения увеличились на 23% (с 1.2±0.1 до 1.5±0.2 мг/кг)'
4. Пиши ПОЛНЫЙ текст, НЕ набросок, НЕ план, НЕ тезисы
5. Используй цитаты из EVIDENCE — вплетай их в текст: «Как показал [Author], "quote..." »

== ФОРМАТ ВЫВОДА ==
Верни JSON (без markdown-обёрток):
{{"heading": "{section_heading}", "content": "ПОЛНЫЙ текст секции с абзацами", "word_count": N}}"""


def build_section_expand_user_prompt(
    section_heading: str,
    section_outline: str,
    section_context: str,
    previous_section_summary: str = "",
    perspective_questions: str = "",
) -> str:
    """User prompt for expanding one section."""
    transition_hint = ""
    if previous_section_summary:
        transition_hint = f"""
== СВЯЗЬ С ПРЕДЫДУЩЕЙ СЕКЦИЕЙ ==
Предыдущая секция закончилась так:
{previous_section_summary[:500]}

Начни эту секцию с логического перехода от предыдущей. Используй фразы:
'Полученные результаты согласуются с...', 'В отличие от...', 'Это может быть объяснено...'
"""
    
    context_block = ""
    if section_context:
        context_block = f"""
== КОНТЕКСТ ИЗ ИСТОЧНИКОВ (используй конкретные цифры, методы, имена, годы) ==
{section_context}

ВАЖНО: Извлекай из контекста конкретные числа, названия методов, имена авторов, годы.
Каждое число из контекста — кандидат для включения в секцию."""

    questions_block = ""
    if perspective_questions:
        questions_block = f"""
== КЛЮЧЕВЫЕ ВОПРОСЫ ДЛЯ ЭТОЙ СЕКЦИИ ==
Ответь на эти вопросы в тексте секции, используя evidence из источников:
{perspective_questions}
"""
    
    return f"""Напиши ПОЛНЫЙ текст секции '{section_heading}'.

== ПЛАН ЭТОЙ СЕКЦИИ ==
{section_outline}
{transition_hint}{context_block}{questions_block}

ПИШИ ПОЛНЫЙ ТЕКСТ. Каждый абзац 4-8 предложений с конкретными данными."""


def build_references_system_prompt(language: str) -> str:
    """System prompt for generating formatted references section."""
    lang_label = "русском" if language == "ru" else "English"
    return f"""Ты — научный редактор. Форматируешь список литературы на {lang_label} языке.

ПРАВИЛА:
- ГОСТ Р 7.0.5-2008 формат
- Каждый источник с DOI если доступен
- Алфавитный порядок (сначала кириллица, потом латиница)
- Верни JSON: {{"heading": "Список литературы", "content": "отформатированный список", "references": [...]}}"""


# ═══════════════════════════════════════════════════════════════
#  REVISION: промпт для переработки по замечаниям Reviewer
# ═══════════════════════════════════════════════════════════════

def build_revision_system_prompt(language: str) -> str:
    """System prompt for revising article based on reviewer feedback."""
    lang_label = "русском" if language == "ru" else "English"
    return f"""Ты — научный писатель, перерабатываешь КОНКРЕТНЫЕ секции статьи по замечаниям рецензента.
Язык: {lang_label}.

ЗАДАЧА: Переписать ТОЛЬКО указанные секции. Вернуть JSON ТОЛЬКО с переписанными секциями.
НЕ переписывай и НЕ возвращай одобренные секции — они будут сохранены как есть.

ПРАВИЛА:
1. Каждый исправленный абзац — 4-8 предложений с конкретными числами
2. Усиливай слабые места конкретными данными и цитатами из источников
3. НЕ СОКРАЩАЙ контент — только расширяй и углубляй
4. Обсуждение: минимум 800 слов (сравнение с другими исследованиями, механизмы, ограничения)
5. Выводы: минимум 300 слов (резюме каждого ключевого вывода, значимость, перспективы)
6. Сохраняй академический стиль без AI-штампов

ФОРМАТ ВЫВОДА (JSON, без markdown):
{{"rewritten_sections": [{{"heading": "Обсуждение", "content": "..."}}], "title_change": null или "новый заголовок"}}"""


def build_revision_user_prompt(article_text: str, revision_instructions: list) -> str:
    """User prompt for article revision — sends only affected sections."""
    edits_text = "\n\n".join(
        f"Правка {i+1} [{e.get('severity', 'medium')}]: {e.get('description', '')}\n"
        f"  Секция: {e.get('section', 'вся статья')}\n"
        f"  Действие: {e.get('action', 'исправить')}"
        for i, e in enumerate(revision_instructions)
    )
    
    # Determine which sections need editing
    affected_sections = set()
    for e in revision_instructions:
        sec = e.get('section', '')
        if sec:
            # Normalize: match partial headings
            affected_sections.add(sec.lower().strip())
    
    # Split article into sections
    sections_dict = {}
    current_heading = ""
    current_content = []
    for line in article_text.split("\n"):
        if line.startswith("## "):
            if current_heading:
                sections_dict[current_heading] = "\n".join(current_content)
            current_heading = line.strip("# ").strip()
            current_content = [line]
        else:
            current_content.append(line)
    if current_heading:
        sections_dict[current_heading] = "\n".join(current_content)
    
    # Build context: ONLY affected sections for rewriting
    affected_text = ""
    kept_sections = []
    for heading, content in sections_dict.items():
        is_affected = (
            heading.lower().strip() in affected_sections
            or any(a in heading.lower() for a in affected_sections)
            or not affected_sections  # if no specific section → all affected
        )
        if is_affected:
            affected_text += content + "\n\n"
        else:
            kept_sections.append(heading)
    
    # If no specific sections identified, send full article (truncated to 20K)
    if not affected_text:
        affected_text = article_text[:20000]
    
    kept_info = ""
    if kept_sections:
        kept_info = f"\n\nСекции БЕЗ ИЗМЕНЕНИЙ (НЕ включай их в ответ): {', '.join(kept_sections)}"
    
    # Section-specific minimums to prevent shrinkage
    min_words = ""
    for heading in sections_dict:
        h = heading.lower()
        if "обсужд" in h or "discuss" in h:
            min_words += f"\n- {heading}: МИНИМУМ 800 слов"
        elif "вывод" in h or "заключ" in h or "conclu" in h:
            min_words += f"\n- {heading}: МИНИМУМ 300 слов"
        elif "введен" in h or "introd" in h:
            min_words += f"\n- {heading}: МИНИМУМ 500 слов"
    
    return f"""ПЕРЕРАБОТАЙ указанные секции по замечаниям рецензента.

== ЗАМЕЧАНИЯ ==
{edits_text}
{kept_info}

== СЕКЦИИ ДЛЯ ПРАВКИ ==
{affected_text}

== МИНИМАЛЬНЫЙ ОБЪЁМ СЕКЦИЙ =={min_words}

Верни JSON ТОЛЬКО с переписанными секциями ( НЕ включай одобренные секции).
Формат: {{"rewritten_sections": [{{"heading": "...", "content": "..."}}]}}"""


# ═══════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ═══════════════════════════════════════════════════════════════

def build_target_word_count(group_type: GroupType) -> dict:
    """Return target word count range for article type."""
    type_key = _group_type_to_article_type(group_type)
    type_info = ARTICLE_TYPES.get(type_key, ARTICLE_TYPES["original_research"])
    return {
        "min": type_info.get("word_range", [3500, 5500])[0],
        "max": type_info.get("word_range", [3500, 5500])[1],
        "target": type_info.get("target_words", 4500),
    }


def build_length_instruction(group_type: GroupType, language: str) -> str:
    """Build explicit length instruction for the prompt."""
    wc = build_target_word_count(group_type)
    lang_label = "слов" if language == "ru" else "words"
    return f"""== ЦЕЛЕВОЙ ОБЪЁМ ==
Минимум: {wc['min']} {lang_label}
Целевой: {wc['target']} {lang_label}
Максимум: {wc['max']} {lang_label}

Каждая секция должна быть содержательной:
- Введение: 3-6 абзацев, 400-700 слов (контекст → история → проблема → цель)
- Методы: 4-8 абзацев, 800-1200 слов (данные → методы → обоснование → параметры)
- Результаты: 5-10 абзацев, 900-1500 слов (цифры → таблицы → сравнения)
- Обсуждение: 5-10 абзацев, 1000-1500 слов (интерпретация → сравнение → механизм → ограничения)
- Выводы: 1-2 абзаца, 200-400 слов (резюме + значимость)"""


def build_max_tokens(group_type: GroupType) -> int:
    """Calculate max_tokens for expand pass based on target word count.
    
    Russian text: ~1.5 tokens per word (subword tokenization).
    Add 50% buffer for JSON structure overhead.
    """
    target = build_target_word_count(group_type)
    return min(int(target["target"] * 1.5 * 1.5), 32000)


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
            if "paragraphs" in sec_def:
                line += f", {sec_def['paragraphs'][0]}-{sec_def['paragraphs'][1]} абзацев"
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
