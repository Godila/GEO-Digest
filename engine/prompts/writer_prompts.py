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
    ARTICLE_TYPES, MULTI_PASS_CONFIG,
)
from engine.schemas import GroupType


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
        context_block = f"\n\nБОГАТЫЙ КОНТЕКСТ ИЗ ИСТОЧНИКОВ:\n{reader_context[:6000]}"

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
#  PASS 3: POLISH — шлифовка финального текста
# ═══════════════════════════════════════════════════════════════

def build_polish_system_prompt(language: str) -> str:
    """System prompt for Pass 3: polishing the article."""
    lang_label = "русском" if language == "ru" else "English"
    
    return f"""Ты — научный редактор высшего класса, специализируешься на геоэкологии и геонауках.
Отшлифовываешь научную статью на {lang_label} языке до уровня журнала ВАК/Scopus.

ТВОИ ЗАДАЧИ:
1. Усиль связность: добавь логические переходы между абзацами и секциями
2. Проверь цитирование: каждое утверждение подкреплено ссылкой [Автор, год]
3. Устрани повторы и избыточность
4. Убедись что КАЖДЫЙ абзац содержит конкретные числа
5. Усиль секцию Обсуждение — добавь интерпретацию если её нет
6. Проверь баланс секций (обсуждение должно быть 25-35% статьи)

СТИЛЬ:
- Безличные конструкции или 'мы'
- Академический, формальный
- Конкретика > общих фраз
- [Автор и др., год] формат цитирования

== ФОРМАТ ВЫВОДА ==
Верни JSON (без markdown):
{{"title": "...", "sections": [{{"heading": "...", "content": "..."}}], "references": ["..."], "word_count": N}}"""


def build_polish_user_prompt(article_json: str) -> str:
    """User prompt for polishing the article."""
    return f"""Отшлифуй эту статью. Усиль связность, проверь цитирование, устрани повторы.

СТАТЬЯ ДЛЯ ШЛИФОВКИ:
{article_json}

УЛУЧШИ:
1. Переходы между секциями — используй фразы: 'Полученные результаты согласуются с...', 'В отличие от...', 'Это может быть объяснено...'
2. Добавь конкретные числа в каждый абзац если их нет
3. Убедись что Обсуждение содержит: интерпретацию, сравнение с литературой, механизм, ограничения
4. Проверь что каждый абзац 4-8 предложений
5. Устрани пустые фразы без содержания"""


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
    return min(int(target["target"] * 1.5 * 1.5), 16384)


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
