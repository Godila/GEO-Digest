"""Article Patterns — извлечённые из реальных научных публикаций.

Источник: журнал «Геология и геофизика Юга России» (Scopus, ВАК, ISSN 2221-3198).
Анализ 22 статей том 16, №1, 2026 (261 стр., ~113 000 слов).

Это НЕ шаблон 1-в-1, а набор паттернов: структура, tone of voice,
стилистические маркеры, типичные формулировки.

Используется Writer'ом и Reviewer'ом.
"""

# ═══════════════════════════════════════════════════════════════
#  СТРУКТУРНЫЕ ПАТТЕРНЫ — калибровка по реальному журналу
# ═══════════════════════════════════════════════════════════════
#
#  Измерено по 22 статьям журнала:
#  - Средний объём: 4407 слов (медиана 4577)
#  - Среднее ссылок: 38.3
#  - Плотность цитирования: 8.6 ссылок/1000 слов
#  - Стиль: безличный (41%), "мы" (36%), смешанный
#  - Цитирование: Автор-год (доминирующий)
#

EXPECTED_SECTIONS = [
    {
        "id": "abstract_ru",
        "required": True,
        "label": "Аннотация (RU)",
        "word_range": [250, 300],
        "rules": [
            "Без формул и ссылок на литературу",
            "Структура: актуальность → цель → методы → результаты → выводы",
            "Начинается с контекста проблемы, не с 'В данной статье...'",
            "Конкретные цифры и факты, не общие фразы",
        ],
        "example_start": (
            "Актуальность работы обусловлена необходимостью "
            "оценки параметров сейсмического режима..."
        ),
    },
    {
        "id": "keywords",
        "required": True,
        "label": "Ключевые слова",
        "count_range": [5, 8],
        "rules": ["В конце каждой аннотации", "Отражают тематику, не слишком широкие"],
    },
    {
        "id": "introduction",
        "required": True,
        "label": "Введение",
        "word_range": [400, 700],  # ~12-18% от 4400 слов
        "paragraphs": [3, 6],
        "structure": [
            "Геологический/экологический контекст региона (1-2 абзаца)",
            "История изучения вопроса со ссылками (1 абзац)",
            "Проблема / gap в знаниях (1 абзац)",
            "Что сделано другими — обзор литературы (1-2 абзаца, 5-10 ссылок)",
            "Цель работы: 'Цель работы заключается в...' (1 предложение)",
            "Задачи: (1)... (2)... (3)... (1 предложение)",
        ],
        "tone_markers": [
            "Известно, что...", "В последние годы...",
            "Основные сейсмогенные структуры протягиваются вдоль...",
            "Однако...", "В то же время...",
            "Целью настоящей работы является...",
            "Актуальность работы обусловлена...",
            "Для решения этой задачи...",
        ],
        "critical_rule": "Введение ДОЛЖНО содержать 3-5 ссылок на фундаментальные работы и конкретные цифры (магнитуды, глубины, площади, периоды наблюдений)",
    },
    {
        "id": "methods",
        "required": True,
        "label": "Материалы и методы",
        "word_range": [800, 1200],  # ~15-25%
        "paragraphs": [4, 8],
        "structure": [
            "Описание объекта исследования: географические координаты, геологическое строение, площадь",
            "Источник данных: каталог, станция, спутник, период наблюдений",
            "Метод обработки с ОБОСНОВАНИЕМ выбора (почему именно этот метод)",
            "Параметры модели/алгоритма (конкретные числа)",
            "Сравнение с альтернативными методами (почему они хуже подходят)",
            "Критерии достоверности и ограничения метода",
        ],
        "tone_markers": [
            "Фактологической основой послужили данные...",
            "Обработка выполнена с помощью...",
            "Для анализа использованы...",
            "Применение данного метода обусловлено...",
            "Параметры модели следующие...",
            "Для верификации результатов применён...",
        ],
        "critical_rule": "Методы должны быть описаны настолько подробно, чтобы другой исследователь мог ПОВТОРИТЬ исследование. Конкретные параметры, модели, версии ПО.",
    },
    {
        "id": "results",
        "required": True,
        "label": "Результаты",
        "word_range": [900, 1500],  # ~20-30%
        "paragraphs": [5, 10],
        "structure": [
            "Главный количественный результат с конкретными цифрами (абзац 1)",
            "Детализация: таблицы, сравнения, статистика (абзацы 2-4)",
            "Пространственное распределение / временные тренды (абзацы 3-5)",
            "Сравнение 'до/после' или 'здесь/там' (абзацы 4-6)",
            "Промежуточные выводы из результатов (абзацы 5-7)",
            "Описание рисунков и таблиц с анализом (на каждый рисунок/таблицу — абзац)",
        ],
        "tone_markers": [
            "Установлено, что...", "Выявлены следующие закономерности:",
            "Результаты показывают, что...", "Полученные значения составляют...",
            "На рисунке X представлено...", "Как видно из таблицы Y...",
            "Особую тревогу вызывает...", "Существенные различия...",
            "Максимальные концентрации приурочены к...",
            "В интервале X-Y значения составляют...",
        ],
        "critical_rule": "КАЖДЫЙ абзац должен содержать минимум 1 конкретное число. Никаких 'значения увеличились' — только 'значения увеличились на 23% (с 1.2 до 1.5 мг/кг)'. Таблицы ОБЯЗАТЕЛЬНЫ.",
    },
    {
        "id": "discussion",
        "required": True,
        "label": "Обсуждение",
        "word_range": [1000, 1500],  # ~25-35% — САМАЯ ВАЖНАЯ СЕКЦИЯ
        "paragraphs": [5, 10],
        "structure": [
            "Интерпретация главного результата: что это значит физически/геологически (абзац 1)",
            "Сравнение с литературой: 'Полученные результаты согласуются с данными [Автор, год]' (абзацы 2-3)",
            "Отличия от других работ: 'В отличие от [Автор, год], в нашем случае...' (абзац 3-4)",
            "Физический/химический/геологический механизм объясняющий результат (абзац 4-5)",
            "Ограничения исследования: 'Следует отметить ограничение...' (абзац 6-7)",
            "Практическая значимость: 'Результаты могут быть использованы для...' (абзац 7-8)",
            "Перспективы дальнейших исследований (абзац 8-10)",
        ],
        "tone_markers": [
            "Полученные результаты согласуются с данными...",
            "В отличие от [Автор, год], наши данные показывают...",
            "Это может быть объяснено...",
            "Данные различия обусловлены особенностями строения...",
            "Следует отметить ограничение данного подхода...",
            "Результаты могут быть использованы для...",
            "Основным источником ... выступают...",
            "Это подтверждает техногенную природу...",
            "Особую роль играют...",
        ],
        "critical_rule": "ОБСУЖДЕНИЕ — самая важная секция. Она должна занимать 25-35% статьи. Каждый результат должен быть ПРОИНТЕРПРЕТИРОВАН, а не просто перефразирован. Минимум 5-8 ссылок на другие работы для сравнения.",
    },
    {
        "id": "conclusion",
        "required": True,
        "label": "Выводы",
        "word_range": [200, 400],  # ~3-5% (было слишком мало)
        "paragraphs": [1, 2],
        "rules": [
            "Главный вывод — 1 предложение",
            "Пронумерованные выводы: (1)... (2)... (3)... (4)...",
            "Практическая значимость — 1-2 предложения",
            "Не повторять дословно результаты",
        ],
        "tone_markers": [
            "Таким образом,...",
            "На основании проведённого исследования можно сделать следующие выводы:",
            "Практическая значимость работы заключается в...",
        ],
    },
    {
        "id": "references",
        "required": True,
        "label": "Список литературы",
        "style": "Автор-год (Harvard/ГОСТ)",
        "order": "Алфавитный: сначала RU источники, затем EN",
        "in_text_format": "[Автор и др., год] или (Author et al., Year)",
        "target_count": [30, 60],  # Измерено: среднее 38.3, медиана 46
        "density": "8-10 ссылок на 1000 слов текста",
        "rules": [
            "Минимум 25 источников для обзорной статьи",
            "Баланс: классические работы + свежие (последние 5 лет, не менее 40%)",
            "DOI для каждого источника если доступен",
            "Самоцитирование минимальное",
        ],
    },
]

# ═══════════════════════════════════════════════════════════════
#  TONE OF VOICE — калибровка по журналу
# ═══════════════════════════════════════════════════════════════

TONE_RULES = {
    "register": "академический, формальный, объективный",
    "person": "безличные конструкции ('установлено', 'показано') или 'мы' (множественное число)",
    "forbidden": [
        "Разговорная лексика ('вообще', 'короче', 'типа')",
        "Эмоциональные эпитеты ('удивительный', 'потрясающий')",
        "Первое лицо единственного числа ('я считаю', 'мне кажется')",
        "Риторические вопросы в научном тексте",
        "Клише без содержания ('следует отметить', 'представляет собой') — только с конкретикой",
        "Повторы одного слова в соседних предложениях",
        "Общие фразы без цифр: 'значения увеличились' → 'значения увеличились на 23%'",
        "Пустые вводные: 'В данной статье рассматривается...' → сразу к делу",
    ],
    "required": [
        "Точные количественные показатели ('на 23% выше', 'p < 0.05', 'Мw=6,9')",
        "Терминологическая точность (один термин = одно значение во всём тексте)",
        "Причинно-следственные связи: 'Это обусловлено...', 'Это может быть объяснено...'",
        "Ограничения и допущения честно указаны",
        "Конкретные цифры в КАЖДОМ абзаце (магнитуды, глубины, координаты, концентрации, площади)",
        "Контекстуализация: каждый факт помещён в рамку существующих исследований",
    ],
}

# Паттерны хороших переходов — из реальных статей журнала
SECTION_TRANSITIONS = {
    "intro→methods": [
        "Для решения поставленной задачи использован следующий подход...",
        "Фактологической основой исследований послужили данные...",
        "Материалами для исследования послужили...",
    ],
    "methods→results": [
        "Результаты проведённого анализа представлены далее.",
        "Анализ полученных данных выявил следующие закономерности.",
        "Результаты исследования и их обсуждение представлены ниже.",
    ],
    "results→discussion": [
        "Полученные результаты требуют детального обсуждения.",
        "Для интерпретации результатов необходимо рассмотреть...",
        "Обсуждение полученных данных позволяет выявить...",
    ],
    "discussion→conclusion": [
        "Таким образом, на основании проведённого исследования можно сделать следующие выводы.",
        "Резюмируя результаты проведённого анализа,...",
        "Основные выводы работы заключаются в следующем.",
    ],
}

# ═══════════════════════════════════════════════════════════════
#  ТИПЫ СТАТЕЙ — калибровка по измеренным объёмам
# ═══════════════════════════════════════════════════════════════

ARTICLE_TYPES = {
    "original_research": {
        "label": "Оригинальное исследование (IMRaD)",
        "word_range": [3500, 5500],
        "target_words": 4500,
        "target_refs": [30, 50],
        "sections": ["introduction", "methods", "results", "discussion", "conclusion", "references"],
        "key_criteria": ["новизна данных", "валидность методов", "конкретные цифры", "глубокое обсуждение"],
    },
    "review": {
        "label": "Обзорная статья",
        "word_range": [5000, 8000],
        "target_words": 6000,
        "target_refs": [40, 70],
        "sections": ["introduction", "systematic_review", "results", "discussion", "future_directions", "conclusion", "references"],
        "key_criteria": ["полнота охвата литературы", "критический анализ", "систематичность", "сравнительные таблицы"],
    },
    "short_communication": {
        "label": "Краткое сообщение",
        "word_range": [1500, 2500],
        "target_words": 2000,
        "target_refs": [10, 20],
        "sections": ["introduction", "methods_brief", "results", "conclusion", "references"],
        "key_criteria": ["важность открытия", "лаконичность", "скорость публикации"],
    },
    "data_paper": {
        "label": "Статья-данные",
        "word_range": [2500, 4000],
        "target_words": 3500,
        "target_refs": [20, 35],
        "sections": ["introduction", "data_description", "methods", "quality_assessment", "access_info", "conclusion", "references"],
        "key_criteria": ["полнота метаданных", "качество данных", "доступность"],
    },
}

# ═══════════════════════════════════════════════════════════════
#  MULTI-PASS WRITER — параметры
# ═══════════════════════════════════════════════════════════════

MULTI_PASS_CONFIG = {
    "passes": 3,
    "pass_1_outline": {
        "purpose": "Создать детальный план статьи с ключевыми тезисами каждого абзаца",
        "target_words": 400,
        "max_tokens": 3000,
        "output": "structured_outline",
    },
    "pass_2_expand": {
        "purpose": "Написать полный текст каждой секции по плану",
        "target_words": "per article_type word_range",
        "max_tokens": "per build_max_tokens()",
        "output": "full_article",
        "per_section": True,  # Каждая секция генерируется ОТДЕЛЬНО
    },
    "pass_3_polish": {
        "purpose": "Шлифовка: связность, переходы, проверка цитирования, устранение повторов",
        "target_words": "same as pass_2",
        "max_tokens": 8000,
        "output": "final_article",
    },
}

# ═══════════════════════════════════════════════════════════════
#  КРИТЕРИИ ОЦЕНКИ (взвешенные по категориям)
# ═══════════════════════════════════════════════════════════════

REVIEW_CRITERIA = {
    "structure": {
        "weight": 0.15,
        "checks": [
            ("has_all_required_sections", "Все обязательные секции присутствуют"),
            ("section_order_correct", "Порядок секций соответствует IMRaD"),
            ("abstract_compliant", "Аннотация 250-300 слов со структурой: актуальность→цель→методы→результаты→выводы"),
            ("keywords_adequate", "5-8 ключевых слов"),
            ("volume_adequate", "Объём статьи соответствует типу (3000-5500 слов для original research)"),
        ],
    },
    "content_quality": {
        "weight": 0.30,
        "checks": [
            ("claims_supported", "Все утверждения подкреплены данными или ссылками"),
            ("gap_clearly_defined", "Исследуемый gap чётко определён"),
            ("contribution_explicit", "Новизна/contribution явно сформулирована"),
            ("methods_reproducible", "Методы описаны достаточно для воспроизводства"),
            ("results_specific", "Результаты содержат конкретные цифры/показатели"),
            ("discussion_deep", "Обсуждение составляет 25-35% статьи с интерпретацией и сравнением"),
        ],
    },
    "scientific_rigor": {
        "weight": 0.25,
        "checks": [
            ("fact_check_pass", "Утверждения соответствуют источникам"),
            ("citations_adequate", "Минимум 25-30 ссылок, плотность 8-10/1000 слов"),
            ("citation_style_correct", "Стиль цитирования: [Автор и др., год]"),
            ("no_plagiarism_indicators", "Нет парафразы без ссылки"),
            ("limitations_stated", "Ограничения исследования указаны"),
        ],
    },
    "style_and_language": {
        "weight": 0.15,
        "checks": [
            ("formal_register", "Академический регистр"),
            ("no_forbidden_phrases", "Нет разговорных конструкций и пустых клише"),
            ("terminology_consistent", "Терминология единообразна"),
            ("transitions_smooth", "Переходы между секциями логичны"),
            ("paragraph_depth", "Абзацы 4-8 предложений, не 1-2"),
        ],
    },
    "depth_and_analysis": {
        "weight": 0.15,
        "checks": [
            ("every_paragraph_has_numbers", "В каждом абзаце есть конкретные числа"),
            ("interpretation_not_just_facts", "Результаты интерпретируются, а не перечисляются"),
            ("comparison_with_literature", "Сравнение с другими работами в обсуждении"),
            ("mechanism_explained", "Физический/геологический механизм объяснён"),
            ("practical_significance", "Практическая значимость указана"),
        ],
    },
}

# ═══════════════════════════════════════════════════════════════
#  REVISION LOOP параметры
# ═══════════════════════════════════════════════════════════════

REVISION_CONFIG = {
    "max_rounds": 3,
    "auto_accept_threshold": 0.80,
    "minor_accept_threshold": 0.65,
    "force_accept_after_max_rounds": True,
    "critical_blocking": True,
    "round_1_strictness": 4,
    "round_2_strictness": 3,
    "round_3_strictness": 2,
}


# ═══════════════════════════════════════════════════════════════
#  FORMATTING RULES — Markdown и LaTeX инструкции для Writer
# ═══════════════════════════════════════════════════════════════

FORMATTING_RULES = {
    "markdown": {
        "equations": (
            "Inline: $V_s$, $M_w$, $\\sigma$ — single dollar signs. "
            "Display: $$ ... $$ on separate lines for standalone equations."
        ),
        "tables": (
            "Markdown pipe tables:\n"
            "| Параметр | Значение | Единица |\n"
            "|----------|----------|---------|\n"
            "| ...      | ...      | ...     |"
        ),
        "figures": (
            "![caption](figure_name.png) — markdown image syntax. "
            "Always include a descriptive caption."
        ),
        "matplotlib": (
            "When describing a chart/plot, include a matplotlib-ready code block:\n"
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "# data and plotting code\n"
            "```"
        ),
        "citations": "[Автор и др., год] or (Author et al., Year) — inline text.",
    },
    "latex": {
        "equations": (
            "Numbered: \\begin{equation}\\n  ... \\label{eq:label}\\n\\end{equation}\n"
            "Unnumbered: $$ ... $$\n"
            "Inline: $V_s$, $M_w$, $\\sigma$"
        ),
        "tables": (
            "\\begin{table}[htbp]\n"
            "\\centering\n"
            "\\caption{...}\\label{tab:label}\n"
            "\\begin{tabular}{lcc}\n"
            "\\hline\n"
            "Параметр & Значение & Единица \\\\\n"
            "\\hline\n"
            "... \\\\\n"
            "\\hline\n"
            "\\end{tabular}\n"
            "\\end{table}"
        ),
        "figures": (
            "\\begin{figure}[htbp]\n"
            "\\centering\n"
            "\\includegraphics[width=\\textwidth]{figure.png}\n"
            "\\caption{...}\\label{fig:label}\n"
            "\\end{figure}"
        ),
        "matplotlib": (
            "If describing a chart/diagram, provide matplotlib code block:\n"
            "```python\n"
            "import matplotlib.pyplot as plt\n"
            "# data and plotting code\n"
            "plt.savefig('figure_name.png', dpi=300)\n"
            "```\n"
            "Simple schemas can use TikZ."
        ),
        "citations": "\\cite{AuthorYear} — BibTeX reference format.",
    },
}


def get_article_type(article_text: str, title: str = "") -> str:
    """Определить тип статьи по контенту."""
    text_lower = (article_text + " " + title).lower()
    
    if any(w in text_lower for w in ["обзор", "систематический обзор", "bibliometric", "библиометрич"]):
        return "review"
    if any(w in text_lower for w in ["набор данных", "dataset", "база данных", "data paper"]):
        return "data_paper"
    if len(article_text) < 3000:
        return "short_communication"
    return "original_research"


def get_criteria_for_type(article_type: str) -> dict:
    """Получить критерии для типа статьи."""
    base = ARTICLE_TYPES.get(article_type, ARTICLE_TYPES["original_research"])
    return {**REVIEW_CRITERIA, "article_type": base}


def format_rubric_prompt() -> str:
    """Сформировать рубричную часть системного промпта для Reviewer'а."""
    sections_desc = "\n".join(
        f"  • {s['id']}: {s.get('description', s['label'])}"
        f"{' (ОБЯЗАТЕЛЬНО)' if s.get('required') else ''}"
        for s in EXPECTED_SECTIONS
        if s.get("required")
    )
    
    criteria_desc = ""
    for cat_name, cat_data in REVIEW_CRITERIA.items():
        checks = "\n".join(f"    - [{cid}] {cdesc}" for cid, cdesc in cat_data["checks"])
        criteria_desc += f"\n{cat_name} (вес {cat_data['weight']:.0%}):\n{checks}\n"
    
    tone_rules = "\n".join(f"  ✗ {r}" for r in TONE_RULES["forbidden"])
    tone_required = "\n".join(f"  ✓ {r}" for r in TONE_RULES["required"])
    
    return f"""=== РУБРИКА ОЦЕНКИ НАУЧНОЙ СТАТЬИ ===

ОБЯЗАТЕЛЬНЫЕ СЕКЦИИ:
{sections_desc}

КРИТЕРИИ ОЦЕНКИ ПО КАТЕГОРИЯМ:
{criteria_desc}
СТИЛИСТИЧЕСКИЕ ПРАВИЛА:
ЗАПРЕЩЕНО:
{tone_rules}
ОБЯЗАТЕЛЬНО:
{tone_required}
"""

# ═══════════════════════════════════════════════════════════════
#  CARS MODEL (Create-A-Research-Space) — Swales 1990
#  Rhetorical structure for Introduction sections
# ═══════════════════════════════════════════════════════════════

CARS_MOVES = {
    "move_1_territory": {
        "label": "Move 1: Establish Territory",
        "steps": [
            "Step 1 — Claim centrality: why this research area matters globally/regionally",
            "Step 2 — Make topic generalizations: broad statements supported by evidence [Author, Year]",
            "Step 3 — Review previous research: who did what, key findings, methods used [Author, Year]",
        ],
    },
    "move_2_niche": {
        "label": "Move 2: Establish Niche",
        "steps": [
            "Step 1A — Counter-claim: however, existing approaches have limitations...",
            "Step 1B — Indicate a gap: little attention has been paid to... / it remains unclear whether...",
            "Step 1C — Question-raising: how does X affect Y under conditions Z?",
            "Step 1D — Continue tradition: building on [Author]'s approach, this study extends...",
        ],
    },
    "move_3_occupy": {
        "label": "Move 3: Occupy Niche",
        "steps": [
            "Step 1 — Outline purpose: the aim of this study is to...",
            "Step 2 — Announce present research: we analyze N studies covering...",
            "Step 3 — Announce principal findings: our analysis reveals...",
            "Step 4 — Indicate paper structure: the paper is organized as follows...",
        ],
    },
}

# ═══════════════════════════════════════════════════════════════
#  EVIDENCE CHAINING PATTERN
#  Structure for grounding claims in source evidence
# ═══════════════════════════════════════════════════════════════

EVIDENCE_CHAINING_EXAMPLE = (
    "ПРИМЕР правильного evidence chaining:\n"
    "«Метод Random Forest показал точность 96.2% при классификации типов землепользования "
    "в условиях горной тайги [Zhang et al., 2021], что значительно превышает результаты "
    "логистической регрессии (78.4%) на том же датасете. Это подтверждается работой "
    "[Yang et al., 2024], где аналогичный подход на данных Sentinel-2 достиг AUC 0.962. "
    "Вместе с тем, [Wang, 2023] отмечает, что при малых обучающих выборках (<50 точек) "
    "преимущество исчезает, что требует дополнительных исследований в условиях недостатка разметки.»\n\n"
    "Структура: Claim → Primary evidence [Author, Year] → Supporting evidence [Author, Year] "
    "→ Counter-evidence [Author, Year] → Implication"
)

# ═══════════════════════════════════════════════════════════════
#  SECTION-SPECIFIC RHETORICAL RULES
#  Each section type has its own discourse pattern
# ═══════════════════════════════════════════════════════════════

SECTION_RHETORICAL_RULES = {
    "introduction": {
        "pattern": "CARS Model (3 moves)",
        "rules": [
            "Move 1: Establish Territory — claim centrality, review previous research",
            "Move 2: Establish Niche — counter-claim, indicate gap, raise question",
            "Move 3: Occupy Niche — state purpose, announce findings, outline structure",
            "Каждый тезис подкреплён ссылкой [Author, Year]",
            "Финальный абзац — аннотация структуры статьи",
        ],
    },
    "literature_review": {
        "pattern": "Synthesis Matrix",
        "rules": [
            "Группируй по ТЕМАМ, не по статьям ( тематический обзор, не хронологический)",
            "Для каждой темы: сравни подходы по ключевым параметрам (данные, метод, точность, регион)",
            "Используй таблицу сравнения если параметров > 3",
            "Завершай каждый тематический блок выводом: что известно, что нет",
            "Минимум 2 цитаты на абзац",
        ],
    },
    "methodology": {
        "pattern": "Reproducibility Protocol",
        "rules": [
            "Пошаговое описание: Data → Preprocessing → Model → Evaluation",
            "Для каждого шага: какие инструменты, параметры, версии ПО",
            "Укажи источники данных (DOI, URL) и период покрытия",
            "Формулы в LaTeX для ключевых метрик",
            "Воспроизводимость: чтобы читатель мог повторить",
        ],
    },
    "results": {
        "pattern": "Claim → Evidence → Interpretation → Bridge",
        "rules": [
            "Каждый абзац: один ключевой результат → evidence → что это значит",
            "Количественные данные: точные числа, таблицы, графики",
            "Сравнение с ожиданиями: совпадает / превосходит / уступает",
            "Bridge к следующему результату: «В дополнение к... мы также обнаружили...»",
            "Evidence chaining: Claim → [Author] → Supporting [Author] → Counter [Author]",
        ],
    },
    "discussion": {
        "pattern": "Start Strong → Compare → Limitations → Future",
        "rules": [
            "Начинай с самого важного finding",
            "Сравнивай с литературой: совпадения и расхождения с [Author, Year]",
            "Обязательно обсуди limitations — честно и конкретно",
            "Предложи directions для будущих исследований",
            "Не повторяй Results — интерпретируй и контекстуализируй",
        ],
    },
    "conclusion": {
        "pattern": "Summary → Contribution → Impact",
        "rules": [
            "Краткое резюме главного вклада (1-2 абзаца)",
            "Практические рекомендации для исследователей/практиков",
            "Broader impact: значение для области в целом",
            "Направления дальнейших исследований (конкретные, не абстрактные)",
        ],
    },
}

def get_rhetorical_rules(section_heading: str) -> dict:
    """Get rhetorical rules for a specific section by heading match."""
    heading_lower = section_heading.lower().replace(" ", "_")
    for key, rules in SECTION_RHETORICAL_RULES.items():
        if key.replace("_", " ") in heading_lower.replace("_", " ") or key in heading_lower:
            return {"section_key": key, **rules}
    # Default: generic rules
    return {
        "section_key": "generic",
        "pattern": "Standard academic",
        "rules": [
            "Каждый абзац содержит чёткий тезис",
            "Тезис подкреплён evidence [Author, Year]",
            "Логичный переход к следующему абзацу",
        ],
    }
