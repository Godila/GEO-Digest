"""Writer Agent — Multi-Pass StructuredDraft → WrittenArticle.

Стратегия (калибровка по журналу «Геология и геофизика Юга России»):
  Pass 1: Outline — детальный план с тезисами каждого абзаца (~400 слов)
  Pass 2: Expand — полный текст каждой секции по плану (~4000-5500 слов)
  Pass 3: Polish — шлифовка связности, цитирования, переходов

Целевые параметры (измерено по 22 статьям):
  Средний объём: 4407 слов
  Среднее ссылок: 38.3
  Плотность цитирования: 8.6/1000 слов
"""

from __future__ import annotations

from engine.agents.base import BaseAgent, LLMCallMixin
from engine.agents.tools import AgentTools
from engine.schemas import (
    GroupType, WrittenArticle, AgentResult,
    StructuredDraft,
)
from engine.prompts.writer_prompts import (
    build_outline_system_prompt, build_outline_user_prompt,
    build_expand_system_prompt, build_expand_user_prompt,
    build_polish_system_prompt, build_polish_user_prompt,
    build_target_word_count, build_length_instruction, build_max_tokens,
)


class WriterAgent(BaseAgent, LLMCallMixin):
    """Пишет статью через Multi-Pass генерацию.

    Uses dedicated Writer LLM (OpenRouter/Gemini Flash Lite) — not self.llm (MiniMax).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._writer_llm = None

    @property
    def writer_llm(self):
        """Lazy-load dedicated Writer LLM (OpenRouter/Gemini Flash Lite)."""
        if self._writer_llm is None:
            from engine.llm.config import get_writer_llm
            self._writer_llm = get_writer_llm()
        return self._writer_llm

    def call_llm(self, prompt, system="", max_tokens=0, temperature=0.3,
                 parse_json=False, timeout=0):
        """Override call_llm to use Writer LLM instead of default MiniMax."""
        from engine.llm.config import WRITER_LLM_CONFIG
        timeout_sec = timeout or self._LLM_TIMEOUT_SECONDS
        # Use writer-specific temperature as default
        effective_temp = temperature if temperature != 0.3 else WRITER_LLM_CONFIG.get("temperature", 0.3)

        def _do_call():
            if parse_json:
                raw = self.writer_llm.complete_json(prompt, system=system, max_tokens=max_tokens or 4096)
                if isinstance(raw, str):
                    import json
                    try:
                        return json.loads(raw)
                    except json.JSONDecodeError:
                        self._log(f"call_llm: JSON parse failed, returning raw string ({len(raw)} chars)")
                        return raw
                return raw
            return self.writer_llm.complete(prompt, system=system, max_tokens=max_tokens or 4096, temperature=effective_temp)

        return self._run_with_timeout(_do_call, timeout_sec)

    @property
    def name(self) -> str:
        return "writer"

    def run(
        self,
        draft: StructuredDraft | None = None,
        style: str = "academic",
        language: str = "ru",
        format_: str = "markdown",
        user_comment: str = "",
        **kwargs,
    ) -> AgentResult:
        """
        Написать статью (Multi-Pass).

        Args:
            draft: StructuredDraft от Reader'а
            style: academic | blog | popular
            language: ru | en
            format_: markdown | latex
            user_comment: дополнительные указания пользователя

        Returns:
            AgentResult с WrittenArticle в .data
        """
        self._log(f"Начало написания (multi-pass): type={draft.group_type.value if draft else '?'}, lang={language}")

        if not draft:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="Не передан draft (StructuredDraft)",
            )

        try:
            group_type = draft.group_type or GroupType.REPLICATION
            tools = AgentTools(self.storage)

            # Собираем контекст источников
            reader_context = kwargs.get("reader_context", "") or ""
            source_info = self._build_source_info(draft, tools)
            full_context = reader_context or source_info

            # ─── PASS 1: OUTLINE ─────────────────────────────
            self._log("Pass 1: Генерация плана статьи...")
            outline = self._pass_outline(draft, full_context, group_type, language)

            # ─── PASS 2: SECTION-BY-SECTION EXPAND ──────────
            self._log("Pass 2: Расширение по секциям...")
            expanded = self._pass_expand_sections(outline, full_context, group_type, language, format_)

            # ─── PASS 3: POLISH ──────────────────────────────
            self._log("Pass 3: Шлифовка статьи...")
            article = self._pass_polish(expanded, draft, format_, language)

            self._log(f"Готово! Объём: {article.word_count} слов")
            return AgentResult(agent_name=self.name, success=True, data=article)

        except Exception as e:
            self._log(f"Ошибка multi-pass: {e}")
            # Fallback 1: single-pass (старый подход)
            self._log("Fallback: переключаюсь на single-pass...")
            single = self._run_single_pass(draft, style, language, format_, user_comment)
            if single.success:
                return single
            # Fallback 2: return outline as minimal article
            try:
                _ = outline  # check if outline exists from Pass 1
                self._log("Fallback: использую outline как черновик статьи")
                from engine.schemas import WrittenArticle
                outline_text = str(outline) if not isinstance(outline, str) else outline
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    data=WrittenArticle(
                        title=draft.title_suggestion or "",
                        text=outline_text,
                        format_=format_,
                        word_count=len(outline_text.split()),
                    ),
                )
            except NameError:
                pass
            return AgentResult(agent_name=self.name, success=False, error=str(e))

    # ─────────────────────────────────────────────────────────
    #  PASS 1: OUTLINE
    # ─────────────────────────────────────────────────────────

    def _pass_outline(self, draft, source_info, group_type, language) -> str:
        """Pass 1: Generate detailed article outline."""
        system = build_outline_system_prompt(group_type, language)
        user = build_outline_user_prompt(draft, source_info)

        result = self.call_llm(
            prompt=user,
            system=system,
            max_tokens=3000,
            parse_json=True,
            temperature=0.3,
        )

        # Outline — это JSON с планом, сохраняем как строку для pass 2
        if isinstance(result, dict):
            import json
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    # ─────────────────────────────────────────────────────────
    #  PASS 2: EXPAND
    # ─────────────────────────────────────────────────────────

    def _pass_expand(self, outline: str, source_info, group_type, language) -> str:
        """Pass 2: Expand outline into full article text."""
        system = build_expand_system_prompt(group_type, language)
        user = build_expand_user_prompt(outline, source_info)
        max_tokens = build_max_tokens(group_type)

        result = self.call_llm(
            prompt=user,
            system=system,
            max_tokens=max_tokens,
            parse_json=True,
            temperature=0.4,
        )

        if isinstance(result, dict):
            import json
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    # ─────────────────────────────────────────────────────────
    #  PASS 2b: SECTION-BY-SECTION EXPAND
    # ─────────────────────────────────────────────────────────

    def _pass_expand_sections(
        self, outline: str, full_context: str, group_type: GroupType,
        language: str, format_: str = "markdown",
    ) -> str:
        """Expand outline section-by-section: one LLM call per section.
        
        Instead of one giant 13K token call, we make 5-6 small calls (2-4K tokens each).
        Each section gets its own relevant context from rich_context.
        """
        import json
        from engine.prompts.writer_prompts import (
            extract_section_context, get_section_target,
            build_section_expand_system_prompt, build_section_expand_user_prompt,
        )

        # Parse outline to extract sections
        outline_data = self._parse_outline_sections(outline)
        sections = outline_data.get("sections", [])
        
        if not sections:
            self._log("Outline has no sections, falling back to full expand")
            return self._pass_expand(outline, full_context, group_type, language)

        expanded_sections = []
        prev_summary = ""
        rich_context = full_context  # Full context — no truncation here

        for i, sec in enumerate(sections):
            heading = sec.get("heading", sec.get("section", f"Секция {i+1}"))
            sec_outline = sec.get("outline", sec.get("paragraphs", str(sec)))
            if isinstance(sec_outline, list):
                sec_outline = json.dumps(sec_outline, ensure_ascii=False, indent=2)
            
            # Extract section-specific context (Step 3: full context by section)
            section_ctx = extract_section_context(heading, rich_context)
            
            target = get_section_target(heading)
            
            self._log(f"  Секция {i+1}/{len(sections)}: '{heading}' → {target['tokens']} tokens")

            system = build_section_expand_system_prompt(group_type, language, heading, format_)
            user = build_section_expand_user_prompt(heading, sec_outline, section_ctx, prev_summary)

            try:
                result = self.call_llm(
                    prompt=user,
                    system=system,
                    max_tokens=target["tokens"],
                    parse_json=True,
                    temperature=0.4,
                )
            except (TimeoutError, Exception) as e:
                self._log(f"  ⚠ Секция '{heading}' timeout/error: {e}. Пропускаю.")
                expanded_sections.append({"heading": heading, "content": sec_outline})
                continue

            # Parse result
            if isinstance(result, dict):
                content = result.get("content", "")
                if content:
                    expanded_sections.append({"heading": heading, "content": content})
                    # Keep last 500 chars as summary for transition to next section
                    prev_summary = content[-500:] if len(content) > 500 else content
                else:
                    expanded_sections.append({"heading": heading, "content": str(result)})
            else:
                expanded_sections.append({"heading": heading, "content": str(result)})

            self._log(f"  ✓ Секция '{heading}': {len(str(expanded_sections[-1].get('content', '')))} chars")

        total_chars = sum(len(s.get("content", "")) for s in expanded_sections)
        self._log(f"Section-by-section done: {len(expanded_sections)} sections, {total_chars} chars total")

        # Return as JSON for pass 3 (polish)
        return json.dumps({
            "title": outline_data.get("title", ""),
            "sections": expanded_sections,
        }, ensure_ascii=False, indent=2)

    def _parse_outline_sections(self, outline: str) -> dict:
        """Parse outline JSON into section list."""
        import json
        try:
            data = json.loads(outline) if isinstance(outline, str) else outline
            if isinstance(data, dict):
                # Outline format: {"title": "...", "outline": [{"section": "...", "paragraphs": [...]}]}
                raw_sections = data.get("outline", data.get("sections", []))
                if isinstance(raw_sections, list):
                    sections = []
                    for s in raw_sections:
                        if isinstance(s, dict):
                            heading = s.get("section", s.get("heading", ""))
                            sections.append({"heading": heading, "outline": s})
                        else:
                            sections.append({"heading": str(s), "outline": str(s)})
                    return {"title": data.get("title", ""), "sections": sections}
                return data
            return {"title": "", "sections": []}
        except (json.JSONDecodeError, TypeError):
            # Try splitting plain text outline by sections
            return self._split_text_outline(outline)

    def _split_text_outline(self, outline: str) -> dict:
        """Fallback: split plain text outline by ## headers."""
        import re
        parts = re.split(r'(?:^|\n)(?:##|#)\s+(.+)', outline)
        sections = []
        if len(parts) > 1:
            for i in range(1, len(parts), 2):
                heading = parts[i].strip()
                content = parts[i+1].strip() if i+1 < len(parts) else ""
                sections.append({"heading": heading, "outline": content})
        if not sections:
            sections.append({"heading": "Основная часть", "outline": outline})
        return {"title": "", "sections": sections}

    # ─────────────────────────────────────────────────────────
    #  REVISION: переработка по замечаниям Reviewer (Шаг 4)
    # ─────────────────────────────────────────────────────────

    def rewrite_article(
        self,
        article_text: str,
        revision_instructions: list,
        language: str = "ru",
        format_: str = "markdown",
    ) -> str:
        """Rewrite article based on reviewer's revision instructions.
        
        Used by orchestrator's review→rewrite loop.
        Returns the revised article text (JSON string).
        """
        from engine.prompts.writer_prompts import (
            build_revision_system_prompt, build_revision_user_prompt,
        )

        self._log(f"Переработка статьи по {len(revision_instructions)} замечаниям...")

        system = build_revision_system_prompt(language)
        user = build_revision_user_prompt(article_text, revision_instructions)

        result = self.call_llm(
            prompt=user,
            system=system,
            max_tokens=16384,
            parse_json=True,
            temperature=0.3,
        )

        import json
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

    # ─────────────────────────────────────────────────────────
    #  PASS 3: POLISH
    # ─────────────────────────────────────────────────────────

    def _pass_polish(self, expanded_json: str, draft, format_, language) -> WrittenArticle:
        """Pass 3: Polish the expanded article."""
        system = build_polish_system_prompt(language)
        user = build_polish_user_prompt(expanded_json)

        raw = self.call_llm(
            prompt=user,
            system=system,
            max_tokens=16384,
            parse_json=True,
            temperature=0.2,
        )

        return self._parse_written(raw, draft, format_, language)

    # ─────────────────────────────────────────────────────────
    #  FALLBACK: Single-Pass (старый подход)
    # ─────────────────────────────────────────────────────────

    def _run_single_pass(
        self, draft, style, language, format_, user_comment
    ) -> AgentResult:
        """Single-pass fallback при ошибке multi-pass."""
        try:
            tools = AgentTools(self.storage)
            source_info = self._build_source_info(draft, tools)

            # Упрощённый промпт
            system = build_expand_system_prompt(draft.group_type, language)
            prompt = self._build_prompt(draft, source_info, style, language, format_, user_comment)
            max_tokens = build_max_tokens(draft.group_type)

            raw = self.call_llm(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                parse_json=True,
                temperature=0.4,
            )

            article = self._parse_written(raw, draft, format_, language)
            return AgentResult(agent_name=self.name, success=True, data=article)
        except Exception as e:
            return AgentResult(agent_name=self.name, success=False, error=str(e))

    # ─────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────

    def _build_source_info(self, draft: StructuredDraft, tools: AgentTools) -> str:
        """Собрать информацию об источниках."""
        if draft.rich_context:
            return f"=== ДЕТАЛЬНЫЙ АНАЛИЗ ИСТОЧНИКОВ ===\n\n{draft.rich_context}"

        parts = []
        for i, doi in enumerate(draft.source_articles[:15], 1):
            art = tools.search_by_doi(doi)
            if art:
                part = f"{i}. {art.title}"
                if art.authors:
                    part += f" ({art.authors})"
                if art.abstract:
                    part += f"\n   Аннотация: {art.abstract[:800]}"
                if art.llm_summary:
                    part += f"\n   LLM-резюме: {art.llm_summary[:500]}"
                parts.append(part)
            else:
                parts.append(f"{i}. DOI: {doi} (не найдена в хранилище)")

        if not parts:
            return "(Источники не указаны)"
        return "\n\n".join(parts)

    def _build_prompt(
        self, draft, source_info, style, language, format_, user_comment
    ) -> str:
        """Собрать промпт для fallback single-pass."""
        length_instruction = build_length_instruction(draft.group_type, language)

        return f"""НАПИСАТЬ СТАТЬЮ

Заголовок: {draft.title_suggestion or '(сгенерировать)'}
Аннотация-набросок: {draft.abstract_suggestion or '(сгенерировать)'}
Тип: {draft.group_type.value}
Язык: {language}
Комментарий пользователя: {user_comment or '(нет)'}

{length_instruction}

== ИСТОЧНИКИ ==
{source_info}

Напиши полную, содержательную статью типа '{draft.group_type.value}' на {language}."""

    def _parse_written(
        self,
        raw: dict | str,
        draft: StructuredDraft,
        format_: str,
        language: str,
    ) -> WrittenArticle:
        """Парсим LLM-ответ в WrittenArticle."""
        if isinstance(raw, str):
            return WrittenArticle(
                text=raw,
                title=draft.title_suggestion or "Generated Article",
                format_=format_,
                language=language,
                word_count=len(raw.split()),
                metadata={"source_draft_id": draft.draft_id},
            )

        if not isinstance(raw, dict):
            raw = {}

        # Собираем секции
        sections = raw.get("sections", [])
        if isinstance(sections, list):
            sections = [
                {"heading": s.get("heading", ""), "content": s.get("content", "")}
                if isinstance(s, dict) else {"heading": "", "content": str(s)}
                for s in sections
            ]
        else:
            sections = []

        # Полный текст из секций
        text_parts = []
        title = raw.get("title", draft.title_suggestion or "Generated Article")
        text_parts.append(f"# {title}\n")
        for sec in sections:
            if sec["heading"]:
                text_parts.append(f"## {sec['heading']}\n")
            text_parts.append(sec["content"] + "\n")
        full_text = "\n".join(text_parts)

        references = raw.get("references", [])
        if isinstance(references, list) and references:
            refs_text = "\n".join(f"- {r}" for r in references)
            full_text += f"\n## Список литературы\n{refs_text}\n"

        word_count = len(full_text.split())

        return WrittenArticle(
            text=full_text,
            title=title,
            format_=format_,
            language=language,
            word_count=word_count,
            references=references if isinstance(references, list) else [],
            sections=sections,
            metadata={
                "source_draft_id": draft.draft_id,
                "group_type": draft.group_type.value,
                "multi_pass": True,
            },
        )

    def estimate_cost(self, draft=None, num_sources: int = 5) -> dict:
        """Оценить стоимость multi-pass генерации."""
        input_tokens = 3000 + (num_sources * 500)
        output_tokens = 4000 + 5000 + 2000  # outline + expand + polish
        return {
            "estimated_tokens": input_tokens + output_tokens,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": output_tokens,
        }

    def validate_input(self, **kwargs) -> tuple[bool, str]:
        draft = kwargs.get("draft")
        if not draft:
            return False, "обязателен параметр 'draft' (StructuredDraft)"
        return True, ""
