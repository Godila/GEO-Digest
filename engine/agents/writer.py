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
    """Пишет статью через Multi-Pass генерацию."""

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
            self._log("Pass 1/3: Генерация плана статьи...")
            outline = self._pass_outline(draft, full_context, group_type, language)

            # ─── PASS 2: EXPAND ──────────────────────────────
            self._log("Pass 2/3: Расширение плана в полный текст...")
            expanded = self._pass_expand(outline, full_context, group_type, language)

            # ─── PASS 3: POLISH ──────────────────────────────
            self._log("Pass 3/3: Шлифовка статьи...")
            article = self._pass_polish(expanded, draft, format_, language)

            self._log(f"Готово! Объём: {article.word_count} слов")
            return AgentResult(agent_name=self.name, success=True, data=article)

        except Exception as e:
            self._log(f"Ошибка multi-pass: {e}")
            # Fallback: single-pass (старый подход)
            self._log("Fallback: переключаюсь на single-pass...")
            return self._run_single_pass(draft, style, language, format_, user_comment)

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
