"""Writer Agent — Multi-Pass StructuredDraft → WrittenArticle.

Стратегия:
  Pass 1: Outline — детальный план с тезисами каждого абзаца
  Pass 2: Section-by-Section Expand — полный текст каждой секции
  Assemble — сборка секций + список литературы (без LLM)

Evidence-Grounded Writing v3: structured evidence extraction (reader.py),
CARS rhetorical model + evidence chaining (article_patterns.py + writer_prompts.py).
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
        timeout_sec = timeout or self._LLM_TIMEOUT_SECONDS

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
            ev_blocks = getattr(draft, 'evidence_blocks', [])
            expanded = self._pass_expand_sections(outline, full_context, group_type, language, format_,
                                                   evidence_blocks=ev_blocks)

            # ─── SKIP PASS 3 (POLISH) for section-by-section ───
            # Section-by-section уже производит качественный текст.
            # Polish отправляет 40K+ chars в LLM → reasoning model обрезает JSON → статья теряется.
            self._log("Пропускаю Pass 3 (полировка) — section-by-section уже достаточно")
            article = self._assemble_sections(expanded, draft, format_, language)

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
            max_tokens=16000,
            parse_json=True,
            temperature=0.3,
        )

        # Outline — JSON с планом, сохраняем как строку для pass 2
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
        evidence_blocks: list[dict] | None = None,
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
        ev_blocks = evidence_blocks or []

        for i, sec in enumerate(sections):
            heading = sec.get("heading", sec.get("section", f"Секция {i+1}"))
            sec_outline = sec.get("outline", sec.get("paragraphs", str(sec)))
            if isinstance(sec_outline, list):
                sec_outline = json.dumps(sec_outline, ensure_ascii=False, indent=2)
            
            # Extract section-specific context (Step 3: full context by section)
            section_ctx = extract_section_context(heading, rich_context)

            # Evidence-grounded: add structured evidence quotes for this section
            if ev_blocks:
                section_evidence = self._gather_section_evidence(heading, ev_blocks)
                if section_evidence:
                    section_ctx = section_evidence + "\n\n" + section_ctx

            # Perspective questions for evidence-grounded structure
            perspective_q = self._generate_perspective_questions(heading, ev_blocks)
            
            target = get_section_target(heading)
            
            self._log(f"  Секция {i+1}/{len(sections)}: '{heading}' → {target['tokens']} tokens"
                      + (f", {sum(len(e.get('quotes',[])) for e in ev_blocks)} evidence" if ev_blocks else ""))

            system = build_section_expand_system_prompt(group_type, language, heading, format_)
            user = build_section_expand_user_prompt(heading, sec_outline, section_ctx, prev_summary,
                                                     perspective_questions=perspective_q)

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

    def _assemble_sections(self, expanded_json: str, draft, format_, language) -> "WrittenArticle":
        """Assemble expanded sections into WrittenArticle without LLM call.
        
        Used instead of _pass_polish for section-by-section writing:
        each section is already polished during expansion.
        Appends formatted references section from DOI metadata.
        """
        import json
        from engine.schemas import WrittenArticle
        
        # Parse expanded sections
        try:
            data = json.loads(expanded_json) if isinstance(expanded_json, str) else expanded_json
        except json.JSONDecodeError:
            data = {"sections": []}
        
        title = data.get("title", "") or getattr(draft, 'title', '')
        sections = data.get("sections", [])
        references = data.get("references", [])
        
        # Assemble markdown article
        parts = []
        if title:
            parts.append(f"# {title}\n")
        
        for sec in sections:
            heading = sec.get("heading", "")
            content = sec.get("content", "")
            if heading:
                parts.append(f"\n## {heading}\n")
            if content:
                parts.append(content)
        
        # ── Generate references from DOI metadata ──
        ref_entries = []
        
        # Try references from LLM first
        if isinstance(references, list) and references:
            ref_entries = [str(r) for r in references if r]
        
        # If LLM didn't return references, build from DOI metadata
        if not ref_entries:
            ref_entries = self._build_references_from_draft(draft)
        
        if ref_entries:
            parts.append(f"\n## Список литературы\n")
            for i, ref in enumerate(ref_entries, 1):
                parts.append(f"{i}. {ref}")
        
        text = "\n\n".join(parts)
        words = len(text.split())
        
        return WrittenArticle(
            text=text,
            word_count=words,
            title=title,
            format_=format_,
            language=language,
            references=ref_entries,
        )

    def _build_references_from_draft(self, draft) -> list[str]:
        """Build formatted references from draft DOI metadata."""
        from engine.agents.tools import AgentTools
        tools = AgentTools(self.storage)
        
        dois = getattr(draft, 'source_articles', []) or getattr(draft, 'key_references', []) or []
        refs = []
        
        for doi in dois[:25]:
            doi_clean = doi.strip().strip('`').strip('*"')
            art = tools.search_by_doi(doi_clean)
            if art:
                # Build bibliographic entry
                authors = getattr(art, 'authors', '') or ''
                year = getattr(art, 'year', '') or ''
                art_title = getattr(art, 'title', '') or ''
                journal = getattr(art, 'journal', '') or getattr(art, 'container_title', '') or ''
                volume = getattr(art, 'volume', '') or ''
                pages = getattr(art, 'pages', '') or ''
                
                entry = ""
                if authors:
                    entry += f"{authors}"
                if year:
                    entry += f" ({year})"
                if art_title:
                    entry += f" {art_title}."
                if journal:
                    entry += f" {journal}"
                if volume:
                    entry += f", {volume}"
                if pages:
                    entry += f", {pages}"
                entry += f" DOI: {doi_clean}"
                
                if entry.strip():
                    refs.append(entry.strip())
            else:
                # Fallback: just DOI
                refs.append(f"DOI: {doi_clean}")
        
        return refs

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
            max_tokens=32000,
            parse_json=True,
            temperature=0.3,
        )

        import json
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        return str(result)

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
    #  EVIDENCE-GROUNDED WRITING HELPERS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _gather_section_evidence(
        heading: str, evidence_blocks: list[dict], max_quotes: int = 12
    ) -> str:
        """Select evidence quotes relevant to a specific article section.

        Uses keyword matching between section heading/keywords and
        evidence quote keywords. Returns formatted evidence string.
        """
        # Section heading → keywords for matching
        heading_lower = heading.lower()
        section_keywords = set(heading_lower.split())
        # Map common section names to search terms
        section_kw_map = {
            "introduction": {"introduction", "background", "context", "motivation", "significance", "relevance", "overview", "territory", "niche"},
            "литературный обзор": {"literature", "review", "previous", "related", "comparison", "synthesis", "trend", "approach"},
            "обзор литературы": {"literature", "review", "previous", "related", "comparison", "synthesis", "trend", "approach"},
            "literature review": {"literature", "review", "previous", "related", "comparison", "synthesis", "trend", "approach"},
            "methodology": {"method", "approach", "data", "model", "algorithm", "tool", "software", "dataset", "parameter"},
            "методология": {"method", "approach", "data", "model", "algorithm", "tool", "software", "dataset", "parameter"},
            "results": {"result", "accuracy", "performance", "score", "metric", "finding", "value", "correlation"},
            "результаты": {"result", "accuracy", "performance", "score", "metric", "finding", "value", "correlation"},
            "discussion": {"discussion", "implication", "limitation", "compare", "disagree", "contrast", "future"},
            "обсуждение": {"discussion", "implication", "limitation", "compare", "disagree", "contrast", "future"},
            "conclusion": {"conclusion", "summary", "contribution", "impact", "recommendation", "future", "direction"},
            "заключение": {"conclusion", "summary", "contribution", "impact", "recommendation", "future", "direction"},
        }
        for kw_key, kw_set in section_kw_map.items():
            if kw_key in heading_lower:
                section_keywords.update(kw_set)
                break

        # Gather and rank relevant quotes
        scored_quotes = []
        for block in evidence_blocks:
            source = block.get("source", "Unknown")
            doi = block.get("doi", "")
            for q in block.get("quotes", []):
                quote_text = q.get("text", "")
                if not quote_text:
                    continue
                # Score by keyword overlap
                quote_kw = set(k.lower() for k in q.get("keywords", []))
                overlap = len(section_keywords & quote_kw)
                # Boost claim_types relevant to section
                claim_type = q.get("claim_type", "")
                type_boost = 0
                if "introduction" in heading_lower and claim_type in ("finding", "gap", "comparison"):
                    type_boost = 2
                elif "methodology" in heading_lower and claim_type == "method_result":
                    type_boost = 3
                elif "result" in heading_lower and claim_type in ("method_result", "finding"):
                    type_boost = 3
                elif "discussion" in heading_lower and claim_type in ("limitation", "comparison", "gap", "recommendation"):
                    type_boost = 3
                elif "conclusion" in heading_lower and claim_type in ("recommendation", "finding"):
                    type_boost = 2
                elif "literature" in heading_lower or "обзор" in heading_lower:
                    type_boost = 1

                score = overlap + type_boost
                if score > 0:
                    cite = f"[{source}]"
                    if doi:
                        cite += f" (doi:{doi})"
                    scored_quotes.append((score, quote_text, cite, q.get("claim_type", "")))

        # Sort by score descending, take top N
        scored_quotes.sort(key=lambda x: x[0], reverse=True)
        top = scored_quotes[:max_quotes]

        if not top:
            return ""

        parts = ["=== СТРУКТУРИРОВАННЫЕ EVIDENCE ДЛЯ ЭТОЙ СЕКЦИИ ===",
                 "Используй эти цитаты verbatim из источников. Каждая цитата подкрепляет конкретный тезис.\n"]
        for i, (score, quote, cite, ctype) in enumerate(top, 1):
            parts.append(f"[E{i}] ({ctype}) {cite}")
            parts.append(f"  «{quote}»")
            parts.append("")

        return "\n".join(parts)

    def _generate_perspective_questions(
        self, heading: str, evidence_blocks: list[dict], language: str = "ru"
    ) -> str:
        """Generate 3-5 key questions for a section based on available evidence.

        ONE call per article (not per section) — questions are pre-generated
        and answers are extracted from evidence without additional LLM calls.
        """
        # Collect unique key_numbers across all blocks for this section
        all_numbers = []
        for block in evidence_blocks:
            all_numbers.extend(block.get("key_numbers", []))

        # Collect claim types present
        claim_types = set()
        for block in evidence_blocks:
            for q in block.get("quotes", []):
                claim_types.add(q.get("claim_type", ""))

        # Generate questions based on section + available evidence
        heading_lower = heading.lower()
        questions = []

        if "introduction" in heading_lower:
            questions = [
                "Почему эта тема важна для геоэкологии? (territory claim)",
                "Какие подходы применялись ранее и в чём их ограничения? (gap identification)",
                "Каков вклад данного исследования по сравнению с существующими работами? (contribution)",
            ]
        elif "literature" in heading_lower or "обзор" in heading_lower:
            questions = [
                "Какие методологии применяются в данной области? (comparison)",
                "Где авторы расходятся в выводах? (contradiction)",
                "Какие пробелы в знаниях выявлены? (gap)",
                "Какие тенденции наблюдаются за последние 5 лет? (trend)",
            ]
        elif "method" in heading_lower:
            questions = [
                "Какие данные и инструменты использованы? (data & tools)",
                "Какие параметры моделей оптимальны? (parameters)",
                "Как обеспечивает воспроизводимость? (reproducibility)",
            ]
        elif "result" in heading_lower:
            questions = [
                "Какие ключевые количественные результаты получены? (numbers)",
                "Как результаты соотносятся с предыдущими работами? (comparison)",
                "Какие закономерности выявлены? (patterns)",
            ]
        elif "discussion" in heading_lower:
            questions = [
                "Какие ограничения у подхода? (limitations)",
                "Где результаты противоречат литературе? (contradictions)",
                "Какие рекомендации для будущих исследований? (future work)",
            ]
        elif "conclusion" in heading_lower:
            questions = [
                "Каков главный вклад исследования? (contribution)",
                "Какие практические рекомендации? (practical implications)",
                "Какие направления дальнейших исследований? (future directions)",
            ]
        else:
            questions = [
                "Какие ключевые факты из источников релевантны этой секции?",
                "Какие противоречия между источниками стоит обсудить?",
            ]

        return "\n".join(f"  Q{i+1}: {q}" for i, q in enumerate(questions))

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
