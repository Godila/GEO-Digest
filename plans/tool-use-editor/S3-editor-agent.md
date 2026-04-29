# Стадия S3: Editor Agent (Core Logic)

## Цель
Создать EditorAgent — главный агент который использует ToolUseLoop + StorageTools
для анализа темы и генерации предложений статей. Это "мозг" новой архитектуры.

## Архитектура

```
EditorAgent (extends BaseAgent)
│
├── run(topic, domain, user_instruction) → EditorResult
│   ├── Phase 1: Анализ хранилища (через ToolUseLoop)
│   │   System prompt: ANALYSIS_SYSTEM_PROMPT
│   │   Tools: count_by_cluster, get_time_range, search_storage, check_existing
│   │   → StorageAnalysis
│   │
│   ├── Phase 2: Генерация предложений (через ToolUseLoop)
│   │   System prompt: PROPOSAL_SYSTEM_PROMPT (включает результаты Phase 1)
│   │   Tools: search_storage (детальный), validate_doi, get_article_detail
│   │   → ArticleProposal[]
│   │
│   └── Post-validation
│       ├── DOI validation против storage
│       ├── Duplicate check
│       └── Confidence scoring
│
├── develop(proposal_id, user_feedback) → DevelopmentResult
│   └── Итеративная доработка выбранного предложения
│
└── resume(job_id) → EditorResult
    └── Возобновление с checkpoint'а
```

## Data Models

```python
# engine/schemas.py (дополнить)

@dataclass
class ArticleProposal:
    id: str                           # "prop_20260426_001"
    title: str                        # Заголовок
    thesis: str                       # Тезис 2-3 предложения
    target_audience: str              # researchers | general_public | policy_makers
    confidence: float                 # 0.0-1.0
    sources_available: int            # Сколько уже есть в базе
    sources_needed: int               # Сколько нужно найти
    key_references: list[str]         # ["DOI:10.xxx/..."]
    gap_filled: str                   # Какую gap закрывает
    estimated_sections: list[str]     # ["введение", "данные", "выводы"]
    status: str = "proposed"          # proposed | selected | in_progress | done
    
@dataclass 
class StorageAnalysis:
    total_articles: int
    relevant_count: int               # по теме
    clusters: list[dict]              # [{theme, count}]
    year_range: tuple[int, int]       # (min, max)
    by_year: dict[int, int]
    source_distribution: dict[str, int]
    existing_articles: list[dict]     # Уже написанные похожие статьи
    gaps: list[str]                   # Выявленные пробелы в покрытии
    raw_llm_analysis: str             # Сырой ответ LLM для отладки

@dataclass
class EditorResult:
    job_id: str
    topic: str
    status: str                       # done | partial | failed
    analysis: StorageAnalysis | None
    proposals: list[ArticleProposal]
    tool_rounds_total: int
    total_tokens_used: int
    duration_sec: float
    warnings: list[str]
    error: str | None

@dataclass
class EditorState:
    """Состояние для checkpoint/resume."""
    job_id: str
    topic: str
    domain: str | None
    phase: str                        # idle | analyzing | proposing | developing | done
    analysis: dict | None             # Сериализованный StorageAnalysis
    proposals: list[dict] | None      # Сериализованные proposals
    selected_proposal_id: str | None
    development_history: list[dict]   # [{user_msg, agent_response, round}]
    error: str | None
    started_at: str
    updated_at: str
```

## Что делаем

### 3.1 Editor Agent
Файл: `engine/agents/editor.py`

```python
class EditorAgent(BaseAgent):
    """Главный редакторский агент с tool-use архитектурой."""
    
    def __init__(self, storage: JsonlStorage = None, llm: LLMProvider = None):
        super().__init__()
        self.storage = storage or JsonlStorage()
        self.llm = llm or get_llm()
        self.tools = create_storage_tools(self.storage)
        self.loop = ToolUseLoop(self.llm, self.tools)
        self._jobs_dir = Path("/app/data/jobs")
        self._jobs_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, topic: str, domain: str | None = None,
            user_instruction: str | None = None,
            max_proposals: int = 5) -> EditorResult:
        """
        Основной метод: анализ + генерация предложений.
        
        Args:
            topic: Тема для анализа
            domain: Более широкий домен (опционально)
            user_instruction: Дополнительная инструкция от пользователя
            max_proposals: Максимум предложений
            
        Returns:
            EditorResult с анализом и предложениями
        """
        import time
        t0 = time.time()
        
        state = EditorState(
            job_id=self._gen_job_id(),
            topic=topic,
            domain=domain,
            phase="analyzing",
            started_at=now_iso(),
            updated_at=now_iso(),
        )
        
        try:
            # ═══ Phase 1: Analysis ═══
            self._log(f"[editor] Phase 1: Анализ хранилища...")
            
            analysis_result = self.loop.run(
                user_message=(
                    f"Проанализируй тему '{topic or domain}' "
                    f"{'(домен: ' + domain + ')' if domain else ''}. "
                    f"{user_instruction or ''}"
                ),
                system_prompt=ANALYSIS_SYSTEM_PROMPT,
                temperature=0.1,  # Минимальная креативность — нужен точный анализ
                max_tokens=2048,
            )
            
            analysis = self._build_analysis(analysis_result)
            state.analysis = asdict(analysis)
            state.phase = "proposing"
            state.updated_at = now_iso()
            self._save_checkpoint(state)
            
            self._log(f"[editor] Phase 1 done: {analysis.relevant_count} статей, "
                     f"{len(analysis.clusters)} кластеров, "
                     f"{len(analysis.gaps)} gaps найдено")
            
            # ═══ Phase 2: Proposals ═══
            self._log(f"[editor] Phase 2: Генерация предложений...")
            
            # Встраиваем результаты анализа в системный промпт
            context = self._format_analysis_context(analysis)
            proposal_prompt = PROPOSAL_SYSTEM_PROMPT.format(
                analysis_context=context,
                max_proposals=max_proposals,
                topic=topic,
            )
            
            proposal_result = self.loop.run(
                user_message=f"Предложи {max_proposals} вариантов статей.",
                system_prompt=proposal_prompt,
                temperature=0.25,  # Немного креативности для заголовков
                max_tokens=4096,
            )
            
            raw_proposals = parse_proposals_from_text(proposal_result.content)
            
            # Post-валидация каждого proposal
            validated = []
            for i, p in enumerate(raw_proposals):
                vp = self._validate_proposal(p, analysis, i)
                validated.append(vp)
            
            state.proposals = [asdict(p) for p in validated]
            state.phase = "done"
            state.updated_at = now_iso()
            self._save_checkpoint(state)
            
            elapsed = time.time() - t0
            
            return EditorResult(
                job_id=state.job_id,
                topic=topic,
                status="done",
                analysis=analysis,
                proposals=validated,
                tool_rounds_total=(analysis_result.total_rounds + 
                                   proposal_result.total_rounds),
                total_tokens_used=(
                    analysis_result.usage.get("input_tokens", 0) +
                    proposal_result.usage.get("input_tokens", 0)
                ),
                duration_sec=elapsed,
                warnings=analysis_result.warnings + proposal_result.warnings,
                error=None,
            )
            
        except Exception as e:
            state.phase = "failed"
            state.error = str(e)
            state.updated_at = now_iso()
            self._save_checkpoint(state)
            raise
    
    def develop(self, proposal_id: str, user_feedback: str,
               job_id: str | None = None) -> DevelopmentResult:
        """
        Итеративная доработка выбранного предложения.
        
        Пользователь говорит: "Добавь про abrupt thaw events"
        Агент: обновляет thesis, ищет дополнительные источники,
               возвращает улучшенный draft.
        """
        ...
    
    def resume(self, job_id: str) -> EditorResult:
        """Возобновление работы из checkpoint'а."""
        state_path = self._jobs_dir / f"{job_id}.json"
        if not state_path.exists():
            raise ValueError(f"Job {job_id} not found")
        
        with open(state_path) as f:
            state = json.load(f)
        
        if state["phase"] == "done":
            return self._reconstruct_result(state)
        
        if state["phase"] == "proposing":
            # Продолжаем с Phase 2
            ...
        
        if state["phase"] == "analyzing":
            # Продолжаем с Phase 1
            ...
        
        raise ValueError(f"Cannot resume from phase: {state['phase']}")
    
    # ── Private methods ──
    
    def _build_analysis(self, loop_result: ToolUseResult) -> StorageAnalysis:
        """Строит StorageAnalysis из результата tool-use loop."""
        # Извлекаем данные из tool calls history
        cluster_data = None
        time_data = None
        search_data = []
        existing_data = []
        
        for tc in loop_result.tool_calls_made:
            if tc["name"] == "count_by_cluster":
                cluster_data = tc["result"]
            elif tc["name"] == "get_time_range":
                time_data = tc["result"]
            elif tc["name"] == "search_storage":
                search_data.append(tc["result"])
            elif tc["name"] == "check_existing_articles":
                existing_data = tc["result"]
        
        # Извлекаем gaps из текстового ответа LLM
        gaps = self._extract_gaps(loop_result.content)
        
        return StorageAnalysis(
            total_articles=time_data.get("total", 0) if time_data else 0,
            relevant_count=sum(sd.get("total_found", 0) for sd in search_data),
            clusters=cluster_data.get("clusters", []) if cluster_data else [],
            year_range=(time_data.get("min_year", 0), time_data.get("max_year", 0)) 
                      if time_data else (0, 0),
            by_year=time_data.get("by_year", {}) if time_data else {},
            source_distribution={},  # можно вычислить из search_data
            existing_articles=existing_data.get("articles", []) if existing_data else [],
            gaps=gaps,
            raw_llm_analysis=loop_result.content,
        )
    
    def _validate_proposal(self, raw: dict, analysis: StorageAnalysis, 
                          index: int) -> ArticleProposal:
        """Валидирует proposal: DOI check, duplicate check, scoring."""
        # 1. Validate DOIs
        valid_dois = []
        for ref in raw.get("key_references", []):
            doi = ref.replace("DOI:", "").strip()
            if self._check_doi_exists(doi):
                valid_dois.append(ref)
        
        # 2. Check for duplicates against existing
        is_duplicate = any(
            self._similarity(raw.get("title", ""), ex.get("file", "")) > 0.7
            for ex in analysis.existing_articles
        )
        
        # 3. Score confidence based on evidence
        base_confidence = raw.get("confidence", 0.5)
        if len(valid_dois) < 3:
            base_confidence *= 0.8  # Штраф за мало источников
        if is_duplicate:
            base_confidence *= 0.3  # Сильный штраф за дубликат
        
        return ArticleProposal(
            id=f"prop_{now_iso().replace('-','').replace(':','')}_{index}",
            title=raw.get("title", "[без заголовка]"),
            thesis=raw.get("thesis", "[без тезиса]"),
            target_audience=raw.get("target_audience", "general_public"),
            confidence=min(base_confidence, 1.0),
            sources_available=len(valid_dois),
            sources_needed=raw.get("sources_needed", 0),
            key_references=valid_dois,
            gap_filled=raw.get("gap_filled", ""),
            estimated_sections=raw.get("estimated_sections", []),
            status="duplicate" if is_duplicate else "proposed",
        )
    
    def _save_checkpoint(self, state: EditorState):
        path = self._jobs_dir / f"{state.job_id}.json"
        with open(path, 'w') as f:
            json.dump(asdict(state), f, ensure_ascii=False, indent=2, default=str)
```

## Acceptance Criteria S3

- [ ] `EditorAgent.run()` создаёт job_id
- [ ] `EditorAgent.run()` вызывает ToolUseLoop для анализа (Phase 1)
- [ ] `EditorAgent.run()` вызывает ToolUseLoop для предложений (Phase 2)
- [ ] `EditorAgent.run()` сохраняет checkpoint после каждой фазы
- [ ] `EditorAgent.run()` возвращает EditorResult с proposals
- [ ] `EditorAgent.run()` валидирует DOI в proposals
- [ ] `EditorAgent.run()` помечает дубликаты (status="duplicate")
- [ ] `EditorAgent.run()` снижает confidence при малом числе источников
- [ ] `EditorAgent.resume()` восстанавливает состояние из файла
- [ ] `EditorAgent.resume()` возвращает результат для completed job
- [ ] `EditorAgent._save_checkpoint()` создаёт читаемый JSON файл
- [ ] Checkpoint содержит все поля EditorState
- [ ] При ошибке checkpoint сохраняется с phase="failed"

## Тесты S3

### Unit тесты

**test_editor_agent.py**
```python
class TestEditorAgentRun:
    @fixture
    def editor(tmp_path):  # EditorAgent с mock storage и mock LLM
        ...
    
    def test_run_creates_job_id(editor):
        result = editor.run(topic="Arctic methane")
        assert result.job_id.startswith("20")  # timestamp-based
        assert len(result.job_id) > 10
    
    def test_run_returns_proposals(editor):
        result = editor.run(topic="test")
        assert len(result.proposals) >= 1
        assert all(hasattr(p, 'title') for p in result.proposals)
        assert all(hasattr(p, 'thesis') for p in result.proposals)
    
    def test_run_calls_tools(editor):
        """Убедимся что tools вызываются."""
        result = editor.run(topic="test")
        assert result.tool_rounds_total >= 2  # Phase 1 + Phase 2
    
    def test_run_saves_checkpoints(editor, tmp_path):
        editor.run(topic="test")
        files = list((tmp_path / "jobs").glob("*.json"))
        assert len(files) >= 1  # минимум один checkpoint
        with open(files[0]) as f:
            data = json.load(f)
        assert "job_id" in data
        assert "phase" in data
        assert data["phase"] == "done"
    
    def test_run_validates_dois(editor):
        """DOI которые не существуют в базе должны быть удалены."""
        # Настраиваем mock чтобы LLM вернул фейковый DOI
        result = editor.run(topic="test")
        for p in result.proposals:
            for ref in p.key_references:
                assert "fake" not in ref.lower()  # фейковые DOI отфильтрованы
    
    def test_run_detects_duplicates(editor):
        """Дубликаты существующих статей отмечаются."""
        result = editor.run(topic="test")
        dupes = [p for p in result.proposals if p.status == "duplicate"]
        # Если есть дубликаты — они должны быть отмечены
        for d in dupes:
            assert d.confidence < 0.5
    
    def test_run_confidence_scoring(editor):
        """Confidence корректно скорится."""
        result = editor.run(topic="test")
        for p in result.proposals:
            assert 0.0 <= p.confidence <= 1.0
    
    def test_error_saves_failed_checkpoint(editor):
        """При ошибке checkpoint сохраняется с failed."""
        editor.llm = FailingLLM()
        with pytest.raises(RuntimeError):
            editor.run(topic="test")
        files = list(Path(editor._jobs_dir).glob("*.json"))
        with open(files[-1]) as f:
            data = json.load(f)
        assert data["phase"] == "failed"
        assert "error" in data

class TestEditorResume:
    def test_resume_completed_job(editor):
        result = editor.run(topic="test")
        resumed = editor.resume(result.job_id)
        assert resumed.job_id == result.job_id
        assert len(resumed.proposals) == len(result.proposals)
    
    def test_resume_nonexistent_raises(editor):
        with pytest.raises(ValueError, match="not found"):
            editor.resume("nonexistent")

class TestBuildAnalysis:
    def test_build_from_tool_history():
        loop_result = ToolUseResult(
            content="Analysis complete",
            tool_calls_made=[
                {"name": "count_by_cluster", "result": {"clusters": [
                    {"theme": "methane", "count": 23}
                ]}},
                {"name": "get_time_range", "result": {
                    "min_year": 2019, "max_year": 2025, "total": 181,
                    "by_year": {2023: 42, 2024: 38}
                }},
            ],
            total_rounds=2, stop_reason="end_turn",
            usage={}, warnings=[]
        )
        analysis = EditorAgent._build_analysis(None, loop_result)
        assert analysis.total_articles == 181
        assert len(analysis.clusters) == 1
        assert analysis.clusters[0]["theme"] == "methane"

class TestValidateProposal:
    def test_valid_proposal_passes():
        p = validate_with_real_doi({"title": "Test", "thesis": "X", 
                                      "key_references": ["DOI:10.real/123"]})
        assert p.status != "duplicate"
        assert p.confidence > 0
    
    def test_fake_doi_removed():
        p = validate_with_fake_doi({"key_references": ["DOI:10.fake/xxx"]})
        assert len(p.key_references) == 0
        assert p.sources_available == 0
    
    def test_duplicate_penalty():
        p = validate_duplicate_title({"title": "Existing Article Title"})
        assert p.status == "duplicate"
        assert p.confidence < 0.5

### Интеграционный тест

**test_editor_integration.py**
```python
@mark.integration
class TestEditorRealWorld:
    """Полные тесты на реальных данных."""
    
    def test_full_editor_cycle():
        """Запуск Editor Agent на реальной базе."""
        storage = JsonlStorage(data_dir="/app/data")
        llm = MiniMaxProvider(api_key=REAL_KEY, model="MiniMax-M2.5")
        editor = EditorAgent(storage=storage, llm=llm)
        
        result = editor.run(
            topic="Arctic permafrost methane emissions",
            max_proposals=3,
        )
        
        assert result.status == "done"
        assert len(result.proposals) >= 1
        assert result.duration_sec < 300  # меньше 5 минут
        
        # Проверяем качество
        best = max(result.proposals, key=lambda p: p.confidence)
        assert best.confidence > 0.3
        assert len(best.thesis) > 20
        assert len(best.title) > 5
    
    def test_resume_after_crash():
        """Resume после того как job был создан."""
        storage = JsonlStorage(data_dir="/app/data")
        llm = MiniMaxProvider(api_key=REAL_KEY, model="MiniMax-M2.5")
        editor = EditorAgent(storage=storage, llm=llm)
        
        result = editor.run(topic="test resume", max_proposals=2)
        
        # Симулируем "новую сессию"
        editor2 = EditorAgent(storage=storage, llm=llm)
        resumed = editor2.resume(result.job_id)
        
        assert resumed.job_id == result.job_id
        assert len(resumed.proposals) == len(result.proposals)
    
    def test_large_topic():
        """Тест на широком домене (больше данных)."""
        storage = JsonlStorage(data_dir="/app/data")
        llm = MiniMaxProvider(api_key=REAL_KEY, model="MiniMax-M2.5")
        editor = EditorAgent(storage=storage, llm=llm)
        
        result = editor.run(
            domain="climate change Arctic region",
            max_proposals=5,
        )
        
        assert result.status in ("done", "partial")
        assert result.tool_rounds_total >= 2
```

## Файлы стадии S3

| Файл | Действие | Строк |
|------|----------|-------|
| `engine/agents/editor.py` | Создать | ~350 |
| `engine/schemas.py` | Изменить (+dataclasses) | +80 |
| `tests/test_editor_agent.py` | Создать | ~300 |
| `tests/test_editor_integration.py` | Создать | ~120 |

**Итого:** ~850 строк
