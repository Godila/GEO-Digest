# Стадия S5: UI — Вкладка "Редактор"

## Цель
Перестроить вкладку "📅 Агенты" в dashboard:
- Убрать старый "Полный пайплайн" (linear orchestrator)
- Оставить "Быстрый Scout" как вспомогательный инструмент
- Добавить "📝 Редактор" — основной режим работы

## UI Layout

```
┌─ 📅 АГЕНТЫ (переработанная) ────────────────────────────────┐
│                                                              │
│  ══ 📝 РЕДАКТОР (основной режим) ═══════════════════════    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Тема/домен: [Arctic permafrost methane_________]     │   │
│  │ Инструкция (опц.): [Focus on 2024-2025 studies__]   │   │
│  │ Кол-во предложений: [5]                               │   │
│  │                                                      │   │
│  │        [🚀 Проанализировать и предложить]            │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ══ Прогресс ═══════════════════════════════════════════    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░ 45%  Фаза: Генерация...       │   │
│  │ 🔍 Tool calls: count_by_cluster ✓ get_time_range ✓  │   │
│  │    search_storage(methane) ✓ validate_doi(3) ✓      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ══ Предложения статей ═════════════════════════════════    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ┌─ Card 1 ──────────────────────────────────────┐   │   │
│  │ │ ★★★★★ 0.87                                   │   │   │
│  │ │ Метан из вечной мерзлоты: новые данные о       │   │   │
│  │ │ внезапных выбросах (2023-2025)                │   │   │
│  │ │                                                │   │   │
│  │ │ За период 2023-2025 опубликовано 12 новых...   │   │   │
│  │ │ Источников: 15 доступно, 3 нужно              │   │   │
│  │ │ Аудитория: researchers + policy makers         │   │   │
│  │ │                                                │   │   │
│  │ │ [✅ Выбрать]  [📋 Детали]                      │   │   │
│  │ └────────────────────────────────────────────────┘   │   │
│  │                                                      │   │
│  │ ┌─ Card 2 ──────────────────────────────────────┐   │   │
│  │ │ ★★★★☆ 0.72                                    │   │   │
│  │ │ Внезапные выбросы метана при abrupt thaw       │   │   │
│  │ │ [...]                                          │   │   │
│  │ │ [✅ Выбрать]  [📋 Детали]                      │   │   │
│  │ └────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ══ Выбранная статья (появляется после выбора) ══════════    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 📄 "Метан из вечной мерзлоты..."                       │   │
│  │ Статус: selected | Confidence: 0.87                   │   │
│  │                                                        │   │
│  │ [Тезис] [Источники] [Доработка] [Написать+Ревью]     │   │
│  │                                                        │   │
│  │ ── Тезис ──────────────────────────────────────       │   │
│  │ За период 2023-2025 опубликовано 12 новых исследований │   │
│  │ показывающих что abrupt thaw события увеличивают...    │   │
│  │                                                        │   │
│  │ ── Ключевые источники (15) ────────────────────       │   │
│  │ • 10.1038/s41558-024-02000-1 Dean et al. 2023          │   │
│  │ • 10.1126/science-abj3440 Nisbet et al. 2021           │   │
│  │ • ...                                                   │   │
│  │                                                        │   │
│  │ ── Доработка ────────────────────────────────         │   │
│  │ Ваш комментарий: [Добавь данные по Сибири_______]      │   │
│  │ [💬 Отправить на доработку]                             │   │
│  │                                                        │   │
│  │ [✍️ Написать статью]  [👀 Отправить на ревью]          │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ══ 🔍 Быстрый Scout (свернутый) ══════════════════════    │
│  ▸ Разовый поиск статей по теме → результаты               │
│  [Развернуть ▾]                                            │
│                                                              │
│  ══ История редактора ═════════════════════════════════    │
│  Job ID              | Тема                    | Статус     │
│  20260426_143052_a1b | Arctic methane...       | done ✓     │
│  20260426_130919_d9d | test pipeline            | failed ✗   │
└──────────────────────────────────────────────────────────────┘
```

## Что делаем

### 5.1 HTML структура
Файл: `dashboard/templates/index.html` (изменить секцию agents tab)

**Удаляем:** Старую секцию `pipelineSection` (оркестратор linear)

**Изменяем:** Scout секция — делаем сворачиваемой (collapsible)

**Добавляем:** Editor секция с формой + карточками + деталями

### 5.2 CSS стили
Новые классы для editor UI:

```css
/* Editor form */
.editor-form { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
.editor-form input { flex: 1; min-width: 200px; }
.editor-form .editor-instr { width: 100%; }

/* Proposal cards */
.proposal-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; margin-top: 16px; }
.proposal-card {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; transition: border-color 0.2s;
}
.proposal-card:hover { border-color: var(--accent); }
.proposal-card.selected { border-color: #22c55e; background: rgba(34,197,94,0.05); }
.proposal-card.duplicate { opacity: 0.6; border-style: dashed; }

.proposal-confidence {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 13px; padding: 3px 10px; border-radius: 20px;
}
.conf-high { background: rgba(34,197,94,0.15); color: #22c55e; }
.conf-mid { background: rgba(234,179,8,0.15); color: #eab308; }
.conf-low { background: rgba(239,68,68,0.15); color: #ef4444; }

.proposal-title { font-size: 16px; font-weight: 600; margin-bottom: 8px; }
.proposal-thesis { font-size: 13px; color: var(--text-secondary); line-height: 1.5; margin-bottom: 12px; }
.proposal-meta { display: flex; gap: 16px; font-size: 12px; color: var(--text-muted); margin-bottom: 16px; }
.proposal-actions { display: flex; gap: 8px; }

/* Selected article detail */
.selected-article { 
    background: var(--bg-secondary); border-radius: 12px; 
    padding: 24px; margin-top: 20px; border: 1px solid var(--accent);
}
.detail-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border); margin-bottom: 16px; }
.detail-tab { padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent; }
.detail-tab.active { border-bottom-color: var(--accent); color: var(--accent); }

/* Progress with tool calls */
.tool-call-log {
    font-family: monospace; font-size: 12px; background: var(--bg-primary);
    border-radius: 8px; padding: 12px; max-height: 200px; overflow-y: auto;
}
.tool-call-item { display: flex; gap: 8px; align-items: center; padding: 4px 0; }
.tool-call-name { color: var(--accent); font-weight: 500; }
```

### 5.3 JavaScript функции

```javascript
// ── Editor: Main Functions ──

async function runEditor() {
    const topic = document.getElementById('editorTopic').value.trim();
    if (!topic) { alert('Введите тему или домен'); return; }
    
    const instruction = document.getElementById('editorInstruction').value.trim();
    const maxProps = parseInt(document.getElementById('editorMaxProposals').value) || 5;
    
    // Show progress
    showEditorProgress(true);
    setEditorStatus('analyzing', 'Запуск анализа...');
    
    try {
        const result = await api('/editor/analyze', 'POST', {
            topic,
            user_instruction: instruction || undefined,
            max_proposals: maxProps
        });
        
        currentEditorJobId = result.job_id;
        setEditorStatus('done', `Анализ завершён: ${result.propposals_count} предложений`);
        
        // Load proposals
        await loadEditorJobDetail(result.job_id);
        
    } catch(e) {
        setEditorStatus('error', 'Ошибка: ' + e.message);
    }
}

async function loadEditorJobDetail(jobId) {
    const job = await api(`/editor/jobs/${jobId}`);
    renderProposals(job.proposals || []);
    
    if (job.selected_proposal_id) {
        selectProposal(jobId, job.selected_proposal_id);
    }
}

function renderProposals(proposals) {
    const container = document.getElementById('proposalGrid');
    if (!proposals.length) {
        container.innerHTML = '<div class="empty-state-sm">Нет предложений</div>';
        return;
    }
    
    container.innerHTML = proposals.map((p, i) => `
        <div class="proposal-card ${p.status}" id="prop-${p.id}" onclick="selectProposal('${currentEditorJobId}', '${p.id}')">
            <div class="proposal-confidence ${confidenceClass(p.confidence)}">
                ★ ${p.confidence.toFixed(2)}
            </div>
            <div class="proposal-title">${escapeHtml(p.title)}</div>
            <div class="proposal-thesis">${escapeHtml(p.thesis)}</div>
            <div class="proposal-meta">
                <span>📚 ${p.sources_available} ист.</span>
                <span>🎯 ${p.target_audience}</span>
                ${p.status === 'duplicate' ? '<span style="color:#ef4444">⚠️ дубликат</span>' : ''}
            </div>
            <div class="proposal-actions">
                <button class="btn-sm btn-primary" onclick="event.stopPropagation();selectProposal('${currentEditorJobId}','${p.id}')">
                    ✅ Выбрать
                </button>
                <button class="btn-sm btn-secondary" onclick="event.stopPropagation();showProposalDetail('${p.id}')">
                    📋 Детали
                </button>
            </div>
        </div>
    `).join('');
}

async function selectProposal(jobId, propId) {
    await api(`/editor/jobs/${jobId}/select/${propId}`, 'POST');
    
    // Visual selection
    document.querySelectorAll('.proposal-card').forEach(c => c.classList.remove('selected'));
    const card = document.getElementById(`prop-${propId}`);
    if (card) card.classList.add('selected');
    
    // Load full detail and show panel
    const job = await api(`/editor/jobs/${jobId}`);
    const proposal = job.proposals.find(p => p.id === propId);
    if (proposal) showSelectedArticle(proposal);
}

function showSelectedArticle(proposal) {
    const panel = document.getElementById('selectedArticle');
    panel.style.display = 'block';
    panel.innerHTML = `
        <h3>📄 ${escapeHtml(proposal.title)}</h3>
        <div class="detail-meta">
            Confidence: ${proposal.confidence} | 
            Источников: ${proposal.sources_available} |
            Status: ${proposal.status}
        </div>
        <div class="detail-tabs">
            <span class="detail-tab active" onclick="switchDetailTab(this,'thesis')">Тезис</span>
            <span class="detail-tab" onclick="switchDetailTab(this,'sources')">Источники (${proposal.key_references.length})</span>
            <span class="detail-tab" onclick="switchDetailTab(this,'develop')">Доработка</span>
        </div>
        <div id="detailContent">
            <p>${escapeHtml(proposal.thesis)}</p>
        </div>
        <div class="detail-actions" style="margin-top:20px;display:flex;gap:12px">
            <button class="btn-primary" onclick="finalizeArticle('${proposal.id}')">✍️ Написать</button>
            <button class="btn-secondary" onclick="reviewArticle('${proposal.id}')">👀 Ревью</button>
        </div>
    `;
    panel.scrollIntoView({behavior: 'smooth'});
}

// ── Helper functions ──
function confidenceClass(c) { return c >= 0.7 ? 'conf-high' : c >= 0.4 ? 'conf-mid' : 'conf-low'; }
function showEditorProgress(show) { /* toggle visibility */ }
function setEditorPhase(phase, text) { /* update status bar */ }
function escapeHtml(s) { /* XSS protection */ }
```

## Acceptance Criteria S5

- [ ] Форма редактора видна во вкладке Агенты
- [ ] Кнопка "Проанализировать" вызывает `/api/editor/analyze`
- [ ] Прогресс-бар показывает фазу работы
- [ ] Tool calls логируются в реальном времени (SSE или polling)
- [ ] Предложения отображаются карточками в grid
- [ ] Карточки показывают: title, thesis, confidence, sources count, audience
- [ ] Карточки дубликатов визуально отличаются (opacity + dashed border)
- [ ] Клик по карточке выбирает предложение (вызов API)
- [ ] Выбранная карточка подсвечивается зелёным
- [ ] Панель выбранной статьи появляется после выбора
- [ ] Панель имеет табы: Тезис / Источники / Доработка
- [ ] Кнопка "Написать" вызывает Writer
- [ ] Кнопка "Ревью" вызывает Reviewer
- [ ] Быстрый Scout сворачиваем/разворачивается
- [ ] История редактора загружается при открытии вкладки
- [ ] Все тексты проходят escapeHtml (XSS защита)
- [ ] UI responsive (работает на мобильных)

## Тесты S5

### Browser/E2E тесты

**test_editor_ui.py**
```python
class TestEditorUI:
    @fixture
    def browser(playwright):  # Playwright browser instance
        ...
    
    def test_editor_form_visible(browser):
        page = browser.goto("http://localhost:3000")
        page.click("text=Агенты")
        assert page.is_visible("#editorTopic")
        assert page.is_visible("#editorAnalyzeBtn")
    
    def test_analyze_button_calls_api(browser, mock_api):
        page.goto("http://localhost:3000")
        page.click("text=Агенты")
        page.fill("#editorTopic", "test topic")
        page.click("text=Проанализировать")
        
        # Проверяем что API вызван
        assert mock_api.last_call["path"] == "/api/editor/analyze"
        assert mock_api.last_call["body"]["topic"] == "test topic"
    
    def test_proposals_rendered(browser):
        """После успешного анализа показываются карточки."""
        mock_api.respond("/api/editor/analyze", {
            "job_id": "test_123", "status": "done", "proposals_count": 3
        })
        mock_api.respond("/api/editor/jobs/test_123", {
            "proposals": [
                {"id": "p1", "title": "Test Article", "thesis": "Test",
                 "confidence": 0.85, "sources_available": 10, "target_audience": "researchers", "status": "proposed"},
                {"id": "p2", "title": "Dupe", "thesis": "X",
                 "confidence": 0.2, "sources_available": 1, "status": "duplicate"},
            ]
        })
        
        page.fill("#editorTopic", "test")
        page.click("text=Проанализировать")
        page.wait_for_selector(".proposal-card")
        
        cards = page.query_all(".proposal-card")
        assert len(cards) == 2
        assert page.text_content("#prop-p1") contains "Test Article"
        assert "#prop-p2" has class "duplicate"
    
    def test_select_proposal(browser):
        """Клик по карточке выбирает предложение."""
        # ... setup ...
        page.click("#prop-p1")
        
        assert "#prop-p1" has class "selected"
        assert page.is_visible("#selectedArticle")
        assert page.text_content("#selectedArticle") contains "Test Article"
    
    def test_detail_tabs(browser):
        """Табы в панели детали переключаются."""
        # ... setup: select proposal ...
        page.click("text=Источники")
        assert page.is_visible(":text('DOI:')")
        
        page.click("text=Доработка")
        assert page.is_visible("#developComment")
    
    def test_scout_collapsible(browser):
        """Быстрый Scout можно свернуть/развернуть."""
        page.goto("http://localhost:3000/?tab=agents")
        
        scout_section = page.query("#scoutSection")
        assert scout_section is visible
        
        page.click("text=Развернуть")  # or collapse toggle
        # Verify collapsed state
    
    def test_history_loads_on_tab_open(browser):
        """История загружается при открытии вкладки."""
        mock_api.respond("/api/editor/jobs", {
            "jobs": [{"job_id": "old_1", "topic": "old topic", "phase": "done"}]
        })
        
        page.goto("http://localhost:3000")
        page.click("text=Агенты")
        
        assert page.text_contains("old topic")

class TestEditorUIResponsive:
    def test_mobile_layout(browser_mobile):
        """На мобильном карточки в 1 колонку."""
        browser_mobile.set_viewport(375, 667)
        browser_mobile.goto("http://localhost:3000/?tab=agents")
        # ... setup proposals ...
        
        grid = browser_mobile.query(".proposal-grid")
        style = grid.get_computed_style()
        assert style["grid-template-columns"] == "1fr"  # single column
```

## Файлы стадии S5

| Файл | Действие | Строк |
|------|----------|-------|
| `dashboard/templates/index.html` | Изменить (agents tab) | +650 |
| `tests/test_editor_ui.py` | Создать | ~350 |

**Итого:** ~1000 строк
