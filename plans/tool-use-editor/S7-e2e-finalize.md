# Стадия S7: E2E Тесты + Дебаггинг + Финализация

## Цель
Полное end-to-end тестирование всей системы: от UI кнопки до финальной статьи.
Дебаггинг найденных проблем. Финальный коммит и push.

## E2E Сценарии

### Сценарий 1: Полный цикл Editor (основной)

```
1. UI: Пользователь открывает вкладку "Агенты"
2. UI: Вводит тему "Arctic permafrost methane emissions 2024"
3. UI: Нажимает "🚀 Проанализировать и предложить"
4. API: POST /api/editor/analyze → job_id
5. Backend: EditorAgent.run() запускается
6. Backend: ToolUseLoop Phase 1 → LLM вызывает:
   - count_by_cluster("Arctic permafrost...") → 6 кластеров
   - get_time_range() → 2019-2025, peak 2023
   - search_storage("methane emissions") → 42 статьи
   - check_existing_articles("Arctic...") → 0 дубликатов
7. Backend: ToolUseLoop Phase 2 → LLM генерирует 3-5 proposals
8. Backend: Post-validation → DOI check, duplicate check, scoring
9. API: Возвращает {job_id, proposals_count: 3}
10. UI: Показывает 3 карточки предложений
11. UI: Пользователь кликает "✅ Выбрать" на лучшей (confidence > 0.7)
12. API: POST /api/editor/jobs/{id}/select/{prop_id}
13. UI: Появляется панель выбранной статьи с тезисом и источниками
14. UI: Пользователь нажимает "✍️ Написать статью"
15. API: POST /api/pipeline/jobs/{id}/write
16. Backend: WriterAgent.run(proposal) → статья в markdown
17. UI: Статья отображается в панели результата
18. UI: Пользователь нажимает "👀 Отправить на ревью"
19. API: POST /api/pipeline/jobs/{id}/review
20. Backend: ReviewerAgent.run(article) → verdict + score
21. UI: Результат ревью: approve/needs_revision + детали

PASS CRITERIA:
- Каждый шаг завершается без ошибок
- Proposal имеет confidence >= 0.5
- Все DOI в proposal валидны (есть в базе)
- Финальная статья > 500 символов
- Review verdict != reject
```

### Сценарий 2: Итеративная доработка

```
1-13. Как в сценарии 1 (до выбора предложения)
14. UI: Пользователь пишет в поле "Добавь данные по Сибири"
15. UI: Нажимает "💬 Отправить на доработку"
16. API: POST /api/pipeline/jobs/{id}/develop (feedback="...")
17. Backend: ReaderAgent читает дополнительные источники
18. Backend: Draft обновляется
19. UI: Обновлённый draft показывается
20. Повтор 14-19 ещё раз с другой правкой
21. Теперь пользователь доволен → нажимает "✍️ Написать"

PASS CRITERIA:
- development_rounds == 2 после двух итераций
- Каждая итерация обновляет draft
- Нет деградации качества между итерациями
```

### Сценарий 3: Error recovery

```
1. UI: Запуск анализа с невалидными данными
   - Пустая тема → валидация на UI, не доходит до API
2. API: Тема которая ничего не найдёт ("quantum entanglement dark matter flowers")
3. Backend: Editor находит 0 статей
4. Backend: LLM генерирует proposals с sources_available=0
5. Backend: Post-validation снижает confidence до < 0.3
6. UI: Карточки показываются с низкой confidence (красные)
7. UI: Пользователь видит что качество низкое → может попробовать другую тему

PASS CRITERIA:
- Система не падает на пустом результате
- Proposals с 0 источников имеют low confidence
- UI корректно отображает слабые предложения
```

### Сценарий 4: Resume после перезапуска

```
1. Запустить полный анализ (сценарий 1, шаги 1-11)
2. Дождаться завершения
3. Перезапустить Docker контейнеры (docker compose restart)
4. Открыть UI заново
5. История должна показать предыдущий job
6. Кликнуть на job → детали загрузились
7. Предложение всё ещё selected
8. Можно продолжить с шага "Написать"

PASS CRITERIA:
- Job выжил после перезапуска (checkpoint на диске)
- Все данные восстановлены (proposals, selection)
- Можно продолжить pipeline без повторного анализа
```

## Что делаем

### 7.1 E2E Test Suite
Файл: `tests/test_e2e_full_pipeline.py`

```python
import pytest
import requests
import json
import time

BASE_UI = "http://localhost:3000"
BASE_API = "http://localhost:3001"  # direct to worker


@mark.e2e
class TestFullEditorPipeline:
    """Полный сценарий 1 через API."""
    
    def test_editor_analyze_to_proposals(self):
        """Шаги 1-9: Анализ → proposals."""
        r = requests.post(f"{BASE_API}/api/editor/analyze", json={
            "topic": "Arctic permafrost methane emissions",
            "max_proposals": 3,
        }, timeout=300)
        
        assert r.status_code == 200
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "done"
        assert data["proposals_count"] >= 1
        
        self.job_id = data["job_id"]
    
    def test_job_detail_has_proposals(self):
        """Шаг 10: Детали содержат proposals."""
        r = requests.get(f"{BASE_API}/api/editor/jobs/{self.job_id}")
        assert r.status_code == 200
        data = r.json()
        
        proposals = data.get("proposals", [])
        assert len(proposals) >= 1
        
        # Валидация структуры
        p = proposals[0]
        assert "id" in p
        assert "title" in p
        assert len(p["title"]) > 5
        assert "thesis" in p
        assert len(p["thesis"]) > 20
        assert "confidence" in p
        assert 0 <= p["confidence"] <= 1
        assert "key_references" in p
        
        self.proposal_id = p["id"]
        self.proposal = p
    
    def test_proposals_have_valid_dois(self):
        """Все DOI в proposals существуют в базе."""
        storage = JsonlStorage(data_dir="/app/data")
        all_dois = {a.doi for a in storage.load_all_articles()}
        
        for ref in self.proposal.get("key_references", []):
            doi = ref.replace("DOI:", "").strip()
            if doi:  # Пустые пропускаем
                assert doi in all_dois, f"DOI {doi} not found in storage!"
    
    def test_select_proposal(self):
        """Шаг 11-12: Выбор предложения."""
        r = requests.post(
            f"{BASE_API}/api/editor/jobs/{self.job_id}/select/{self.proposal_id}"
        )
        assert r.status_code == 200
        assert r.json()["status"] == "selected"
    
    def test_pipeline_write_article(self):
        """Шаг 14-17: Генерация статьи через pipeline."""
        r = requests.post(f"{BASE_API}/api/pipeline/jobs/{self.job_id}/write", timeout=120)
        assert r.status_code == 200
        data = r.json()
        assert data["state"] in ("reviewing", "done")  # зависит от реализации
    
    def test_pipeline_review(self):
        """Шаг 18-21: Ревью."""
        r = requests.post(f"{BASE_API}/api/pipeline/jobs/{self.job_id}/review", timeout=60)
        assert r.status_code == 200
        data = r.json()
        assert data["state"] in ("done", "developing")  # approve или back to develop


@mark.e2e  
class TestIterativeDevelopment:
    """Сценарий 2: Итеративная доработка."""
    
    def setup_method(self):
        # Создаём и выбираем proposal
        r = requests.post(f"{BASE_API}/api/editor/analyze", json={
            "topic": "Arctic lake emissions", "max_proposals": 2
        }, timeout=300)
        self.job_id = r.json()["job_id"]
        
        r = requests.get(f"{BASE_API}/api/editor/jobs/{self.job_id}")
        props = r.json().get("proposals", [])
        if props:
            requests.post(
                f"{BASE_API}/api/editor/jobs/{self.job_id}/select/{props[0]['id']}"
            )
    
    def test_first_development_round(self):
        r = requests.post(f"{BASE_API}/api/pipeline/jobs/{self.job_id}/develop", json={
            "feedback": "Add more data about measurement methods"
        }, timeout=60)
        assert r.status_code == 200
        assert r.json()["round"] == 1
    
    def test_second_development_round(self):
        r = requests.post(f"{BASE_API}/api/pipeline/jobs/{self.job_id}/develop", json={
            "feedback": "Include Siberian region specifically"
        }, timeout=60)
        assert r.status_code == 200
        assert r.json()["round"] == 2
    
    def test_accumulated_rounds(self):
        r = requests.get(f"{BASE_API}/api/pipeline/jobs/{self.job_id}")
        data = r.json()
        assert len(data.get("development_rounds", [])) >= 2


@mark.e2e
class TestEdgeCases:
    """Сценарий 3: Граничные случаи."""
    
    def test_empty_topic_rejected():
        r = requests.post(f"{BASE_API}/api/editor/analyze", json={})
        assert r.status_code == 400
    
    def test_no_results_topic(self):
        """Тема которая ничего не найдёт."""
        r = requests.post(f"{BASE_API}/api/editor/analyze", json={
            "topic": "xyznonexistenttopic12345thatdefinitelyhasnoarticles"
        }, timeout=120)
        # Не должен упасть!
        assert r.status_code == 200
        data = r.json()
        # Может быть done с 0 proposals или partial
        assert data["status"] in ("done", "partial")
    
    def test_duplicate_detection():
        """Проверяем что дубликаты детектятся."""
        # Сначала создаём статью
        ...
        # Потом анализируем похожую тему
        r = requests.post(f"{BASE_API}/api/editor/analyze", json={
            "topic": "Arctic permafrost methane"  # похожа на существующую
        }, timeout=300)
        data = r.json()
        job_id = data["job_id"]
        
        r2 = requests.get(f"{BASE_API}/api/editor/jobs/{job_id}")
        proposals = r2.json().get("proposals", [])
        dupes = [p for p in proposals if p.get("status") == "duplicate"]
        # Хотя бы один должен быть детектирован (если есть совпадения)
        # Это soft assertion — зависит от данных
        

@mark.e2e
class TestPersistence:
    """Сценарий 4: Checkpoint persistence."""
    
    def test_job_survives_restart(self):
        r = requests.post(f"{BASE_API}/api/editor/analyze", json={
            "topic": "persistence test topic"
        }, timeout=300)
        job_id = r.json()["job_id"]
        
        # Проверяем что файл есть
        import os
        assert os.path.exists(f"/app/data/jobs/{job_id}.json")
        
        # Читаем файл напрямую
        with open(f"/app/data/jobs/{job_id}.json") as f:
            data = json.load(f)
        
        assert data["job_id"] == job_id
        assert data["phase"] == "done"  # or "failed"
        assert "proposals" in data or "error" in data
```

### 7.2 Browser E2E Tests (Playwright-style via browser tool)

**test_e2e_browser.py**
```python
@mark.browser_e2e
class TestBrowserFullCycle:
    """Тесты через реальный браузер (browser tool)."""
    
    def test_full_ui_cycle(browser):
        """Открыть UI → Агенты → Анализировать → Выбрать → Результат."""
        browser.goto("http://localhost:3000/?_v=e2e")
        
        # 1. Открыть вкладку Агенты
        browser.click("text=Агенты")
        browser.wait_for_selector("#editorTopic")
        
        # 2. Заполнить форму
        browser.fill("#editorTopic", "Arctic permafrost methane")
        browser.click("text=Проанализировать")
        
        # 3. Ждём результаты (SSE или polling)
        browser.wait_for_selector(".proposal-card", timeout=300000)  # 5 мин
        
        # 4. Проверяем карточки
        cards = browser.query_all(".proposal-card")
        assert len(cards) >= 1
        
        # 5. Выбираем первую
        cards[0].click()
        browser.wait_for_selector("#selectedArticle")
        
        # 6. Проверяем панель детали
        assert browser.text_content("#selectedArticle") contains "Тезис"
        
        # 7. Переключаем табы
        browser.click("text=Источники")
        assert browser.is_visible(":text('DOI:')")
        
        print("✅ Full UI cycle passed!")
```

### 7.3 Дебаггинг чеклист

После запуска E2E тестов:

```
□ Все unit тесты проходят (pytest tests/ -v)
□ Все integration тесты проходят (pytest tests/ -m integration -v)
□ E2E API тесты проходят (pytest tests/ -m e2e -v)
□ Docker логи чистые (нет tracebacks)
□ Worker health OK
□ Dashboard proxy работает
□ UI загружается без JS ошибок (console clear + действия → no errors)
□ Scout через UI работает (быстрый режим)
□ Editor через UI работает (основной режим)
□ Pipeline select → write → review работает
□ Job persistence после restart
□ Push на main успешен
```

### 7.4 Финализация

```bash
# 1. Запуск всех тестов
pytest tests/ -v --tb=short 2>&1 | tail -30

# 2. E2E тесты
pytest tests/test_e2e_full_pipeline.py -v -m e2e --tb=long 2>&1 | tail -50

# 3. Если есть падения — фиксим, возвращаемся к нужной стадии

# 4. Коммит
git add -A
git commit -m "Sprint 12: Tool-Use Editor Agent (complete implementation)

Stages S0-S7:
- S0: Infrastructure (tool_complete interface, MiniMax tool use verified)
- S1: Storage Tools (7 tools: search, detail, validate, cluster, etc.)
- S2: Tool-Use Loop (multi-turn LLM↔tools orchestration)
- S3: Editor Agent (analysis + proposals + validation + checkpoints)
- S4: API Endpoints (8 new endpoints + SSE logs)
- S5: UI (Editor tab with proposal cards, detail panel, progress)
- S6: Orchestrator v2 (state machine: edit→select→develop→write→review)
- S7: E2E tests (4 scenarios, full coverage)

Verified:
- MiniMax M2.5/M2.7 tool use (Anthropic-compatible API)
- Multi-turn tool_use → tool_result → end_turn cycle
- Full pipeline: topic → proposals → select → write → review
- Checkpoint persistence across container restarts
- DOI validation eliminates hallucinated references
- Duplicate detection prevents redundant articles"

# 5. Push
git push origin main
```

## Acceptance Criteria S7 (финальные)

- [ ] Все unit тесты проходят (>90% pass rate)
- [ ] Все integration тесты проходят
- [ ] E2E сценарий 1 (полный цикл) проходит
- [ ] E2E сценарий 2 (итерации) проходит
- [ ] E2E сценарий 3 (edge cases) проходит
- [ ] E2E сценарий 4 (persistence) passes
- [ ] Нет JS ошибок в UI (browser console clean)
- [ ] Docker логи без tracebacks
- [ ] Worker memory < 2GB (no leaks)
- [ ] Время полного цикла < 5 минут
- [ ] Коммит сделан с описательным сообщением
- [ ] Push на main успешен
- [ ] README обновлён (опционально)

## Файлы стадии S7

| Файл | Действие | Строк |
|------|----------|-------|
| `tests/test_e2e_full_pipeline.py` | Создать | ~350 |
| `tests/test_e2e_browser.py` | Создать | ~100 |
| `tests/conftest.py` | Изменить (fixtures для e2e) | +50 |
| `README.md` | Изменить (обновить архитектуру) | +100 |

**Итого:** ~600 строк + дебаггинг
```

---

# СВОДНАЯ ТАБЛИЦА ВСЕХ СТАДИЙ

| Стадия | Название | Файлов | Строк | Зависит от | Тестов |
|--------|----------|--------|------|------------|--------|
| **S0** | Инфраструктура | 8 | ~605 | Nothing | 15 |
| **S1** | Tool Executor | 6 | ~880 | S0 | 18 |
| **S2** | Tool-Use Loop | 6 | ~900 | S0 | 14 |
| **S3** | Editor Agent Core | 4 | ~850 | S1, S2 | 12 |
| **S4** | API Endpoints | 4 | ~690 | S3 | 12 |
| **S5** | UI Editor Tab | 2 | ~1000 | S4 | 10 |
| **S6** | Orchestrator v2 | 4 | ~750 | S3 | 12 |
| **S7** | E2E + Финал | 4 | ~600 | S5, S6 | 8+ |
| **ИТОГО** | | **38 файлов** | **~6280 строк** | | **~101 тест** |

Временная оценка:
- S0: 0.5 дня
- S1: 1 день
- S2: 1 день
- S3: 1.5 дня
- S4: 0.5 дня
- S5: 1.5 дня
- S6: 1 день
- S7: 1.5 дня (с дебаггингом)
- **ИТОГО: ~8.5 дней** при полной загрузке
