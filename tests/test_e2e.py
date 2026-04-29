"""E2E test for GEO-Digest Agent Engine. (Sprint 8)"""

import sys
import os

# Add project root to path so 'engine' package is importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_imports():
    """Test all public imports."""
    separator("1. IMPORTS")

    from engine import (
        EngineConfig, get_config,
        Article, ArticleGroup,
        StructuredDraft, GroupDraft,
        ReviewedDraft, WrittenArticle,
        AgentResult, JobState,
        DataRequirements, InfrastructureNeeds,
        GroupType, Severity, ReviewVerdict,
        ScoutResult,
        BaseAgent, LLMProvider, StorageBackend,
        Edit, FactCheck,
    )
    print("  [OK] All public imports resolved")
    return True


def test_config():
    """Test config loading."""
    separator("2. CONFIG")

    from engine.config import EngineConfig, get_config, LLMConfig, ReviewerConfig

    cfg = EngineConfig.load()
    assert isinstance(cfg.llm, LLMConfig), "llm config"
    assert isinstance(cfg.reviewer, ReviewerConfig), "reviewer config"
    assert cfg.data_dir.exists(), f"data_dir: {cfg.data_dir}"
    assert cfg.jobs_dir.exists(), f"jobs_dir: {cfg.jobs_dir}"
    print(f"  [OK] Config: {cfg}")

    singleton = get_config()
    # Note: load() creates new instance, get_config() caches
    # They may differ if load() was called first without setting singleton
    assert isinstance(singleton, EngineConfig), "singleton is EngineConfig"
    print(f"  [OK] Singleton works")
    return True


def test_schemas():
    """Test all schemas create and serialize."""
    separator("3. SCHEMAS")

    from engine.schemas import (
        Article, ArticleGroup, StructuredDraft, GroupDraft,
        WrittenArticle, ReviewedDraft, Edit, FactCheck,
        ScoutResult, AgentResult, JobState, JobStatus,
        GroupType, Severity, ReviewVerdict,
        DataRequirements, InfrastructureNeeds,
    )

    # Article
    a = Article(data={"doi": "10.1234/test", "title": "Test Paper", "authors": "Smith J"})
    assert a.doi == "10.1234/test"
    d = a.to_dict()
    assert "doi" in d
    print(f"  [OK] Article: {a.title}")

    # StructuredDraft with all group types
    for gt in GroupType:
        draft = StructuredDraft(
            draft_id=f"d_{gt.value}",
            group_type=gt,
            confidence=0.8,
            gap_identified="test gap",
        )
        assert draft.group_type == gt
        dd = draft.to_dict()
        assert dd["group_type"] == gt.value
    print(f"  [OK] StructuredDraft: all {len(GroupType)} types")

    # WrittenArticle
    wa = WrittenArticle(text="# Test\nBody", title="T", format_="markdown",
                        language="ru", word_count=3)
    wd = wa.to_dict()
    assert wd["word_count"] == 3
    print(f"  [OK] WrittenArticle: {wa.word_count} words")

    # ReviewedDraft
    rd = ReviewedDraft(original_text="test", verdict=ReviewVerdict.ACCEPT_WITH_MINOR,
                       overall_score=0.9, summary="Good")
    assert rd.critical_issues == 0
    print(f"  [OK] ReviewedDraft: {rd.verdict.value} score={rd.overall_score}")

    # Edit + FactCheck
    edit = Edit(location="sec1", severity=Severity.MAJOR, original="bad",
                suggested="good", reason="style", category="style")
    fc = FactCheck(claim="X > Y", source_doi="10.x/y", verified=True)
    print(f"  [OK] Edit + FactCheck")

    # ScoutResult
    sr = ScoutResult(groups=[], total_found=5, topic="test")
    sd = sr.to_dict()
    assert sd["total_found"] == 5
    print(f"  [OK] ScoutResult")

    # AgentResult
    ar = AgentResult(agent_name="test", success=True, data=sr)
    ard = ar.to_dict()
    assert ard["success"] is True
    print(f"  [OK] AgentResult")

    # JobState
    js = JobState(job_id="test_001", status=JobStatus.CREATED, input_topic="ML rocks")
    jsd = js.to_dict()
    assert jsd["job_id"] == "test_001"
    assert js.is_running is False
    assert js.is_terminal is False
    print(f"  [OK] JobState: {js.job_id}")

    # DataRequirements + InfrastructureNeeds
    dr = DataRequirements(input_data="seismic data", data_format="csv")
    infra = InfrastructureNeeds(hardware="GPU", software=["python"])
    print(f"  [OK] DataRequirements + InfrastructureNeeds")

    return True


def test_agents():
    """Test all agents instantiate and validate."""
    separator("4. AGENTS")

    from engine.agents.scout import ScoutAgent
    from engine.agents.reader import ReaderAgent
    from engine.agents.writer import WriterAgent
    from engine.agents.reviewer import ReviewerAgent
    from engine.schemas import WrittenArticle

    agents = [
        ("scout", ScoutAgent),
        ("reader", ReaderAgent),
        ("writer", WriterAgent),
        ("reviewer", ReviewerAgent),
    ]

    for name, cls in agents:
        agent = cls()
        assert agent.name == name, f"{cls.__name__}.name != '{name}'"
        print(f"  [OK] {name}: {agent.name}")

    # Test validate_input for each
    scout = ScoutAgent()
    ok, msg = scout.validate_input(topic="")
    assert not ok, "empty topic should fail"
    ok2, msg2 = scout.validate_input(topic="valid topic")
    assert ok2, "valid topic should pass"
    print(f"  [OK] Scout validate_input")

    reader = ReaderAgent()
    rok, _ = reader.validate_input()
    assert not rok, "no input should fail"
    rok2, _ = reader.validate_input(dois=["10.x/y"])
    assert rok2, "dois should pass"
    print(f"  [OK] Reader validate_input")

    writer = WriterAgent()
    wok, _ = writer.validate_input()
    assert not wok, "no draft should fail"

    reviewer = ReviewerAgent()
    rvk, _ = reviewer.validate_input(strictness=99)
    assert not rvk, "strictness>5 should fail"
    rvk2, _ = reviewer.validate_input(article=WrittenArticle(text="t", title="t", format_="m", language="ru", word_count=1), strictness=3)
    assert rvk2, "strictness=3 with article should pass"
    print(f"  [OK] Writer/Reviewer validate_input")

    # Test estimate_cost
    sc = scout.estimate_cost(topic="test", max_articles=10)
    assert sc["estimated_tokens"] > 0
    rc = reader.estimate_cost(num_articles=3)
    assert rc["estimated_tokens"] > 0
    wc = writer.estimate_cost(num_sources=5)
    assert wc["estimated_tokens"] > 0
    rvc = reviewer.estimate_cost(word_count=2000)
    assert rvc["estimated_tokens"] > 0
    print(f"  [OK] All estimate_cost() return values")

    return True


def test_api():
    """Test API app creation and routes."""
    separator("5. API")

    from engine.api import app, JobResponse, CreateJobRequest, ApproveScoutRequest

    assert app.title == "GEO-Digest Agent API"
    assert len(app.routes) >= 14, f"expected 14+ routes, got {len(app.routes)}"

    route_paths = sorted([r.path for r in app.routes if hasattr(r, 'path')])
    expected = [
        "/health",
        "/api/v1/jobs",
        "/api/v1/jobs/{job_id}",
        "/api/v1/jobs/{job_id}/approve",
        "/api/v1/jobs/{job_id}/approve-draft",
        "/api/v1/jobs/{job_id}/revise",
        "/api/v1/jobs/{job_id}/skip-review",
        "/api/v1/jobs/{job_id}/start",
    ]
    for ep in expected:
        assert any(ep.replace("{job_id}", "") in r or ep in r for r in route_paths), \
            f"missing endpoint: {ep}"
    print(f"  [OK] API: {len(app.routes)} routes registered")

    # Test request models
    req = CreateJobRequest(topic="test topic", pipeline="full")
    assert req.topic == "test topic"
    apr = ApproveScoutRequest(group_index=0, comment="ok")
    assert apr.group_index == 0
    print(f"  [OK] Pydantic models valid")

    return True


def test_cli():
    """Test CLI argument parsing."""
    separator("6. CLI")

    from engine.cli import main

    # Test help (returns 0)
    assert main(["--help"]) == 0
    print(f"  [OK] --help works")

    # Test subcommand helps
    assert main(["scout", "--help"]) == 0
    assert main(["serve", "--help"]) == 0
    assert main(["jobs", "--help"]) == 0
    assert main(["job", "--help"]) == 0
    print(f"  [OK] Subcommand --help works")

    # Test invalid (no command) returns 0 (prints help)
    assert main([]) == 0
    print(f"  [OK] No-command handling")

    return True


def test_orchestrator():
    """Test orchestrator job lifecycle."""
    separator("7. ORCHESTRATOR")

    from engine.orchestrator import Orchestrator
    from engine.schemas import JobStatus

    orc = Orchestrator()

    # Create job
    job = orc.create_job(topic="E2E test topic")
    assert job.status == JobStatus.CREATED
    job_id = job.job_id
    print(f"  [OK] Created: {job_id}")

    # Load it back
    loaded = orc.load_state(job_id)
    assert loaded.job_id == job_id
    assert loaded.input_topic == "E2E test topic"
    print(f"  [OK] Loaded back")

    # List jobs
    jobs = orc.list_jobs()
    assert len(jobs) >= 1
    print(f"  [OK] Jobs list: {len(jobs)} jobs")

    # Cancel
    cancelled = orc.cancel_job(job_id)
    assert cancelled.status == JobStatus.CANCELLED
    print(f"  [OK] Cancelled")

    return True


def run_all():
    """Run all E2E tests."""
    print("=" * 60)
    print("  GEO-Digest Agent Engine — E2E Test Suite")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Schemas", test_schemas),
        ("Agents", test_agents),
        ("API", test_api),
        ("CLI", test_cli),
        ("Orchestrator", test_orchestrator),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, fn in tests:
        try:
            if fn():
                passed += 1
            else:
                failed += 1
                errors.append(name)
        except Exception as e:
            failed += 1
            errors.append(f"{name}: {e}")
            import traceback
            traceback.print_exc()

    separator("RESULTS")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    if errors:
        print(f"  Errors:")
        for e in errors:
            print(f"    - {e}")
    print()

    if failed == 0:
        print("  *** ALL TESTS PASSED ***")
        return 0
    else:
        print(f"  *** {failed} TEST(S) FAILED ***")
        return 1


if __name__ == "__main__":
    sys.exit(run_all())
