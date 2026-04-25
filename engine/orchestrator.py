"""Orchestrator -- State machine coordinating the agent pipeline."""
from __future__ import annotations
import json, uuid, time
from datetime import datetime
from pathlib import Path
from engine.config import EngineConfig, get_config
from engine.schemas import (
    JobStatus, JobState, GroupType, ScoutResult, GroupDraft,
    StructuredDraft, WrittenArticle, ReviewedDraft, AgentResult,
)
from engine.agents.base import BaseAgent
from engine.llm.base import LLMProvider
from engine.storage.jsonl_backend import JsonlStorage

# Lazy imports for agents (not all exist yet)
def _get_agent_class(name: str):
    """Lazy load agent class to avoid circular imports."""
    classes = {}
    try:
        from engine.agents.scout import ScoutAgent
        classes["scout"] = ScoutAgent
    except ImportError:
        pass
    try:
        from engine.agents.reader import ReaderAgent
        classes["reader"] = ReaderAgent
    except ImportError:
        pass
    try:
        from engine.agents.writer import WriterAgent
        classes["writer"] = WriterAgent
    except ImportError:
        pass
    try:
        from engine.agents.reviewer import ReviewerAgent
        classes["reviewer"] = ReviewerAgent
    except ImportError:
        pass
    return classes.get(name)


class Orchestrator:
    PAUSE_STATUSES = {JobStatus.SCOUT_DONE, JobStatus.READ_DONE, JobStatus.WRITE_DONE}

    def __init__(self, config: EngineConfig | None = None,
                 jobs_dir: str = "", output_dir: str = ""):
        self.config = config or get_config()
        self.jobs_dir = Path(jobs_dir or str(self.config.jobs_dir))
        self.output_dir = Path(output_dir or str(self.config.output_dir))
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._agents = {}

    def _get_agent(self, name: str) -> BaseAgent:
        if name not in self._agents:
            cls = _get_agent_class(name)
            if not cls:
                raise ValueError(f"Unknown agent: {name} (not implemented yet)")
            self._agents[name] = cls()
        return self._agents[name]

    def create_job(self, topic: str, pipeline: str = "full",
                   user_comment: str = "") -> JobState:
        job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        state = JobState(job_id=job_id, status=JobStatus.CREATED,
                         input_topic=topic, user_comment=user_comment,
                         pipeline=pipeline)
        self._save_state(state)
        return state

    def start_job(self, job_id: str) -> JobState:
        state = self.load_state(job_id)
        if state.is_running:
            raise RuntimeError(f"Job {job_id} already running (status={state.status.value})")
        state.status = JobStatus.SCOUTING
        self._save_state(state)
        self._run_scout_async(state)
        return state

    def approve_group(self, job_id: str, group_index: int, comment: str = "") -> JobState:
        state = self.load_state(job_id)
        if state.status != JobStatus.SCOUT_DONE:
            raise RuntimeError(f"Cannot approve: job is {state.status.value}, expected scout_done")
        state.selected_group_index = group_index
        state.add_approval("scout", "approve", f"group={group_index} comment={comment}")
        state.status = JobStatus.READING
        self._save_state(state)
        self._run_reader_async(state)
        return state

    def approve_draft(self, job_id: str, comment: str = "") -> JobState:
        state = self.load_state(job_id)
        if state.status != JobStatus.READ_DONE:
            raise RuntimeError(f"Cannot approve: job is {state.status.value}, expected read_done")
        state.add_approval("read", "approve", comment)
        state.status = JobStatus.WRITING
        self._save_state(state)
        self._run_writer_async(state)
        return state

    def request_revision(self, job_id: str, comment: str = "") -> JobState:
        state = self.load_state(job_id)
        if state.status != JobStatus.WRITE_DONE:
            raise RuntimeError(f"Cannot revise: job is {state.status.value}")
        state.add_approval("write", "revise", comment)
        state.status = JobStatus.WRITING
        self._save_state(state)
        self._run_writer_async(state)
        return state

    def skip_review(self, job_id: str) -> JobState:
        state = self.load_state(job_id)
        if state.status != JobStatus.WRITE_DONE:
            raise RuntimeError(f"Cannot skip review: job is {state.status.value}")
        state.status = JobStatus.COMPLETE
        self._save_state(state)
        return state

    def cancel_job(self, job_id: str) -> JobState:
        state = self.load_state(job_id)
        state.status = JobStatus.CANCELLED
        state.touch()
        self._save_state(state)
        return state

    def load_state(self, job_id: str) -> JobState:
        path = self._job_path(job_id)
        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        with open(path, encoding="utf-8") as f:
            return JobState.from_dict(json.load(f))

    def list_jobs(self) -> list:
        jobs = []
        for p in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            try:
                jobs.append(JobState.from_dict(json.loads(p.read_text())))
            except Exception:
                pass
        return jobs

    # ── Internal async runners ──

    def _run_scout_async(self, state: JobState):
        import threading
        def target():
            try:
                agent = self._get_agent("scout")
                result, elapsed = self._timeit(agent.run, topic=state.input_topic)
                state.set_result("scout", result.data)
                if result.success:
                    state.status = JobStatus.SCOUT_DONE
                else:
                    state.status = JobStatus.FAILED
                    state.error = result.error
                state.touch()
                self._save_state(state)
            except Exception as e:
                state.status = JobStatus.FAILED
                state.error = str(e)
                self._save_state(state)
        t = threading.Thread(target=target, daemon=True)
        t.start()

    def _run_reader_async(self, state: JobState):
        import threading
        def target():
            try:
                scout_result = state.get_result("scout", ScoutResult)
                groups = scout_result.groups if scout_result else []
                idx = state.selected_group_index
                selected = groups[idx] if 0 <= idx < len(groups) else (groups[0] if groups else None)
                agent = self._get_agent("reader")
                result, elapsed = self._timeit(agent.run, group=selected)
                state.set_result("read", result.data)
                if result.success:
                    state.status = JobStatus.READ_DONE
                else:
                    state.status = JobStatus.FAILED
                    state.error = result.error
                state.touch()
                self._save_state(state)
            except Exception as e:
                state.status = JobStatus.FAILED
                state.error = str(e)
                self._save_state(state)
        t = threading.Thread(target=target, daemon=True)
        t.start()

    def _run_writer_async(self, state: JobState):
        import threading
        def target():
            try:
                draft = state.get_result("read", GroupDraft)
                agent = self._get_agent("writer")
                result, elapsed = self._timeit(agent.run, draft=draft,
                                                user_comment=state.user_comment)
                state.set_result("write", result.data)
                if result.success:
                    state.status = JobStatus.WRITE_DONE
                else:
                    state.status = JobStatus.FAILED
                    state.error = result.error
                state.touch()
                self._save_state(state)
            except Exception as e:
                state.status = JobStatus.FAILED
                state.error = str(e)
                self._save_state(state)
        t = threading.Thread(target=target, daemon=True)
        t.start()

    def _timeit(self, fn, *args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        return result, time.time() - t0

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _save_state(self, state: JobState):
        path = self._job_path(state.job_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
