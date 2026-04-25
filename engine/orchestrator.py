"""
Orchestrator — State machine that coordinates the agent pipeline.

Pipeline flow:
  CREATED → SCOUTING → SCOUT_DONE ⏸️ → READING → READ_DONE ⏸️ 
  → WRITING → WRITE_DONE ⏸️ → REVIEWING → COMPLETE

Each "DONE" state is an approval gate (unless auto_approve=True).
The orchestrator:
  - Runs agents in sequence
  - Persists state to JSON files
  - Supports pause/resume
  - Exposes status via REST API
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from engine.config import Config
from engine.schemas import (
    JobState, JobStatus, AgentResult,
    ScoutResult, GroupDraft, WrittenArticle, ReviewedDraft,
    ArticleGroup, StructuredDraft,
)
from engine.llm.base import LLMProvider
from engine.storage.base import StorageBackend


class Orchestrator:
    """
    Pipeline orchestrator — coordinates agents through the full pipeline.
    
    Usage:
        orch = Orchestrator()
        
        # Start full pipeline
        job = orch.start(topic="ML for landslide detection")
        print(job.status)  # scouting / scout_done / ...
        
        # After scout_done, approve a group
        job = orch.approve(job.job_id, group_index=0)  # → reading
        
        # Check status anytime
        status = orch.get_status(job.job_id)
    """
    
    def __init__(
        self,
        llm: LLMProvider | None = None,
        storage: StorageBackend | None = None,
        reviewer_llm: LLMProvider | None = None,
    ):
        self._cfg = Config.get_instance()
        self._llm = llm
        self._storage = storage
        self._reviewer_llm = reviewer_llm
    
    @property
    def jobs_dir(self) -> Path:
        p = Path(self._cfg.get("agent.jobs_dir", "/app/data/agent_jobs"))
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def output_dir(self) -> Path:
        p = Path(self._cfg.get("agent.output_dir", "/app/data/output"))
        p.mkdir(parents=True, exist_ok=True)
        return p
    
    @property
    def auto_approve(self) -> bool:
        return self._cfg.get("agent.auto_approve", False)
    
    # ── Job Lifecycle ──────────────────────────────────────────
    
    def start(
        self,
        topic: str,
        pipeline: str = "full",
        user_comment: str = "",
        **kwargs,
    ) -> JobState:
        """
        Start a new pipeline job.
        
        Args:
            topic: Research topic or query
            pipeline: "full" | "scout_only" | "read_write" | "write_only" | "review_only"
            user_comment: User's additional instructions for writer
            **kwargs: Passed to first agent in chain
        
        Returns:
            JobState with status=SCOUTING (or appropriate start state)
        """
        job_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        state = JobState(
            job_id=job_id,
            status=JobStatus.CREATED,
            input_topic=topic,
            user_comment=user_comment,
            pipeline=pipeline,
        )
        self._save_state(state)
        
        # Start running based on pipeline type
        if pipeline == "scout_only":
            return self._run_scout(state, **kwargs)
        elif pipeline == "write_only":
            # User provides draft directly
            state.status = JobStatus.WRITING
            return self._run_writer(state, **kwargs)
        elif pipeline == "review_only":
            state.status = JobStatus.REVIEWING
            return self._run_reviewer(state, **kwargs)
        else:
            # Full or read_write: start with scout
            return self._run_scout(state, **kwargs)
    
    def get_status(self, job_id: str) -> JobState | None:
        """Load current state of a job."""
        return self._load_state(job_id)
    
    def approve(self, job_id: str, group_index: int = -1, comment: str = "") -> JobState:
        """
        Approve current stage and proceed to next.
        
        Args:
            job_id: Job identifier
            group_index: Which group/draft to select (-1 = best)
            comment: Additional instruction for next stage
        
        Returns:
            Updated JobState (now running next stage)
        """
        state = self._load_state(job_id)
        if not state:
            raise ValueError(f"Job {job_id} not found")
        
        if not state.is_paused:
            raise ValueError(f"Job {job_id} is not paused (status={state.status.value})")
        
        state.selected_group_index = group_index
        state.user_comment = comment or state.user_comment
        state.add_approval(stage=state.status.value.replace("_done", ""), action="approve", detail=f"group={group_index}")
        
        # Determine what stage just completed and run next
        if state.status == JobStatus.SCOUT_DONE:
            return self._run_reader(state)
        elif state.status == JobStatus.READ_DONE:
            return self._run_writer(state)
        elif state.status == JobStatus.WRITE_DONE:
            return self._run_reviewer(state)
        else:
            raise ValueError(f"No next stage after {state.status.value}")
    
    def skip(self, job_id: str, reason: str = "") -> JobState:
        """Skip current optional stage (e.g., skip review)."""
        state = self._load_state(job_id)
        if not state:
            raise ValueError(f"Job {job_id} not found")
        
        state.add_approval(stage=state.status.value.replace("_done", ""), action="skip", detail=reason)
        
        if state.status == JobStatus.WRITE_DONE:
            # Skip review → complete
            state.status = JobStatus.COMPLETE
            self._save_state(state)
            return state
        elif state.status == JobStatus.READ_DONE:
            # Skip write? Unusual but possible
            return self._run_writer(state)
        else:
            raise ValueError(f"Cannot skip from {state.status.value}")
    
    def reject(self, job_id: str, reason: str = "") -> JobState:
        """Cancel the job at current approval gate."""
        state = self._load_state(job_id)
        if not state:
            raise ValueError(f"Job {job_id} not found")
        
        state.status = JobStatus.CANCELLED
        state.add_approval(
            stage=state.status.value.replace("_done", ""),
            action="reject",
            detail=reason,
        )
        self._save_state(state)
        return state
    
    def revise(self, job_id: str, instruction: str = "") -> JobState:
        """
        Re-run the last stage with new instructions.
        E.g., after review says "fix style" → revise(instruction="make more formal")
        """
        state = self._load_state(job_id)
        if not state:
            raise ValueError(f"Job {job_id} not found")
        
        state.user_comment = instruction
        state.add_approval(stage="revise", action="revise", detail=instruction)
        
        # Re-run the last completed stage
        if state.status == JobStatus.WRITE_DONE:
            return self._run_writer(state)
        elif state.status == JobStatus.REVIEW_DONE:
            return self._run_reviewer(state)
        elif state.status == JobStatus.READ_DONE:
            return self._run_reader(state)
        else:
            raise ValueError(f"Cannot revise from {state.status.value}")
    
    def list_jobs(self, limit: int = 20) -> list[dict]:
        """List recent jobs with summary info."""
        jobs = []
        for f in sorted(self.jobs_dir.glob("*.json"), reverse=True)[:limit]:
            try:
                data = json.loads(f.read_text())
                jobs.append({
                    "job_id": data.get("job_id"),
                    "status": data.get("status"),
                    "topic": data.get("input", {}).get("topic", ""),
                    "pipeline": data.get("input", {}).get("pipeline", ""),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return jobs
    
    # ── Stage Runners (private) ────────────────────────────────
    
    def _run_scout(self, state: JobState, **kwargs) -> JobState:
        """Run Scout Agent."""
        from engine.agents.scout import ScoutAgent
        
        state.status = JobStatus.SCOUTING
        self._save_state(state)
        
        agent = ScoutAgent(llm=self._llm, storage=self._storage)
        result: AgentResult = agent.run(
            topic=state.input_topic,
            **kwargs,
        )
        
        if result.success:
            state.set_result("scout", result.data)
            
            if self.auto_approve:
                # Auto-select best group
                if isinstance(result.data, ScoutResult) and result.data.best_group:
                    state.selected_group_index = 0
                return self._run_reader(state)
            else:
                state.status = JobStatus.SCOUT_DONE
        else:
            state.status = JobStatus.FAILED
            state.error = result.error
        
        self._save_state(state)
        return state
    
    def _run_reader(self, state: JobState) -> JobState:
        """Run Reader Agent on selected group."""
        from engine.agents.reader import ReaderAgent
        
        state.status = JobStatus.READING
        self._save_state(state)
        
        # Get selected group from scout results
        scout_data = state.get_result("scout")
        groups = []
        if scout_data and isinstance(scout_data, dict):
            raw_groups = scout_data.get("groups", [])
            for rg in raw_groups:
                groups.append(ArticleGroup.from_dict(rg))
        
        # Select group
        idx = state.selected_group_index
        if idx < 0 and groups:
            idx = 0  # default to first
        selected_group = groups[idx] if idx < len(groups) else None
        
        if not selected_group:
            state.status = JobStatus.FAILED
            state.error = "No group selected for reading"
            self._save_state(state)
            return state
        
        agent = ReaderAgent(llm=self._llm, storage=self._storage)
        result: AgentResult = agent.run(
            group=selected_group,
            user_comment=state.user_comment,
        )
        
        if result.success:
            state.set_result("read", result.data)
            
            if self.auto_approve:
                return self._run_writer(state)
            else:
                state.status = JobStatus.READ_DONE
        else:
            state.status = JobStatus.FAILED
            state.error = result.error
        
        self._save_state(state)
        return state
    
    def _run_writer(self, state: JobState) -> JobState:
        """Run Writer Agent."""
        from engine.agents.writer import WriterAgent
        
        state.status = JobStatus.WRITING
        self._save_state(state)
        
        # Get draft from reader (or direct input)
        draft_data = state.get_result("read")
        draft = None
        if draft_data and isinstance(draft_data, dict):
            draft = GroupDraft.from_dict(draft_data)
        
        agent = WriterAgent(llm=self._llm, storage=self._storage)
        result: AgentResult = agent.run(
            draft=draft,
            user_comment=state.user_comment,
            topic=state.input_topic,
        )
        
        if result.success:
            state.set_result("write", result.data)
            
            # Save output file
            if isinstance(result.data, WrittenArticle):
                self._save_output(state, result.data)
            
            if self.auto_approve:
                return self._run_reviewer(state)
            else:
                state.status = JobStatus.WRITE_DONE
        else:
            state.status = JobStatus.FAILED
            state.error = result.error
        
        self._save_state(state)
        return state
    
    def _run_reviewer(self, state: JobState) -> JobState:
        """Run Reviewer Agent (if enabled)."""
        cfg_reviewer = self._cfg.get("reviewer.enabled", False)
        if not cfg_reviewer:
            # Reviewer disabled → mark as complete
            state.status = JobStatus.COMPLETE
            self._save_state(state)
            return state
        
        from engine.agents.reviewer import ReviewerAgent
        
        state.status = JobStatus.REVIEWING
        self._save_state(state)
        
        written_data = state.get_result("write")
        written = None
        if written_data and isinstance(written_data, dict):
            written = WrittenArticle(**written_data)
        
        agent = ReviewerAgent(llm=self._reviewer_llm, storage=self._storage)
        result: AgentResult = agent.run(
            article=written,
            source_articles=[],  # TODO: extract from draft
        )
        
        if result.success:
            state.set_result("review", result.data)
            state.status = JobStatus.COMPLETE
        else:
            state.status = JobStatus.FAILED
            state.error = result.error
        
        self._save_state(state)
        return state
    
    # ── Persistence ────────────────────────────────────────────
    
    def _state_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"
    
    def _save_state(self, state: JobState):
        path = self._state_path(state.job_id)
        path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    
    def _load_state(self, job_id: str) -> JobState | None:
        path = self._state_path(job_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return JobState.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None
    
    def _save_output(self, state: JobState, article: WrittenArticle):
        """Save written article to output directory."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() else "_" for c in state.input_topic[:40])
        
        # Markdown
        md_path = self.output_dir / f"{ts}_{safe_topic}.md"
        md_path.write_text(article.text, encoding="utf-8")
        
        # Metadata JSON
        meta_path = self.output_dir / f"{ts}_{safe_topic}.meta.json"
        meta_path.write_text(json.dumps(article.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# ── Convenience: classmethod shortcuts ─────────────────────────

def run_pipeline(
    topic: str,
    pipeline: str = "full",
    auto_approve: bool = True,
    **kwargs,
) -> JobState:
    """
    One-liner: run entire pipeline and return final state.
    
    Usage:
        state = run_pipeline("ML for earthquake prediction", auto_approve=True)
        print(state.status)  # COMPLETE or FAILED
        result = state.get_result("write", WrittenArticle)
    """
    orch = Orchestrator()
    if auto_approve:
        # Temporarily set auto_approve
        from engine.config import Config
        cfg = Config.get_instance()
        old_val = cfg.get("agent.auto_approve", False)
        cfg.set("agent.auto_approve", True)
    
    state = orch.start(topic=topic, pipeline=pipeline, **kwargs)
    
    # If auto_approve and not yet complete, it means we need to wait
    # (agents run synchronously in this mode)
    while state.is_running:
        time.sleep(1)
        state = orch.get_status(state.job_id)
        if not state:
            break
    
    if auto_approve:
        from engine.config import Config
        cfg = Config.get_instance()
        cfg.set("agent.auto_approve", old_val)
    
    return state
