"""REST API for GEO-Digest Agent Engine. (Sprint 4)

FastAPI application with endpoints for:
- Job management (create, list, get status)
- Pipeline control (start, approve, revise, cancel)
- Human-in-the-loop approval gates

Usage:
    uvicorn engine.api:app --host 0.0.0.0 --port 3002

Or from CLI:
    python -m engine.api
"""

from __future__ import annotations

import datetime
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel as PydanticModel, Field

from engine.config import EngineConfig, get_config
from engine.schemas import JobState, JobStatus, ScoutResult, GroupDraft
from engine.orchestrator import Orchestrator


# ── Request/Response Models ─────────────────────────────────


class CreateJobRequest(PydanticModel):
    topic: str = Field(..., min_length=3, description="Research topic")
    pipeline: str = Field(default="full", description="Pipeline type: full|scout_only")
    user_comment: str = Field(default="", description="User notes for writer")


class ApproveScoutRequest(PydanticModel):
    group_index: int = Field(..., ge=0, description="Selected group index from scout results")
    comment: str = Field(default="", description="Approval comment")


class ApproveDraftRequest(PydanticModel):
    comment: str = Field(default="", description="Approval comment for writer")


class ReviseRequest(PydanticModel):
    comment: str = Field(..., min_length=1, description="Revision instructions")


class JobResponse(PydanticModel):
    """Serializable job state for API responses."""
    job_id: str
    status: str
    input_topic: str
    user_comment: str
    pipeline: str
    created_at: str
    updated_at: str
    selected_group_index: int = -1
    error: str = ""
    is_running: bool = False
    is_paused: bool = False
    is_terminal: bool = False
    results_summary: dict = {}
    approval_history: list = []

    @classmethod
    def from_job_state(cls, state: JobState) -> "JobResponse":
        return cls(
            job_id=state.job_id,
            status=state.status.value,
            input_topic=state.input_topic,
            user_comment=state.user_comment,
            pipeline=state.pipeline,
            created_at=state.created_at,
            updated_at=state.updated_at,
            selected_group_index=state.selected_group_index,
            error=state.error,
            is_running=state.is_running,
            is_paused=state.is_paused,
            is_terminal=state.is_terminal,
            results_summary={k: type(v).__name__ for k, v in state.results.items()},
            approval_history=state.approval_history,
        )


class HealthResponse(PydanticModel):
    status: str
    version: str
    config: str


# ── App Lifecycle ───────────────────────────────────────────

_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        cfg = get_config()
        _orchestrator = Orchestrator(config=cfg)
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize orchestrator on startup."""
    global _orchestrator
    cfg = get_config()
    _orchestrator = Orchestrator(config=cfg)
    print(f"[API] Engine initialized: {cfg}")
    yield
    print("[API] Shutdown")


app = FastAPI(
    title="GEO-Digest Agent API",
    version="1.0.0",
    description="REST API for multi-agent article writing pipeline",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    cfg = get_config()
    return HealthResponse(
        status="ok",
        version="1.0.0",
        config=str(cfg),
    )


@app.post("/api/v1/jobs", response_model=JobResponse)
async def create_job(req: CreateJobRequest):
    """
    Create a new pipeline job.

    The job starts in 'created' status. Use POST /jobs/{id}/start to begin.
    """
    orc = get_orchestrator()
    state = orc.create_job(
        topic=req.topic,
        pipeline=req.pipeline,
        user_comment=req.user_comment,
    )
    return JobResponse.from_job_state(state)


@app.get("/api/v1/jobs", response_model=list[JobResponse])
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    status_filter: Optional[str] = Query(default=None, description="Filter by status"),
):
    """List all jobs, most recent first."""
    orc = get_orchestrator()
    jobs = orc.list_jobs()
    if status_filter:
        jobs = [j for j in jobs if j.status.value == status_filter]
    jobs = jobs[:limit]
    return [JobResponse.from_job_state(j) for j in jobs]


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and details."""
    orc = get_orchestrator()
    try:
        state = orc.load_state(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobResponse.from_job_state(state)


@app.post("/api/v1/jobs/{job_id}/start", response_model=JobResponse)
async def start_job(job_id: str):
    """
    Start the pipeline job (begins scouting phase).

    Transitions: created -> scouting
    """
    orc = get_orchestrator()
    try:
        state = orc.start_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return JobResponse.from_job_state(state)


@app.post("/api/v1/jobs/{job_id}/approve", response_model=JobResponse)
async def approve_scout(job_id: str, req: ApproveScoutRequest):
    """
    Approval Gate 1: Select which article group to develop.

    Human-in-the-loop: user reviews scout results and picks a group.
    Transitions: scout_done -> reading
    """
    orc = get_orchestrator()
    try:
        state = orc.approve_group(job_id, req.group_index, req.comment)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return JobResponse.from_job_state(state)


@app.post("/api/v1/jobs/{job_id}/approve-draft", response_model=JobResponse)
async def approve_draft(job_id: str, req: ApproveDraftRequest):
    """
    Approval Gate 2: Approve the structured draft for writing.

    Human-in-the-loop: user reviews draft and approves or requests changes.
    Transitions: read_done -> writing
    """
    orc = get_orchestrator()
    try:
        state = orc.approve_draft(job_id, req.comment)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return JobResponse.from_job_state(state)


@app.post("/api/v1/jobs/{job_id}/revise", response_model=JobResponse)
async def request_revision(job_id: str, req: ReviseRequest):
    """
    Request revision of the written article.

    Transitions: write_done -> writing (with revision comment)
    """
    orc = get_orchestrator()
    try:
        state = orc.request_revision(job_id, req.comment)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return JobResponse.from_job_state(state)


@app.post("/api/v1/jobs/{job_id}/skip-review", response_model=JobResponse)
async def skip_review(job_id: str):
    """
    Skip reviewer and mark job complete.

    Transitions: write_done -> complete
    """
    orc = get_orchestrator()
    try:
        state = orc.skip_review(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return JobResponse.from_job_state(state)


@app.delete("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def cancel_job(job_id: str):
    """Cancel a running or paused job."""
    orc = get_orchestrator()
    try:
        state = orc.cancel_job(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobResponse.from_job_state(state)


# ── CLI Entry Point ────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 3002):
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
    print(f"[API] Server running on http://{host}:{port}")


if __name__ == "__main__":
    run_server()
