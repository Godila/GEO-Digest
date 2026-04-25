"""Writer Agent — generates article from StructuredDraft. (Sprint 5)"""
from engine.agents.base import BaseAgent
from engine.schemas import AgentResult, WrittenArticle

class WriterAgent(BaseAgent):
    """Generates a full article draft from structured input."""
    
    @property
    def name(self) -> str:
        return "writer"
    
    def run(self, **kwargs) -> AgentResult:
        # TODO Sprint 5 implementation
        raise NotImplementedError("WriterAgent not yet implemented (Sprint 5)")
