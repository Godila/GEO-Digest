"""Reviewer Agent — reviews article with a different LLM model. (Sprint 6)"""
from engine.agents.base import BaseAgent
from engine.schemas import AgentResult, ReviewedDraft

class ReviewerAgent(BaseAgent):
    """Reviews draft for style, facts, and scientific accuracy."""
    
    @property
    def name(self) -> str:
        return "reviewer"
    
    def run(self, **kwargs) -> AgentResult:
        # TODO Sprint 6 implementation
        raise NotImplementedError("ReviewerAgent not yet implemented (Sprint 6)")
