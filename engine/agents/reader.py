"""Reader Agent — reads articles and produces StructuredDraft. (Sprint 3)"""
from engine.agents.base import BaseAgent
from engine.schemas import AgentResult, GroupDraft

class ReaderAgent(BaseAgent):
    """Reads full text (or abstract) and produces structured draft."""
    
    @property
    def name(self) -> str:
        return "reader"
    
    def run(self, **kwargs) -> AgentResult:
        # TODO Sprint 3 implementation
        raise NotImplementedError("ReaderAgent not yet implemented (Sprint 3)")
