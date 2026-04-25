"""Scout Agent — finds and groups articles by potential. (Sprint 2)"""
from engine.agents.base import BaseAgent
from engine.schemas import AgentResult, ScoutResult

class ScoutAgent(BaseAgent):
    """Finds articles and groups them by potential: replication / review / data_paper."""
    
    @property
    def name(self) -> str:
        return "scout"
    
    def run(self, **kwargs) -> AgentResult:
        # TODO Sprint 2 implementation
        raise NotImplementedError("ScoutAgent not yet implemented (Sprint 2)")
