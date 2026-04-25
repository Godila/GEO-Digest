"""
Agent Registry — centralized registry of all available agents.

Usage:
    from engine.agents import get_agent, list_agents
    
    scout = get_agent("scout")
    result = scout.run(topic="ML geology")
"""

from .base import BaseAgent
from .tools import AgentTools

# Lazy imports to avoid circular deps
_AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    """Decorator to register an agent class."""
    _AGENT_REGISTRY[cls.name] = cls
    return cls


def get_agent(name: str, **kwargs) -> BaseAgent:
    """Instantiate agent by name."""
    if name not in _AGENT_REGISTRY:
        # Try lazy import of known agents
        _lazy_load(name)
    
    if name not in _AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name!r}. Available: {list(_AGENT_REGISTRY.keys())}")
    
    return _AGENT_REGISTRY[name](**kwargs)


def list_agents() -> list[dict]:
    """List all registered agents with metadata."""
    _ensure_all_loaded()
    return [
        {
            "name": cls.name,
            "description": getattr(cls, "description", ""),
            "required_input": getattr(cls, "required_input_keys", []),
        }
        for cls in _AGENT_REGISTRY.values()
    ]


def _lazy_load(name: str):
    """Lazy import agent module on first use."""
    imports = {
        "scout": ".scout",
        "reader": ".reader",
        "writer": ".writer",
        "reviewer": ".reviewer",
    }
    if name in imports:
        import importlib
        importlib.import_module(imports[name], package=__name__)


def _ensure_all_loaded():
    """Ensure all known agents are imported and registered."""
    for name in ("scout", "reader", "writer", "reviewer"):
        if name not in _AGENT_REGISTRY:
            _lazy_load(name)


__all__ = ["BaseAgent", "AgentTools", "get_agent", "list_agents", "register_agent"]
