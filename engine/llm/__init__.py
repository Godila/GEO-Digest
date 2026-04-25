"""
LLM Providers — unified interface for any LLM backend.
"""

from .base import LLMProvider
from .minimax import MiniMaxProvider
from .openai_compat import OpenAICompatProvider

__all__ = ["LLMProvider", "MiniMaxProvider", "OpenAICompatProvider"]


def get_llm(provider: str = "", **kwargs) -> LLMProvider:
    """
    Factory: get the right LLM provider instance.
    
    Args:
        provider: "minimax" | "openai_compat" (or empty = from config)
        **kwargs: Override config values
    
    Returns:
        Configured LLMProvider instance
    """
    if not provider:
        from engine.config import Config
        cfg = Config.get_instance()
        provider = cfg.get("llm.provider", "minimax")
    
    providers = {
        "minimax": MiniMaxProvider,
        "openai_compat": OpenAICompatProvider,
    }
    
    cls = providers.get(provider)
    if not cls:
        raise ValueError(f"Unknown LLM provider: {provider}. Available: {list(providers.keys())}")
    
    return cls(**kwargs)
