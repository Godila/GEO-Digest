"""LLM Providers."""
from .base import LLMProvider
from .minimax import MiniMaxProvider
from .openai_compat import OpenAICompatProvider

def create_provider(cfg) -> LLMProvider:
    p = cfg.llm.provider
    if p == "minimax":
        return MiniMaxProvider(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url,
                               model=cfg.llm.model, timeout=cfg.llm.timeout,
                               retries=cfg.llm.retries)
    elif p == "openai_compat":
        return OpenAICompatProvider(api_key=cfg.llm.api_key, base_url=cfg.llm.base_url,
                                    model=cfg.llm.model, timeout=cfg.llm.timeout)
    else:
        raise ValueError(f"Unknown LLM provider: {p}")


def get_llm():
    """Get default LLM provider from config singleton."""
    from engine.config import get_config
    return create_provider(get_config())


__all__ = ["LLMProvider", "MiniMaxProvider", "OpenAICompatProvider", "create_provider", "get_llm"]
