"""
Engine configuration loader.

Loads from:
  1. config.yaml (project root)
  2. .env file (for API keys)
  3. Environment variables (override)
  4. Defaults
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ── Project paths ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # .../geo_digest/
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = Path(os.environ.get("GEO_DATA_DIR", str(PROJECT_ROOT / "data")))
OUTPUT_DIR = DATA_DIR / "output"
JOBS_DIR = DATA_DIR / "agent_jobs"
RUNS_DIR = Path(os.environ.get("GEO_RUNS_DIR", str(PROJECT_ROOT / "runs")))

# Ensure dirs exist
for d in (DATA_DIR, OUTPUT_DIR, JOBS_DIR, RUNS_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ── Data classes ───────────────────────────────────────────────

@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "minimax"              # minimax / openai_compat
    model: str = "MiniMax-M2.7"
    base_url: str = ""
    api_key: str = ""                       # loaded from env
    api_key_env: str = "MINIMAX_API_KEY"
    disable_thinking: bool = True
    max_tokens: int = 4096
    timeout: int = 120
    retries: int = 3


@dataclass
class ReviewerConfig:
    """Reviewer LLM configuration (separate model)."""
    enabled: bool = True
    provider: str = "openai_compat"         # different from writer!
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    max_tokens: int = 8192


@dataclass
class ScoutConfig:
    """Scout agent configuration."""
    max_results_per_source: int = 25
    min_citations: int = 0
    time_range: str = ""                    # e.g. "2023-2025"
    languages: list[str] = field(default_factory=lambda: ["en"])
    min_confidence_for_group: float = 0.5
    enable_semantic_scholar: bool = True
    enable_crossref: bool = True


@dataclass
class ReaderConfig:
    """Reader agent configuration."""
    download_pdfs: bool = True
    pdf_timeout: int = 30
    use_grobid: bool = False               # Grobid for structured extraction
    grobid_url: str = "http://localhost:8070"
    fallback_to_abstract: bool = True       # If no PDF available


@dataclass
class WriterConfig:
    """Writer agent configuration."""
    default_style: str = "review"           # review / replication / data_paper / short_comm
    default_language: str = "ru"
    target_words_review: int = 4000
    target_words_replication: int = 2500
    target_words_short_comm: int = 1500
    include_bibliography: bool = True
    mark_gaps_as: str = "[NEED RESEARCH]"


@dataclass
class EngineConfig:
    """
    Complete engine configuration.
    
    Usage:
        cfg = EngineConfig.load()
        cfg.llm.api_key → "xxx"
        cfg.reviewer.model → "gpt-4o"
    """
    llm: LLMConfig = field(default_factory=LLMConfig)
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)
    scout: ScoutConfig = field(default_factory=ScoutConfig)
    reader: ReaderConfig = field(default_factory=ReaderConfig)
    writer: WriterConfig = field(default_factory=WriterConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    jobs_dir: Path = field(default_factory=lambda: JOBS_DIR)
    runs_dir: Path = field(default_factory=lambda: RUNS_DIR)
    
    # Raw config dict (from yaml)
    _raw: dict = field(default_factory=dict)
    
    # ── Loaders ──
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "EngineConfig":
        """Load configuration from yaml + env + defaults."""
        path = config_path or CONFIG_PATH
        
        raw = {}
        if HAS_YAML and path.exists():
            try:
                raw = yaml.safe_load(path.read_text()) or {}
            except Exception:
                pass
        
        # Load .env for API keys
        env_vars = cls._load_env()
        
        return cls._from_raw(raw, env_vars)
    
    @classmethod
    def _load_env(cls) -> dict:
        """Load key=value from .env file."""
        env = {}
        if ENV_FILE.exists():
            for line in ENV_FILE.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
        return env
    
    @classmethod
    def _from_raw(cls, raw: dict, env_vars: dict) -> "EngineConfig":
        """Build EngineConfig from raw yaml + env vars."""
        
        # LLM config
        llm_raw = raw.get("llm", {})
        llm = LLMConfig(
            provider="minimax",
            model=llm_raw.get("model", "MiniMax-M2.7"),
            base_url=llm_raw.get("base_url", "https://api.minimax.chat/v1"),
            api_key=env_vars.get("MINIMAX_API_KEY", ""),
            api_key_env="MINIMAX_API_KEY",
            disable_thinking=llm_raw.get("disable_thinking", True),
            max_tokens=llm_raw.get("max_tokens", 4096),
        )
        
        # Reviewer config
        rev_raw = raw.get("reviewer", {})
        reviewer = ReviewerConfig(
            enabled=rev_raw.get("enabled", True),
            provider=rev_raw.get("provider", "openai_compat"),
            model=rev_raw.get("model", "gpt-4o"),
            base_url=rev_raw.get("base_url", "https://api.openai.com/v1"),
            api_key_env=rev_raw.get("api_key_env", "OPENAI_API_KEY"),
        )
        
        # Scout config
        scout_raw = raw.get("scout", {})
        scout = ScoutConfig(
            max_results_per_source=scout_raw.get("max_results", 25),
            min_citations=scout_raw.get("min_citations", 0),
            time_range=scout_raw.get("time_range", ""),
            enable_semantic_scholar=(
                raw.get("sources", {}).get("semantic_scholar", {}).get("enabled", True)
            ),
            enable_crossref=(
                raw.get("sources", {}).get("crossref", {}).get("enabled", True)
            ),
        )
        
        # Reader config
        reader_raw = raw.get("reader", {})
        reader = ReaderConfig(
            download_pdfs=reader_raw.get("download_pdfs", True),
            use_grobid=reader_raw.get("use_grobid", False),
            grobid_url=reader_raw.get("grobid_url", "http://localhost:8070"),
        )
        
        # Writer config
        writer_raw = raw.get("writer", {})
        writer = WriterConfig(
            default_style=writer_raw.get("default_style", "review"),
            default_language=writer_raw.get("default_language", "ru"),
            target_words_review=writer_raw.get("target_words_review", 4000),
            target_words_replication=writer_raw.get("target_words_replication", 2500),
            mark_gaps_as=writer_raw.get("mark_gaps", "[NEED RESEARCH]"),
        )
        
        return cls(
            llm=llm,
            reviewer=reviewer,
            scout=scout,
            reader=reader,
            writer=writer,
            _raw=raw,
        )
    
    def get_api_key(self, env_name: str) -> str:
        """Get API key by env variable name."""
        # Check os.environ first (runtime override), then .env cache
        return os.environ.get(env_name, "")
    
    def to_dict(self) -> dict:
        """Serialize for debugging/API."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "base_url": self.llm.base_url[:50] + "..." if len(self.llm.base_url) > 50 else self.llm.base_url,
                "has_api_key": bool(self.get_api_key(self.llm.api_key_env)),
            },
            "reviewer": {
                "enabled": self.reviewer.enabled,
                "provider": self.reviewer.provider,
                "model": self.reviewer.model,
                "has_api_key": bool(self.get_api_key(self.reviewer.api_key_env)),
            },
            "scout": {
                "max_results": self.scout.max_results_per_source,
                "time_range": self.scout.time_range,
            },
            "paths": {
                "data_dir": str(self.data_dir),
                "output_dir": str(self.output_dir),
                "jobs_dir": str(self.jobs_dir),
            },
        }
    
    def __repr__(self) -> str:
        return (
            f"EngineConfig(llm={self.llm.provider}/{self.llm.model}, "
            f"reviewer={self.reviewer.model if self.reviewer.enabled else 'disabled'}, "
            f"data={self.data_dir})"
        )


# ── Global singleton (lazy load) ───────────────────────────────

_config_instance: Optional[EngineConfig] = None


def get_config() -> EngineConfig:
    """Get global config instance (lazy, cached)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = EngineConfig.load()
    return _config_instance


def reload_config() -> EngineConfig:
    """Force reload config (useful after .env changes)."""
    global _config_instance
    _config_instance = EngineConfig.load()
    return _config_instance
