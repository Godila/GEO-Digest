"""Engine configuration loader."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("GEO_DATA_DIR", str(PROJECT_ROOT / "data")))
OUTPUT_DIR = DATA_DIR / "output"
JOBS_DIR = DATA_DIR / "agent_jobs"
for d in (DATA_DIR, OUTPUT_DIR, JOBS_DIR):
    d.mkdir(parents=True, exist_ok=True)


@dataclass
class LLMConfig:
    """LLM provider settings."""
    provider: str = "minimax"
    model: str = "MiniMax-M2.7"
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = "MINIMAX_API_KEY"
    disable_thinking: bool = True
    max_tokens: int = 4096
    timeout: int = 120
    retries: int = 3


@dataclass
class ReviewerConfig:
    """Reviewer (second model) settings."""
    enabled: bool = True
    provider: str = "openai_compat"
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"


@dataclass
class EngineConfig:
    """Root engine configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    jobs_dir: Path = field(default_factory=lambda: JOBS_DIR)
    _raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: str | None = None) -> "EngineConfig":
        """Load from config.yaml + .env."""
        path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
        raw = {}
        if HAS_YAML and path.exists():
            try:
                raw = yaml.safe_load(path.read_text()) or {}
            except Exception:
                pass

        # Load .env
        env = {}
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()

        # LLM config
        lr = raw.get("llm", {})
        llm = LLMConfig(
            provider=lr.get("provider", "minimax"),
            model=lr.get("model", "MiniMax-M2.7"),
            base_url=env.get("MINIMAX_BASE_URL", lr.get("base_url", "https://api.minimax.chat/anthropic")),
            api_key=env.get("MINIMAX_API_KEY", ""),
            api_key_env="MINIMAX_API_KEY",
        )

        # Reviewer config
        rv = raw.get("reviewer", {})
        rev = ReviewerConfig(
            enabled=rv.get("enabled", True),
            provider=rv.get("provider", "openai_compat"),
            model=rv.get("model", "gpt-4o"),
        )

        return cls(llm=llm, reviewer=rev, _raw=raw)

    def get_api_key(self, name: str) -> str:
        return os.environ.get(name, "")

    def __repr__(self) -> str:
        return f"EngineConfig(llm={self.llm.provider}/{self.llm.model})"


# ── Singleton ──────────────────────────────────────────────

_instance: EngineConfig | None = None


def get_config() -> EngineConfig:
    """Get or create singleton config instance."""
    global _instance
    if _instance is None:
        _instance = EngineConfig.load()
    return _instance
