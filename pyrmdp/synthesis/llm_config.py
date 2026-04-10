"""
Centralised LLM configuration for the synthesis pipeline.

Loads settings from a YAML config file (default: ``llm.yaml`` in the repo
root, overridable via the ``PYRMDP_LLM_CONFIG`` env-var).  Any field can
also be overridden via environment variables prefixed with ``PYRMDP_``:

    PYRMDP_MODEL=gpt-4-turbo  PYRMDP_TEMPERATURE=0.5  python …

Provides a single ``build_llm_fn()`` helper that both ``llm_failure.py``
and ``delta_minimizer.py`` (and anything else) can call instead of each
inlining their own OpenAI client setup.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════
#  Config dataclass
# ════════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    """All tunables for the LLM backend used across the synthesis pipeline."""

    # Connection
    provider: str = "openai"              # "openai" | "azure" | "custom"
    api_key: str = ""                     # fallback: $OPENAI_API_KEY
    base_url: Optional[str] = None        # for self-hosted / vLLM / Ollama
    api_version: Optional[str] = None     # Azure only

    # Generation
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    timeout: float = 60.0                 # seconds per request

    # Retry
    max_retries: int = 2

    # ── Serialization helpers ──────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Build from a flat or nested dict (e.g. parsed YAML / JSON)."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# ════════════════════════════════════════════════════════════════════
#  Loading logic
# ════════════════════════════════════════════════════════════════════

_ENV_PREFIX = "PYRMDP_"

# Search order for the config file (first hit wins)
_DEFAULT_SEARCH_PATHS = [
    Path("llm.yaml"),                                       # cwd
    Path(__file__).resolve().parents[2] / "llm.yaml",       # repo root
    Path.home() / ".config" / "pyrmdp" / "llm.yaml",       # user-global
]


def _find_config_path() -> Optional[Path]:
    """Return the first config file that exists, or None."""
    env_path = os.environ.get("PYRMDP_LLM_CONFIG")
    if env_path:
        p = Path(env_path)
        return p if p.is_file() else None

    for candidate in _DEFAULT_SEARCH_PATHS:
        if candidate.is_file():
            return candidate
    return None


def _apply_env_overrides(cfg: LLMConfig) -> None:
    """Overlay ``PYRMDP_*`` env-vars onto the config in-place."""
    for field_name, field_obj in cfg.__dataclass_fields__.items():
        env_key = _ENV_PREFIX + field_name.upper()
        env_val = os.environ.get(env_key)
        if env_val is None:
            continue
        # Cast to the field's type
        ftype = field_obj.type
        if ftype in ("float", float):
            setattr(cfg, field_name, float(env_val))
        elif ftype in ("int", int):
            setattr(cfg, field_name, int(env_val))
        elif ftype in ("bool", bool):
            setattr(cfg, field_name, env_val.lower() in ("1", "true", "yes"))
        else:
            setattr(cfg, field_name, env_val)


def load_config(path: Optional[str] = None) -> LLMConfig:
    """
    Load :class:`LLMConfig` from YAML file + env-var overrides.

    Resolution order (each layer overrides the previous):
      1. Built-in defaults (the dataclass fields).
      2. YAML file (explicit *path*, or auto-discovered).
      3. ``PYRMDP_*`` environment variables.
      4. ``OPENAI_API_KEY`` env-var as a final api_key fallback.

    Returns
    -------
    LLMConfig
    """
    cfg = LLMConfig()

    config_path = Path(path) if path else _find_config_path()
    if config_path and config_path.is_file():
        try:
            import yaml  # optional dep — only if a config file is present
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            cfg = LLMConfig.from_dict(data)
            logger.info(f"Loaded LLM config from {config_path}")
        except ImportError:
            logger.warning(
                f"Found {config_path} but PyYAML is not installed — "
                "using defaults.  pip install pyyaml"
            )
        except Exception as exc:
            logger.warning(f"Failed to parse {config_path}: {exc} — using defaults.")

    # Env-var overrides
    _apply_env_overrides(cfg)

    # Final fallback: OPENAI_API_KEY
    if not cfg.api_key:
        cfg.api_key = os.environ.get("OPENAI_API_KEY", "")

    return cfg


# ════════════════════════════════════════════════════════════════════
#  LLM function builder
# ════════════════════════════════════════════════════════════════════

def build_llm_fn(config: Optional[LLMConfig] = None) -> Callable[[str], str]:
    """
    Return a ``fn(prompt) → response_text`` callable configured by *config*.

    If *config* is ``None`` it is loaded via :func:`load_config`.

    Raises
    ------
    EnvironmentError
        If no API key is available.
    ImportError
        If the ``openai`` package is missing.
    """
    if config is None:
        config = load_config()

    if not config.api_key:
        raise EnvironmentError(
            "No LLM API key found.  Set OPENAI_API_KEY, PYRMDP_API_KEY, "
            "or add 'api_key' to llm.yaml."
        )

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install 'pyrmdp[llm]'")

    client_kwargs: Dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    if config.timeout:
        client_kwargs["timeout"] = config.timeout
    if config.max_retries:
        client_kwargs["max_retries"] = config.max_retries

    client = OpenAI(**client_kwargs)

    def _call(prompt: str, *, system: str | None = None) -> str:
        """Send a chat completion request.

        Parameters
        ----------
        prompt : str
            User message (or concatenated system+user for legacy callers).
        system : str, optional
            If provided, sent as a separate ``system`` role message for
            better instruction following.  Otherwise *prompt* is sent as
            a single ``user`` message.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
        )
        return resp.choices[0].message.content

    return _call
