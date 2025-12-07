"""Simple runtime config manager.

Provides a small, explicit API to store and retrieve the active
`ConfSearchConfig` instance. This reduces ad-hoc use of globals and
centralizes where the active configuration is kept.

This module is intentionally tiny and synchronous.
"""
from typing import Optional

from default_vals import ConfSearchConfig
from config_loader import load_config as _load_config, ConfigError
import logging

logger = logging.getLogger(__name__)

# Module-level holder for the active config
_config: Optional[ConfSearchConfig] = None


def set_config(config: ConfSearchConfig) -> None:
    """Set the active configuration object."""
    global _config
    _config = config
    logger.debug("Runtime config set: %s", config)


def get_config() -> Optional[ConfSearchConfig]:
    """Return the active configuration, or None if not set."""
    return _config


def load_config_from_file(path: str) -> ConfSearchConfig:
    """Load configuration from a YAML file and set it as active.

    Raises ConfigError or FileNotFoundError as from `config_loader.load_config`.
    """
    cfg = load_config(path)
    set_config(cfg)
    return cfg


def load_config(path: str) -> ConfSearchConfig:
    """Compatibility wrapper: load config from `path`, set as active and return it.

    This mirrors the old `config_loader.load_config` API so callers can import
    `load_config` from this module directly.
    """
    cfg = _load_config(path)
    set_config(cfg)
    return cfg


# Re-export ConfigError for compatibility
__all__ = ["set_config", "get_config", "load_config_from_file", "load_config", "clear_config", "ConfigError"]


def clear_config() -> None:
    """Clear the stored configuration (useful in tests)."""
    global _config
    _config = None
