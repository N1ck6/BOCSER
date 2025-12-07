"""Config loader for BOCSER.

Provides `load_config(path)` which reads YAML, validates keys against
`ConfSearchConfig` and returns a typed config object. Unknown keys are
warned about; missing keys use defaults from `ConfSearchConfig`.
"""
from dataclasses import fields
from typing import Any, Dict

import yaml

from default_vals import ConfSearchConfig
import logging
logger = logging.getLogger(__name__)


class ConfigError(Exception):
    pass


def _coerce_type(value: Any, target_type: type):
    """Try to coerce value to target_type, raise ConfigError on failure."""
    # Basic coercions for common types
    try:
        if target_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                val = value.strip().lower()
                if val in ("true", "1", "yes", "y"):  # noqa: W503
                    return True
                if val in ("false", "0", "no", "n"):  # noqa: W503
                    return False
            raise ValueError()
        return target_type(value)
    except Exception as e:
        raise ConfigError(f"Failed to coerce value {value!r} to {target_type}: {e}")


def load_config(path: str) -> ConfSearchConfig:
    """Load YAML config from `path` and return `ConfSearchConfig`.

    - Unknown keys are ignored with a warning.
    - Values are coerced to the dataclass field types when possible.
    - Missing values use the dataclass defaults.
    """
    try:
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ConfigError(f"Failed to read config file {path}: {e}")

    if not isinstance(raw, dict):
        raise ConfigError("Config file must contain a mapping (YAML dict) at top level")

    cfg_kwargs: Dict[str, Any] = {}
    valid_fields = {f.name: f.type for f in fields(ConfSearchConfig)}

    for k, v in raw.items():
        if k not in valid_fields:
            # don't fail hard on unknown keys, just warn
            logger.warning("Unknown config key '%s' - ignoring", k)
            continue
        target_type = valid_fields[k]
        try:
            # Attempt to coerce simple common types; if it's e.g. Union[str, None],
            # leave the value as-is and rely on the dataclass to validate later.
            if getattr(target_type, "__origin__", None) is None and isinstance(v, (str, int, float, bool)):
                v = _coerce_type(v, target_type)
        except ConfigError:
            # re-raise with context
            raise
        cfg_kwargs[k] = v

    # Build the dataclass using explicit kwargs; missing fields default
    try:
        config = ConfSearchConfig(**cfg_kwargs)
    except TypeError as e:
        raise ConfigError(f"Failed to construct ConfSearchConfig: {e}")

    return config
