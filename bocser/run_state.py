"""Runtime state helpers.

This module holds small pieces of mutable runtime state that previously lived
as module-level globals in other modules (e.g. `_CURRENT_STRUCTURE_ID` in
`calc.py`). Centralizing them here keeps state explicit and easier to test.
"""
from typing import Final

_CURRENT_STRUCTURE_ID: int = 0


def increase_structure_id() -> int:
    """Allocate and return the next structure id (0-based).

    Returns the id that was just allocated (previous value), matching the
    previous behaviour where callers used the old value and then incremented
    the counter.
    """
    global _CURRENT_STRUCTURE_ID
    _CURRENT_STRUCTURE_ID += 1
    return _CURRENT_STRUCTURE_ID - 1


def peek_structure_id() -> int:
    """Return the current structure id without incrementing."""
    return _CURRENT_STRUCTURE_ID


def reset() -> None:
    """Reset the runtime state (useful for tests)."""
    global _CURRENT_STRUCTURE_ID
    _CURRENT_STRUCTURE_ID = 0
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class RunState:
    mol_file_name: Optional[str] = None
    norm_energy: float = 0.0
    dihedral_ids: List[Any] = field(default_factory=list)
    cur_add_points: List[Any] = field(default_factory=list)
    global_degrees: List[Any] = field(default_factory=list)
    structures_path: str = ""
    exp_name: str = ""
    asked_points: List[Any] = field(default_factory=list)
    model_chk: Optional[Any] = None
    current_minima: float = 1e9
    acq_vals_log: List[float] = field(default_factory=list)
    last_opt_ok: bool = True
    minima: List[Any] = field(default_factory=list)


# Single shared state instance used by the application.
STATE = RunState()
