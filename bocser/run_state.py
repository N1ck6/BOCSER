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
