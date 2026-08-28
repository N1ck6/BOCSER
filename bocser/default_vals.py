from dataclasses import dataclass, field
from typing import List, Tuple, Union, Dict

_CONSTRAINT_ATOM_COUNT = {"bond": 2, "angle": 3, "dihedral": 4}

@dataclass
class ConfSearchConfig:
    mol_file_name : str
    spin_multiplicity : int = 1
    charge : int = 0
    orca_exec_command : str = "/opt/orca5/orca"
    num_of_procs : int = 8
    orca_method : str = "lda sto-3g"
    broken_struct_energy : float = 100.
    bond_length_threshold : float = 0.7 # Deprecated
    clash_vdw_scale: float = 0.0  # 0.0 = legacy fixed 0.7 Å threshold.
                                  # >0.0 = use clash_vdw_scale * (vdw_radius_a + vdw_radius_b) instead.
    ts : bool = False
    ring_bond_threshold : float = 1.75
    ts_ring_bond_threshold : float = 2.5
    use_grass : bool = False
    path_to_grass : str = ""
    grass_options : str = ""
    orca_poll_timeout_minutes: int = 60
    orca_poll_interval_ms: int = 1000
    sbatch_template_name: str = "sbatch_temp"
    rolling_window_size : int = 5
    rolling_std_threshold : float = 0.15
    rolling_mean_threshold : float = 1.
    num_initial_points : int = 3
    max_steps : int = 50
    exp_name : str = "cs"
    load_ensemble : Union[str, None] = None
    acquisition_function : str = "iv"
    ts_bonds: List[Tuple[int, int]] = field(default_factory=list)   # 1-индексация, как в исходном .mol
    ts_bond_max_length: float = 5.0
    fixed_double_bonds: List[Tuple[int, int]] = field(default_factory=list)  # 1-индексация, как в исходном .mol
    extra_constraints: List[Dict] = field(default_factory=list)
    ts_max_iter: int = 45
    clear_working_folder: bool = False
    checkpoint_step: Union[int, None] = None   # None - default behaviour, else - start at chosen step
    tf_eager_mode: bool = False # Управляет tf.config.run_functions_eagerly() - отладка shape-ошибок, замедление расчетов acquisition
    bond_stretch_factor: float = 1.4  # порог для _check_bond_topology_intact

    def __post_init__(self): # Форматировать в список кортежей двойных связей
        self.ts_bonds = [tuple(int(x) for x in pair) for pair in self.ts_bonds]
        self.fixed_double_bonds = [tuple(int(x) for x in pair) for pair in self.fixed_double_bonds]
        for name in ("ts_bonds", "fixed_double_bonds"):
            for pair in getattr(self, name):
                if len(pair) != 2:
                    raise ValueError(f"{name} entries must be [atom_a, atom_b] pairs, got {pair}")

        normalized = []
        for i, raw in enumerate(self.extra_constraints):
            if "type" not in raw or "atoms" not in raw:
                raise ValueError(f"extra_constraints[{i}]: нужны поля 'type' и 'atoms', получено {raw}")
            ctype = raw["type"]
            if ctype not in _CONSTRAINT_ATOM_COUNT:
                raise ValueError(f"extra_constraints[{i}]: type должен быть bond/angle/dihedral, получено {ctype!r}")
            atoms = tuple(int(a) for a in raw["atoms"])
            if len(atoms) != _CONSTRAINT_ATOM_COUNT[ctype]:
                raise ValueError(
                    f"extra_constraints[{i}]: для type={ctype} нужно "
                    f"{_CONSTRAINT_ATOM_COUNT[ctype]} атомов, получено {len(atoms)}: {atoms}"
                )
            value = raw.get("value", "current")
            normalized.append({"type": ctype, "atoms": atoms, "value": value})
        self.extra_constraints = normalized
