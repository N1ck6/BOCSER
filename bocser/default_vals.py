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
    clash_vdw_scale: float = 0.0  # 0.0 = legacy fixed 0.7 Å threshold; >0.0 = use clash_vdw_scale * (vdw_radius_a + vdw_radius_b) instead.
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
    vary_ts_bond_lengths: bool = True
    exclude_dof_slack: bool = True # На каждом кольце оставляет (N - 3) торсионных осей для GP. Излишние оси фиксируются на значении из исходной геометрии
    ts_bond_min_length: float = 1.0
    ts_bond_kernel_lengthscale: float = 0.5 # lengthscale RBF-ядра по измерениям длины TS-связи (единицы Å)
    ts_bond_scan_step: float = 0.1
    ts_bond_scan_min_steps: int = 15 # нижняя граница числа точек скана
    ts_bond_scan_max_steps: int = 50 # верхняя граница
    fixed_double_bonds: List[Tuple[int, int]] = field(default_factory=list)  # 1-индексация, как в исходном .mol
    extra_constraints: List[Dict] = field(default_factory=list)
    ts_max_iter: int = 45
    clear_working_folder: bool = False
    checkpoint_step: Union[int, None] = None   # None - default behaviour, else - start at chosen step
    tf_eager_mode: bool = False # Управляет tf.config.run_functions_eagerly() - отладка shape-ошибок, замедление расчетов acquisition
    bond_stretch_factor: float = 1.4  # порог для _check_bond_topology_intact
    preopt_max_iter: int = 100

    def __post_init__(self):
        # Форматировать в список кортежей двойных связей
        self.ts_bonds = [tuple(int(x) for x in pair) for pair in self.ts_bonds]
        self.fixed_double_bonds = [tuple(int(x) for x in pair) for pair in self.fixed_double_bonds]
        for name in ("ts_bonds", "fixed_double_bonds"):
            for pair in getattr(self, name):
                if len(pair) != 2:
                    raise ValueError(f"{name} entries must be [atom_a, atom_b] pairs, got {pair}")
        
        if self.ts_bond_min_length <= 0 or self.ts_bond_min_length >= self.ts_bond_max_length:
            raise ValueError(
                f"ts_bond_min_length ({self.ts_bond_min_length}) must be > 0 and < "
                f"ts_bond_max_length ({self.ts_bond_max_length})."
            )

        if self.ts_bond_scan_step <= 0:
            raise ValueError(f"ts_bond_scan_step ({self.ts_bond_scan_step}) must be > 0.")
        if self.ts_bond_scan_min_steps < 4:
            # calc_bond_coefs fits a 4-parameter Morse potential (De, a, re, c) —
            # fewer than 4 points makes curve_fit underdetermined.
            raise ValueError(
                f"ts_bond_scan_min_steps ({self.ts_bond_scan_min_steps}) must be "
                f">= 4 (calc_bond_coefs fits 4 Morse parameters)."
            )
        if self.ts_bond_scan_min_steps > self.ts_bond_scan_max_steps:
            raise ValueError(
                f"ts_bond_scan_min_steps ({self.ts_bond_scan_min_steps}) must be "
                f"<= ts_bond_scan_max_steps ({self.ts_bond_scan_max_steps})."
            )

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
