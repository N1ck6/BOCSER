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
    clash_vdw_scale: float = 0.0  # 0.0 = legacy fixed 0.7 Å threshold for all atom pairs; >0.0 = clash_vdw_scale * (vdw_radius_a + vdw_radius_b) for pairs more than 2 bonds apart (bonded, 1-3 and TS-bond pairs are skipped). Sensible values: ~0.5-0.7
    ts : bool = False
    use_grass : bool = False
    path_to_grass : str = ""
    grass_options : str = ""
    orca_poll_timeout_minutes: int = 60
    sbatch_template_name: str = "sbatch_temp"
    rolling_window_size : int = 5
    rolling_std_threshold : float = 0.15
    rolling_mean_threshold : float = 1.
    num_initial_points : int = 3
    max_steps : int = 50  # counts only BO steps whose candidate passed the structure check and went to ORCA
    max_geometry_rejections_in_a_row: int = 50  # stop the run after this many uncounted (rejected) candidates in a row; 0 = no limit
    exp_name : str = "cs"
    load_ensemble : Union[str, None] = None
    acquisition_function : str = "evm"  # ei | evm | ik
    ts_bonds: List[Tuple[int, int]] = field(default_factory=list)   # 1-индексация, как в исходном .mol
    ts_bond_max_length: float = 5.0
    vary_ts_bond_lengths: bool = True
    exclude_dof_slack: bool = True # На каждом кольце оставляет (N - 6) торсионных осей для GP. Излишние оси фиксируются на значении из исходной геометрии, разреженно по кольцу
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

    # TS only (ts: true). Fixed TS-bond lengths are held in the ORCA pre-optimization only,
    # never in the final OptTS. Sources: bond entries of extra_constraints on a ts_bonds pair,
    # or (when True) the lengths in the lowest-energy conformer of load_ensemble.
    infer_ts_bond_lengths_from_ensemble: bool = False

    # Full opt failed in ORCA (MaxIter, SCF, abort) after a successful pre-opt: give the GP the
    # pre-opt energy of the point instead of broken_struct_energy. A geometry rejected by the
    # structure checks is always broken. False = old behaviour (penalize the region).
    use_preopt_energy_on_opt_failure: bool = True

    # Also freeze the BO dihedrals in the final Opt/OptTS (they are always frozen in the pre-opt).
    # The result is then a constrained structure, not a free stationary point.
    constrain_bo_dihedrals_in_final_opt: bool = False

    # Final ensemble: two minima are one conformer when BOTH hold (defaults as in CREST)
    dedup_rmsd_threshold: float = 0.125   # heavy-atom RMSD after symmetry-aware alignment, A
    dedup_energy_threshold: float = 0.05  # kcal/mol

    # Optional MMFF94 clean-up of each candidate before the ORCA pre-optimization.
    use_mmff_regularization: bool = False
    mmff_max_iter: int = 1000

    # Optional: feed more frames of each ORCA trajectory into the GP (thinned in BO coordinates).
    augment_gp_from_orca_trj: bool = False
    orca_trj_max_gp_points: int = 12
    orca_trj_min_dihedral_separation: float = 0.10  # radians
    orca_trj_energy_window: float = 15.0  # kcal/mol above the trajectory minimum

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

        if self.dedup_rmsd_threshold < 0 or self.dedup_energy_threshold < 0:
            raise ValueError("dedup_rmsd_threshold and dedup_energy_threshold must be >= 0.")
        if self.mmff_max_iter < 1:
            raise ValueError(f"mmff_max_iter ({self.mmff_max_iter}) must be >= 1.")
        if self.orca_trj_max_gp_points < 2:
            raise ValueError("orca_trj_max_gp_points must be >= 2 (first and last trajectory frames are always kept).")
        if self.orca_trj_min_dihedral_separation < 0 or self.orca_trj_energy_window < 0:
            raise ValueError("orca_trj_min_dihedral_separation and orca_trj_energy_window must be >= 0.")

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
