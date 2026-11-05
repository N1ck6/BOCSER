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
    exclude_dof_slack: bool = False # True => keep (N - 6) torsion axes in the GP
    scan_ring_torsions: bool = True
    ik_loss_weight: float = 1.0  # acquisition "ik" = IV - ik_loss_weight * IK loss (rings whose torsions are all GP axes)
    ts_bond_min_length: float = 1.0
    ts_bond_kernel_lengthscale: float = 0.5 # lengthscale RBF-ядра по измерениям длины TS-связи (единицы Å)
    ts_bond_scan_step: float = 0.1
    ts_bond_scan_min_steps: int = 15 # нижняя граница числа точек скана
    ts_bond_scan_max_steps: int = 50 # верхняя граница
    fixed_double_bonds: List[Tuple[int, int]] = field(default_factory=list)  # 1-индексация, как в исходном .mol
    extra_constraints: List[Dict] = field(default_factory=list)
    ts_max_iter: int = 45
    clear_working_folder: bool = False
    checkpoint_step: Union[int, str, None] = None   # None - default behaviour, N - continue after step N, "last" - after the last saved step (raise max_steps to run more)
    tf_eager_mode: bool = False # Управляет tf.config.run_functions_eagerly() - отладка shape-ошибок, замедление расчетов acquisition
    bond_stretch_factor: float = 1.4  # порог для _check_bond_topology_intact
    preopt_max_iter: int = 100
    infer_ts_bond_lengths_from_ensemble: bool = False
    ts_mode_bond: List[int] = field(default_factory=list)
    ts_bond_result_margin: float = 0.5
    use_preopt_energy_on_opt_failure: bool = True
    constrain_bo_dihedrals_in_final_opt: bool = False
    constrain_final_opt: bool = False # Hold the double bonds and extra_constraints in the Full opt

    # Separate model of P() - the candidate gives a valid energy by structure check and ORKA runs
    use_feasibility_model: bool = True
    feasibility_lengthscale: float = 0.5  # per torsion, in 2*sin(dTheta/2) units (0.5 ~ 30 deg, 1.0 ~ 60 deg)
    feasibility_prior: float = 0.8        # P far from every evaluated point

    build_from_nearest_conformer: bool = True# Not GP torsion axes come from a real low-energy structure.

    dedup_rmsd_threshold: float = 0.125   # heavy-atom RMSD after symmetry-aware alignment, A
    dedup_energy_threshold: float = 0.05  # kcal/mol

    use_mmff_regularization: bool = False
    mmff_max_iter: int = 1000

    #  above
    # the lowest MMFF94 energy of the known structures (load_ensemble conformers, minima found,
    # candidates built so far). Same force field and same constraints-free energy on both sides.
    
    mmff_energy_window: Union[float, None] = 30.0 # Reject a candidate if MMFF94 energy exceeds this (kcal/mol); null = no filter.

    candidate_batch_size: int = 5 # Amount of candidates, taken from GP on each step

    # Optional: feed more frames of each ORCA trajectory into the GP (thinned in BO coordinates).
    augment_gp_from_orca_trj: bool = False
    orca_trj_max_gp_points: int = 12
    orca_trj_min_dihedral_separation: float = 0.10  # radians
    orca_trj_energy_window: float = 15.0  # kcal/mol above the trajectory minimum

    def __post_init__(self):
        # orca_method: first line = ORCA keywords (a leading "!" is optional), next lines
        # (e.g. a "%scf ... end" block) are copied into every input as they are
        lines = [line.rstrip() for line in str(self.orca_method).strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("orca_method must not be empty.")
        lines[0] = lines[0].strip().removeprefix("!").strip()
        if not lines[0]:
            raise ValueError("orca_method: the first line must hold the ORCA keywords (method, basis).")
        self.orca_method = "\n".join(lines)

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
        if self.mmff_energy_window is not None and self.mmff_energy_window <= 0:
            raise ValueError(f"mmff_energy_window ({self.mmff_energy_window}) must be > 0 (null = no filter).")
        if self.candidate_batch_size < 1:
            raise ValueError(f"candidate_batch_size ({self.candidate_batch_size}) must be >= 1.")
        if self.mmff_max_iter < 1:
            raise ValueError(f"mmff_max_iter ({self.mmff_max_iter}) must be >= 1.")
        if self.orca_trj_max_gp_points < 2:
            raise ValueError("orca_trj_max_gp_points must be >= 2 (first and last trajectory frames are always kept).")
        if self.orca_trj_min_dihedral_separation < 0 or self.orca_trj_energy_window < 0:
            raise ValueError("orca_trj_min_dihedral_separation and orca_trj_energy_window must be >= 0.")
        if self.feasibility_lengthscale <= 0:
            raise ValueError(f"feasibility_lengthscale ({self.feasibility_lengthscale}) must be > 0.")
        if not 0.0 < self.feasibility_prior < 1.0:
            raise ValueError(f"feasibility_prior ({self.feasibility_prior}) must be between 0 and 1.")
        if self.ik_loss_weight < 0:
            raise ValueError(f"ik_loss_weight ({self.ik_loss_weight}) must be >= 0.")
        if self.ts_bond_result_margin < 0:
            raise ValueError(f"ts_bond_result_margin ({self.ts_bond_result_margin}) must be >= 0 (0 = no check).")
        self.ts_mode_bond = [int(x) for x in (self.ts_mode_bond or [])]
        if self.ts_mode_bond and len(self.ts_mode_bond) != 2:
            raise ValueError(f"ts_mode_bond must be an [atom_a, atom_b] pair, got {self.ts_mode_bond}")

        normalized = []
        for i, raw in enumerate(self.extra_constraints):
            if "type" not in raw or "atoms" not in raw:
                raise ValueError(f"extra_constraints[{i}]: fields 'type' and 'atoms' are required, got {raw}")
            ctype = raw["type"]
            if ctype not in _CONSTRAINT_ATOM_COUNT:
                raise ValueError(f"extra_constraints[{i}]: type must be bond/angle/dihedral, got {ctype!r}")
            atoms = tuple(int(a) for a in raw["atoms"])
            if len(atoms) != _CONSTRAINT_ATOM_COUNT[ctype]:
                raise ValueError(
                    f"extra_constraints[{i}]: type={ctype} needs "
                    f"{_CONSTRAINT_ATOM_COUNT[ctype]} atoms, got {len(atoms)}: {atoms}"
                )
            value = raw.get("value", "current")
            normalized.append({"type": ctype, "atoms": atoms, "value": value})
        self.extra_constraints = normalized
