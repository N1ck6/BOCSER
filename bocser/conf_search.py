"""
Bayesian Optimization for Conformational Search - Refactored Class-Based Orchestrator

This module provides ConfSearchRunner, a class-based orchestrator that encapsulates
all state and behavior for the conformational search workflow.
"""

import os
import stat
import sys


def _redirected_logs() -> dict:
    """Regular files that stdout/stderr are redirected to (`> run.log 2>&1`), captured before
    any import writes to them: (st_dev, st_ino) -> (size at start, opened in append mode).
    With `>>` the size at start is the previous run's log; with `>` the shell has already emptied it."""
    found = {}
    for stream in (sys.stdout, sys.stderr):
        try:
            fd = stream.fileno()
            st = os.fstat(fd)
        except (AttributeError, OSError, ValueError):
            continue
        if not stat.S_ISREG(st.st_mode):
            continue
        try:
            import fcntl
            append = bool(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_APPEND)
        except (ImportError, OSError):
            append = None  # unknown (Windows)
        found[(st.st_dev, st.st_ino)] = (st.st_size, append)
    return found


_OWN_LOGS = _redirected_logs()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="gpflow")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.warning")
from rdkit.Chem import AllChem, rdMolAlign
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import hashlib
import json
import re
import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import trieste
import gpflow
from trieste.data import Dataset
from trieste.space import Box
from trieste.models.gpflow.models import GaussianProcessRegression
from trieste.acquisition.rule import EfficientGlobalOptimization
from trieste.acquisition.function import ExpectedImprovement

from transform_kernel import TransformKernel
from coef_from_grid import (
    pes_tf, pes_tf_grad,
    calc_bond_coefs,
    morse_tf
)
from calc import (
    calc_energy,
    load_last_optimized_structure_xyz_block,
    parse_points_from_trj,
    _qc_calcs_dir,
    _check_rings_intact,
    raw1_to_with_h_canonical,
    with_h_canonical_to_raw1,
    raw1_to_heavy_canonical,
    _heavy_and_h_order,
    HARTRI_TO_KCAL,
    resolve_extra_constraints,
    _validate_extra_constraints_atoms,
    compute_mol_hash,
    build_reference_with_h_mol,
    submit_calc, wait_for_jobs,
)
from run_state import increase_structure_id
import run_state
import config_manager
from config_loader import ConfigError
from coef_calc import (
    CoefCalculator,
    log_and_combine_double_bonds,
    parse_bond_scan_results,
    generate_bond_scan_inp,
)
from db_connector import LocalConnector
from ensemble_processor import EnsembleProcessor
from evm import ExplorationalVarianceMinimizer
from default_vals import ConfSearchConfig
from ik_loss import IKLoss
from imp_var_with_ik import ImprovementVarianceWithIK
import time
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import logging
logger = logging.getLogger(__name__)

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)



class PotentialFunction:
    """Wrapper for mean function coefficients used in kernel computations.

    Only reads the first len(mean_func_coefs) columns of any
    input tensor — i.e. only the dihedral-angle dimensions. This is what
    makes it safe to call on the FULL search vector (dihedrals + TS-bond
    lengths, when vary_ts_bond_lengths is enabled): it silently ignores any
    trailing columns rather than needing an explicit active_dims slice. Do
    NOT change the loop to range(X.shape[1]) — that would break the moment
    TS-bond dimensions are appended to the search space.
    """

    def __init__(self, mean_func_coefs) -> None:
        self.mean_func_coefs = mean_func_coefs

    @tf.function
    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        return tf.stack(
            [
                pes_tf(X[:, dim], *self.mean_func_coefs[dim])
                for dim in range(len(self.mean_func_coefs))
            ],
            axis=1,
        )

    @tf.function
    def grad(self, X: tf.Tensor) -> tf.Tensor:
        return tf.stack(
            [
                pes_tf_grad(X[:, dim], *self.mean_func_coefs[dim])
                for dim in range(len(self.mean_func_coefs))
            ],
            axis=1,
        )

class TSBondPotentialFunction:
    """mean function for TransformKernel based on TS bond length measurements
    Based on the Morse potential."""
    def __init__(self, mean_func_coefs, n_dih: int) -> None:
        self.mean_func_coefs = mean_func_coefs
        self.n_dih = n_dih  # offset

    @tf.function
    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        return tf.stack(
            [
                morse_tf(X[:, self.n_dih + dim], *self.mean_func_coefs[dim])
                for dim in range(len(self.mean_func_coefs))
            ],
            axis=1,
        )

@dataclass
class ConfSearchState:
    """Internal state container for ConfSearchRunner."""

    mol_file_name: Optional[str] = None
    exp_name: str = ""
    structures_path: str = ""
    working_folder: str = ""
    db_file: str = ""
    norm_energy: float = 0.0
    dihedral_ids: list = field(default_factory=list)
    global_degrees: list = field(default_factory=list)
    asked_points: list = field(default_factory=list)
    minima: list = field(default_factory=list)
    ensemble_processor: Optional[Any] = None
    broken_structs_path: str = ""
    success_out_dir: str = ""
    model_chk: Optional[Any] = None
    current_minima: float = 1e9
    acq_vals_log: list = field(default_factory=list)
    deepest_minima: list = field(default_factory=list)
    config_path: str = ""
    last_opt_ok: bool = True
    ik_loss: Optional[Any] = None
    ik_loss_dihedrals_idxs: list = field(default_factory=list)
    mean_func_coefs: list = field(default_factory=list)
    search_dim: int = 0
    mol: Optional[Chem.Mol] = None
    config: Optional[ConfSearchConfig] = None
    extra_constraints: list = field(default_factory=list)
    vary_ts_bond_lengths: bool = False
    fixed_ts_bond_targets: list = field(default_factory=list)  # [((a, b), length)], pre-opt only
    _dihedral_ids_finalized: bool = False


def _is_broken(energy: float, broken_ref: float, tol: float = 5.0) -> bool:
    """Return True if energy is within tol of broken_struct_energy sentinel."""
    return abs(energy - broken_ref) <= tol

def _print_progress(step: int, max_steps: int, best_energy: float, n_points: int, n_broken: int) -> None:
    """Visible one-line progress print to stdout, independent of the
    logging configuration (which may be routed only to a file)."""
    print(
        f"[BOCSER] step {step:>4}/{max_steps}  "
        f"best_energy={best_energy:>10.3f} kcal/mol  "
        f"points={n_points:>4}  broken={n_broken:>3}",
        flush=True,
    )


class ConfSearchRunner:
    """
    Orchestrator for Bayesian Optimization-based conformational search.

    Encapsulates all state and workflow logic, eliminating module-level globals.
    """

    def __init__(self, working_folder: str = ".", db_file: Optional[str] = None):
        """
        Initialize the conformational search runner.
        
        Args:
            working_folder: Directory where config, input files are read from
                           and output files are written to. Defaults to current directory.
            db_file: Path to dihedral_logs.db database file. If not provided, defaults to
                    the parent directory of the bocser module (../dihedral_logs.db).
        """
        self.state = ConfSearchState()
        self.state.working_folder = str(Path(working_folder).resolve())
        
        # Set database file path
        if db_file is None:
            # Default: one directory up from bocser folder
            bocser_dir = Path(__file__).resolve().parent
            parent_dir = bocser_dir.parent
            db_file = str(parent_dir / "dihedral_logs.db")
        
        self.state.db_file = db_file
        
        # Ensure working folder exists
        Path(working_folder).mkdir(parents=True, exist_ok=True)

    _RING_DOF_SLACK = 6

    def _require_dihedral_ids_finalized(self, caller: str) -> None:
        """Guard against setup() ordering regressions. Call this at the top of
        any block that reads self.state.dihedral_ids' final length (n_dihedral,
        search_dim, IK-index validation, TS-bond Morse scans). Fails loudly
        with a clear message instead of silently corrupting search_dim downstream."""
        if not self.state._dihedral_ids_finalized:
            raise RuntimeError(
                f"{caller}: self.state.dihedral_ids is not finalized yet. "
                f"This must run AFTER the coef_matrix()/dihedral_ids-populate loop in setup()."
            )

    def _apply_ring_dof_cap(
        self,
        dihedral_list_all: list,
        ik_loss_dihedrals_idxs: list,
    ) -> None:
        """Cap the number of GP-searched (independently sampled) torsion axes
        per ring at (n_flexible_positions_in_ring - _RING_DOF_SLACK)

        Ring-fusion axes are never demoted: they encode the relative orientation of two rings.
        A ring whose optional-axis count does not exceed its cap is left completely untouched.
        """
        idx_ring_membership: dict = {}
        for ring_i, cycle_idx in enumerate(ik_loss_dihedrals_idxs):
            for idx in set(cycle_idx):
                if idx >= 0:
                    idx_ring_membership.setdefault(idx, set()).add(ring_i)
        shared_axes = {idx for idx, rings in idx_ring_membership.items() if len(rings) > 1}

        drop_idx_to_window: dict = {}

        for ring_i, (cycle_d, cycle_idx) in enumerate(zip(dihedral_list_all, ik_loss_dihedrals_idxs)):
            n_flexible = sum(1 for idx in cycle_idx if idx != -2)
            cap = max(0, n_flexible - self._RING_DOF_SLACK)

            seen_in_ring: set = set()
            shared_in_ring = []
            optional_in_ring = []
            for d, idx in zip(cycle_d, cycle_idx):
                if idx < 0 or idx in seen_in_ring:
                    continue
                seen_in_ring.add(idx)
                if idx in shared_axes:
                    shared_in_ring.append(idx)
                else:
                    optional_in_ring.append((idx, d))

            total_in_ring = len(shared_in_ring) + len(optional_in_ring)
            if total_in_ring <= cap:
                logger.info(
                    "Ring DOF cap: ring #%d (size=%d) has %d GP-searched axes "
                    "(%d shared, %d optional), within cap=%d (n_flexible=%d - "
                    "slack=%d) — no axes demoted.",
                    ring_i, len(cycle_idx), total_in_ring, len(shared_in_ring),
                    len(optional_in_ring), cap, n_flexible, self._RING_DOF_SLACK,
                )
                continue

            budget_for_optional = max(0, cap - len(shared_in_ring))
            n_opt = len(optional_in_ring)
            if budget_for_optional <= 0:
                keep_positions: set = set()
            elif budget_for_optional >= n_opt:
                keep_positions = set(range(n_opt))
            else:
                positions = np.linspace(0, n_opt - 1, budget_for_optional)
                keep_positions = {int(round(p)) for p in positions}
                pos = 0
                while len(keep_positions) < budget_for_optional and pos < n_opt:
                    keep_positions.add(pos)
                    pos += 1

            keep = [optional_in_ring[p] for p in sorted(keep_positions)]
            drop = [optional_in_ring[p] for p in range(n_opt) if p not in keep_positions]

            logger.info(
                "Ring DOF cap: ring #%d (size=%d) has %d GP-searched axes "
                "(%d shared, %d optional) > cap=%d (n_flexible=%d - slack=%d) "
                "— demoting %d optional axis(es), evenly spaced around the "
                "ring, to fixed (original-geometry) torsions: %s. Kept %d "
                "optional axis(es): %s. Shared (ring-fusion) axes are never "
                "demoted: %s.",
                ring_i, len(cycle_idx), total_in_ring, len(shared_in_ring),
                len(optional_in_ring), cap, n_flexible, self._RING_DOF_SLACK,
                len(drop), [idx for idx, _ in drop],
                len(keep), [idx for idx, _ in keep],
                shared_in_ring,
            )
            for idx, d in drop:
                drop_idx_to_window[idx] = d

        if not drop_idx_to_window:
            logger.info("Ring DOF cap: no rings exceeded their cap — dihedral_ids unchanged.")
            return

        for idx, d in drop_idx_to_window.items():
            val = -Chem.rdMolTransforms.GetDihedralRad(self.state.mol.GetConformer(), *d)
            self.state.fixed_dihedrals.append((list(d), val))

        orig_len = len(self.state.dihedral_ids)
        old_to_new: dict = {}
        new_dihedral_ids = []
        new_mean_func_coefs = []
        for old_idx, (ids, coefs) in enumerate(zip(self.state.dihedral_ids, self.state.mean_func_coefs)):
            if old_idx in drop_idx_to_window:
                continue
            old_to_new[old_idx] = len(new_dihedral_ids)
            new_dihedral_ids.append(ids)
            new_mean_func_coefs.append(coefs)

        self.state.dihedral_ids = new_dihedral_ids
        self.state.mean_func_coefs = new_mean_func_coefs

        # Rewrite EVERY reference across all rings
        for cycle_idx in ik_loss_dihedrals_idxs:
            for pos in range(len(cycle_idx)):
                idx = cycle_idx[pos]
                if idx in drop_idx_to_window:
                    cycle_idx[pos] = -4  # demoted: left unconstrained, IK uses static reference
                elif idx >= 0:
                    cycle_idx[pos] = old_to_new[idx]

        self.state.ik_loss_dihedrals_idxs = ik_loss_dihedrals_idxs

        logger.info(
            "Ring DOF cap: demoted %d axis(es) in total. dihedral_ids reduced "
            "from %d to %d entries.",
            len(drop_idx_to_window), orig_len, len(new_dihedral_ids),
        )

    def _dump_status_hook(self, dumping_value: bool, filename: Optional[str] = None) -> None:
        """Save optimization status to JSON file."""
        if filename is None:
            filename = Path(self.state.working_folder) / f"{self.state.exp_name}_last_opt_status.json"
        filename = Path(filename)
        filename.write_text(json.dumps({"LAST_OPT_OK": dumping_value}))

    def _calc_point(self, dihedrals: list[float]) -> float:
        """Performs energy calculation for given dihedral angles.
        Uses full vector for BO search: first
        len(self.state.dihedral_ids) dihedral angles,
        then — target lengths for TS-bonds."""

        if self.state.model_chk:
            logger.info("Checkpoint is not null, calculating previous acq. func. max!")
            dihedrals_tf = tf.constant(dihedrals, dtype=tf.float64)
            if len(dihedrals_tf.shape) == 1:
                dihedrals_tf = tf.reshape(dihedrals_tf, [1, dihedrals_tf.shape[0]])
            logger.debug("Cur dihedrals_tf: %s", dihedrals_tf)
            logger.debug("Current minima: %s", self.state.current_minima)
            mean, variance = self.state.model_chk.predict_f(dihedrals_tf)
            normal = tfp.distributions.Normal(mean, tf.sqrt(variance))
            tau = self.state.current_minima + 3.0
            acq_val = (
                normal.cdf(tau) * (((tau - mean) ** 2) * (1 - normal.cdf(tau)) + variance)
                + tf.sqrt(variance) * normal.prob(tau) * (tau - mean) * (1 - 2 * normal.cdf(tau))
                - variance * (normal.prob(tau) ** 2)
            )
            self.state.acq_vals_log.append(acq_val.numpy().flatten()[0])

        if tf.is_tensor(dihedrals):
            dihedrals = list(dihedrals.numpy())

        self.state.asked_points.append(dihedrals)

        logger.debug("Point: %s", dihedrals)

        # Pre-opt
        n_dih = len(self.state.dihedral_ids)
        dihedral_vals = dihedrals[:n_dih]
        # TS-bond lengths held in the pre-optimization only: sampled by BO, or fixed
        # (user values / lowest-energy ensemble conformer, see _resolve_fixed_ts_bond_targets)
        if self.state.vary_ts_bond_lengths:
            ts_bond_targets = list(zip(self.state.ts_bonds, dihedrals[n_dih:]))
        elif self.state.fixed_ts_bond_targets:
            ts_bond_targets = list(self.state.fixed_ts_bond_targets)
        else:
            ts_bond_targets = None

        logger.info("Optimizing constrained struct")
        try:
            en, preopt_status = calc_energy(
                self.state.mol_file_name,
                list(zip(self.state.dihedral_ids, dihedral_vals)),
                self.state.norm_energy,
                True,
                constrained_opt=True,
                ik_loss=self.state.ik_loss,
                original_mol=self.state.mol,
                broken_structs_dir=self.state.broken_structs_path,
                success_out_dir=self.state.success_out_dir,
                ts_bonds=self.state.ts_bonds,
                ts_bond_max_length=self.state.config.ts_bond_max_length,
                fixed_dihedrals=self.state.fixed_dihedrals,
                extra_constraints=self.state.extra_constraints,
                ts_bond_targets=ts_bond_targets,
            )
        except Exception:
            logger.exception("calc_energy (preopt) fell with unexpected exception — conting point as broken.")
            en, preopt_status = self.state.config.broken_struct_energy, False
            # Dump the status file immediately, right to avoidcrash on exception
            self._dump_status_hook(dumping_value=False)
        self.state.last_opt_ok = preopt_status
        logger.info("Status of preopt: %s; LAST_OPT_OK: %s", preopt_status, self.state.last_opt_ok)
        if not preopt_status:
            self._dump_status_hook(dumping_value=self.state.last_opt_ok)
            skipped_structure_id = increase_structure_id()
            logger.error("Preopt finished with error! Structure with number %s will be skipped!", skipped_structure_id)
            return en + np.random.randn()
        preopt_en = en
        full_opt_failure = {}
        try:
            logger.info("Optimized! Loading xyz from preopt")
            xyz_from_constrained = load_last_optimized_structure_xyz_block(self.state.mol_file_name)
            logger.info("Loaded! Full opt")
            en, opt_status = calc_energy(
                self.state.mol_file_name,
                list(zip(self.state.dihedral_ids, dihedral_vals)),
                self.state.norm_energy,
                True,
                force_xyz_block=xyz_from_constrained,
                ik_loss=self.state.ik_loss,
                original_mol=self.state.mol,
                broken_structs_dir=self.state.broken_structs_path,
                success_out_dir=self.state.success_out_dir,
                ts_bonds=self.state.ts_bonds,
                ts_bond_max_length=self.state.config.ts_bond_max_length,
                fixed_dihedrals=self.state.fixed_dihedrals,
                extra_constraints=self.state.extra_constraints,
                ts_bond_targets=None,  # TS-bond lengths are free in the final Opt/OptTS
                failure_info=full_opt_failure,
            )
        except Exception:
            # e.g. sbatch -W reports a non-zero ORCA exit code: lose this point, not the whole run
            logger.exception("calc_energy (full opt) fell with unexpected exception — counting point as broken.")
            en, opt_status = self.state.config.broken_struct_energy, False
        self.state.last_opt_ok = opt_status
        logger.info("Status of opt: %s; LAST_OPT_OK: %s", opt_status, self.state.last_opt_ok)
        logger.info("Optimized! En = %s", en)
        self._dump_status_hook(dumping_value=self.state.last_opt_ok)

        if not opt_status:
            skipped_structure_id = increase_structure_id()
            logger.error("Opt finished with error! Structure with number %s will be skipped!", skipped_structure_id)
            if self.state.config.use_preopt_energy_on_opt_failure and full_opt_failure.get("stage") == "orca":
                # The pre-opt energy at this point is a valid observation (it is what the first
                # trajectory frame gives on success); only a rejected geometry is not.
                logger.warning(
                    "Full optimization failed in ORCA — using the pre-optimization energy %.3f kcal/mol "
                    "for this point instead of broken_struct_energy.", preopt_en,
                )
                return preopt_en

        return en + ((not opt_status) * np.random.randn())

    def _func_objective(self, cur: tf.Tensor) -> tf.Tensor:
        """Defines the objective function for the Bayesian optimization."""
        cur_np = cur.numpy() if hasattr(cur, "numpy") else np.asarray(cur)
        results = [[self._calc_point(x)] for x in cur_np]
        return tf.constant(results, dtype=tf.float64)

    def _extract_dofs_values(self, m: Chem.Mol) -> tf.Tensor:
        """Extract dihedral angles from a molecule conformer."""
        conf = m.GetConformer()
        dihedral_vals = [
            -Chem.rdMolTransforms.GetDihedralRad(conf, *self.state.dihedral_ids[i])
            for i in range(len(self.state.dihedral_ids))
        ]

        ts_bond_vals = []
        if self.state.vary_ts_bond_lengths:
            lo = self.state.config.ts_bond_min_length
            hi = self.state.config.ts_bond_max_length
            for (a, b) in self.state.ts_bonds:
                raw_len = Chem.rdMolTransforms.GetBondLength(conf, a, b)
                # Length in random ETKDG-embedding is random. Clip in box,
                # so starting point will be valid for GP/trieste.
                clipped = float(np.clip(raw_len, lo, hi))
                if abs(clipped - raw_len) > 1e-6:
                    logger.debug(
                        "Initial embedded TS-bond length for pair (%d, %d) was "
                        "%.3f A, clipped to %.3f A to fit search bounds [%.3f, %.3f].",
                        a, b, raw_len, clipped, lo, hi,
                    )
                ts_bond_vals.append(clipped)

        return tf.constant([dihedral_vals + ts_bond_vals], dtype=tf.float64)

    def _upd_dataset_from_trj(self, trj_filename: str, dataset: Optional[Dataset]) -> Dataset:
        """Update the GP dataset with observations from an ORCA trajectory.

        Default: first frame + cluster representatives. With augment_gp_from_orca_trj,
        more frames are kept but thinned in BO coordinates (consecutive optimization
        steps are strongly correlated, not independent observations).
        """
        logger.debug("Input dataset is: %s", dataset)
        cfg = self.state.config
        augment = bool(cfg.augment_gp_from_orca_trj)
        # Points must have the same dimensions as the search space (TS-bond lengths too)
        ts_pairs = self.state.ts_bonds if self.state.vary_ts_bond_lengths else []
        parsed_data, last_point = parse_points_from_trj(
            trj_file_name=trj_filename,
            dihedrals=self.state.dihedral_ids,
            norm_en=self.state.norm_energy,
            save_structs=not augment,
            structures_path=self.state.structures_path,
            return_minima=True,
            extra_distance_pairs=ts_pairs,
            distance_bounds=(cfg.ts_bond_min_length, cfg.ts_bond_max_length),
            return_all_frames=augment,
        )
        if augment:
            parsed_data = self._select_trj_training_frames(parsed_data)
            self._save_structures([xyz for _, _, xyz in parsed_data])
        # Filter out frames where rings opened during ORCA optimization
        valid_degrees, valid_energies = [], []
        for coords, energy, xyz_block in parsed_data:
            if self.state.ik_loss is not None:
                rings_intact, _ = _check_rings_intact(xyz_block, self.state.mol, ts_bonds=self.state.ts_bonds)
                if not rings_intact:
                    logger.warning(
                        "Discarding trajectory frame (energy=%.3f kcal/mol): "
                        "ring opened during optimization",
                        energy
                    )
                    continue
            valid_degrees.append(coords)
            valid_energies.append(energy)

        if not valid_degrees:
            logger.warning(
                "All trajectory frames from %s discarded (rings broken in all frames). "
                "Dataset unchanged.",
                trj_filename
            )
            return dataset

        # Validate last_point before saving to minima list.
        last_xyz = last_point["xyz_block"]
        if self.state.ik_loss is None or _check_rings_intact(last_xyz, self.state.mol, ts_bonds=self.state.ts_bonds)[0]:
            minima_file = os.path.join(
                self.state.working_folder,
                f"{self.state.exp_name}_minima/{len(self.state.minima)}.xyz"
            )
            with open(minima_file, "w") as minima_xyz_writer:
                minima_xyz_writer.write(last_xyz)
            self.state.minima.append((last_point["coords"], last_point["rel_en"]))
        else:
            logger.warning(
                "Last trajectory point (energy=%.3f kcal/mol) discarded: "
                "ring opened during optimization. Not added to minima.",
                last_point["rel_en"]
            )

        logger.debug("Valid frames: %d / %d", len(valid_degrees), len(parsed_data))
        logger.debug("Degrees: %s\nEnergies: %s", valid_degrees, valid_energies)

        self.state.global_degrees.extend(valid_degrees)

        add_part = Dataset(
            tf.constant(valid_degrees, dtype="double"),
            tf.constant(valid_energies, dtype="double").reshape(len(valid_energies), 1),
        )

        return add_part if not dataset else dataset + add_part

    def _save_structures(self, xyz_blocks: list) -> None:
        """Save trajectory frames used for the GP as {exp_name}/<structure id>.xyz."""
        for xyz in xyz_blocks:
            path = Path(self.state.structures_path) / f"{run_state.peek_structure_id()}.xyz"
            path.write_text(xyz)
            increase_structure_id()

    def _select_trj_training_frames(self, parsed_data: list) -> list:
        """Pick informative frames of one ORCA trajectory for the GP (augment_gp_from_orca_trj).

        Always keeps the first and the last frame; then greedily adds the frame farthest
        (in BO coordinates) from the frames already chosen, among frames within
        orca_trj_energy_window of the trajectory minimum, until orca_trj_max_gp_points
        or until the farthest one is closer than orca_trj_min_dihedral_separation.
        """
        cfg = self.state.config
        indexed = [(i, x) for i, x in enumerate(parsed_data) if np.isfinite(x[1])]
        if len(indexed) <= 2:
            return [x for _, x in indexed]

        n_dih = len(self.state.dihedral_ids)
        ts_scale = max(float(cfg.ts_bond_kernel_lengthscale), 1e-6)

        def distance(a, b):
            a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
            delta = np.abs(a[:n_dih] - b[:n_dih])
            delta = np.minimum(delta, 2.0 * np.pi - delta)
            d2 = float(np.sum(delta ** 2)) + float(np.sum(((a[n_dih:] - b[n_dih:]) / ts_scale) ** 2))
            return float(np.sqrt(d2))

        selected = [indexed[0], indexed[-1]]
        e_min = min(x[1] for _, x in indexed)
        candidates = [(i, x) for i, x in indexed[1:-1] if x[1] <= e_min + cfg.orca_trj_energy_window]
        while candidates and len(selected) < cfg.orca_trj_max_gp_points:
            gaps = [min(distance(x[0], s[0]) for _, s in selected) for _, x in candidates]
            best = int(np.argmax(gaps))
            if gaps[best] < cfg.orca_trj_min_dihedral_separation:
                break  # every remaining frame is a near-duplicate of a chosen one
            selected.append(candidates.pop(best))
        keep = {i for i, _ in selected}
        logger.info("Trajectory augmentation: %d of %d frames used for the GP", len(keep), len(parsed_data))
        return [x for i, x in indexed if i in keep]

    def _erase_last_from_dataset(self, dataset: Dataset, n: int = 1) -> Dataset:
        """Remove last n points from dataset."""
        query_points = tf.slice(
            dataset.query_points,
            [0, 0],
            [dataset.query_points.shape[0] - n, dataset.query_points.shape[1]],
        )
        observations = tf.slice(
            dataset.observations,
            [0, 0],
            [dataset.observations.shape[0] - n, dataset.observations.shape[1]],
        )
        return Dataset(query_points, observations)

    def load_config(self, config_path: str) -> None:
        """Load and validate configuration from file.
        
        Args:
            config_path: Path to config file. Can be relative to working_folder or absolute.
        """
        from config_manager import load_config, ConfigError

        if not os.path.isabs(config_path):
            config_path = str(Path(config_path).resolve())

        try:
            config = load_config(config_path)
        except FileNotFoundError:
            logger.error("No config file %s! Finishing!", config_path)
            raise
        except ConfigError as e:
            logger.error("Config error: %s. Finishing!", e)
            raise
        except Exception:
            logger.exception("Something went wrong while loading config! Finishing!")
            raise

        self.state.config = config
        self.state.config_path = config_path

    def setup(self) -> None:
        """Initialize the search from configuration."""
        if not self.state.config:
            raise RuntimeError("Config not loaded. Call load_config first.")
        config = self.state.config

        tf.config.run_functions_eagerly(config.tf_eager_mode)
        logger.info("tf.config.run_functions_eagerly(%s)", config.tf_eager_mode)
        
        if config.clear_working_folder:
            self._clear_working_folder()

        logger.info("Performing conf. search with config: %s", config)

        # Propagate configuration to central config manager and internal state
        config_manager.set_config(config)
        
        mol_file = Path(config.mol_file_name)
        if not mol_file.is_absolute():
            mol_file = mol_file.resolve()
        self.state.mol_file_name = str(mol_file)

        _validate_extra_constraints_atoms(config)
        self.state.extra_constraints = resolve_extra_constraints(config.extra_constraints, self.state.mol_file_name)
        if self.state.extra_constraints:
            logger.info("Загружено %d пользовательских extra_constraints", len(self.state.extra_constraints))

        self.state.exp_name = config.exp_name
        self.state.structures_path = str(Path(self.state.working_folder) / f"{config.exp_name}/")

        Path(self.state.structures_path).mkdir(parents=True, exist_ok=True)
        minima_path = Path(self.state.working_folder) / f"{config.exp_name}_minima/"
        minima_path.mkdir(parents=True, exist_ok=True)

        broken_path = Path(self.state.working_folder) / f"{config.exp_name}_broken/"
        broken_path.mkdir(parents=True, exist_ok=True)
        self.state.broken_structs_path = str(broken_path)

        success_path = Path(self.state.working_folder) / f"{config.exp_name}_success/"
        success_path.mkdir(parents=True, exist_ok=True)
        self.state.success_out_dir = str(success_path)

        if config.acquisition_function not in {"ei", "evm", "ik"}:
            logger.warning(
                "Acquisition function should be one of the following: 'ei', 'evm', 'ik'; got %s; Continue with default: 'evm'",
                config.acquisition_function,
            )
            config.acquisition_function = "evm"

        self.state.ts_bonds = [
            (raw1_to_with_h_canonical(a, self.state.mol_file_name),
            raw1_to_with_h_canonical(b, self.state.mol_file_name))
            for a, b in config.ts_bonds
        ]
        if self.state.ts_bonds:
            logger.info("TS-связи (config 1-idx -> canonical 0-idx): %s -> %s", config.ts_bonds, self.state.ts_bonds)

        # Must run before vary_ts_bond_lengths is read below (it may be switched off here)
        self.state.fixed_ts_bond_targets = self._resolve_fixed_ts_bond_targets(config)

        logger.info("Coef calculator creating")

        self.state.mol = Chem.RemoveHs(Chem.MolFromMolFile(self.state.mol_file_name))
        
        scans_dir = str(Path(self.state.working_folder) / f"{self.state.exp_name}_scans/")

        user_double_bonds = [
            (raw1_to_heavy_canonical(a, self.state.mol_file_name),
            raw1_to_heavy_canonical(b, self.state.mol_file_name))
            for a, b in config.fixed_double_bonds
        ]
        fixed_double_bonds = log_and_combine_double_bonds(self.state.mol, user_double_bonds)
        
        ref_with_h_mol = build_reference_with_h_mol(self.state.mol_file_name)
        ts_bonds_heavy_canonical = set()
        for a, b in config.ts_bonds:
            sym_a = ref_with_h_mol.GetAtomWithIdx(a - 1).GetSymbol()
            sym_b = ref_with_h_mol.GetAtomWithIdx(b - 1).GetSymbol()
            if sym_a == 'H' or sym_b == 'H':
                logger.info(
                    "ts_bonds pair (%d, %d) involves a hydrogen atom — no "
                    "CoefCalculator exclusion mapping needed (H atoms are "
                    "always terminal, so this pair could never be picked up "
                    "as a rotatable-dihedral candidate anyway).",
                    a, b,
                )
                continue
            heavy_a = raw1_to_heavy_canonical(a, self.state.mol_file_name)
            heavy_b = raw1_to_heavy_canonical(b, self.state.mol_file_name)
            
            with_h_a = raw1_to_with_h_canonical(a, self.state.mol_file_name)
            with_h_b = raw1_to_with_h_canonical(b, self.state.mol_file_name)
            if (heavy_a, heavy_b) != (with_h_a, with_h_b):
                raise RuntimeError(
                    f"Atom numbering mismatch for ts_bonds pair (raw 1-idx "
                    f"{a},{b}): heavy-only canonical = ({heavy_a},{heavy_b}), "
                    f"with-H canonical = ({with_h_a},{with_h_b}). These two "
                    f"numbering schemes are assumed to coincide for heavy "
                    f"atoms throughout the codebase — this assumption has "
                    f"just been violated. Do not proceed: CoefCalculator "
                    f"and calc.py would silently operate on different atoms "
                    f"for what should be the same TS bond."
                )
            ts_bonds_heavy_canonical.add((heavy_a, heavy_b))

        coef_calc = CoefCalculator(
            mol=self.state.mol,
            config=config,
            dir_for_inps=scans_dir,
            db_connector=LocalConnector(self.state.db_file),
            fixed_double_bonds=fixed_double_bonds,
            ts_bonds=ts_bonds_heavy_canonical,
        )

        logger.info("Coef calculator created!")
        
        try:
            dihedral_list_all, ring_atoms_list, ik_loss_dihedrals_idxs = coef_calc.get_ring_dihedrals(
                self.state.mol
            )

            self.state.fixed_dihedrals = []
            for cycle_d, cycle_idx in zip(dihedral_list_all, ik_loss_dihedrals_idxs):
                for d, idx in zip(cycle_d, cycle_idx):
                    if idx == -2:
                        val = -Chem.rdMolTransforms.GetDihedralRad(self.state.mol.GetConformer(), *d)
                        self.state.fixed_dihedrals.append((list(d), val))
            if self.state.fixed_dihedrals:
                logger.info("Fixed torsions (at double bonds): %s", self.state.fixed_dihedrals)

            if ik_loss_dihedrals_idxs:
                self.state.ik_loss = IKLoss.from_rdkit(self.state.mol, ring_atoms_list)
                self.state.ik_loss_dihedrals_idxs = ik_loss_dihedrals_idxs
                logger.info("IK loss prepared. IK dihedral indices: %s", ik_loss_dihedrals_idxs)
            else:
                self.state.ik_loss = None
                self.state.ik_loss_dihedrals_idxs = []
                logger.warning("No ring dihedrals detected; IK acquisition will be unavailable.")
        except Exception as e:
            self.state.ik_loss = None
            self.state.ik_loss_dihedrals_idxs = []
            ik_loss_dihedrals_idxs = []
            dihedral_list_all = []
            logger.exception(
                "Failed to prepare IK loss: %s. "
                "Possible cause: unsupported ring topology. Falling back to evm.", e
            )
        
        coef_matrix = coef_calc.coef_matrix()

        for ids, coefs in coef_matrix:
            central_axis = frozenset((ids[1], ids[2]))
            if central_axis in fixed_double_bonds:
                raise RuntimeError(
                    f"Axis {ids} is fixed, but still exists in  "
                    f"coef_matrix()/self.frags — filtration in CoefCalculator is incomplete. "
                )
            self.state.dihedral_ids.append(ids)
            self.state.mean_func_coefs.append(coefs)

        central_bonds_seen: dict = {}
        for ids in self.state.dihedral_ids:
            central_bond = frozenset((ids[1], ids[2]))
            if central_bond in central_bonds_seen:
                raise RuntimeError(
                    f"Duplicate GP axis detected for the same physical bond "
                    f"{tuple(central_bond)}: {central_bonds_seen[central_bond]} "
                    f"and {ids} both ended up in self.state.dihedral_ids. This "
                    f"would produce two independent hard 'D' constraints for "
                    f"the same bond in the Pre-OPT .inp — an overdetermined, "
                    f"usually physically unsatisfiable geometry. This should "
                    f"be impossible given CoefCalculator's internal dedup "
                    f"layers (get_interesting_frags, get_ring_dihedrals "
                    f"axis-reuse, _dedup_frags_by_central_bond, coef_matrix) "
                    f"— one of them has regressed if this fires."
                )
            central_bonds_seen[central_bond] = ids

        if config.exclude_dof_slack:
            self._apply_ring_dof_cap(dihedral_list_all, ik_loss_dihedrals_idxs)

        # Mark dihedral_ids as finalized BEFORE anything downstream is allowed to read it.
        self.state._dihedral_ids_finalized = True

        logger.info("Dihedral ids: %s", self.state.dihedral_ids)
        logger.info("Mean func coefs: %s", self.state.mean_func_coefs)

        self._require_dihedral_ids_finalized("n_dihedral computation")
        n_dihedral = len(self.state.dihedral_ids)

        for cycle_idx in ik_loss_dihedrals_idxs:
            for idx in cycle_idx:
                if idx >= 0 and idx >= n_dihedral:
                    raise RuntimeError(
                        f"IK dihedral index {idx} is out of range n_dihedral={n_dihedral}. "
                        f"ik_loss_dihedrals_idxs={ik_loss_dihedrals_idxs}. "
                        f"Check frag_key_to_position in get_ring_dihedrals() and filter "
                        f"fixed_double_bonds в get_interesting_frags()."
                    )

        if config.vary_ts_bond_lengths and self.state.ts_bonds:
            self.state.vary_ts_bond_lengths = True
            self.state.search_dim = n_dihedral + len(self.state.ts_bonds)
            logger.info(
                "TS-bond length variation ENABLED: %d TS-bond(s) added as extra GP "
                "search dimensions, bounds=[%.3f, %.3f] A, appended AFTER the %d "
                "dihedral dimensions (index layout: [0:%d)=dihedrals, [%d:%d)=ts-bond "
                "lengths). Set vary_ts_bond_lengths: false in config to fall back to "
                "the legacy behaviour (ts_bonds used only for break/clash safety "
                "checks, not searched).",
                len(self.state.ts_bonds), config.ts_bond_min_length, config.ts_bond_max_length,
                n_dihedral, n_dihedral, n_dihedral, self.state.search_dim,
            )
        else:
            self.state.vary_ts_bond_lengths = False
            self.state.search_dim = n_dihedral
            if self.state.ts_bonds:
                logger.info(
                    "vary_ts_bond_lengths=False: %d TS-bond(s) configured but used "
                    "ONLY for legacy safety checks — not added as GP search dimensions.",
                    len(self.state.ts_bonds),
                )

        self._require_dihedral_ids_finalized("TS-bond Morse scan block")
        self.state.ts_bond_mean_coefs = []
        n_ts_cached = 0
        n_ts_computed = 0
        if self.state.vary_ts_bond_lengths:
            theory_level = f"{config.orca_method}|charge={config.charge}|mult={config.spin_multiplicity}"
            mol_hash = compute_mol_hash(self.state.mol_file_name, config.charge, config.spin_multiplicity)
            db = LocalConnector(self.state.db_file)

            pairs_to_scan = []  # [(a, b, pair_key, inp_name), ...]
            cached_coefs = {}   # pair_key -> coefs, чтобы сохранить порядок self.state.ts_bonds

            # mol_hash ignores the atom order of the .mol file, raw indices don't: key the cache by
            # canonical atom ranks + graph distance so another .mol of the same molecule can't
            # pick up the coefficients of a different atom pair
            ref_mol = build_reference_with_h_mol(self.state.mol_file_name)
            ranks = list(Chem.CanonicalRankAtoms(ref_mol, breakTies=False))
            topo_dist = Chem.GetDistanceMatrix(ref_mol)

            def db_pair_key(a: int, b: int) -> str:
                r1, r2 = sorted((ranks[a], ranks[b]))
                return f"rank{r1}-{r2}|d{int(topo_dist[a][b])}"

            for (a, b) in self.state.ts_bonds:
                pair_key = f"{min(a, b)}-{max(a, b)}"
                cached = db.get_ts_bond_coefs(mol_hash, theory_level, db_pair_key(a, b))
                if cached is not None:
                    logger.info("TS-bond Morse coefs cache HIT for pair (%d,%d): %s", a, b, cached)
                    cached_coefs[pair_key] = cached
                    continue

                scans_dir = Path(self.state.working_folder) / f"{self.state.exp_name}_ts_bond_scans"
                scans_dir.mkdir(exist_ok=True)
                inp_name = str(scans_dir / f"tsbond_{pair_key}.inp")

                mol_for_scan = Chem.MolFromMolFile(self.state.mol_file_name, removeHs=False)
                heavy_idx, h_idx = _heavy_and_h_order(mol_for_scan)
                mol_for_scan = Chem.RenumberAtoms(mol_for_scan, heavy_idx + h_idx)
                xyz = "\n".join(Chem.MolToXYZBlock(mol_for_scan).split("\n")[2:])

                # Grid density scales with scan range width instead of a value
                range_width = config.ts_bond_max_length - config.ts_bond_min_length
                nsteps = int(round(range_width / config.ts_bond_scan_step))
                nsteps = max(config.ts_bond_scan_min_steps, min(config.ts_bond_scan_max_steps, nsteps))
                actual_step = range_width / nsteps if nsteps > 0 else range_width
                logger.info(
                    "TS-bond pair (%d,%d): scanning [%.3f, %.3f] A with nsteps=%d "
                    "(target step=%.3f A, actual step=%.3f A).",
                    a, b, config.ts_bond_min_length, config.ts_bond_max_length,
                    nsteps, config.ts_bond_scan_step, actual_step,
                )

                generate_bond_scan_inp(
                    xyz, a + 1, b + 1, inp_name,
                    num_of_procs=config.num_of_procs, method_of_calc=config.orca_method,
                    charge=config.charge, multipl=config.spin_multiplicity,
                    lo=config.ts_bond_min_length, hi=config.ts_bond_max_length,
                    nsteps=nsteps,
                )
                pairs_to_scan.append((a, b, pair_key, inp_name))

            if pairs_to_scan:
                job_ids = [submit_calc(inp_name, scan=True) for (_, _, _, inp_name) in pairs_to_scan]
                logger.info("Submitted %d independent TS-bond length scans concurrently: %s", len(job_ids), job_ids)
                wait_for_jobs(job_ids, timeout_minutes=config.orca_poll_timeout_minutes)

                for (a, b, pair_key, inp_name) in pairs_to_scan:
                    try:
                        lengths, energies = parse_bond_scan_results(inp_name)
                        coefs = calc_bond_coefs(lengths, energies)
                    except (OSError, ValueError, IndexError, RuntimeError, TypeError) as e:
                        # De=0 gives a flat Morse mean; not cached, so the scan is retried next run
                        logger.error(
                            "TS-bond scan %s failed (%s) — using flat mean function for pair (%d,%d).",
                            inp_name, e, a, b,
                        )
                        cached_coefs[pair_key] = (0.0, 1.0, (config.ts_bond_min_length + config.ts_bond_max_length) / 2, 0.0)
                        continue
                    logger.info("TS-bond pair (%d,%d) Morse coefs: De=%.3f a=%.3f re=%.3f c=%.3f", a, b, *coefs)
                    db.set_ts_bond_coefs(mol_hash, theory_level, db_pair_key(a, b), coefs)
                    cached_coefs[pair_key] = tuple(coefs)

            for (a, b) in self.state.ts_bonds:
                pair_key = f"{min(a, b)}-{max(a, b)}"
                self.state.ts_bond_mean_coefs.append(cached_coefs[pair_key])

            n_ts_computed = len(pairs_to_scan)
            n_ts_cached = len(self.state.ts_bonds) - n_ts_computed

        logger.info("Cur search dim is %s", self.state.search_dim)

        n_fourier_cached = getattr(coef_calc, "n_fourier_cached", 0)
        n_fourier_computed = getattr(coef_calc, "n_fourier_computed", 0)
        logger.info(
            "Setup cache summary: torsion Fourier coefs — %d from cache, %d "
            "newly computed; TS-bond Morse coefs — %d from cache, %d newly computed.",
            n_fourier_cached, n_fourier_computed, n_ts_cached, n_ts_computed,
        )

        for cycle_idx in ik_loss_dihedrals_idxs:
            for idx in cycle_idx:
                if idx >= 0 and idx >= self.state.search_dim:
                    raise RuntimeError(
                        f"IK dihedral index {idx} is out of range search_dim={self.state.search_dim}. "
                        f"ik_loss_dihedrals_idxs={ik_loss_dihedrals_idxs}. "
                        f"Check frag_key_to_position in get_ring_dihedrals() and filter "
                        f"fixed_double_bonds в get_interesting_frags()."
                    )

    def _build_model_and_acquisition(self) -> Tuple[Any, Any, Any]:
        """Build GPR model, BO optimizer, and acquisition rule."""
        potential_func = PotentialFunction(self.state.mean_func_coefs)

        n_dih = len(self.state.dihedral_ids)
        n_ts = len(self.state.ts_bonds) if self.state.vary_ts_bond_lengths else 0
        dih_dims = list(range(n_dih))
        ts_dims = list(range(n_dih, n_dih + n_ts))

        kernel = (
            gpflow.kernels.White(0.001)
            + gpflow.kernels.Periodic(
                gpflow.kernels.RBF(
                    variance=0.07,
                    lengthscales=0.005,
                    active_dims=dih_dims,
                ),
                period=[2 * np.pi for _ in dih_dims],
            )
            + TransformKernel(
                potential_func,
                gpflow.kernels.RBF(
                    variance=0.12,
                    lengthscales=0.005,
                    active_dims=dih_dims,
                ),
            )
        )

        kernel.kernels[1].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(
            loc=tf.constant(0.005, dtype=tf.float64), scale=tf.constant(0.001, dtype=tf.float64)
        )
        kernel.kernels[2].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(
            loc=tf.constant(0.005, dtype=tf.float64), scale=tf.constant(0.001, dtype=tf.float64)
        )

        if n_ts > 0:
            ts_potential_func = TSBondPotentialFunction(self.state.ts_bond_mean_coefs, n_dih)
            ts_kernel = TransformKernel(
                ts_potential_func,
                gpflow.kernels.RBF(variance=0.12, lengthscales=0.5, active_dims=ts_dims),
            )
            kernel = kernel + ts_kernel

        lower = [0.0 for _ in range(n_dih)] + [self.state.config.ts_bond_min_length for _ in range(n_ts)]
        upper = [2 * np.pi for _ in range(n_dih)] + [self.state.config.ts_bond_max_length for _ in range(n_ts)]
        search_space = Box(lower, upper)

        config = self.state.config

        # Compute normalizing energy (in kcal/mol)
        if config.load_ensemble:
            ensemble_path = self._resolve_ensemble_path()
            logger.info(
                "load_ensemble was set (%s); skipping ORCA norm_energy calculation on mol, "
                "min energy from ensemble selected instead.",
                ensemble_path,
            )
            self.state.ensemble_processor = self._make_ensemble_processor()
            if not self.state.ensemble_processor.energies:
                raise RuntimeError(
                    f"load_ensemble at {ensemble_path}: no conformers/energies found. "
                    f"norm_energy can't be calculated."
                )
            # BO energies are E_orca * HARTRI_TO_KCAL - norm_energy, so the ensemble must hold absolute
            # kcal/mol energies (< -9e4 for any molecule with a heavy-atom torsion). Hartree or
            # relative values would silently mix scales with every later ORCA result.
            if max(self.state.ensemble_processor.energies) > -1e4:
                raise RuntimeError(
                    f"load_ensemble at {ensemble_path}: energies (max={max(self.state.ensemble_processor.energies):.4f}) "
                    f"are not absolute energies in kcal/mol. Put the absolute energy in kcal/mol "
                    f"(Hartree * {HARTRI_TO_KCAL}) as the last token of each comment line."
                )
            self.state.norm_energy = min(self.state.ensemble_processor.energies)
            logger.info(
                "Norm energy on %d structures: %s",
                len(self.state.ensemble_processor.energies), self.state.norm_energy,
            )
        else:
            theory_level = f"{config.orca_method}|charge={config.charge}|mult={config.spin_multiplicity}"
            # norm_energy comes from an Opt/OptTS of the input geometry under the hard constraints,
            # so both change the cached value
            if config.ts:
                theory_level += "|ts"
            hard_constraints = sorted(
                [("dihedral", tuple(atoms), round(float(np.rad2deg(val)), 1)) for atoms, val in self.state.fixed_dihedrals]
                + [(c.type, tuple(c.atoms), round(float(c.value), 3)) for c in self.state.extra_constraints]
            )
            if hard_constraints:
                theory_level += "|c=" + hashlib.sha1(repr(hard_constraints).encode()).hexdigest()[:12]
            mol_hash = compute_mol_hash(self.state.mol_file_name, config.charge, config.spin_multiplicity)
            db = LocalConnector(self.state.db_file)

            cached = db.get_norm_energy(mol_hash, theory_level)
            if cached is not None:
                logger.info("norm_energy cache (theory=%s): %.6f kcal/mol — skipping ORCA calc.",
                            theory_level, cached)
                self.state.norm_energy = cached
            else:
                self.state.norm_energy, ok = calc_energy(
                    self.state.mol_file_name, dihedrals=[], norm_energy=0.0, ik_loss=self.state.ik_loss,
                    original_mol=self.state.mol, broken_structs_dir=self.state.broken_structs_path,
                    success_out_dir=self.state.success_out_dir, ts_bonds=self.state.ts_bonds,
                    ts_bond_max_length=self.state.config.ts_bond_max_length,
                    fixed_dihedrals=self.state.fixed_dihedrals, extra_constraints=self.state.extra_constraints,
                )
                logger.info("Norm energy: %s", self.state.norm_energy)
                if not ok:
                    raise RuntimeError(
                        f"Initial geometry of {self.state.mol_file_name} failed energy calculation "
                        f"(norm_energy={self.state.norm_energy}). Check the log immediately above this "
                        f"error for the specific cause (atom clash, ring-opened, or ORCA optimization "
                        f"failure) — increasing thresholds will not help if the geometry itself is being "
                        f"altered before ORCA ever runs."
                    )
                db.set_norm_energy(mol_hash, theory_level, self.state.norm_energy,
                            source_mol_file=self.state.mol_file_name)
                logger.info("norm_energy cached (theory=%s).", theory_level)

        observer = trieste.objectives.utils.mk_observer(self._func_objective)

        return kernel, search_space, observer

    def _initialize_dataset(self, observer: Any) -> Dataset:
        """Build initial dataset from ensemble or random points."""
        config = self.state.config
        dataset = None

        if config.load_ensemble:
            ep = self.state.ensemble_processor
            if ep is None:
                logger.warning(
                    "ensemble_processor was not created ahead — checking load_ensemble again."
                )
                ep = self._make_ensemble_processor()
            
            logger.info(
                "Loading init points from given ensemble! Normalizing against norm_energy=%.3f kcal/mol.",
                self.state.norm_energy,
            )
            ep.normalize_energy(self.state.norm_energy)
            dataset = Dataset(*ep.get_tf_data())

            obs = dataset.observations.numpy().flatten()
            obs_range = obs.max() - obs.min()
            obs_mean = obs.mean()
            logger.info(
                "Dataset energy range check: min=%.2f, max=%.2f, range=%.2f, mean=%.2f kcal/mol",
                obs.min(), obs.max(), obs_range, obs_mean
            )
            if obs_range > 200 or abs(obs_mean) > 100:
                logger.error(
                    "ENERGY RANGE WARNING: dataset observations span %.1f kcal/mol "
                    "with mean=%.1f. GP is configured for ~0-50 kcal/mol range. "
                    "This WILL cause Inf in GP hyperparameters. "
                    "Check ensemble normalization — norm_energy=%.3f may be incompatible "
                    "with ensemble energy scale.",
                    obs_range, obs_mean, self.state.norm_energy
                )
                raise RuntimeError(
                    f"Dataset energy range ({obs_range:.1f} kcal/mol) is too large for GP. "
                    f"Use ensemble_min normalization instead of ORCA norm_energy for TS ensembles."
                )
        else:
            for idx in range(config.num_initial_points):
                # Check conformer existance
                mol_copy = Chem.RWMol(self.state.mol)
                mol_copy = Chem.AddHs(mol_copy) # Silence RDkit warning
                mol_copy.RemoveAllConformers()
                res = AllChem.EmbedMolecule(mol_copy, AllChem.ETKDGv3())
                mol_copy = Chem.RemoveHs(mol_copy)

                if res == -1:
                    res = AllChem.EmbedMolecule(mol_copy, randomSeed=idx, useRandomCoords=True)
                if res == -1:
                    # Fallback: use geometry from .mol file
                    logger.warning(
                        "EmbedMolecule failed for initial point %d "
                        "Using original .mol geometry with random dihedral perturbation.", idx
                    )
                    mol_copy = self.state.mol
                    # Perturbation: random dihedrals instead of geometry from a file
                    n_dih = len(self.state.dihedral_ids)
                    random_vals = [np.random.uniform(0, 2 * np.pi) for _ in range(n_dih)]
                    # Uses different range
                    if self.state.vary_ts_bond_lengths:
                        lo = self.state.config.ts_bond_min_length
                        hi = self.state.config.ts_bond_max_length
                        random_vals += [np.random.uniform(lo, hi) for _ in range(len(self.state.ts_bonds))]
                    initial_query_points = tf.constant([random_vals], dtype=tf.float64)
                else:
                    initial_query_points = self._extract_dofs_values(mol_copy)

                observed_point = observer(initial_query_points)
                if not self.state.last_opt_ok:
                    logger.warning(
                        "Optimization didn't finish well. Continue only with broken_struct_energy in required point: %s",
                        observed_point,
                    )
                    dataset = observed_point if not dataset else dataset + observed_point
                else:
                    dataset = self._upd_dataset_from_trj(
                        str(_qc_calcs_dir(self.state.mol_file_name) / (Path(self.state.mol_file_name).stem + "_trj.xyz")), dataset
                    )
            logger.info(
                "Initial dataset observed! %s minima observed, total %s points has been collected!",
                config.num_initial_points,
                dataset.query_points.shape[0],
            )

        return dataset

    def _clear_working_folder(self) -> None:
        """Start a clean run: move the previous run's files from working_folder into a new
        backupN/ subfolder (backup1/, backup2/, ...). Nothing is deleted except the
        intermediate files in qc_calcs/: from there only the last .out and _trj.xyz are kept.

        Stay in place: input files (.mol, load_ensemble, config, .db, sbatch template),
        folders that contain an input file and earlier backupN/ folders.
        The file this process's stdout/stderr is redirected to is split: what it held at
        start (previous run, only with `>>`) goes to the backup, this run's lines stay."""
        config = self.state.config
        working_folder = Path(self.state.working_folder).resolve()

        # Never wipe a project/home/root folder (e.g. the default --folder=. run from the repo root)
        code_dir = Path(__file__).resolve().parent
        if (
            (working_folder / ".git").exists()
            or working_folder == code_dir
            or working_folder in code_dir.parents
            or working_folder == Path.home().resolve()
            or working_folder == Path(working_folder.anchor)
        ):
            raise RuntimeError(
                f"clear_working_folder=True refused for {working_folder}: it is a repository, "
                f"contains the BOCSER code, or is a home/root folder. Use a dedicated run folder."
            )

        keep = set()
        mol_path = Path(config.mol_file_name)
        if not mol_path.is_absolute():
            mol_path = (working_folder / mol_path).resolve()
        keep.add(mol_path.resolve())
        # setup() resolves mol_file_name against the CWD, keep that location too
        keep.add(Path(config.mol_file_name).resolve())
        # sbatch template: looked up in qc_calcs/ first, then relative to the CWD
        keep.add(Path(config.sbatch_template_name).resolve())
        qc_dir = (Path(config.mol_file_name).resolve().parent / "qc_calcs").resolve()
        keep.add(qc_dir / config.sbatch_template_name)

        if config.load_ensemble:
            # same lookup as _resolve_ensemble_path: working folder, then CWD
            keep.add(Path(self._resolve_ensemble_path()).resolve())

        if getattr(self.state, "config_path", None):
            keep.add(Path(self.state.config_path).resolve())
        keep.add(Path(self.state.db_file).resolve())

        protected_suffixes = {".mol", ".db", ".yaml", ".yml"}
        for item in working_folder.iterdir():
            if item.is_file() and item.suffix.lower() in protected_suffixes:
                keep.add(item.resolve())

        backup_re = re.compile(r"backup(\d+)$")
        existing = [int(m.group(1)) for item in working_folder.iterdir()
                    if item.is_dir() and (m := backup_re.match(item.name))]
        backup_dir = working_folder / f"backup{max(existing, default=0) + 1}"

        def move(src: Path) -> None:
            dest = backup_dir / src.resolve().relative_to(working_folder)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))

        moved = 0
        # qc_calcs/: keep only the last ORCA output and its trajectory, drop inputs/scripts/scratch
        if qc_dir.is_dir() and qc_dir.is_relative_to(working_folder):
            for f in sorted(qc_dir.iterdir()):
                if f.resolve() in keep:
                    continue
                try:
                    if f.is_file() and (f.suffix == ".out" or f.name.endswith("_trj.xyz")):
                        move(f)
                        moved += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
                except Exception:
                    logger.exception("Failed to back up / remove %s", f)
            if not any(qc_dir.iterdir()):
                qc_dir.rmdir()

        for item in sorted(working_folder.iterdir()):
            resolved = item.resolve()
            if resolved in keep or backup_re.match(item.name) or resolved == qc_dir:
                continue
            if item.is_dir() and any(k.is_relative_to(resolved) for k in keep):
                logger.info("Leaving folder %s in place: it contains an input file", item)
                continue
            if item.is_file():
                st = item.stat()
                own = _OWN_LOGS.get((st.st_dev, st.st_ino))
                if own is not None:
                    moved += self._split_own_log(item, backup_dir / item.name, *own)
                    continue
            try:
                move(item)
                moved += 1
            except Exception:
                logger.exception("Failed to move %s to %s", item, backup_dir)

        if moved:
            logger.info("clear_working_folder: previous results moved to %s (%d items)", backup_dir, moved)
        else:
            logger.info("clear_working_folder: nothing to back up in %s", working_folder)
        if config.checkpoint_step is not None:
            logger.warning(
                "checkpoint_step is set together with clear_working_folder: the old checkpoint "
                "was moved to the backup, the run starts from scratch."
            )

    @staticmethod
    def _split_own_log(path: Path, dest: Path, start_size: int, append: Optional[bool]) -> int:
        """The current run writes its output to `path`. Copy the part written before this
        process started (previous run) to `dest` and keep only this run's lines in `path`.
        Returns 1 if something was backed up."""
        if start_size == 0:
            logger.info(
                "%s is this run's output; the previous run's log was already emptied by the shell "
                "(launch with '>> %s' instead of '>' to keep it in the backup).", path, path.name,
            )
            return 0
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (AttributeError, OSError, ValueError):
                pass
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "rb") as src, open(dest, "wb") as out:
            out.write(src.read(start_size))
        if append is not True:
            # Not confirmed O_APPEND (or Windows, where it can't be checked): after a truncation
            # the writer would continue at its old offset and leave a hole of zero bytes,
            # so the file is left as is (previous + current run)
            logger.info("Previous run's log copied to %s; %s left unchanged (append mode not confirmed).", dest, path)
            return 1
        with open(path, "r+b") as f:
            f.seek(start_size)
            current = f.read()
            f.seek(0)
            f.truncate()
            f.write(current)
        logger.info("Previous run's log moved to %s; %s now holds only this run.", dest, path)
        return 1

    def _build_rule(self) -> Any:
        """Build acquisition rule based on configuration."""
        config = self.state.config

        rule = None

        match config.acquisition_function:
            case "evm":
                logger.info("Continue with Explorational Variance Minimizer acquisition function!")
                rule = EfficientGlobalOptimization(ExplorationalVarianceMinimizer(threshold=3))
            case "ei":
                logger.info("Continue with ExpectedImprovement acquisition function!")
                rule = EfficientGlobalOptimization(ExpectedImprovement())
            case "ik":
                logger.info("Continue with ImprovementVarianceWithIK acquisition function!")
                if self.state.ik_loss is None or len(self.state.ik_loss_dihedrals_idxs) == 0:
                    logger.warning("IK loss is not available; falling back to ExplorationalVarianceMinimizer")
                    rule = EfficientGlobalOptimization(ExplorationalVarianceMinimizer(threshold=3))
                else:
                    rule = EfficientGlobalOptimization(
                        ImprovementVarianceWithIK(
                            threshold=3.0,
                            ik_loss=self.state.ik_loss,
                            ik_loss_idxs=self.state.ik_loss_dihedrals_idxs,
                            ik_loss_weight=1.0,
                        )
                    )
            case _:
                raise ValueError(f"Unknown acquisition function {config.acquisition_function}")

        return rule

    def _checkpoint_path(self) -> Path:
        return Path(self.state.working_folder) / f"{self.state.exp_name}_checkpoint.json"
    
    def _load_checkpoint_file(self) -> Optional[dict]:
        """Load full checkpoint JSON (history of all saved steps)."""
        path = self._checkpoint_path()
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
        except Exception:
            logger.exception(
                "Checkpoint file %s is corrupted — ignoring.", path
            )
            return None
        # Backward compat: old single-snapshot format → wrap as history
        if isinstance(data, dict) and "history" not in data and "step" in data:
            return {"history": [data]}
        if not isinstance(data, dict) or "history" not in data:
            logger.warning("Checkpoint file %s has unknown format — ignoring.", path)
            return None
        return data

    def _save_checkpoint(self, dataset: Dataset, step: int) -> None:
        """Append / update this step in the checkpoint history.

        Full query_points + observations are stored so a later resume
        rebuilds the same GP training set as if ORCA had been re-run.
        """
        snapshot = {
            "step": step,
            "query_points": dataset.query_points.numpy().tolist(),
            "observations": dataset.observations.numpy().tolist(),
            "minima": self.state.minima,
            "current_minima": float(self.state.current_minima),
            "acq_vals_log": list(self.state.acq_vals_log),
            "deepest_minima": [float(v) for v in self.state.deepest_minima],
        }

        existing = self._load_checkpoint_file()
        history = []
        if existing is not None:
            # Drop same/later steps so re-running from an earlier step stays consistent
            history = [
                h for h in existing.get("history", [])
                if isinstance(h, dict) and h.get("step", -1) < step
            ]
        history.append(snapshot)

        payload = {"history": history}
        tmp_path = self._checkpoint_path().with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload))
        tmp_path.replace(self._checkpoint_path())
        logger.info("Checkpoint saved")
    
    def _resolve_checkpoint_snapshot(self) -> Optional[dict]:
        """Pick a history entry according to config.checkpoint_step.

        Rules:
        - checkpoint_step is None → do not resume (normal run).
        - no / empty checkpoint file → log, no resume.
        - checkpoint_step > last available step → use last, log warning.
        - checkpoint_step matches an entry → use it.
        - otherwise (e.g. step missing in the middle, or < 1) → log, no resume.
        """
        config = self.state.config
        requested = config.checkpoint_step
        if requested is None:
            return None

        data = self._load_checkpoint_file()
        if data is None:
            logger.warning(
                "checkpoint_step=%s set, but no usable checkpoint file at %s — starting fresh.",
                requested, self._checkpoint_path(),
            )
            return None

        history = [
            h for h in data.get("history", [])
            if isinstance(h, dict) and "step" in h
        ]
        if not history:
            logger.warning(
                "checkpoint_step=%s set, but checkpoint history is empty — starting fresh.",
                requested,
            )
            return None

        history_sorted = sorted(history, key=lambda h: h["step"])
        steps_available = [h["step"] for h in history_sorted]
        last = history_sorted[-1]

        try:
            req = int(requested)
        except (TypeError, ValueError):
            logger.warning(
                "checkpoint_step=%r is not a valid integer — ignoring checkpoint.",
                requested,
            )
            return None

        if req < 1:
            logger.warning(
                "checkpoint_step=%d is out of range (need >= 1) — ignoring checkpoint.",
                req,
            )
            return None

        if req > last["step"]:
            logger.warning(
                "checkpoint_step=%d is greater than last saved step=%d (available=%s). "
                "Using last checkpoint.",
                req, last["step"], steps_available,
            )
            return last

        for h in history_sorted:
            if h["step"] == req:
                logger.info(
                    "Resuming from checkpoint_step=%d (%d dataset points, %d minima).",
                    req, len(h.get("observations", [])), len(h.get("minima", [])),
                )
                return h

        logger.warning(
            "checkpoint_step=%d not found in checkpoint history (available=%s) — "
            "ignoring checkpoint, starting fresh.",
            req, steps_available,
        )
        return None

    def _dataset_from_checkpoint(self, snapshot: dict) -> Dataset:
        """Restore Dataset + runner state fields from a checkpoint snapshot."""
        query_points = snapshot["query_points"]
        ckpt_dim = len(query_points[0]) if query_points else None
        if ckpt_dim is not None and ckpt_dim != self.state.search_dim:
            raise RuntimeError(
                f"Checkpoint at {self._checkpoint_path()} has search_dim="
                f"{ckpt_dim}, but the current config/molecule implies "
                f"search_dim={self.state.search_dim} "
                f"(n_dihedral={len(self.state.dihedral_ids)}, "
                f"vary_ts_bond_lengths={self.state.vary_ts_bond_lengths}, "
                f"n_ts_bonds={len(self.state.ts_bonds)}). The config was "
                f"likely changed (ts_bonds / vary_ts_bond_lengths / ring "
                f"topology) since this checkpoint was written — loading it "
                f"as-is would silently corrupt the GP training tensors. "
                f"Delete the checkpoint to start fresh, or restore the "
                f"original config that produced it."
            )

        dataset = Dataset(
            tf.constant(query_points, dtype=tf.float64),
            tf.constant(snapshot["observations"], dtype=tf.float64),
        )
        self.state.minima = snapshot.get("minima", [])
        self.state.current_minima = float(snapshot.get("current_minima", 1e9))
        self.state.acq_vals_log = list(snapshot.get("acq_vals_log", []))
        self.state.deepest_minima = list(snapshot.get("deepest_minima", []))

        # The structure-id counter restarts at 0 in a new process: continue after the
        # highest id already on disk so resumed steps don't overwrite earlier files
        next_id = self._next_free_structure_id()
        run_state.set_structure_id(next_id)
        logger.info("Structure numbering continues from %d.", next_id)
        return dataset

    def _next_free_structure_id(self) -> int:
        """1 + highest numeric file-name prefix in the structures/broken/success folders."""
        max_id = -1
        for folder in (self.state.structures_path, self.state.broken_structs_path, self.state.success_out_dir):
            if not folder or not Path(folder).is_dir():
                continue
            for item in Path(folder).iterdir():
                m = re.match(r"(\d+)", item.name)
                if m:
                    max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def run(self) -> None:
        """Execute the full Bayesian optimization loop."""
        config = self.state.config

        kernel, search_space, observer = self._build_model_and_acquisition()
        
        ckpt = self._resolve_checkpoint_snapshot()
        start_step = 1
        if ckpt is not None:
            dataset = self._dataset_from_checkpoint(ckpt)
            start_step = ckpt["step"] + 1
            logger.info(
                "BO loop will continue from step %d (restored step %d).",
                start_step, ckpt["step"],
            )
        else:
            dataset = self._initialize_dataset(observer)
        
        obs = dataset.observations.numpy().flatten()
        obs_std = float(np.std(obs))
        obs_mean = float(np.mean(obs))
        obs_range = float(obs.max() - obs.min())

        logger.info(
            "Dataset stats before GP init: mean=%.2f, std=%.2f, range=%.2f kcal/mol",
            obs_mean, obs_std, obs_range
        )
        
        if obs_range > 200 or abs(obs_mean) > 100:
            raise RuntimeError(
                f"Dataset energy range ({obs_range:.1f} kcal/mol, mean={obs_mean:.1f}) "
                f"is too large for GP. Check ensemble normalization."
            )

        # Adapt hyperparameters to initial data scale.
        init_variance = max(0.07, obs_std ** 2 * 0.1)   # 10% from the variance of observations
        init_lengthscale = 0.5                          # typical distance between dihedra in rad

        kernel.kernels[1].base_kernel.variance.assign(init_variance)
        kernel.kernels[2].base_kernel.variance.assign(init_variance)
        kernel.kernels[2].base_kernel.lengthscales.assign(init_lengthscale)

        if self.state.vary_ts_bond_lengths:
            # kernel.kernels[3] = RBF-term for TS-bonds,
            # used only when vary_ts_bond_lengths: True
            kernel.kernels[3].variance.assign(init_variance)
            logger.info(
                "GP kernel initialized (TS-bond term): variance=%.4f, lengthscales=%.3f A",
                init_variance, self.state.config.ts_bond_kernel_lengthscale,
            )

        logger.info(
            "GP kernel initialized: variance=%.4f, lengthscales=%.3f",
            init_variance, init_lengthscale
        )

        gpr = gpflow.models.GPR(dataset.astuple(), kernel)
        gpflow.set_trainable(gpr.likelihood, False)
        gpflow.set_trainable(gpr.kernel.kernels[0].variance, False)
        gpflow.set_trainable(gpr.kernel.kernels[1].period, False)
        model = GaussianProcessRegression(gpr, num_kernel_samples=100)

        bo = trieste.bayesian_optimizer.BayesianOptimizer(observer, search_space)

        logger.debug("Initial data: %s", dataset)

        # Check if all values are the same
        obs = dataset.observations.numpy().flatten()
        n_broken = int(np.sum([_is_broken(v, config.broken_struct_energy) for v in obs]))
        if n_broken == len(obs):
            logger.error(
                "All %d observations are sentinel broken_struct_energy=%.1f (tol=±5). "
                "GP training will be numerically unstable. "
                "Check norm_energy calculation and ring geometry.",
                len(obs), config.broken_struct_energy
            )

        try:
            model.optimize(dataset)
        except Exception as e:
            logger.error(
                "Initial GP optimization failed: %s. "
                "Likely cause: degenerate dataset — all observations are broken_struct_energy (%.1f). "
                "Check norm_energy calculation and ring geometry in .mol file.",
                e, config.broken_struct_energy
            )
            # Диагностика датасета
            obs = dataset.observations.numpy().flatten()
            n_broken = int(np.sum([_is_broken(v, config.broken_struct_energy) for v in obs]))
            logger.error(
                "Dataset diagnostics: %d points total, %d broken (>=%.1f), "
                "min=%.3f, max=%.3f",
                len(obs), n_broken, config.broken_struct_energy * 0.9,
                float(obs.min()), float(obs.max())
            )
            raise RuntimeError(
                f"Cannot start BO loop: initial GP optimization failed. "
                f"{n_broken}/{len(obs)} dataset points are broken_struct_energy. "
                f"See log for details."
            ) from e

        self.state.model_chk = gpflow.utilities.deepcopy(model.model)
        self.state.current_minima = tf.reduce_min(dataset.observations).numpy()

        rule = self._build_rule()

        deepest_minima = self.state.deepest_minima  # non-empty when resumed from a checkpoint
        early_termination_flag = False

        logger.info("MINIMA: %s", self.state.minima)

        if start_step > config.max_steps:
            logger.info(
                "start_step=%d > max_steps=%d — nothing left to run; saving results.",
                start_step, config.max_steps,
            )
            self._save_results(dataset)
            return
        
        for step in range(start_step, config.max_steps + 1):
            logger.debug("Previous last_opt_ok: %s", self.state.last_opt_ok)
            logger.debug("Step number %s", step)

            try:
                t0 = time.monotonic()
                result = bo.optimize(1, dataset, model, rule, fit_initial_model=False)
                logger.info("Optimization step %s succeed! (bo.optimize took %.1f s)", step, time.monotonic() - t0)
            except Exception:
                logger.exception(
                    "trieste bo.optimize() raised on step %s — skipping this step, "
                    "keeping previous dataset/model unchanged.", step
                )
                continue

            logger.debug("After step: %s", self.state.last_opt_ok)

            last_opt_status = None
            status_file = Path(self.state.working_folder) / f"{self.state.exp_name}_last_opt_status.json"
            if status_file.is_file():
                try:
                    with open(status_file, "r") as file:
                        last_opt_status = json.load(file)
                except (json.JSONDecodeError, OSError) as e:
                    logger.error(
                        "Status file %s exists but could not be parsed (%s) "
                        "— treating as LAST_OPT_OK=False instead of crashing.",
                        status_file, e,
                    )
                    last_opt_status = {"LAST_OPT_OK": False}
            else:
                last_opt_status = {"LAST_OPT_OK": False}
                logger.error("Status file missing — acq/objective failed before dump")
            logger.debug("Last opt status: %s", last_opt_status)

            try:
                dataset = result.try_get_final_dataset()
                model = result.try_get_final_model()
            except Exception as e:
                if "GatherV2" in str(e) or "not in [0" in str(e):
                    logger.error(
                        "Step %s: trieste result unavailable due to an indexing error in "
                        "acquisition-function. Keeping previous dataset and model.",
                        step,
                    )
                else:
                    logger.error(
                        "Step %d: could not get final dataset/model from trieste result: %s. "
                        "Keeping previous dataset and model. "
                        "Likely cause: GP hyperparameter optimization failed (Inf/NaN). "
                        "This usually means dataset observations are in a bad range — "
                        "check norm_energy and ensemble normalization.",
                        step, e
                    )
                # Continue with old dataset, model
                model.update(dataset)
                try:
                    model.optimize(dataset)
                except Exception as e2:
                    logger.error(
                        "Step %d: fallback model.optimize also failed: %s. "
                        "Continuing with stale hyperparameters.",
                        step, e2
                    )
                continue

            logger.debug("Last asked point was %s", self.state.asked_points[-1])

            deepest_minima.append(tf.reduce_min(dataset.observations).numpy())

            logs = {
                "acq_vals": self.state.acq_vals_log,
                "deepest_minima": deepest_minima,
                "norm_en": self.state.norm_energy,
            }

            logs_file = Path(self.state.working_folder) / f"{self.state.exp_name}_logs.json"
            with open(logs_file, "w") as file:
                json.dump(logs, file)

            logger.debug("Eta is %s", getattr(rule._acquisition_function, "_eta", None))
            if self.state.last_opt_ok:
                dataset = self._erase_last_from_dataset(dataset, 1)
                dataset = self._upd_dataset_from_trj(
                    str(_qc_calcs_dir(self.state.mol_file_name) / (Path(self.state.mol_file_name).stem + "_trj.xyz")), dataset
                )
            else:
                logger.warning("Last optimization finished with error, skipping trj parsing!")
            model.update(dataset)
            try:
                model.optimize(dataset)
            except Exception as e:
                logger.warning(
                    "GP optimization failed at step %d: %s. "
                    "Continuing with current hyperparameters.",
                    step, e
                )

            logger.info("Updating model checkpoint!")
            self.state.model_chk = gpflow.utilities.deepcopy(model.model)
            try:
                self.state.current_minima = rule._acquisition_function._eta.numpy()[0]
            except Exception:
                logger.debug("Unable to read current minima from acquisition function" )
            logger.info("Updated!")

            logger.info("Step %s completed!", step)
            obs = dataset.observations.numpy().flatten()
            n_broken = int(np.sum([_is_broken(v, config.broken_struct_energy) for v in obs]))
            logger.info(
                "Step %d completed. Dataset: %d points, best=%.3f, "
                "broken=%d, mean_valid=%.3f",
                step, len(obs), float(obs.min()), n_broken,
                float(obs[obs < config.broken_struct_energy * 0.9].mean())
                if n_broken < len(obs) else float("nan"),
            )
            _print_progress(step, config.max_steps, float(obs.min()), len(obs), n_broken)

            all_minima_file = Path(self.state.working_folder) / f"{self.state.exp_name}_all_minima.json"
            with open(all_minima_file, "w") as json_minima_writer:
                json.dump(self.state.minima, json_minima_writer)
            
            try:
                self._save_checkpoint(dataset, step)
            except Exception:
                logger.exception("Failed to write checkpoint at step %s", step)

            # Last N acquisition values: a step can add no value (bo.optimize failed before
            # the objective ran), so indexing the list by step number would misalign the window
            window_vals = logs["acq_vals"][-config.rolling_window_size:] if config.rolling_window_size > 0 else []
            if config.rolling_window_size <= 0 or len(window_vals) < config.rolling_window_size:
                continue

            logger.debug("Checking termination criterion!")
            logger.debug("Acq vals in window: %s", window_vals)

            rolling_mean = np.mean(window_vals)
            rolling_std = np.std(window_vals)

            logger.debug("After step %s:", step)
            logger.info(
                "Current rolling mean of acquisition function maximum is: %s, threshold is %s",
                rolling_mean,
                config.rolling_mean_threshold,
            )
            logger.info(
                "Current rolling std of acquisition function maximum is: %s, threshold is %s",
                rolling_std,
                config.rolling_std_threshold,
            )
            if (
                rolling_std < config.rolling_std_threshold
                and rolling_mean < config.rolling_mean_threshold
            ):
                logger.info("Termination criterion reached on step %s! Terminating search!", step)
                early_termination_flag = True
                break

        if not early_termination_flag:
            logger.info("Max number of steps has been reached!")
            print(f"[BOCSER] Finished: reached max_steps={config.max_steps}", flush=True)
        else:
            print(f"[BOCSER] Finished early: convergence criterion met at step {step}", flush=True)

        logger.info("MINIMA: %s", self.state.minima)
        self._save_results(dataset)

    def _save_results(self, dataset: Dataset) -> None:
        """Save final results and ensembles."""
        query_points = dataset.query_points.numpy()
        observations = dataset.observations.numpy()

        if not self.state.minima:
            # No successful full-optimization ever completed this run
            logger.error(
                "No minima were found this run (%d dataset points collected, "
                "all candidates ended up broken/failed — see per-step logs "
                "above for the specific cause of each). Writing empty result "
                "files instead of clustering.",
                len(observations),
            )
            clustering_file = str(Path(self.state.working_folder) / f"{self.state.exp_name}_clustering_results.json")
            json.dump({}, open(clustering_file, "w"))

            final_ensemble_file = str(Path(self.state.working_folder) / f"{self.state.exp_name}_final_ensemble.xyz")
            open(final_ensemble_file, "w").close()

            all_points_file = str(Path(self.state.working_folder) / f"{self.state.exp_name}_all_points.json")
            json.dump(
                {"query_points": query_points.tolist(), "observations": observations.tolist()},
                open(all_points_file, "w"),
            )
            ckpt_path = self._checkpoint_path()
            if ckpt_path.is_file():
                ckpt_path.unlink()
                logger.info("Run finished (with zero minima) — checkpoint file removed.")
            return

        res = self._cluster_minima_by_rmsd()

        clustering_file = str(Path(self.state.working_folder) / f"{self.state.exp_name}_clustering_results.json")
        logger.info(
            "Results of clustering: %s. There are relative energy and number of structure for each cluster. Saved in %s",
            res,
            clustering_file,
        )
        json.dump(res, open(clustering_file, "w"))

        final_ensemble_file = str(Path(self.state.working_folder) / f"{self.state.exp_name}_final_ensemble.xyz")
        logger.info("Saving final ensemble into %s", final_ensemble_file)
        ens_xyz_str = ""
        for _, structure_id in res.values():
            cur_xyz = ""
            minima_file = str(Path(self.state.working_folder) / f"{self.state.exp_name}_minima" / f"{structure_id}.xyz")
            with open(minima_file, "r") as cur_xyz_reader:
                cur_xyz = "".join([line for line in cur_xyz_reader])
            ens_xyz_str += cur_xyz + "\n"

        with open(final_ensemble_file, "w") as ens_writer:
            ens_writer.write(ens_xyz_str)

        all_points_file = str(Path(self.state.working_folder) / f"{self.state.exp_name}_all_points.json")
        logger.info("Saving all points at %s", all_points_file)
        json.dump(
            {"query_points": query_points.tolist(), "observations": observations.tolist()},
            open(all_points_file, "w"),
        )
        ckpt_path = self._checkpoint_path()
        if ckpt_path.is_file():
            ckpt_path.unlink()
            logger.info("Run finished successfully — checkpoint file removed.")

    def _cluster_minima_by_rmsd(self) -> dict:
        """Group the found minima into unique conformers.

        Two minima are the same conformer when their energies differ by less than
        dedup_energy_threshold AND the heavy-atom RMSD after symmetry-aware alignment
        (RDKit GetBestRMS) is below dedup_rmsd_threshold. All atoms take part, so ring
        torsions outside the GP axes and TS-bond lengths are compared too.
        Minima are processed from the lowest energy, so each group is represented by its
        lowest-energy member and groups come out sorted by energy.
        Returns {group: (relative energy, minimum index)}.
        """
        cfg = self.state.config
        minima_dir = Path(self.state.working_folder) / f"{self.state.exp_name}_minima"
        ref = build_reference_with_h_mol(self.state.mol_file_name)

        candidates = []  # (energy, minimum index, heavy-atom mol)
        for i, (_, energy) in enumerate(self.state.minima):
            path = minima_dir / f"{i}.xyz"
            try:
                lines = [l for l in path.read_text().splitlines() if l.strip()]
                if lines and lines[0].strip().isdigit():
                    lines = lines[2:]
                if len(lines) != ref.GetNumAtoms():
                    raise ValueError(f"{len(lines)} atoms instead of {ref.GetNumAtoms()}")
                mol = Chem.Mol(ref)
                conf = mol.GetConformer()
                for k, line in enumerate(lines):
                    conf.SetAtomPosition(k, [float(v) for v in line.split()[1:4]])
                candidates.append((float(energy), i, Chem.RemoveHs(mol)))
            except (OSError, ValueError) as e:
                logger.warning("Minimum %d (%s) skipped in the final ensemble: %s", i, path, e)
        candidates.sort(key=lambda c: c[0])

        groups = []  # (energy, minimum index, heavy-atom mol) of each group's representative
        for energy, idx, mol in candidates:
            for g_energy, g_idx, g_mol in groups:
                if abs(energy - g_energy) >= cfg.dedup_energy_threshold:
                    continue
                rmsd = rdMolAlign.GetBestRMS(Chem.Mol(mol), g_mol)
                if rmsd < cfg.dedup_rmsd_threshold:
                    logger.debug("Minimum %d duplicates %d (dE=%.3f kcal/mol, RMSD=%.3f A)",
                                 idx, g_idx, energy - g_energy, rmsd)
                    break
            else:
                groups.append((energy, idx, mol))

        logger.info(
            "Final ensemble: %d unique conformers out of %d minima (RMSD < %.3f A and dE < %.3f kcal/mol = duplicate)",
            len(groups), len(candidates), cfg.dedup_rmsd_threshold, cfg.dedup_energy_threshold,
        )
        return {label: (energy, idx) for label, (energy, idx, _) in enumerate(groups)}

    def _resolve_ensemble_path(self) -> str:
        """Returns absolute path to load_ensemble: relative paths are looked up in the
        working folder (--folder) first, then in the current working directory."""
        config = self.state.config
        load_ensemble_filename = Path(config.load_ensemble)
        if load_ensemble_filename.is_absolute():
            return str(load_ensemble_filename)
        in_folder = Path(self.state.working_folder) / load_ensemble_filename
        in_cwd = Path.cwd() / load_ensemble_filename
        if not in_folder.is_file() and in_cwd.is_file():
            logger.info("load_ensemble %s not found in %s — using %s",
                        config.load_ensemble, self.state.working_folder, in_cwd.resolve())
            return str(in_cwd.resolve())
        return str(in_folder)

    def _make_ensemble_processor(self, dihedral_idxs: Optional[list] = None) -> EnsembleProcessor:
        """EnsembleProcessor with atom-order check against the .mol and, when TS-bond
        lengths are searched, the TS-bond dimensions of the search vector."""
        raw_mol = Chem.MolFromMolFile(self.state.mol_file_name, removeHs=False)
        heavy_idx, h_idx = _heavy_and_h_order(raw_mol)
        raw_order = heavy_idx + h_idx
        ts_kwargs = {}
        if self.state.vary_ts_bond_lengths:
            ts_kwargs = dict(
                ts_bonds=self.state.ts_bonds,
                ts_bounds=(self.state.config.ts_bond_min_length, self.state.config.ts_bond_max_length),
            )
        return EnsembleProcessor(
            self._resolve_ensemble_path(),
            dihedral_idxs=self.state.dihedral_ids if dihedral_idxs is None else dihedral_idxs,
            ref_symbols=[raw_mol.GetAtomWithIdx(i).GetSymbol() for i in raw_order],
            raw_order=raw_order,
            **ts_kwargs,
        )

    def _resolve_fixed_ts_bond_targets(self, config: ConfSearchConfig) -> list:
        """Fixed TS-bond lengths, held ONLY in the ORCA pre-optimization (ts: true only).

        Two sources:
          - user input: `bond` entries of extra_constraints on a ts_bonds pair
            (value: number or "current" = length in the .mol);
          - infer_ts_bond_lengths_from_ensemble: lengths in the lowest-energy conformer of
            load_ensemble (overrides user values).
        Such TS-pair bond entries are taken out of the hard constraints, which are applied
        in every ORCA stage, so the final OptTS can move along the reaction coordinate.
        Returns [((a, b), length)] with canonical 0-idx atoms.
        """
        if not config.ts:
            if config.infer_ts_bond_lengths_from_ensemble:
                logger.warning("infer_ts_bond_lengths_from_ensemble is ignored: ts is false.")
            return []

        ts_pairs = {frozenset(p) for p in self.state.ts_bonds}
        lengths, kept = {}, []
        for c in self.state.extra_constraints:
            if c.type == "bond" and frozenset(c.atoms) in ts_pairs:
                lengths[frozenset(c.atoms)] = float(c.value)
            else:
                kept.append(c)
        self.state.extra_constraints = kept
        source = "extra_constraints (pre-opt only)"

        if config.infer_ts_bond_lengths_from_ensemble:
            if not config.load_ensemble or not self.state.ts_bonds:
                raise ConfigError("infer_ts_bond_lengths_from_ensemble: true needs both load_ensemble and ts_bonds.")
            ep = self._make_ensemble_processor(dihedral_idxs=[])
            if not ep.energies:
                raise RuntimeError(f"load_ensemble {self._resolve_ensemble_path()}: no conformers found.")
            best = int(np.argmin(ep.energies))
            coords = np.asarray(ep.coords[best], dtype=float)
            if max(max(p) for p in self.state.ts_bonds) >= len(coords):
                raise ConfigError("load_ensemble has fewer atoms than the .mol (no H?): TS-bond lengths can't be measured.")
            if lengths:
                logger.warning("TS-bond values from extra_constraints are replaced by the ensemble values.")
            lengths = {frozenset((a, b)): float(np.linalg.norm(coords[a] - coords[b])) for a, b in self.state.ts_bonds}
            source = f"lowest-energy ensemble conformer #{best + 1}/{len(ep.energies)} (E={ep.energies[best]:.6f})"

        targets = [((a, b), lengths[frozenset((a, b))]) for a, b in self.state.ts_bonds if frozenset((a, b)) in lengths]
        if not targets:
            return []

        for (a, b), length in targets:
            if not 0.0 < length <= config.ts_bond_max_length:
                raise ConfigError(
                    f"Fixed TS-bond length {length:.3f} A for atoms "
                    f"({with_h_canonical_to_raw1(a, self.state.mol_file_name)}, "
                    f"{with_h_canonical_to_raw1(b, self.state.mol_file_name)}) is outside "
                    f"(0, ts_bond_max_length={config.ts_bond_max_length}]: every point would be rejected."
                )
            logger.info(
                "Fixed TS bond (%d, %d) [1-idx .mol] = %.4f A from %s",
                with_h_canonical_to_raw1(a, self.state.mol_file_name),
                with_h_canonical_to_raw1(b, self.state.mol_file_name), length, source,
            )
        if config.vary_ts_bond_lengths:
            logger.warning("vary_ts_bond_lengths switched off: TS-bond lengths are fixed (%s).", source)
            config.vary_ts_bond_lengths = False
        return targets


def main():
    """Entry point for the conformational search orchestrator."""
    import argparse

    logging.basicConfig( # Output all project info
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="bo_confsearch",
        description="Bayesian optimization for conformational search",
    )
    parser.add_argument(
        "--folder",
        default=".",
        help="Working folder for input files and output results (default: current directory)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file name (relative to --folder or absolute path)"
    )

    args = parser.parse_args()

    logger.info("Working folder: %s", args.folder)
    logger.info("Reading config from: %s", args.config)

    runner = ConfSearchRunner(working_folder=args.folder)
    runner.load_config(args.config)
    runner.setup()
    runner.run()

    logger.info("Conformational search completed!")


if __name__ == "__main__":
    main()