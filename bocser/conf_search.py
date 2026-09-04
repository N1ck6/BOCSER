"""
Bayesian Optimization for Conformational Search - Refactored Class-Based Orchestrator

This module provides ConfSearchRunner, a class-based orchestrator that encapsulates
all state and behavior for the conformational search workflow.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="gpflow")
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.warning")
from rdkit.Chem import AllChem
from dataclasses import dataclass, field
from typing import Optional, Tuple, Any
import json
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
    raw1_to_heavy_canonical,
    _heavy_and_h_order,
    resolve_extra_constraints,
    _validate_extra_constraints_atoms,
    compute_mol_hash,
    build_reference_with_h_mol,
    submit_calc, wait_for_jobs,
)
from run_state import increase_structure_id
import config_manager
from coef_calc import (
    CoefCalculator,
    log_and_combine_double_bonds,
    parse_bond_scan_results,
    generate_bond_scan_inp,
)
from db_connector import LocalConnector
from ensemble_processor import EnsembleProcessor
from evm import ExplorationalVarianceMinimizer
from dbscan import DBSCAN
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
    last_opt_ok: bool = True
    ik_loss: Optional[Any] = None
    ik_loss_dihedrals_idxs: list = field(default_factory=list)
    mean_func_coefs: list = field(default_factory=list)
    search_dim: int = 0
    mol: Optional[Chem.Mol] = None
    config: Optional[ConfSearchConfig] = None
    extra_constraints: list = field(default_factory=list)
    vary_ts_bond_lengths: bool = False
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

    # Cremer & Pople result that an N-membered ring has exactly N-3 independent shape coordinates
    _RING_DOF_SLACK = 3

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

        FLAT_COEFS = (0.0,) * 7
        drop_idx_to_window: dict = {}

        for ring_i, (cycle_d, cycle_idx) in enumerate(zip(dihedral_list_all, ik_loss_dihedrals_idxs)):
            n_flexible = sum(1 for idx in cycle_idx if idx != -2 and idx != -3)
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
            # Prefer keeping axes with non-flat Fourier mean function
            ranked = sorted(
                optional_in_ring,
                key=lambda pair: self.state.mean_func_coefs[pair[0]] == FLAT_COEFS,
            )
            keep = ranked[:budget_for_optional]
            drop = ranked[budget_for_optional:]

            logger.info(
                "Ring DOF cap: ring #%d (size=%d) has %d GP-searched axes "
                "(%d shared, %d optional) > cap=%d (n_flexible=%d - slack=%d) "
                "— demoting %d optional axis(es) to fixed (original-geometry) "
                "torsions: %s. Kept %d optional axis(es): %s. Shared "
                "(ring-fusion) axes are never demoted: %s.",
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
                    cycle_idx[pos] = -4  # demoted: fixed at original-geometry value, IK uses static reference
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
        ts_bond_vals = dihedrals[n_dih:] if self.state.vary_ts_bond_lengths else []
        ts_bond_targets = list(zip(self.state.ts_bonds, ts_bond_vals)) if self.state.vary_ts_bond_lengths else None

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
        self.state.last_opt_ok = preopt_status
        logger.info("Status of preopt: %s; LAST_OPT_OK: %s", preopt_status, self.state.last_opt_ok)
        if not preopt_status:
            self._dump_status_hook(dumping_value=self.state.last_opt_ok)
            skipped_structure_id = increase_structure_id()
            logger.error("Preopt finished with error! Structure with number %s will be skipped!", skipped_structure_id)
            return en + np.random.randn()
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
            ts_bond_targets=ts_bond_targets,
        )
        self.state.last_opt_ok = opt_status
        logger.info("Status of opt: %s; LAST_OPT_OK: %s", opt_status, self.state.last_opt_ok)
        logger.info("Optimized! En = %s", en)
        self._dump_status_hook(dumping_value=self.state.last_opt_ok)

        if not opt_status:
            skipped_structure_id = increase_structure_id()
            logger.error("Opt finished with error! Structure with number %s will be skipped!", skipped_structure_id)

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
        """Update dataset by parsing trajectory file."""
        logger.debug("Input dataset is: %s", dataset)
        parsed_data, last_point = parse_points_from_trj(
            trj_file_name=trj_filename,
            dihedrals=self.state.dihedral_ids,
            norm_en=self.state.norm_energy,
            save_structs=True,
            structures_path=self.state.structures_path,
            return_minima=True,
        )
        # Filter out frames where rings opened during ORCA optimization
        valid_degrees, valid_energies = [], []
        for coords, energy, xyz_block in parsed_data:
            if self.state.ik_loss is not None:
                if not _check_rings_intact(xyz_block, self.state.mol):
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
        if self.state.ik_loss is None or _check_rings_intact(last_xyz, self.state.mol):
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

            for (a, b) in self.state.ts_bonds:
                pair_key = f"{min(a, b)}-{max(a, b)}"
                cached = db.get_ts_bond_coefs(mol_hash, theory_level, pair_key)
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
                    lengths, energies = parse_bond_scan_results(inp_name)
                    coefs = calc_bond_coefs(lengths, energies)
                    logger.info("TS-bond pair (%d,%d) Morse coefs: De=%.3f a=%.3f re=%.3f c=%.3f", a, b, *coefs)
                    db.set_ts_bond_coefs(mol_hash, theory_level, pair_key, coefs)
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
            self.state.ensemble_processor = EnsembleProcessor(
                ensemble_path, dihedral_idxs=self.state.dihedral_ids,
            )
            if not self.state.ensemble_processor.energies:
                raise RuntimeError(
                    f"load_ensemble at {ensemble_path}: no conformers/energies found. "
                    f"norm_energy can't be calculated."
                )
            self.state.norm_energy = min(self.state.ensemble_processor.energies)
            logger.info(
                "Norm energy on %d structures: %s",
                len(self.state.ensemble_processor.energies), self.state.norm_energy,
            )
        else:
            theory_level = f"{config.orca_method}|charge={config.charge}|mult={config.spin_multiplicity}"
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
                ep = EnsembleProcessor(
                    self._resolve_ensemble_path(), dihedral_idxs=self.state.dihedral_ids,
                )
            
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
        """Remove all files/subfolders in working_folder except the mol file,
        the ensemble file (if provided), so restarting a run never gets
        contaminated by a previous run's outputs."""
        config = self.state.config
        working_folder = Path(self.state.working_folder)

        keep = set()
        mol_path = Path(config.mol_file_name)
        if not mol_path.is_absolute():
            mol_path = (working_folder / mol_path).resolve()
        keep.add(mol_path.resolve())

        if config.load_ensemble:
            ens_path = Path(config.load_ensemble)
            if not ens_path.is_absolute():
                ens_path = (working_folder / ens_path).resolve()
            keep.add(ens_path.resolve())

        if getattr(self.state, "config_path", None):
            keep.add(Path(self.state.config_path).resolve())
        keep.add(Path(self.state.db_file).resolve())
        
        ckpt_path = working_folder / f"{config.exp_name}_checkpoint.json"
        keep.add(ckpt_path.resolve())

        protected_suffixes = {".mol", ".log", ".db", ".yaml"}
        for item in working_folder.iterdir():
            if item.is_file() and item.suffix.lower() in protected_suffixes:
                keep.add(item.resolve())

        logger.info("clear_working_folder=True: wiping %s except %s", working_folder, keep)

        for item in working_folder.iterdir():
            resolved = item.resolve()
            if resolved in keep:
                logger.debug("Preserving %s (protected: config/ensemble/.db/.mol/.log/.yaml/checkpoint)", item)
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                logger.debug("Removed %s", item)
            except Exception:
                logger.exception("Failed to remove %s during working-folder cleanup", item)

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
        return dataset
    
    def run(self) -> None:
        """Execute the full Bayesian optimization loop."""
        config = self.state.config

        kernel, search_space, observer = self._build_model_and_acquisition()
        
        ckpt = self._resolve_checkpoint_snapshot()
        start_step = 1
        if ckpt is not None:
            dataset = self._dataset_from_checkpoint(ckpt)
            logger.info(
                "BO loop will continue from step %d (restored step %d).",
                start_step, ckpt["step"],
            )
            start_step = ckpt["step"] + 1
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

        deepest_minima = []
        early_termination_flag = False

        logger.info("MINIMA: %s", self.state.minima)

        if start_step > config.max_steps:
            logger.info(
                "start_step=%d > max_steps=%d — nothing left to run; saving results.",
                start_step, config.max_steps,
            )
            self._save_results(dataset)
            return
        
        for step in range(1, config.max_steps + 1):
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
                with open(status_file, "r") as file:
                    last_opt_status = json.load(file)
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

            if step < config.rolling_window_size:
                continue

            logger.debug("Checking termination criterion!")
            logger.debug("Acq vals in window: %s", logs['acq_vals'][max(0, step - config.rolling_window_size) : step])

            rolling_mean = np.mean(
                logs["acq_vals"][max(0, step - config.rolling_window_size) : step]
            )
            rolling_std = np.std(
                logs["acq_vals"][max(0, step - config.rolling_window_size) : step]
            )

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
                step >= config.rolling_window_size
                and rolling_std < config.rolling_std_threshold
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

        dbscan_labels = DBSCAN(
            eps=np.pi / 12,
            min_pts=1,
        ).fit_predict(np.asarray([cur[0] for cur in self.state.minima]))

        res = {int(label): (1e9, -1) for label in np.unique(dbscan_labels)}

        for i in range(len(self.state.minima)):
            cluster_id = dbscan_labels[i]
            if self.state.minima[i][1] < res[cluster_id][0]:
                res[cluster_id] = self.state.minima[i][1], i

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

    def _resolve_ensemble_path(self) -> str:
        """Returns absolute path to load_ensemble."""
        config = self.state.config
        load_ensemble_filename = Path(config.load_ensemble)
        if not load_ensemble_filename.is_absolute():
            load_ensemble_filename = Path(self.state.working_folder) / load_ensemble_filename
        return str(load_ensemble_filename)


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
