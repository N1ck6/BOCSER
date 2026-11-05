"""
Bayesian Optimization for Conformational Search - Refactored Class-Based Orchestrator

This module provides ConfSearchRunner, a class-based orchestrator that encapsulates
all state and behavior for the conformational search workflow.
"""

import os
import signal
import stat
import sys


def _redirected_logs() -> dict:
    """Regular files that stdout/stderr are redirected to (`> run.log 2>&1`), captured before
    any import writes to them: (st_dev, st_ino) -> (size at start, opened in append mode, path).
    With `>>` the size at start is the previous run's log; with `>` the shell has already emptied it.
    The path is known on Linux (/proc/self/fd), None elsewhere."""
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
        try:
            path = os.readlink(f"/proc/self/fd/{fd}")
            path = path if os.path.isfile(path) else None
        except OSError:
            path = None
        found[(st.st_dev, st.st_ino)] = (st.st_size, append, path)
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
import inspect
import atexit
import json
import re
import socket
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
    mmff_energy,
    load_last_optimized_structure_xyz_block,
    parse_points_from_trj,
    _qc_calcs_dir,
    check_structure,
    _save_broken_struct,
    raw1_to_with_h_canonical,
    with_h_canonical_to_raw1,
    raw1_to_heavy_canonical,
    _heavy_and_h_order,
    HARTREE_TO_KCAL,
    resolve_extra_constraints,
    _validate_extra_constraints_atoms,
    compute_mol_hash,
    build_reference_with_h_mol,
    submit_calc, wait_for_jobs,
    xyz_coords,
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
from feasibility import FeasibilityModel, FeasibilityWeightedAcquisition
from candidate_batch import BatchRecordingOptimizer
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
    # Last objective call: candidate rejected by the structure checks before the first ORCA run
    geometry_rejected: bool = False
    # Last objective call: (query point, energy), to rebuild the dataset if trieste fails after it
    last_observation: Optional[tuple] = None
    # Outcome of every evaluated candidate, "category:reason" -> count (run summary)
    outcomes: dict = field(default_factory=dict)
    n_unique_conformers: int = 0
    # Ring torsions left out of the GP by exclude_dof_slack: free, set by ring closure
    free_ring_dihedrals: list = field(default_factory=list)
    ring_atoms_list: list = field(default_factory=list)
    # IK loss of the acquisition: only rings whose torsions are all GP axes (or fixed)
    ik_acq_loss: Optional[Any] = None
    ik_acq_dihedrals_idxs: list = field(default_factory=list)
    feasibility: Optional[FeasibilityModel] = None
    # Last objective call: it gave an energy usable by the GP, and the BO point of the
    # geometry actually built (the GP observation goes there, not to the asked point)
    last_eval_valid: bool = False
    last_built_point: Optional[list] = None
    last_failure_reason: Optional[str] = None
    # Known structures to build candidates from: [(BO point, canonical coords)]
    templates: list = field(default_factory=list)
    # Lowest MMFF94 energy of the known structures (mmff_energy_window), None until one is known
    mmff_reference: Optional[float] = None
    ts_mode_bond: Optional[tuple] = None
    ts_bond_windows: dict = field(default_factory=dict)  # (a, b) -> (lo, hi), final OptTS check
    _dihedral_ids_finalized: bool = False


LOCK_FILE_NAME = "bocser.lock"


def _pid_alive(pid) -> bool:
    if os.name == "nt":
        return True  # os.kill(pid, 0) is not a harmless probe on Windows
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


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
        self.dry_run = False  # --dry-run: no ORCA, see dry_run_report()
        
        # Ensure working folder exists
        Path(working_folder).mkdir(parents=True, exist_ok=True)

    _RING_DOF_SLACK = 6
    _GP_TORSION_LENGTHSCALE = 0.7  # prior median of the per-torsion GP lengthscales

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

        The demoted axes are free: no constraint in the MMFF build or in the pre-opt, so ring
        closure sets them (6 torsions close a ring once the others are given). Freezing them
        at the .mol values left the ring over-constrained and tied every candidate to the
        .mol conformation.
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
                "ring, to free torsions set by ring closure: %s. Kept %d "
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

        self.state.free_ring_dihedrals = [list(d) for d in drop_idx_to_window.values()]

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
                    cycle_idx[pos] = -4  # demoted: free torsion, set by ring closure
                elif idx >= 0:
                    cycle_idx[pos] = old_to_new[idx]

        self.state.ik_loss_dihedrals_idxs = ik_loss_dihedrals_idxs

        logger.info(
            "Ring DOF cap: demoted %d axis(es) in total. dihedral_ids reduced "
            "from %d to %d entries.",
            len(drop_idx_to_window), orig_len, len(new_dihedral_ids),
        )

    def _prepare_acquisition_ik_loss(self) -> None:
        """IK loss for the "ik" acquisition: only rings whose torsions are all GP axes (or fixed).
        A ring with free (demoted) torsions closes itself in the MMFF build; its IK loss would
        hold those torsions at the .mol values and pull every candidate back to the .mol
        conformation. Whether such a ring can be built is learned by the feasibility model."""
        idxs = self.state.ik_loss_dihedrals_idxs
        self.state.ik_acq_loss, self.state.ik_acq_dihedrals_idxs = None, []
        if self.state.ik_loss is None or not idxs:
            return
        keep = [i for i, cycle in enumerate(idxs) if -4 not in cycle and any(j >= 0 for j in cycle)]
        if len(keep) == len(idxs):
            self.state.ik_acq_loss, self.state.ik_acq_dihedrals_idxs = self.state.ik_loss, idxs
            return
        logger.info(
            "IK loss of the acquisition: %d of %d ring(s) (rings with free torsions or no GP axis are left out)",
            len(keep), len(idxs),
        )
        if not keep:
            return
        try:
            self.state.ik_acq_loss = IKLoss.from_rdkit(
                self.state.mol, [self.state.ring_atoms_list[i] for i in keep],
                bond_length_overrides=getattr(self, "_ik_ts_lengths", None),
                ts_bond_ranges=getattr(self, "_ik_ts_ranges", None),
            )
            self.state.ik_acq_dihedrals_idxs = [idxs[i] for i in keep]
        except Exception:
            logger.exception("Could not build the IK loss of the acquisition — the acquisition runs without it.")

    def _resolve_ts_mode_bond(self, config: ConfSearchConfig) -> Optional[tuple]:
        """Canonical pair for TS_Mode {B a b} in the final OptTS: ts_mode_bond from the config,
        else the shortest TS bond (fixed TS-bond length, else the .mol length)."""
        if not config.ts or config.use_grass or not self.state.ts_bonds:
            if config.ts_mode_bond:
                logger.warning("ts_mode_bond is ignored: it needs ts: true, ts_bonds and no GRASS.")
            return None
        if config.ts_mode_bond:
            try:
                pair = tuple(raw1_to_with_h_canonical(a, self.state.mol_file_name) for a in config.ts_mode_bond)
            except ValueError as e:
                raise ConfigError(f"ts_mode_bond {config.ts_mode_bond}: atom not in the .mol") from e
            source = "ts_mode_bond"
        else:
            conf = build_reference_with_h_mol(self.state.mol_file_name).GetConformer()
            fixed = {frozenset(p): length for p, length in self.state.fixed_ts_bond_targets}
            lengths = {
                tuple(p): fixed.get(frozenset(p), Chem.rdMolTransforms.GetBondLength(conf, *p))
                for p in self.state.ts_bonds
            }
            pair = min(lengths, key=lengths.get)
            source = f"shortest TS bond, {lengths[pair]:.3f} A"
        logger.info(
            "OptTS follows the Hessian mode of bond %s (1-idx .mol; %s)",
            tuple(with_h_canonical_to_raw1(a, self.state.mol_file_name) for a in pair), source,
        )
        return pair

    def _resolve_ts_bond_windows(self) -> dict:
        """Allowed TS-bond lengths of the final OptTS geometry: the range over load_ensemble
        (else over the .mol and the fixed TS-bond lengths) widened by ts_bond_result_margin."""
        config = self.state.config
        margin = float(config.ts_bond_result_margin)
        if not config.ts or not self.state.ts_bonds or margin <= 0:
            return {}
        ref = {tuple(p): [] for p in self.state.ts_bonds}
        source = ".mol and fixed TS-bond lengths"
        ep = self.state.ensemble_processor
        n_atoms = build_reference_with_h_mol(self.state.mol_file_name).GetNumAtoms()
        if ep is not None and ep.coords and len(ep.coords[0]) == n_atoms:
            for coords in ep.coords:
                c = np.asarray(coords, dtype=float)
                for a, b in ref:
                    ref[(a, b)].append(float(np.linalg.norm(c[a] - c[b])))
            source = f"{len(ep.coords)} load_ensemble conformers"
        else:
            conf = build_reference_with_h_mol(self.state.mol_file_name).GetConformer()
            for a, b in ref:
                ref[(a, b)].append(Chem.rdMolTransforms.GetBondLength(conf, a, b))
            for (a, b), length in self.state.fixed_ts_bond_targets:
                key = (a, b) if (a, b) in ref else (b, a)
                ref[key].append(float(length))
        windows = {pair: (max(0.0, min(v) - margin), max(v) + margin) for pair, v in ref.items()}
        for (a, b), (lo, hi) in windows.items():
            logger.info(
                "Final OptTS check: TS bond %s must stay within [%.3f, %.3f] A (%s +- %.2f A)",
                tuple(with_h_canonical_to_raw1(i, self.state.mol_file_name) for i in (a, b)), lo, hi, source, margin,
            )
        return windows

    def acquire_lock(self) -> None:
        """One BOCSER run per qc_calcs/ folder (next to the .mol): two runs there overwrite
        each other's ORCA inputs, outputs and scratch files. A lock left by a dead process
        on this host is removed; a lock from another host has to be deleted by hand."""
        qc_dir = Path(self.state.config.mol_file_name).resolve().parent / "qc_calcs"
        qc_dir.mkdir(parents=True, exist_ok=True)
        path = qc_dir / LOCK_FILE_NAME
        host = socket.gethostname()
        for _ in range(2):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                try:
                    owner = json.loads(path.read_text())
                except (OSError, ValueError):
                    owner = {}
                if owner.get("host") == host and not _pid_alive(owner.get("pid")):
                    logger.warning("Removing a stale lock %s (process %s is not running)", path, owner.get("pid"))
                    path.unlink(missing_ok=True)
                    continue
                raise RuntimeError(
                    f"Another BOCSER run uses {qc_dir}: host {owner.get('host', '?')}, pid {owner.get('pid', '?')}, "
                    f"folder {owner.get('working_folder', '?')}, started {owner.get('started', '?')}. Two runs with "
                    f"the same .mol folder overwrite each other's ORCA files. Wait for that run to finish or put "
                    f"the .mol into a folder of its own. If no run is active, delete {path}."
                )
            with os.fdopen(fd, "w") as fh:
                json.dump({
                    "host": host, "pid": os.getpid(), "working_folder": self.state.working_folder,
                    "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, fh)
            self._lock_path = path
            atexit.register(self.release_lock)
            logger.info("Lock taken: %s", path)
            return
        raise RuntimeError(f"Could not take the lock {path}")

    def release_lock(self) -> None:
        path = getattr(self, "_lock_path", None)
        if path is None:
            return
        try:
            if json.loads(path.read_text()).get("pid") == os.getpid():
                path.unlink()
        except (OSError, ValueError):
            pass
        self._lock_path = None

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
        self.state.geometry_rejected = False
        self.state.last_observation = None
        self.state.last_eval_valid = False
        self.state.last_built_point = None
        self.state.last_failure_reason = None
        en = self._calc_point_energy(dihedrals)
        asked = [float(v) for v in self.state.asked_points[-1]]
        self.state.last_observation = (asked, float(en))
        self._record_feasibility(asked)
        return en

    # A candidate built farther than this from the asked point (max over the BO torsions,
    # radians, ~30 deg) is a different point: the asked one counts as not buildable
    _BUILD_TOLERANCE = 0.5

    def _torsion_deviation(self, a, b) -> float:
        """Largest absolute difference over the BO torsions, radians."""
        n_dih = len(self.state.dihedral_ids)
        if n_dih == 0:
            return 0.0
        d = np.abs(np.asarray(a[:n_dih], dtype=float) - np.asarray(b[:n_dih], dtype=float)) % (2 * np.pi)
        return float(np.max(np.minimum(d, 2 * np.pi - d)))

    # ORCA failures that say nothing about the geometry (no output file, unreadable output):
    # such a candidate gets no label, so a queue or file-system hiccup does not close a region
    _INFRA_FAILURES = ("no_out_file", "parse_error")

    def _record_feasibility(self, asked: list) -> None:
        """Label the evaluated candidate for the feasibility model (use_feasibility_model).
        Candidates the MMFF build could not bring near the asked torsions are rejected before
        ORCA (mmff_off_target), so a labelled point is always the asked one."""
        fm = self.state.feasibility
        if fm is None:
            return
        if not self.state.last_eval_valid:
            if self.state.last_failure_reason in self._INFRA_FAILURES:
                logger.warning(
                    "Candidate failed for a technical reason (%s) — not labelled in the feasibility model.",
                    self.state.last_failure_reason,
                )
                return
            fm.add(asked, False)
        else:
            fm.add(asked, True)
        logger.info(
            "Feasibility model: %d labels, %d failed; P at this point is now %.2f",
            len(fm), fm.n_failed, fm.probability_at(asked),
        )

    def _nearest_template(self, point) -> Optional[list]:
        """Coordinates of the known structure closest to `point` in the BO torsions
        (build_from_nearest_conformer), or None to build from the .mol."""
        if not self.state.config.build_from_nearest_conformer or not self.state.templates:
            return None
        n_dih = len(self.state.dihedral_ids)
        target = np.asarray(point, dtype=float)
        pts = np.asarray([p for p, _ in self.state.templates], dtype=float)
        d2 = np.sum((2.0 * np.sin(0.5 * (pts[:, :n_dih] - target[None, :n_dih]))) ** 2, axis=1)
        if pts.shape[1] > n_dih:
            ts_scale = max(float(self.state.config.ts_bond_kernel_lengthscale), 1e-6)
            d2 += np.sum(((pts[:, n_dih:] - target[None, n_dih:]) / ts_scale) ** 2, axis=1)
        best = int(np.argmin(d2))
        logger.info(
            "Candidate built from known structure #%d of %d (largest BO-torsion difference %.0f deg)",
            best, len(self.state.templates), np.degrees(self._torsion_deviation(pts[best], target)),
        )
        return self.state.templates[best][1]

    def _add_template(self, point, coords) -> None:
        """Known structure (canonical-order coordinates) and its BO point, for _nearest_template."""
        coords = np.asarray(coords, dtype=float)
        if getattr(self, "_n_atoms_with_h", None) is None:
            self._n_atoms_with_h = build_reference_with_h_mol(self.state.mol_file_name).GetNumAtoms()
        if len(coords) != self._n_atoms_with_h or len(point) != self.state.search_dim:
            return
        self.state.templates.append((np.asarray(point, dtype=float), coords.tolist()))
        self._note_mmff_reference(coords)

    def _note_mmff_reference(self, coords) -> None:
        """Lower the MMFF94 reference (mmff_energy_window) with a known good structure."""
        if self.state.config.mmff_energy_window is None:
            return
        block = "\n".join(f"X {x:.6f} {y:.6f} {z:.6f}" for x, y, z in np.asarray(coords, dtype=float))
        try:
            energy = mmff_energy(block, self.state.mol_file_name, ts_bonds=self.state.ts_bonds)
        except Exception:
            logger.debug("MMFF94 energy of a known structure failed", exc_info=True)
            return
        self._note_mmff_energy(energy)

    def _note_mmff_energy(self, energy: Optional[float]) -> None:
        if energy is None or not np.isfinite(energy):
            return
        if self.state.mmff_reference is None or energy < self.state.mmff_reference:
            self.state.mmff_reference = float(energy)
            logger.info(
                "MMFF94 reference energy is now %.1f kcal/mol: candidates above %.1f are rejected before ORCA",
                energy, energy + self.state.config.mmff_energy_window,
            )

    def _mmff_energy_limit(self) -> Optional[float]:
        window = self.state.config.mmff_energy_window
        if window is None or self.state.mmff_reference is None:
            return None
        return self.state.mmff_reference + float(window)

    def _ts_bond_targets(self, point) -> Optional[list]:
        """TS-bond lengths held in the pre-optimization only: sampled by BO, or fixed
        (user values / lowest-energy ensemble conformer, see _resolve_fixed_ts_bond_targets)."""
        n_dih = len(self.state.dihedral_ids)
        if self.state.vary_ts_bond_lengths:
            return list(zip(self.state.ts_bonds, point[n_dih:]))
        if self.state.fixed_ts_bond_targets:
            return list(self.state.fixed_ts_bond_targets)
        return None

    def dry_run_report(self) -> None:
        """--dry-run: write the pre-opt and full-opt ORCA inputs of one candidate (the .mol
        geometry at its own torsions) to the working folder, without running ORCA."""
        point = [float(v) for v in self._extract_dofs_values(self.state.mol).numpy()[0]]
        n_dih = len(self.state.dihedral_ids)
        folder = Path(self.state.working_folder)
        pre_inp = folder / f"{self.state.exp_name}_dry_run_preopt.inp"
        opt_inp = folder / f"{self.state.exp_name}_dry_run_opt.inp"
        common = dict(
            ik_loss=self.state.ik_loss, original_mol=self.state.mol, broken_structs_dir=None,
            success_out_dir=None, ts_bonds=self.state.ts_bonds,
            ts_bond_max_length=self.state.config.ts_bond_max_length,
            fixed_dihedrals=self.state.fixed_dihedrals, extra_constraints=self.state.extra_constraints,
        )
        failure = {}
        _, ok = calc_energy(
            self.state.mol_file_name, list(zip(self.state.dihedral_ids, point[:n_dih])), 0.0,
            constrained_opt=True, ts_bond_targets=self._ts_bond_targets(point), failure_info=failure,
            write_input_only=str(pre_inp), **common,
        )
        if ok:
            # Geometry here is the pre-opt input; in a real run the full opt starts from the pre-opt result
            calc_energy(
                self.state.mol_file_name, list(zip(self.state.dihedral_ids, point[:n_dih])), 0.0,
                write_input_only=str(opt_inp), ts_mode_bond=self.state.ts_mode_bond, **common,
            )
        unknown = list(getattr(self.state.config, "unknown_keys", []))
        lines = [
            "===== BOCSER dry run =====",
            f"GP search dimensions: {self.state.search_dim} ({n_dih} torsions"
            + (f", {len(self.state.ts_bonds)} TS-bond lengths)" if self.state.vary_ts_bond_lengths else ")"),
            "Test candidate (.mol geometry): "
            + ("passes the structure check" if ok else f"REJECTED by the structure check ({failure.get('reason')})"),
        ]
        if ok:
            lines.append(f"ORCA inputs written: {pre_inp.name}, {opt_inp.name}")
        if unknown:
            lines.append(f"Unknown config keys (ignored): {', '.join(unknown)}")
        lines.append("norm_energy and torsion scans not in the cache are not computed in a dry run.")
        lines.append("==========================")
        logger.info("\n".join(lines))
        print("[BOCSER] Dry run finished, see the log" + (f"; inputs in {folder}" if ok else ""), flush=True)

    def _count_outcome(self, category: str, reason: str = "") -> None:
        key = f"{category}:{reason}" if reason else category
        self.state.outcomes[key] = self.state.outcomes.get(key, 0) + 1

    def _calc_point_energy(self, dihedrals: list[float]) -> float:
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
        ts_bond_targets = self._ts_bond_targets(dihedrals)

        logger.info("Optimizing constrained struct")
        preopt_failure = {}
        build_info = {}
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
                failure_info=preopt_failure,
                start_coords=self._nearest_template(dihedrals),
                build_info=build_info,
                max_build_deviation=self._BUILD_TOLERANCE,
                mmff_energy_limit=self._mmff_energy_limit(),
            )
        except Exception:
            logger.exception("calc_energy (preopt) fell with unexpected exception — conting point as broken.")
            en, preopt_status = self.state.config.broken_struct_energy, False
            # Dump the status file immediately, right to avoidcrash on exception
            self._dump_status_hook(dumping_value=False)
        self.state.last_built_point = self._built_point(dihedrals, build_info)
        self.state.last_opt_ok = preopt_status
        if preopt_status:
            self._note_mmff_energy(build_info.get("mmff_energy"))
        else:
            self.state.last_failure_reason = preopt_failure.get("reason")
        logger.info("Status of preopt: %s; LAST_OPT_OK: %s", preopt_status, self.state.last_opt_ok)
        if not preopt_status:
            self._dump_status_hook(dumping_value=self.state.last_opt_ok)
            skipped_structure_id = increase_structure_id()
            self.state.geometry_rejected = preopt_failure.get("stage") == "geometry"
            self._count_outcome(
                "rejected_before_orca" if self.state.geometry_rejected else "preopt_failed",
                preopt_failure.get("reason") or "error",
            )
            if self.state.geometry_rejected:
                logger.error(
                    "Candidate %s rejected by the structure check before ORCA — %s, the BO step is "
                    "not counted.", skipped_structure_id,
                    "labelled infeasible (not added to the GP)" if self.state.feasibility is not None
                    else "broken_struct_energy goes to the GP",
                )
            else:
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
                ts_mode_bond=self.state.ts_mode_bond,
            )
        except Exception:
            # e.g. sbatch -W reports a non-zero ORCA exit code: lose this point, not the whole run
            logger.exception("calc_energy (full opt) fell with unexpected exception — counting point as broken.")
            en, opt_status = self.state.config.broken_struct_energy, False
        if opt_status:
            opt_status = self._check_full_opt_result(full_opt_failure)
        if not opt_status:
            self.state.last_failure_reason = full_opt_failure.get("reason")
        self.state.last_opt_ok = opt_status
        logger.info("Status of opt: %s; LAST_OPT_OK: %s", opt_status, self.state.last_opt_ok)
        logger.info("Optimized! En = %s", en)
        self._dump_status_hook(dumping_value=self.state.last_opt_ok)

        if opt_status:
            self._count_outcome("success")
            self.state.last_eval_valid = True
        else:
            self._count_outcome(
                {"geometry": "opt_input_rejected", "result": "opt_result_broken"}.get(
                    full_opt_failure.get("stage"), "opt_failed"),
                full_opt_failure.get("reason") or "error",
            )
            skipped_structure_id = increase_structure_id()
            logger.error("Opt finished with error! Structure with number %s will be skipped!", skipped_structure_id)
            if self.state.config.use_preopt_energy_on_opt_failure and full_opt_failure.get("stage") in ("orca", "result"):
                # The pre-opt energy at this point is a valid observation (it is what the first
                # trajectory frame gives on success); only a rejected input geometry is not.
                logger.warning(
                    "Full optimization failed (%s) — using the pre-optimization energy %.3f kcal/mol "
                    "for this point instead of broken_struct_energy.",
                    "ORCA" if full_opt_failure["stage"] == "orca" else "broken final geometry", preopt_en,
                )
                self.state.last_eval_valid = True
                return preopt_en

        return en + ((not opt_status) * np.random.randn())

    def _built_point(self, asked: list, build_info: dict) -> list:
        """BO point of the geometry sent to the pre-opt (the asked point if it was not built)."""
        n_dih = len(self.state.dihedral_ids)
        built = [float(v) for v in asked]
        values = build_info.get("dihedrals")
        if values is None or len(values) != n_dih:
            return built
        built[:n_dih] = [float(v) % (2 * np.pi) for v in values]
        if self.state.vary_ts_bond_lengths:
            lo, hi = self.state.config.ts_bond_min_length, self.state.config.ts_bond_max_length
            lengths = build_info.get("ts_bond_lengths", {})
            for k, pair in enumerate(self.state.ts_bonds):
                if tuple(pair) in lengths:
                    built[n_dih + k] = float(np.clip(lengths[tuple(pair)], lo, hi))
        deviation = self._torsion_deviation(asked, built)
        logger.info("Built candidate: largest BO-torsion difference from the asked point %.1f deg", np.degrees(deviation))
        return built

    def _check_full_opt_result(self, failure_info: dict) -> bool:
        """Run the structure checks on the final geometry of a successful full optimization.
        A broken result counts as a failed full opt (failure_info["stage"] = "result")."""
        try:
            xyz_final = load_last_optimized_structure_xyz_block(self.state.mol_file_name)
            reason, atoms_raw1 = check_structure(
                xyz_final, self.state.mol_file_name, original_mol=self.state.mol,
                ts_bonds=self.state.ts_bonds, check_rings=self.state.ik_loss is not None,
            )
            if reason is None:
                reason, atoms_raw1 = self._left_ts_region(xyz_final)
        except Exception:
            logger.exception("Could not check the final full-opt geometry — keeping the result.")
            return True
        if reason is None:
            return True
        logger.error("Final full-opt geometry failed the structure check (%s, atoms %s).", reason, atoms_raw1)
        _save_broken_struct(xyz_final, self.state.broken_structs_path, f"opt_result_{reason}", atoms_raw1)
        failure_info["stage"] = "result"
        failure_info["reason"] = reason
        return False

    def _left_ts_region(self, xyz_block: str) -> tuple:
        """("left_ts_region", atoms) when a TS bond of the final OptTS geometry is outside its
        ts_bond_result_margin window (the OptTS fell to the product or the reactant)."""
        if not self.state.ts_bond_windows:
            return None, None
        coords = xyz_coords(xyz_block)
        for (a, b), (lo, hi) in self.state.ts_bond_windows.items():
            length = float(np.linalg.norm(coords[a] - coords[b]))
            if not lo <= length <= hi:
                atoms = tuple(with_h_canonical_to_raw1(i, self.state.mol_file_name) for i in (a, b))
                logger.error(
                    "OptTS left the TS region: TS bond %s is %.3f A, allowed [%.3f, %.3f] A "
                    "(ts_bond_result_margin=%.2f)", atoms, length, lo, hi, self.state.config.ts_bond_result_margin,
                )
                return "left_ts_region", atoms
        return None, None

    def _func_objective(self, cur: tf.Tensor) -> tf.Tensor:
        """Defines the objective function for the Bayesian optimization."""
        cur_np = cur.numpy() if hasattr(cur, "numpy") else np.asarray(cur)
        results = [[self._calc_point(x)] for x in cur_np]
        return tf.constant(results, dtype=tf.float64)

    def _extract_dofs_values(self, m: Chem.Mol) -> tf.Tensor:
        """Extract dihedral angles from a molecule conformer, in the BO convention: RDKit
        dihedrals in [0, 2*pi), as calc.dihedral_angle gives for the ensemble and trajectories
        (the old -GetDihedralRad gave the mirror torsions and could fall outside the box)."""
        conf = m.GetConformer()
        dihedral_vals = [
            Chem.rdMolTransforms.GetDihedralRad(conf, *self.state.dihedral_ids[i]) % (2 * np.pi)
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

    def _last_trj_path(self) -> str:
        return str(_qc_calcs_dir(self.state.mol_file_name) / (Path(self.state.mol_file_name).stem + "_trj.xyz"))

    def _trj_dataset_part(self, trj_filename: str) -> Optional[Dataset]:
        """GP observations from the trajectory of a successful full optimization, which replace
        the energy at the asked point. None (caller keeps the asked point) if the trajectory
        can't be read or every frame fails the structure checks."""
        try:
            return self._parse_trj_dataset_part(trj_filename)
        except Exception:
            logger.exception(
                "Failed to read trajectory %s — the GP keeps the full-opt energy at the asked point.",
                trj_filename,
            )
            return None

    def _parse_trj_dataset_part(self, trj_filename: str) -> Optional[Dataset]:
        """Default: first frame + cluster representatives. With augment_gp_from_orca_trj,
        more frames are kept but thinned in BO coordinates (consecutive optimization
        steps are strongly correlated, not independent observations).
        """
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
        def frame_problem(xyz_block):
            reason, _ = check_structure(
                xyz_block, self.state.mol_file_name, original_mol=self.state.mol,
                ts_bonds=self.state.ts_bonds, check_rings=self.state.ik_loss is not None,
            )
            return reason

        # Filter out frames with broken bonds, clashes or opened rings
        valid_degrees, valid_energies = [], []
        for coords, energy, xyz_block in parsed_data:
            reason = frame_problem(xyz_block)
            if reason is not None:
                logger.warning("Discarding trajectory frame (energy=%.3f kcal/mol): %s", energy, reason)
                continue
            valid_degrees.append(coords)
            valid_energies.append(energy)

        if not valid_degrees:
            logger.warning("All trajectory frames from %s failed the structure check.", trj_filename)
            return None

        # Validate last_point before saving to minima list.
        last_xyz = last_point["xyz_block"]
        last_reason = frame_problem(last_xyz)
        if last_reason is None:
            minima_file = os.path.join(
                self.state.working_folder,
                f"{self.state.exp_name}_minima/{len(self.state.minima)}.xyz"
            )
            with open(minima_file, "w") as minima_xyz_writer:
                minima_xyz_writer.write(last_xyz)
            self.state.minima.append((last_point["coords"], last_point["rel_en"]))
            self._add_template(last_point["coords"], xyz_coords(last_xyz))
        else:
            logger.warning(
                "Last trajectory point (energy=%.3f kcal/mol) discarded: %s. Not added to minima.",
                last_point["rel_en"], last_reason,
            )

        logger.info("GP: %d trajectory frame(s) added in place of the asked point", len(valid_degrees))
        logger.debug("Valid frames: %d / %d", len(valid_degrees), len(parsed_data))
        logger.debug("Degrees: %s\nEnergies: %s", valid_degrees, valid_energies)

        self.state.global_degrees.extend(valid_degrees)

        return Dataset(
            tf.constant(valid_degrees, dtype="double"),
            tf.constant(valid_energies, dtype="double").reshape(len(valid_energies), 1),
        )

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
        
        if self.dry_run:
            logger.info("Dry run: the working folder is left as it is, ORCA is not started.")
        elif config.clear_working_folder:
            self._clear_working_folder()
        else:
            self._drop_previous_log()

        logger.info("Performing conf. search with config: %s", config)
        final_opt_constraints = [
            name for name, on in (("BO dihedrals", config.constrain_bo_dihedrals_in_final_opt),
                                  ("fixed torsions + extra_constraints", config.constrain_final_opt)) if on
        ]
        logger.info(
            "Full optimization constraints: %s (pre-opt holds all of them)",
            ", ".join(final_opt_constraints) if final_opt_constraints else "none",
        )

        # Propagate configuration to central config manager and internal state
        config_manager.set_config(config)
        
        mol_file = Path(config.mol_file_name)
        if not mol_file.is_absolute():
            mol_file = mol_file.resolve()
        self.state.mol_file_name = str(mol_file)

        _validate_extra_constraints_atoms(config)
        self.state.extra_constraints = resolve_extra_constraints(config.extra_constraints, self.state.mol_file_name)
        if self.state.extra_constraints:
            logger.info("Loaded %d user extra_constraints", len(self.state.extra_constraints))

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
            logger.info("TS bonds (config 1-idx -> canonical 0-idx): %s -> %s", config.ts_bonds, self.state.ts_bonds)

        # Must run before vary_ts_bond_lengths is read below (it may be switched off here)
        self.state.fixed_ts_bond_targets = self._resolve_fixed_ts_bond_targets(config)
        self.state.ts_mode_bond = self._resolve_ts_mode_bond(config)

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
            submit_scans=not self.dry_run,
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
                # The ring closure must use the TS-bond lengths the pre-opt holds, not the .mol ones
                ts_lengths = {frozenset(pair): float(length) for pair, length in self.state.fixed_ts_bond_targets}
                ring_bonds = {frozenset((r[k], r[(k + 1) % len(r)])) for r in ring_atoms_list for k in range(len(r))}
                for pair in ts_lengths.keys() & ring_bonds:
                    logger.info(
                        "IK loss: ring TS bond %s uses the pre-opt length %.4f A instead of the .mol length",
                        tuple(with_h_canonical_to_raw1(a, self.state.mol_file_name) for a in sorted(pair)),
                        ts_lengths[pair],
                    )
                # A TS bond in a ring may close it at any length in [ts_bond_min_length, ts_bond_max_length]
                ts_ranges = {
                    frozenset(pair): (config.ts_bond_min_length, config.ts_bond_max_length)
                    for pair in self.state.ts_bonds
                } if config.ts else {}
                for pair in ts_ranges.keys() & ring_bonds:
                    logger.info(
                        "IK loss: ring TS bond %s closes the ring at any length in [%.2f, %.2f] A",
                        tuple(with_h_canonical_to_raw1(a, self.state.mol_file_name) for a in sorted(pair)),
                        *ts_ranges[pair],
                    )
                self.state.ik_loss = IKLoss.from_rdkit(
                    self.state.mol, ring_atoms_list, bond_length_overrides=ts_lengths, ts_bond_ranges=ts_ranges,
                )
                self._ik_ts_ranges = ts_ranges
                self.state.ik_loss_dihedrals_idxs = ik_loss_dihedrals_idxs
                self.state.ring_atoms_list = ring_atoms_list
                self._ik_ts_lengths = ts_lengths
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
        self._prepare_acquisition_ik_loss()

        # Mark dihedral_ids as finalized BEFORE anything downstream is allowed to read it.
        self.state._dihedral_ids_finalized = True

        logger.info("Dihedral ids: %s", self.state.dihedral_ids)
        logger.info("Mean func coefs: %s", self.state.mean_func_coefs)
        n_flat = sum(1 for coefs in self.state.mean_func_coefs if not any(coefs))
        logger.info(
            "GP axes: %d — %d with a torsion-scan profile, %d with a flat mean "
            "(ring torsions are not scanned, or the scan failed)",
            len(self.state.mean_func_coefs), len(self.state.mean_func_coefs) - n_flat, n_flat,
        )

        self._require_dihedral_ids_finalized("n_dihedral computation")
        n_dihedral = len(self.state.dihedral_ids)

        for cycle_idx in ik_loss_dihedrals_idxs:
            for idx in cycle_idx:
                if idx >= 0 and idx >= n_dihedral:
                    raise RuntimeError(
                        f"IK dihedral index {idx} is out of range n_dihedral={n_dihedral}. "
                        f"ik_loss_dihedrals_idxs={ik_loss_dihedrals_idxs}. "
                        f"Check frag_key_to_position in get_ring_dihedrals() and filter "
                        f"fixed_double_bonds in get_interesting_frags()."
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

            if pairs_to_scan and self.dry_run:
                logger.info("Dry run: %d TS-bond scan(s) not submitted, flat mean used for them", len(pairs_to_scan))
                for (_, _, pair_key, _) in pairs_to_scan:
                    cached_coefs[pair_key] = (0.0, 1.0, (config.ts_bond_min_length + config.ts_bond_max_length) / 2, 0.0)
                pairs_to_scan = []

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
                        f"fixed_double_bonds in get_interesting_frags()."
                    )

    def _build_model_and_acquisition(self) -> Tuple[Any, Any, Any]:
        """Build GPR model, BO optimizer, and acquisition rule."""
        potential_func = PotentialFunction(self.state.mean_func_coefs)

        n_dih = len(self.state.dihedral_ids)
        n_ts = len(self.state.ts_bonds) if self.state.vary_ts_bond_lengths else 0
        dih_dims = list(range(n_dih))
        ts_dims = list(range(n_dih, n_dih + n_ts))

        # Periodic term: one lengthscale per torsion (ARD), so the GP can learn which torsions
        # change the energy quickly. The prior (median 0.7, x/e^0.5) replaces the old
        # LogNormal(0.005, 0.001), which pinned one lengthscale for all torsions at 1.0.
        # In the Periodic kernel r = sin(dTheta/2) / l: with l = 0.7 the correlation is 0.77
        # at 60 deg and 0.46 at 120 deg (gauche and anti are told apart).
        kernel = (
            gpflow.kernels.White(0.001)
            + gpflow.kernels.Periodic(
                gpflow.kernels.RBF(
                    variance=0.07,
                    lengthscales=[self._GP_TORSION_LENGTHSCALE] * n_dih,
                    active_dims=dih_dims,
                ),
                period=[2 * np.pi for _ in dih_dims],
            )
            + TransformKernel(
                potential_func,
                gpflow.kernels.RBF(
                    variance=0.12,
                    lengthscales=1.0,
                    active_dims=dih_dims,
                ),
            )
        )

        kernel.kernels[1].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(
            loc=tf.constant(np.log(self._GP_TORSION_LENGTHSCALE), dtype=tf.float64),
            scale=tf.constant(0.5, dtype=tf.float64),
        )
        # Lengthscale over the scan-profile energies (kcal/mol): same centre (1.0) as before,
        # no longer pinned
        kernel.kernels[2].base_kernel.lengthscales.prior = tfp.distributions.LogNormal(
            loc=tf.constant(0.0, dtype=tf.float64), scale=tf.constant(0.5, dtype=tf.float64)
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
            # BO energies are E_orca * HARTREE_TO_KCAL - norm_energy, so the ensemble must hold absolute
            # kcal/mol energies (< -9e4 for any molecule with a heavy-atom torsion). Hartree or
            # relative values would silently mix scales with every later ORCA result.
            if max(self.state.ensemble_processor.energies) > -1e4:
                raise RuntimeError(
                    f"load_ensemble at {ensemble_path}: energies (max={max(self.state.ensemble_processor.energies):.4f}) "
                    f"are not absolute energies in kcal/mol. Put the absolute energy in kcal/mol "
                    f"(Hartree * {HARTREE_TO_KCAL}) as the last token of each comment line."
                )
            self.state.norm_energy = min(self.state.ensemble_processor.energies)
            logger.info(
                "Norm energy on %d structures: %s",
                len(self.state.ensemble_processor.energies), self.state.norm_energy,
            )
        else:
            theory_level = f"{config.orca_method}|charge={config.charge}|mult={config.spin_multiplicity}"
            # norm_energy comes from an Opt/OptTS of the input geometry (under the hard constraints
            # with constrain_final_opt), so both change the cached value
            if config.ts:
                theory_level += "|ts"
            hard_constraints = sorted(
                [("dihedral", tuple(atoms), round(float(np.rad2deg(val)), 1)) for atoms, val in self.state.fixed_dihedrals]
                + [(c.type, tuple(c.atoms), round(float(c.value), 3)) for c in self.state.extra_constraints]
            ) if config.constrain_final_opt else []
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
                    ts_mode_bond=self.state.ts_mode_bond,
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
            if self.state.feasibility is not None:
                for point in dataset.query_points.numpy():
                    self.state.feasibility.add(point, True)
                logger.info("Feasibility model seeded with %d ensemble conformers", len(self.state.feasibility))
        else:
            use_fm = self.state.feasibility is not None
            # Rejected embeddings are cheap (no ORCA): with the feasibility model they are
            # retried, like rejected BO candidates, so the GP still starts from num_initial_points
            max_retries = config.max_geometry_rejections_in_a_row or 50
            n_done, n_retried, idx = 0, 0, -1
            while n_done < config.num_initial_points:
                idx += 1
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
                if use_fm and not self.state.last_eval_valid:
                    if self.state.geometry_rejected and n_retried < max_retries:
                        n_retried += 1
                        logger.warning("Initial point rejected before ORCA — trying another embedding (%d)", n_retried)
                        continue
                    n_done += 1
                    logger.warning("Initial point gave no energy — kept only in the feasibility model.")
                    continue
                n_done += 1
                part = self._trj_dataset_part(self._last_trj_path()) if self.state.last_opt_ok else None
                if part is None:
                    if not self.state.last_opt_ok:
                        logger.warning(
                            "Optimization didn't finish well. Continue only with the returned energy in required point: %s",
                            observed_point,
                        )
                    part = self._move_last_to_built_point(observed_point)
                dataset = part if dataset is None else dataset + part
            if dataset is None:
                raise RuntimeError(
                    f"None of the {config.num_initial_points} initial points gave an energy (see the log above "
                    f"for each). The GP needs at least one: check the .mol geometry and the ORCA settings, "
                    f"or start from load_ensemble."
                )
            logger.info(
                "Initial dataset observed! %s minima observed, total %s points has been collected!",
                config.num_initial_points,
                dataset.query_points.shape[0],
            )

        return dataset

    def _move_last_to_built_point(self, dataset: Dataset) -> Dataset:
        """The energy of the last candidate belongs to the geometry that was built, which can
        differ from the asked point (ring closure in the MMFF build): move the observation there."""
        built = self.state.last_built_point
        if built is None or not self.state.last_eval_valid or dataset.query_points.shape[0] == 0:
            return dataset
        query_points = dataset.query_points.numpy().copy()
        if np.allclose(query_points[-1], built):
            return dataset
        query_points[-1] = built
        return Dataset(tf.constant(query_points, dtype=tf.float64), dataset.observations)

    def _drop_invalid_observations(self, dataset: Dataset) -> Dataset:
        """Feasibility model on: broken_struct_energy values are not energies — take them out
        of the GP data (old checkpoints and runs without the model kept them there)."""
        if self.state.feasibility is None:
            return dataset
        obs = dataset.observations.numpy().reshape(-1)
        keep = np.array([np.isfinite(e) and not _is_broken(float(e), self.state.config.broken_struct_energy) for e in obs])
        if keep.all():
            return dataset
        if not keep.any():
            raise RuntimeError(
                "Every GP observation is broken_struct_energy: no valid energy to start from. "
                "Check the .mol geometry and the ORCA settings."
            )
        logger.warning("%d broken_struct_energy observation(s) taken out of the GP data", int((~keep).sum()))
        return Dataset(
            tf.boolean_mask(dataset.query_points, keep), tf.boolean_mask(dataset.observations, keep),
        )

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
        keep.add(qc_dir / LOCK_FILE_NAME)  # held by this run

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
                    moved += self._split_own_log(item, backup_dir / item.name, own[0], own[1])
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
    def _split_own_log(path: Path, dest: Optional[Path], start_size: int, append: Optional[bool]) -> int:
        """The current run writes its output to `path`. Copy the part written before this
        process started (previous run) to `dest` (None = drop it) and keep only this run's
        lines in `path`. Returns 1 if something was backed up."""
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
        if dest is not None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "rb") as src, open(dest, "wb") as out:
                out.write(src.read(start_size))
        if append is not True:
            if dest is None:
                logger.info("%s still holds the previous run's log (append mode not confirmed).", path)
                return 0
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
        if dest is None:
            logger.info("Previous run's log removed from %s (clear_working_folder is off).", path)
            return 0
        logger.info("Previous run's log moved to %s; %s now holds only this run.", dest, path)
        return 1

    def _drop_previous_log(self) -> None:
        """clear_working_folder is off: like a `>` redirect, keep only this run's output in the
        log the current process writes to (the previous run's part is not kept anywhere)."""
        for start_size, append, path in _OWN_LOGS.values():
            if path is not None and start_size > 0:
                self._split_own_log(Path(path), None, start_size, append)

    def _build_rule(self) -> Any:
        """Build acquisition rule based on configuration."""
        config = self.state.config

        match config.acquisition_function:
            case "evm":
                logger.info("Continue with Explorational Variance Minimizer acquisition function!")
                builder = ExplorationalVarianceMinimizer(threshold=3)
            case "ei":
                logger.info("Continue with ExpectedImprovement acquisition function!")
                builder = ExpectedImprovement()
            case "ik":
                logger.info("Continue with ImprovementVarianceWithIK acquisition function!")
                if self.state.ik_acq_loss is None or config.ik_loss_weight == 0:
                    # Same improvement variance, without the ring-closure term
                    logger.info(
                        "No ring for the IK loss of the acquisition (or ik_loss_weight: 0) — using the "
                        "improvement variance alone; the MMFF ring build is unchanged."
                    )
                    builder = ExplorationalVarianceMinimizer(threshold=3)
                else:
                    builder = ImprovementVarianceWithIK(
                        threshold=3.0,
                        ik_loss=self.state.ik_acq_loss,
                        ik_loss_idxs=self.state.ik_acq_dihedrals_idxs,
                        ik_loss_weight=config.ik_loss_weight,
                    )
            case _:
                raise ValueError(f"Unknown acquisition function {config.acquisition_function}")

        if self.state.feasibility is not None:
            logger.info(
                "Acquisition is weighted by the feasibility model (%d labels, %d failed)",
                len(self.state.feasibility), self.state.feasibility.n_failed,
            )
            builder = FeasibilityWeightedAcquisition(builder, self.state.feasibility)
        self._batch_optimizer = None
        if config.candidate_batch_size > 1 and self.state.feasibility is not None:
            # Candidates rejected before ORCA are replaced by the next maximum of the same search
            self._batch_optimizer = BatchRecordingOptimizer(config.candidate_batch_size, len(self.state.dihedral_ids))
            return EfficientGlobalOptimization(builder, optimizer=self._batch_optimizer)
        return EfficientGlobalOptimization(builder)

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
            "norm_energy": float(self.state.norm_energy),  # all observations are relative to it
            "outcomes": dict(self.state.outcomes),
            "events": run_state.events(),
        }
        if self.state.feasibility is not None:
            snapshot["feasibility_points"] = self.state.feasibility.points
            snapshot["feasibility_labels"] = self.state.feasibility.labels

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
        - checkpoint_step is "last" → use the last saved step.
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

        if isinstance(requested, str) and requested.strip().lower() == "last":
            logger.info(
                "checkpoint_step=last: resuming after step %d (%d dataset points, %d minima).",
                last["step"], len(last.get("observations", [])), len(last.get("minima", [])),
            )
            return last

        try:
            req = int(requested)
        except (TypeError, ValueError):
            logger.warning(
                "checkpoint_step=%r is neither an integer nor 'last' — ignoring checkpoint.",
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
        ckpt_norm = snapshot.get("norm_energy")
        if ckpt_norm is not None:
            if abs(float(ckpt_norm) - self.state.norm_energy) > 1e-4:
                logger.warning(
                    "norm_energy of this run (%.6f) differs from the checkpoint (%.6f) — using the "
                    "checkpoint value so old and new points share one energy scale.",
                    self.state.norm_energy, float(ckpt_norm),
                )
            self.state.norm_energy = float(ckpt_norm)
        self.state.minima = snapshot.get("minima", [])
        self.state.outcomes = dict(snapshot.get("outcomes", {}))
        run_state.set_events(snapshot.get("events", {}))
        self.state.current_minima = float(snapshot.get("current_minima", 1e9))
        self.state.acq_vals_log = list(snapshot.get("acq_vals_log", []))
        self.state.deepest_minima = list(snapshot.get("deepest_minima", []))

        fm = self.state.feasibility
        if fm is not None:
            points, labels = snapshot.get("feasibility_points"), snapshot.get("feasibility_labels")
            try:
                fm.set_data(points, labels)
                logger.info("Feasibility model restored: %d labels, %d failed", len(fm), fm.n_failed)
            except (TypeError, ValueError):
                # Checkpoint without labels (older version or the model was off): every GP point
                # with an energy counts as feasible, every broken_struct_energy point as failed
                obs = dataset.observations.numpy().reshape(-1)
                fm.set_data(
                    dataset.query_points.numpy().tolist(),
                    [not _is_broken(float(e), self.state.config.broken_struct_energy) for e in obs],
                )
                logger.info("Feasibility model seeded from the checkpoint data: %d labels, %d failed", len(fm), fm.n_failed)

        minima_dir = Path(self.state.working_folder) / f"{self.state.exp_name}_minima"
        for i, (coords, _) in enumerate(self.state.minima):
            try:
                self._add_template(coords, xyz_coords((minima_dir / f"{i}.xyz").read_text()))
            except (OSError, ValueError):
                pass

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

        n_ts_dims = len(self.state.ts_bonds) if self.state.vary_ts_bond_lengths else 0
        if config.use_feasibility_model:
            self.state.feasibility = FeasibilityModel(
                n_dihedrals=len(self.state.dihedral_ids), n_ts_bonds=n_ts_dims,
                lengthscale=config.feasibility_lengthscale,
                ts_lengthscale=config.ts_bond_kernel_lengthscale,
                prior=config.feasibility_prior,
            )
        ep = self.state.ensemble_processor
        if ep is not None and config.build_from_nearest_conformer:
            for point, coords in zip(ep.processed_ens, ep.coords):
                self._add_template(point, coords)
            if not self.state.templates and ep.coords:
                logger.warning("load_ensemble structures can't be used to build candidates (atom count differs from the .mol)")
        elif ep is not None:
            for coords in ep.coords:
                self._note_mmff_reference(coords)
        self.state.ts_bond_windows = self._resolve_ts_bond_windows()

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
        dataset = self._drop_invalid_observations(dataset)
        if config.build_from_nearest_conformer:
            logger.info("Candidates are built from the closest of %d known structures", len(self.state.templates))

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

        # Constant prior mean (trained) instead of 0: 0 is the best energy, so far from the data
        # the GP used to predict "as good as the best" and sent the search to empty regions
        gpr = gpflow.models.GPR(dataset.astuple(), kernel, mean_function=gpflow.mean_functions.Constant(obs_mean))
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
            self._log_run_summary(start_step - 1, "nothing left to run")
            return
        
        # A BO step counts towards max_steps only if its candidate reached ORCA: candidates
        # rejected by the structure check (or no candidate at all) are retried, at most
        # max_geometry_rejections_in_a_row times in a row (0 = no limit). With the feasibility
        # model a candidate without an energy only teaches that model; without it, it goes
        # into the GP as broken_struct_energy.
        step = start_step - 1  # counted steps done
        rejected_in_a_row = 0
        tried_in_batch: list = []  # points of the current acquisition search already evaluated
        max_rejected = config.max_geometry_rejections_in_a_row
        # The model is refitted below after every step (after the trajectory frames replace
        # the asked point), so trieste's own refit inside bo.optimize is skipped
        bo_kwargs = {"fit_initial_model": False}
        if "fit_model" in inspect.signature(bo.optimize).parameters:
            bo_kwargs["fit_model"] = False
        while step < config.max_steps:
            if max_rejected > 0 and rejected_in_a_row >= max_rejected:
                break
            logger.debug("Previous last_opt_ok: %s", self.state.last_opt_ok)
            logger.debug("Step number %s", step + 1)

            self.state.last_observation = None
            self.state.geometry_rejected = False
            t0 = time.monotonic()
            try:
                queued = None
                if rejected_in_a_row > 0 and self._batch_optimizer is not None:
                    queued = self._batch_optimizer.pop_next(rule._acquisition_function, tried_in_batch)
                if queued is not None:
                    # Same GP, same acquisition search: only the feasibility weight changed
                    tried_in_batch.append(queued)
                    dataset = dataset + observer(tf.constant([queued], dtype=tf.float64))
                    logger.info("Step %s: kept candidate of the last acquisition search evaluated", step + 1)
                else:
                    result = bo.optimize(1, dataset, model, rule, **bo_kwargs)
                    new_dataset = result.try_get_final_dataset()
                    model = result.try_get_final_model()
                    dataset = new_dataset
                    if self._batch_optimizer is not None and self.state.asked_points:
                        tried_in_batch = [np.asarray(self.state.asked_points[-1], dtype=float)]
                    logger.info("Optimization step %s succeed! (bo.optimize took %.1f s)", step + 1, time.monotonic() - t0)
                if self.state.feasibility is not None and not self.state.last_eval_valid:
                    # No energy for this candidate: it lives in the feasibility model only
                    dataset = self._erase_last_from_dataset(dataset, 1)
                else:
                    dataset = self._move_last_to_built_point(dataset)
            except Exception as e:
                if self.state.last_observation is None:
                    # Failed before the objective ran (acquisition): nothing new to keep
                    logger.exception(
                        "Step %d: trieste failed before the objective was called — no new point, "
                        "keeping previous dataset and model. The step is not counted.", step + 1,
                    )
                    rejected_in_a_row += 1
                    model.update(dataset)
                    try:
                        model.optimize(dataset)
                    except Exception as e2:
                        logger.error("Step %d: fallback model.optimize also failed: %s.", step + 1, e2)
                    continue
                # Failed after ORCA (GP update/fit): keep the observed point instead of losing hours of ORCA
                logger.error(
                    "Step %d: trieste failed after the objective was evaluated (%s) — adding the "
                    "observed point to the dataset directly.", step + 1, e,
                )
                query_point, energy = self.state.last_observation
                if self.state.feasibility is None or self.state.last_eval_valid:
                    dataset = self._move_last_to_built_point(dataset + Dataset(
                        tf.constant([query_point], dtype=tf.float64), tf.constant([[energy]], dtype=tf.float64),
                    ))

            if self.state.geometry_rejected:
                rejected_in_a_row += 1
            else:
                rejected_in_a_row = 0
                step += 1

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
                part = self._trj_dataset_part(self._last_trj_path())
                if part is not None:
                    dataset = self._erase_last_from_dataset(dataset, 1) + part
            else:
                logger.warning("Last optimization finished with error, skipping trj parsing!")
            if self.state.feasibility is not None and not self.state.last_eval_valid:
                # No new energy: the GP data did not change, refitting it would only cost time
                logger.info("GP data unchanged (candidate without an energy) — GP not refitted")
            else:
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

            obs = dataset.observations.numpy().flatten()
            if self.state.feasibility is not None:
                # Candidates without an energy (outside the GP data)
                n_broken = self.state.feasibility.n_failed
            else:
                n_broken = int(np.sum([_is_broken(v, config.broken_struct_energy) for v in obs]))
            if self.state.geometry_rejected:
                logger.info(
                    "Candidate rejected by the structure check (%d in a row) — %s, "
                    "step counter stays at %d/%d. Dataset: %d points, broken=%d",
                    rejected_in_a_row,
                    "labelled in the feasibility model" if self.state.feasibility is not None else "added to the GP as broken",
                    step, config.max_steps, len(obs), n_broken,
                )
                print(
                    f"[BOCSER] rejected by structure check ({rejected_in_a_row} in a row), "
                    f"step stays {step}/{config.max_steps}  points={len(obs):>4}  broken={n_broken:>3}",
                    flush=True,
                )
            else:
                logger.info("Step %s completed!", step)
                logger.info(
                    "Step %d completed. Dataset: %d points, best=%.3f, "
                    "broken=%d, mean_valid=%.3f",
                    step, len(obs), float(obs.min()), n_broken,
                    float(obs[obs < config.broken_struct_energy * 0.9].mean())
                    if np.any(obs < config.broken_struct_energy * 0.9) else float("nan"),
                )
                _print_progress(step, config.max_steps, float(obs.min()), len(obs), n_broken)

            all_minima_file = Path(self.state.working_folder) / f"{self.state.exp_name}_all_minima.json"
            with open(all_minima_file, "w") as json_minima_writer:
                json.dump(self.state.minima, json_minima_writer)
            
            try:
                # A rejected candidate overwrites the snapshot of the current step, so it is kept on resume
                self._save_checkpoint(dataset, step)
            except Exception:
                logger.exception("Failed to write checkpoint at step %s", step)

            if self.state.geometry_rejected:
                continue

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

        if step < config.max_steps and not early_termination_flag:
            stop_reason = f"{rejected_in_a_row} rejected candidates in a row"
            logger.error(
                "Stopping: %d candidates in a row were rejected before ORCA (max_geometry_rejections_in_a_row=%d); "
                "%d of %d steps done.", rejected_in_a_row, max_rejected, step, config.max_steps,
            )
            print(f"[BOCSER] Stopped: {rejected_in_a_row} rejected candidates in a row at step {step}", flush=True)
        elif not early_termination_flag:
            stop_reason = "max_steps reached"
            logger.info("Max number of steps has been reached!")
            print(f"[BOCSER] Finished: reached max_steps={config.max_steps}", flush=True)
        else:
            stop_reason = "convergence criterion met"
            print(f"[BOCSER] Finished early: convergence criterion met at step {step}", flush=True)

        logger.info("MINIMA: %s", self.state.minima)
        self._save_results(dataset)
        self._log_run_summary(step, stop_reason)

    _OUTCOME_TITLES = (
        ("success", "successful (pre-opt + full opt)"),
        ("rejected_before_orca", "rejected by the structure check before ORCA"),
        ("preopt_failed", "pre-opt failed in ORCA"),
        ("opt_input_rejected", "pre-opt result rejected by the structure check"),
        ("opt_failed", "full opt failed in ORCA"),
        ("opt_result_broken", "full opt result rejected by the structure check"),
    )

    def _log_run_summary(self, steps_done: int, stop_reason: str) -> None:
        """Counts of candidate outcomes by stage and reason (whole run, resumed parts included),
        logged and written to {exp_name}_run_summary.json."""
        config = self.state.config
        by_category: dict = {}
        for key, n in self.state.outcomes.items():
            category, _, reason = key.partition(":")
            by_category.setdefault(category, {})
            by_category[category][reason or "-"] = by_category[category].get(reason or "-", 0) + n
        events = run_state.events()
        unknown_keys = list(getattr(config, "unknown_keys", []))

        lines = [
            "===== BOCSER run summary =====",
            f"Steps counted: {steps_done} of max_steps={config.max_steps} (stopped: {stop_reason})",
            f"Candidates evaluated: {sum(self.state.outcomes.values())}",
        ]
        for category, title in self._OUTCOME_TITLES:
            reasons = by_category.get(category, {})
            total = sum(reasons.values())
            detail = ", ".join(f"{r} {n}" for r, n in sorted(reasons.items(), key=lambda x: -x[1]) if r != "-")
            lines.append(f"  {title}: {total}" + (f" ({detail})" if detail else ""))
        lines.append(
            f"MMFF not converged: candidate build {events.get('mmff_build_not_converged', 0)}, "
            f"regularization {events.get('mmff_not_converged', 0)}"
        )
        if self.state.feasibility is not None:
            lines.append(
                f"Feasibility model: {len(self.state.feasibility)} labels, "
                f"{self.state.feasibility.n_failed} without an energy"
            )
        lines.append(
            f"Minima found: {len(self.state.minima)}, unique conformers in the final ensemble: "
            f"{self.state.n_unique_conformers}"
        )
        if unknown_keys:
            lines.append(f"Unknown config keys (ignored): {', '.join(unknown_keys)}")
        lines.append("==============================")
        logger.info("\n".join(lines))

        summary = {
            "steps_counted": steps_done,
            "max_steps": config.max_steps,
            "stop_reason": stop_reason,
            "outcomes": by_category,
            "events": events,
            "minima_found": len(self.state.minima),
            "unique_conformers": self.state.n_unique_conformers,
            "unknown_config_keys": unknown_keys,
        }
        try:
            (Path(self.state.working_folder) / f"{self.state.exp_name}_run_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
        except OSError:
            logger.exception("Could not write the run summary file")

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
            self._log_checkpoint_kept()
            return

        res = self._cluster_minima_by_rmsd()
        self.state.n_unique_conformers = len(res)

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
        self._log_checkpoint_kept()

    def _log_checkpoint_kept(self) -> None:
        """The checkpoint stays after the run, so it can be continued with more steps."""
        if self._checkpoint_path().is_file():
            logger.info(
                "Checkpoint kept at %s: to continue this run, set checkpoint_step: last and a larger max_steps.",
                self._checkpoint_path(),
            )

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
        Such TS-pair bond entries are taken out of the hard constraints, which
        constrain_final_opt also applies in the final opt, so the final OptTS can always move
        along the reaction coordinate.
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check the config: build the GP axes and write the ORCA inputs of one candidate, without running ORCA"
    )

    args = parser.parse_args()

    # kill / Slurm stop / closed terminal: raise SystemExit instead of dying at once, so the
    # running ORCA job is cancelled (calc.start_calc) before BOCSER exits
    def _stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _stop)
    if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) is signal.SIG_DFL:  # not under nohup
        signal.signal(signal.SIGHUP, _stop)

    logger.info("Working folder: %s", args.folder)
    logger.info("Reading config from: %s", args.config)

    runner = ConfSearchRunner(working_folder=args.folder)
    runner.load_config(args.config)
    if args.dry_run:
        runner.dry_run = True
        runner.setup()
        runner.dry_run_report()
        return
    runner.acquire_lock()
    try:
        runner.setup()
        runner.run()
    finally:
        runner.release_lock()

    logger.info("Conformational search completed!")


if __name__ == "__main__":
    main()