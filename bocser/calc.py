import os
import os.path
import time
import math
from typing import Union
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import numpy as np

from sklearn.cluster import KMeans

from default_vals import ConfSearchConfig
from config_loader import ConfigError
import subprocess
import shutil
from pathlib import Path
import hashlib
import tempfile
import logging
logger = logging.getLogger(__name__)
import config_manager
import run_state
from functools import lru_cache
from typing import NamedTuple

logging.getLogger("tensorflow").setLevel(logging.ERROR)

HARTRI_TO_KCAL = 627.509474063

_CONSTRAINT_LETTER = {"bond": "B", "angle": "A", "dihedral": "D"}

_VDW_RADII: dict[str, float] = {
    'H':  1.20, 'C':  1.70, 'N':  1.55, 'O':  1.52,
    'F':  1.47, 'P':  1.80, 'S':  1.80, 'Cl': 1.75,
    'Br': 1.85, 'I':  1.98, 'Se': 1.90, 'Si': 2.10,
}

_COVALENT_RADII: dict[str, float] = {
    'H':  0.31, 'C':  0.76, 'N':  0.71, 'O':  0.66,
    'F':  0.57, 'P':  1.07, 'S':  1.05, 'Cl': 1.02,
    'Br': 1.20, 'I':  1.39, 'Se': 1.20, 'Si': 1.11,
}

# fallback: carbon
_COVALENT_RADII_DEFAULT = 0.76
_VDW_RADII_DEFAULT = 1.70


def _clash_threshold(sym_a: str, sym_b: str) -> float:
    """Minimum allowed distance between two atoms before they are considered clashing."""
    cfg = config_manager.get_config()
    scale = cfg.clash_vdw_scale if cfg is not None else 0.0

    if scale <= 0.0:
        return 0.7  # legacy behaviour, unchanged default

    ra = _VDW_RADII.get(sym_a, _VDW_RADII_DEFAULT)
    rb = _VDW_RADII.get(sym_b, _VDW_RADII_DEFAULT)
    return scale * (ra + rb)

def _bond_break_threshold(sym_a: str, sym_b: str, delta: float = 0.1) -> float:
    """Maximum allowed distance for a ring bond before it is considered broken."""
    ra = _VDW_RADII.get(sym_a, _VDW_RADII_DEFAULT)
    rb = _VDW_RADII.get(sym_b, _VDW_RADII_DEFAULT)
    return ra + rb + delta

#Alias for type of node about dihedral angle 
#that consists of list with four atoms and value of degree
dihedral = tuple[list[int], float]

class Constraint(NamedTuple):
    type: str      # "bond" | "angle" | "dihedral"
    atoms: tuple   # canonical 0-idx (with-H)
    value: float   # angle/dihedral, A for bond

def _get_config_or_raise() -> ConfSearchConfig:
    """Return the runtime config or raise RuntimeError if it's not set.

    This enforces that calculations always use the central config and avoids
    falling back to outdated module-level globals.
    """
    cfg = config_manager.get_config()
    if cfg is None:
        raise RuntimeError("Configuration is not set. Call `config_manager.set_config()` or load a config before using calc functions.")
    return cfg

def dist_between_atoms(mol : Chem.rdchem.Mol, i : int, j : int) -> float:
    pos_i = mol.GetConformer().GetAtomPosition(i)
    pos_j = mol.GetConformer().GetAtomPosition(j)
    
    return np.sqrt((pos_i.x - pos_j.x) ** 2 + (pos_i.y - pos_j.y) ** 2 + (pos_i.z - pos_j.z) ** 2)

def _heavy_and_h_order(mol) -> tuple[list[int], list[int]]:
    heavy_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != 'H']
    h_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == 'H']
    return heavy_idx, h_idx

@lru_cache(maxsize=8)
def _with_h_order_cached(mol_file_name: str) -> tuple:
    mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)
    heavy_idx, h_idx = _heavy_and_h_order(mol)
    return tuple(heavy_idx + h_idx)

def build_reference_with_h_mol(mol_file_name: str):
    """Canonical (with-H) renumber, needed for saving bond/dihedral value: 'current'."""
    mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)
    heavy_idx, h_idx = _heavy_and_h_order(mol)
    return Chem.RenumberAtoms(mol, heavy_idx + h_idx)

def raw1_to_with_h_canonical(raw_1idx: int, mol_file_name: str) -> int:
    """1-idx number from .mol -> 0-idx"""
    order = _with_h_order_cached(mol_file_name)
    return order.index(raw_1idx - 1)

def raw1_to_heavy_canonical(raw_1idx: int, mol_file_name: str) -> int:
    """Same canonical, but to heavy once"""
    mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)
    heavy_idx, _ = _heavy_and_h_order(mol)
    raw0 = raw_1idx - 1
    if raw0 not in heavy_idx:
        raise ValueError(f"Atom {raw_1idx} — hydrogen; "
                          f"Double bonds are specified only between heavy atoms.")
    return heavy_idx.index(raw0)

def with_h_canonical_to_raw1(canon_idx: int, mol_file_name: str) -> int:
    """Opposite of raw1_to_with_h_canonical"""
    order = _with_h_order_cached(mol_file_name)
    return order[canon_idx] + 1

def _validate_extra_constraints_atoms(config: ConfSearchConfig) -> None:
    if not config.extra_constraints:
        return

    ref_mol = Chem.MolFromMolFile(config.mol_file_name, removeHs=False)
    n_atoms = ref_mol.GetNumAtoms()

    errors = []
    for i, c in enumerate(config.extra_constraints):
        for raw_idx in c["atoms"]:
            # config uses 1-indexed atom numbers as in the raw .mol file
            if not (1 <= raw_idx <= n_atoms):
                errors.append(
                    f"extra_constraints[{i}] (type={c['type']}): atom index {raw_idx} "
                    f"is out of range — molecule has {n_atoms} atoms (1-indexed, with H)."
                )

    if errors:
        msg = "Invalid extra_constraints in config:\n" + "\n".join(errors)
        logger.error(msg)
        raise ConfigError(msg)

def resolve_extra_constraints(raw_constraints: list, mol_file_name: str) -> list:
    """config.extra_constraints (1-idx, raw .mol) -> list[Constraint]"""
    if not raw_constraints:
        return []
    ref_mol = build_reference_with_h_mol(mol_file_name)
    conf = ref_mol.GetConformer()
    result = []
    for raw in raw_constraints:
        ctype = raw["type"]
        canon_atoms = tuple(raw1_to_with_h_canonical(a, mol_file_name) for a in raw["atoms"])
        value = raw["value"]
        if value == "current" or value is None:
            if ctype == "bond":
                value = rdMolTransforms.GetBondLength(conf, *canon_atoms)
            elif ctype == "angle":
                value = rdMolTransforms.GetAngleDeg(conf, *canon_atoms)
            elif ctype == "dihedral":
                value = rdMolTransforms.GetDihedralDeg(conf, *canon_atoms)
        else:
            value = float(value)
        c = Constraint(ctype, canon_atoms, round(value, 6))
        logger.info(
            "extra_constraint: type=%s raw_atoms(1-idx, с H)=%s -> canonical(0-idx)=%s value=%.4f",
            ctype, raw["atoms"], canon_atoms, value,
        )
        result.append(c)
    return result

def submit_calc(gjf_name: str, scan=False) -> str:
    """Submit an ORCA job to SLURM WITHOUT waiting for completion (no `-W`).
    Returns the SLURM job id so callers can poll/wait on a batch of jobs
    submitted together. Mirrors start_calc() otherwise (sbatch script content,
    cleanup happens later, per-job, after the wait step)."""
    cfg = _get_config_or_raise()
    orca_cmd = cfg.orca_exec_command

    gjf_path = Path(gjf_name).resolve()
    gjf_dir = gjf_path.parent
    gjf_base = gjf_path.stem
    sbatch_name = str(gjf_dir / (gjf_base + ".sh"))
    template_to_copy = _select_sbatch_template(gjf_dir, cfg)
    shutil.copy(template_to_copy, sbatch_name)

    with open(sbatch_name, "a") as fh:
        if cfg.ts and cfg.use_grass and not scan:
            fh.write(f"python -u {cfg.path_to_grass} {gjf_name} -OPATH {orca_cmd[:-4]} -p orca -onp {cfg.num_of_procs} -oms \"{cfg.orca_method}\" {cfg.grass_options} > {gjf_name[:-4]}.grass\n")
        else:
            fh.write(f"{orca_cmd} {gjf_name} > {gjf_name[:-4]}.out\n")

    proc = subprocess.run(
        ["sbatch", "-o", "/dev/null", sbatch_name],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        logger.error("sbatch submission FAILED for %s: %s", sbatch_name, proc.stderr.strip() or proc.stdout.strip())
        raise RuntimeError(f"sbatch submission failed for {sbatch_name}: {proc.stderr.strip()}")

    # sbatch stdout is typically "Submitted batch job 123456"
    job_id = proc.stdout.strip().split()[-1]
    logger.info("Submitted %s as SLURM job %s (not waiting yet)", gjf_name, job_id)
    return job_id


def wait_for_jobs(job_ids: list[str], timeout_minutes: int) -> None:
    """Block until all given SLURM job ids finish, or timeout. Uses a simple
    polling loop against squeue rather than one blocking sbatch -W per job,
    so independent jobs actually run concurrently on the cluster."""
    import time
    deadline = time.monotonic() + timeout_minutes * 60
    pending = set(job_ids)

    while pending and time.monotonic() < deadline:
        proc = subprocess.run(
            ["squeue", "-h", "-j", ",".join(pending), "-o", "%i"],
            capture_output=True, text=True,
        )
        still_running = set(proc.stdout.split())
        finished = pending - still_running
        if finished:
            logger.debug("Jobs finished: %s", finished)
        pending = still_running
        if pending:
            time.sleep(10)

    if pending:
        logger.warning(
            "Timeout (%d min) reached with %d jobs still pending: %s. "
            "Continuing — their .out files may be incomplete/missing.",
            timeout_minutes, len(pending), pending,
        )

def change_dihedrals(mol_file_name: str,
                     dihedrals: list[list[tuple[tuple[int,int,int,int], float]]], ik_loss=None,
                    full_block=False, ts_bonds=None, fixed_dihedrals=None, extra_constraints=None):
    ts_bond_set = {frozenset(b) for b in (ts_bonds or [])}

    try:
        mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)
        heavy_idx, h_idx = _heavy_and_h_order(mol)
        mol = Chem.RenumberAtoms(mol, heavy_idx + h_idx)


        # Return reference geometry if no torsion angles are provided
        if not dihedrals and not fixed_dihedrals and not extra_constraints:
            if full_block:
                return Chem.MolToXYZBlock(mol)
            return '\n'.join(Chem.MolToXYZBlock(mol).split('\n')[2:])
        
        # Read acquisition function from central config (require config to be set)
        _cfg = _get_config_or_raise()
        _af = _cfg.acquisition_function

        if _af != 'ik':
            for cycle in dihedrals:
                for atoms, degree in cycle:
                    rdMolTransforms.SetDihedralRad(mol.GetConformer(), *atoms, degree)

        else:
            with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
                Chem.MolToMolFile(mol, tmp.name)
                tmp_mol = Chem.RWMol(Chem.MolFromMolFile(tmp.name, removeHs=False))

            # for a, b in ts_bond_set: # Remove TS bonds to allow free movement
            #     if tmp_mol.GetBondBetweenAtoms(a, b) is not None:
            #         tmp_mol.RemoveBond(a, b)
            # if ts_bond_set:
            #     Chem.FastFindRings(tmp_mol)

            mp = AllChem.MMFFGetMoleculeProperties(tmp_mol, mmffVariant='MMFF94')
            # if mp is None:
            #     raise RuntimeError("MMFFGetMoleculeProperties returned None after removing TS bonds")
            ff = AllChem.MMFFGetMoleculeForceField(tmp_mol, mp)

            for bl_dict in ik_loss.bond_lengths:
                for (a, b), value in bl_dict.items():
                    if frozenset((a, b)) in ts_bond_set:
                        continue  # TS-bond is not fixed
                    ff.MMFFAddDistanceConstraint(a, b, False, value, value, 1e3)

            for va_dict in ik_loss.valence_angles:
                for (a, b, c), value in va_dict.items():
                    ff.MMFFAddAngleConstraint(a, b, c, False, np.rad2deg(value), np.rad2deg(value), 1e2)
            
            # Prevent deformation of discarded rings that IKLoss does not cover
            ik_covered_bonds = {
                frozenset(bond) for bl_dict in ik_loss.bond_lengths for bond in bl_dict
            }
            conf0 = mol.GetConformer()
            for ring in mol.GetRingInfo().AtomRings():
                if len(ring) >= 4:
                    continue  # already constrained above
                n = len(ring)
                for k in range(n):
                    a, b = ring[k], ring[(k + 1) % n]
                    if frozenset((a, b)) in ts_bond_set or frozenset((a, b)) in ik_covered_bonds:
                        continue
                    dist = conf0.GetAtomPosition(a).Distance(conf0.GetAtomPosition(b))
                    ff.MMFFAddDistanceConstraint(a, b, False, dist, dist, 1e3)
                for k in range(n):
                    a, b, c = ring[k - 1], ring[k], ring[(k + 1) % n]
                    angle_deg = rdMolTransforms.GetAngleDeg(conf0, a, b, c)
                    ff.MMFFAddAngleConstraint(a, b, c, False, angle_deg, angle_deg, 1e2)

            for (a, b, c, d), value in dihedrals:
                ff.MMFFAddTorsionConstraint(a, b, c, d, False, np.rad2deg(-value), np.rad2deg(-value), 1)

            for atoms, value in (fixed_dihedrals or []):
                a, b, c, d = atoms
                ff.MMFFAddTorsionConstraint(a, b, c, d, False, np.rad2deg(-value), np.rad2deg(-value), 1e2)

            for c in (extra_constraints or []):
                if c.type == "bond":
                    ff.MMFFAddDistanceConstraint(*c.atoms, False, c.value, c.value, 1e3)
                elif c.type == "angle":
                    ff.MMFFAddAngleConstraint(*c.atoms, False, c.value, c.value, 1e2)
                elif c.type == "dihedral":
                    a, b, cc, d = c.atoms
                    ff.MMFFAddTorsionConstraint(a, b, cc, d, False, -c.value, -c.value, 1e2)

            ff.Minimize(maxIts=1000)
            mol = tmp_mol

            for i, (atoms, old_val) in enumerate(dihedrals):
                positions = [mol.GetConformer().GetAtomPosition(idx) for idx in atoms]
                new_val = dihedral_angle(*positions)

                dihedrals[i] = (atoms, new_val)

        if full_block:
            return Chem.MolToXYZBlock(mol)
        return '\n'.join(Chem.MolToXYZBlock(mol).split('\n')[2:])

    except OSError:
        logger.error("No such file: %s", mol_file_name)
        return None
    
def to_degrees(dihedrals : list[dihedral]) -> list[dihedral]:
    """
        Convert rads to degrees in dihedrals
    """
    res = []
    for cur in dihedrals:
        a, d = cur
        res.append((a, d * 180 / math.pi))
    
    return res

def read_xyz(name : str) -> list[str]:
    """
        read coords from 'filename' and return that as a list of strings
    """
    xyz = []
    with open(name, 'r') as file:
        for line in file:
            xyz.append(line)
    return '\n'.join(xyz)

def generate_oinp(
        coords : str, 
        dihedrals : list[dihedral], # BO-targets; only when constrained_opt=True
        gjf_name : str, 
        num_of_procs : int, 
        method_of_calc : str,
        charge : int,
        multipl : int,
        constrained_opt : bool = False,
        hard_constraints: list = None,   # every ORCA call (pre-opt И full opt) 
    ) -> None:
    """
        generates orca .inp file
    """
    # Require runtime config to be set; config provides TS flag
    cfg = _get_config_or_raise()
    parent = Path(gjf_name).parent
    parent.mkdir(parents=True, exist_ok=True)

    # Write atomically into the same directory to avoid partial writes being
    # picked up by monitoring code. Use a temp file and then replace.
    with tempfile.NamedTemporaryFile(mode="w", dir=str(parent), delete=False, suffix=".tmp") as tmp:
        
        if cfg.ts and cfg.use_grass:
            tmp.write(str(coords.count('\n')))
            tmp.write("\n\n")
            tmp.write(coords)
        else:
            opt_cmd = "OptTS" if cfg.ts else "Opt"
            tmp.write("!" + method_of_calc + f" {opt_cmd}\n")
            tmp.write("%pal\nnprocs " + str(num_of_procs) + "\nend\n")

            hard_by_axis = {}
            all_constraints = []
            for c in (hard_constraints or []):
                if c.type == "dihedral":
                    axis = frozenset((c.atoms[1], c.atoms[2]))
                    if axis in hard_by_axis:
                        logger.warning(
                            "Dublicate dihedral-constraint on axis %s: %s",
                            tuple(axis), c,
                        )
                    hard_by_axis[axis] = c
                all_constraints.append(c)

            if constrained_opt:
                dihedrals_deg = to_degrees(dihedrals)
                for atoms, value in dihedrals_deg:
                    axis = frozenset((atoms[1], atoms[2]))
                    if axis in hard_by_axis:
                        logger.info(
                            "BO-target for zxis %s was skipped in .inp: fixed with %s=%s.",
                            tuple(axis), hard_by_axis[axis].type, hard_by_axis[axis].value,
                        )
                        continue
                    all_constraints.append(Constraint("dihedral", tuple(atoms), value))

            need_geom = bool(all_constraints) or cfg.ts
            if need_geom:
                if all_constraints:
                    tmp.write("%geom Constraints\n")
                    for c in all_constraints:
                        letter = _CONSTRAINT_LETTER[c.type]
                        atoms_str = " ".join(str(a) for a in c.atoms)
                        tmp.write("{ " + letter + " " + atoms_str + " " + str(c.value) + " C }\n")
                    tmp.write("end\n")
                if cfg.ts:
                    max_iter = cfg.ts_max_iter if constrained_opt else cfg.ts_max_iter * 2
                    tmp.write(f"MaxIter {max_iter}\n")
                    tmp.write("Calc_Hess true\n")
                tmp.write("end\n")

            tmp.write("* xyz " + str(charge) + " " + str(multipl) + "\n")
            tmp.write(coords)
            tmp.write("END\n")
        
        tmp_name = tmp.name

    try:
        os.replace(tmp_name, gjf_name)
        os.chmod(gjf_name, 0o644)
    except Exception:
        # If atomic replace fails, try a best-effort move
        shutil.move(tmp_name, gjf_name)

def _select_sbatch_template(gjf_dir: Path, cfg: ConfSearchConfig) -> str:
    """Return the path to the sbatch template to copy for `gjf_dir`.

    Prefers a template found inside `gjf_dir`; otherwise falls back to the
    configured template name (which may be a path in the current working dir).
    Returns a string path suitable for passing to `shutil.copy`.
    """
    candidate = gjf_dir / cfg.sbatch_template_name
    if candidate.exists():
        return str(candidate)
    return cfg.sbatch_template_name


def start_calc(gjf_name: str, scan=False):
    """
        Running calculation
    """	
    cfg = _get_config_or_raise()
    orca_cmd = cfg.orca_exec_command

    # Place the generated sbatch script next to the input file so scripts
    # and outputs live inside the working folder instead of the module cwd.
    gjf_path = Path(gjf_name).resolve()
    gjf_dir = gjf_path.parent
    gjf_base = gjf_path.stem
    sbatch_name = str(gjf_dir / (gjf_base + ".sh"))
    template_to_copy = _select_sbatch_template(gjf_dir, cfg)
    shutil.copy(template_to_copy, sbatch_name)
    
    with open(sbatch_name, "a") as fh:
        if cfg.ts and cfg.use_grass and not scan:
            fh.write(f"python -u {cfg.path_to_grass} {gjf_name} -OPATH {orca_cmd[:-4]} -p orca -onp {cfg.num_of_procs} -oms \"{cfg.orca_method}\" {cfg.grass_options} > {gjf_name[:-4]}.grass\n")
        else:
            fh.write(f"{orca_cmd} {gjf_name} > {gjf_name[:-4]}.out\n")
    
    timeout_minutes = cfg.orca_poll_timeout_minutes
    logger.info("Submitting ORCA job %s to SLURM (sbatch -W, timeout=%d min)...", gjf_name, timeout_minutes)
    proc = subprocess.run(
        ["sbatch", "-W", "-t", str(timeout_minutes), "-o", "/dev/null", sbatch_name],
        capture_output=True, text=True,
    )

    if proc.returncode != 0:
        logger.error(
            "sbatch submission FAILED for %s (returncode=%s). stdout=%r stderr=%r. "
            "This is an infrastructure failure, not a chemistry failure — "
            "the resulting '.out' will not exist",
            sbatch_name, proc.returncode, proc.stdout.strip(), proc.stderr.strip(),
        )
        raise RuntimeError(
            f"sbatch submission failed for {sbatch_name} (code {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )

    # cleanup heavy ORCA files after calculations
    _cleanup_orca_tempfiles(gjf_path)

_ORCA_JUNK_SUFFIXES = {
    ".gbw", ".densities", ".tmp", ".ges",
    ".prop", ".bas", ".engrad", ".pcgrad",
    ".hess", ".bibtex",
}

def _cleanup_orca_tempfiles(inp_path: Path) -> None:
    """Delete large ORCA scratch files after each calculation.
    
    Keeps: .inp, .out, .xyz, _trj.xyz, .sh
    Deletes: .gbw, .densities, .tmp and other scratch files.
    """
    stem = inp_path.stem
    parent = inp_path.parent
    deleted_bytes = 0

    for f in parent.iterdir():
        if f.stem == stem and f.suffix in _ORCA_JUNK_SUFFIXES:
            try:
                size = f.stat().st_size
                f.unlink()
                deleted_bytes += size
                logger.debug("Deleted ORCA scratch file: %s (%.1f MB)", f.name, size/1e6)
            except OSError as e:
                logger.warning("Could not delete %s: %s", f.name, e)

    if deleted_bytes > 0:
        logger.info(
            "Cleaned up ORCA scratch files: %.1f MB freed", deleted_bytes / 1e6
        )
    
def _qc_calcs_dir(mol_file_name: str) -> Path:
    """Return path to the QC calculation subfolder, creating it if needed."""
    qc_dir = Path(mol_file_name).parent / "qc_calcs"
    qc_dir.mkdir(exist_ok=True)
    return qc_dir

def mol_to_inp_name(mol_file_name : str) -> str:
    """
        generating name of inp file from mol file name
    """
    cfg = _get_config_or_raise()
    stem = Path(mol_file_name).stem
    ext = ".inp" if not cfg.ts or not cfg.use_grass else ".xyz"
    return str(_qc_calcs_dir(mol_file_name) / (stem + ext))

def inp_to_out_name(inp_file_name : str) -> str:
    """
        generating name of out file from inp file name
    """
    cfg = _get_config_or_raise()
    return (inp_file_name[:-4] + ".out") if not cfg.ts or not cfg.use_grass else (os.path.dirname(inp_file_name) + "/outfile.out")

def find_energy_in_log(log_name : str) -> tuple[float, bool, int | None]:
    """
        finds energy of structure in log file
    """
    import re
    energy_re = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
    alt_re = re.compile(r"TOTAL ENERGY\s*[:=]\s*(-?\d+\.\d+)")
    terminated_re = re.compile(r"\*{4}ORCA TERMINATED NORMALLY\*{4}")
    cycle_re  = re.compile(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", re.IGNORECASE)
    maxiter_re = re.compile(r"The optimization did not converge but reached the maximum number of", re.IGNORECASE)
    m = None

    try:
        with open(log_name, 'r', errors='ignore') as fh:
            # scan from end to find the last occurrence without loading whole huge files
            from collections import deque
            last_lines = deque(fh, maxlen=4000)
            joined = "\n".join(last_lines)
            cfg = _get_config_or_raise()

            if not terminated_re.search(joined):
                # Don't search for energy if run didn't succeed
                logger.warning(
                    "ORCA TERMINATED NORMALLY not found in %s (last %d lines); "
                    "treating optimization as failed, not parsing energy.",
                    log_name, len(last_lines),
                )
                return cfg.broken_struct_energy, False, None

            cycles = [int(c) for c in cycle_re.findall(joined)]
            n_cycles = max(cycles) if cycles else None

            if maxiter_re.search(joined):
                # Iteration threshhold is hit, point can be not a minimum
                logger.warning("ORCA reached MaxIter (cycle=%s) — returning broken_struct_energy", n_cycles)
                return cfg.broken_struct_energy, False, n_cycles

            m = energy_re.search(joined) or alt_re.search(joined)
            if not m:
                # no energy line found -> optimization likely failed; return broken_struct_energy
                logger.warning("No energy line found in %s; returning broken_struct_energy", log_name)
                return cfg.broken_struct_energy, False, None
            
            return float(m.group(1)), True, n_cycles
            
    except FileNotFoundError:
        logger.error("No log file: %s. Returning broken_struct_energy", log_name)
        return cfg.broken_struct_energy, False, None
    except Exception:
        logger.exception("Failed to parse energy from line: %s", m.group(0))
        matched_text = m.group(0) if m is not None else "<no match — exception occurred before energy regex ran>"
        logger.exception("Failed to parse energy from log (matched text: %s)", matched_text)
        return cfg.broken_struct_energy, False, None

def _save_broken_struct(
    coords_block: str,
    broken_structs_dir: Union[str, None],
    reason: str,
    atoms_raw1: Union[tuple, None] = None,
    extra_files: Union[list, None] = None,
) -> None:
    """Save a discarded/broken candidate geometry for later inspection (e.g. in ChemCraft)."""

    if not broken_structs_dir:
        return
    try:
        lines = [l for l in coords_block.strip("\n").split("\n") if l.strip()]
        n_atoms = len(lines)
        struct_id = run_state.peek_structure_id()
        Path(broken_structs_dir).mkdir(parents=True, exist_ok=True)
        atom_tag = "_".join(str(a) for a in atoms_raw1) + "_" if atoms_raw1 else ""
        out_path = Path(broken_structs_dir) / f"{struct_id}_{atom_tag}{reason}.xyz"
        with open(out_path, "w") as fh:
            fh.write(f"{n_atoms}\n{atom_tag}{reason}\n")
            fh.write("\n".join(lines))
            fh.write("\n")

        for src in extra_files or []:
            src = Path(src)
            if src.is_file():
                shutil.copy2(src, Path(broken_structs_dir) / f"{struct_id}_{atom_tag}{reason}{src.suffix}")
        logger.info("Saved broken candidate (%s) to %s", reason, out_path)
    except Exception:
        logger.exception("Failed to save broken structure (%s) to %s", reason, broken_structs_dir)

def _save_successful_out(out_name: str, constrained_opt: bool, success_out_dir: Union[str, None]) -> None:
    """Copy a successfully converged ORCA .out file of final (unconstrained)
    optimization stage conformers."""
    if not success_out_dir:
        return
    try:
        Path(success_out_dir).mkdir(parents=True, exist_ok=True)
        struct_id = run_state.peek_structure_id()
        dest = Path(success_out_dir) / f"{struct_id}_{'PreOPT' if constrained_opt else 'OPT'}_{Path(out_name).name}"
        shutil.copy(out_name, dest)
        logger.debug("Saved successful conformer .out to %s", dest)
    except Exception:
        logger.exception("Failed to save successful .out file %s to %s", out_name, success_out_dir)

def check_is_broken(
    xyz_block: str,
    len_threshold: float | None = None,
) -> tuple[bool, Union[tuple[int, int], None]]:
    """Return True if any two atoms in xyz_block are unphysically close.
    When len_threshold is None sum of radii is used instead,
    which is physically more accurate and avoids false positives for heavy atoms.
    """
    lines = [l for l in xyz_block.strip().split('\n') if l.strip()]

    # Strip XYZ header (first two lines) if present
    if lines and lines[0].strip().lstrip('-').isdigit():
        lines = lines[2:]

    symbols = []
    coords = []
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    coord_matrix = np.array(coords)
    n = coord_matrix.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coord_matrix[i] - coord_matrix[j])
            # Legacy behaviour: single global threshold
            threshold = len_threshold if len_threshold is not None else _clash_threshold(symbols[i], symbols[j])
            if dist <= threshold:
                logger.warning(
                    "Clash detected: atoms %d(%s) and %d(%s) distance=%.3f A "
                    "<= threshold=%.3f A",
                    i, symbols[i], j, symbols[j], dist, threshold
                )
                return True, (i, j)
    return False, None

def _check_rings_intact(
    xyz_block: str,
    original_mol: Chem.rdchem.Mol,
    bond_threshold: float | None = None,
    ts_bonds=None,
) -> tuple[bool, Union[tuple[int, int], None]]:
    """
    Verifies that all rings of molecule are remained 
    intact in proposed structure.
    bond_threshold: the maximum allowed bond length.
    """

    cfg = _get_config_or_raise()
    ts_multiplier = 0.8 if cfg.ts else 0.5
    ts_bond_set = {frozenset(b) for b in (ts_bonds or [])}

    lines = [l for l in xyz_block.strip().split('\n') if l.strip()]
    if lines and lines[0].strip().lstrip('-').isdigit():
        lines = lines[2:]

    coords = {}
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4:
            continue
        coords[i] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

    for ring in original_mol.GetRingInfo().AtomRings():
        for j in range(len(ring)):
            a = ring[j]
            b = ring[(j + 1) % len(ring)]
            if frozenset((a, b)) in ts_bond_set:
                continue
            
            if a not in coords or b not in coords:
                logger.warning(
                    "Ring atom %d or %d missing from xyz block", a, b
                )
                return False

            dist = np.linalg.norm(coords[a] - coords[b])

            if bond_threshold is not None:
                threshold = bond_threshold
            else:
                sym_a = original_mol.GetAtomWithIdx(a).GetSymbol()
                sym_b = original_mol.GetAtomWithIdx(b).GetSymbol()
                threshold = _bond_break_threshold(sym_a, sym_b) * ts_multiplier

            if dist > threshold:
                sym_a = original_mol.GetAtomWithIdx(a).GetSymbol()
                sym_b = original_mol.GetAtomWithIdx(b).GetSymbol()
                logger.warning(
                    "Ring bond %d(%s)-%d(%s) opened: length %.3f A "
                    "> vdw threshold %.3f A",
                    a, sym_a, b, sym_b, dist, threshold
                )
                return False, (a, b)

    return True, None

def _check_ts_bonds_within_limit(xyz_block: str, ts_bonds, max_length: float) -> tuple[bool, Union[tuple[int, int], None]]:
    """Checks if TS-bonds are within max_length, because RingInfo does not see bonds with H"""
    if not ts_bonds:
        return True
    lines = [l for l in xyz_block.strip().split('\n') if l.strip()]
    if lines and lines[0].strip().lstrip('-').isdigit():
        lines = lines[2:]
    coords = {}
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4:
            continue
        coords[i] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
    for a, b in ts_bonds:
        if a not in coords or b not in coords:
            logger.warning("TS-bond %d-%d: atom missing in xyz", a, b)
            return False
        dist = np.linalg.norm(coords[a] - coords[b])
        if dist > max_length:
            logger.warning("TS-bond %d-%d surpassed the limit: %.3f A > %.3f A", a, b, dist, max_length)
            return False, (a, b)
    return True, None

def calc_energy(
        mol_file_name: str,
        dihedrals: list[dihedral] = [],
        norm_energy: float = 0,
        save_structs: bool = True,
        constrained_opt: bool = False,
        force_xyz_block: Union[None, str] = None,
        ik_loss=None,
        config: ConfSearchConfig = None,
        original_mol=None,
        broken_structs_dir: Union[str, None] = None,
        success_out_dir: Union[str, None] = None,
        ts_bonds=None, ts_bond_max_length=5.0,
        fixed_dihedrals=None,
        extra_constraints=None,
) -> float:
    """
        Calculates energy of molecule from 'mol_file_name'
        with current properties and returns it as float.
        If config is provided, uses its values; otherwise uses module-level defaults.
    """
    # Use explicit config if provided, otherwise require central config
    if config is None:
        cfg = _get_config_or_raise()
    else:
        cfg = config

    logger.debug("Calc with save_struct=%s", save_structs)

    xyz_upd = None

    hard_constraints = []
    for atoms, value in (fixed_dihedrals or []):
        hard_constraints.append(Constraint("dihedral", tuple(atoms), round(np.rad2deg(value), 6)))
    hard_constraints.extend(extra_constraints or [])
    
    logger.debug("dihedrals before: %s", dihedrals)
    logger.debug(
        "ORCA hard_constraints (%d): %s",
        len(hard_constraints),
        hard_constraints,
    )
    
    if force_xyz_block:
        xyz_upd = force_xyz_block
    else:
        xyz_upd = change_dihedrals(
            mol_file_name, dihedrals, ik_loss,
            ts_bonds=ts_bonds, fixed_dihedrals=fixed_dihedrals,
            extra_constraints=extra_constraints,
        )
    
    logger.debug("dihedrals after: %s", dihedrals)

    broken, clash_atoms = check_is_broken(xyz_upd)
    if broken:
        broken_energy = cfg.broken_struct_energy
        logger.warning(
            "Seems that some atoms in current structure are closer than the threshold! Returning broken_struct_energy=%s",
            broken_energy,
        )
        atoms_raw1 = tuple(with_h_canonical_to_raw1(a, mol_file_name) for a in clash_atoms) if clash_atoms else None
        _save_broken_struct(xyz_upd, broken_structs_dir, "clash", atoms_raw1)
        return broken_energy, False

    if ts_bonds:
        within_limit, exceeded_atoms = _check_ts_bonds_within_limit(xyz_upd, ts_bonds, ts_bond_max_length)
        if not within_limit:
            atoms_raw1 = tuple(with_h_canonical_to_raw1(a, mol_file_name) for a in exceeded_atoms) if exceeded_atoms else None
            _save_broken_struct(xyz_upd, broken_structs_dir, "ts_bond_exceeded", atoms_raw1)
            return cfg.broken_struct_energy, False

    if ik_loss is not None:
        rings_intact, opened_atoms = _check_rings_intact(xyz_upd, original_mol, ts_bonds=ts_bonds)
        if not rings_intact:
            logger.warning("Ring has opened in candidate — skipping ORCA")
            atoms_raw1 = tuple(with_h_canonical_to_raw1(a, mol_file_name) for a in opened_atoms) if opened_atoms else None
            _save_broken_struct(xyz_upd, broken_structs_dir, "ring_open", atoms_raw1)
            return cfg.broken_struct_energy, False

    opt_status = True

    inp_name = mol_to_inp_name(mol_file_name)
    out_name = inp_to_out_name(inp_name)

    if Path(out_name).is_file():
        try:
            Path(out_name).unlink(missing_ok=True)
        except Exception:
            # fallback to remove via shell
            subprocess.run(["rm", "-f", out_name])
    
    generate_oinp(
        xyz_upd,
        dihedrals,
        inp_name,
        constrained_opt=constrained_opt,
        num_of_procs=cfg.num_of_procs,
        method_of_calc=cfg.orca_method,
        charge=cfg.charge,
        multipl=cfg.spin_multiplicity,
        hard_constraints=hard_constraints,
    )
    start_calc(inp_name)
    
    res, opt_status, n_cycles = find_energy_in_log(out_name)
    
    logger.info(
        "ORCA finished: success=%s  cycles=%s  energy=%.6f",
        opt_status, n_cycles, res
    )

    res = res if not opt_status else res * HARTRI_TO_KCAL - norm_energy
    logger.debug("opt status in calc_energy is %s", opt_status)
    if not opt_status:
        _save_broken_struct(xyz_upd, broken_structs_dir, "opt_failed",
                            extra_files=[out_name])
    else:
        _save_successful_out(out_name, constrained_opt, success_out_dir)
    return res, opt_status

def load_last_optimized_structure_xyz_block(mol_file_name : str) -> str:
    xyz_path = _qc_calcs_dir(mol_file_name) / (Path(mol_file_name).stem + ".xyz")
    with open(xyz_path, 'r') as xyz_file:
        full_xyz = xyz_file.readlines()
    return ''.join(full_xyz[2:])

# `increase_structure_id` is provided by `run_state`.

def dihedral_angle(a : list[float], b : list[float], c : list[float], d : list[float]) -> float:
    """
    Calculates dihedral angle between 4 points
    """
    
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    
    # Next: compute signed dihedral angle in terms used by RDKit
    #Vars named like in rdkit source code

    lengthSq = lambda u : np.sum(u ** 2)
    
    nIJK = np.cross(b - a, c - b)
    nJKL = np.cross(c - b, d - c)
    m = np.cross(nIJK, c - b)

    res =  -np.arctan2(np.dot(m, nJKL) / np.sqrt(lengthSq(m) * lengthSq(nJKL)),\
                       np.dot(nIJK, nJKL) / np.sqrt(lengthSq(nIJK) * lengthSq(nJKL)))
    return (res + 2 * np.pi) % (2 * np.pi)
       
def parse_points_from_trj(
    trj_file_name : str,
    dihedrals : list,
    norm_en : float, 
    save_structs : bool = True,
    structures_path : str = "structs/", 
    return_minima : bool = True,
) -> Union[list[tuple[list[dihedral], float]], tuple[list[tuple[list[dihedral], float]], tuple[list[dihedral], float]]]:
    """
        Parse more points from trj orca file
        returns list of description of dihedrals
        for every point
    """

    logger.debug("Parsing starts with norm_en=%s, save_struct=%s", norm_en, save_structs)

    result = []

    structures = []

    # use internal counter for structure ids (from run_state)

    with open(trj_file_name, "r") as file:
        lines = [line[:-1] for line in file]
        n = int(lines[0])
        for i in range(len(lines) // (n + 2)):
            structures.append("\n".join(lines[i * (n + 2) : (i + 1) * (n + 2)]))
            
            energy = float(lines[i * (n + 2) + 1].split()[-1]) * HARTRI_TO_KCAL - norm_en
            cur_d = []
            for a, b, c, d in dihedrals:
                a_coord = list(map(float, lines[i * (n + 2) + 2 + a].split()[1:]))
                b_coord = list(map(float, lines[i * (n + 2) + 2 + b].split()[1:]))
                c_coord = list(map(float, lines[i * (n + 2) + 2 + c].split()[1:]))
                d_coord = list(map(float, lines[i * (n + 2) + 2 + d].split()[1:]))    
                cur_d.append(dihedral_angle(a_coord, b_coord, c_coord, d_coord))
            result.append((cur_d, energy, structures[i]))
    
    logger.debug("Points in trj: %s", len(result))
    
    if len(result) == 1:
        minima_node = {
            "coords": result[0][0],
            "rel_en": result[0][1],
            "xyz_block": result[0][2],
        }
        return result, minima_node

    points, obs, _ = list(zip(*result[1:]))

    num_of_clusters = min(3, len(points))
    logger.debug("Num of clusters: %s", num_of_clusters)

    vals = {cluster_id : (1e9, -1) for cluster_id in range(num_of_clusters)}

    model = KMeans(n_clusters=num_of_clusters)
    model.fit(points)
    
    for i in range(len(points)):
        cluster = model.predict([points[i]])[0]
        #print(cluster)
        if vals[cluster][0] > obs[i]:
            vals[cluster] = obs[i], i
    
    logger.debug("PARSING POINTS, CLUSTER NUM = %s", num_of_clusters)
    if save_structs:
        logger.info("SAVING STRUCTS")
        cur_id = run_state.peek_structure_id()
        logger.info("Saving first struct from trj. Current structure number: %s", cur_id)
        with open(structures_path + str(cur_id) + ".xyz", "w") as file:
            file.write(structures[0])
        logger.info("saved")
        run_state.increase_structure_id()

        for cluster_id in vals:
            cur_id = run_state.peek_structure_id()
            logger.info("saving struct number %s", cur_id)
            with open(structures_path + str(cur_id) + ".xyz", "w") as file:
                file.write(structures[vals[cluster_id][1] + 1]) # because points parsed from result[1:]
            logger.info("saved")
            run_state.increase_structure_id()
   
    minima_node = {
        "coords" : result[-1][0],
        "rel_en" : result[-1][1],
        "xyz_block" : structures[-1]
    }

    return (
        [result[0]] + [
            (points[vals[cluster_id][1]], vals[cluster_id][0], result[vals[cluster_id][1] + 1][2])
            for cluster_id in vals
        ],
        minima_node
    )

def compute_mol_hash(mol_file_name: str, charge: int, multipl: int) -> str:
    """Canonical, filename-independent identity for norm_energy caching."""
    mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)
    mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(mol, canonical=True)
    key = f"{smiles}|q={charge}|m={multipl}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()