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
import subprocess
import shlex
import shutil
from pathlib import Path

import tempfile
import logging
logger = logging.getLogger(__name__)
import config_manager
import run_state

HARTRI_TO_KCAL = 627.509474063 

#Alias for type of node about dihedral angle 
#that consists of list with four atoms and value of degree
dihedral = tuple[list[int], float]


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

def change_dihedrals(mol_file_name: str,
                     dihedrals: list[list[tuple[tuple[int,int,int,int], float]]],
                     ik_loss=None,
                     full_block=False):
    try:
        mol = Chem.MolFromMolFile(mol_file_name, removeHs=False)
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

            mp = AllChem.MMFFGetMoleculeProperties(tmp_mol, mmffVariant='MMFF94')
            ff = AllChem.MMFFGetMoleculeForceField(tmp_mol, mp)

            for bl_dict in ik_loss.bond_lengths:
                for (a, b), value in bl_dict.items():
                    ff.MMFFAddDistanceConstraint(a, b, False, value, value, 1e3)

            for va_dict in ik_loss.valence_angles:
                for (a, b, c), value in va_dict.items():
                    ff.MMFFAddAngleConstraint(a, b, c, False,
                                              np.rad2deg(value),
                                              np.rad2deg(value), 1e2)

            for (a, b, c, d), value in dihedrals:
                ff.MMFFAddTorsionConstraint(a, b, c, d, False,
                                            np.rad2deg(-value),
                                            np.rad2deg(-value), 1)

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
        dihedrals : list[dihedral],
        gjf_name : str, 
        num_of_procs : int, 
        method_of_calc : str,
        charge : int,
        multipl : int,
        constrained_opt : bool = False
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
        opt_cmd = "opt"
        if cfg.ts:
            opt_cmd = "OptTS"
        tmp.write("!" + method_of_calc + f" {opt_cmd}\n")
        tmp.write("%pal\nnprocs " + str(num_of_procs) + "\nend\n")
        if constrained_opt:
            tmp.write("%geom Constraints\n")
            dihedrals = to_degrees(dihedrals)
            for cur in dihedrals:
                a, d = cur
                tmp.write("{ D " + " ".join(map(str, a)) + " " + str(d) + " C }\n")
            tmp.write("end\n")
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

def generate_default_oinp(
    coords: str,
    dihedrals: list[dihedral],
    oinp_name: str,
    constrained_opt: bool = False,
    num_of_procs: int = None,
    orca_method: str = None,
    charge: int = None,
    multipl: int = None,
) -> None:
    """Generate ORCA input file using module globals or explicit parameters.
    
    If optional parameters are not provided, uses module-level defaults set by set_config().
    """
    if num_of_procs is None or orca_method is None or charge is None or multipl is None:
        cfg = _get_config_or_raise()
    num_of_procs = num_of_procs if num_of_procs is not None else cfg.num_of_procs
    orca_method = orca_method if orca_method is not None else cfg.orca_method
    charge = charge if charge is not None else cfg.charge
    multipl = multipl if multipl is not None else cfg.spin_multiplicity
    generate_oinp(
        coords,
        dihedrals,
        oinp_name,
        num_of_procs,
        orca_method,
        charge,
        multipl,
        constrained_opt=constrained_opt,
    )


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


def start_calc(gjf_name: str, sbatch: bool = True):
    """
        Running calculation
    """	
    cfg = _get_config_or_raise()
    orca_cmd = cfg.orca_exec_command
    if sbatch:
        # Place the generated sbatch script next to the input file so scripts
        # and outputs live inside the working folder instead of the module cwd.
        gjf_path = Path(gjf_name).resolve()
        gjf_dir = gjf_path.parent
        gjf_base = gjf_path.stem
        sbatch_name = str(gjf_dir / (gjf_base + ".sh"))
        try:
            # Prefer a template located in the same directory as the input file.
            template_to_copy = _select_sbatch_template(gjf_dir, cfg)
            shutil.copy(template_to_copy, sbatch_name)
            # append the orca call into the sbatch script
            with open(sbatch_name, "a") as fh:
                fh.write(f"{orca_cmd} {gjf_name} > {gjf_name[:-4]}.out\n")
            subprocess.run(["sbatch", sbatch_name], check=True)
        except Exception:
            # fall back to running the previous approach if sbatch fails
            subprocess.run(shlex.split(orca_cmd) + [gjf_name], check=False)
    else:
        out_path = gjf_name[:-4] + ".out"
        try:
            with open(out_path, "wb") as out_f:
                subprocess.run(shlex.split(orca_cmd) + [gjf_name], stdout=out_f, check=False)
        except Exception:
            # best-effort fallback using shell in case ORCA_EXEC_COMMAND is complex
            subprocess.run(f"{orca_cmd} {gjf_name} > {out_path}", shell=True)

def mol_to_inp_name(mol_file_name : str) -> str:
    """
        generating name of inp file from mol file name
    """
    return mol_file_name[:-4] + ".inp"

def inp_to_out_name(inp_file_name : str) -> str:
    """
        generating name of out file from inp file name
    """
    return inp_file_name[:-4] + ".out"

def wait_for_the_end_of_calc(log_name: str, poll_interval_ms: int | None = None, timeout_seconds: int | None = None) -> bool:
    """Monitor ORCA output file until the run finishes or times out.

    Returns True if ORCA terminated normally, False if it finished with an
    error. Raises `TimeoutError` if the file doesn't reach a terminal state
    within `timeout_seconds`.

    Both `poll_interval_ms` and `timeout_seconds` default to values from the
    runtime config if not provided.
    """
    cfg = config_manager.get_config()
    if poll_interval_ms is None:
        poll_interval_ms = cfg.orca_poll_interval_ms if cfg is not None else 1000
    if timeout_seconds is None:
        timeout_seconds = cfg.orca_poll_timeout_seconds if cfg is not None else 3600

    poll_s = max(0.05, poll_interval_ms / 1000.0)
    deadline = time.time() + float(timeout_seconds)

    # Keep a small rolling buffer of recent lines to avoid reading entire huge
    # log files repeatedly.
    from collections import deque

    while True:
        if time.time() > deadline:
            raise TimeoutError(f"Waiting for ORCA output {log_name} timed out after {timeout_seconds} seconds")

        try:
            with open(log_name, "r", errors="ignore") as fh:
                last_lines = deque(fh, maxlen=200)
                joined = "\n".join(last_lines)

                if "ORCA TERMINATED NORMALLY" in joined:
                    return True
                if "ORCA finished by error" in joined or "Error" in joined or "GSTEP" in joined:
                    return False
        except FileNotFoundError:
            # File may not yet exist; continue polling until timeout
            logger.debug("Output file %s not present yet; polling...", log_name)
        except Exception:
            logger.exception("Error while monitoring output file %s", log_name)

        time.sleep(poll_s)

def find_energy_in_log(log_name : str) -> tuple[float, bool]:
    """
        finds energy of structure in log file
    """
    import re
    energy_re = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")
    alt_re = re.compile(r"TOTAL ENERGY\s*[:=]\s*(-?\d+\.\d+)")

    try:
        with open(log_name, 'r', errors='ignore') as fh:
            # scan from end to find the last occurrence without loading whole huge files
            from collections import deque
            last_lines = deque(fh, maxlen=500)
            joined = "\n".join(last_lines)

            m = energy_re.search(joined)
            if not m:
                m = alt_re.search(joined)
            if m:
                try:
                    en = float(m.group(1))
                    return en, True
                except Exception:
                    logger.exception("Failed to parse energy from line: %s", m.group(0))
                    cfg = _get_config_or_raise()
                    return cfg.broken_struct_energy, False
            # no energy line found -> optimization likely failed; return broken_struct_energy
            logger.warning("No energy line found in %s; returning broken_struct_energy", log_name)
            cfg = _get_config_or_raise()
            return cfg.broken_struct_energy, False
    except FileNotFoundError:
        logger.error("No log file: %s. Returning broken_struct_energy", log_name)
        cfg = _get_config_or_raise()
        return cfg.broken_struct_energy, False

def check_is_broken(
    xyz_block : str,
    len_threshold : float | None = None
) -> bool:
    # Determine bond-length threshold from config unless explicitly provided
    if len_threshold is None:
        cfg = _get_config_or_raise()
        len_threshold = cfg.bond_length_threshold

    coord_matrix = np.asarray(
        list(
            map(
                lambda s: list(
                    map(
                        float, 
                        s.split()[1:]
                    )
                ),
                xyz_block.strip().split('\n')
            )
        )
    )
    for i in range(coord_matrix.shape[0]):
        for j in range(i+1, coord_matrix.shape[1]):
            #print(np.linalg.norm(coord_matrix[i, :] - coord_matrix[j, :]))
            if np.linalg.norm(coord_matrix[i, :] - coord_matrix[j, :]) <= len_threshold:
                return True
    return False
 
def calc_energy(
        mol_file_name: str,
        dihedrals: list[dihedral] = [],
        norm_energy: float = 0,
        save_structs: bool = True,
        constrained_opt: bool = False,
        force_xyz_block: Union[None, str] = None,
        ik_loss=None,
        config: ConfSearchConfig = None,
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
    bond_len_threshold = cfg.bond_length_threshold

    logger.debug("Calc with save_struct=%s", save_structs)

    xyz_upd = None

    logger.debug("dihedrals before: %s", dihedrals)
    if force_xyz_block:
        xyz_upd = force_xyz_block
    else:
        xyz_upd = change_dihedrals(mol_file_name, dihedrals, ik_loss)

    logger.debug("dihedrals after: %s", dihedrals)

    if check_is_broken(xyz_upd, bond_len_threshold):
        broken_energy = cfg.broken_struct_energy
        logger.warning(
            "Seems that some atoms in current structure are closer than %s! Returning broken_struct_energy=%s",
            bond_len_threshold,
            broken_energy,
        )
        return broken_energy, False

    opt_status = True

    inp_name = mol_to_inp_name(mol_file_name)
    out_name = inp_to_out_name(inp_name)

    if Path(out_name).is_file():
        try:
            Path(out_name).unlink(missing_ok=True)
        except Exception:
            # fallback to remove via shell
            subprocess.run(["rm", "-f", out_name])
    generate_default_oinp(
        xyz_upd,
        dihedrals,
        inp_name,
        constrained_opt=constrained_opt,
        num_of_procs=cfg.num_of_procs,
        orca_method=cfg.orca_method,
        charge=cfg.charge,
        multipl=cfg.spin_multiplicity,
    )
    start_calc(inp_name)
    # Use configured polling defaults inside wait_for_the_end_of_calc
    wait_for_the_end_of_calc(out_name)
    
    res, opt_status = find_energy_in_log(out_name) 
    res = res if not opt_status else res * HARTRI_TO_KCAL - norm_energy
    logger.debug("opt status in calc_energy is %s", opt_status)
    return res, opt_status

def load_last_optimized_structure_xyz_block(mol_file_name : str) -> str:
    full_xyz = []
    with open(mol_file_name[:-4] + '.xyz', 'r') as xyz_file:
        for line in xyz_file:
            full_xyz.append(line)
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
            result.append((cur_d, energy))
    
    logger.debug("Points in trj: %s", len(result))
    
    if len(result) == 1:
        return result

    points, obs = list(zip(*result[1:]))

    num_of_clusters = min(3, len(points))
    logger.debug("Num of clusters: %s", num_of_clusters)

    vals = {cluster_id : (1e9, -1) for cluster_id in range(num_of_clusters)}

    #print(points)
    #print(len(points))

    model = KMeans(n_clusters=num_of_clusters)
    model.fit(points)
    
    for i in range(len(points)):
        cluster = model.predict([points[i]])[0]
        #print(cluster)
        if vals[cluster][0] > obs[i]:
            vals[cluster] = obs[i], i
    #print(len(vals))
    #print(vals)
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

    return [result[0]] + [(points[vals[cluster_id][1]], vals[cluster_id][0]) for cluster_id in vals], minima_node
