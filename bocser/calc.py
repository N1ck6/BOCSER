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
        
        if cfg.ts and cfg.use_grass:
            tmp.write(str(coords.count('\n')))
            tmp.write("\n\n")
            tmp.write(coords)
        else:
            opt_cmd = "OptTS" if cfg.ts else "Opt"
                
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
    subprocess.run(["sbatch", "-W", "-t", str(timeout_minutes), "-o", "/dev/null", sbatch_name])
    
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
        for j in range(i+1, coord_matrix.shape[0]):
            #print(np.linalg.norm(coord_matrix[i, :] - coord_matrix[j, :]))
            if np.linalg.norm(coord_matrix[i, :] - coord_matrix[j, :]) <= len_threshold:
                return True
    return False

def _check_rings_intact(
    xyz_block: str,
    original_mol: Chem.rdchem.Mol,
    bond_threshold: float = 1.8
) -> bool:
    """
    Verifies that all rings of molecule are remained 
    intact in proposed structure.
    bond_threshold: the maximum allowed bond length.
    """
    cfg.ts
    lines = [l for l in xyz_block.strip().split('\n') if l.strip()]
    start = 2 if lines[0].strip().isdigit() else 0
    lines = lines[start:]

    coords = {}
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 4:
            continue
        coords[i] = np.array([float(parts[1]), float(parts[2]), float(parts[3])])

    ring_info = original_mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        for j in range(len(ring)):
            a = ring[j]
            b = ring[(j + 1) % len(ring)]
            if a not in coords or b not in coords:
                logger.warning("Atom %d or %d is not found in XYZ block", a, b)
                return False
            dist = np.linalg.norm(coords[a] - coords[b])
            if dist > bond_threshold:
                logger.warning(
                    "Ring bond %d-%d opened: length %.3f Å > %.3f Å",
                    a, b, dist, bond_threshold
                )
                return False
    return True

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

    if ik_loss is not None:
        bond_length = cfg.bond_length_threshold * 2.5 # 1.75 for normal
        if cfg.ts: bond_length += 0.75 # more length for ts
        if not _check_rings_intact(xyz_upd, original_mol, bond_length):
            logger.warning("Ring has opened in candidate — skipping ORCA")
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
    )
    start_calc(inp_name)
    
    res, opt_status = find_energy_in_log(out_name) 
    res = res if not opt_status else res * HARTRI_TO_KCAL - norm_energy
    logger.debug("opt status in calc_energy is %s", opt_status)
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
