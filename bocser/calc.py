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

HARTRI_TO_KCAL = 627.509474063 

ORCA_EXEC_COMMAND = "/opt/orca5/orca"
NUM_OF_PROCS = 8
DEFAULT_METHOD = None
ORCA_METHOD = "lda sto-3g"
CHARGE = 0
MULTIPL = 1
TS = False
BROKEN_STRUCT_ENERGY = 100.0
BOND_LENGTH_THRESHOLD = 0.7
_CURRENT_STRUCTURE_ID = 0  # global id for every structure that we would save (use accessors)

ACQUISITION_FUNCTION = ConfSearchConfig.acquisition_function

WRONG_GEOMETRY = False

#Alias for type of node about dihedral angle 
#that consists of list with four atoms and value of degree
dihedral = tuple[list[int], float]

def set_config(config: ConfSearchConfig) -> None:
    """Set runtime parameters for calculation functions from a `ConfSearchConfig`.

    This replaces the previous dict-based loader and centralizes config
    propagation. Call this early in your program (e.g. from `conf_search.py`).
    """
    global MULTIPL, CHARGE, ORCA_EXEC_COMMAND, NUM_OF_PROCS, ORCA_METHOD, TS, BROKEN_STRUCT_ENERGY, BOND_LENGTH_THRESHOLD, ACQUISITION_FUNCTION

    ORCA_EXEC_COMMAND = config.orca_exec_command
    NUM_OF_PROCS = config.num_of_procs
    ORCA_METHOD = config.orca_method
    CHARGE = config.charge
    MULTIPL = config.spin_multiplicity
    TS = config.ts
    BROKEN_STRUCT_ENERGY = config.broken_struct_energy
    BOND_LENGTH_THRESHOLD = config.bond_length_threshold
    ACQUISITION_FUNCTION = config.acquisition_function

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

        if ACQUISITION_FUNCTION != 'ik':
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
        print("No such file!")
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
    with open(gjf_name, 'w+') as file:
        opt_cmd = "opt"
        if TS:
            opt_cmd = "OptTS"
        file.write("!" + method_of_calc + f" {opt_cmd}\n")
        file.write("%pal\nnprocs " + str(num_of_procs) + "\nend\n")
        if constrained_opt:
            file.write("%geom Constraints\n")
            dihedrals = to_degrees(dihedrals)
            for cur in dihedrals:
                a, d = cur
                file.write("{ D " + " ".join(map(str, a)) + " " + str(d) + " C }\n")
            file.write("end\n")
            file.write("end\n")    
        file.write("* xyz " + str(charge) + " " + str(multipl) + "\n")
        file.write(coords)
        file.write("END\n")

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
    num_of_procs = num_of_procs if num_of_procs is not None else NUM_OF_PROCS
    orca_method = orca_method if orca_method is not None else ORCA_METHOD
    charge = charge if charge is not None else CHARGE
    multipl = multipl if multipl is not None else MULTIPL
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


def start_calc(gjf_name: str, sbatch: bool = True):
    """
        Running calculation
    """	
    if sbatch:
        sbatch_name = gjf_name.split('/')[-1][:-4] + ".sh"
        try:
            shutil.copy("sbatch_temp", sbatch_name)
            # append the orca call into the sbatch script
            with open(sbatch_name, "a") as fh:
                fh.write(f"{ORCA_EXEC_COMMAND} {gjf_name} > {gjf_name[:-4]}.out\n")
            subprocess.run(["sbatch", sbatch_name], check=True)
        except Exception:
            # fall back to running the previous approach if sbatch fails
            subprocess.run(shlex.split(ORCA_EXEC_COMMAND) + [gjf_name], check=False)
    else:
        out_path = gjf_name[:-4] + ".out"
        try:
            with open(out_path, "wb") as out_f:
                subprocess.run(shlex.split(ORCA_EXEC_COMMAND) + [gjf_name], stdout=out_f, check=False)
        except Exception:
            # best-effort fallback using shell in case ORCA_EXEC_COMMAND is complex
            subprocess.run(f"{ORCA_EXEC_COMMAND} {gjf_name} > {out_path}", shell=True)

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

def wait_for_the_end_of_calc(log_name : str, timeout):
    """
        waiting fot the end of calculation in gaussian 
        by checking log file every 'timeout' ms
    """
    while True:
        try: 
            with open(log_name, 'r') as file:
                log_file = [line for line in file]               
                if "ORCA TERMINATED NORMALLY" in log_file[-2] or\
                        "ORCA finished by error" in log_file[-5] or\
                        "Error" in log_file[-2] or\
                        "GSTEP" in log_file[-2]:
                    break
        except FileNotFoundError:
            pass
        except IndexError:
            pass
        finally:
            time.sleep(timeout / 1000)

def find_energy_in_log(log_name : str) -> tuple[float, bool]:
    """
        finds energy of structure in log file
    """
    try:
        with open(log_name, 'r') as file:
            en_line = [line for line in file if "FINAL SINGLE POINT ENERGY" in line][-1]
            en = float(en_line.split()[4])
            return en, True
    except FileNotFoundError:
        print("No log file! Something went wrong! Finishing!")
        exit(0) #TODO: Make hooks for finishing
    except IndexError:
        print(f"Seems that optimization finished with error! Check it carefuly by yourself! Returning default energy for broken structures: {BROKEN_STRUCT_ENERGY}")
        return BROKEN_STRUCT_ENERGY, False #TODO: find better way to handle errors in orca

def check_is_broken(
    xyz_block : str,
    len_threshold : float = BOND_LENGTH_THRESHOLD
) -> bool:
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
    # Use config values if provided, else fall back to globals set by set_config()
    bond_len_threshold = config.bond_length_threshold if config else BOND_LENGTH_THRESHOLD

    print(f"Calc with save_struct={save_structs}")

    xyz_upd = None

    print("dihedrals before: ", dihedrals)
    if force_xyz_block:
        xyz_upd = force_xyz_block
    else:
        xyz_upd = change_dihedrals(mol_file_name, dihedrals, ik_loss)

    print("dihedrals after: ", dihedrals)

    if check_is_broken(xyz_upd, bond_len_threshold):
        broken_energy = config.broken_struct_energy if config else BROKEN_STRUCT_ENERGY
        print(f"Seems that some atoms in current structure is closer than {bond_len_threshold}!")
        print(f"Returning broken_struct_energy that is {broken_energy}")
        return broken_energy, False

    opt_status = True

    inp_name = mol_to_inp_name(mol_file_name)
    out_name = inp_to_out_name(inp_name)

    if os.path.isfile(out_name):
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
        num_of_procs=config.num_of_procs if config else None,
        orca_method=config.orca_method if config else None,
        charge=config.charge if config else None,
        multipl=config.spin_multiplicity if config else None,
    )
    start_calc(inp_name)
    wait_for_the_end_of_calc(out_name, 1000)
    
    res, opt_status = find_energy_in_log(out_name) 
    res = res if not opt_status else res * HARTRI_TO_KCAL - norm_energy
    print(f"opt status in calc_energy is {opt_status}")
    return res, opt_status

def load_last_optimized_structure_xyz_block(mol_file_name : str) -> str:
    full_xyz = []
    with open(mol_file_name[:-4] + '.xyz', 'r') as xyz_file:
        for line in xyz_file:
            full_xyz.append(line)
    return ''.join(full_xyz[2:])

def increase_structure_id() -> int:
    """Increment and return a new structure id.

    The module-level counter is internal; use this accessor instead of
    touching the variable directly.
    """
    global _CURRENT_STRUCTURE_ID
    _CURRENT_STRUCTURE_ID += 1
    return _CURRENT_STRUCTURE_ID - 1

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

    print(f"Parsing starts with norm_en={norm_en}, save_struct={save_structs}")

    result = []

    structures = []

    # use internal counter for structure ids
    global _CURRENT_STRUCTURE_ID

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
    
    print(f"Points in trj: {len(result)}")
    
    if len(result) == 1:
        return result

    points, obs = list(zip(*result[1:]))

    num_of_clusters = min(3, len(points))
    print(f"Num of clusters: {num_of_clusters}")

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
    print(f"PARSING POINTS, CLUSTER NUM = {num_of_clusters}")
    if save_structs:

        print("SAVING STRUCTS")
        print("Saving first struct from trj. Current structure number: {}".format(_CURRENT_STRUCTURE_ID))
        with open(structures_path + str(_CURRENT_STRUCTURE_ID) + ".xyz", "w") as file:
            file.write(structures[0])
        print("saved")
        _CURRENT_STRUCTURE_ID += 1

        for cluster_id in vals:
            print("saving struct number {}".format(_CURRENT_STRUCTURE_ID))
            with open(structures_path + str(_CURRENT_STRUCTURE_ID) + ".xyz", "w") as file:
                file.write(structures[vals[cluster_id][1] + 1]) # because points parsed from result[1:]
            print("saved")
            _CURRENT_STRUCTURE_ID += 1
   
    minima_node = {
        "coords" : result[-1][0],
        "rel_en" : result[-1][1],
        "xyz_block" : structures[-1]
    }

    return [result[0]] + [(points[vals[cluster_id][1]], vals[cluster_id][0]) for cluster_id in vals], minima_node
