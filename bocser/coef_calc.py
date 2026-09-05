from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import SetDihedralRad

import numpy as np

from typing import Union
import os

from default_vals import ConfSearchConfig
from calc import start_calc
from coef_from_grid import calc_coefs
from db_connector import Connector
import config_manager

import networkx as nx
from ik_loss import CyclicCollection
import logging
logger = logging.getLogger(__name__)


class CoefCalculator:
    """
        This class performs splitting given molecule on
        small parts with only one interesting rotable dihedral.
        Scanning energies of torsion rotation with given method.
        Calculating coefs for mean function for GPRegressor
    """

    def __init__(
        self,
        mol : Chem.rdchem.Mol,
        config : ConfSearchConfig,
        dir_for_inps : str="",
        skip_triple_equal_terminal_atoms=True,
        aromatic_to_aliphatic : bool = True,         
        degrees : np.ndarray = np.linspace(0, 2 * np.pi, 37).reshape(37, 1),
        db_connector : Union[Connector, None] = None,
        fixed_double_bonds: set = None,
        ts_bonds: set = None,
    ) -> None:
        """
            mol - rdkit molecule
            dir_for_inps - path to directory, where scan .inp files will generates
            skip_triple_equal_terminal_atoms - skip dihedrals,
                where one of atoms is RX3, where X is a terminal atom
            num_of_procs - num of procs to calculate
            method_of_calc - method in orca format
            charge - charge of molecule
            multipl - multiplicity
            degrees - degree grid to scan
        """

        self.mol = mol
        self.dir_for_inps = (dir_for_inps.rstrip("/") + "/") if dir_for_inps else ""
        self.skip_triple_equal_terminal_atoms = skip_triple_equal_terminal_atoms
        self.num_of_procs = config.num_of_procs
        self.method_of_calc = config.orca_method
        self.charge = config.charge
        self.multipl = config.spin_multiplicity
        self.af = config.acquisition_function
        self.degrees = degrees
        self.fixed_double_bonds = fixed_double_bonds or set()
        self.ts_bond_set = {frozenset(b) for b in (ts_bonds or set())}
        # Key is SMILES, val is idx
        self.unique_frags = {}
        # Key is atom idxs, val is idx
        self.frags = {}
        self.db_connector = db_connector
        self.aromatic_to_aliphatic = aromatic_to_aliphatic

        self.case_sensetive_atoms = [
            cur for cur in [
                Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), idx) for idx in range(1, 119)
            ] if cur.upper() != cur
        ]

        self.scanfile2smiles = {} # k - scan_file, v - smiles
        self.fetched_coefs = {} # k - smiles, v - coefs

        if not os.path.exists(self.dir_for_inps):
            os.makedirs(self.dir_for_inps)

    def is_terminal(self,
                    atom : Chem.rdchem.Atom) -> bool:
        """
            Returns True if atom is terminal(Hs not counted)
        """
        return len(atom.GetNeighbors()) == 1

    def get_second_atom_in_bond(self,
                                bond : Chem.rdchem.Bond,
                                atom : Chem.rdchem.Atom) -> Chem.rdchem.Atom:
        """
            retruns another atom from this bond
        """
        return bond.GetEndAtom() if bond.GetBeginAtom().GetIdx() == atom.GetIdx() else bond.GetBeginAtom()

    def is_triple_eq_neighbors(self,
                               atom : Chem.rdchem.Atom) -> bool:
        """
        check if current atom has three equal neighbors

        """

        in_bond = None

        for bond in atom.GetBonds():
            if not self.is_terminal(self.get_second_atom_in_bond(bond, atom)):
                in_bond = bond
                break

        if in_bond is None:
            return False

        neighbor_atoms = [cur.GetSymbol() for cur in atom.GetNeighbors()]
        neighbor_atoms.remove(self.get_second_atom_in_bond(in_bond, atom).GetSymbol())

        neighbor_bonds = [cur.GetBondType() for cur in atom.GetBonds()]
        neighbor_bonds.remove(in_bond.GetBondType())

        terminal_neighbors = False

        # 3 terminal neighbors
        if sum([self.is_terminal(cur) for cur in atom.GetNeighbors()]) == 3:
            terminal_neighbors = True

        if(terminal_neighbors and len(neighbor_atoms) == 3 and len(set(neighbor_atoms)) == 1 and len(set(neighbor_bonds)) == 1):
            return True

        return False
    
    def is_terminal_bond(self, bond : Chem.rdchem.Bond) -> bool:
        if len([cur for cur in bond.GetBeginAtom().GetBonds()]) < 2 or\
           len([cur for cur in bond.GetEndAtom().GetBonds()]) < 2 :
            return True
        
        return False
    
    def is_interesting(self,
                       bond : Chem.rdchem.Atom) -> bool:
        """
            Returns True if we should scan this bond
            if skip_triple_equale_terminal_atoms == True - dihedral
            angles, where on one atom there are three equal terminal atoms,
            are not interesting
        """
        if bond.IsInRing() and self.af != 'ik':
            return False

        # If one of atoms is terminal
        if self.is_terminal_bond(bond):
            return False

        # If bond isn't single
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False
        # User defined double bonds
        if frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) in self.fixed_double_bonds:
            return False
        # NEW: TS-bond length is a separate BO search dimension
        if frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) in self.ts_bond_set:
            return False

        if not self.skip_triple_equal_terminal_atoms:
            return True

        # If one of atoms has three equal terminal atom neighbors
        for t_atom in (bond.GetBeginAtom(), bond.GetEndAtom()):
            if self.is_triple_eq_neighbors(t_atom):
                return False

        return True

    def get_unique_mols(
        self,
        lst : list[Chem.rdchem.Mol]
    ) -> list[Chem.rdchem.Mol]:
        """
            Return unqiue mols from lst. Leave first occurance only
        """

        occured_smiles = set()
        result = []
        
        for cur_mol in lst:
            cur_smiles = Chem.MolToSmiles(cur_mol)
            if cur_smiles in occured_smiles:
                continue
            result.append(cur_mol)
            occured_smiles.add(cur_smiles)

        return result

    def generate_3d_coords(self,
                           lst : list[Chem.rdchem.Mol]) -> list[Chem.rdchem.Mol]:
        """
            returns list with same molecules but with
            Hs and generated coords by ETKDG
        """
        result = []
        for mol in lst:
            mol = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            if res == -1:
                # Fallback: random coordinates + MMFF minimization
                res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
            if res == -1:
                logger.error(
                    "EmbedMolecule failed for %s — skipping fragment",
                    Chem.MolToSmiles(Chem.RemoveAllHs(mol))
                )
                continue
            result.append(mol)
        return result

    def get_idxs_to_rotate(self,
                           mol : Chem.rdchem.Mol) -> list[int]:
        """
            Returns idxs of dihedral angel in correct order
        """
        for bond in Chem.RemoveAllHs(mol).GetBonds():

            if self.is_terminal_bond(bond):
                continue

            return ([cur.GetIdx() for cur in bond.GetBeginAtom().GetNeighbors() if cur.GetIdx() != bond.GetEndAtomIdx()][0],
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                    [cur.GetIdx() for cur in bond.GetEndAtom().GetNeighbors() if cur.GetIdx() != bond.GetBeginAtomIdx()][0])

        raise ValueError(
            f"No non-terminal bond found in molecule {Chem.MolToSmiles(Chem.RemoveAllHs(mol))} — cannot determine dihedral indices"
        )

    def _dedup_frags_by_central_bond(self):
        """Keep exactly one 4-tuple key per physical central bond in self.frags.

        Different substituent choices or reverse order previously produced
        multiple keys for the same bond; all were emitted as simultaneous
        hard D constraints in Pre-OPT .inp. Called before the IK position
        map is built so dihedral_ids and ik_loss_dihedrals_idxs stay consistent.
        """
        unique = {}
        for key, coef_idx in self.frags.items():
            axis = frozenset((key[1], key[2]))
            if axis not in unique:
                unique[axis] = (key, coef_idx)
            else:
                logger.info(
                    "Deduplicating frags: dropping %s for central bond %s "
                    "(already have %s).",
                    key, tuple(axis), unique[axis][0],
                )
        self.frags = {key: coef_idx for key, coef_idx in unique.values()}

    def get_ring_dihedrals(self, mol):
        all_rings = [list(r) for r in Chem.GetSymmSSSR(mol)]
        rings = [r for r in all_rings if len(r) >= 4]
        logger.info("GetSymmSSSR found %d cycles, %d for use", len(all_rings), len(rings))

        if not rings:
            logger.warning("Rings are not found in molecule, IK unavailable")
            return [], [], []

        # Ensure self.frags is unique-by-axis before building the position
        # map that IK indices will reference.
        self._dedup_frags_by_central_bond()
        
        frag_key_to_position = {key: i for i, key in enumerate(self.frags.keys())}

        edges = []
        for bond in mol.GetBonds():
            if bond.IsInRing():
                edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

        graph = nx.from_edgelist(edges)
        if graph.number_of_edges() == 0:
            logger.warning("Ring graph has no edges, IK unavailable")
            return [], [], []

        graph.remove_edges_from(nx.bridges(graph))

        all_dihedrals = []
        all_ring_traversals = []
        all_dihedral_idxs = []

        for comp in nx.connected_components(graph):
            if len(comp) <= 2:
                continue

            subg = graph.subgraph(comp)
            
            cycles = nx.cycle_basis(subg)
            cycles = sorted(cycles, key=lambda c: min(c))

            for cycle in cycles:
                ring_nodes = cycle

                if len(ring_nodes) < 4:
                    logger.warning(
                        "Skipping ring of size %d (atoms %s): "
                        "too small for dihedral definition, IK not applicable",
                        len(ring_nodes), ring_nodes
                    )
                    continue

                ring_traversal = CyclicCollection(ring_nodes)
                all_ring_traversals.append(ring_traversal.a)

                dihedrals = [
                    tuple(ring_traversal[i + step] for step in (-1, 0, 1, 2))
                    for i in range(len(ring_nodes))
                ]

                dihedral_idxs = []
                for d in dihedrals:
                    central_bond = frozenset((d[1], d[2]))
                    if central_bond in self.fixed_double_bonds:
                        dihedral_idxs.append(-2) # Fixed
                        continue
                    
                    if central_bond in self.ts_bond_set:
                        dihedral_idxs.append(-3)  # TS-bond axis: IK tracks it via static reference, GP never controls it as a torsion
                        continue

                    found = False
                    for f in self.frags.keys():
                        if all(atom in f for atom in d):
                            dihedral_idxs.append(frag_key_to_position[f])
                            found = True
                            break
                        
                    if not found:
                        existing_axis = None
                        for f in self.frags.keys():
                            if frozenset((f[1], f[2])) == central_bond:
                                existing_axis = f
                                break

                        if existing_axis is not None:
                            logger.info(
                                "Ring window %s (bond %s) does not exactly match existing "
                                "frag axis %s (different substituent atom at a ring-fusion "
                                "branch point, or reverse order) — reusing %s instead of "
                                "creating a second, physically conflicting rotation axis "
                                "for the same bond.",
                                d, tuple(central_bond), existing_axis, existing_axis,
                            )
                            dihedral_idxs.append(frag_key_to_position[existing_axis])
                        else:
                            new_idx = len(self.frags)
                            self.frags[d] = new_idx
                            frag_key_to_position[d] = new_idx
                            dihedral_idxs.append(new_idx)

                all_dihedrals.append(dihedrals)
                all_dihedral_idxs.append(dihedral_idxs)

        return all_dihedrals, all_ring_traversals, all_dihedral_idxs
            
    def convert_all_aromatic_to_aliphatic(
        self,
        cur_smiles : str
    ) -> str:
        """
            Converts all aromatic atoms in SMILES to aliphatic
        """
        tmp_smiles = cur_smiles
        for case_sensetive_atom in self.case_sensetive_atoms:
            if case_sensetive_atom in tmp_smiles:
                tmp_smiles = tmp_smiles.replace(case_sensetive_atom, f"<{case_sensetive_atom}>")
        counter = 0
        result = ""
        for cur in tmp_smiles:
            if cur == '<':
                counter += 1
            if cur == '>':
                counter -= 1
            if counter == 0 and cur.islower():
                result += cur.upper()
            else:
                result += cur
        for case_sensetive_atom in self.case_sensetive_atoms:
            if case_sensetive_atom in result:
                result = result.replace(f"<{case_sensetive_atom}>", case_sensetive_atom)
        return result

    def _sanitize_smiles(
        self,
        cur_smiles : str
    ) -> str:
        cur_mol = Chem.MolFromSmiles(cur_smiles)
        while True:
            logger.debug("Cur smiles: %s; Num of radical electrons: %s", Chem.MolToSmiles(cur_mol), sum([cur.GetNumRadicalElectrons() for cur in cur_mol.GetAtoms()]))
            found_radical_electrons = False
            for atom in cur_mol.GetAtoms():
                found_radical_electrons |= atom.GetNumRadicalElectrons()
                atom.SetNumExplicitHs(atom.GetNumExplicitHs()+atom.GetNumRadicalElectrons())
            cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
            if not found_radical_electrons:
                break
            
        return Chem.MolToSmiles(Chem.RemoveAllHs(cur_mol))

    def get_interesting_frags(self) -> list[Chem.rdchem.Mol]:
        """
            returns a list of simple molecules with one
            rotable interesting torsion angle
            if skip_triple_equale_terminal_atoms == True - dihedral
            angels, where on one atom there are three equal terminal atoms,
            are not interesting
        """

        rotable_frags = []

        count = 0

        ring_info = self.mol.GetRingInfo()

        for bond in self.mol.GetBonds():
            if not self.is_interesting(bond):
                continue

            # For ring bonds under the ik acquisition function, process each
            # ring the bond belongs to separately so that self.frags gets a
            # distinct 4-atom key per ring. This is necessary for fused ring
            # systems (e.g. decalin) where a shared bond must contribute to
            # the ring-closure constraints of both rings.
            if self.af == 'ik' and bond.IsInRing():
                if frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) in self.fixed_double_bonds:
                    continue
                if frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())) in self.ts_bond_set:
                    continue
                bond_rings = [
                    set(ring) for ring in ring_info.AtomRings()
                    if bond.GetBeginAtomIdx() in ring and bond.GetEndAtomIdx() in ring
                ]
                # skip 3-atom rings
                bond_rings = [r for r in bond_rings if len(r) >= 4]
                if not bond_rings:
                    continue
                rings_to_process = bond_rings
            else:
                rings_to_process = [None]

            minor_rings = []
            for ring_atoms in rings_to_process:
                if ring_atoms is not None:
                    atoms_to_use = ring_atoms
                else:
                    atoms_to_use = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                    for atom in [*bond.GetBeginAtom().GetNeighbors(),
                                 *bond.GetEndAtom().GetNeighbors()]:
                        atoms_to_use.add(atom.GetIdx())

                rotable_frag_smiles = Chem.rdmolfiles.MolFragmentToSmiles(self.mol, atomsToUse=list(atoms_to_use))

                if not Chem.MolFromSmiles(rotable_frag_smiles):
                    if self.aromatic_to_aliphatic:
                        rotable_frag_smiles = self.convert_all_aromatic_to_aliphatic(rotable_frag_smiles)
                    else:
                        continue

                rotable_frag_smiles = self._sanitize_smiles(rotable_frag_smiles)
                frag_mol = Chem.MolFromSmiles(rotable_frag_smiles)
                frag_smiles = Chem.MolToSmiles(frag_mol)

                # Add to rotable_frags only the first time we see this SMILES;
                # duplicate entries share the same coefficient index.
                if frag_smiles not in self.unique_frags:
                    rotable_frags.append(frag_mol)
                    self.unique_frags[frag_smiles] = count
                    count += 1

                logger.debug("rot_frag_smiles: %s idxs_to_rotate: %s", frag_smiles, self.get_idxs_to_rotate(frag_mol))

                if ring_atoms is not None:
                    # compute the dihedral axis directly instead of round-tripping through the shared fragment.
                    b_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    b_nbrs = [a.GetIdx() for a in self.mol.GetAtomWithIdx(b_idx).GetNeighbors() if a.GetIdx() != e_idx]
                    e_nbrs = [a.GetIdx() for a in self.mol.GetAtomWithIdx(e_idx).GetNeighbors() if a.GetIdx() != b_idx]
                    if not b_nbrs or not e_nbrs:
                        logger.error(
                            "Ring bond %d-%d has no substituent on one side — skipping dihedral",
                            b_idx, e_idx,
                        )
                        continue
                    old_idxs = (b_nbrs[0], b_idx, e_idx, e_nbrs[0])
                else:
                    query_result = self.mol.GetSubstructMatches(
                        Chem.MolFromSmiles(
                            self._sanitize_smiles(
                                Chem.rdmolfiles.MolFragmentToSmiles(
                                    frag_mol,
                                    atomsToUse=self.get_idxs_to_rotate(frag_mol)
                                )
                            )
                        )
                    )
                    logger.debug("query_result: %s", query_result)

                    old_idxs = ()
                    minor_cycles = []
                    for res in query_result:
                        if len(res) == 4 and all(cur in atoms_to_use for cur in res):
                            old_idxs = res
                            break
                        elif len(res) != 4:
                            if res not in minor_cycles:
                                logger.warning("Skipping dihedral match with %d atoms (need 4): %s", len(res), res)
                                minor_cycles.append(res)

                    if not old_idxs:
                        if atoms_to_use not in minor_rings:
                            logger.error(
                                "No matching substructure found for fragment %s (atoms %s) in molecule — skipping dihedral",
                                frag_smiles, atoms_to_use,
                            )
                            minor_rings.append(atoms_to_use)
                        continue

                real_axis = frozenset((old_idxs[1], old_idxs[2]))
                if real_axis in self.fixed_double_bonds:
                    logger.info(
                        "get_idxs_to_rotate() choose axis %s for bond %s-%s — "
                        "fragment was fixed, so it is skipped",
                        old_idxs, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(),
                    )
                    continue
                
                existing_for_axis = None
                for existing_key in self.frags:
                    if frozenset((existing_key[1], existing_key[2])) == real_axis:
                        existing_for_axis = existing_key
                        break
                if existing_for_axis is not None:
                    if existing_for_axis != old_idxs:
                        logger.debug(
                            "Skipping duplicate window %s for central bond %s "
                            "(already registered as %s) — one axis per bond.",
                            old_idxs, tuple(real_axis), existing_for_axis,
                        )
                    continue

                self.frags[old_idxs] = self.unique_frags[frag_smiles]

        return self.generate_3d_coords(self.get_unique_mols(rotable_frags))

    def get_list_of_xyz(self,
                        lst : list[Chem.rdchem.Mol]) -> list[str]:
        """
            returns list of xyz-blocks of given molecules
        """

        return list(map(Chem.MolToXYZBlock, lst))

    def generate_scan_inp(
        self,
        xyz : str,
        idxs_to_rotate : list[int],
        filename : str,
        submol_charge : int
    ) -> None:
        """
            Generates .inp file with "filename" for scan
            of mol, described by "xyz" xyz block, in orca
            Note that we rotate 0-1-2-3 angle, I think,
            that it should work always
        """
        with open(filename, 'w+') as file:
            file.write("!" + self.method_of_calc + " opt\n")
            file.write("%pal\nnprocs " + str(self.num_of_procs) + "\nend\n")
            file.write("%geom Scan\n")
            file.write("D " + " ".join(list(map(str, idxs_to_rotate))) + " = 0.0, 360.0, 37\n")
            file.write("end\nend\n")
            file.write("* xyz " + str(submol_charge) + " " + str(self.multipl) + "\n")
            file.write(xyz)
            file.write("END\n")

    def get_coords_from_xyz_block(self,
                                  xyz : str) -> str:
        """
            returns xyz-coords from xyz block
            erase first info lines
        """

        return "\n".join(xyz.split("\n")[2:])

    def generate_scan_inps_from_mol(self) -> list[str]:
        """
            Generates inp files for scanning of all interesting
            unique fragments from molecule.
            Returns list of .inp filenames
            dir_for_inps - path of directory including folders separator
        """

        inp_names = []

        angle_number = 0

        for sub_mol in self.get_interesting_frags():

            cur_mol = sub_mol
            cur_mol_smiles = Chem.MolToSmiles(Chem.RemoveHs(cur_mol))

            db_response = self.db_connector.get_request_params(
                "SELECT a1, a2, a3, b1, b2, b3, c FROM dihedrals "
                "WHERE dihedral_smiles = ? AND method = ?",
                (cur_mol_smiles, self.method_of_calc.lower()),
            )
            
            if len(db_response) > 0:
                self.fetched_coefs[cur_mol_smiles] = db_response[0]

            idxs_to_rotate = self.get_idxs_to_rotate(cur_mol)
            heavy_mol = Chem.RemoveAllHs(cur_mol)
            central_bond = heavy_mol.GetBondBetweenAtoms(idxs_to_rotate[1], idxs_to_rotate[2])
            if not central_bond.IsInRing():
                SetDihedralRad(cur_mol.GetConformer(), *idxs_to_rotate, 0)
            elif cur_mol_smiles not in self.fetched_coefs:
                # A ring dihedral cannot be freely scanned (the ring breaks);
                # use a flat GP mean prior so the IK loss alone guides ring geometry.
                self.fetched_coefs[cur_mol_smiles] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

            xyz = Chem.MolToXYZBlock(cur_mol)
            filename = self.dir_for_inps + "scan_" + str(angle_number) + ".inp"
            self.generate_scan_inp(
                xyz=self.get_coords_from_xyz_block(xyz), 
                idxs_to_rotate=idxs_to_rotate, 
                filename=filename,
                submol_charge=Chem.GetFormalCharge(cur_mol)
            )
            inp_names.append(filename)
            angle_number += 1
            self.scanfile2smiles[filename] = cur_mol_smiles
    
        return inp_names

    def get_energies_from_scans(self,
                                lst : list[str]) -> list[tuple[str, list[float]]]:
        """
            lst - list of input file paths,
            return list of lists of energies in
            [0.0, 360.0] with step = 10 degrees
        """
        result = []

        for inp_name in lst:
            res_file_name = inp_name[:-3] + "relaxscanact.dat"

            if self.scanfile2smiles[inp_name] in self.fetched_coefs:
                result.append(None)
                continue

            cur_res = []
            with open(res_file_name, "r") as file:
                for line in file:
                    cur_res.append(float(line.strip().split()[1]))
            result.append(np.array(cur_res))

        return list(zip(lst, result))

    def get_scans_of_dihedrals(self) -> np.ndarray:
        """
            Returns list of energie dependecies.
            Independent torsion-scan fragments are submitted to SLURM
            concurrently instead of one-by-one, so wall-clock time scales
            with the slowest single scan rather than the sum of all scans.
        """
        from calc import submit_calc, wait_for_jobs

        inp_files = self.generate_scan_inps_from_mol()

        to_run = [
            cur for cur in inp_files
            if self.scanfile2smiles[cur] not in self.fetched_coefs
        ]

        if to_run:
            job_ids = [submit_calc(cur, scan=True) for cur in to_run]
            logger.info("Submitted %d independent torsion scans concurrently: %s", len(job_ids), job_ids)
            cfg = config_manager.get_config()
            wait_for_jobs(job_ids, timeout_minutes=cfg.orca_poll_timeout_minutes)

        return self.get_energies_from_scans(inp_files)

    def calc(self) -> list[list[float]]:
        """
            Calculate coefs for mean function
        """
        res = []
        inp_filenames = []
        for inp_filename, energies in self.get_scans_of_dihedrals():
            inp_filenames.append(inp_filename)
            if self.scanfile2smiles[inp_filename] in self.fetched_coefs:
                res.append(self.fetched_coefs[self.scanfile2smiles[inp_filename]])
                continue
            res.append(calc_coefs(self.degrees, energies))
        
        self.n_fourier_computed = len(inp_filenames) - len(self.fetched_coefs)
        self.n_fourier_cached = len(self.fetched_coefs)
        logger.info("Sucessful calculated %s coefs and fetched from db %s coefs!", self.n_fourier_computed, self.n_fourier_cached)
                
        for inp_filename, coefs in zip(inp_filenames, res):
            if self.scanfile2smiles[inp_filename] in self.fetched_coefs:
                continue
            self.db_connector.set_request_params(
                "INSERT INTO dihedrals (dihedral_smiles, method, a1, a2, a3, b1, b2, b3, c) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.scanfile2smiles[inp_filename], self.method_of_calc.lower(),
                    coefs[0], coefs[1], coefs[2], coefs[3], coefs[4], coefs[5], coefs[6],
                ),
            )

        return res

    def coef_matrix(self) -> list[tuple[tuple, list[float]]]:
        """
            Get matrix of coefficients for mean function 
            for all dihedral angels 
        """  
        unique_coefs = self.calc()
        result = []
        # Prefer window with Fourier coefficients over a flat (ring-fusion) one.
        seen_axes = {}
        for idxs in self.frags:
            axis = frozenset((idxs[1], idxs[2]))
            coef_idx = self.frags[idxs]
            has_real_coefs = coef_idx < len(unique_coefs)
            if axis in seen_axes:
                prev_idxs, prev_has_real = seen_axes[axis]
                if has_real_coefs and not prev_has_real:
                    # Replace the flat placeholder with the scanned one.
                    seen_axes[axis] = (idxs, True)
                    logger.info(
                        "coef_matrix: preferring scanned window %s over flat "
                        "ring-fusion window %s for central bond %s.",
                        idxs, prev_idxs, tuple(axis),
                    )
                else:
                    logger.info(
                        "coef_matrix: dropping duplicate window %s for central "
                        "bond %s (already keeping %s).",
                        idxs, tuple(axis), prev_idxs,
                    )
                continue
            seen_axes[axis] = (idxs, has_real_coefs)

        for axis, (idxs, has_real) in seen_axes.items():
            coef_idx = self.frags[idxs]
            if not has_real:
                logger.debug("No Fourier coefs for ring-fusion axis %s — using flat mean function.", idxs)
                result.append((list(idxs), (0.0,) * 7))
            else:
                result.append((list(idxs), unique_coefs[coef_idx]))
        return result


def detect_double_bonds(mol: Chem.rdchem.Mol) -> list[tuple[int, int]]:
    """Canonical (heavy-only, 0-idx) atoms pairs of double bonds."""
    return sorted(
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for b in mol.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
    )


def log_and_combine_double_bonds(mol, user_specified: list[tuple[int, int]]) -> set:
    auto = detect_double_bonds(mol)
    logger.info("Auto detected double bonds (canonical 0-idx): %s", auto)
    if user_specified:
        logger.info("User set double bonds (canonical 0-idx): %s", user_specified)
        for a, b in user_specified:
            bond = mol.GetBondBetweenAtoms(a, b)
            if bond is None:
                logger.warning("fixed_double_bonds (%d,%d): no bonds detected between specified atoms.", a, b)
            elif bond.GetBondType() != Chem.BondType.DOUBLE:
                logger.warning("fixed_double_bonds (%d,%d): bond detected, but not double (type=%s).", a, b, bond.GetBondType())
    combined = {frozenset(p) for p in user_specified} | {frozenset(p) for p in auto}
    logger.info("Resulting set of double bonds (%d total): %s", len(combined), sorted(tuple(sorted(p)) for p in combined))
    return combined

def generate_bond_scan_inp(
    xyz: str,
    atom_a: int, atom_b: int,
    filename: str,
    num_of_procs: int,
    method_of_calc: str,
    charge: int, multipl: int,
    lo: float, hi: float, nsteps: int = 20,
) -> None:
    """Relaxed scan of the bond length atom_a-atom_b from lo to hi."""
    with open(filename, 'w+') as file:
        file.write("!" + method_of_calc + " opt\n")
        file.write("%pal\nnprocs " + str(num_of_procs) + "\nend\n")
        file.write("%geom Scan\n")
        file.write(f"B {atom_a} {atom_b} = {lo}, {hi}, {nsteps}\n")
        file.write("end\nend\n")
        file.write("* xyz " + str(charge) + " " + str(multipl) + "\n")
        file.write(xyz)
        file.write("END\n")

def parse_bond_scan_results(inp_name: str) -> tuple[np.ndarray, np.ndarray]:
    res_file_name = inp_name[:-4] + ".relaxscanact.dat"
    lengths, energies = [], []
    with open(res_file_name, "r") as file:
        for line in file:
            parts = line.strip().split()
            lengths.append(float(parts[0]))
            energies.append(float(parts[1]))
    return np.array(lengths), np.array(energies)