import numpy as np
import tensorflow as tf

from typing import List, Optional, Tuple, Callable#, Self

from calc import dihedral_angle
import logging
logger = logging.getLogger(__name__)

def parse_energy_from_xyz_2nd_line(s):
    return float(s.split()[-1]) #*HARTRI_TO_KCAL

class EnsembleProcessor:
    def __init__(
        self,
        ensemble_filename : str,
        dihedral_idxs : List[List[int]],
        parse_energy_from_xyz_2nd_line : Callable[[str], float] = parse_energy_from_xyz_2nd_line,
        ref_symbols : Optional[List[str]] = None,
        raw_order : Optional[List[int]] = None,
        ts_bonds : Optional[List[Tuple[int, int]]] = None,
        ts_bounds : Optional[Tuple[float, float]] = None,
    ) -> None:
        """
            Processes .xyz ensemble to list of torsion angle values
            and their energies in kcal/mol.

            Args:
                ensemble_filename - filename of the ensemble
                dihedral_idxs - list of 4-element lists with atom idxs.
                    angles given in expected order
                parse_energy_from_xyz_2nd_line - function that will apply
                    to the 2nd line of .xyz file and return Energy in kcal/mol
                ref_symbols - element symbols in the canonical (heavy atoms first) order
                    used by dihedral_idxs; enables the atom-order check
                raw_order - canonical index -> index in the original .mol file; a block in
                    the original .mol order is reordered to the canonical one
                ts_bonds, ts_bounds - when given, TS-bond lengths (clipped to ts_bounds)
                    are appended after the dihedrals, as in the BO search vector
        """
        self.processed_ens = []
        self.energies: list[float] = []
        xyz_blocks = []

        try:
            with open(ensemble_filename, 'r') as file:
                lines = [l.rstrip('\n') for l in file]

            i = 0
            nlines = len(lines)
            while i < nlines:
                if not lines[i].strip():
                    # blank separator / trailing lines
                    i += 1
                    continue
                n_atoms = int(lines[i])
                block = lines[i : i + 2 + n_atoms]
                xyz_blocks.append("\n".join(block))
                i += 2 + n_atoms

        except FileNotFoundError:
            logger.error("Error! No such file: %s; Finishing with empty ensemble!", ensemble_filename)
            # Keep object in a valid state with empty data so callers can continue.
            self.processed_ens = []
            self.energies = []
            return

        for block_idx, xyz_block in enumerate(xyz_blocks):
            comment = xyz_block.split('\n')[1]
            try:
                self.energies.append(parse_energy_from_xyz_2nd_line(comment))
            except (ValueError, IndexError) as e:
                raise ValueError(
                    f"{ensemble_filename}: conformer #{block_idx} — cannot read energy from "
                    f"comment line {comment!r} ({e})"
                ) from e

        reordered = False
        for block_idx, raw_xyz in enumerate(xyz_blocks):
            atom_lines = [cur.split() for cur in raw_xyz.split('\n')[2:] if cur.strip()]
            symbols = [parts[0] for parts in atom_lines]
            coords = [list(map(float, parts[1:4])) for parts in atom_lines]

            if ref_symbols is not None:
                # dihedral_idxs are canonical (heavy atoms first) indices: check the file uses that order
                if symbols == list(ref_symbols) or symbols == list(ref_symbols[:len(symbols)]):
                    pass
                elif raw_order is not None and symbols == [ref_symbols[raw_order.index(k)] for k in range(len(raw_order))]:
                    coords = [coords[raw_idx] for raw_idx in raw_order]
                    reordered = True
                else:
                    raise ValueError(
                        f"{ensemble_filename}: conformer #{block_idx} atom order matches neither the "
                        f"canonical (heavy atoms first) nor the original .mol order."
                    )

            dihedrals = []
            for i, j, k, l in dihedral_idxs:
                dihedrals.append(
                    dihedral_angle(
                        coords[i], coords[j], coords[k], coords[l]
                    )
                )
            for a, b in (ts_bonds or []):
                length = float(np.linalg.norm(np.array(coords[a]) - np.array(coords[b])))
                dihedrals.append(float(np.clip(length, *ts_bounds)))
            self.processed_ens.append(dihedrals)

        if reordered:
            logger.warning(
                "%s is in the original .mol atom order — reordered to the canonical "
                "(heavy atoms first) order before computing dihedrals.", ensemble_filename,
            )

    def normalize_energy(
        self,
        norm_en : float
        ):# -> Self:
        """
            Substracts `norm_en` from all energies in ensemble
            Args:
                norm_en - normalization energy
            Returns:
                self object

        """

        self.energies = [cur - norm_en for cur in self.energies]
        return self
    

    def get_data(
        self    
    ) -> Tuple[List[List[float]], List[float]]:
        """
            Returns parsed data

            Return:
                tuple of list of dihedral values and list of energies 
        """
        return self.processed_ens, self.energies

    def get_tf_data(
        self,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
            Returns parsed data in tf float64 tensors

            Return:
                tuple of tf tensor of torsion values and tf tensor of energies
        """

        return tf.constant(self.processed_ens, dtype=tf.float64), tf.reshape(tf.constant(self.energies, dtype=tf.float64), [-1, 1])
