import tensorflow as tf
import numpy as np
from rdkit import Chem
import ringo

import math
import tempfile
import typing

from calc import dihedral_angle

T = typing.TypeVar('T')

class CyclicCollection(typing.Generic[T]):
    """This behaves like a list but enforces cyclicity
    when accessing elements by index"""

    def __init__(self, a: list[T]) -> None:
        self.a = a

    def __len__(self) -> int:
        return len(self.a)

    def __getitem__(self, idx: int) -> T:
        if not self.a:
            raise IndexError("CyclicCollection is empty")
        return self.a[idx % len(self.a)]

xx = lambda t: tuple(i + 1 for i in t)

class IKLoss:
    def __init__(
        self,
        bond_lengths: list[list[float]],
        valence_angles: list[list[float]],
        dihedral_angles: list[list[float]],
        bond_lengths_dicts,
        valence_angles_dicts,
        dihedral_angles_dicts
    ):
        self.bond_lengths = bond_lengths_dicts
        self.valence_angles = valence_angles_dicts
        self.dihedral_angles = dihedral_angles_dicts

        self.D_matrices = [
            tf.stack([self.D_matrix(d) for d in cycle_lengths])
            for cycle_lengths in bond_lengths
        ]

        self.V_matrices = [
            tf.stack([self.V_matrix(a) for a in cycle_angles])
            for cycle_angles in valence_angles
        ]
        
        self.T_matrices = [
            tf.stack([self.T_matrix(da) for da in cycle_dihedrals])
            for cycle_dihedrals in dihedral_angles
        ]

    @classmethod
    def from_rdkit(cls, mol, all_ring_atoms):
        mol = Chem.AddHs(mol)
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
            Chem.MolToXYZFile(mol, tmp.name)
            p = ringo.Confpool()
            p.include_from_file(tmp.name)
        conf = p[0]

        bond_lengths = []
        valence_angles = []
        dihedral_angles = []
        
        bond_lengths_dicts = []
        valence_angles_dicts = []
        dihedral_angles_dicts = []

        for ring_atoms_list in all_ring_atoms:
            ring_atoms = CyclicCollection(ring_atoms_list)

            bl = [
                conf.l(*xx(ring_atoms[i] for i in (k, k + 1)))
                for k in range(len(ring_atoms))
            ]
            va = [
                np.deg2rad(conf.v(*xx(ring_atoms[i] for i in (k - 1, k, k + 1))))
                for k in range(len(ring_atoms))
            ]
            da = [
                dihedral_angle(*(mol.GetConformer().GetAtomPosition(ring_atoms[i]) 
                for i in (k - 1, k, k + 1, k + 2)))
                for k in range(len(ring_atoms))
            ]

            bond_lengths.append(bl)
            valence_angles.append(va)
            dihedral_angles.append(da)

            bond_lengths_dicts.append({
                tuple(ring_atoms[k] for k in (i, i + 1)): v
                for i, v in enumerate(bl)
            })
            valence_angles_dicts.append({
                tuple(ring_atoms[k] for k in (i - 1, i, i + 1)): v
                for i, v in enumerate(va)
            })
            dihedral_angles_dicts.append({
                tuple(ring_atoms[k] for k in (i - 1, i, i + 1, i + 2)): v
                for i, v in enumerate(da)
            })

        return cls(bond_lengths, valence_angles, dihedral_angles, bond_lengths_dicts, valence_angles_dicts, dihedral_angles_dicts)

    @staticmethod
    def D_matrix(d):
        return tf.constant(
            [
                [1.0, 0.0, 0.0, d],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=tf.float64,
        )

    @staticmethod
    def V_matrix(a):
        return tf.constant(
            [
                [-math.cos(a), math.sin(a), 0.0, 0.0],
                [-math.sin(a), -math.cos(a), 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=tf.float64,
        )

    @staticmethod
    def T_matrix(a):
        a = tf.convert_to_tensor(a, dtype=tf.float64)
        a = a.reshape(-1, 1)
        
        batch_size = tf.shape(a)[0]
        
        cos_a = tf.cos(a)
        sin_a = tf.sin(a)
        zeros = tf.zeros_like(a)
        ones = tf.ones_like(a)
        
        row1 = tf.stack([ones, zeros, zeros, zeros], axis=1)
        row2 = tf.stack([zeros, cos_a, -sin_a, zeros], axis=1)
        row3 = tf.stack([zeros, sin_a, cos_a, zeros], axis=1)
        row4 = tf.stack([zeros, zeros, zeros, ones], axis=1)
        
        return tf.squeeze(tf.stack([row1, row2, row3, row4], axis=1))

    def build_T_matrices(self, angles_list):
        result = []
        
        for i, angles in enumerate(angles_list):
            n, l = tf.shape(angles)[0], tf.shape(angles)[1]
            
            result_columns = []
            
            for j in range(l):
                column_angles = angles[:, j]
                column_has_nan = tf.math.is_nan(column_angles)
                
                if tf.reduce_all(column_has_nan):
                    T_col = tf.expand_dims(self.T_matrices[i][j], 0)
                    T_col = tf.tile(T_col, [n, 1, 1])
                else:
                    T_col = self.T_matrix(column_angles)
                
                result_columns.append(tf.expand_dims(T_col, 1))
            
            result.append(tf.concat(result_columns, axis=1))
            
        return result
    
    @staticmethod
    def compute_total_motions(V, D, T):
        V = V[None, ...]
        D = D[None, ...]

        S = V @ D @ T
        S = tf.transpose(S, [1, 0, 2, 3])
        
        def step(prev, cur):
            return prev @ cur

        total = tf.scan(fn=step, elems=S)

        return total[-1]
    
    def __call__(self, dihedrals):
        T_matrices = self.build_T_matrices(dihedrals)
        
        total_loss = 0
        for i in range(len(dihedrals)):            
            total_motions = self.compute_total_motions(
                self.V_matrices[i], self.D_matrices[i], T_matrices[i]
            )

            ik_loss = tf.map_fn(
                fn=lambda total_motion: tf.reduce_sum(
                    tf.map_fn(
                        fn=lambda x: x**2,
                        elems=tf.stack([
                            total_motion[0, 3],
                            total_motion[1, 3],
                            total_motion[2, 3],
                            total_motion[0, 0] + total_motion[1, 1] + total_motion[2, 2] - 3.0,
                            total_motion[0, 1],
                            total_motion[0, 2],
                            total_motion[1, 2],
                        ]),
                    ),
                    keepdims=True,
                ),
                elems=total_motions,
            )

            total_loss = total_loss + ik_loss

        return total_loss
