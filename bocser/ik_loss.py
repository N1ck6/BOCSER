"""Inverse-kinematics loss helpers for ring closure constraints.

This module provides `IKLoss` which computes differentiable losses for
ring closures based on bond lengths, valence angles and dihedral angles.
The implementation focuses on correctness and readability; small API
improvements and type hints were added.
"""

from __future__ import annotations

import math
import tempfile
from typing import Dict, Generic, List, Tuple, TypeVar

import numpy as np
import ringo
import tensorflow as tf
from rdkit import Chem

from calc import dihedral_angle

T = TypeVar("T")


class CyclicCollection(Generic[T]):
    """A list-like view that wraps indices cyclically.

    Accessing an empty collection raises IndexError.
    """

    def __init__(self, a: List[T]) -> None:
        self.a = list(a)

    def __len__(self) -> int:
        return len(self.a)

    def __getitem__(self, idx: int) -> T:
        if not self.a:
            raise IndexError("CyclicCollection is empty")
        return self.a[idx % len(self.a)]


def _one_based(t: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert 0-based indices to 1-based tuple used by ringo.

    This improves readability over the previous inline lambda.
    """
    return tuple(i + 1 for i in t)


class IKLoss:
    """Encapsulate IK loss construction and evaluation.

    The constructor expects precomputed lists of bond lengths, valence
    angles and dihedral angles (in radians for angles). Additionally the
    corresponding dictionaries (mapping index tuples to values) are
    stored for use by external code.
    """

    def __init__(
        self,
        bond_lengths: List[List[float]],
        valence_angles: List[List[float]],
        dihedral_angles: List[List[float]],
        bond_lengths_dicts: List[Dict[Tuple[int, int], float]],
        valence_angles_dicts: List[Dict[Tuple[int, int, int], float]],
        dihedral_angles_dicts: List[Dict[Tuple[int, int, int, int], float]],
    ) -> None:
        self.bond_lengths = bond_lengths_dicts
        self.valence_angles = valence_angles_dicts
        self.dihedral_angles = dihedral_angles_dicts

        # Store TF constants / tensors for matrix operations
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
    def from_rdkit(cls, mol: Chem.Mol, all_ring_atoms: List[List[int]]) -> "IKLoss":
        """Build IKLoss from an RDKit molecule and ring atom index lists.

        The helper uses `ringo` to extract reference geometry. It returns a
        complete IKLoss instance with tensors prepared for TF computations.
        """
        mol = Chem.AddHs(mol)
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=True) as tmp:
            Chem.MolToXYZFile(mol, tmp.name)
            p = ringo.Confpool()
            p.include_from_file(tmp.name)
        conf = p[0]

        bond_lengths: List[List[float]] = []
        valence_angles: List[List[float]] = []
        dihedral_angles: List[List[float]] = []

        bond_lengths_dicts = []
        valence_angles_dicts = []
        dihedral_angles_dicts = []

        for ring_atoms_list in all_ring_atoms:
            ring_atoms = CyclicCollection(ring_atoms_list)

            bl = [
                conf.l(*_one_based(tuple(ring_atoms[i] for i in (k, k + 1))))
                for k in range(len(ring_atoms))
            ]
            va = [
                np.deg2rad(
                    conf.v(*_one_based(tuple(ring_atoms[i] for i in (k - 1, k, k + 1))))
                )
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

        return cls(
            bond_lengths,
            valence_angles,
            dihedral_angles,
            bond_lengths_dicts,
            valence_angles_dicts,
            dihedral_angles_dicts,
        )

    @staticmethod
    def D_matrix(d: float) -> tf.Tensor:
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
    def V_matrix(a: float) -> tf.Tensor:
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
    def T_matrix(a: tf.Tensor) -> tf.Tensor:
        """Return a batch of 4x4 rotation-translation matrices for angles a.

        Input `a` may be a scalar, 1D tensor or batched tensor. The returned
        shape will be `[..., 4, 4]` where `...` corresponds to the batch
        dimension(s) implied by `a`.
        """
        a = tf.convert_to_tensor(a, dtype=tf.float64)
        a = tf.reshape(a, (-1, 1))

        cos_a = tf.cos(a)
        sin_a = tf.sin(a)
        zeros = tf.zeros_like(a)
        ones = tf.ones_like(a)

        row1 = tf.stack([ones, zeros, zeros, zeros], axis=1)
        row2 = tf.stack([zeros, cos_a, -sin_a, zeros], axis=1)
        row3 = tf.stack([zeros, sin_a, cos_a, zeros], axis=1)
        row4 = tf.stack([zeros, zeros, zeros, ones], axis=1)

        # Stack rows to form matrices, then squeeze batch axis if needed
        mats = tf.stack([row1, row2, row3, row4], axis=1)
        return tf.squeeze(mats)

    def build_T_matrices(self, angles_list: List[tf.Tensor]) -> List[tf.Tensor]:
        """Build T matrices for the provided dihedral-angle batches.

        Each element in `angles_list` is expected to be a 2D tensor shape
        `[n, l]` where `n` is batch size and `l` is the number of dihedrals
        in that ring. If an entire column is NaN, the reference T matrix is
        tiled for that column.
        """
        result: List[tf.Tensor] = []

        for i, angles in enumerate(angles_list):
            n = tf.shape(angles)[0]
            l = tf.shape(angles)[1]

            result_columns: List[tf.Tensor] = []
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
    def compute_total_motions(V: tf.Tensor, D: tf.Tensor, T: tf.Tensor) -> tf.Tensor:
        V = V[None, ...]
        D = D[None, ...]

        S = V @ D @ T
        S = tf.transpose(S, [1, 0, 2, 3])

        def step(prev, cur):
            return prev @ cur

        total = tf.scan(fn=step, elems=S)

        return total[-1]

    def __call__(self, dihedrals: List[tf.Tensor]) -> tf.Tensor:
        T_matrices = self.build_T_matrices(dihedrals)

        total_loss = 0
        for i in range(len(dihedrals)):
            total_motions = self.compute_total_motions(
                self.V_matrices[i], self.D_matrices[i], T_matrices[i]
            )

            ik_loss = tf.map_fn(
                fn=lambda total_motion: tf.reduce_sum(
                    tf.map_fn(
                        fn=lambda x: x ** 2,
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
