"""Feasibility model: P(a candidate gives a valid energy | x), kept apart from the energy GP.

A candidate gives no energy when its geometry is rejected by the structure check, ORCA
fails, or the geometry could not be built near the asked point. Such candidates are labelled
0 here and never enter the energy GP; candidates with a valid energy (and the conformers of
load_ensemble) are labelled 1. The acquisition is multiplied by P (FeasibilityWeightedAcquisition).

P is a kernel-weighted Beta posterior (a Nadaraya-Watson classifier with a prior):
    w_i(x) = exp(-0.5 * (sum_torsions (2 sin(dTheta/2) / l)^2 + sum_ts_bonds (dL / l_ts)^2))
    P(x)   = (a + sum_i w_i y_i) / (a + b + sum_i w_i),   a = prior * s,  b = (1 - prior) * s
The prior strength s is small, so a single failure at x already brings P(x) down to ~0.3
(the same candidate is not asked again), while far from every label P stays at the prior.
It is a smooth function of x, so the gradient-based acquisition optimizer can use it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf
from trieste.acquisition.interface import (
    AcquisitionFunction,
    AcquisitionFunctionClass,
    SingleModelAcquisitionBuilder,
)
from trieste.data import Dataset
from trieste.models import ProbabilisticModel

PRIOR_STRENGTH = 0.5
MIN_PROBABILITY = 0.01


class FeasibilityModel:
    """Pass/fail labels of evaluated points and P(feasible | x) over the BO search vector
    (n_dihedrals torsions in radians, then n_ts_bonds TS-bond lengths in A)."""

    def __init__(
        self,
        n_dihedrals: int,
        n_ts_bonds: int = 0,
        lengthscale: float = 0.5,
        ts_lengthscale: float = 0.5,
        prior: float = 0.8,
        prior_strength: float = PRIOR_STRENGTH,
        min_probability: float = MIN_PROBABILITY,
    ) -> None:
        if lengthscale <= 0 or ts_lengthscale <= 0:
            raise ValueError("Feasibility lengthscales must be > 0")
        if not 0.0 < prior < 1.0:
            raise ValueError("Feasibility prior must be between 0 and 1")
        self.n_dihedrals = int(n_dihedrals)
        self.n_dims = self.n_dihedrals + int(n_ts_bonds)
        if self.n_dims < 1:
            raise ValueError("Feasibility model needs at least one search dimension")
        self._lengthscale = float(lengthscale)
        self._ts_lengthscale = float(ts_lengthscale)
        self._a = float(prior) * float(prior_strength)
        self._b = (1.0 - float(prior)) * float(prior_strength)
        self._min_probability = float(min_probability)
        self.points: list = []
        self.labels: list = []
        # Variables with a free first dimension: new labels are seen by an acquisition
        # function that trieste has already traced into a tf.function
        self._x = tf.Variable(np.empty((0, self.n_dims)), shape=tf.TensorShape([None, self.n_dims]),
                              dtype=tf.float64, trainable=False, name="feasibility_x")
        self._y = tf.Variable(np.empty((0,)), shape=tf.TensorShape([None]),
                              dtype=tf.float64, trainable=False, name="feasibility_y")

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def n_failed(self) -> int:
        return int(sum(1 for y in self.labels if not y))

    def add(self, point, feasible: bool) -> bool:
        """Record one label; returns False (nothing recorded) for a point of the wrong size
        or with non-finite coordinates."""
        values = [float(v) for v in np.asarray(point, dtype=float).reshape(-1)]
        if len(values) != self.n_dims or not np.all(np.isfinite(values)):
            return False
        self.points.append(values)
        self.labels.append(1 if feasible else 0)
        self._sync()
        return True

    def set_data(self, points, labels) -> None:
        points = [[float(v) for v in p] for p in points]
        labels = [1 if y else 0 for y in labels]
        if len(points) != len(labels):
            raise ValueError("Feasibility points and labels differ in length")
        if any(len(p) != self.n_dims for p in points):
            raise ValueError(f"Feasibility points must have {self.n_dims} coordinates")
        self.points, self.labels = points, labels
        self._sync()

    def _sync(self) -> None:
        x = np.asarray(self.points, dtype=np.float64).reshape(-1, self.n_dims)
        self._x.assign(x)
        self._y.assign(np.asarray(self.labels, dtype=np.float64).reshape(-1))

    def probability(self, x) -> tf.Tensor:
        """P(feasible | x) for x of shape [N, n_dims]; returns [N, 1]."""
        x = tf.reshape(tf.cast(x, tf.float64), [-1, self.n_dims])
        train_x = self._x.read_value()
        train_y = self._y.read_value()
        delta = x[:, None, :] - train_x[None, :, :]                      # [N, M, d]
        n_dih = self.n_dihedrals
        chord = 2.0 * tf.sin(0.5 * delta[:, :, :n_dih]) / self._lengthscale
        d2 = tf.reduce_sum(tf.square(chord), axis=-1)
        if self.n_dims > n_dih:
            d2 += tf.reduce_sum(tf.square(delta[:, :, n_dih:] / self._ts_lengthscale), axis=-1)
        w = tf.exp(-0.5 * d2)                                             # [N, M]
        success = tf.reduce_sum(w * train_y[None, :], axis=1)
        total = tf.reduce_sum(w, axis=1)
        p = (self._a + success) / (self._a + self._b + total)
        p = tf.clip_by_value(p, self._min_probability, 1.0)
        return tf.reshape(p, [-1, 1])

    def probability_at(self, point) -> float:
        return float(self.probability(np.asarray(point, dtype=np.float64)[None, :]).numpy()[0, 0])


class _FeasibilityWeightedFunction(AcquisitionFunctionClass):
    def __init__(self, base_function, feasibility: FeasibilityModel):
        self._base_function = base_function
        self._feasibility = feasibility

    @property
    def _eta(self):
        # conf_search reads the current best predicted energy from the acquisition function
        return self._base_function._eta

    def __call__(self, x):
        value = self._base_function(x)
        p = self._feasibility.probability(tf.squeeze(x, axis=-2))
        # value * P keeps the scale of a non-negative acquisition (EVM, EI). The "ik"
        # acquisition (IV - IK loss) can be negative: there a lower P must make it more
        # negative, hence value / P. Both branches are 0 at value = 0.
        return tf.where(value >= 0, value * p, value / p)


class FeasibilityWeightedAcquisition(SingleModelAcquisitionBuilder):
    """Acquisition of a base builder multiplied by P(feasible | x)."""

    def __init__(self, base_builder: SingleModelAcquisitionBuilder, feasibility: FeasibilityModel) -> None:
        self._base_builder = base_builder
        self._feasibility = feasibility

    def __repr__(self) -> str:
        return f"FeasibilityWeightedAcquisition({self._base_builder!r})"

    def prepare_acquisition_function(
        self, model: ProbabilisticModel, dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        base = self._base_builder.prepare_acquisition_function(model, dataset)
        return _FeasibilityWeightedFunction(base, self._feasibility)

    def update_acquisition_function(
        self, function: AcquisitionFunction, model: ProbabilisticModel, dataset: Optional[Dataset] = None,
    ) -> AcquisitionFunction:
        if not isinstance(function, _FeasibilityWeightedFunction):
            return self.prepare_acquisition_function(model, dataset)
        function._base_function = self._base_builder.update_acquisition_function(
            function._base_function, model, dataset
        )
        return function
