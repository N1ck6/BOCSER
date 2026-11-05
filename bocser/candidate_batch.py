"""Several candidates from one acquisition search (candidate_batch_size).

trieste's default optimizer for a Box runs L-BFGS from NUM_RUNS_DIM * D starting points and
returns the best end point only. BatchRecordingOptimizer runs the same search and also keeps
the other distinct end points, best first. When the first candidate of a BO step is rejected
before ORCA (structure check, MMFF build, MMFF energy), the runner tries the next kept one
instead of a new acquisition search, which is the slow part of a step.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from trieste.acquisition.optimizer import automatic_optimizer_selector
from trieste.space import Box

logger = logging.getLogger(__name__)

try:  # trieste internals; without them the default optimizer runs and no batch is kept
    from trieste.acquisition.optimizer import (
        NUM_RUNS_DIM,
        NUM_SAMPLES_DIM,
        NUM_SAMPLES_MIN,
        _perform_parallel_continuous_optimization,
        generate_initial_points,
        sample_from_space,
    )
    _HAVE_INTERNALS = True
except ImportError:  # pragma: no cover - depends on the installed trieste
    _HAVE_INTERNALS = False


def periodic_distance(a, b, n_dihedrals: int) -> float:
    """Largest difference over the torsions (radians, periodic) and the TS-bond lengths (A)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = np.abs(a[:n_dihedrals] - b[:n_dihedrals]) % (2 * np.pi)
    worst = float(np.max(np.minimum(d, 2 * np.pi - d))) if n_dihedrals else 0.0
    if len(a) > n_dihedrals:
        worst = max(worst, float(np.max(np.abs(a[n_dihedrals:] - b[n_dihedrals:]))))
    return worst


class BatchRecordingOptimizer:
    """AcquisitionOptimizer for EfficientGlobalOptimization that keeps up to `batch_size`
    distinct maxima of the last search in `candidates` (best first; the first is returned)."""

    def __init__(self, batch_size: int, n_dihedrals: int, min_separation: float = 0.35) -> None:
        self.batch_size = int(batch_size)
        self.n_dihedrals = int(n_dihedrals)
        self.min_separation = float(min_separation)  # ~20 deg: closer end points are one maximum
        self.candidates: list = []

    def __call__(self, space, target_func):
        self.candidates = []
        if self.batch_size <= 1 or not _HAVE_INTERNALS or isinstance(target_func, tuple) or not isinstance(space, Box):
            return automatic_optimizer_selector(space, target_func)
        dim = space.dimension
        num_samples = max(int(NUM_SAMPLES_MIN), int(NUM_SAMPLES_DIM) * int(dim))
        num_runs = int(NUM_RUNS_DIM) * int(dim)
        try:
            starts = self._spread_starts(space, target_func, num_samples, num_runs)
            _, fun_values, chosen_x, _ = _perform_parallel_continuous_optimization(target_func, space, starts, {})
        except Exception:
            logger.exception("Batch acquisition search failed — using trieste's default optimizer")
            return automatic_optimizer_selector(space, target_func)
        values = fun_values.numpy()[:, 0]
        points = chosen_x.numpy()[:, 0, :]
        order = np.argsort(-values)
        kept = []
        for i in order:
            if not np.all(np.isfinite(points[i])) or not np.isfinite(values[i]):
                continue
            if all(periodic_distance(points[i], points[j], self.n_dihedrals) > self.min_separation for j in kept):
                kept.append(int(i))
            if len(kept) >= self.batch_size:
                break
        if not kept:
            return automatic_optimizer_selector(space, target_func)
        self.candidates = [points[i].copy() for i in kept]
        logger.info(
            "Acquisition search: %d distinct maxima kept (values %s)",
            len(kept), ", ".join(f"{values[i]:.3g}" for i in kept),
        )
        return tf.constant(points[kept[0]][None, :], dtype=chosen_x.dtype)

    def _spread_starts(self, space, target_func, num_samples: int, num_runs: int) -> tf.Tensor:
        """L-BFGS starting points [num_runs, 1, D]: the best random samples, but at most half
        of them may sit near a better one, so the runs end in different maxima (trieste takes
        the plain top samples, which all lie around the global maximum)."""
        samples = space.sample(num_samples)
        values = np.asarray(target_func(samples[:, None, :])).reshape(-1)
        points = samples.numpy()
        order = [i for i in np.argsort(-values) if np.isfinite(values[i])]
        spread, crowded = [], []
        for i in order:
            if len(spread) >= num_runs:
                break
            if all(periodic_distance(points[i], points[j], self.n_dihedrals) > self.min_separation for j in spread):
                spread.append(i)
            elif len(crowded) < num_runs // 2:
                crowded.append(i)
        chosen = (spread + crowded)[:num_runs]
        if not chosen:
            return generate_initial_points(num_runs, sample_from_space(num_samples), space, target_func, 1)
        return tf.constant(points[chosen][:, None, :], dtype=samples.dtype)

    def pop_next(self, acquisition_function, tried: list) -> Optional[np.ndarray]:
        """Best of the remaining kept maxima under the current acquisition (the feasibility
        weight already holds the rejections), skipping those near an already tried point.
        None when nothing usable is left."""
        remaining = [
            x for x in self.candidates[1:]
            if all(periodic_distance(x, t, self.n_dihedrals) > self.min_separation for t in tried)
        ]
        self.candidates = self.candidates[:1] + remaining
        if not remaining:
            return None
        try:
            values = acquisition_function(tf.constant(np.asarray(remaining)[:, None, :], dtype=tf.float64))
            values = np.asarray(values).reshape(-1)
        except Exception:
            logger.exception("Could not re-score the kept candidates — using them in the original order")
            values = -np.arange(len(remaining), dtype=float)
        best = int(np.argmax(values))
        if not np.isfinite(values[best]):
            self.candidates = self.candidates[:1]
            return None
        chosen = remaining.pop(best)
        self.candidates = self.candidates[:1] + remaining
        logger.info("Next kept candidate of the last acquisition search (value now %.3g)", values[best])
        return chosen
