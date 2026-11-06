import numpy as np
from scipy.optimize import curve_fit

import tensorflow as tf

from calc import HARTREE_TO_KCAL

# ============================================================================
# NEW FOURIER MODEL (2026-11-06)
# Fixed integer harmonics guarantee exact 2π periodicity. The cache namespace
# is versioned in coef_calc.py so legacy seven-parameter rows are never reused.
# ============================================================================
def pes(x: np.ndarray, A1: float, A2: float, A3: float, C: float) -> np.ndarray:
    """Strictly 2π-periodic three-harmonic cosine torsional profile."""
    theta = np.asarray(x, dtype=float).reshape(-1)
    return A1 * np.cos(theta) + A2 * np.cos(2.0 * theta) + A3 * np.cos(3.0 * theta) + C


def pes_tf(x: tf.Tensor, A1: float, A2: float, A3: float, C: float) -> tf.Tensor:
    """TensorFlow implementation of the strictly periodic torsional profile."""
    return A1 * tf.cos(x) + A2 * tf.cos(2.0 * x) + A3 * tf.cos(3.0 * x) + C


def pes_tf_grad(x: tf.Tensor, A1: float, A2: float, A3: float, C: float) -> tf.Tensor:
    """Derivative with respect to the dihedral angle."""
    return -(A1 * tf.sin(x) + 2.0 * A2 * tf.sin(2.0 * x) + 3.0 * A3 * tf.sin(3.0 * x))


def calc_coefs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit [A1, A2, A3, C] by linear least squares; y is in Hartree."""
    theta = np.asarray(x, dtype=float).reshape(-1)
    energy = (np.asarray(y, dtype=float).reshape(-1) - np.mean(y)) * HARTREE_TO_KCAL
    if theta.size != energy.size or theta.size < 4:
        raise ValueError(f"Fourier fit requires at least four paired points; got {theta.size} angles and {energy.size} energies")
    design = np.column_stack((np.cos(theta), np.cos(2.0 * theta), np.cos(3.0 * theta), np.ones_like(theta)))
    coefs, _, rank, _ = np.linalg.lstsq(design, energy, rcond=None)
    if rank < 4 or not np.all(np.isfinite(coefs)):
        raise ValueError(f"Periodic Fourier design matrix is rank deficient (rank={rank})")
    return coefs

def morse(
    r: np.ndarray,
    De: float, a: float, re: float, c: float,
) -> np.ndarray:
    """
        The Morse potential for communication:
        De is the depth of the pit (dissociation energy), a - "rigidity" of the bond,
        re is the equilibrium length, c is the energy shift (baseline).
    """
    return De * (1 - np.exp(-a * (r - re))) ** 2 + c

def morse_tf(
    r: tf.Tensor,
    De: float, a: float, re: float, c: float,
) -> tf.Tensor:
    return De * (1 - tf.exp(-a * (r - re))) ** 2 + c

def morse_grad_tf(
    r: tf.Tensor,
    De: float, a: float, re: float, c: float,
) -> tf.Tensor:
    return 2 * De * a * tf.exp(-a * (r - re)) * (1 - tf.exp(-a * (r - re)))

def calc_bond_coefs(
    r: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
        r - bond length in scan [N]
        y - energy in Hartree [N]
        returns [4] set of morze coefficients: De, a, re, c
    """
    y = (y - y.min()) * HARTREE_TO_KCAL
    # Начальные приближения: De ~ размах кривой, re ~ точка минимума
    p0 = [max(y.max() - y.min(), 1.0), 2.0, r[np.argmin(y)], 0.0]
    coefs, _ = curve_fit(morse, r, y, p0=p0, maxfev=10000)
    return coefs