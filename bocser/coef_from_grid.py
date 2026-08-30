import numpy as np
from scipy.optimize import curve_fit

import tensorflow as tf

from calc import HARTRI_TO_KCAL

def pes(
    x : np.ndarray, 
    A11 : float, A21 : float, A31 : float,
    b11 : float, b21 : float, b31 : float,
    c1 : float
) -> np.ndarray:
    """
        Function, that could describe PES
    """
    return A11 * np.cos(b11 * x[:, 0]) + A21 * np.cos(2 * b21 * x[:, 0]) + A31 * np.cos(3 * b31 * x[:, 0]) + c1

def pes_tf(
    x : tf.Tensor,
    A11 : float, A21 : float, A31 : float,
    b11 : float, b21 : float, b31 : float,
    c1 : float
) -> tf.Tensor:
    return A11 * tf.cos(b11 * x) + A21 * tf.cos(2 * b21 * x) + A31 * tf.cos(3 * b31 * x) + c1

def pes_tf_grad(
    x : tf.Tensor,
    A11 : float, A21 : float, A31 : float,
    b11 : float, b21 : float, b31 : float,
    c1 : float
) -> tf.Tensor:
    return -(A11 * b11 * tf.sin(b11 * x) + A21 * 2 * b21 * tf.sin(2 * b21 * x) + A31 * 3 * b31 * tf.sin(3 * b31 * x))

def calc_coefs(
    x : np.ndarray,
    y : np.ndarray,
) -> np.ndarray:
    """
        x - observed points [N, inp_dims]
        y - observed signal [N]
        returns [7, inp_dims] array of coefs
    """ 
    y = (y - y.mean()) * HARTRI_TO_KCAL
    coefs, cov_matrix = curve_fit(pes, x, y, p0=np.ones(7), maxfev=10000)
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
    y = (y - y.min()) * HARTRI_TO_KCAL
    # Начальные приближения: De ~ размах кривой, re ~ точка минимума
    p0 = [max(y.max() - y.min(), 1.0), 2.0, r[np.argmin(y)], 0.0]
    coefs, _ = curve_fit(morse, r, y, p0=p0, maxfev=10000)
    return coefs