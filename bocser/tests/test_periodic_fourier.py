"""Regression tests for the fixed-harmonic periodic torsion profile."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from coef_from_grid import calc_coefs, pes


def test_fourier_profile_is_exactly_two_pi_periodic():
    theta = np.linspace(0.0, 2.0 * np.pi, 37).reshape(-1, 1)
    coefs = (1.7, -0.45, 0.2, 0.0)
    values = pes(theta, *coefs)
    shifted = pes(theta + 2.0 * np.pi, *coefs)
    np.testing.assert_allclose(values, shifted, rtol=0.0, atol=1.0e-12)


def test_fourier_fit_recovers_three_fixed_harmonics_after_energy_centering():
    theta = np.linspace(0.0, 2.0 * np.pi, 37).reshape(-1, 1)
    expected = np.array([1.7, -0.45, 0.2])
    energy_shape = (
        expected[0] * np.cos(theta[:, 0])
        + expected[1] * np.cos(2.0 * theta[:, 0])
        + expected[2] * np.cos(3.0 * theta[:, 0])
    )
    # calc_coefs expects Hartree and intentionally removes the mean before fitting.
    energies_hartree = energy_shape / 627.509474063 + 12.0
    fitted = calc_coefs(theta, energies_hartree)
    np.testing.assert_allclose(fitted[:3], expected, rtol=1.0e-10, atol=1.0e-10)
    assert fitted.shape == (4,)
    assert abs(fitted[3]) < 1.0e-10
