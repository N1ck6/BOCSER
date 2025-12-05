"""Fixtures and utilities for unit tests."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from default_vals import ConfSearchConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict():
    """Return a sample config dictionary."""
    return {
        "mol_file_name": "test.mol",
        "spin_multiplicity": 1,
        "charge": 0,
        "orca_exec_command": "/opt/orca5/orca",
        "num_of_procs": 4,
        "orca_method": "lda sto-3g",
        "broken_struct_energy": 100.0,
        "bond_length_threshold": 0.7,
        "ts": False,
        "rolling_window_size": 5,
        "rolling_std_threshold": 0.15,
        "rolling_mean_threshold": 1.0,
        "num_initial_points": 3,
        "max_steps": 50,
        "exp_name": "test_search",
        "load_ensemble": None,
        "acquisition_function": "evm",
    }


@pytest.fixture
def sample_config_file(temp_dir, sample_config_dict):
    """Create a temporary config YAML file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def sample_config(sample_config_dict):
    """Return a ConfSearchConfig instance."""
    return ConfSearchConfig(**sample_config_dict)


@pytest.fixture
def sample_xyz_content():
    """Return sample XYZ file content."""
    return """3
Energy: -12.345
C    0.000000    0.000000    0.000000
H    0.629118    0.629118    0.629118
H   -0.629118   -0.629118   -0.629118
"""


@pytest.fixture
def sample_mol_file(temp_dir, sample_xyz_content):
    """Create a temporary MOL-like file."""
    mol_path = temp_dir / "test.mol"
    with open(mol_path, "w") as f:
        f.write(sample_xyz_content)
    return mol_path


@pytest.fixture
def sample_traj_file(temp_dir):
    """Create a temporary trajectory XYZ file with multiple structures."""
    traj_path = temp_dir / "traj.xyz"
    content = """3
Energy: -12.345
C    0.000000    0.000000    0.000000
H    0.629118    0.629118    0.629118
H   -0.629118   -0.629118   -0.629118
3
Energy: -12.350
C    0.000000    0.000000    0.000000
H    0.629118    0.629118    0.629118
H   -0.629118   -0.629118   -0.629118
"""
    with open(traj_path, "w") as f:
        f.write(content)
    return traj_path
