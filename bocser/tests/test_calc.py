"""Unit tests for calc module utility functions."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from calc import (
    dihedral_angle,
    to_degrees,
    mol_to_inp_name,
    inp_to_out_name,
    check_is_broken,
    increase_structure_id,
    set_config,
)
from default_vals import ConfSearchConfig


class TestDihedralAngle:
    """Tests for dihedral angle calculation."""

    def test_dihedral_angle_linear(self):
        """Test dihedral angle for linear atoms (180 degrees)."""
        # Four collinear points along z-axis
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 1.0]
        c = [0.0, 0.0, 2.0]
        d = [0.0, 0.0, 3.0]

        angle = dihedral_angle(a, b, c, d)
        # Should be close to 180 degrees (π radians) or 0, depending on convention
        assert isinstance(angle, (float, np.floating))
        assert 0 <= angle < 2 * np.pi

    def test_dihedral_angle_tetrahedron(self):
        """Test dihedral angle for typical tetrahedral geometry."""
        # Approximate tetrahedral geometry (methane-like)
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        c = [1.0, 1.0, 0.0]
        d = [1.0, 1.0, 1.0]

        angle = dihedral_angle(a, b, c, d)
        assert isinstance(angle, (float, np.floating))
        assert 0 <= angle < 2 * np.pi

    def test_dihedral_angle_numpy_arrays(self):
        """Test that function works with numpy arrays."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([1.0, 1.0, 0.0])
        d = np.array([1.0, 1.0, 1.0])

        angle = dihedral_angle(a, b, c, d)
        assert isinstance(angle, (float, np.floating))


class TestToDegreesConversion:
    """Tests for radian to degree conversion."""

    def test_to_degrees_empty_list(self):
        """Test conversion of empty list."""
        result = to_degrees([])
        assert result == []

    def test_to_degrees_single_dihedral(self):
        """Test conversion of single dihedral."""
        dihedrals = [([0, 1, 2, 3], np.pi)]  # π radians = 180 degrees
        result = to_degrees(dihedrals)

        assert len(result) == 1
        atoms, degrees = result[0]
        assert atoms == [0, 1, 2, 3]
        assert abs(degrees - 180.0) < 1e-6

    def test_to_degrees_multiple_dihedrals(self):
        """Test conversion of multiple dihedrals."""
        dihedrals = [
            ([0, 1, 2, 3], 0.0),
            ([1, 2, 3, 4], np.pi / 2),  # 90 degrees
            ([2, 3, 4, 5], np.pi),  # 180 degrees
        ]
        result = to_degrees(dihedrals)

        assert len(result) == 3
        assert abs(result[0][1] - 0.0) < 1e-6
        assert abs(result[1][1] - 90.0) < 1e-6
        assert abs(result[2][1] - 180.0) < 1e-6


class TestFileNameConversions:
    """Tests for file name conversion utilities."""

    def test_mol_to_inp_name(self):
        """Test MOL to INP conversion."""
        mol_name = "/path/to/molecule.mol"
        inp_name = mol_to_inp_name(mol_name)
        assert inp_name == "/path/to/molecule.inp"

    def test_mol_to_inp_name_no_extension(self):
        """Test conversion when file has no extension."""
        mol_name = "molecule"
        inp_name = mol_to_inp_name(mol_name)
        assert inp_name == ".inp"

    def test_inp_to_out_name(self):
        """Test INP to OUT conversion."""
        inp_name = "/path/to/molecule.inp"
        out_name = inp_to_out_name(inp_name)
        assert out_name == "/path/to/molecule.out"

    def test_mol_to_inp_to_out_chain(self):
        """Test chaining MOL -> INP -> OUT conversions."""
        mol_name = "test.mol"
        inp_name = mol_to_inp_name(mol_name)
        out_name = inp_to_out_name(inp_name)
        assert out_name == "test.out"


class TestCheckIsBroken:
    """Tests for broken geometry detection."""

    def test_check_is_broken_valid_geometry(self):
        """Test with valid (non-broken) geometry."""
        xyz_block = """
C    0.0    0.0    0.0
H    1.0    0.0    0.0
H    0.0    1.0    0.0
H    0.0    0.0    1.0
"""
        # All atoms are at least 1.0 Angstrom apart, threshold is 0.7
        is_broken = check_is_broken(xyz_block, len_threshold=0.7)
        assert is_broken is False

    def test_check_is_broken_close_atoms(self):
        """Test with atoms too close together."""
        xyz_block = """
C    0.0    0.0    0.0
H    0.5    0.0    0.0
"""
        # H is only 0.5 Angstrom from C, threshold is 0.7
        is_broken = check_is_broken(xyz_block, len_threshold=0.7)
        assert is_broken is True

    def test_check_is_broken_empty_block(self):
        """Test with empty XYZ block."""
        xyz_block = ""
        # Should handle gracefully
        with pytest.raises((ValueError, IndexError)):
            check_is_broken(xyz_block)

    def test_check_is_broken_single_atom(self):
        """Test with single atom (no pairs to check)."""
        xyz_block = """
C    0.0    0.0    0.0
"""
        is_broken = check_is_broken(xyz_block, len_threshold=0.7)
        assert is_broken is False


class TestStructureId:
    """Tests for structure ID tracking."""

    def test_increase_structure_id_initial(self):
        """Test that initial structure ID is 0."""
        # Reset to known state
        import calc
        calc._CURRENT_STRUCTURE_ID = 0

        id1 = increase_structure_id()
        assert id1 == 0

    def test_increase_structure_id_increments(self):
        """Test that structure ID increments correctly."""
        # Reset to known state
        import calc
        calc._CURRENT_STRUCTURE_ID = 0

        id1 = increase_structure_id()
        id2 = increase_structure_id()
        id3 = increase_structure_id()

        assert id1 == 0
        assert id2 == 1
        assert id3 == 2

    def test_increase_structure_id_returns_before_increment(self):
        """Test that structure ID returns pre-increment value."""
        import calc
        calc._CURRENT_STRUCTURE_ID = 100

        returned_id = increase_structure_id()
        assert returned_id == 100

        next_id = increase_structure_id()
        assert next_id == 101


class TestSetConfig:
    """Tests for the set_config function."""

    def test_set_config_updates_globals(self, sample_config):
        """Test that set_config updates module-level globals."""
        import calc

        set_config(sample_config)

        assert calc.ORCA_EXEC_COMMAND == sample_config.orca_exec_command
        assert calc.NUM_OF_PROCS == sample_config.num_of_procs
        assert calc.ORCA_METHOD == sample_config.orca_method
        assert calc.CHARGE == sample_config.charge
        assert calc.MULTIPL == sample_config.spin_multiplicity
        assert calc.TS == sample_config.ts
        assert calc.BROKEN_STRUCT_ENERGY == sample_config.broken_struct_energy
        assert calc.BOND_LENGTH_THRESHOLD == sample_config.bond_length_threshold
        assert calc.ACQUISITION_FUNCTION == sample_config.acquisition_function

    def test_set_config_partial_update(self, sample_config):
        """Test that set_config completely replaces config."""
        import calc

        original_ts = sample_config.ts
        set_config(sample_config)
        assert calc.TS == original_ts

        # Create a new config with different TS value
        modified_config = ConfSearchConfig(
            mol_file_name=sample_config.mol_file_name,
            ts=not original_ts,  # Flip the value
        )
        set_config(modified_config)
        assert calc.TS == (not original_ts)


@pytest.fixture
def sample_config(sample_config_dict):
    """Return a ConfSearchConfig instance."""
    return ConfSearchConfig(**sample_config_dict)
