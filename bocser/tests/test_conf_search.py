"""Unit tests for conf_search module (ConfSearchRunner)."""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from conf_search import ConfSearchRunner, ConfSearchState, PotentialFunction
from default_vals import ConfSearchConfig


class TestConfSearchState:
    """Tests for the ConfSearchState dataclass."""

    def test_initial_state(self):
        """Test that initial state is properly initialized."""
        state = ConfSearchState()
        assert state.mol_file_name is None
        assert state.exp_name == ""
        assert state.norm_energy == 0.0
        assert state.dihedral_ids == []
        assert state.asked_points == []
        assert state.minima == []
        assert state.current_minima == 1e9
        assert state.last_opt_ok is True

    def test_state_modification(self):
        """Test that state fields can be modified."""
        state = ConfSearchState()
        state.mol_file_name = "test.mol"
        state.exp_name = "test_exp"
        state.norm_energy = 42.0

        assert state.mol_file_name == "test.mol"
        assert state.exp_name == "test_exp"
        assert state.norm_energy == 42.0


class TestConfSearchRunner:
    """Tests for the ConfSearchRunner orchestrator."""

    def test_runner_initialization(self):
        """Test that ConfSearchRunner initializes with empty state."""
        runner = ConfSearchRunner()
        assert isinstance(runner.state, ConfSearchState)
        assert runner.state.config is None

    def test_load_config(self, sample_config_file):
        """Test loading config from file."""
        runner = ConfSearchRunner()
        runner.load_config(str(sample_config_file))

        assert runner.state.config is not None
        assert runner.state.config.mol_file_name == "test.mol"
        assert runner.state.config.num_of_procs == 4

    def test_load_config_file_not_found(self):
        """Test error handling for missing config file."""
        runner = ConfSearchRunner()
        with pytest.raises(FileNotFoundError):
            runner.load_config("/nonexistent/config.yaml")

    def test_dump_status_hook(self, runner_with_config, temp_dir):
        """Test that status hook writes JSON correctly."""
        runner_with_config.state.exp_name = str(temp_dir / "test")
        runner_with_config._dump_status_hook(True)

        status_file = Path(f"{runner_with_config.state.exp_name}_last_opt_status.json")
        assert status_file.exists()

        with open(status_file, "r") as f:
            data = json.load(f)
        assert data["LAST_OPT_OK"] is True

        # Cleanup
        status_file.unlink()

    def test_extract_dofs_values_mock(self, runner_with_config):
        """Test dihedral extraction (with mocked molecule)."""
        runner_with_config.state.dihedral_ids = [(0, 1, 2, 3), (1, 2, 3, 4)]

        # Mock molecule
        mock_mol = MagicMock()
        mock_conf = MagicMock()
        mock_mol.GetConformer.return_value = mock_conf

        # Mock GetDihedralRad to return known values
        mock_conf.GetDihedralRad.side_effect = [0.5, 1.0]

        # This will be called but we're mainly testing the structure
        with patch("rdkit.Chem.rdMolTransforms.GetDihedralRad", side_effect=[0.5, 1.0]):
            result = runner_with_config._extract_dofs_values(mock_mol)
            # Result should be a TensorFlow constant
            assert result is not None

    def test_erase_last_from_dataset_mock(self, runner_with_config):
        """Test dataset editing functionality."""
        # Mock Dataset and TensorFlow
        mock_dataset = MagicMock()
        mock_query_points = MagicMock()
        mock_observations = MagicMock()

        mock_dataset.query_points = mock_query_points
        mock_dataset.observations = mock_observations

        mock_query_points.shape = (10, 5)
        mock_observations.shape = (10, 1)

        # Mock tf.slice to return predictable values
        with patch("tensorflow.slice") as mock_slice:
            mock_slice.side_effect = [mock_query_points, mock_observations]
            # Test that the function is callable and handles slicing
            result = runner_with_config._erase_last_from_dataset(mock_dataset, 1)
            assert result is not None


class TestPotentialFunction:
    """Tests for the PotentialFunction class."""

    def test_initialization(self):
        """Test that PotentialFunction initializes with coefficients."""
        mean_func_coefs = [
            [1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.1],
            [1.1, 2.1, 3.1, 0.5, 0.6, 0.7, 0.15],
        ]
        pf = PotentialFunction(mean_func_coefs)
        assert pf.mean_func_coefs == mean_func_coefs

    def test_callable(self):
        """Test that PotentialFunction is callable."""
        mean_func_coefs = [
            [1.0, 2.0, 3.0, 0.5, 0.6, 0.7, 0.1],
        ]
        pf = PotentialFunction(mean_func_coefs)
        assert callable(pf)


@pytest.fixture
def runner_with_config(sample_config):
    """Return a ConfSearchRunner with loaded config."""
    runner = ConfSearchRunner()
    runner.state.config = sample_config
    return runner


@pytest.fixture
def runner_with_state(runner_with_config, temp_dir):
    """Return a ConfSearchRunner with initialized state."""
    runner = runner_with_config
    runner.state.mol_file_name = "test.mol"
    runner.state.exp_name = "test_exp"
    runner.state.structures_path = str(temp_dir / "structures/")
    runner.state.dihedral_ids = [(0, 1, 2, 3)]
    runner.state.norm_energy = 10.0
    runner.state.last_opt_ok = True
    return runner
