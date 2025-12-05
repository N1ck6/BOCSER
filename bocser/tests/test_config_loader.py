"""Unit tests for config_loader module."""

import pytest
import yaml
from pathlib import Path

from config_loader import load_config, ConfigError
from default_vals import ConfSearchConfig


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_config(self, sample_config_file):
        """Test loading a valid config file."""
        config = load_config(str(sample_config_file))
        assert isinstance(config, ConfSearchConfig)
        assert config.mol_file_name == "test.mol"
        assert config.num_of_procs == 4
        assert config.acquisition_function == "evm"

    def test_load_config_with_defaults(self, temp_dir):
        """Test that missing keys use defaults from dataclass."""
        config_path = temp_dir / "minimal_config.yaml"
        minimal_config = {"mol_file_name": "molecule.mol"}
        with open(config_path, "w") as f:
            yaml.dump(minimal_config, f)

        config = load_config(str(config_path))
        assert config.mol_file_name == "molecule.mol"
        assert config.spin_multiplicity == 1  # default
        assert config.charge == 0  # default
        assert config.max_steps == 50  # default

    def test_load_config_file_not_found(self):
        """Test error handling when config file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_load_config_invalid_yaml(self, temp_dir):
        """Test error handling for invalid YAML."""
        config_path = temp_dir / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("{ invalid yaml: [ missing bracket")

        with pytest.raises(ConfigError):
            load_config(str(config_path))

    def test_load_config_not_dict(self, temp_dir):
        """Test error handling when YAML is not a dict."""
        config_path = temp_dir / "list_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(["item1", "item2"], f)

        with pytest.raises(ConfigError):
            load_config(str(config_path))

    def test_unknown_config_key_warns(self, temp_dir, capsys):
        """Test that unknown keys trigger a warning but don't fail."""
        config_path = temp_dir / "config_with_unknown.yaml"
        config_data = {
            "mol_file_name": "test.mol",
            "unknown_key": "should_warn",
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        assert config.mol_file_name == "test.mol"
        captured = capsys.readouterr()
        assert "unknown_key" in captured.out

    def test_type_coercion_bool_true(self, temp_dir):
        """Test type coercion for boolean values."""
        config_path = temp_dir / "bool_config.yaml"
        config_data = {
            "mol_file_name": "test.mol",
            "ts": "true",  # string instead of bool
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        assert config.ts is True

    def test_type_coercion_bool_false(self, temp_dir):
        """Test type coercion for boolean values (false)."""
        config_path = temp_dir / "bool_config_false.yaml"
        config_data = {
            "mol_file_name": "test.mol",
            "ts": "false",
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        assert config.ts is False

    def test_type_coercion_int(self, temp_dir):
        """Test type coercion for int values."""
        config_path = temp_dir / "int_config.yaml"
        config_data = {
            "mol_file_name": "test.mol",
            "num_of_procs": "8",  # string instead of int
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        assert config.num_of_procs == 8
        assert isinstance(config.num_of_procs, int)

    def test_type_coercion_float(self, temp_dir):
        """Test type coercion for float values."""
        config_path = temp_dir / "float_config.yaml"
        config_data = {
            "mol_file_name": "test.mol",
            "bond_length_threshold": "1.5",  # string instead of float
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_path))
        assert config.bond_length_threshold == 1.5
        assert isinstance(config.bond_length_threshold, float)

    def test_empty_config_file(self, temp_dir):
        """Test loading an empty config file (should use all defaults)."""
        config_path = temp_dir / "empty_config.yaml"
        with open(config_path, "w") as f:
            f.write("")

        # Should raise error because mol_file_name is required
        with pytest.raises(ConfigError):
            load_config(str(config_path))
