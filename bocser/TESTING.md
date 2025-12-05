"""Testing Guide for BOCSER Refactored Codebase

This document explains the test structure, how to run tests, and guidelines
for adding new tests to the project.
"""

# ============================================================================
# TEST STRUCTURE
# ============================================================================

"""
tests/
├── __init__.py                 - Package marker
├── conftest.py                 - Shared fixtures and test utilities
├── test_config_loader_only.py  - Lightweight tests (no TF/sklearn deps)
├── test_config_loader.py       - Full config_loader tests
├── test_conf_search.py         - ConfSearchRunner orchestrator tests
└── test_calc.py                - calc module utility function tests

DEPENDENCY LEVELS:
- test_config_loader_only.py: LIGHT (yaml only)
- test_config_loader.py: LIGHT (yaml only)
- test_conf_search.py: HEAVY (TensorFlow, gpflow, trieste)
- test_calc.py: HEAVY (TensorFlow, sklearn, rdkit)
"""

# ============================================================================
# RUNNING TESTS
# ============================================================================

"""
Quick Test (no heavy dependencies):
  $ pytest tests/test_config_loader_only.py -v
  Expected: 13 tests pass

Full Test Suite (requires all dependencies):
  $ pytest tests/ -v
  Expected: ~40+ tests covering all modules

Test with Coverage Report:
  $ pytest tests/ --cov=. --cov-report=html
  Coverage report generated in htmlcov/index.html

Test Specific Class:
  $ pytest tests/test_config_loader_only.py::TestLoadConfig -v

Test Specific Function:
  $ pytest tests/test_config_loader_only.py::TestLoadConfig::test_load_valid_config -v

Verbose Output with Print Statements:
  $ pytest tests/ -v -s
"""

# ============================================================================
# TEST ORGANIZATION BY MODULE
# ============================================================================

"""
config_loader.py
  Tests: test_config_loader_only.py, test_config_loader.py
  Coverage:
    - load_config() with valid/invalid files
    - Type coercion (string -> bool, int, float)
    - Error handling (FileNotFoundError, ConfigError)
    - Default values and unknown keys
  Status: ✅ 13/13 tests passing (lightweight suite)

ConfSearchRunner (conf_search.py)
  Tests: test_conf_search.py
  Coverage:
    - ConfSearchState initialization
    - Config loading (load_config)
    - State management (_dump_status_hook, _extract_dofs_values)
    - Dataset operations (_erase_last_from_dataset, _upd_dataset_from_trj)
    - PotentialFunction class
  Status: ⏸️ Requires TensorFlow/gpflow/trieste

calc.py
  Tests: test_calc.py
  Coverage:
    - Dihedral angle calculation
    - Unit conversions (radians ↔ degrees)
    - File naming conventions (MOL ↔ INP ↔ OUT)
    - Geometry validation (check_is_broken)
    - Structure ID tracking (increase_structure_id)
    - Config propagation (set_config)
  Status: ⏸️ Requires TensorFlow/sklearn/rdkit
"""

# ============================================================================
# SHARED FIXTURES
# ============================================================================

"""
conftest.py provides:

temp_dir
  - Temporary directory cleaned up after test
  - Useful for file I/O tests

sample_config_dict
  - Dictionary with all ConfSearchConfig fields
  - Ready to convert to YAML or dataclass

sample_config_file
  - YAML file in temp_dir with sample config
  - Used by config loader tests

sample_config
  - ConfSearchConfig instance
  - Used by ConfSearchRunner and calc tests

sample_xyz_content
  - Sample XYZ file content (3 atoms)
  - Used by molecule-related tests

sample_mol_file
  - Temporary MOL file in temp_dir
  - Used by calc tests

sample_traj_file
  - Temporary trajectory file (multiple structures)
  - Used by parsing tests
"""

# ============================================================================
# ADDING NEW TESTS
# ============================================================================

"""
1. Create a new test class:

   class TestMyFeature:
       \"\"\"Tests for my feature.\"\"\"
       
       def test_basic_behavior(self):
           \"\"\"Test basic behavior.\"\"\"
           # Arrange
           obj = MyClass()
           
           # Act
           result = obj.method()
           
           # Assert
           assert result == expected

2. Use fixtures:

   class TestWithFixtures:
       def test_with_config(self, sample_config):
           \"\"\"Uses sample_config fixture.\"\"\"
           assert sample_config.mol_file_name == "test.mol"
       
       def test_with_files(self, temp_dir, sample_mol_file):
           \"\"\"Uses temp_dir and sample_mol_file.\"\"\"
           assert sample_mol_file.exists()

3. Mock external dependencies:

   from unittest.mock import Mock, patch, MagicMock
   
   def test_with_mock(self):
       mock_mol = MagicMock()
       mock_mol.GetConformer.return_value = Mock()
       # ... use mock_mol

4. Test error conditions:

   import pytest
   
   def test_error_handling(self):
       with pytest.raises(ValueError):
           bad_function()

5. Parametrized tests:

   @pytest.mark.parametrize("input,expected", [
       (1, 2),
       (3, 6),
   ])
   def test_multiple_cases(self, input, expected):
       assert double(input) == expected
"""

# ============================================================================
# TESTING BEST PRACTICES
# ============================================================================

"""
1. Isolation: Each test should be independent
   ✓ Use fixtures to set up state
   ✓ Clean up with teardown or context managers
   ✗ Don't rely on test execution order

2. Clarity: Test names should describe what they test
   ✓ test_load_config_with_defaults
   ✓ test_type_coercion_bool_true
   ✗ test_config
   ✗ test_1

3. Coverage: Test happy path and error cases
   ✓ Valid input and output
   ✓ Invalid input and error handling
   ✓ Edge cases (empty, None, etc.)
   ✗ Only happy path

4. Mocking: Use mocks for heavy dependencies
   ✓ Mock TensorFlow objects in unit tests
   ✓ Test with real objects in integration tests
   ✗ Mix mocks and real objects inconsistently

5. Fixtures: Reuse common setup
   ✓ Define in conftest.py
   ✓ Use @pytest.fixture decorator
   ✗ Duplicate setup in every test
"""

# ============================================================================
# CONTINUOUS INTEGRATION
# ============================================================================

"""
CI/CD Integration (example):

1. Lightweight tests on every commit:
   pytest tests/test_config_loader_only.py -v

2. Full test suite on pull request:
   pytest tests/ -v --cov

3. Track coverage over time:
   pytest tests/ --cov --cov-report=term-missing

4. Generate reports:
   pytest tests/ --html=report.html --self-contained-html
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Issue: ModuleNotFoundError: No module named 'tensorflow'
Solution:
  - Full tests require: pip install tensorflow gpflow trieste scikit-learn
  - Lightweight tests work: pip install pytest pyyaml
  - Or run: pytest tests/test_config_loader_only.py

Issue: ImportError: No module named 'rdkit'
Solution:
  - Install: conda install -c conda-forge rdkit
  - Or skip calc tests: pytest tests/test_config_loader*.py

Issue: Test hangs or takes too long
Solution:
  - Use pytest -v -s to see progress
  - Run single test class: pytest tests/file.py::TestClass
  - Add timeout: pip install pytest-timeout
  - Use: pytest --timeout=10 (10 second timeout)

Issue: Fixture not found
Solution:
  - Check fixture defined in conftest.py
  - Fixture name matches parameter name exactly
  - conftest.py in tests/ directory (not elsewhere)
"""
