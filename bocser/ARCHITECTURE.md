# BOCSER Refactoring: Architecture Overview

This document describes the refactored codebase architecture, design patterns, and migration guide.

## Table of Contents
- [Overview](#overview)
- [Key Changes](#key-changes)
- [Architecture Patterns](#architecture-patterns)
- [Module Reference](#module-reference)
- [Migration Guide](#migration-guide)
- [Testing](#testing)

---

## Overview

The BOCSER codebase has been refactored to improve:
- **Testability**: Eliminated module-level global state
- **Safety**: Replaced unsafe shell execution with subprocess
- **Configuration Management**: Centralized robust config loading
- **Code Quality**: Improved type hints, documentation, and lint compliance
- **Maintainability**: Class-based orchestrator for clearer control flow

### Design Philosophy
- **Explicit State**: All runtime state owned by classes, not module globals
- **Type Safety**: Configuration validated and type-coerced at load time
- **Backward Compatibility**: Dual-mode APIs support both old (globals) and new (parameters) styles
- **Dependency Injection**: Prefer passing dependencies over module-level imports

---

## Key Changes

### 1. **Class-Based Orchestrator** (conf_search.py)

**Before**: Module-level globals, implicit state
```python
# Old style (don't use)
config = None  # Global state
def load_config(path):
    global config
    config = ...
```

**After**: Explicit class ownership
```python
# New style (preferred)
runner = ConfSearchRunner()
runner.load_config("config.yaml")
runner.run()  # All state tracked in runner.state
```

**Benefits**:
- Testable: Instantiate multiple runners in tests
- Traceable: State visible through `runner.state` object
- Parallelizable: Multiple runners can run independently

### 2. **Centralized Configuration** (config_loader.py)

**Before**: Inline YAML parsing, scattered validation
```python
# Old style (don't use)
with open(config_file) as f:
    config = yaml.safe_load(f)
    num_of_procs = int(config.get('num_of_procs', 4))
```

**After**: Robust typed loader
```python
# New style (preferred)
from config_loader import load_config
config = load_config("config.yaml")
# config is now ConfSearchConfig with validated types
```

**Features**:
- Type coercion: `"true"` → `True`, `"42"` → `42`, `"3.14"` → `3.14`
- Error handling: ConfigError with clear messages
- Validation: Unknown keys trigger warnings but don't crash
- Defaults: Missing fields use sensible defaults

### 3. **Safe Subprocess Execution** (calc.py)

**Before**: Dangerous shell execution
```python
# Old style (UNSAFE - don't use)
os.system(f"orca {oinp_file} > {out_file}")
```

**After**: Safe subprocess call
```python
# New style (preferred)
import subprocess
import shlex
process = subprocess.run(
    shlex.split(f"orca {oinp_file}"),
    capture_output=True,
    timeout=300,
    check=True
)
```

**Benefits**:
- No command injection vulnerabilities
- Better error handling (exceptions vs return codes)
- Timeout support
- Output capture without shell redirection

### 4. **Dual-Mode Config Propagation** (calc.py)

**Before**: Global state only
```python
# Old style (still works but)
set_config(config)
energy = calc_energy(mol, dihedral)
```

**After**: Functions accept optional config parameters
```python
# New style (preferred)
energy = calc_energy(mol, dihedral, config=config)

# Old style (still supported for backward compatibility)
set_config(config)
energy = calc_energy(mol, dihedral)  # Uses global config
```

**Benefits**:
- Explicit dependencies: Who needs what?
- Testable: Pass test configs directly
- No hidden state: Functions are deterministic

---

## Architecture Patterns

### Pattern 1: State Encapsulation

```python
class ConfSearchRunner:
    """Orchestrates the entire workflow."""
    
    def __init__(self):
        self.state = ConfSearchState()  # Single source of truth
    
    def load_config(self, path):
        """Load config and update state."""
        config = load_config(path)
        self.state.config = config
    
    def run(self):
        """Execute workflow; state updated throughout."""
        self._build_model_and_acquisition()
        self._initialize_dataset()
        # ... workflow proceeds, state tracked in self.state
```

### Pattern 2: Dependency Injection

```python
# Instead of:
def calc_energy(mol, dihedral):
    config = get_global_config()  # Hidden dependency
    ...

# Do this:
def calc_energy(mol, dihedral, config=None):
    if config is None:
        config = get_global_config()  # Backward compat fallback
    ...

# Call with explicit config:
energy = calc_energy(mol, dihedral, config=runner.state.config)
```

### Pattern 3: Configuration Validation

```python
# config_loader.py handles type coercion
config_dict = yaml.safe_load(file)  # Raw data

# Apply type coercion and validation
config = load_config("config.yaml")  # ConfSearchConfig with validated types

# Access with confidence:
for i in range(config.num_of_procs):  # Always int
    ...
```

### Pattern 4: Graceful Degradation

```python
# Unknown config keys don't crash:
config = load_config("config.yaml")  # Warns on unknown keys, continues

# Defaults apply automatically:
if config.orca_method is None:
    config.orca_method = "PM6"  # Uses default

# Missing files are handled:
try:
    config = load_config("nonexistent.yaml")
except ConfigError as e:
    print(f"Config error: {e}")
    config = ConfSearchConfig()  # Use defaults
```

---

## Module Reference

### config_loader.py

**Purpose**: Centralized, type-safe configuration loading

**Key Functions**:
- `load_config(path: str) -> ConfSearchConfig`
  - Load YAML config file
  - Apply type coercion and validation
  - Raise `ConfigError` on problems
  - Return `ConfSearchConfig` instance

**Key Classes**:
- `ConfSearchConfig`: Dataclass with all configuration fields
  ```python
  @dataclass
  class ConfSearchConfig:
      mol_file_name: str
      charge: int
      multipl: int
      num_of_procs: int
      orca_method: str
      # ... more fields with defaults
  ```

**Usage**:
```python
from config_loader import load_config

# Load from file
config = load_config("config.yaml")

# Or create with defaults
config = ConfSearchConfig()

# Access fields
print(config.mol_file_name)
print(config.charge)
```

### conf_search.py

**Purpose**: Main orchestrator for Bayesian optimization workflow

**Key Classes**:
- `ConfSearchRunner`: Orchestrates entire workflow
  - `load_config(path)`: Load config from YAML
  - `setup()`: Initialize datasets and models
  - `run()`: Execute Bayesian optimization
  - `_save_results()`: Export results
  
- `ConfSearchState`: Runtime state dataclass
  - `config`: Current configuration
  - `dataset`: Current dataset state
  - `trajectory`: Optimization trajectory

- `PotentialFunction`: Wraps objective function
  - `__call__(x)`: Evaluate objective at point x

**Usage**:
```python
from conf_search import ConfSearchRunner

# Create runner
runner = ConfSearchRunner()

# Load config
runner.load_config("config.yaml")

# Run optimization
runner.setup()
runner.run()

# Access results
print(runner.state.dataset)
print(runner.state.trajectory)
```

### calc.py

**Purpose**: Chemistry utilities, ORCA integration, trajectory parsing

**Key Functions**:
- `set_config(config)`: Set global config (backward compat)
- `calc_energy(mol, dihedral, config=None)`: Calculate energy
- `generate_default_oinp(mol_file, config=None)`: Generate ORCA input
- `dihedral_angle(coords, atoms)`: Calculate dihedral angle
- `check_is_broken(mol)`: Validate molecule geometry
- `increase_structure_id(file_name)`: Increment structure ID

**Usage**:
```python
from calc import calc_energy, generate_default_oinp
from config_loader import load_config

config = load_config("config.yaml")

# Calculate energy (explicit config)
energy = calc_energy(mol, dihedral, config=config)

# Generate ORCA input (explicit config)
oinp = generate_default_oinp(mol_file, config=config)

# Or use global config (backward compat)
calc.set_config(config)
energy = calc_energy(mol, dihedral)
```

### Other Modules

- **ik_loss.py**: Type hints, improved documentation (no state changes)
- **coef_calc.py**: Coefficient calculations (stable)
- **evm.py**: Extra model validation (stable)
- **imp_var_with_ik.py**: Importance-weighted variables (stable)
- **db_connector.py**: Database operations (stable)
- **dbscan.py**: Clustering utilities (stable)

---

## Migration Guide

### For Users (Running Existing Code)

**Option A: No changes needed** (backward compat)
```python
# Old code still works
from calc import set_config, calc_energy
from config_loader import load_config

config = load_config("config.yaml")
set_config(config)
energy = calc_energy(mol, dihedral)  # Works as before
```

**Option B: Use new orchestrator** (recommended)
```python
# Cleaner, more testable
from conf_search import ConfSearchRunner

runner = ConfSearchRunner()
runner.load_config("config.yaml")
runner.run()
```

### For Developers (Writing New Code)

**Rule 1: Use dependency injection**
```python
# ✓ Good: Dependencies explicit
def my_function(config, mol):
    energy = calc_energy(mol, dihedral, config=config)
    ...

# ✗ Avoid: Hidden global dependency
def my_function(mol):
    energy = calc_energy(mol, dihedral)  # Where does config come from?
    ...
```

**Rule 2: Use ConfSearchConfig**
```python
# ✓ Good: Type-safe
from config_loader import load_config
config = load_config("config.yaml")
assert isinstance(config.num_of_procs, int)

# ✗ Avoid: Untyped dictionaries
config = yaml.safe_load(open("config.yaml"))
# Is config['num_of_procs'] int or string?
```

**Rule 3: Use ConfSearchRunner for orchestration**
```python
# ✓ Good: Clear control flow
runner = ConfSearchRunner()
runner.load_config("config.yaml")
runner.setup()
runner.run()

# ✗ Avoid: Scattered function calls with implicit state
load_config_global("config.yaml")
setup_global()
run_global()  # What config? What state?
```

### Migrating Existing Functions

**Step 1: Add config parameter**
```python
# Before
def calc_energy(mol, dihedral):
    num_procs = _global_config.num_of_procs
    ...

# After
def calc_energy(mol, dihedral, config=None):
    if config is None:
        config = _get_global_config()
    num_procs = config.num_of_procs
    ...
```

**Step 2: Update docstring**
```python
def calc_energy(mol, dihedral, config=None):
    """Calculate energy of molecule at given dihedral angle.
    
    Args:
        mol: RDKit molecule object
        dihedral: Dihedral angle in degrees
        config: ConfSearchConfig instance (optional, uses global if None)
    
    Returns:
        float: Energy in Hartree
    """
```

**Step 3: Add tests**
```python
def test_calc_energy_with_explicit_config():
    config = ConfSearchConfig(num_of_procs=4, orca_method="PM6")
    energy = calc_energy(test_mol, 120.0, config=config)
    assert energy > 0
```

---

## Testing

### Running Tests

**Quick test** (no heavy dependencies):
```bash
pytest tests/test_config_loader_only.py -v
```

**Full test suite** (requires TensorFlow, scikit-learn):
```bash
pytest tests/ -v
```

**See TESTING.md for details**

### Test Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── test_config_loader_only.py  # Lightweight config tests ✅ 13/13
├── test_config_loader.py       # Full config tests
├── test_conf_search.py         # Orchestrator tests
└── test_calc.py                # Utility function tests
```

### Writing Tests

```python
def test_my_feature(self, sample_config):
    """Test my feature with sample config fixture."""
    # Arrange
    runner = ConfSearchRunner()
    
    # Act
    runner.state.config = sample_config
    
    # Assert
    assert runner.state.config.num_of_procs == 4
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"
- Full tests require: `pip install tensorflow gpflow trieste scikit-learn`
- Lightweight tests work: `pip install pytest pyyaml`
- Run lightweight only: `pytest tests/test_config_loader_only.py`

### "Global config not working"
- Check `set_config()` was called: `calc.set_config(config)`
- Use explicit config parameter instead: `calc_energy(mol, d, config=config)`

### "Config loading fails with unknown key"
- This is expected behavior (graceful degradation)
- Unknown keys trigger warning but don't crash
- Check your config file for typos

### "Tests hang or are slow"
- Run specific test: `pytest tests/file.py::TestClass::test_name`
- Check for infinite loops in geometry optimization
- Use timeout: `pytest --timeout=10`

---

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| State Management | Module globals | Class ownership (ConfSearchRunner) |
| Config Loading | Inline YAML parsing | Centralized config_loader |
| Type Safety | Untyped dictionaries | ConfSearchConfig dataclass |
| Process Execution | `os.system()` | `subprocess.run()` |
| Testability | Difficult (globals) | Easy (explicit dependencies) |
| Backward Compat | N/A | Full (set_config() still works) |

---

## Next Steps

1. **For Users**: Start with `ConfSearchRunner` for new projects
2. **For Developers**: Follow migration guide when updating functions
3. **For CI/CD**: Run lightweight tests on commit, full tests on PR
4. **For Documentation**: Update team wiki with new patterns

---

**Questions?** See TESTING.md for testing details or check the docstrings in refactored modules.
