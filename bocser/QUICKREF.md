# Quick Reference: BOCSER Refactored API

## Essential Commands

### Running the Optimization

```python
from conf_search import ConfSearchRunner

runner = ConfSearchRunner()
runner.load_config("config.yaml")
runner.setup()
runner.run()
```

### Loading Configuration

```python
from config_loader import load_config

config = load_config("config.yaml")
# config is now ConfSearchConfig with validated types
print(config.num_of_procs)  # int
print(config.charge)        # int
print(config.orca_method)   # str
```

### Calculating Energy

```python
from calc import calc_energy
from config_loader import load_config

config = load_config("config.yaml")
energy = calc_energy(mol, dihedral=120.0, config=config)
```

### Generating ORCA Input

```python
from calc import generate_default_oinp
from config_loader import load_config

config = load_config("config.yaml")
oinp_content = generate_default_oinp("molecule.mol", config=config)
```

## ConfSearchConfig Fields

```python
@dataclass
class ConfSearchConfig:
    # Required or with defaults
    mol_file_name: str = "mol.mol"
    charge: int = 0
    multipl: int = 1
    num_of_procs: int = 4
    orca_method: str = "PM6"
    num_of_optimization_steps: int = 10
    budget_limit: int = 200
    # ... and more fields
```

## ConfSearchRunner Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `load_config(path)` | Load YAML configuration | None |
| `setup()` | Initialize models and datasets | None |
| `run()` | Execute Bayesian optimization | None |
| `_save_results()` | Export results to files | None |

## ConfSearchState

Access runtime state through `runner.state`:

```python
runner = ConfSearchRunner()
runner.load_config("config.yaml")

# Before setup
print(runner.state.config)  # Loaded config

# After setup
print(runner.state.dataset)
print(runner.state.model)
print(runner.state.model_data)
print(runner.state.dof_idx_in_atoms)
```

## Common Patterns

### Pattern 1: Basic Workflow
```python
from conf_search import ConfSearchRunner

runner = ConfSearchRunner()
runner.load_config("config.yaml")
runner.setup()
runner.run()
```

### Pattern 2: Custom Processing
```python
from conf_search import ConfSearchRunner
from calc import calc_energy

runner = ConfSearchRunner()
runner.load_config("config.yaml")

config = runner.state.config
for mol in molecules:
    energy = calc_energy(mol, 120.0, config=config)
    print(f"Energy: {energy}")
```

### Pattern 3: Backward Compatibility
```python
# Old code still works
from calc import set_config, calc_energy
from config_loader import load_config

config = load_config("config.yaml")
set_config(config)
energy = calc_energy(mol, dihedral)
```

### Pattern 4: Testing
```python
from conf_search import ConfSearchRunner
from config_loader import ConfSearchConfig

# Create runner with custom config
runner = ConfSearchRunner()
runner.state.config = ConfSearchConfig(
    num_of_procs=1,
    orca_method="PM6"
)
```

## Error Handling

### Config Errors
```python
from config_loader import load_config, ConfigError

try:
    config = load_config("config.yaml")
except ConfigError as e:
    print(f"Config error: {e}")
    config = ConfSearchConfig()  # Use defaults
```

### ORCA Errors
```python
from calc import calc_energy

try:
    energy = calc_energy(mol, dihedral, config=config)
except subprocess.CalledProcessError as e:
    print(f"ORCA failed: {e.stderr}")
```

## Type Hints

All functions use type hints. Examples:

```python
from config_loader import load_config, ConfSearchConfig
from typing import Optional

def my_function(
    config: ConfSearchConfig,
    num_iterations: int = 10,
    debug: bool = False
) -> float:
    """Do something with config."""
    return 0.0

# Load config with type safety
config = load_config("config.yaml")
result = my_function(config, num_iterations=20, debug=True)
```

## Configuration File Example

```yaml
# config.yaml
mol_file_name: molecule.mol
charge: 0
multipl: 1
num_of_procs: 4
orca_method: PM6
num_of_optimization_steps: 50
budget_limit: 500
```

## Running Tests

```bash
# Lightweight tests (no heavy dependencies)
pytest tests/test_config_loader_only.py -v

# Full test suite (requires TensorFlow, sklearn)
pytest tests/ -v

# Specific test
pytest tests/test_config_loader_only.py::TestLoadConfig::test_load_valid_config -v

# With coverage
pytest tests/ --cov --cov-report=html
```

## Migration Checklist

When updating existing code:

- [ ] Replace inline YAML parsing with `load_config()`
- [ ] Replace `os.system()` with `subprocess.run()`
- [ ] Add `config` parameter to functions (with `config=None` default)
- [ ] Use `ConfSearchRunner` for orchestration
- [ ] Add docstrings with `Args` and `Returns`
- [ ] Add type hints to function signatures
- [ ] Add unit tests for new functions
- [ ] Run linter: `pylint *.py`
- [ ] Run tests: `pytest tests/ -v`

## Key Imports

```python
# Configuration
from config_loader import load_config, ConfSearchConfig, ConfigError

# Orchestration
from conf_search import ConfSearchRunner, ConfSearchState, PotentialFunction

# Chemistry utilities
from calc import calc_energy, generate_default_oinp, set_config

# Other modules (stable, no changes)
from coef_calc import ...
from evm import ...
from ik_loss import ...
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: tensorflow` | `pip install tensorflow gpflow trieste scikit-learn` |
| `ConfigError: Unknown key` | Check YAML file for typos (it's a warning, not fatal) |
| `calc_energy not working` | Ensure config passed or `set_config()` called |
| `Tests fail with dependencies` | Use `pytest tests/test_config_loader_only.py` instead |
| `ORCA not found` | Check `config.orca_method` value and ORCA installation |

## Documentation

- **ARCHITECTURE.md**: Detailed architecture explanation
- **TESTING.md**: Complete testing guide
- **This file**: Quick reference for common tasks

## Version Info

- **Python**: 3.7+
- **Key Dependencies**: TensorFlow 2.x, GPflow, trieste, RDKit, scikit-learn
- **Testing**: pytest 9.0+

---

**Need help?** Check ARCHITECTURE.md for detailed explanations or TESTING.md for test setup.
