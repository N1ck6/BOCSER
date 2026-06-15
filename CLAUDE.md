# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is BOCSER?

Physics-informed Bayesian Optimization for Conformational Ensemble Augmentation. It uses Gaussian Process Regression with ORCA quantum chemistry calculations to explore molecular conformational space. Requires ORCA and (for cluster runs) Slurm.

## Commands

```bash
# Install dependencies (Python 3.10)
conda create -n bocser python=3.10
conda activate bocser
pip install tensorflow gpflow trieste rdkit scikit-learn pyyaml pytest ringo-ik

# Run conformational search
python bocser/conf_search.py --folder=<working_dir> --config=config.yaml

# Via Slurm
sbatch run_cs.sh <working_dir>

# Run all tests
pytest bocser/tests/ -v

# Run lightweight tests only (no TF/GPflow needed)
pytest bocser/tests/test_config_loader_only.py -v

# Run a specific test
pytest bocser/tests/test_calc.py::TestClass::test_name -v

# With coverage
pytest bocser/tests/ --cov --cov-report=html
```

## Architecture

All source code lives in `bocser/`. The module imports use relative names (e.g. `from calc import ...`), so scripts must be run from the `bocser/` directory or with it on `PYTHONPATH`. The entry point `conf_search.py` adds its own directory to `sys.path` via the `main()` function.

### Core data flow

1. **`config_loader.py`** — reads `config.yaml`, validates keys against `ConfSearchConfig` (defined in `default_vals.py`), returns a typed dataclass. Raises `ConfigError` on bad values.

2. **`coef_calc.py` (`CoefCalculator`)** — decomposes the molecule into rotatable dihedrals, runs 1D ORCA torsion scans (or fetches cached results from SQLite), and fits Fourier coefficients for the GP mean function. Results are cached in `dihedral_logs.db` (sibling to `bocser/`).

3. **`conf_search.py` (`ConfSearchRunner`)** — main orchestrator. Holds all mutable state in `ConfSearchState`. Key lifecycle: `load_config()` → `setup()` → `run()`. The `run()` method builds a GPR model with a physics-informed kernel (`TransformKernel` + Periodic + White), collects initial points (random RDKit embeddings or a pre-existing ensemble), then iterates Bayesian optimization steps until convergence or `max_steps`.

4. **`calc.py`** — wraps ORCA invocations via `subprocess` (no `shell=True`). Handles two-stage optimization: constrained pre-opt then full opt. Reads runtime config from `config_manager` singleton. Energy unit conversion: Hartree → kcal/mol via `HARTRI_TO_KCAL = 627.509474063`.

5. **`db_connector.py`** — SQLite-backed cache for 1D scan results. Avoids redundant ORCA runs for already-scanned dihedrals.

### Acquisition functions (configured via `acquisition_function` in config)

- `ei` — standard Expected Improvement (trieste built-in)
- `evm` — Explorational Variance Minimizer (`evm.py`), custom acquisition that balances exploration with a threshold
- `ik` — Improvement Variance with Inverse Kinematics loss (`imp_var_with_ik.py`), incorporates ring-closure constraints via `ik_loss.py` (uses the `ringo` library)

### Output files (all prefixed with `exp_name`, written to `--folder`)

- `{exp_name}/` — optimized structures per step
- `{exp_name}_minima/` — XYZ for each local minimum found
- `{exp_name}_logs.json` — acquisition function values and deepest minima per step
- `{exp_name}_all_minima.json` — all minima coordinates and relative energies
- `{exp_name}_final_ensemble.xyz` — deduplicated ensemble after DBSCAN clustering
- `{exp_name}_clustering_results.json` — cluster assignments
- `{exp_name}_last_opt_status.json` — boolean flag written after each ORCA call

### Configuration (`config.yaml`)

Key fields in `ConfSearchConfig` (`default_vals.py`):
- `mol_file_name` — path to `.mol` file (relative to `--folder`)
- `exp_name` — prefix for all output files
- `orca_exec_command` — path to ORCA binary
- `orca_method` — method string in ORCA format (e.g. `"M062X"`)
- `acquisition_function` — one of `ei`, `evm`, `ik`
- `load_ensemble` — optional path to `.xyz` to seed initial dataset
- `rolling_window_size`, `rolling_std_threshold`, `rolling_mean_threshold` — early termination criteria

### Global state pattern

`config_manager.py` holds a module-level singleton accessed via `get_config()`/`set_config()`. `calc.py` reads config exclusively through `_get_config_or_raise()` — never accept a config parameter directly from user code without calling `config_manager.set_config()` first in the `ConfSearchRunner.setup()` method.
