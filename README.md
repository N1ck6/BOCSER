# BOCSER: Physics-Informed Bayesian Optimization for Conformational Ensemble Augmentation

**Physics-informed Bayesian Optimization for Conformational Ensemble Augmentation**

BOCSER finds all conformers of a molecule using Gaussian Process Regression with a physics-informed kernel.

---

## Quick Start

```bash
# 1. Install dependencies
conda create -n bocser python=3.10 && conda activate bocser
pip install -r requirements.txt

# 2. Prepare your working directory
mkdir my_run && cd my_run
# Place molecule.mol and config.yaml here

# 3. Run
python bocser/conf_search.py --folder=my_run --config=config.yaml

# Or via Slurm (requiers installed ORCA to perform calculations and Slurm workload manager to manage them)
sbatch run_cs.sh my_run
```

---

## Input

| File | Description |
|---|---|
| `molecule.mol` | RDKit MDL MOL file with 3D coordinates |
| `config.yaml` | Experiment settings (see below) |
| ORCA binary | Quantum chemistry engine, installed separately |
| `sbatch_temp` | Slurm batch script template |

### Structure of `config.yaml`

```yaml
mol_file_name: "molecule.mol"
exp_name: "my_experiment"
orca_exec_command: "/path/to/orca"
orca_method: "PBE def2-SVP"
num_of_procs: 8
charge: 0
spin_multiplicity: 1
acquisition_function: "evm"   # "ei", "evm", or "ik"
num_initial_points: 5
max_steps: 100
```

---

## Output

All files are written to `--folder`, prefixed with `exp_name`:

| File | Contents |
|---|---|
| `<exp_name>_final_ensemble.xyz` | **Main output.** Deduplicated stable conformers |
| `<exp_name>_all_minima.json` | All local minima with relative energies (kcal/mol) |
| `<exp_name>_logs.json` | Acquisition values and best energy per BO step |
| `<exp_name>_minima/` | Individual `.xyz` files for each minimum |
| `<exp_name>_clustering_results.json` | DBSCAN cluster assignments |

---

## How It Works

1. **Detect rotatable bonds** in the molecule (RDKit).
2. **Pre-scan each bond** through 360° with ORCA → fit Fourier coefficients → these become the GP prior mean (cached in `dihedral_logs.db`).
3. **Initialize dataset** from random RDKit embeddings or a seed ensemble.
4. **Bayesian optimization loop** (up to `max_steps`):
   - Acquisition function selects next dihedral angles to evaluate.
   - ORCA runs a two-stage optimization: constrained pre-opt → full geometry opt.
   - Result updates the GP model.
   - Early-stop when acquisition values plateau (rolling window criterion).
5. **Cluster** all discovered minima with DBSCAN (angular distance) → write final ensemble.

---

## Acquisition Functions

| Value | Description |
|---|---|
| `evm` | Explorational Variance Minimizer — balances exploration and exploitation (default) |
| `ei` | Expected Improvement — classic BO, more exploitative |
| `ik` | IK-aware EVM — adds ring-closure penalty for macrocycles/ring molecules |

---

## Running Tests

```bash
# Fast tests only (no TF/ORCA needed)
pytest bocser/tests/test_config_loader_only.py -v

# Full test suite
pytest bocser/tests/ -v --timeout=30
```
