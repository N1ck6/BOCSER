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

# 3. Optional check without ORCA: builds the GP axes and writes the pre-opt and
#    full-opt ORCA inputs of one candidate ({exp_name}_dry_run_*.inp) to my_run/
python bocser/conf_search.py --folder my_run --config config.yaml --dry-run

# 4. Run (in the background; '>>' keeps the previous run's log for the backup)
# At the end the log shows a run summary (outcomes by stage, minima, unknown config keys),
# also saved to {exp_name}_run_summary.json
nohup python -u bocser/conf_search.py --folder my_run --config config.yaml >> my_run/run.log 2>&1 &

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
orca_method: "PBE def2-SVP"   # ORCA keywords; Opt/OptTS is appended by BOCSER
# Several lines: the first one is the "!" keyword line (the "!" is optional),
# the rest is copied into every ORCA input as is:
# orca_method: |
#   ! M062X 6-31+G(d) DefGrid3
#   %scf
#     MaxIter 300
#   end
num_of_procs: 8
charge: 0
spin_multiplicity: 1
acquisition_function: "evm"   # "ei", "evm", or "ik"
num_initial_points: 5
max_steps: 100
fixed_double_bonds: # torsion with double bonds are not oprimized
  - [3, 24]
  - [9, 23]

ts_bonds: # transition state bonds, which are allowed to be stretched up to 5 Å
  - [3, 9]

extra_constraints:
  - type: bond
    atoms: [9, 17]
    value: 1.54          # value in Å

  - type: angle
    atoms: [9, 17, 18]
    value: current        # keep the start-geometry value (written to ORCA without a value)

  - type: dihedral
    atoms: [1, 2, 3, 4]
    value: 180.0          # degrees

# Optional, defaults shown:
# use_feasibility_model: true       # failed candidates teach P(feasible) instead of entering the GP
# feasibility_lengthscale: 0.5      # how far around a failure P is lowered (0.5 ~ 30 deg, 1.0 ~ 60 deg)
# feasibility_prior: 0.8            # P(feasible) far from every evaluated point
# build_from_nearest_conformer: true
# exclude_dof_slack: false          # true = old behaviour: only N - 6 torsions of each ring are GP axes
# scan_ring_torsions: true          # "ik": scan ring torsions on an open fragment (else flat GP mean)
# mmff_energy_window: 30.0          # kcal/mol above the lowest known MMFF94 energy; null = no filter
# candidate_batch_size: 5           # maxima kept per acquisition search, tried after a rejection
# ik_loss_weight: 1.0               # weight of the ring-closure loss in "ik"; 0 = plain EVM
# ts_mode_bond: []                  # 1-based TS bond followed by OptTS; empty = shortest TS bond
# ts_bond_result_margin: 0.5        # Å; 0 turns the TS-region check of the result off
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
| `<exp_name>_clustering_results.json` | Unique conformers: relative energy and minimum index |

---

## How It Works

1. **Detect rotatable bonds** in the molecule (RDKit).
2. **Pre-scan each bond** through 360° with ORCA → fit A1·cos θ + A2·cos 2θ + A3·cos 3θ + C (fixed harmonics, exactly 2π-periodic) → this becomes the GP prior mean (cached in `dihedral_logs.db`; profiles of the older free-frequency fit are not reused, so the first run scans again). In `ik` mode every rotatable ring bond is a GP axis too; it is scanned on an open fragment: the bond, its neighbours and their double-bond partners, the ring cut and the cut atoms filled with H (`scan_ring_torsions`).
3. **Initialize dataset** from random RDKit embeddings or a seed ensemble.
4. **Bayesian optimization loop** (up to `max_steps`):
   - Acquisition function selects next dihedral angles to evaluate.
   - The candidate geometry is checked before ORCA (atom clashes, stretched bonds, TS bonds outside `[ts_bond_min_length, ts_bond_max_length]`, opened rings). A rejected candidate does not count towards `max_steps`; after `max_geometry_rejections_in_a_row` rejections in a row the run stops.
   - **Feasibility model** (`use_feasibility_model: true`, default). A candidate without an energy (rejected geometry, failed ORCA, or a geometry that could not be built near the asked torsions) is not put into the GP as `broken_struct_energy`. It is labelled "failed" in a separate classifier P(feasible | torsions), and the acquisition is multiplied by P, so BO stops asking points next to known failures. ORCA failures with no output file or an unreadable one (queue, file system) get no label. Seed-ensemble conformers enter the GP only, not the feasibility model; a checkpoint from before this change resumes with no feasibility labels. Points with an energy are stored at the torsions of the geometry that was actually built. With `use_feasibility_model: false` the old behaviour (`broken_struct_energy` in the GP) is kept.
   - The start geometry is built from the nearest known conformer (seed ensemble or minima found so far, `build_from_nearest_conformer`), with the BO torsions and TS bonds held by MMFF. There is no fallback geometry: if MMFF fails, does not converge, ends more than ~30° from an asked torsion, or gives an MMFF94 energy more than `mmff_energy_window` above the lowest known structure, the candidate is rejected before ORCA, labelled in the feasibility model, and the next maximum of the same acquisition search is tried (`candidate_batch_size`).
   - In the `ik` ring-closure loss a TS bond in a ring may have any length in `[ts_bond_min_length, ts_bond_max_length]`: the ring counts as closed when the other ring bonds bring the two TS atoms within that range.
   - ORCA stages. Non-TS: constrained `Opt` (BO torsions, fixed torsions, `extra_constraints`) → unconstrained `Opt`. TS: constrained `Opt` with the TS bonds also held → `OptTS` with the BO torsions held and the TS bonds free (TS-bond `extra_constraints` are dropped here) → unconstrained `OptTS` that reads the Hessian of the previous `OptTS`. The GP gets the trajectory of the last stage (its first frame is the result of the previous stage, plus cluster representatives); only its end point can become a minimum. The last `OptTS` does not recompute the Hessian. If the TS `OptTS` with held torsions fails in ORCA, the GP gets the energy of the constrained `Opt`; if the last stage fails in ORCA or its result is broken, the GP gets the energy of the previous stage at the asked point. The final result goes through the same checks. OptTS follows the TS bond given by `ts_mode_bond` (default: the shortest TS bond), and a result whose TS bonds left the range of the seed ensemble ± `ts_bond_result_margin` counts as broken.
   - Result updates the GP model.
   - Early-stop when acquisition values plateau (rolling window criterion).
5. **Deduplicate** all discovered minima (heavy-atom RMSD + energy window) → write final ensemble sorted by energy.

**One run per `.mol` folder.** ORCA files go to `qc_calcs/` next to the `.mol`, and a run takes the lock `qc_calcs/bocser.lock`. A second run with a `.mol` in the same folder stops with an error naming the running one. A lock left by a crashed run is removed automatically on the same machine; from another machine, delete it by hand.

**Continuing a run.** `{exp_name}_checkpoint.json` is kept after the run. To add steps in the same folder, set `checkpoint_step: last`, a larger `max_steps` (e.g. 20 → 30) and `clear_working_folder: false`: the run continues from the last saved step and rewrites the final files. A run that stopped early on the convergence criterion stops again soon with the same thresholds.

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
