# Running conf_search.py with Working Folders

## Overview

The refactored `conf_search.py` now supports specifying a working folder where all input files are read from and output files are written to. This makes it easy to organize multiple conformational search runs in separate directories.

## Usage

### Basic Usage (Current Directory)

If your config and input files are in the current directory:

```bash
python conf_search.py
# or explicitly:
python conf_search.py --config config.yaml
```

### Using a Specific Working Folder

To use a different working folder:

```bash
python conf_search.py --folder /path/to/working/folder
```

This will:
- Look for `config.yaml` in `/path/to/working/folder/`
- Read input files (e.g., `molecule.mol`) from `/path/to/working/folder/`
- Write all output files to `/path/to/working/folder/`

### Custom Config File in Working Folder

```bash
python conf_search.py --folder ./experiment1 --config my_config.yaml
```

This will:
- Look for `my_config.yaml` in `./experiment1/`
- Read input files from `./experiment1/`
- Write output to `./experiment1/`

### Using Absolute Path for Config

```bash
python conf_search.py --folder ./results --config /absolute/path/to/config.yaml
```

If you provide an absolute path to config, it will be used as-is (not relative to `--folder`).

## Directory Structure Example

### Setup

```
projects/
├── experiment1/
│   ├── config.yaml
│   └── molecule.mol
├── experiment2/
│   ├── config.yaml
│   └── molecule.mol
└── bocser/
    ├── conf_search.py
    └── ... other modules
```

### Usage

```bash
# Run experiment 1
cd bocser
python conf_search.py --folder ../experiment1

# Run experiment 2
python conf_search.py --folder ../experiment2
```

### Results

After running, each experiment folder will contain:

```
experiment1/
├── config.yaml
├── molecule.mol
├── experiment1/                    # structures
│   ├── 0.xyz
│   ├── 1.xyz
│   └── ...
├── experiment1_minima/             # minima structures
│   ├── 0.xyz
│   ├── 1.xyz
│   └── ...
├── experiment1_scans/              # ORCA scan files
│   ├── 0.inp
│   ├── 0.out
│   └── ...
├── experiment1_clustering_results.json
├── experiment1_final_ensemble.xyz
├── experiment1_all_points.json
├── experiment1_last_opt_status.json
└── dihedral_logs.db
```

## Command-Line Arguments

### `--folder` (optional)
- **Type**: string
- **Default**: `.` (current directory)
- **Description**: Working folder for input files and output results

### `--config` (optional)
- **Type**: string
- **Default**: `config.yaml`
- **Description**: Config file name (relative to `--folder` or absolute path)

## Environment Setup Example

Here's a complete workflow example:

```bash
# Create project structure
mkdir -p conformer_search/{exp1,exp2,code}
cd conformer_search

# Prepare experiment 1
cd exp1
# Copy your config.yaml and molecule.mol here
cd ..

# Prepare experiment 2
cd exp2
# Copy your config.yaml and molecule.mol here
cd ..

# Copy or clone bocser code
cd code
# git clone ... or copy files

# Run experiments
cd ..
python code/conf_search.py --folder exp1 --config config.yaml
python code/conf_search.py --folder exp2 --config config.yaml

# Check results
ls exp1/exp1_final_ensemble.xyz
ls exp2/exp2_final_ensemble.xyz
```

## Important Notes

1. **Relative Paths in Config**: If your `config.yaml` contains relative paths (e.g., `mol_file_name: molecule.mol`), they will be resolved relative to the working folder.

2. **Working Folder Creation**: If the working folder doesn't exist, it will be created automatically.

3. **Output Structure**: All output files are created inside the specified working folder, maintaining the same structure as before:
   - `{exp_name}/` - structure outputs
   - `{exp_name}_minima/` - minima structures
   - `{exp_name}_scans/` - ORCA input/output files
   - `{exp_name}_*` - result files (clustering, ensemble, all_points, etc.)

4. **Backward Compatibility**: Running without `--folder` argument works exactly as before (uses current directory).

## Programmatic Usage

You can also use the new feature in Python code:

```python
from conf_search import ConfSearchRunner

# Use custom working folder
runner = ConfSearchRunner(working_folder="./my_experiment")
runner.load_config("config.yaml")  # Looks in ./my_experiment/
runner.setup()
runner.run()
```

## Troubleshooting

### Error: "No config file config.yaml!"
- Check that your config file exists in the working folder
- Use absolute path: `--config /path/to/config.yaml`
- Check spelling and file extension

### Error: "No module file..."
- Ensure input files (e.g., `molecule.mol`) are in the working folder
- Use relative paths in `config.yaml` (e.g., `mol_file_name: molecule.mol`)

### Output files not found
- Check that the working folder was created correctly
- Look for files in `{working_folder}/{exp_name}_*` paths
- Check file permissions

---

**Examples:**

```bash
# Simple - current directory
python conf_search.py

# Specify folder
python conf_search.py --folder ./my_experiment

# Specify both folder and config
python conf_search.py --folder ./exp1 --config settings.yaml

# With absolute paths
python conf_search.py --folder /home/user/conformer_exp --config /home/user/settings.yaml
```

---

**For more information**, see:
- [QUICKREF.md](QUICKREF.md) for general usage
- [ARCHITECTURE.md](ARCHITECTURE.md) for technical details
