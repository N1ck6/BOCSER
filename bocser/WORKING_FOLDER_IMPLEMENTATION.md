# Working Folder Feature - Implementation Summary

## Overview

The `conf_search.py` module has been enhanced to support a working folder feature. This allows users to specify a directory where:
- **Input files** are read from (config.yaml, molecule.mol, etc.)
- **Output files** are written to (results, databases, etc.)

## Changes Made

### 1. **ConfSearchState Enhancement**
Added `working_folder` field to track the active working directory:
```python
@dataclass
class ConfSearchState:
    working_folder: str = ""
    # ... other fields
```

### 2. **ConfSearchRunner Constructor**
Updated to accept optional `working_folder` parameter:
```python
def __init__(self, working_folder: str = "."):
    self.state = ConfSearchState()
    self.state.working_folder = working_folder
    Path(working_folder).mkdir(parents=True, exist_ok=True)
```

### 3. **Path Resolution in Key Methods**

#### `load_config(config_path)`
- Resolves relative config paths to working_folder
- Supports absolute paths (unchanged)

#### `setup()`
- Resolves mol_file_name to working_folder
- Creates output directories in working_folder:
  - `{exp_name}/`
  - `{exp_name}_minima/`
  - `{exp_name}_scans/`
- Passes correct paths to CoefCalculator and LocalConnector

#### `_dump_status_hook()`
- Saves status JSON files to working_folder

#### `_save_results()`
- Reads/writes all result files from/to working_folder:
  - clustering_results.json
  - final_ensemble.xyz
  - all_points.json

#### `_upd_dataset_from_trj()`
- Writes minima files to working_folder

### 4. **Command-Line Interface (CLI)**
Enhanced `main()` function with new arguments:

```bash
# New arguments
python conf_search.py --folder /path/to/folder --config config.yaml

# Legacy usage still works
python conf_search.py  # Uses current directory
```

Argument details:
- `--folder`: Working folder path (default: `.`)
- `--config`: Config filename (default: `config.yaml`)

### 5. **Import Addition**
Added `Path` import for directory creation:
```python
from pathlib import Path
```

## Usage Examples

### Basic Usage
```bash
# Run in current directory (backward compatible)
python conf_search.py

# Use specific working folder
python conf_search.py --folder ./experiment1

# Use custom config name
python conf_search.py --folder ./results --config settings.yaml
```

### Python API
```python
from conf_search import ConfSearchRunner

runner = ConfSearchRunner(working_folder="./my_experiment")
runner.load_config("config.yaml")
runner.setup()
runner.run()
```

### Multiple Experiments
```bash
# Run several experiments in parallel
python conf_search.py --folder exp1 &
python conf_search.py --folder exp2 &
python conf_search.py --folder exp3 &
wait
```

## File Structure

### Before (Current Directory)
```
.
├── config.yaml
├── molecule.mol
├── experiment1/          (output)
├── experiment1_minima/   (output)
├── experiment1_scans/    (output)
└── result files          (output)
```

### After (With Working Folder)
```
experiment1/
├── config.yaml          (input)
├── molecule.mol         (input)
├── experiment1/         (output)
├── experiment1_minima/  (output)
├── experiment1_scans/   (output)
└── result files         (output)
```

## Key Features

✅ **Isolated Experiments**
- Each experiment in its own folder
- Easy to organize and maintain

✅ **Portable**
- Move folder to different location
- Works with both relative and absolute paths

✅ **Parallel Execution**
- Run multiple experiments simultaneously
- No file conflicts (each in own folder)

✅ **Backward Compatible**
- Works without `--folder` argument
- Existing scripts continue to work

✅ **Clean Separation**
- All inputs in working folder
- All outputs in working folder
- Easy cleanup (delete folder)

## Implementation Details

### Path Resolution Strategy
1. **Config file**: 
   - Relative path → look in working_folder
   - Absolute path → use as-is

2. **Input files** (e.g., mol_file):
   - Relative path in config → resolve to working_folder
   - Absolute path → use as-is

3. **Output files**:
   - Always written to working_folder
   - Uses `os.path.join()` for cross-platform compatibility

### Directory Creation
- Working folder auto-created if missing
- All output subdirectories created in setup()
- Uses `Path.mkdir(parents=True, exist_ok=True)`

## Testing

### Basic Test
```bash
# Create test structure
mkdir test_exp
echo "mol_file_name: molecule.mol" > test_exp/config.yaml

# Run with working folder
python conf_search.py --folder test_exp

# Verify structure
ls test_exp/
# Should show: molecule.mol, config.yaml, experiment_*, dihedral_logs.db, etc.
```

### Multiple Folders Test
```bash
for i in {1..3}; do
    mkdir exp$i
    cp config.yaml exp$i/
    cp molecule.mol exp$i/
    python conf_search.py --folder exp$i &
done
wait
```

## Files Modified

### conf_search.py
1. Added `Path` import
2. Updated `ConfSearchState` dataclass
3. Modified `ConfSearchRunner.__init__()` 
4. Updated `load_config()` method
5. Modified `setup()` method
6. Updated `_dump_status_hook()` method
7. Modified `_save_results()` method
8. Updated `_upd_dataset_from_trj()` method
9. Enhanced `main()` function with CLI arguments

## Documentation

### New Files Created
- `WORKING_FOLDER_USAGE.md` - Comprehensive usage guide
- `working_folder_examples.sh` - Example bash script

### Updated Files
- `README.md` - Updated with new feature info
- `QUICKREF.md` - Added usage examples
- `ARCHITECTURE.md` - Documented path resolution strategy

## Backward Compatibility

✅ **100% Compatible**
- Existing code works without changes
- Default behavior unchanged (current directory)
- All arguments optional

### Old Code
```python
runner = ConfSearchRunner()
runner.load_config("config.yaml")
runner.setup()
runner.run()
# Still works exactly the same
```

### New Code
```python
runner = ConfSearchRunner(working_folder="./my_exp")
runner.load_config("config.yaml")  # Looks in ./my_exp/
runner.setup()
runner.run()
```

## Error Handling

### Handled Cases
- Missing working folder → auto-created
- Relative config path → resolved to working_folder
- Absolute config path → used as-is
- Missing input files → proper error messages

### Error Messages
```
No config file /path/to/config.yaml!
No module file /path/to/molecule.mol!
```

## Performance Impact

✅ **Minimal**
- Path operations using os.path (efficient)
- Directory creation only happens once (setup)
- No additional file I/O

## Use Cases

### 1. **Parameter Sweeps**
```bash
for method in PM6 PM6-D3 PM6-D3H4; do
    mkdir exp_$method
    sed "s/PM6/$method/" config_template.yaml > exp_$method/config.yaml
    python conf_search.py --folder exp_$method &
done
wait
```

### 2. **Multiple Molecules**
```bash
for mol in molecules/*.mol; do
    dir=$(basename $mol .mol)
    mkdir $dir
    cp $mol $dir/molecule.mol
    cp config.yaml $dir/
    python conf_search.py --folder $dir &
done
wait
```

### 3. **Production Runs**
```bash
# Organized by date and time
mkdir -p results/$(date +%Y%m%d/%H%M%S)
python conf_search.py --folder results/$(date +%Y%m%d/%H%M%S)
```

### 4. **Reproducible Science**
```bash
# All experiment data in one place
experiment_folder/
├── config.yaml
├── molecule.mol
├── README.md (experiment notes)
└── (all results)

# Share entire folder for reproducibility
```

## Summary

The working folder feature makes `conf_search.py` more user-friendly and flexible:

| Aspect | Benefit |
|--------|---------|
| **Organization** | Multiple experiments in separate folders |
| **Portability** | Move experiments between machines/locations |
| **Parallel Execution** | Run multiple experiments without conflicts |
| **Isolation** | Clean separation of input/output |
| **Cleanup** | Delete folder to remove all results |
| **Compatibility** | 100% backward compatible |

---

**Status**: ✅ **Complete and Tested**

All path handling updated to support working folders while maintaining backward compatibility.
