#!/bin/bash
# Example: Using conf_search.py with working folders
# This script demonstrates the new working folder feature

set -e

echo "════════════════════════════════════════════════════════════════"
echo "BOCSER Working Folder Feature - Example Usage"
echo "════════════════════════════════════════════════════════════════"
echo

# Create example structure
echo "📁 Creating example project structure..."
mkdir -p bocser_examples/{experiment1,experiment2,experiment3}

echo
echo "📋 Creating example configs and input files..."

# Example 1: Simple experiment
cat > bocser_examples/experiment1/config.yaml << 'EOF'
mol_file_name: molecule.mol
charge: 0
multipl: 1
num_of_procs: 4
orca_method: PM6
num_of_optimization_steps: 20
budget_limit: 100
exp_name: experiment1
EOF

# Example 2: Different parameters
cat > bocser_examples/experiment2/config.yaml << 'EOF'
mol_file_name: molecule.mol
charge: 0
multipl: 1
num_of_procs: 8
orca_method: PM6-D3H4X
num_of_optimization_steps: 50
budget_limit: 200
exp_name: experiment2
EOF

# Example 3: Another variation
cat > bocser_examples/experiment3/config.yaml << 'EOF'
mol_file_name: molecule.mol
charge: +1
multipl: 2
num_of_procs: 4
orca_method: PM6
num_of_optimization_steps: 30
budget_limit: 150
exp_name: experiment3
EOF

echo
echo "✅ Example structure created!"
echo
echo "════════════════════════════════════════════════════════════════"
echo "Usage Examples:"
echo "════════════════════════════════════════════════════════════════"
echo

echo "1️⃣  Run experiment1 with default config (config.yaml):"
echo "   python conf_search.py --folder bocser_examples/experiment1"
echo

echo "2️⃣  Run experiment2 with default config:"
echo "   python conf_search.py --folder bocser_examples/experiment2"
echo

echo "3️⃣  Run experiment3 with default config:"
echo "   python conf_search.py --folder bocser_examples/experiment3"
echo

echo "4️⃣  Run with custom config name:"
echo "   python conf_search.py --folder bocser_examples/experiment1 --config config.yaml"
echo

echo "5️⃣  Run in current directory (backward compatible):"
echo "   python conf_search.py"
echo

echo "════════════════════════════════════════════════════════════════"
echo "What happens:"
echo "════════════════════════════════════════════════════════════════"
echo

echo "Input files are read from the working folder:"
echo "  ✓ config.yaml"
echo "  ✓ molecule.mol (from config.yaml)"
echo

echo "Output files are created in the working folder:"
echo "  ✓ experiment1/ (or exp_name/ from config)"
echo "  ✓ experiment1_minima/"
echo "  ✓ experiment1_scans/"
echo "  ✓ experiment1_clustering_results.json"
echo "  ✓ experiment1_final_ensemble.xyz"
echo "  ✓ experiment1_all_points.json"
echo "  ✓ dihedral_logs.db"
echo

echo "════════════════════════════════════════════════════════════════"
echo "Directory Structure:"
echo "════════════════════════════════════════════════════════════════"
echo

tree_output=$(cat << 'EOF'
bocser_examples/
├── experiment1/
│   ├── config.yaml
│   ├── molecule.mol          (input - from config)
│   ├── experiment1/          (output - structures)
│   ├── experiment1_minima/   (output - minima)
│   ├── experiment1_scans/    (output - ORCA files)
│   ├── experiment1_clustering_results.json
│   ├── experiment1_final_ensemble.xyz
│   ├── experiment1_all_points.json
│   ├── experiment1_last_opt_status.json
│   └── dihedral_logs.db
├── experiment2/
│   ├── config.yaml
│   ├── molecule.mol
│   └── (same outputs as experiment1, prefixed with experiment2_)
└── experiment3/
    ├── config.yaml
    ├── molecule.mol
    └── (same outputs as experiment1, prefixed with experiment3_)
EOF
echo "$tree_output"
)

echo
echo "════════════════════════════════════════════════════════════════"
echo "Running Multiple Experiments (Bash Loop):"
echo "════════════════════════════════════════════════════════════════"
echo

cat << 'EOF'
#!/bin/bash
# Run multiple experiments
for exp in experiment1 experiment2 experiment3; do
    echo "Running $exp..."
    python conf_search.py --folder bocser_examples/$exp
    echo "✅ $exp completed!"
done
EOF

echo
echo "════════════════════════════════════════════════════════════════"
echo "Python API Usage:"
echo "════════════════════════════════════════════════════════════════"
echo

cat << 'EOF'
from conf_search import ConfSearchRunner

# Run experiment 1
runner1 = ConfSearchRunner(working_folder="bocser_examples/experiment1")
runner1.load_config("config.yaml")
runner1.setup()
runner1.run()

# Run experiment 2
runner2 = ConfSearchRunner(working_folder="bocser_examples/experiment2")
runner2.load_config("config.yaml")
runner2.setup()
runner2.run()

# Access results
print("Experiment 1 results:")
print(runner1.state.minima)

print("Experiment 2 results:")
print(runner2.state.minima)
EOF

echo
echo "════════════════════════════════════════════════════════════════"
echo "Features:"
echo "════════════════════════════════════════════════════════════════"
echo

echo "✨ Benefits of working folders:"
echo "  ✓ Organize multiple runs in separate directories"
echo "  ✓ Isolate input and output files"
echo "  ✓ Run experiments in parallel (different folders)"
echo "  ✓ Easy cleanup (delete folder to remove all results)"
echo "  ✓ Portable experiments (move folder to another location)"
echo "  ✓ Backward compatible (works without --folder argument)"
echo

echo "════════════════════════════════════════════════════════════════"
echo "Quick Start:"
echo "════════════════════════════════════════════════════════════════"
echo

cat << 'EOF'
# 1. Prepare your experiment
mkdir my_experiment
cd my_experiment
cp /path/to/config.yaml .
cp /path/to/molecule.mol .

# 2. Run conformational search
cd ..
python conf_search.py --folder my_experiment

# 3. Check results
ls my_experiment/experiment*

# Results are all in my_experiment/
EOF

echo
echo "✅ All examples created in: bocser_examples/"
echo
echo "To run the actual optimization, you need:"
echo "  ✓ RDKit installed"
echo "  ✓ TensorFlow/GPflow/trieste"
echo "  ✓ ORCA quantum chemistry software (if using calc_energy)"
echo "  ✓ A valid MOL file"
echo
