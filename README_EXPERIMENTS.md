# Experiment Organization Structure

The Inception-CRO project now uses an organized, hierarchical experiment structure that groups experiments by date and then by key parameter configurations. This makes it much easier to find, compare, and analyze experimental results.

## New Structure Overview

```
experiments/
├── YYYY-MM-DD/                          # Date-based grouping
│   ├── dataset_reefNxN_genN_lrXe-X/     # Parameter-based grouping
│   │   ├── exp_HHMMSS_seedN/             # Individual experiment
│   │   │   ├── plots/
│   │   │   │   └── exp_HHMMSS_seedN_fitness.png
│   │   │   ├── checkpoints/
│   │   │   │   └── best_model.pth
│   │   │   ├── visualizations/
│   │   │   │   └── best_coral/
│   │   │   ├── results.csv
│   │   │   └── config_used.json          # Complete configuration backup
│   │   └── exp_HHMMSS_seedM/             # Another experiment with same params
│   └── dataset_reefNxN_genM_lrYe-Y/      # Different parameter group
└── YYYY-MM-DD+1/                        # Next day's experiments
```

## Example Structure

```
experiments/
├── 2025-06-03/
│   ├── medmnist-chestmnist_reef2x2_gen10_lr1e-03/
│   │   ├── exp_140530_seed42/
│   │   ├── exp_141205_seed123/
│   │   └── exp_142315_seed999/
│   ├── medmnist-chestmnist_reef3x3_gen20_lr1e-03/
│   │   └── exp_150445_seed42/
│   └── mnist_reef2x2_gen15_lr5e-03/
│       └── exp_160122_seed42/
└── 2025-06-04/
    └── medmnist-pathmnist_reef4x4_gen25_lr1e-02_poly/
        └── exp_091234_seed777/
```

## Parameter Group Naming Convention

Parameter groups are automatically named based on key differentiating factors:

### Base Format
```
{dataset}_{reef_size}_gen{max_generations}_lr{learning_rate}
```

### Extended Format (when non-default)
```
{base}_{fitness_method}_{branch_range}_{mutation_rate}
```

### Examples
- `medmnist-chestmnist_reef2x2_gen10_lr1e-03` - Default parameters
- `mnist_reef3x3_gen20_lr5e-03_poly` - Polynomial fitness method
- `medmnist-pathmnist_reef2x2_gen15_lr1e-03_br2-6` - Custom branch range (2-6)
- `medmnist-chestmnist_reef4x4_gen25_lr1e-02_mut0.30` - Custom mutation rate
- `mnist_reef2x2_gen10_lr1e-03_poly_br1-2_mut0.15` - Multiple custom parameters

## Benefits of New Structure

### ✅ **Easy Navigation**
- Find experiments by date: "What did I run yesterday?"
- Compare similar configurations: "How do different reef sizes perform?"
- Group analysis: "Compare all polynomial fitness experiments"

### ✅ **Automatic Organization**
- No manual folder creation needed
- Consistent naming conventions
- Parameter-based grouping prevents confusion

### ✅ **Enhanced Analysis**
- Built-in experiment analyzer tool
- Automated visualization generation
- Easy batch comparison of parameter groups

### ✅ **Reproducibility**
- Complete configuration backup in each experiment
- Experiment metadata tracking
- Migration support for old experiments

## Key Files in Each Experiment

### `config_used.json`
```json
{
  "dataset_name": "medmnist",
  "medmnist_subset": "chestmnist",
  "reef_size": [2, 2],
  "max_generations": 10,
  "learning_rate": 0.001,
  "fitness_method": "linear",
  "seed": 42,
  "experiment_metadata": {
    "created_at": "2025-06-03T05:14:05",
    "experiment_structure_version": "2.0",
    "description": "Organized experiment structure with date and parameter grouping"
  }
}
```

### `results.csv`
```csv
experiment,dataset,subset,task_type,best_fitness,test_fitness,test_accuracy,param_count,branch_min,branch_max,mutation_rate,max_generations,num_epochs,batch_size,learning_rate
exp_051405_seed42,medmnist,chestmnist,multi-label,0.95,0.947,94.74,12840,1,4,0.2,2,1,32,0.001
```

## Using the New Structure

### 1. Running Experiments
The new structure is automatically used when running experiments:

```bash
# Default run - creates organized structure automatically
python main.py

# Custom parameters - grouped by configuration
python main.py --reef-size "(3,3)" --max-generations 20 --learning-rate 0.01
```

### 2. Analyzing Experiments
Use the experiment analyzer to explore results:

```bash
# List all experiments
python experiment_analyzer.py --list-only

# Full analysis with visualizations
python experiment_analyzer.py

# Analyze only recent experiments
python experiment_analyzer.py --days-back 7
```

### 3. Migrating Old Experiments
Migrate experiments from the old flat structure:

```bash
# Preview migration (dry run)
python migrate_experiments.py --dry-run

# Perform migration with backup
python migrate_experiments.py --backup
```

## Analysis Tools

### Experiment Analyzer (`experiment_analyzer.py`)

**Features:**
- Automatic experiment discovery
- Parameter group analysis
- Statistical summaries
- Visualization generation
- Export combined results

**Generated Visualizations:**
- Test accuracy by parameter group (boxplots)
- Accuracy trends over time
- Accuracy vs model complexity
- Dataset comparison charts

**Usage Examples:**
```bash
# Basic analysis
python experiment_analyzer.py

# Custom output directory
python experiment_analyzer.py --output-dir my_analysis

# Analyze last 3 days only
python experiment_analyzer.py --days-back 3

# Just list experiments
python experiment_analyzer.py --list-only
```

### Migration Tool (`migrate_experiments.py`)

**Features:**
- Automatic old experiment detection
- Safe migration with dry-run option
- Backup creation before migration
- Configuration inference for old experiments
- Progress tracking

**Usage Examples:**
```bash
# Preview what will be migrated
python migrate_experiments.py --dry-run

# Migrate with automatic backup
python migrate_experiments.py --backup

# Custom experiments directory
python migrate_experiments.py --experiments-dir /path/to/experiments
```

## Configuration Management

### Automatic Config Backup
Every experiment automatically saves its complete configuration to `config_used.json`, including:
- All CLI arguments used
- Default values for unspecified parameters
- Experiment metadata (timestamp, version, etc.)
- Migration information (if migrated from old structure)

### Version Tracking
The structure includes version information to handle future changes:
```json
{
  "experiment_metadata": {
    "experiment_structure_version": "2.0",
    "created_at": "2025-06-03T05:14:05",
    "description": "Organized experiment structure with date and parameter grouping"
  }
}
```

## Best Practices

### 🎯 **Parameter Exploration**
```bash
# Compare reef sizes
python main.py --reef-size "(2,2)" --seed 42
python main.py --reef-size "(3,3)" --seed 42
python main.py --reef-size "(4,4)" --seed 42

# Compare fitness methods
python main.py --fitness-method linear --seed 42
python main.py --fitness-method poly --seed 42
```

### 📊 **Regular Analysis**
```bash
# Weekly analysis
python experiment_analyzer.py --days-back 7

# Generate monthly report
python experiment_analyzer.py --days-back 30 --output-dir monthly_report
```

### 🔄 **Reproducible Experiments**
```bash
# Multiple seeds for same configuration
for seed in 42 123 999; do
    python main.py --reef-size "(3,3)" --max-generations 20 --seed $seed
done
```

## Troubleshooting

### Common Issues

**Q: Old experiments not showing in analyzer**
A: Use the migration tool to move them to the new structure:
```bash
python migrate_experiments.py --backup
```

**Q: Parameter group names too long**
A: The system automatically abbreviates when necessary, but you can identify experiments by their unique timestamp and seed.

**Q: Missing visualization dependencies**
A: The analyzer works without seaborn, using basic matplotlib as fallback.

### Directory Structure Validation
The system automatically creates all necessary directories and validates the structure on each run.

## Advanced Usage

### Custom Analysis Scripts
You can create custom analysis scripts using the experiment discovery functions:

```python
from experiment_analyzer import find_experiments, load_experiment_results

# Find experiments programmatically
experiments = find_experiments("experiments", days_back=7)
df = load_experiment_results(experiments)

# Custom analysis
best_configs = df.groupby('param_group')['test_accuracy'].max()
print(best_configs.sort_values(ascending=False))
```

### Batch Processing
```bash
# Run parameter sweep
for reef in "(2,2)" "(3,3)" "(4,4)"; do
    for gen in 10 20 30; do
        python main.py --reef-size "$reef" --max-generations $gen
    done
done
```

This organized structure makes experiment management much more efficient and enables powerful analysis capabilities for your research.

