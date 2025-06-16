# CLI Arguments Integration

The Inception-CRO project now supports command line arguments to override configuration parameters from `configs/default_config.py`. This allows for flexible experimentation without modifying configuration files.

## Quick Start

```bash
# Show all available options
python main.py --help

# Basic run with custom parameters
python main.py --dataset-name mnist --batch-size 64 --max-generations 15

# Quick test with minimal parameters
python main.py --reef-size "(2, 2)" --max-generations 3 --num-epochs 2
```

## Available CLI Arguments

### General Parameters
- `--seed`: Random seed for reproducibility (default: 42)
- `--experiments-dir`: Base directory for experiments (default: "experiments")

### Dataset Parameters
- `--dataset-name`: Choose between "mnist" or "medmnist" (default: "medmnist")
- `--medmnist-subset`: MedMNIST subset when using medmnist dataset (default: "chestmnist")
- `--task-type`: Task type - "auto", "multi-class", or "multi-label" (default: "auto")

### Data Loader Parameters
- `--batch-size`: Batch size for training (default: 128)
- `--validation-split`: Fraction of training data for validation (default: 0.1)
- `--shuffle-dataset` / `--no-shuffle-dataset`: Enable/disable dataset shuffling (default: enabled)

### Training Parameters
- `--learning-rate`: Learning rate for training (default: 0.001)
- `--num-batches`: Number of batches for partial training (default: 50)
- `--num-epochs`: Number of epochs for full training (default: 10)
- `--patience`: Early stopping patience (default: 3)

### CRO Hyperparameters
- `--reef-size`: Reef size as tuple, e.g., "(3, 3)" (default: (2, 2))
- `--rho-0`: Initial reef occupation probability (default: 0.6)
- `--fb`: Broadcast spawning probability (default: 0.98)
- `--fa`: Asexual reproduction probability (default: 0.05)
- `--pa`: Asexual reproduction attempt probability (default: 0.001)
- `--fd`: Depredation probability (default: 0.05)
- `--pd`: Depredation attempt probability (default: 0.01)
- `--kappa`: Number of attempts for larvae settlement (default: 3)
- `--mutation-rate`: Mutation rate for genetic operations (default: 0.2)
- `--max-generations`: Maximum number of generations (default: 10)
- `--max-no-improve`: Max generations without improvement before stopping (default: 5)

### Inception Architecture Parameters
- `--branch-min`: Minimum number of branches in Inception modules (default: 1)
- `--branch-max`: Maximum number of branches in Inception modules (default: 4)

### Fitness Parameters
- `--fitness-method`: Fitness computation method - "linear" or "poly" (default: "linear")
- `--fitness-alpha`: Alpha parameter for fitness computation (default: 7)
- `--fitness-beta`: Beta parameter for fitness computation (default: 1.0)

## Usage Examples

### Basic Dataset Switching
```bash
# Use MNIST instead of MedMNIST
python main.py --dataset-name mnist

# Use different MedMNIST subset
python main.py --dataset-name medmnist --medmnist-subset pathmnist
```

### Training Configuration
```bash
# Longer training with higher learning rate
python main.py --learning-rate 0.01 --num-epochs 20 --patience 5

# Faster training with larger batches
python main.py --batch-size 256 --num-epochs 5
```

### CRO Optimization Settings
```bash
# Larger reef with more generations
python main.py --reef-size "(4, 4)" --max-generations 25

# Higher mutation rate for more exploration
python main.py --mutation-rate 0.3 --max-no-improve 8
```

### Architecture Search Space
```bash
# Allow more branches in Inception modules
python main.py --branch-min 2 --branch-max 6

# Constrain to simpler architectures
python main.py --branch-min 1 --branch-max 2
```

### Fitness Function Tuning
```bash
# Use polynomial fitness with custom parameters
python main.py --fitness-method poly --fitness-alpha 10 --fitness-beta 0.5
```

### Comprehensive Example
```bash
python main.py \
    --dataset-name medmnist \
    --medmnist-subset chestmnist \
    --batch-size 64 \
    --learning-rate 0.005 \
    --num-epochs 15 \
    --reef-size "(4, 4)" \
    --max-generations 25 \
    --mutation-rate 0.25 \
    --branch-min 2 \
    --branch-max 5 \
    --fitness-method poly \
    --seed 123
```

## Configuration Validation

The system includes automatic validation of configuration parameters:
- Reef size must be a tuple of 2 integers
- Branch limits: min ≤ max and min ≥ 1
- Probabilities must be between 0 and 1
- Validation split must be between 0 and 1
- All numeric parameters must be positive

Invalid configurations will be caught with descriptive error messages.

## Implementation Details

### How It Works
1. **Argument Parsing**: Uses `argparse` to define all CLI options with proper types and help text
2. **Config Override**: Updates the global `CONFIG` dictionary with CLI values
3. **Validation**: Validates all parameters for consistency and correctness
4. **Configuration Summary**: Displays active configuration before execution

### Key Functions
- `parse_args()`: Defines and parses all CLI arguments
- `update_config_from_args()`: Updates CONFIG with CLI values
- `validate_config()`: Validates configuration consistency

### Special Handling
- **Tuple Parameters**: `reef_size` accepts string tuples like "(3, 3)" and parses them safely
- **Boolean Flags**: `shuffle_dataset` supports both `--shuffle-dataset` and `--no-shuffle-dataset`
- **Type Safety**: All parameters are validated for correct types and ranges

## Running Examples Script

A helper script is provided to show usage examples:

```bash
python cli_examples.py
```

This script demonstrates various CLI usage patterns without actually running the optimization.

## Default Configuration

All CLI arguments default to values from `configs/default_config.py`. CLI arguments only override specific parameters, leaving others unchanged. This ensures backward compatibility with existing configurations.

