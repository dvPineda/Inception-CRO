# Training Guide

This guide covers everything you need to know about training Inception-CRO models, from basic usage to advanced configuration.

## Quick Start

### Basic Training
```bash
# Train on MNIST with default settings
python main.py --dataset-name mnist

# Train on MedMNIST chest X-rays
python main.py --dataset-name medmnist --medmnist-subset chestmnist
```

### View All Options
```bash
python main.py --help
```

## Configuration Options

### Dataset Parameters
- `--dataset-name`: Choose between "mnist" or "medmnist" (default: "medmnist")
- `--medmnist-subset`: MedMNIST subset when using medmnist dataset (default: "chestmnist")
- `--task-type`: Task type - "auto", "multi-class", or "multi-label" (default: "auto")

### Training Parameters
- `--batch-size`: Batch size for training (default: 128)
- `--learning-rate`: Learning rate for training (default: 0.001)
- `--num-epochs`: Number of epochs for full training (default: 10)
- `--patience`: Early stopping patience (default: 3)

### CRO Optimization
- `--reef-size`: Reef size as tuple, e.g., "(3, 3)" (default: (2, 2))
- `--max-generations`: Maximum number of generations (default: 10)
- `--mutation-rate`: Mutation rate for genetic operations (default: 0.2)
- `--max-no-improve`: Max generations without improvement before stopping (default: 5)

### Architecture Parameters
- `--branch-min`: Minimum number of branches in Inception modules (default: 1)
- `--branch-max`: Maximum number of branches in Inception modules (default: 4)

### Fitness Function
- `--fitness-method`: Fitness computation method - "linear" or "poly" (default: "linear")
- `--fitness-alpha`: Alpha parameter for fitness computation (default: 7)
- `--fitness-beta`: Beta parameter for fitness computation (default: 1.0)

## Example Training Configurations

### Quick Testing
```bash
python main.py --reef-size "(2, 2)" --max-generations 3 --num-epochs 2
```

### Thorough Training
```bash
python main.py \
    --dataset-name medmnist \
    --medmnist-subset chestmnist \
    --reef-size "(4,4)" \
    --max-generations 25 \
    --learning-rate 0.005 \
    --num-epochs 20 \
    --mutation-rate 0.25
```

### Architecture Search Focus
```bash
python main.py \
    --branch-min 2 \
    --branch-max 6 \
    --reef-size "(3,3)" \
    --max-generations 30
```

## Understanding Training Output

### Console Output
During training, you'll see:
- **Generation Progress**: Current generation and fitness scores
- **Best Individual**: Architecture and performance of best solution
- **Training Metrics**: Loss and accuracy during neural network training
- **Early Stopping**: Notifications when training stops early

### Generated Files
Training creates a structured experiment directory:
```
experiments/YYYY-MM-DD/config_group/exp_HHMMSS_seedN/
├── results.csv              # Experiment results
├── config_used.json         # Complete configuration
├── plots/                   # Training visualizations
├── checkpoints/             # Model checkpoints
└── visualizations/          # Architecture visualizations
```

## Tips for Effective Training

### For Quick Prototyping
- Use small reef sizes: `--reef-size "(2,2)"`
- Limit generations: `--max-generations 5`
- Reduce epochs: `--num-epochs 3`

### For Research Results
- Use larger reefs: `--reef-size "(4,4)" or "(5,5)"`
- More generations: `--max-generations 25-50`
- Sufficient epochs: `--num-epochs 15-25`
- Multiple seeds for statistical significance

### For Resource-Constrained Environments
- Smaller batch sizes: `--batch-size 32`
- Conservative architecture search: `--branch-max 3`
- Early stopping: `--patience 3`

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce batch size: `--batch-size 32`
- Limit architecture complexity: `--branch-max 2`
- Use smaller reef: `--reef-size "(2,2)"`

**Slow Training**
- Increase batch size: `--batch-size 256`
- Reduce epochs for fitness evaluation: `--num-batches 25`
- Use GPU if available

**Poor Convergence**
- Increase mutation rate: `--mutation-rate 0.3`
- Extend generations: `--max-generations 30`
- Adjust learning rate: `--learning-rate 0.01`

**Architecture Search Not Exploring**
- Increase reef size: `--reef-size "(4,4)"`
- Higher mutation rate: `--mutation-rate 0.4`
- Wider branch range: `--branch-min 1 --branch-max 6`

## Advanced Configuration

### Fitness Function Tuning
The fitness function balances accuracy and efficiency:

```bash
# Emphasize accuracy over efficiency
python main.py --fitness-method poly --fitness-alpha 10 --fitness-beta 0.1

# Balance accuracy and efficiency
python main.py --fitness-method linear --fitness-alpha 7 --fitness-beta 1.0

# Emphasize efficiency over accuracy
python main.py --fitness-method poly --fitness-alpha 5 --fitness-beta 2.0
```

### Custom Datasets
To add support for new datasets:
1. Implement dataset loading in `src/data_loading.py`
2. Add dataset configuration to `configs/default_config.py`
3. Update CLI arguments in `main.py`

### Reproducibility
For reproducible results:
```bash
python main.py --seed 42 --dataset-name mnist
```

## Performance Monitoring

### Real-time Monitoring
- Monitor GPU usage: `nvidia-smi -l 1`
- Track training progress through console output
- Check experiment directory for intermediate results

### Post-training Analysis
```bash
# Analyze results across experiments
python experiment_analyzer.py

# View specific experiment
ls experiments/YYYY-MM-DD/
```

## Next Steps

After training:
1. **Analyze Results**: Use `experiment_analyzer.py` to compare experiments
2. **Compare Models**: Run `python scripts/run_comprehensive_experiments.py`
3. **Research Evaluation**: Use the comprehensive evaluation framework
4. **Deployment**: Export best models for production use

