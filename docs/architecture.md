# Architecture Guide

This guide explains the architecture and design principles of the Inception-CRO system.

## Overview

Inception-CRO combines two key components:
1. **Inception Modules**: Flexible, multi-branch convolutional architectures
2. **Coral Reef Optimization (CRO)**: Metaheuristic optimization algorithm

## System Architecture

```
Inception-CRO System
├── 🧬 Coral Reef Optimization (CRO)
│   ├── Population Management
│   ├── Genetic Operations (Mutation, Crossover)
│   ├── Selection & Survival
│   └── Fitness Evaluation
├── 🏗️ Inception Module Generator
│   ├── Branch Configuration
│   ├── Layer Generation
│   └── Architecture Validation
├── 🎯 Training Framework
│   ├── Partial Training (Fitness)
│   ├── Full Training (Final)
│   └── Evaluation Metrics
└── 📊 Evaluation System
    ├── Performance Metrics
    ├── Efficiency Analysis
    └── Statistical Testing
```

## Core Components

### 1. Coral Reef Optimization (CRO)

**File**: `src/cro.py`

CRO is a metaheuristic optimization algorithm inspired by coral reef ecosystems. It evolves a population of solutions (coral architectures) through natural processes:

#### Key Parameters
- **Reef Size**: 2D grid dimensions (e.g., 3×3, 4×4)
- **Population Density**: Initial occupation probability (ρ₀)
- **Reproduction Rates**: 
  - Broadcast spawning (Fb): High probability for successful corals
  - Asexual reproduction (Fa): Lower probability for mutation
- **Survival Factors**:
  - Depredation (Fd, Pd): Removes poor performers
  - Settlement attempts (κ): Tries to place new larvae

#### CRO Process
1. **Initialization**: Randomly populate reef with coral architectures
2. **Reproduction**: Generate offspring through crossover and mutation
3. **Competition**: Larvae compete for settlement spots
4. **Selection**: Best architectures survive and reproduce
5. **Iteration**: Repeat until convergence or max generations

### 2. Inception Module Architecture

**File**: `src/models.py`

Inception modules use parallel branches with different filter sizes to capture multi-scale features:

```
Inception Module
├── Branch 1: 1×1 Conv
├── Branch 2: 1×1 Conv → 3×3 Conv
├── Branch 3: 1×1 Conv → 5×5 Conv
├── Branch 4: 3×3 MaxPool → 1×1 Conv
└── Concatenation → Output
```

#### Configurable Parameters
- **Number of Branches**: 1-6 parallel paths
- **Filter Sizes**: 1×1, 3×3, 5×5, 7×7
- **Channel Dimensions**: Adaptive based on input
- **Activation Functions**: ReLU, Batch Normalization

#### Architecture Encoding
Each architecture is encoded as a parameter vector:
```python
{
    'num_branches': 4,
    'branch_configs': [
        {'filters': [32], 'kernel_sizes': [1]},
        {'filters': [16, 32], 'kernel_sizes': [1, 3]},
        {'filters': [8, 16], 'kernel_sizes': [1, 5]},
        {'filters': [32], 'kernel_sizes': [1], 'pooling': True}
    ]
}
```

### 3. Fitness Function

**File**: `src/fitness_utils.py`

The fitness function balances accuracy and efficiency:

#### Linear Fitness
```
fitness = α × accuracy - β × (parameters / 1M)
```

#### Polynomial Fitness
```
fitness = (α × accuracy)^β / (parameters / 1M)
```

#### Parameters
- **α (alpha)**: Accuracy weight (default: 7)
- **β (beta)**: Efficiency weight (default: 1.0)
- **Method**: 'linear' or 'poly'

### 4. Training Framework

**File**: `src/trainer.py`

#### Two-Stage Training

1. **Partial Training** (Fitness Evaluation)
   - Limited epochs/batches for speed
   - Quick assessment of architecture potential
   - Used during CRO evolution

2. **Full Training** (Final Model)
   - Complete training with early stopping
   - Best architecture from CRO
   - Production-ready model

#### Training Configuration
- **Optimizer**: Adam with adaptive learning rate
- **Loss Function**: CrossEntropyLoss
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Patience-based validation monitoring

## Design Principles

### 1. Modularity
- **Separated Concerns**: CRO, architecture generation, training, evaluation
- **Pluggable Components**: Easy to swap fitness functions, optimizers
- **Configuration-Driven**: Extensive parameterization

### 2. Efficiency
- **Partial Training**: Quick fitness evaluation
- **Lazy Evaluation**: Models created on-demand
- **Memory Management**: Careful GPU memory usage
- **Parallel-Ready**: Designed for multi-GPU scaling

### 3. Flexibility
- **Multiple Datasets**: MNIST, MedMNIST, extensible
- **Various Architectures**: Different Inception configurations
- **Configurable Search Space**: Adjustable complexity bounds

### 4. Research Focus
- **Statistical Rigor**: Multiple runs, confidence intervals
- **Comprehensive Metrics**: Beyond accuracy
- **Green AI**: Efficiency and sustainability focus
- **Reproducibility**: Deterministic with fixed seeds

## Key Algorithms

### 1. Architecture Generation

```python
def generate_inception_architecture(params):
    """Generate Inception module from parameter vector."""
    branches = []
    for branch_config in params['branch_configs']:
        branch = Sequential()
        for i, (filters, kernel_size) in enumerate(
            zip(branch_config['filters'], branch_config['kernel_sizes'])
        ):
            if i == 0 or not branch_config.get('pooling', False):
                branch.add_module(f'conv_{i}', 
                    Conv2d(in_channels, filters, kernel_size, padding='same')
                )
            else:
                branch.add_module('pool', MaxPool2d(3, stride=1, padding=1))
                branch.add_module(f'conv_{i}', 
                    Conv2d(in_channels, filters, 1)
                )
            branch.add_module(f'bn_{i}', BatchNorm2d(filters))
            branch.add_module(f'relu_{i}', ReLU(inplace=True))
        branches.append(branch)
    
    return InceptionModule(branches)
```

### 2. CRO Evolution Step

```python
def evolution_step(self):
    """Single CRO evolution step."""
    # Broadcast spawning (crossover)
    for coral in self.reef:
        if coral and random.random() < self.Fb:
            mate = self.select_mate(coral)
            offspring = self.crossover(coral, mate)
            self.attempt_settlement(offspring)
    
    # Asexual reproduction (mutation)
    for coral in self.reef:
        if coral and random.random() < self.Fa:
            offspring = self.mutate(coral)
            self.attempt_settlement(offspring)
    
    # Depredation (remove poor performers)
    for coral in self.reef:
        if coral and random.random() < self.Fd:
            if coral.fitness < self.fitness_threshold:
                self.remove_coral(coral)
```

### 3. Fitness Evaluation

```python
def evaluate_fitness(model, data_loader, device, method='linear', alpha=7, beta=1.0):
    """Evaluate model fitness."""
    # Get accuracy through training/validation
    accuracy = train_and_evaluate(model, data_loader, device)
    
    # Get model complexity
    parameters = sum(p.numel() for p in model.parameters())
    param_millions = parameters / 1e6
    
    # Calculate fitness
    if method == 'linear':
        fitness = alpha * accuracy - beta * param_millions
    elif method == 'poly':
        fitness = (alpha * accuracy) ** beta / param_millions
    
    return fitness, accuracy, parameters
```

## Performance Optimizations

### 1. Memory Management
- **Model Cleanup**: Explicit deletion after evaluation
- **GPU Cache**: Clear CUDA cache between evaluations
- **Batch Size Adaptation**: Dynamic based on available memory

### 2. Training Efficiency
- **Partial Training**: Limited epochs for fitness evaluation
- **Early Stopping**: Prevent overfitting and save time
- **Learning Rate Scheduling**: Adaptive optimization

### 3. Architecture Constraints
- **Parameter Limits**: Prevent extremely large models
- **Depth Limits**: Avoid vanishing gradients
- **Branch Limits**: Control search space complexity

## Extensibility

### Adding New Components

1. **New Optimization Algorithms**
   - Implement base optimizer interface
   - Add to `src/optimizers/`
   - Update configuration

2. **New Architecture Types**
   - Extend model base class
   - Implement parameter encoding
   - Add to model factory

3. **New Fitness Functions**
   - Add to `src/fitness_utils.py`
   - Update configuration options
   - Validate with existing models

4. **New Datasets**
   - Implement data loader
   - Add preprocessing pipeline
   - Update evaluation framework

## Configuration System

The system uses hierarchical configuration:

1. **Default Config** (`configs/default_config.py`)
2. **CLI Overrides** (command-line arguments)
3. **Environment Variables** (for deployment)
4. **Experiment Configs** (for research)

### Key Configuration Categories
- **Dataset Parameters**: Batch size, validation split
- **Training Parameters**: Learning rate, epochs, patience
- **CRO Parameters**: Reef size, mutation rates, generations
- **Architecture Parameters**: Branch limits, filter ranges
- **Fitness Parameters**: Method, alpha, beta values

## Research Applications

The architecture supports various research directions:

1. **Architecture Search**: Automated design space exploration
2. **Green AI**: Efficiency-focused optimization
3. **Transfer Learning**: Pre-trained feature extraction
4. **Multi-Objective Optimization**: Accuracy vs efficiency trade-offs
5. **Evolutionary Computation**: Novel metaheuristic algorithms

## Best Practices

1. **Reproducibility**: Always set random seeds
2. **Validation**: Use separate validation sets for fitness
3. **Statistical Significance**: Multiple runs for research
4. **Resource Monitoring**: Track GPU memory and training time
5. **Hyperparameter Tuning**: Systematic exploration of CRO parameters

## Debugging and Profiling

### Common Issues
1. **Memory Errors**: Reduce batch size or model complexity
2. **Slow Convergence**: Adjust learning rates or CRO parameters
3. **Poor Diversity**: Increase mutation rates or reef size
4. **Overfitting**: Enable early stopping, add regularization

### Profiling Tools
- **PyTorch Profiler**: Detailed performance analysis
- **NVIDIA Nsight**: GPU utilization monitoring
- **Memory Profiler**: Track memory usage patterns
- **TensorBoard**: Training visualization and debugging

This architecture provides a solid foundation for research in evolutionary neural architecture search with a focus on efficiency and sustainability.

