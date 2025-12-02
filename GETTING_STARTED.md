# Getting Started with Inception-CRO

🎉 **Welcome to your clean, organized Inception-CRO project!**

This guide will help you get started with the newly refactored and organized project structure.

## 📰 What's New

Your project has been completely reorganized and enhanced with:

✅ **Clean Structure**: Removed redundant files and organized everything logically  
✅ **Unified Interface**: Single `inception_cro` command for all operations  
✅ **Advanced Evaluation**: Research-level statistical analysis framework  
✅ **Comprehensive Documentation**: Detailed guides in `docs/` directory  
✅ **Legacy Preservation**: Old scripts moved to `__archive__/` for reference  

## 🗺️ Project Structure

```
Inception-CRO/
├── 📁 src/                          # Core source code
│   ├── cro.py                       # Coral Reef Optimization
│   ├── models.py                    # Inception-CRO and baseline models
│   ├── trainer.py                   # Training framework
│   ├── research_metrics.py          # Advanced evaluation metrics
│   └── ... (other core modules)
├── 📁 scripts/                      # Execution scripts
│   ├── demo_research_evaluation.py  # Quick demonstration
│   ├── enhanced_comparison.py       # Individual comparisons
│   └── run_comprehensive_experiments.py  # Batch experiments
├── 📁 docs/                         # Documentation
│   ├── training.md                  # Training guide
│   ├── evaluation.md                # Evaluation framework
│   └── architecture.md              # System architecture
├── 📁 configs/                      # Configuration files
├── 📁 tests/                        # Unit tests
├── 📁 __archive__/                  # Legacy scripts (preserved)
├── inception_cro                     # ✨ Unified command interface
├── main.py                          # Core training script
├── experiment_analyzer.py           # Results analysis
├── run_tests.py                     # Test runner
└── README.md                        # Main documentation
```

## 🚀 Quick Start Commands

### 0. Check Dependencies (1 minute)
```bash
# Verify all required packages are installed
python check_dependencies.py

# Install missing packages if needed
pip install -r requirements.txt
```

### 1. Basic Demo (5 minutes)
```bash
# Quick demonstration of the framework on MedMNIST
./inception_cro demo

# Or view capabilities without running
./inception_cro demo --capabilities
```

### 2. Train Inception-CRO Model
```bash
# Train on MedMNIST ChestMNIST with default settings
./inception_cro train

# Train on MNIST with custom parameters
./inception_cro train --dataset-name mnist --max-generations 20 --reef-size "(3,3)"

# Train on different MedMNIST subset
./inception_cro train --dataset-name pathmnist --max-generations 25 --num-epochs 20
```

### 3. Compare Against TorchVision Models
```bash
# Quick comparison with TorchVision baselines (30-60 minutes)
./inception_cro compare --mode quick

# Full research evaluation (several hours)
./inception_cro compare --mode full

# Ablation studies
./inception_cro compare --mode ablation
```

### 4. Analyze Results
```bash
# Analyze experimental results
./inception_cro analyze

# Analyze last 7 days only
./inception_cro analyze --days-back 7
```

### 5. Run Tests
```bash
# Run test suite
./inception_cro test

# Quick tests only
./inception_cro test --quick
```

## 📚 Available Documentation

| Guide | Description | When to Use |
|-------|-------------|-------------|
| [Training Guide](docs/training.md) | Complete training instructions | When training models |
| [Evaluation Guide](docs/evaluation.md) | Research-level evaluation | For comprehensive analysis |
| [Architecture Guide](docs/architecture.md) | System design and internals | For development/extension |
| [Main README](README.md) | Project overview | First time users |

## 🎯 Common Use Cases

### For Research Paper
```bash
# Complete evaluation with statistical rigor
./inception_cro compare --mode all

# Results will include:
# - Multiple runs for statistical significance
# - Confidence intervals and effect sizes
# - Publication-ready visualizations
# - Comprehensive reports in markdown and JSON
```

### For Development
```bash
# Quick iterative testing
./inception_cro demo
./inception_cro compare --mode quick

# Analyze results
./inception_cro analyze --list-only
```

### For Production
```bash
# Train optimized model for deployment
./inception_cro train --dataset-name medmnist --medmnist-subset chestmnist \
    --reef-size "(4,4)" --max-generations 30 --num-epochs 25
```

## 📊 Understanding Results

### Generated Output Structure
```
results/
├── plots/                    # 📈 Visualizations
│   ├── training_curves_*.png
│   ├── model_comparison_*.png
│   └── efficiency_analysis_*.png
├── reports/                  # 📝 Research reports
│   ├── research_report.md    # Human-readable
│   └── comprehensive_report.json  # Machine-readable
└── raw_data/                 # 💾 Raw experimental data
```

### Key Metrics
- **Accuracy**: Classification performance
- **Efficiency**: Accuracy per million parameters
- **Statistical Significance**: p-values and confidence intervals
- **Training Time**: Resource utilization
- **Green AI Metrics**: Sustainability scores

## 🛠️ Advanced Usage

### Direct Script Execution
If you prefer direct script execution:

```bash
# Training
python main.py --dataset-name mnist --max-generations 15

# Research evaluation
python scripts/run_comprehensive_experiments.py --mode full

# Quick demo
python scripts/demo_research_evaluation.py
```

### Legacy Scripts
Old scripts are preserved in `__archive__/legacy_scripts/` for reference:
- `compare_models.py` - Original comparison script
- `quick_demo.py` - Original demo
- `migrate_experiments.py` - Experiment migration tool

## 🐛 Troubleshooting

### Common Issues

**Permission Error with `inception_cro`**
```bash
chmod +x inception_cro
```

**Missing Dependencies**
```bash
pip install -r requirements.txt
```

**Out of Memory**
```bash
# Reduce batch size
./inception_cro train --batch-size 32

# Or use smaller models
./inception_cro compare --mode quick
```

**Slow Evaluation**
```bash
# Use quick mode for testing
./inception_cro compare --mode quick

# Disable extended training
./inception_cro compare --mode full --no-extended-training
```

### Getting Help
```bash
# Command help
./inception_cro --help
./inception_cro train --help
./inception_cro compare --help

# Specific documentation
cat docs/training.md
cat docs/evaluation.md
```

## 🚀 Next Steps

1. **Start with the demo**: `./inception_cro demo`
2. **Read the guides**: Check `docs/` directory
3. **Train your first model**: `./inception_cro train`
4. **Run comparisons**: `./inception_cro compare --mode quick`
5. **Analyze results**: `./inception_cro analyze`

## 🎆 What Was Removed/Reorganized

### Removed Files
- `README_CLI.md` → Content moved to `docs/training.md`
- `README_EXPERIMENTS.md` → Content integrated into main README
- `README_MODEL_COMPARISON.md` → Content moved to `docs/evaluation.md`
- `README_RESEARCH_EVALUATION.md` → Content moved to `docs/evaluation.md`
- `cli_examples.py` → Examples integrated into documentation
- `show_comparison_framework.py` → Replaced by unified interface
- `test_baseline_models.py` → Functionality integrated into test suite

### Moved to Archive
- `compare_models.py` → `__archive__/legacy_scripts/`
- `quick_demo.py` → `__archive__/legacy_scripts/`
- `migrate_experiments.py` → `__archive__/legacy_scripts/`

### New/Enhanced
- `inception_cro` → ✨ **New unified command interface**
- `scripts/` → ✨ **New research-level evaluation framework**
- `docs/` → ✨ **Comprehensive documentation**
- `src/research_metrics.py` → ✨ **Advanced statistical analysis**

---

**🎉 Your project is now clean, organized, and ready for research-level work!**

Start with `./inception_cro demo` to see everything in action.

