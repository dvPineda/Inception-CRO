# Evaluation Framework Guide

This guide covers the comprehensive research-level evaluation framework for benchmarking Inception-CRO against state-of-the-art models.

## Overview

The evaluation framework provides:
- **Statistical rigor** with multiple runs and significance testing
- **Comprehensive metrics** beyond simple accuracy
- **Efficiency analysis** for Green AI research
- **Publication-ready** reports and visualizations

## Quick Start

### 1. Demo Evaluation
```bash
# Quick demonstration (5 minutes)
python scripts/demo_research_evaluation.py

# View capabilities without running
python scripts/demo_research_evaluation.py --capabilities
```

### 2. Quick Test
```bash
# Fast evaluation with minimal settings
python scripts/run_comprehensive_experiments.py --mode quick
```

### 3. Full Research Evaluation
```bash
# Complete research-level evaluation
python scripts/run_comprehensive_experiments.py --mode full
```

## Evaluation Modes

### Quick Mode (`--mode quick`)
- **Duration**: 30-60 minutes
- **Datasets**: iris, wine
- **Models**: Inception-CRO, SimpleFFNN, EnhancedFFNN
- **Runs**: 3 per model
- **Training**: 50 epochs, 30 generations
- **Purpose**: Rapid prototyping and testing

### Full Mode (`--mode full`)
- **Duration**: Several hours
- **Datasets**: iris, wine, breast_cancer, digits
- **Models**: All model architectures (FFNNs, CNNs, RNNs, Transformers)
- **Runs**: 5-7 per model
- **Training**: 200-300 epochs, 100-200 generations
- **Purpose**: Complete research evaluation

### Ablation Mode (`--mode ablation`)
- **Duration**: 1-2 hours
- **Focus**: Component analysis of Inception-CRO
- **Runs**: 7 per model (high statistical power)
- **Purpose**: Understanding model contributions

### Scalability Mode (`--mode scalability`)
- **Duration**: 2-3 hours
- **Focus**: Performance scaling analysis
- **Datasets**: Different sizes and complexities
- **Purpose**: Resource utilization benchmarks

### All Mode (`--mode all`)
- **Duration**: Extended time
- **Scope**: Complete evaluation suite
- **Purpose**: Maximum research rigor

## Research Metrics

### Performance Metrics
- **Accuracy**: Mean, standard deviation, 95% confidence intervals
- **Classification**: Precision, Recall, F1-Score (macro, micro, weighted)
- **Advanced**: Matthews Correlation Coefficient, AUC-ROC
- **Error Analysis**: Confusion matrices

### Statistical Analysis
- **Multiple Runs**: 3-7 evaluation runs for significance
- **Confidence Intervals**: 95% CIs for all metrics
- **Significance Testing**: Pairwise model comparisons
- **Effect Sizes**: Cohen's d for practical significance
- **Reliability**: Consistency and stability measures

### Efficiency Analysis
- **Model Complexity**: Parameter count, depth, layer analysis
- **Computational Cost**: Training time, inference time, FLOPs
- **Memory Usage**: GPU/CPU consumption
- **Green AI**: Sustainability scores, carbon footprint estimates

### Training Analysis
- **Learning Curves**: Loss and accuracy over time
- **Convergence**: Early stopping, overfitting detection
- **Generalization**: Training vs validation performance gap

## Command Line Options

### Basic Usage
```bash
python scripts/run_comprehensive_experiments.py \
    --mode [quick|full|ablation|scalability|all] \
    --output-dir results/my_evaluation
```

### Advanced Configuration
```bash
python scripts/enhanced_comparison.py \
    --datasets iris wine breast_cancer \
    --models Inception-CRO SimpleFFNN EnhancedFFNN \
    --num-runs 5 \
    --max-epochs 200 \
    --max-generations 100 \
    --extended-training \
    --batch-size 32 \
    --output-dir results/custom_evaluation
```

### Available Options
- `--datasets`: List of datasets to evaluate
- `--models`: List of models to compare
- `--num-runs`: Number of evaluation runs (3-10)
- `--max-epochs`: Maximum training epochs
- `--max-generations`: Maximum CRO generations
- `--extended-training`: Use extended training parameters
- `--batch-size`: Batch size for training/evaluation
- `--output-dir`: Output directory for results

## Output Structure

```
results/
├── plots/                          # 📈 Visualizations
│   ├── training_curves_*.png       # Training/validation curves
│   ├── model_comparison_*.png      # Performance comparisons
│   ├── efficiency_analysis_*.png   # Efficiency analysis
│   ├── statistical_analysis_*.png  # Statistical significance
│   └── cross_dataset_heatmap.png   # Cross-dataset performance
├── reports/                        # 📝 Research reports
│   ├── research_report.md          # Human-readable report
│   ├── comprehensive_report.json   # Machine-readable data
│   ├── model_ranking_analysis.json # Model rankings
│   └── *_summary.json             # Dataset-specific summaries
├── raw_data/                       # 💾 Raw experimental data
│   └── *_results.json             # Individual model results
└── statistical_analysis/           # 📊 Statistical data
    └── significance_tests.json     # Pairwise comparisons
```

## Interpreting Results

### Performance Analysis
- **Best Model**: Highest mean accuracy with statistical significance
- **Most Consistent**: Lowest standard deviation across runs
- **Most Efficient**: Best accuracy-to-parameter ratio
- **Statistical Significance**: p-values and effect sizes

### Efficiency Analysis
- **Parameter Efficiency**: Accuracy per million parameters
- **Time Efficiency**: Accuracy per second of training
- **Memory Efficiency**: Model size vs peak memory usage
- **Green AI Score**: Composite sustainability metric

### Research Report Contents

1. **Executive Summary**: Key findings and best models
2. **Methodology**: Complete experimental setup
3. **Results**: Performance tables and statistical analysis
4. **Statistical Analysis**: Significance tests and effect sizes
5. **Efficiency Analysis**: Resource utilization metrics
6. **Recommendations**: Model selection guidance
7. **Limitations**: Experimental constraints
8. **Future Work**: Extension suggestions

## Example Use Cases

### Research Paper Evaluation
```bash
# Complete evaluation for publication
python scripts/run_comprehensive_experiments.py --mode all

# Results suitable for:
# - Academic methodology sections
# - Statistical significance claims
# - Performance comparison tables
# - Efficiency analysis figures
```

### Model Development
```bash
# Quick iterative testing
python scripts/run_comprehensive_experiments.py --mode quick

# Component analysis
python scripts/run_comprehensive_experiments.py --mode ablation
```

### Deployment Analysis
```bash
# Resource constraints focus
python scripts/run_comprehensive_experiments.py --mode scalability

# Custom evaluation
python scripts/enhanced_comparison.py \
    --datasets production_dataset \
    --models Inception-CRO AdvancedFFNN \
    --num-runs 10
```

## Advanced Features

### Statistical Rigor
- Multiple evaluation runs for significance testing
- 95% confidence intervals for all metrics
- Pairwise statistical comparisons between models
- Effect size calculations for practical significance
- Bonferroni correction for multiple comparisons

### Green AI Integration
- Carbon footprint estimation during training
- Energy efficiency metrics
- Sustainability scoring
- Pareto frontier analysis (accuracy vs resources)

### Publication Support
- LaTeX-ready tables and figures
- Statistical significance annotations
- Professional visualization styling
- Complete methodology documentation
- Reproducible experimental setup

## Customization

### Adding New Models
1. Implement model in `src/models.py`
2. Add to model factory in evaluation scripts
3. Update experiment configurations

### Adding New Datasets
1. Add dataset loader to `src/data_loading.py`
2. Configure preprocessing parameters
3. Update evaluation scripts

### Custom Metrics
1. Extend `ResearchMetrics` class
2. Add visualization functions
3. Update report generation

### Extending Analysis
1. Add new analysis functions to evaluation framework
2. Create custom visualization scripts
3. Implement domain-specific metrics

## Performance Tips

### For Quick Results
- Use `--mode quick` for rapid iteration
- Limit datasets: `--datasets iris wine`
- Reduce runs: `--num-runs 3`

### For Comprehensive Analysis
- Use `--mode full` or `--mode all`
- Include all relevant datasets
- Use sufficient runs: `--num-runs 7`
- Enable extended training: `--extended-training`

### For Resource Management
- Monitor GPU memory usage
- Use appropriate batch sizes
- Consider distributed evaluation for large experiments

## Troubleshooting

### Common Issues

**Out of Memory**
- Reduce batch size: `--batch-size 16`
- Limit model complexity
- Use CPU for evaluation if necessary

**Slow Evaluation**
- Reduce number of runs: `--num-runs 3`
- Use quick mode for testing
- Limit training epochs

**Statistical Issues**
- Increase number of runs for better statistics
- Check for data leakage between train/test
- Verify random seed settings

### Getting Help

- Check the console output for detailed error messages
- Review generated log files in output directory
- Use `--help` flag for command-line options
- Check the troubleshooting section in training guide

## Next Steps

After evaluation:
1. **Analyze Results**: Review generated reports and visualizations
2. **Statistical Analysis**: Examine confidence intervals and significance tests
3. **Publication**: Use results in research papers or presentations
4. **Model Selection**: Choose best models for deployment
5. **Further Research**: Identify areas for improvement

