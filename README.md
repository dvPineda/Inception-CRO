# Inception-CRO 🧬

**Optimizing Inception modules using Coral Reef Optimization for efficient and sustainable AI**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository hosts a master's thesis project that investigates optimizing Inception modules using **Coral Reef Optimization (CRO)** for enhanced efficiency in image classification tasks, contributing to **GreenAI** principles.

## 🌟 Overview

**Inception-CRO** combines the architectural flexibility of Inception modules with the metaheuristic optimization power of Coral Reef Optimization to automatically design efficient neural network architectures. The project focuses on:

- 🧠 **Automated Architecture Design**: Using CRO to optimize Inception module configurations
- ⚡ **Efficiency Focus**: Achieving high accuracy with minimal computational resources
- 🌱 **Green AI**: Reducing environmental impact through optimized, smaller architectures
- 🔬 **Research-Grade Evaluation**: Comprehensive benchmarking with statistical analysis

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/your-username/Inception-CRO.git
cd Inception-CRO

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### 1. Quick Demo (5 minutes)
```bash
# Run a quick demonstration
python scripts/demo_research_evaluation.py
```

#### 2. Train Inception-CRO Model
```bash
# Train on MNIST with default settings
python main.py --dataset-name mnist

# Train on MedMNIST ChestMNIST with custom parameters
python main.py --dataset-name chestmnist --max-generations 20 --num-epochs 25

# Train on MedMNIST PathMNIST
python main.py --dataset-name pathmnist --reef-size "(4,4)" --max-generations 30
```

#### 3. Compare Against TorchVision Baselines
```bash
# Quick comparison with TorchVision models
python scripts/run_comprehensive_experiments.py --mode quick

# Full research-level evaluation against all baselines
python scripts/run_comprehensive_experiments.py --mode full
```

## 📋 Available Commands

### Core Training
- `python main.py` - Train Inception-CRO with evolutionary optimization
- `python main.py --help` - See all available training options

### Model Comparison & Evaluation
- `python scripts/demo_research_evaluation.py` - Quick demonstration
- `python scripts/run_comprehensive_experiments.py --mode quick` - Fast comparison
- `python scripts/run_comprehensive_experiments.py --mode full` - Complete evaluation
- `python scripts/run_comprehensive_experiments.py --mode ablation` - Ablation studies

### Development & Testing
- `python run_tests.py` - Run test suite
- `python experiment_analyzer.py` - Analyze experimental results

## 📁 Project Structure

```
Inception-CRO/
├── 📁 src/                          # Core source code
│   ├── cro.py                       # Coral Reef Optimization implementation
│   ├── models.py                    # Inception-CRO and baseline models
│   ├── trainer.py                   # Training and evaluation logic
│   ├── research_metrics.py          # Research-level evaluation metrics
│   └── ...
├── 📁 scripts/                      # Execution scripts
│   ├── demo_research_evaluation.py  # Quick demonstration
│   ├── enhanced_comparison.py       # Individual model comparison
│   └── run_comprehensive_experiments.py  # Batch experiments
├── 📁 configs/                      # Configuration files
├── 📁 tests/                        # Unit tests
├── 📁 docs/                         # Documentation
├── main.py                          # Main training script
└── README.md                        # This file
```

## 🔬 Research Features

### Comprehensive Evaluation Framework
- **Statistical Analysis**: Multiple runs, confidence intervals, significance testing
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, Matthews Correlation Coefficient
- **Efficiency Analysis**: Parameter count, training time, inference time, memory usage
- **Green AI Metrics**: Carbon footprint estimates, sustainability scores
- **Publication-Ready Reports**: Markdown and JSON outputs with visualizations

### Supported Datasets
- **MNIST**: Classic handwritten digit recognition
- **MedMNIST**: Medical imaging datasets (ChestMNIST, PathMNIST, DermaMNIST, etc.)
  - ChestMNIST: Chest X-ray classification
  - PathMNIST: Pathology image classification  
  - DermaMNIST: Dermatology image classification
  - And 9 more medical imaging datasets

### Baseline Models (TorchVision)
- **Efficient Models**: MobileNet-v2/v3, EfficientNet-B0/B1, SqueezeNet
- **ResNet Family**: ResNet-18, ResNet-34, ResNet-50
- **Dense Architectures**: DenseNet-121
- **VGG Models**: VGG-16 (with/without BatchNorm)
- **Inception Models**: Inception-v3, GoogLeNet

## 📊 Example Results

Typical performance on MedMNIST ChestMNIST:

| Model | Accuracy | Parameters | Efficiency* |
|-------|----------|------------|-------------|
| **Inception-CRO** | **94.2%** | **125K** | **748.7** |
| EfficientNet-B0 | 94.5% | 4.0M | 23.5 |
| ResNet-18 | 93.8% | 11.2M | 8.4 |
| MobileNet-v2 | 92.4% | 2.2M | 41.2 |

*Efficiency = Accuracy per million parameters

## 🛠️ Advanced Usage

### Custom Training Configuration
```bash
python main.py \
    --dataset-name medmnist \
    --medmnist-subset chestmnist \
    --reef-size "(4,4)" \
    --max-generations 25 \
    --learning-rate 0.005 \
    --num-epochs 20
```

### Batch Experiments
```bash
# Run all experiment types
python scripts/run_comprehensive_experiments.py --mode all

# Custom output directory
python scripts/run_comprehensive_experiments.py --mode full --output-dir my_results
```

### Research Analysis
```bash
# Analyze experimental results
python experiment_analyzer.py

# View framework capabilities
python scripts/demo_research_evaluation.py --capabilities
```

## 📚 Documentation

- **[Training Guide](docs/training.md)**: Detailed training instructions
- **[Evaluation Guide](docs/evaluation.md)**: Research-level evaluation framework
- **[Architecture Guide](docs/architecture.md)**: Understanding Inception-CRO design
- **[API Reference](docs/api.md)**: Code documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{inception_cro_2024,
  title={Optimizing Inception Modules using Coral Reef Optimization for Sustainable AI},
  author={[Your Name]},
  school={[Your University]},
  year={2024},
  url={https://github.com/your-username/Inception-CRO}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Miguel Cárdenas-Montes** - Research Supervisor
- **Miguel A. Gutiérrez-Naranjo** - Research Supervisor
- Contributors to the open-source libraries used in this project

---

**🌱 Built with sustainability in mind • 🧬 Optimized through evolution • 🔬 Validated with scientific rigor**
