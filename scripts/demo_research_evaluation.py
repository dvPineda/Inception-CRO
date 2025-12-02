#!/usr/bin/env python3
# scripts/demo_research_evaluation.py

"""
Demonstration script for the research-level evaluation framework.
This script runs a quick example to show the capabilities of the system.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_comparison import EnhancedComparisonFramework

def run_demo_evaluation():
    """
    Run a demonstration of the research-level evaluation framework.
    """
    
    print("🎯 Research-Level Model Evaluation Framework Demo")
    print("=" * 60)
    print("This demo will run a quick evaluation to demonstrate the framework capabilities.\n")
    
    # Create framework with demo output directory
    demo_output_dir = "results/demo_evaluation"
    framework = EnhancedComparisonFramework(output_dir=demo_output_dir)
    
    # Demo configuration - small scale for quick execution
    demo_config = {
        "datasets": ["chestmnist"],  # Focus on MedMNIST for demo
        "models": ["inception_cro", "resnet18", "mobilenet_v2"],  # InceptionCRO vs TorchVision baselines
        "num_evaluation_runs": 3,  # Fewer runs for demo
        "extended_training": False,  # Faster training for demo
        "max_epochs": 10,
        "max_generations": 5,
        "batch_size": 32
    }
    
    print(f"📊 Demo Configuration:")
    print(f"  • Datasets: {demo_config['datasets']}")
    print(f"  • Models: {demo_config['models']}")
    print(f"  • Evaluation runs: {demo_config['num_evaluation_runs']}")
    print(f"  • Max epochs: {demo_config['max_epochs']}")
    print(f"  • Max generations: {demo_config['max_generations']}")
    print(f"  • Output directory: {demo_output_dir}")
    print()
    
    # Estimate runtime
    estimated_time = len(demo_config['datasets']) * len(demo_config['models']) * 2  # rough estimate in minutes
    print(f"⏱️ Estimated runtime: ~{estimated_time} minutes")
    print()
    
    # Ask for confirmation
    try:
        response = input("🚀 Start demo evaluation? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Demo cancelled.")
            return
    except KeyboardInterrupt:
        print("\nDemo cancelled.")
        return
    
    # Run the evaluation
    start_time = time.time()
    
    try:
        framework.run_comprehensive_evaluation(**demo_config)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"⏱️ Total runtime: {duration/60:.1f} minutes")
        print(f"📁 Results saved to: {demo_output_dir}")
        
        # Show what was generated
        output_path = Path(demo_output_dir)
        if output_path.exists():
            print(f"\n📋 Generated outputs:")
            
            # Check for plots
            plots_dir = output_path / "plots"
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                print(f"  📊 {len(plot_files)} visualization plots")
            
            # Check for reports
            reports_dir = output_path / "reports"
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.json")) + list(reports_dir.glob("*.md"))
                print(f"  📄 {len(report_files)} research reports")
            
            # Check for raw data
            raw_data_dir = output_path / "raw_data"
            if raw_data_dir.exists():
                data_files = list(raw_data_dir.glob("*.json"))
                print(f"  💾 {len(data_files)} raw data files")
            
            print(f"\n📖 To view the comprehensive report, open:")
            print(f"  {output_path / 'reports' / 'research_report.md'}")
            print(f"\n📊 To view visualizations, check:")
            print(f"  {plots_dir}/")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\nThis might be due to missing dependencies or incompatible model implementations.")
        print("Please check that all required modules are properly implemented.")
        return
    
    print(f"\n✨ Demo Features Demonstrated:")
    print(f"  ✅ Comprehensive performance metrics (accuracy, F1, precision, recall)")
    print(f"  ✅ Statistical significance testing with confidence intervals")
    print(f"  ✅ Computational efficiency analysis (parameters, timing, memory)")
    print(f"  ✅ Training curve visualization and analysis")
    print(f"  ✅ Model comparison across multiple datasets")
    print(f"  ✅ Research-level reporting with methodology and findings")
    print(f"  ✅ Publication-ready plots and statistical analysis")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Examine the generated reports and visualizations")
    print(f"  2. Run the full experimental suite with:")
    print(f"     python scripts/run_comprehensive_experiments.py --mode full")
    print(f"  3. Customize the evaluation for your specific research needs")
    print(f"  4. Use the statistical analysis for your research publications")

def show_framework_capabilities():
    """
    Show the capabilities of the research framework without running experiments.
    """
    
    print("🔬 Research-Level Evaluation Framework Capabilities")
    print("=" * 70)
    
    capabilities = {
        "📊 Performance Metrics": [
            "Accuracy with statistical significance testing",
            "Precision, Recall, F1-Score (macro, micro, weighted)",
            "Matthews Correlation Coefficient",
            "Confusion Matrix Analysis",
            "AUC-ROC for multi-class problems",
            "Confidence intervals and effect sizes"
        ],
        "🔢 Statistical Analysis": [
            "Multiple evaluation runs for statistical significance",
            "95% confidence intervals",
            "Pairwise statistical significance testing",
            "Effect size calculations (Cohen's d)",
            "Reliability and consistency measures",
            "Coefficient of variation analysis"
        ],
        "⚡ Efficiency Analysis": [
            "Parameter count and model complexity",
            "Training time measurement",
            "Inference time benchmarking",
            "Memory usage analysis (GPU/CPU)",
            "FLOPs estimation",
            "Green AI and sustainability metrics"
        ],
        "📈 Visualization & Reporting": [
            "Training and validation curves",
            "Model performance comparison charts",
            "Efficiency frontier analysis",
            "Statistical significance plots",
            "Cross-dataset performance heatmaps",
            "Publication-ready figures"
        ],
        "📝 Research Reports": [
            "Comprehensive methodology documentation",
            "Executive summary with key findings",
            "Statistical analysis summary",
            "Recommendations and limitations",
            "Human-readable markdown reports",
            "Machine-readable JSON data files"
        ],
        "🔧 Framework Features": [
            "Support for multiple model architectures",
            "Extensible to new datasets and models",
            "Configurable evaluation parameters",
            "Batch experiment execution",
            "Ablation study support",
            "Scalability analysis"
        ]
    }
    
    for category, features in capabilities.items():
        print(f"\n{category}:")
        for feature in features:
            print(f"  ✅ {feature}")
    
    print(f"\n🎯 Usage Examples:")
    print(f"  📊 Quick demo: python scripts/demo_research_evaluation.py")
    print(f"  🚀 Full evaluation: python scripts/run_comprehensive_experiments.py --mode full")
    print(f"  ⚡ Quick test: python scripts/run_comprehensive_experiments.py --mode quick")
    print(f"  🔬 Ablation studies: python scripts/run_comprehensive_experiments.py --mode ablation")
    print(f"  📈 Scalability: python scripts/run_comprehensive_experiments.py --mode scalability")
    
    print(f"\n📚 For research purposes, this framework provides:")
    print(f"  • Statistical rigor required for academic publications")
    print(f"  • Comprehensive metrics beyond simple accuracy")
    print(f"  • Reproducible experimental methodology")
    print(f"  • Professional visualizations and reports")
    print(f"  • Efficiency analysis for practical deployment")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Demo script for research-level model evaluation framework'
    )
    parser.add_argument('--capabilities', action='store_true',
                       help='Show framework capabilities without running demo')
    
    args = parser.parse_args()
    
    if args.capabilities:
        show_framework_capabilities()
    else:
        run_demo_evaluation()

if __name__ == "__main__":
    main()

