#!/usr/bin/env python3
# scripts/run_comprehensive_experiments.py

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

def run_experiment_suite(
    output_base_dir: str = "results/comprehensive_experiments",
    quick_test: bool = False,
    extended_training: bool = True
):
    """
    Run comprehensive experimental suite with all models and datasets.
    
    Args:
        output_base_dir: Base directory for all experimental results
        quick_test: If True, run quick experiments for testing
        extended_training: If True, use extended training parameters
    """
    
    # Create base output directory
    base_dir = Path(output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment configurations
    if quick_test:
        print("📦 Running QUICK TEST experiments...")
        experiments = {
            "quick_test": {
                "datasets": ["chestmnist"],
                "models": ["inception_cro", "resnet18", "mobilenet_v2"],
                "num_runs": 3,
                "max_epochs": 15,
                "max_generations": 10,
                "batch_size": 32
            }
        }
    else:
        print("🚀 Running COMPREHENSIVE experiments...")
        experiments = {
            "mnist_baseline_comparison": {
                "datasets": ["mnist"],
                "models": ["inception_cro", "resnet18", "resnet34", "mobilenet_v2", "efficientnet_b0"],
                "num_runs": 5,
                "max_epochs": 25,
                "max_generations": 20,
                "batch_size": 64
            },
            "medmnist_chest_comparison": {
                "datasets": ["chestmnist"],
                "models": ["inception_cro", "resnet18", "mobilenet_v2", "densenet121", "efficientnet_b0"],
                "num_runs": 5,
                "max_epochs": 30,
                "max_generations": 25,
                "batch_size": 32
            },
            "medmnist_path_comparison": {
                "datasets": ["pathmnist"],
                "models": ["inception_cro", "resnet18", "mobilenet_v2", "squeezenet1_0"],
                "num_runs": 5,
                "max_epochs": 30,
                "max_generations": 25,
                "batch_size": 32
            },
            "comprehensive_evaluation": {
                "datasets": ["mnist", "chestmnist", "pathmnist"],
                "models": ["inception_cro", "resnet18", "mobilenet_v2", "efficientnet_b0"],
                "num_runs": 7,
                "max_epochs": 35,
                "max_generations": 30,
                "batch_size": 32
            }
        }
    
    # Run each experiment
    experiment_results = {}
    total_experiments = len(experiments)
    
    print(f"\n📄 Total experiment suites to run: {total_experiments}")
    print(f"📅 Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    for exp_idx, (exp_name, exp_config) in enumerate(experiments.items(), 1):
        print(f"\n🏃 Running experiment suite {exp_idx}/{total_experiments}: {exp_name}")
        print(f"📁 Datasets: {exp_config['datasets']}")
        print(f"🤖 Models: {exp_config['models']}")
        print(f"🔄 Runs per model: {exp_config['num_runs']}")
        
        # Create experiment-specific output directory
        exp_output_dir = base_dir / exp_name
        
        try:
            # Build command
            cmd = [
                sys.executable, 
                os.path.join(os.path.dirname(__file__), "enhanced_comparison.py"),
                "--datasets"] + exp_config["datasets"] + [
                "--models"] + exp_config["models"] + [
                "--num-runs", str(exp_config["num_runs"]),
                "--max-epochs", str(exp_config["max_epochs"]),
                "--max-generations", str(exp_config["max_generations"]),
                "--batch-size", str(exp_config["batch_size"]),
                "--output-dir", str(exp_output_dir)
            ]
            
            if extended_training:
                cmd.append("--extended-training")
            
            print(f"⚡ Executing: {' '.join(cmd)}")
            
            # Run experiment
            start_time = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = datetime.now()
            
            duration = end_time - start_time
            
            print(f"✅ Experiment {exp_name} completed successfully!")
            print(f"⏱️ Duration: {duration}")
            print(f"📁 Results saved to: {exp_output_dir}")
            
            # Store experiment metadata
            experiment_results[exp_name] = {
                "status": "success",
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds(),
                "output_directory": str(exp_output_dir),
                "configuration": exp_config
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Experiment {exp_name} failed!")
            print(f"Error: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            
            experiment_results[exp_name] = {
                "status": "failed",
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr,
                "configuration": exp_config
            }
        
        print("-" * 60)
    
    # Generate final summary
    print(f"\n🏁 All experiments completed!")
    print(f"📅 End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Count successes and failures
    successes = sum(1 for result in experiment_results.values() if result["status"] == "success")
    failures = sum(1 for result in experiment_results.values() if result["status"] == "failed")
    
    print(f"✅ Successful experiments: {successes}/{total_experiments}")
    print(f"❌ Failed experiments: {failures}/{total_experiments}")
    
    # Save experiment summary
    summary_file = base_dir / "experiment_summary.json"
    import json
    
    summary = {
        "total_experiments": total_experiments,
        "successful_experiments": successes,
        "failed_experiments": failures,
        "experiment_results": experiment_results,
        "execution_summary": {
            "quick_test_mode": quick_test,
            "extended_training": extended_training,
            "base_output_directory": str(base_dir)
        }
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Experiment summary saved to: {summary_file}")
    
    if successes > 0:
        print(f"\n📊 To view results, check the output directories:")
        for exp_name, result in experiment_results.items():
            if result["status"] == "success":
                print(f"  - {exp_name}: {result['output_directory']}")
    
    return experiment_results

def run_ablation_studies(
    output_base_dir: str = "results/ablation_studies",
    dataset: str = "chestmnist"
):
    """
    Run ablation studies on Inception-CRO to understand component contributions.
    
    Args:
        output_base_dir: Base directory for ablation study results
        dataset: Dataset to use for ablation studies (chestmnist, pathmnist, or mnist)
    """
    
    print(f"\n🔬 Running Inception-CRO Ablation Studies on {dataset}...")
    
    base_dir = Path(output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Ablation study configurations
    ablation_studies = {
        "full_inception_cro": {
            "description": "Full Inception-CRO with all components",
            "models": ["inception_cro"],
            "config": "standard"
        },
        "inception_cro_vs_strong_baselines": {
            "description": "Inception-CRO vs strong TorchVision baselines",
            "models": ["inception_cro", "resnet18", "efficientnet_b0", "densenet121"],
            "config": "comparison"
        },
        "inception_cro_vs_efficient_models": {
            "description": "Inception-CRO vs efficiency-focused models",
            "models": ["inception_cro", "mobilenet_v2", "squeezenet1_0", "mobilenet_v3_small"],
            "config": "efficiency_comparison"
        }
    }
    
    ablation_results = {}
    
    for study_name, study_config in ablation_studies.items():
        print(f"\n🎯 Running ablation study: {study_name}")
        print(f"📝 Description: {study_config['description']}")
        
        study_output_dir = base_dir / study_name
        
        try:
            # Build command for ablation study
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "enhanced_comparison.py"),
                "--datasets", dataset,
                "--models"] + study_config["models"] + [
                "--num-runs", "7",  # More runs for ablation studies
                "--max-epochs", "300",
                "--max-generations", "200",
                "--batch-size", "32",
                "--extended-training",
                "--output-dir", str(study_output_dir)
            ]
            
            print(f"⚡ Executing ablation study...")
            
            start_time = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = datetime.now()
            
            duration = end_time - start_time
            
            print(f"✅ Ablation study {study_name} completed!")
            print(f"⏱️ Duration: {duration}")
            
            ablation_results[study_name] = {
                "status": "success",
                "duration_seconds": duration.total_seconds(),
                "output_directory": str(study_output_dir)
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Ablation study {study_name} failed: {e}")
            ablation_results[study_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    # Save ablation study summary
    ablation_summary_file = base_dir / "ablation_study_summary.json"
    import json
    
    with open(ablation_summary_file, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    print(f"\n📄 Ablation study summary saved to: {ablation_summary_file}")
    
    return ablation_results

def run_scalability_analysis(
    output_base_dir: str = "results/scalability_analysis"
):
    """
    Run scalability analysis with different dataset sizes and model configurations.
    
    Args:
        output_base_dir: Base directory for scalability analysis results
    """
    
    print("\n📈 Running Scalability Analysis...")
    
    base_dir = Path(output_base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Scalability test configurations
    scalability_tests = {
        "small_scale": {
            "datasets": ["chestmnist"],
            "models": ["inception_cro", "mobilenet_v2"],
            "generations": [10, 20],
            "batch_sizes": [16, 32]
        },
        "medium_scale": {
            "datasets": ["pathmnist"],
            "models": ["inception_cro", "resnet18"],
            "generations": [20, 40],
            "batch_sizes": [32, 64]
        },
        "large_scale": {
            "datasets": ["mnist"],
            "models": ["inception_cro", "efficientnet_b0"],
            "generations": [30, 50],
            "batch_sizes": [64, 128]
        }
    }
    
    scalability_results = {}
    
    for test_name, test_config in scalability_tests.items():
        print(f"\n🔍 Running scalability test: {test_name}")
        
        test_output_dir = base_dir / test_name
        
        try:
            # Use moderate settings for scalability testing
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "enhanced_comparison.py"),
                "--datasets"] + test_config["datasets"] + [
                "--models"] + test_config["models"] + [
                "--num-runs", "3",  # Fewer runs for scalability tests
                "--max-epochs", "150",
                "--max-generations", str(max(test_config["generations"])),
                "--batch-size", str(max(test_config["batch_sizes"])),
                "--extended-training",
                "--output-dir", str(test_output_dir)
            ]
            
            print(f"⚡ Executing scalability test...")
            
            start_time = datetime.now()
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            end_time = datetime.now()
            
            duration = end_time - start_time
            
            print(f"✅ Scalability test {test_name} completed!")
            print(f"⏱️ Duration: {duration}")
            
            scalability_results[test_name] = {
                "status": "success",
                "duration_seconds": duration.total_seconds(),
                "output_directory": str(test_output_dir),
                "configuration": test_config
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Scalability test {test_name} failed: {e}")
            scalability_results[test_name] = {
                "status": "failed",
                "error": str(e),
                "configuration": test_config
            }
    
    # Save scalability analysis summary
    scalability_summary_file = base_dir / "scalability_analysis_summary.json"
    import json
    
    with open(scalability_summary_file, 'w') as f:
        json.dump(scalability_results, f, indent=2)
    
    print(f"\n📄 Scalability analysis summary saved to: {scalability_summary_file}")
    
    return scalability_results

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Experimental Suite Runner')
    parser.add_argument('--mode', choices=['full', 'quick', 'ablation', 'scalability', 'all'], 
                       default='full',
                       help='Experiment mode to run')
    parser.add_argument('--output-dir', type=str, default='results/comprehensive_experiments',
                       help='Base output directory for all experiments')
    parser.add_argument('--no-extended-training', action='store_true',
                       help='Disable extended training (faster but less thorough)')
    parser.add_argument('--ablation-dataset', type=str, default='chestmnist',
                       help='Dataset to use for ablation studies (mnist, chestmnist, pathmnist, etc.)')
    
    args = parser.parse_args()
    
    # Configuration
    extended_training = not args.no_extended_training
    
    print(f"🚀 Starting Comprehensive Experimental Suite")
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Mode: {args.mode}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"⚙️ Extended training: {extended_training}")
    print("=" * 80)
    
    all_results = {}
    
    if args.mode in ['full', 'quick', 'all']:
        # Run main experiment suite
        quick_test = (args.mode == 'quick')
        exp_output_dir = os.path.join(args.output_dir, 'main_experiments')
        
        print(f"\n📋 Running main experimental suite...")
        main_results = run_experiment_suite(
            output_base_dir=exp_output_dir,
            quick_test=quick_test,
            extended_training=extended_training
        )
        all_results['main_experiments'] = main_results
    
    if args.mode in ['ablation', 'all']:
        # Run ablation studies
        ablation_output_dir = os.path.join(args.output_dir, 'ablation_studies')
        
        print(f"\n🔬 Running ablation studies...")
        ablation_results = run_ablation_studies(
            output_base_dir=ablation_output_dir,
            dataset=args.ablation_dataset
        )
        all_results['ablation_studies'] = ablation_results
    
    if args.mode in ['scalability', 'all']:
        # Run scalability analysis
        scalability_output_dir = os.path.join(args.output_dir, 'scalability_analysis')
        
        print(f"\n📈 Running scalability analysis...")
        scalability_results = run_scalability_analysis(
            output_base_dir=scalability_output_dir
        )
        all_results['scalability_analysis'] = scalability_results
    
    # Generate master summary
    master_summary_file = Path(args.output_dir) / "master_experiment_summary.json"
    import json
    
    master_summary = {
        "execution_timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "extended_training": extended_training,
        "total_experiment_categories": len(all_results),
        "results_by_category": all_results
    }
    
    with open(master_summary_file, 'w') as f:
        json.dump(master_summary, f, indent=2)
    
    print(f"\n🏆 ALL EXPERIMENTS COMPLETED!")
    print(f"📄 Master summary saved to: {master_summary_file}")
    print(f"📅 Final timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show summary of all results
    total_successful = 0
    total_failed = 0
    
    for category, category_results in all_results.items():
        successful = sum(1 for result in category_results.values() if result.get("status") == "success")
        failed = sum(1 for result in category_results.values() if result.get("status") == "failed")
        total_successful += successful
        total_failed += failed
        
        print(f"\n📊 {category.upper()}: {successful} successful, {failed} failed")
    
    print(f"\n🔢 OVERALL SUMMARY: {total_successful} successful, {total_failed} failed experiments")
    
    if total_successful > 0:
        print(f"\n👍 Success! Your comprehensive evaluation results are ready for analysis.")
        print(f"📁 Check the output directory: {args.output_dir}")
    
    return all_results

if __name__ == "__main__":
    main()

