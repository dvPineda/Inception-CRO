#!/usr/bin/env python3
"""
Experiment Analyzer for Inception-CRO

This script provides utilities to analyze and visualize experiments organized
in the new hierarchical structure: experiments/YYYY-MM-DD/parameter_group/experiment/
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import argparse

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not available, using basic matplotlib styling")

def find_experiments(base_dir="experiments", days_back=None):
    """
    Find all experiments in the organized structure.
    
    Args:
        base_dir: Base experiments directory
        days_back: Only include experiments from the last N days
    
    Returns:
        List of experiment paths with metadata
    """
    experiments = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Experiments directory {base_dir} does not exist")
        return experiments
    
    # Filter dates if specified
    cutoff_date = None
    if days_back:
        cutoff_date = datetime.now() - timedelta(days=days_back)
    
    # Walk through date directories
    for date_dir in base_path.iterdir():
        if not date_dir.is_dir():
            continue
            
        try:
            # Parse date from directory name
            date_obj = datetime.strptime(date_dir.name, "%Y-%m-%d")
            if cutoff_date and date_obj < cutoff_date:
                continue
        except ValueError:
            # Skip non-date directories (old format)
            continue
        
        # Walk through parameter group directories
        for param_dir in date_dir.iterdir():
            if not param_dir.is_dir():
                continue
                
            # Walk through individual experiments
            for exp_dir in param_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                    
                # Check if it's a valid experiment (has results.csv)
                results_file = exp_dir / "results.csv"
                config_file = exp_dir / "config_used.json"
                
                if results_file.exists():
                    experiments.append({
                        'path': str(exp_dir),
                        'date': date_dir.name,
                        'param_group': param_dir.name,
                        'experiment_name': exp_dir.name,
                        'results_file': str(results_file),
                        'config_file': str(config_file) if config_file.exists() else None,
                        'date_obj': date_obj
                    })
    
    return sorted(experiments, key=lambda x: x['date_obj'], reverse=True)

def load_experiment_results(experiments):
    """
    Load results from all experiments into a combined DataFrame.
    """
    all_results = []
    
    for exp in experiments:
        try:
            df = pd.read_csv(exp['results_file'])
            
            # Add metadata columns
            df['exp_date'] = exp['date']
            df['param_group'] = exp['param_group']
            df['exp_path'] = exp['path']
            
            # Load config if available
            if exp['config_file'] and os.path.exists(exp['config_file']):
                with open(exp['config_file'], 'r') as f:
                    config = json.load(f)
                    # Add key config parameters
                    df['reef_size'] = str(config.get('reef_size', 'unknown'))
                    df['fitness_method'] = config.get('fitness_method', 'unknown')
                    df['seed'] = config.get('seed', 'unknown')
            
            all_results.append(df)
            
        except Exception as e:
            print(f"Error loading {exp['results_file']}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    else:
        return pd.DataFrame()

def analyze_parameter_groups(df):
    """
    Analyze results grouped by parameter configurations.
    """
    if df.empty:
        print("No results to analyze")
        return
    
    print("=== Parameter Group Analysis ===")
    
    # Group by parameter group
    grouped = df.groupby('param_group').agg({
        'test_accuracy': ['mean', 'std', 'max', 'count'],
        'best_fitness': ['mean', 'std', 'max'],
        'param_count': ['mean', 'std']
    }).round(4)
    
    print(grouped)
    
    # Best performing configurations
    print("\n=== Top 5 Parameter Groups by Test Accuracy ===")
    top_groups = df.groupby('param_group')['test_accuracy'].mean().sort_values(ascending=False).head(5)
    for group, acc in top_groups.items():
        count = df[df['param_group'] == group].shape[0]
        print(f"{group}: {acc:.2f}% (from {count} experiments)")

def create_visualizations(df, output_dir="experiment_analysis"):
    """
    Create visualizations of experiment results.
    """
    if df.empty:
        print("No data to visualize")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    if HAS_SEABORN:
        sns.set_palette("husl")
    
    # 1. Test accuracy by parameter group
    plt.figure(figsize=(15, 8))
    if HAS_SEABORN:
        sns.boxplot(data=df, x='param_group', y='test_accuracy')
    else:
        # Fallback to basic matplotlib
        grouped = df.groupby('param_group')['test_accuracy']
        plt.boxplot([group[1].values for group in grouped], labels=grouped.groups.keys())
    plt.xticks(rotation=45, ha='right')
    plt.title('Test Accuracy Distribution by Parameter Group')
    plt.ylabel('Test Accuracy (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_param_group.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Test accuracy over time
    plt.figure(figsize=(12, 6))
    df_time = df.copy()
    df_time['exp_date'] = pd.to_datetime(df_time['exp_date'])
    if HAS_SEABORN:
        sns.scatterplot(data=df_time, x='exp_date', y='test_accuracy', hue='param_group', alpha=0.7)
    else:
        # Fallback to basic matplotlib
        for param_group in df_time['param_group'].unique():
            group_data = df_time[df_time['param_group'] == param_group]
            plt.scatter(group_data['exp_date'], group_data['test_accuracy'], 
                       label=param_group, alpha=0.7)
    plt.title('Test Accuracy Over Time')
    plt.ylabel('Test Accuracy (%)')
    plt.xlabel('Date')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Fitness vs Parameter Count
    if 'param_count' in df.columns:
        plt.figure(figsize=(10, 6))
        if HAS_SEABORN:
            sns.scatterplot(data=df, x='param_count', y='test_accuracy', hue='param_group', alpha=0.7)
        else:
            # Fallback to basic matplotlib
            for param_group in df['param_group'].unique():
                group_data = df[df['param_group'] == param_group]
                plt.scatter(group_data['param_count'], group_data['test_accuracy'], 
                           label=param_group, alpha=0.7)
        plt.title('Test Accuracy vs Model Parameter Count')
        plt.xlabel('Parameter Count')
        plt.ylabel('Test Accuracy (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_params.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Dataset comparison if multiple datasets
    if df['dataset'].nunique() > 1:
        plt.figure(figsize=(10, 6))
        if HAS_SEABORN:
            sns.boxplot(data=df, x='dataset', y='test_accuracy', hue='param_group')
        else:
            # Basic matplotlib fallback
            datasets = df['dataset'].unique()
            param_groups = df['param_group'].unique()
            
            x_pos = []
            accuracies = []
            labels = []
            
            for i, dataset in enumerate(datasets):
                for j, param_group in enumerate(param_groups):
                    subset = df[(df['dataset'] == dataset) & (df['param_group'] == param_group)]
                    if not subset.empty:
                        x_pos.extend([i + j * 0.2] * len(subset))
                        accuracies.extend(subset['test_accuracy'].tolist())
                        labels.extend([f"{dataset}_{param_group}"] * len(subset))
            
            plt.scatter(x_pos, accuracies, alpha=0.7)
            plt.xticks(range(len(datasets)), datasets)
        
        plt.title('Test Accuracy by Dataset and Parameter Group')
        plt.ylabel('Test Accuracy (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}/")

def generate_experiment_summary(df):
    """
    Generate a comprehensive summary of experiments.
    """
    if df.empty:
        print("No experiments to summarize")
        return
    
    print("\n=== Experiment Summary ===")
    print(f"Total experiments: {len(df)}")
    print(f"Date range: {df['exp_date'].min()} to {df['exp_date'].max()}")
    print(f"Parameter groups: {df['param_group'].nunique()}")
    print(f"Datasets: {', '.join(df['dataset'].unique())}")
    
    if 'subset' in df.columns:
        print(f"Dataset subsets: {', '.join(df['subset'].unique())}")
    
    print(f"\nAccuracy statistics:")
    print(f"  Mean: {df['test_accuracy'].mean():.2f}%")
    print(f"  Std:  {df['test_accuracy'].std():.2f}%")
    print(f"  Max:  {df['test_accuracy'].max():.2f}%")
    print(f"  Min:  {df['test_accuracy'].min():.2f}%")

def list_experiments_by_date(experiments):
    """
    List experiments organized by date.
    """
    print("\n=== Experiments by Date ===")
    
    current_date = None
    for exp in experiments:
        if exp['date'] != current_date:
            current_date = exp['date']
            print(f"\n📅 {current_date}:")
        
        print(f"  └─ {exp['param_group']}/")
        print(f"     └─ {exp['experiment_name']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Inception-CRO experiments")
    parser.add_argument('--experiments-dir', default='experiments', help='Experiments directory')
    parser.add_argument('--days-back', type=int, help='Only analyze experiments from last N days')
    parser.add_argument('--output-dir', default='experiment_analysis', help='Output directory for visualizations')
    parser.add_argument('--list-only', action='store_true', help='Only list experiments, no analysis')
    
    args = parser.parse_args()
    
    print("🔍 Finding experiments...")
    experiments = find_experiments(args.experiments_dir, args.days_back)
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"Found {len(experiments)} experiments")
    
    if args.list_only:
        list_experiments_by_date(experiments)
        return
    
    print("📊 Loading experiment results...")
    df = load_experiment_results(experiments)
    
    if df.empty:
        print("No results data found!")
        return
    
    # Generate analysis
    generate_experiment_summary(df)
    analyze_parameter_groups(df)
    
    print(f"\n📈 Creating visualizations...")
    create_visualizations(df, args.output_dir)
    
    # Save combined results
    combined_file = os.path.join(args.output_dir, 'combined_results.csv')
    df.to_csv(combined_file, index=False)
    print(f"💾 Combined results saved to {combined_file}")

if __name__ == "__main__":
    main()

