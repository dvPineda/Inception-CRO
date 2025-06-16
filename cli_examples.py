#!/usr/bin/env python3
"""
CLI Examples for Inception-CRO

This script demonstrates various ways to override configuration parameters
from the command line when running the main.py script.
"""

import subprocess
import sys

def run_example(description, command):
    """
    Print and optionally run an example command.
    """
    print(f"\n{description}:")
    print(f"python {command}")
    print("─" * 50)

def main():
    print("=== CLI Arguments Examples for Inception-CRO ===")
    print("\nNote: Add --help to see all available options")
    
    # Basic help
    run_example(
        "Show all available CLI options",
        "main.py --help"
    )
    
    # Dataset examples
    run_example(
        "Run with MNIST dataset",
        "main.py --dataset-name mnist --batch-size 64"
    )
    
    run_example(
        "Run with MedMNIST pathmnist subset",
        "main.py --dataset-name medmnist --medmnist-subset pathmnist --batch-size 32"
    )
    
    # Training parameter examples
    run_example(
        "Override training parameters",
        "main.py --learning-rate 0.01 --num-epochs 20 --patience 5"
    )
    
    # CRO hyperparameter examples
    run_example(
        "Override CRO hyperparameters",
        'main.py --reef-size "(3, 3)" --max-generations 20 --mutation-rate 0.3'
    )
    
    run_example(
        "Quick test run with small parameters",
        'main.py --reef-size "(2, 2)" --max-generations 5 --num-epochs 3 --batch-size 32'
    )
    
    # Branch configuration
    run_example(
        "Configure Inception branch limits",
        "main.py --branch-min 2 --branch-max 6"
    )
    
    # Fitness method examples
    run_example(
        "Use polynomial fitness method",
        "main.py --fitness-method poly --fitness-alpha 10 --fitness-beta 0.5"
    )
    
    # Comprehensive example
    run_example(
        "Comprehensive configuration override",
        'main.py --dataset-name medmnist --medmnist-subset chestmnist --batch-size 64 --learning-rate 0.005 --num-epochs 15 --reef-size "(4, 4)" --max-generations 25 --mutation-rate 0.25 --branch-min 2 --branch-max 5 --fitness-method poly --seed 123'
    )
    
    # Disable shuffling example
    run_example(
        "Disable dataset shuffling",
        "main.py --no-shuffle-dataset"
    )
    
    print("\n=== Configuration File Locations ===")
    print("Default config: configs/default_config.py")
    print("Results will be saved in: experiments/InceptionCRO_<timestamp>/")
    
    print("\n=== Notes ===")
    print("• All CLI arguments override values from default_config.py")
    print("• Use quotes around tuple values like reef-size: '(3, 3)'")
    print("• Boolean flags: --shuffle-dataset (enable) or --no-shuffle-dataset (disable)")
    print("• Configuration validation will catch invalid parameter combinations")
    print("• Experiment results are timestamped and saved automatically")

if __name__ == "__main__":
    main()

