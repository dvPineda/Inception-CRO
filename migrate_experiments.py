#!/usr/bin/env python3
"""
Migration utility for Inception-CRO experiments

Migrates old experiment structure (flat) to new organized structure
(date-based with parameter grouping).
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
import argparse

def parse_experiment_name(exp_name):
    """
    Parse old experiment name format: InceptionCRO_YYYYMMDD_HHMMSS
    """
    parts = exp_name.split('_')
    if len(parts) >= 3 and parts[0] == 'InceptionCRO':
        try:
            date_str = parts[1]  # YYYYMMDD
            time_str = parts[2]  # HHMMSS
            
            # Parse date
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            formatted_date = date_obj.strftime("%Y-%m-%d")
            
            return formatted_date, time_str
        except ValueError:
            pass
    
    return None, None

def infer_parameter_group(exp_path, default_config):
    """
    Infer parameter group name from experiment configuration.
    Since old experiments don't have saved configs, use defaults.
    """
    # Try to load config if it exists
    config_file = os.path.join(exp_path, "config_used.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except:
            config = default_config.copy()
    else:
        config = default_config.copy()
    
    # Build parameter group name using same logic as new structure
    dataset_name = config.get('dataset_name', 'medmnist')
    if dataset_name == 'medmnist':
        dataset_name = f"medmnist-{config.get('medmnist_subset', 'chestmnist')}"
    
    reef_size = config.get('reef_size', (2, 2))
    if isinstance(reef_size, list):
        reef_size = tuple(reef_size)
    reef_str = f"{reef_size[0]}x{reef_size[1]}"
    
    max_generations = config.get('max_generations', 10)
    learning_rate = config.get('learning_rate', 0.001)
    
    param_folder = f"{dataset_name}_reef{reef_str}_gen{max_generations}_lr{learning_rate:.0e}"
    
    # Add non-default parameters
    fitness_method = config.get('fitness_method', 'linear')
    if fitness_method != 'linear':
        param_folder += f"_{fitness_method}"
    
    branch_min = config.get('branch_min', 1)
    branch_max = config.get('branch_max', 4)
    if branch_min != 1 or branch_max != 4:
        param_folder += f"_br{branch_min}-{branch_max}"
    
    mutation_rate = config.get('mutation_rate', 0.2)
    if mutation_rate != 0.2:
        param_folder += f"_mut{mutation_rate:.2f}"
    
    return param_folder

def migrate_experiment(old_path, new_base, default_config, dry_run=False):
    """
    Migrate a single experiment to the new structure.
    """
    exp_name = os.path.basename(old_path)
    
    # Parse date from experiment name
    date_str, time_str = parse_experiment_name(exp_name)
    
    if not date_str:
        print(f"⚠️  Could not parse date from {exp_name}, skipping")
        return False
    
    # Infer parameter group
    param_group = infer_parameter_group(old_path, default_config)
    
    # Create new experiment name
    # Try to extract seed from config or use default
    config_file = os.path.join(old_path, "config_used.json")
    seed = 42  # default
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                seed = config.get('seed', 42)
        except:
            pass
    
    new_exp_name = f"exp_{time_str}_seed{seed}"
    
    # Build new path
    new_path = os.path.join(new_base, date_str, param_group, new_exp_name)
    
    print(f"📦 {exp_name}")
    print(f"   → {date_str}/{param_group}/{new_exp_name}")
    
    if dry_run:
        return True
    
    # Create directories and move
    try:
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.move(old_path, new_path)
        
        # Create/update config file with migration info
        config_file = os.path.join(new_path, "config_used.json")
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            config = default_config.copy()
        
        config['migration_info'] = {
            'migrated_at': datetime.now().isoformat(),
            'original_name': exp_name,
            'migration_version': '1.0'
        }
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return True
        
    except Exception as e:
        print(f"❌ Error migrating {exp_name}: {e}")
        return False

def find_old_experiments(experiments_dir):
    """
    Find experiments in the old flat structure.
    """
    old_experiments = []
    
    for item in os.listdir(experiments_dir):
        item_path = os.path.join(experiments_dir, item)
        
        # Skip new structure directories (YYYY-MM-DD format)
        if os.path.isdir(item_path):
            try:
                datetime.strptime(item, "%Y-%m-%d")
                continue  # This is new structure, skip
            except ValueError:
                pass  # Not a date, might be old structure
            
            # Check if it looks like old experiment
            if item.startswith('InceptionCRO_'):
                old_experiments.append(item_path)
    
    return old_experiments

def main():
    parser = argparse.ArgumentParser(description="Migrate old Inception-CRO experiments")
    parser.add_argument('--experiments-dir', default='experiments', help='Experiments directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without actually migrating')
    parser.add_argument('--backup', action='store_true', help='Create backup before migration')
    
    args = parser.parse_args()
    
    # Load default config for inference
    try:
        from configs.default_config import CONFIG
        default_config = CONFIG.copy()
    except ImportError:
        print("Could not load default config, using minimal defaults")
        default_config = {
            'dataset_name': 'medmnist',
            'medmnist_subset': 'chestmnist',
            'reef_size': (2, 2),
            'max_generations': 10,
            'learning_rate': 0.001,
            'fitness_method': 'linear',
            'branch_min': 1,
            'branch_max': 4,
            'mutation_rate': 0.2,
            'seed': 42
        }
    
    print("🔍 Finding old experiments...")
    old_experiments = find_old_experiments(args.experiments_dir)
    
    if not old_experiments:
        print("✅ No old experiments found to migrate")
        return
    
    print(f"Found {len(old_experiments)} old experiments to migrate")
    
    if args.dry_run:
        print("\n🧪 DRY RUN - No files will be moved\n")
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        backup_dir = f"{args.experiments_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"📋 Creating backup at {backup_dir}")
        shutil.copytree(args.experiments_dir, backup_dir)
    
    # Migrate experiments
    print("\n🚀 Starting migration...\n")
    
    success_count = 0
    for old_path in old_experiments:
        if migrate_experiment(old_path, args.experiments_dir, default_config, args.dry_run):
            success_count += 1
    
    print(f"\n✅ Migration complete: {success_count}/{len(old_experiments)} experiments migrated")
    
    if args.dry_run:
        print("\nRun without --dry-run to perform the actual migration")
    else:
        print("\nYou can now use experiment_analyzer.py to analyze your organized experiments!")

if __name__ == "__main__":
    main()

