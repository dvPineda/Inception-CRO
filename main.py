# main.py

import argparse
import ast
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from configs.default_config import CONFIG
from src.cro import CoralReefOptimization
from src.models import InceptionMNISTModel
from src.trainer import Trainer
from src.utils import evaluate_model, load_data, load_medmnist, save_results_to_csv


MEDMNIST_SUBSETS = [
    "chestmnist",
    "pathmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
]


def parse_reef_size(reef_size_str):
    """Parse reef size tuples coming from CLI strings."""
    try:
        reef_size = ast.literal_eval(reef_size_str)
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Reef size must be a tuple like \"(4,4)\"")

    if (
        not isinstance(reef_size, tuple)
        or len(reef_size) != 2
        or not all(isinstance(x, int) and x > 0 for x in reef_size)
    ):
        raise argparse.ArgumentTypeError("Reef size must be a tuple of two positive integers")

    return reef_size


def main():
    parser = argparse.ArgumentParser(
        description="Train Inception-CRO on MNIST or MedMNIST datasets"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="medmnist",
        choices=["mnist", "medmnist"] + MEDMNIST_SUBSETS,
        help="Dataset to train on",
    )
    parser.add_argument(
        "--medmnist-subset",
        type=str,
        default="chestmnist",
        choices=MEDMNIST_SUBSETS,
        help="MedMNIST subset to use when dataset-name is medmnist",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=None,
        help="Maximum generations for CRO",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size",
    )
    parser.add_argument(
        "--reef-size",
        type=parse_reef_size,
        default=None,
        help='Reef size as tuple string, e.g., "(4,4)"',
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=None,
        help="Mutation rate for CRO",
    )
    parser.add_argument(
        "--fitness-method",
        type=str,
        choices=["linear", "poly"],
        default=None,
        help="Fitness computation method",
    )
    parser.add_argument(
        "--fitness-alpha",
        type=int,
        default=None,
        help="Alpha parameter for fitness computation",
    )
    parser.add_argument(
        "--fitness-beta",
        type=float,
        default=None,
        help="Beta parameter for fitness computation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience for full training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-shuffle-dataset",
        action="store_true",
        help="Disable dataset shuffling",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Whether to save the trained model checkpoint",
    )

    args = parser.parse_args()

    # Override config with command line arguments
    if args.max_generations is not None:
        CONFIG["max_generations"] = args.max_generations
    if args.num_epochs is not None:
        CONFIG["num_epochs"] = args.num_epochs
    if args.learning_rate is not None:
        CONFIG["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        CONFIG["batch_size"] = args.batch_size
    if args.reef_size is not None:
        CONFIG["reef_size"] = args.reef_size
    if args.mutation_rate is not None:
        CONFIG["mutation_rate"] = args.mutation_rate
    if args.fitness_method is not None:
        CONFIG["fitness_method"] = args.fitness_method
    if args.fitness_alpha is not None:
        CONFIG["fitness_alpha"] = args.fitness_alpha
    if args.fitness_beta is not None:
        CONFIG["fitness_beta"] = args.fitness_beta
    if args.patience is not None:
        CONFIG["patience"] = args.patience
    if args.seed is not None:
        CONFIG["seed"] = args.seed
    if args.no_shuffle_dataset:
        CONFIG["shuffle_dataset"] = False
    
    # Create experiment name based on dataset
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = (
        args.medmnist_subset
        if args.dataset_name == "medmnist"
        else args.dataset_name
    )
    dynamic_experiment_name = f"{dataset_name}_InceptionCRO_{timestamp}"
    CONFIG["experiment_name"] = dynamic_experiment_name
    
    # Create experiment directories
    base_exp_dir = os.path.join(CONFIG["experiments_dir"], dynamic_experiment_name)
    CONFIG["plots_dir"] = os.path.join(base_exp_dir, "plots")
    CONFIG["checkpoints_dir"] = os.path.join(base_exp_dir, "checkpoints")
    CONFIG["visualizations_dir"] = os.path.join(base_exp_dir, "visualizations")
    CONFIG["results_csv"] = os.path.join(base_exp_dir, "results.csv")
    
    # Ensure directories exist
    os.makedirs(CONFIG["plots_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoints_dir"], exist_ok=True)
    os.makedirs(CONFIG["visualizations_dir"], exist_ok=True)
    
    print(f"🚀 Starting Inception-CRO Training")
    print(f"📊 Dataset: {dataset_name}")
    print(f"📁 Experiment: {CONFIG['experiment_name']}")
    print(f"⚙️ Max generations: {CONFIG['max_generations']}")
    print(f"🔄 Epochs: {CONFIG['num_epochs']}")
    print(f"📦 Batch size: {CONFIG['batch_size']}")
    print(f"📏 Reef size: {CONFIG['reef_size']}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")
    
    # Set seeds for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    
    # Load dataset based on choice
    print(f"📥 Loading {dataset_name} dataset...")
    if args.dataset_name == "mnist":
        train_loader, val_loader, test_loader, num_classes, input_channels = load_data(
            batch_size=CONFIG["batch_size"],
            validation_split=CONFIG["validation_split"],
            shuffle_dataset=CONFIG["shuffle_dataset"],
            random_seed=CONFIG["seed"]
        )
    else:
        # Use MedMNIST
        subset = (
            args.medmnist_subset
            if args.dataset_name == "medmnist"
            else args.dataset_name
        )
        train_loader, val_loader, test_loader, num_classes, input_channels = load_medmnist(
            batch_size=CONFIG["batch_size"],
            subset=subset,
            validation_split=CONFIG["validation_split"],
            shuffle_dataset=CONFIG["shuffle_dataset"],
            random_seed=CONFIG["seed"]
        )
    
    print(f"✅ Dataset loaded - Classes: {num_classes}, Input channels: {input_channels}")
    
    # Create trainer
    trainer = Trainer(device, CONFIG)
    
    def fitness_function(model_params):
        """Fitness function for CRO optimization."""
        try:
            model = InceptionMNISTModel(
                model_params, 
                input_channels=input_channels, 
                num_classes=num_classes
            ).to(device)
            trainer.partial_train(model, train_loader)
            fitness, accuracy, _ = evaluate_model(
                model,
                val_loader,
                device,
                fitness_method=CONFIG["fitness_method"],
                alpha=CONFIG["fitness_alpha"],
                beta=CONFIG["fitness_beta"]
            )
            return fitness
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return 0.0
    
    # Initialize and run CRO
    print(f"🧬 Initializing Coral Reef Optimization...")
    cro = CoralReefOptimization(
        reef_size=CONFIG["reef_size"],
        rho_0=CONFIG["rho_0"],
        Fb=CONFIG["Fb"],
        Fa=CONFIG["Fa"],
        Pa=CONFIG["Pa"],
        Fd=CONFIG["Fd"],
        Pd=CONFIG["Pd"],
        kappa=CONFIG["kappa"],
        mutation_rate=CONFIG["mutation_rate"],
        fitness_function=fitness_function,
        max_generations=CONFIG["max_generations"],
        max_no_improve=CONFIG["max_no_improve"],
        visualization_dir=CONFIG["visualizations_dir"]
    )
    
    print(f"🔄 Running optimization for {CONFIG['max_generations']} generations...")
    cro.run()
    
    # Plot fitness evolution
    print(f"📊 Generating fitness plots...")
    generations = range(1, len(cro.fitness_history) + 1)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(generations, cro.fitness_history, label='Best Fitness', linewidth=2)
    plt.plot(generations, cro.avg_fitness_history, label='Average Fitness', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(generations, cro.fitness_history, label='Best Fitness', linewidth=2, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = os.path.join(CONFIG["plots_dir"], f"{CONFIG['experiment_name']}_fitness.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📈 Fitness plot saved: {plot_filename}")
    
    # Train best model
    print(f"🎯 Training best model with full epochs...")
    best_model_params = cro.best_coral['solution']
    best_model = InceptionMNISTModel(
        best_model_params,
        input_channels=input_channels,
        num_classes=num_classes
    ).to(device)
    
    trainer.full_train(best_model, train_loader, val_loader)
    
    # Final evaluation
    print(f"📊 Final evaluation on test set...")
    test_fitness, test_accuracy, _ = evaluate_model(
        best_model,
        test_loader,
        device,
        fitness_method=CONFIG["fitness_method"],
        alpha=CONFIG["fitness_alpha"],
        beta=CONFIG["fitness_beta"]
    )
    
    num_trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    
    print(f"\n🎉 Training completed!")
    print(f"✅ Test accuracy: {test_accuracy:.2f}%")
    print(f"📊 Test fitness: {test_fitness:.4f}")
    print(f"🔢 Model parameters: {num_trainable_params:,}")
    print(f"🧬 Best generation fitness: {cro.best_coral['fitness']:.4f}")
    
    # Save results
    results = {
        "experiment": CONFIG["experiment_name"],
        "dataset": dataset_name,
        "num_classes": num_classes,
        "input_channels": input_channels,
        "best_fitness": cro.best_coral["fitness"],
        "test_fitness": test_fitness,
        "test_accuracy": test_accuracy,
        "param_count": num_trainable_params,
        "fitness_method": CONFIG["fitness_method"],
        "alpha": CONFIG["fitness_alpha"],
        "beta": CONFIG["fitness_beta"],
        "max_generations": CONFIG["max_generations"],
        "num_epochs": CONFIG["num_epochs"],
        "batch_size": CONFIG["batch_size"],
        "learning_rate": CONFIG["learning_rate"],
        "mutation_rate": CONFIG["mutation_rate"],
        "reef_size": str(CONFIG["reef_size"]),
    }
    
    save_results_to_csv(results, CONFIG["results_csv"])
    print(f"💾 Results saved: {CONFIG['results_csv']}")
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(CONFIG["checkpoints_dir"], "best_model.pth")
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'model_params': best_model_params,
            'results': results,
            'config': CONFIG
        }, model_path)
        print(f"💾 Model saved: {model_path}")
    
    print(f"\n📁 All outputs saved to: {base_exp_dir}")

if __name__ == "__main__":
    main()
