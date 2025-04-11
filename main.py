# main.py

import os
import torch
import datetime
import matplotlib.pyplot as plt
import numpy as np

from configs.default_config import CONFIG
from src.cro import CoralReefOptimization
from src.trainer import Trainer
from src.utils import load_data, evaluate_model, save_results_to_csv
from src.models import InceptionMNISTModel

def main():
    # 1. Use a unique timestamp in your experiment name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dynamic_experiment_name = f"MNIST_InceptionCRO_{timestamp}"

    # 2. Override the placeholder in config
    CONFIG["experiment_name"] = dynamic_experiment_name

    # 3. Create a base directory for this experiment
    base_exp_dir = os.path.join(CONFIG["experiments_dir"], dynamic_experiment_name)

    # 4. Define subfolders and files
    CONFIG["plots_dir"] = os.path.join(base_exp_dir, "plots")
    CONFIG["checkpoints_dir"] = os.path.join(base_exp_dir, "checkpoints")
    CONFIG["visualizations_dir"] = os.path.join(base_exp_dir, "visualizations")
    CONFIG["results_csv"] = os.path.join(base_exp_dir, "results.csv")

    # 5. Ensure directories exist
    os.makedirs(CONFIG["plots_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoints_dir"], exist_ok=True)
    os.makedirs(CONFIG["visualizations_dir"], exist_ok=True)

    # --- The rest of your main code remains as is ---
    print(f"Experiment Name: {CONFIG['experiment_name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seeds for reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Load data
    train_loader, val_loader, test_loader = load_data(
        batch_size=CONFIG["batch_size"],
        validation_split=CONFIG["validation_split"],
        shuffle_dataset=CONFIG["shuffle_dataset"],
        random_seed=CONFIG["seed"]
    )

    # Create a Trainer
    trainer = Trainer(device, CONFIG)

    def fitness_function(model_params):
        # Partial train + evaluate
        try:
            model = InceptionMNISTModel(model_params).to(device)
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
            print(f"Evaluación fallida: {e}")
            return 0.0

    # Instantiate CRO, passing the custom visualization directory
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
        visualization_dir=CONFIG["visualizations_dir"]  # NEW
    )

    cro.run()

    # Plot the fitness evolution
    generations = range(1, len(cro.fitness_history) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(generations, cro.fitness_history, label='Mejor Fitness')
    plt.plot(generations, cro.avg_fitness_history, label='Fitness Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evolución del Fitness a lo Largo de las Generaciones')
    plt.legend()
    plt.grid(True)

    # Name the plot with the dynamic experiment name
    plot_filename = os.path.join(CONFIG["plots_dir"], f"{CONFIG['experiment_name']}_fitness.png")
    plt.savefig(plot_filename)
    print(f"Fitness convergence plot guardado en: {plot_filename}")

    # Full train best model
    best_model_params = cro.best_coral['solution']
    best_model = InceptionMNISTModel(best_model_params).to(device)
    trainer.full_train(best_model, train_loader, val_loader)

    # Evaluate on test
    test_fitness, test_accuracy, _ = evaluate_model(
        best_model,
        test_loader,
        device,
        fitness_method=CONFIG["fitness_method"],
        alpha=CONFIG["fitness_alpha"],
        beta=CONFIG["fitness_beta"]
    )
    print(f"Precisión del mejor modelo en el conjunto de prueba: {test_accuracy:.2f}%")

    num_trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    results = {
        "experiment": CONFIG["experiment_name"],
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
    }

    # Store results.csv in your experiment folder
    save_results_to_csv(results, CONFIG["results_csv"])
    print(f"Resultados guardados en: {CONFIG['results_csv']}")

if __name__ == "__main__":
    main()
