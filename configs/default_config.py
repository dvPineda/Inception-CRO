# configs/default_config.py

CONFIG = {
    "experiment_name": "PLACEHOLDER", # Will override in main.py
    "seed": 42,
    "experiments_dir": "experiments", # Base directory for experiments

    # Data loading
    "batch_size": 128,
    "validation_split": 0.1,
    "shuffle_dataset": True,

    # Training (partial training for fitness, full training for final best model)
    "learning_rate": 0.001,
    "num_batches": 100,   # Number of mini-batches for partial training
    "num_epochs": 100,    # Full training epochs
    "patience": 40,       # Early stopping patience

    # Coral Reef Optimization (CRO) parameters
    "reef_size": (20, 10),
    "rho_0": 0.6,
    "Fb": 0.98,
    "Fa": 0.05,
    "Pa": 0.001,
    "Fd": 0.05,
    "Pd": 0.01,
    "kappa": 3,
    "mutation_rate": 0.2,
    "max_generations": 40,
    "max_no_improve": 40,

    # Fitness function parameters
    "fitness_method": "linear",  # possible values: "linear", "poly"
    "fitness_alpha": 7,
    "fitness_beta": 1.0,

    # The following 4 keys will be set dynamically in main.py:
    #   "plots_dir"
    #   "checkpoints_dir"
    #   "visualizations_dir"
    #   "results_csv"
}
