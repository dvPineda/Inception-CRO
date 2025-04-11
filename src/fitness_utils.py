# src/fitness_utils.py

import numpy as np

def linear_scale_num_params(num_params: int, alpha: int = 7) -> float:
    """
    Linearly scales the number of parameters (num_params) to [0,1].
    We assume num_params ∈ [1, 10^alpha]. If num_params exceeds 10^alpha,
    the scaled value can exceed 1 (which effectively punishes complexity).
    """
    return (num_params - 1) / (10**alpha - 1)

def compute_fitness(accuracy: float,
                    num_params: int,
                    method: str = 'linear',
                    alpha: int = 7,
                    beta: float = 1.0) -> float:
    """
    Computes a fitness value given an accuracy (in [0,100]) and a number of parameters.

    Args:
        accuracy (float): Model accuracy (0 to 100).
        num_params (int): Number of trainable parameters.
        method (str): Fitness formulation. Options: ['linear', 'poly'].
        alpha (int): Scaling exponent for the linear scaling of num_params.
        beta (float): Penalty/weight factor balancing accuracy vs. complexity.

    Returns:
        float: The computed fitness value.
    """
    # Convert accuracy from [0,100] to [0,1].
    accuracy_norm = accuracy / 100.0

    # Scale the number of parameters to [0,1].
    npt = linear_scale_num_params(num_params, alpha)

    if method == 'linear':
        # f = accuracy_norm - beta * npt
        fitness = accuracy_norm - beta * npt
    elif method == 'poly':
        # A polynomial-based approach. 
        # Example from notebooks:  f = accuracy^2 + (1 - npt)^2,
        # with optional weighting on the second term.
        fitness = (accuracy_norm ** 2) + beta * ((1.0 - npt) ** 2)
    else:
        raise ValueError(f"Unknown fitness method: {method}")

    return fitness
