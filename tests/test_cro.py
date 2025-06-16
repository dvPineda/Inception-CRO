import pytest
import random
import os
import numpy as np
from copy import deepcopy

from src.cro import CoralReefOptimization

def deterministic_fitness(solution: dict) -> float:
    """
    A dummy fitness function that equals the number of branches.
    This guarantees that any solution with more branches has strictly higher fitness.
    """
    return float(len(solution["branches_params"]))

@pytest.fixture
def small_config():
    return {
        "reef_size": (3, 3),
        "rho_0": 0.5,
        "Fb": 0.5,
        "Fa": 0.5,
        "Pa": 1.0,
        "Fd": 0.5,
        "Pd": 0.5,
        "kappa": 1,
        "mutation_rate": 1.0,   # Ensure every mutation can produce a new branch
        "max_generations": 3,
        "max_no_improve": 2,
        "branch_min": 1,
        "branch_max": 3,
    }

def test_initialize_reef_fills_correct_number(small_config):
    cro = CoralReefOptimization(
        **small_config,
        fitness_function=deterministic_fitness,
        visualization_dir=""  # no need to actually write images in this test
    )
    reef = cro.reef
    total_cells = small_config["reef_size"][0] * small_config["reef_size"][1]
    expected_count = int(small_config["rho_0"] * total_cells)
    actual_count = sum(1 for row in reef for cell in row if cell is not None)
    assert actual_count == expected_count

def test_random_solution_branch_range(small_config):
    cro = CoralReefOptimization(
        **small_config,
        fitness_function=deterministic_fitness,
        visualization_dir=""
    )
    for _ in range(10):
        sol = cro.random_solution()
        assert "branches_params" in sol
        bcount = len(sol["branches_params"])
        assert small_config["branch_min"] <= bcount <= small_config["branch_max"]

def test_crossover_and_mutate_validity(small_config):
    cro = CoralReefOptimization(
        **small_config,
        fitness_function=deterministic_fitness,
        visualization_dir=""
    )
    p1 = cro.random_solution()
    p2 = cro.random_solution()
    child = cro.crossover(p1, p2)
    assert "branches_params" in child
    b1 = len(p1["branches_params"])
    b2 = len(p2["branches_params"])
    bc = len(child["branches_params"])
    assert min(b1, b2) <= bc <= max(b1, b2)

    mutated = cro.mutate(deepcopy(p1))
    assert "branches_params" in mutated
    for branch in mutated["branches_params"]:
        assert set(branch.keys()) == {"depth", "filter_sizes", "filter_channels", "use_pooling"}

def test_cro_evolution_non_decreasing_fitness(tmp_path, small_config):
    """
    Use deterministic_fitness (which equals branch count) with mutation_rate=1.0.
    This ensures that any brooded child may have equal or larger branch count.
    Verify that fitness_history is non-decreasing and that the final best fitness
    is at least the configured branch_max.
    """
    viz_dir = str(tmp_path / "viz_dummy")
    os.makedirs(viz_dir, exist_ok=True)

    cro = CoralReefOptimization(
        **small_config,
        fitness_function=deterministic_fitness,
        visualization_dir=viz_dir
    )
    cro.run()

    # Ensure best_coral is set
    assert cro.best_coral is not None

    # Check that fitness_history is non-decreasing
    for earlier, later in zip(cro.fitness_history, cro.fitness_history[1:]):
        assert later >= earlier

    # Final best fitness should be at least branch_max (3.0)
    assert cro.best_coral["fitness"] >= float(small_config["branch_max"])

def test_larvae_settlement_replaces_lower_fitness(small_config):
    """
    Manually create a 2×2 reef where every coral has fitness=1.0.
    Inject one larva with fitness=3.0. After settlement, at least one cell
    in the reef must be replaced by that higher‐fitness larva.
    """
    cro = CoralReefOptimization(
        **small_config,
        fitness_function=deterministic_fitness,
        visualization_dir=""
    )

    # Build a 2×2 NumPy array reef with fitness=1.0 in every position
    empty_reef = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            sol = {
                "branches_params": [
                    {
                        "depth": 1,
                        "filter_sizes": [(1, 1)],
                        "filter_channels": [4],
                        "use_pooling": False
                    }
                ]
            }
            empty_reef[i, j] = {"solution": sol, "fitness": 1.0}

    cro.reef = empty_reef
    cro.N, cro.M = 2, 2

    # Create a larva with branch_count=3 → fitness = 3.0
    larva_sol = {
        "branches_params": [
            {"depth": 1, "filter_sizes": [(1, 1)], "filter_channels": [4], "use_pooling": False},
            {"depth": 1, "filter_sizes": [(1, 1)], "filter_channels": [4], "use_pooling": False},
            {"depth": 1, "filter_sizes": [(1, 1)], "filter_channels": [4], "use_pooling": False}
        ]
    }
    larva = {"solution": larva_sol, "fitness": 3.0, "attempts": cro.kappa}
    cro.larvae_pool = [larva]

    # Run settlement: since every reef entry has fitness=1.0 and larva=3.0,
    # it must replace at least one cell.
    cro.larvae_settlement()

    any_replaced = False
    for i in range(2):
        for j in range(2):
            if cro.reef[i, j]["fitness"] == 3.0:
                any_replaced = True

    assert any_replaced, "Larva with fitness=3.0 did not replace any coral"
