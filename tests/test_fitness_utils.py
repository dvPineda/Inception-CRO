import pytest
from src.fitness_utils import (
    linear_scale_num_params,
    compute_fitness
)

def test_linear_scale_num_params_below_alpha():
    # If num_params = 1 should map to 0.0 when alpha >= 1
    val = linear_scale_num_params(num_params=1, alpha=3)
    assert pytest.approx(val, rel=1e-6) == 0.0

    # If num_params = 10^alpha, then scaled = 1.0
    max_val = 10 ** 3
    val2 = linear_scale_num_params(num_params=max_val, alpha=3)
    assert pytest.approx(val2, rel=1e-6) == 1.0

def test_linear_scale_num_params_above_alpha():
    # If num_params > 10^alpha, result can exceed 1.0
    val3 = linear_scale_num_params(num_params=(10**2 + 50), alpha=2)
    assert val3 > 1.0

def test_compute_fitness_linear():
    # accuracy = 80% -> 0.80, num_params small => penalty near 0
    fitness = compute_fitness(accuracy=80.0, num_params=100, method="linear", alpha=4, beta=0.1)
    # accuracy_norm = 0.8, npt = (100 -1)/(10^4 -1) ≈ 0.0099 -> penalty ~0.00099
    assert pytest.approx(fitness, rel=1e-3) == 0.8 - 0.1 * ((100 - 1) / (10**4 - 1))

def test_compute_fitness_poly():
    # Test the polynomial variant
    fitness_poly = compute_fitness(accuracy=50.0, num_params=500, method="poly", alpha=3, beta=1.0)
    acc_norm = 0.5
    npt = (500 - 1) / (10**3 - 1)
    expected = acc_norm**2 + 1.0 * (1 - npt)**2
    assert pytest.approx(fitness_poly, rel=1e-4) == expected

def test_compute_fitness_invalid_method():
    with pytest.raises(ValueError):
        compute_fitness(accuracy=10.0, num_params=10, method="bogus", alpha=1, beta=1.0)
