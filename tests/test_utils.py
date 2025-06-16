import torch
import pytest
import os
import tempfile
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.utils import (
    load_data,
    load_medmnist,
    evaluate_model,
    save_results_to_csv
)

def test_save_results_to_csv(tmp_path):
    # Create a dummy CSV and write one entry
    csv_file = tmp_path / "results.csv"
    result_dict = {
        "experiment": "test_exp",
        "dataset": "mnist",
        "best_fitness": 0.9,
        "test_accuracy": 95.0
    }
    save_results_to_csv(result_dict, str(csv_file))
    df = pd.read_csv(str(csv_file))
    assert "experiment" in df.columns
    assert df.loc[0, "experiment"] == "test_exp"
    # Append again and check row count
    save_results_to_csv(result_dict, str(csv_file))
    df2 = pd.read_csv(str(csv_file))
    assert len(df2) == 2

def test_evaluate_model_trivial():
    # Build a tiny model and a trivial dataset
    import torch.nn as nn
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)
        def forward(self, x):
            return self.linear(x)

    model = TinyModel()
    # Create a DataLoader with four 2-d points, labels 0 or 1
    data = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    ds = TensorDataset(data, labels)
    loader = DataLoader(ds, batch_size=2)
    # Evaluate (loss should be defined, accuracy ~50% since untrained)
    fitness, accuracy, loss = evaluate_model(model, loader, device="cpu",
                                            fitness_method="linear",
                                            alpha=2, beta=0.5)
    assert 0.0 <= accuracy <= 100.0
    assert loss >= 0.0
    assert isinstance(fitness, float)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU for MedMNIST download")
def test_load_medmnist_smoke():
    # This will only run if GPU is available, because MedMNIST can be heavy
    train_loader, val_loader, test_loader, n_classes, n_channels, task_type = load_medmnist(
        batch_size=4, subset="pathmnist", validation_split=0.1, shuffle_dataset=False, random_seed=42
    )
    # Basic shape checks
    xs, ys = next(iter(train_loader))
    assert xs.ndim == 4  # batch, C, H, W
    assert isinstance(n_classes, int)
    assert isinstance(n_channels, int)
    assert task_type in ["multi-class", "multi-label"]

def test_load_data_mnist_smoke():
    train_loader, val_loader, test_loader, n_classes, n_channels, task_type = load_data(
        batch_size=4, validation_split=0.1, shuffle_dataset=False, random_seed=123
    )
    xs, ys = next(iter(train_loader))
    assert xs.shape[1:] == (1, 28, 28)
    assert n_classes == 10
    assert n_channels == 1
    assert task_type == "multi-class"
