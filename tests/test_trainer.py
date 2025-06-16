import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from src.trainer import Trainer

def make_binary_dataset(n=20):
    # Simple XOR dataset; two dims, labels 0/1
    X = torch.randn(n, 2)
    Y = (X[:, 0] + X[:, 1] > 0).long()
    ds = TensorDataset(X, Y)
    loader = DataLoader(ds, batch_size=4)
    return loader, 2  # two classes

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)
    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def dummy_cfg(tmp_path):
    # Minimal config dictionary
    return {
        "learning_rate": 1e-3,
        "num_batches": 2,
        "num_epochs": 3,
        "patience": 1
    }

def test_partial_train_runs_without_error(dummy_cfg):
    loader, n_classes = make_binary_dataset()
    # Add required task_type for trainer
    dummy_cfg["task_type"] = "multi-class"
    trainer = Trainer(device="cpu", config=dummy_cfg)  # Fixed: cfg -> config
    model = SimpleModel()
    # Should not raise any exceptions
    trainer.partial_train(model, loader)

def test_full_train_early_stopping_and_checkpoint(tmp_path, dummy_cfg):
    # Build train & val loaders
    train_loader, n_classes = make_binary_dataset(n=10)
    val_loader, _ = make_binary_dataset(n=10)
    # Add required parameters for trainer
    dummy_cfg.update({
        "task_type": "multi-class",
        "fitness_method": "linear",
        "fitness_alpha": 7,
        "fitness_beta": 1.0,
        "checkpoints_dir": str(tmp_path / "ckpt")
    })
    trainer = Trainer(device="cpu", config=dummy_cfg)  # Fixed: cfg -> config
    model = SimpleModel()
    trainer.full_train(model, train_loader, val_loader)
    # After training, best_model.pth should exist
    ckpt_file = tmp_path / "ckpt" / "best_model.pth"
    assert ckpt_file.exists()
