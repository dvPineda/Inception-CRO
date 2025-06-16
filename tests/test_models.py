import torch
import pytest
from src.models import InceptionModule, InceptionMNISTModel

def make_dummy_branches(num_branches=2, in_channels=3):
    """Build a simple two-branch configuration for testing."""
    branches = []
    for b in range(num_branches):
        # Each branch: depth=2, fixed kernel sizes, fixed channels
        branch = {
            "depth": 2,
            "filter_sizes": [(3, 3), (3, 3)],
            "filter_channels": [8, 8],
            "use_pooling": (b % 2 == 0)  # alternate pooling on/off
        }
        branches.append(branch)
    return branches

def test_inception_module_forward_shape():
    in_ch = 3
    branches = make_dummy_branches(num_branches=2, in_channels=in_ch)
    module = InceptionModule(in_channels=in_ch, branches_params=branches)
    x = torch.randn(4, in_ch, 28, 28)   # batch of 4, 28×28 RGB
    out = module(x)
    # Each branch ends in 8 channels, two branches => 16 output channels
    assert out.shape[0] == 4
    assert out.shape[1] == 16
    # Spatial dims should be unchanged (pooling is stride=1 + adaptive pooling)
    assert out.shape[2] == 28 and out.shape[3] == 28

def test_inception_module_param_count_variations():
    # Config A: one branch, 2 conv layers (1→8, 8→8)
    branches_A = [{
        "depth": 2,
        "filter_sizes": [(3, 3), (3, 3)],
        "filter_channels": [8, 8],
        "use_pooling": False
    }]
    module_A = InceptionModule(in_channels=1, branches_params=branches_A)
    params_A = sum(p.numel() for p in module_A.parameters())

    # Config B: two identical branches (so param count should double)
    branches_B = branches_A + branches_A
    module_B = InceptionModule(in_channels=1, branches_params=branches_B)
    params_B = sum(p.numel() for p in module_B.parameters())

    assert params_B == 2 * params_A

def test_inception_mnist_model_forward_and_output_dims():
    # Build a simple two-branch Inception for grayscale MNIST (1 channel → 10 classes)
    in_ch = 1
    branches = make_dummy_branches(num_branches=2, in_channels=in_ch)
    model = InceptionMNISTModel(
        model_params={"branches_params": branches},
        input_channels=in_ch,
        num_classes=10
    )
    x = torch.randn(2, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (2, 10)
    # Ensure the Inception block's combined out_channels equals sum of last-layer channels
    # In our dummy, each branch’s final filter_channel = 8, so total = 2*8 = 16.
    # After conv1 (1→32 → pool), Inception sees 32→16 channels, then FC maps 16→128→10.
    total_branch_out = sum(bp["filter_channels"][-1] for bp in branches)
    assert model.out_channels == total_branch_out

def test_inception_mnist_model_param_count_consistency():
    # Compare two InceptionMNISTModels with different branch counts
    in_ch = 1
    # One branch:
    branches1 = [{
        "depth": 1,
        "filter_sizes": [(3, 3)],
        "filter_channels": [8],
        "use_pooling": False
    }]
    model1 = InceptionMNISTModel(
        model_params={"branches_params": branches1},
        input_channels=in_ch,
        num_classes=10
    )
    p1 = sum(p.numel() for p in model1.parameters())

    # Two branches (same single-layer conv per branch)
    branches2 = [
        {
            "depth": 1,
            "filter_sizes": [(3, 3)],
            "filter_channels": [8],
            "use_pooling": False
        },
        {
            "depth": 1,
            "filter_sizes": [(3, 3)],
            "filter_channels": [8],
            "use_pooling": False
        }
    ]
    model2 = InceptionMNISTModel(
        model_params={"branches_params": branches2},
        input_channels=in_ch,
        num_classes=10
    )
    p2 = sum(p.numel() for p in model2.parameters())

    # p2 should be strictly larger than p1 (since there’s an extra branch’s worth of convs)
    assert p2 > p1
