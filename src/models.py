# src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    """
    A dynamic Inception-like module with variable branches, depths, and filter sizes.
    Each branch can optionally include a MaxPool2d layer if indicated by 'use_pooling'.
    """

    def __init__(self, in_channels, branches_params):
        """
        Args:
            in_channels (int): Number of input channels (e.g. from a previous convolution).
            branches_params (list): A list of dictionaries, each describing a branch. 
                Example of branch_param:
                    {
                        'depth': 2,
                        'filter_sizes': [(3, 3), (5, 5)],
                        'filter_channels': [32, 64],
                        'use_pooling': True
                    }
        """
        super(InceptionModule, self).__init__()
        self.branches = nn.ModuleList()

        for branch_param in branches_params:
            layers = []
            current_in_channels = in_channels
            depth = branch_param.get('depth', 1)
            filter_sizes = branch_param.get('filter_sizes', [(1, 1)] * depth)
            filter_channels = branch_param.get('filter_channels', [32] * depth)
            use_pooling = branch_param.get('use_pooling', False)

            # Build each layer in the branch
            for idx in range(depth):
                kernel_size = filter_sizes[idx]
                padding = (
                    kernel_size[0] // 2,
                    kernel_size[1] // 2
                )
                out_channels = filter_channels[idx]
                layers.append(
                    nn.Conv2d(
                        current_in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=padding
                    )
                )
                current_in_channels = out_channels

            if use_pooling:
                layers.append(
                    nn.MaxPool2d(
                        kernel_size=(3, 3),
                        stride=1,
                        padding=1
                    )
                )

            self.branches.append(nn.Sequential(*layers))

    def forward(self, x):
        """
        Forward pass of the InceptionModule. 
        Concatenates the outputs from all branches across the channel dimension.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Concatenated feature map combining all branch outputs.
        """
        branch_outputs = [branch(x) for branch in self.branches]

        # Some branches may alter spatial dimension (especially if pooling is used).
        # To concatenate, we adapt them to the smallest (H, W) among branches 
        # via adaptive_avg_pool2d or any consistent approach.
        min_height = min(out.size(2) for out in branch_outputs)
        min_width = min(out.size(3) for out in branch_outputs)

        resized_outputs = [
            F.adaptive_avg_pool2d(out, (min_height, min_width))
            if (out.size(2) != min_height or out.size(3) != min_width)
            else out
            for out in branch_outputs
        ]

        outputs = torch.cat(resized_outputs, dim=1)
        return outputs


class InceptionMNISTModel(nn.Module):
    """
    An Inception-based model configured for MNIST (single-channel, 28x28 images).
    Uses a stem layer, a dynamic InceptionModule, global pooling, and a small MLP head.
    """

    def __init__(self, model_params):
        """
        Args:
            model_params (dict): Dict with at least 'branches_params', describing 
                                 how the InceptionModule should be built.
        """
        super(InceptionMNISTModel, self).__init__()
        # Stem: single Conv2d for input dimension (1 channel for MNIST)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # InceptionModule
        self.inception = InceptionModule(
            in_channels=32,
            branches_params=model_params.get('branches_params', [])
        )

        # Compute output channels after the InceptionModule
        self.output_channels = self._get_output_channels(
            in_channels=32,
            branches_params=model_params.get('branches_params', [])
        )

        # Head: global average pooling + linear layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.output_channels, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST

    def _get_output_channels(self, in_channels, branches_params):
        """
        Calculates the total number of output channels from the InceptionModule
        by summing the final out_channels of each branch.

        Args:
            in_channels (int): number of input channels to the inception module (unused in the sum).
            branches_params (list): each item is a dict describing a branch.

        Returns:
            int: total number of output channels after concatenating all branches.
        """
        total_channels = 0
        for branch_param in branches_params:
            depth = branch_param.get('depth', 1)
            filter_channels = branch_param.get('filter_channels', [32] * depth)
            total_channels += filter_channels[-1]
        return total_channels

    def forward(self, x):
        """
        Forward pass through the stem, Inception module, and classification head.

        Args:
            x (torch.Tensor): Input image batch, shape (B, 1, 28, 28) for MNIST.

        Returns:
            torch.Tensor: Raw logits with shape (B, 10).
        """
        x = self.conv1(x)
        x = self.pool(x)
        x = self.inception(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
