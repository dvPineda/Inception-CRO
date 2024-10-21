import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, branches_params):
        """
        Módulo Inception con ramas dinámicas y tamaños de filtro no cuadrados.

        Args:
            in_channels (int): Número de canales de entrada.
            branches_params (list): Lista de diccionarios con los parámetros de cada rama.
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
        branch_outputs = [branch(x) for branch in self.branches]
        # Ajustar los tamaños de las salidas
        min_height = min(out.size(2) for out in branch_outputs)
        min_width = min(out.size(3) for out in branch_outputs)
        resized_outputs = [
            F.adaptive_avg_pool2d(out, (min_height, min_width))
            if (out.size(2) != min_height or out.size(3) != min_width)
            else out
            for out in branch_outputs
        ]
        outputs = torch.cat(resized_outputs, 1)
        return outputs

class InceptionMNISTModel(nn.Module):
    def __init__(self, model_params):
        """
        Modelo de clasificación MNIST con módulo Inception dinámico.

        Args:
            model_params (dict): Parámetros del modelo, incluyendo 'branches_params'.
        """
        super(InceptionMNISTModel, self).__init__()
        # Raíz (Stem)
        self.conv1 = nn.Conv2d(
            1,
            32,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        # Módulo Inception dinámico
        self.inception = InceptionModule(
            32,
            model_params['branches_params']
        )
        # Calcular el número de canales de salida después del InceptionModule
        self.output_channels = self._get_output_channels(
            32,
            model_params['branches_params']
        )
        # Cabeza (Head)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(
            self.output_channels,
            128
        )
        self.fc2 = nn.Linear(
            128,
            10
        )

    def _get_output_channels(self, in_channels, branches_params):
        """
        Calcula el número total de canales de salida del InceptionModule.

        Args:
            in_channels (int): Número de canales de entrada.
            branches_params (list): Parámetros de las ramas del InceptionModule.

        Returns:
            int: Número total de canales de salida.
        """
        total_channels = 0
        for branch_param in branches_params:
            depth = branch_param.get('depth', 1)
            filter_channels = branch_param.get('filter_channels', [32] * depth)
            total_channels += filter_channels[-1]
        return total_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.inception(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward_inception(self, x):
        """
        Método auxiliar para pasar la entrada directamente al módulo Inception.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Salida del módulo Inception.
        """
        return self.inception(x)
