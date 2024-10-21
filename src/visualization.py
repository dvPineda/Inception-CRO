import os
import torch
import torch.nn as nn
from graphviz import Digraph

def visualize_inception_module(model, generation, idx, output_dir='visualizations'):
    """
    Visualiza y guarda la arquitectura del módulo Inception del modelo en formato de imagen.
    
    Args:
        model (nn.Module): El modelo a visualizar.
        generation (int): Número de la generación actual.
        idx (int): Índice del coral dentro del arrecife.
        output_dir (str): Directorio donde se guardarán las imágenes.
    """
    # Crear un directorio para la generación si no existe
    gen_dir = os.path.join(output_dir, f'generation_{generation}')
    os.makedirs(gen_dir, exist_ok=True)

    # Crear un objeto Digraph de graphviz
    dot = Digraph(comment=f'Inception Module - Coral {idx}')

    # Agregar nodo de entrada
    dot.node('Input', 'Input')

    # Agregar las ramas del módulo Inception
    for branch_idx, branch in enumerate(model.inception.branches):
        branch_name = f'Branch_{branch_idx}'
        dot.node(branch_name, f'Branch {branch_idx}')
        dot.edge('Input', branch_name)
        parent_name = branch_name
        for layer_idx, layer in enumerate(branch):
            layer_name = f'{branch_name}_Layer_{layer_idx}'
            if isinstance(layer, nn.Conv2d):
                kernel_size = layer.kernel_size
                label = f"Conv2d\nin_channels={layer.in_channels}\nout_channels={layer.out_channels}\nkernel_size={kernel_size}"
                dot.node(layer_name, label)
                dot.edge(parent_name, layer_name)
                parent_name = layer_name
            elif isinstance(layer, nn.MaxPool2d):
                kernel_size = layer.kernel_size
                label = f"MaxPool2d\nkernel_size={kernel_size}"
                dot.node(layer_name, label)
                dot.edge(parent_name, layer_name)
                parent_name = layer_name
            # Ignorar otras capas (e.g., ReLU)

    # Guardar el gráfico
    file_path = os.path.join(gen_dir, f'coral_{idx}_inception.gv')
    dot.render(file_path, view=False, format='png')
