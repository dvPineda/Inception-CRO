# src/visualization.py

import os
import torch.nn as nn
from graphviz import Digraph

def visualize_inception_module(model, generation, idx, output_dir='visualizations'):
    """
    Visualiza y guarda la arquitectura del módulo Inception del modelo en formato de imagen,
    incluyendo un bloque final 'Output' que muestra cómo se concatenan las salidas de todas las ramas.

    Args:
        model (nn.Module): El modelo a visualizar (se asume que contiene un atributo 'inception').
        generation (int): Número de la generación actual.
        idx (int or str): Índice del coral dentro del arrecife o 'best_coral'.
        output_dir (str): Directorio donde se guardarán las imágenes.
    """
    # Crear un directorio para la generación si no existe
    if idx == 'best_coral':
        gen_dir = output_dir  # Guardar en la carpeta 'best_coral'
    else:
        gen_dir = os.path.join(output_dir, f'generation_{generation}')
    os.makedirs(gen_dir, exist_ok=True)

    # Crear un objeto Digraph de graphviz
    if idx == 'best_coral':
        dot = Digraph(comment=f'Best Coral - Generation {generation}')
    else:
        dot = Digraph(comment=f'Inception Module - Coral {idx}')

    # Agregar nodo de entrada
    dot.node('Input', 'Input')

    # Lista para almacenar los nombres de los últimos nodos de cada rama
    branch_output_names = []

    # Verificar que el modelo tenga un atributo 'inception'
    # Este script asume un atributo model.inception.branches
    if not hasattr(model, 'inception'):
        dot.node('Error', 'Model does not have an `inception` attribute.')
        file_name = f'error_coral_{idx}'
        file_path = os.path.join(gen_dir, file_name)
        dot.render(file_path, view=False, format='png')
        return

    # Recorrer las ramas del InceptionModule
    for branch_idx, branch in enumerate(model.inception.branches):
        branch_name = f'Branch_{branch_idx}'
        dot.node(branch_name, f'Branch {branch_idx}')
        dot.edge('Input', branch_name)
        parent_name = branch_name

        for layer_idx, layer in enumerate(branch):
            layer_name = f'{branch_name}_Layer_{layer_idx}'

            if isinstance(layer, nn.Conv2d):
                kernel_size = layer.kernel_size
                label = (f"Conv2d\n"
                         f"in_channels={layer.in_channels}\n"
                         f"out_channels={layer.out_channels}\n"
                         f"kernel_size={kernel_size}")
                dot.node(layer_name, label)
                dot.edge(parent_name, layer_name)
                parent_name = layer_name

            elif isinstance(layer, nn.MaxPool2d):
                kernel_size = layer.kernel_size
                label = f"MaxPool2d\nkernel_size={kernel_size}"
                dot.node(layer_name, label)
                dot.edge(parent_name, layer_name)
                parent_name = layer_name

            # If there were other layers to visualize (e.g. BatchNorm, ReLU),
            # we could add them here. For now, we only highlight Conv2d / MaxPool2d.

        # Al finalizar la rama, almacenar el nombre del último nodo
        branch_output_names.append(parent_name)

    # Agregar nodo de salida y conectar las ramas
    dot.node('Output', 'Output (Concatenation)')
    for output_name in branch_output_names:
        dot.edge(output_name, 'Output')

    # Guardar el gráfico
    if idx == 'best_coral':
        file_name = f'generation_{generation}_best_coral'
    else:
        file_name = f'coral_{idx}_inception'

    file_path = os.path.join(gen_dir, file_name)
    dot.render(file_path, view=False, format='png')
