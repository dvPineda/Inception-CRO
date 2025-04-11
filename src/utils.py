# src/utils.py

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.fitness_utils import compute_fitness
import csv
import os

def load_data(batch_size=128, validation_split=0.1, shuffle_dataset=True, random_seed=42):
    """
    Carga el conjunto de datos MNIST y devuelve los loaders de entrenamiento, validación y prueba.

    Args:
        batch_size (int): Tamaño del lote.
        validation_split (float): Porcentaje del conjunto de entrenamiento utilizado para validación.
        shuffle_dataset (bool): Si se mezclan los datos antes de dividir.
        random_seed (int): Semilla para la reproducibilidad.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("Cargando el dataset MNIST...")

    # Ejemplo de transformaciones para MNIST (sin data augmentation, 
    # aunque puedes añadir transforms.RandomRotation, etc. si deseas)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Crear conjuntos de entrenamiento y validación
    dataset_size = len(train_dataset)
    train_size = int((1 - validation_split) * dataset_size)
    val_size = dataset_size - train_size

    if shuffle_dataset:
        # Establecer semillas para reproducibilidad
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_dataset
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_dataset
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print("Dataset MNIST cargado correctamente.")
    return train_loader, val_loader, test_loader


def evaluate_model(model,
                   data_loader,
                   device,
                   fitness_method='linear',
                   alpha=7,
                   beta=1.0):
    """
    Evalúa el modelo en un conjunto de datos y devuelve la aptitud, la precisión y la 
    pérdida promedio usando la función de fitness elegida (e.g. 'linear', 'poly').

    Args:
        model (nn.Module): El modelo a evaluar.
        data_loader (DataLoader): DataLoader del conjunto de datos.
        device (torch.device): Dispositivo ('cpu' o 'cuda').
        fitness_method (str): Opción para la función de fitness. Default: 'linear'.
        alpha (int): Parámetro de escalado para el número de parámetros. Default: 7.
        beta (float): Factor de penalización/peso. Default: 1.0.

    Returns:
        tuple: (fitness, accuracy, average_loss)
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(data_loader)

    # Calcular el número de parámetros del modelo
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calcular el fitness usando la función importada de fitness_utils.py
    fitness = compute_fitness(
        accuracy=accuracy,
        num_params=num_params,
        method=fitness_method,
        alpha=alpha,
        beta=beta
    )

    return fitness, accuracy, avg_loss


def save_results_to_csv(results, filename):
    """
    Guarda los resultados de la ejecución en un archivo CSV, añadiendo 
    cabeceras si el archivo no existe.
    
    Args:
        results (dict): Diccionario con los resultados y parámetros a registrar.
        filename (str): Ruta al archivo CSV donde guardar los datos.
    """
    file_exists = os.path.isfile(filename)
    fieldnames = list(results.keys())

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)
