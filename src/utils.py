import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

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

    # Crear conjuntos de entrenamiento y validación usando random_split
    dataset_size = len(train_dataset)
    train_size = int((1 - validation_split) * dataset_size)
    val_size = dataset_size - train_size

    if shuffle_dataset:
        # Establecer semillas para reproducibilidad en la mezcla de datos (opcional)
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

def evaluate_model(model, data_loader, device, alpha=0.0001):
    """
    Evalúa el modelo en un conjunto de datos y devuelve la precisión y la pérdida promedio,
    incluyendo una penalización basada en el número de parámetros del modelo.

    Args:
        model (nn.Module): El modelo a evaluar.
        data_loader (DataLoader): DataLoader del conjunto de datos.
        device (torch.device): Dispositivo ('cpu' o 'cuda').
        alpha (float): Coeficiente de penalización por complejidad.

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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)

    # Calcular el número de parámetros del modelo
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Aplicar penalización por complejidad
    penalty = alpha * np.log(num_params)
    # Calcular fitness
    fitness = accuracy - penalty  # Maximizar fitness

    return fitness, accuracy, avg_loss
