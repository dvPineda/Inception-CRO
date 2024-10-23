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

    # Crear conjuntos de entrenamiento y validación
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(validation_split * dataset_size)
    if shuffle_dataset:
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.random.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.utils.data.random_split(train_dataset, [split, dataset_size - split])
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=val_sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    print("Dataset MNIST cargado correctamente.")
    return train_loader, val_loader, test_loader

def evaluate_model(model, data_loader, device):
    """
    Evalúa el modelo en un conjunto de datos y devuelve la precisión y la pérdida promedio.

    Args:
        model (nn.Module): El modelo a evaluar.
        data_loader (DataLoader): DataLoader del conjunto de datos.
        device (torch.device): Dispositivo ('cpu' o 'cuda').

    Returns:
        tuple: (accuracy, average_loss)
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
    return accuracy, avg_loss
