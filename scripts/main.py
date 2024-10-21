import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

# Importar las clases definidas en otros archivos
from cro import CoralReefOptimization
from models import InceptionMNISTModel

# Configuración de dispositivos
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el dataset MNIST
def load_data():
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
    validation_split = 0.1
    shuffle_dataset = True
    random_seed = 42

    # Crear conjuntos de entrenamiento y validación
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        train_dataset,
        batch_size=128,
        sampler=val_sampler
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False
    )
    print("Dataset MNIST cargado correctamente.")
    return train_loader, val_loader, test_loader

def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
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

if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data()

    def fitness_function(model_params):
        try:
            model = InceptionMNISTModel(model_params).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            model.train()
            num_batches = 2  # Reducido para acelerar las pruebas
            for batch_idx, (images, labels) in enumerate(train_loader):
                if batch_idx >= num_batches:
                    break
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            # Evaluar en el conjunto de validación
            model.eval()
            with torch.no_grad():
                total = 0
                correct = 0
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            # Calcular el número de parámetros del modelo
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # Penalización por complejidad (ajustar alpha según sea necesario)
            alpha = 0.0001  # Coeficiente de penalización
            penalty = alpha * math.log(num_params)
            # Función de aptitud combinada
            fitness = accuracy - penalty  # Maximizar fitness
            return fitness
        except Exception as e:
            print(f"Evaluación del modelo fallida: {e}")
            return 0  # Peor aptitud posible

    # Parámetros del CRO - actualmente usando valores validados en el TFG
    cro = CoralReefOptimization(
        reef_size=(3, 3),  # Arrecife más pequeño
        rho_0=0.6,
        Fb=0.98,
        Fa=0.05,
        Pd=0.05,
        kappa=3,
        mutation_rate=0.2,
        fitness_function=fitness_function,
        max_generations=3
    )

    print(" ======================== ")
    cro.run()
    print(" ======================== ")

    # Evaluar el mejor modelo en el conjunto de prueba
    best_model_params = cro.best_coral['solution']
    best_model = InceptionMNISTModel(best_model_params).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # Entrenar el mejor modelo completo
    epochs = 10
    for epoch in range(epochs):
        best_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = best_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")
    # Evaluar en el conjunto de prueba
    test_accuracy, _ = evaluate_model(best_model, test_loader)
    print(f"Precisión del mejor modelo en el conjunto de prueba: {test_accuracy:.2f}%")
