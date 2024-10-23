import os
import torch
import torch.optim as optim
import torch.nn as nn
import math

from src.models import InceptionMNISTModel
from src.cro import CoralReefOptimization
from src.utils import load_data, evaluate_model

def main():
    # Configuración del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Cargar datos
    train_loader, val_loader, test_loader = load_data()

    # Definir la función de aptitud
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
            accuracy, _ = evaluate_model(model, val_loader, device)
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
    test_accuracy, _ = evaluate_model(best_model, test_loader, device)
    print(f"Precisión del mejor modelo en el conjunto de prueba: {test_accuracy:.2f}%")

if __name__ == '__main__':
    main()
