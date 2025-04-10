import os
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import math
import csv
import matplotlib.pyplot as plt

from src.models import InceptionMNISTModel
from src.cro import CoralReefOptimization
from src.utils import load_data, evaluate_model

def main():
    # Configuración del dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")

    # Cargar datos
    train_loader, val_loader, test_loader = load_data(shuffle_dataset=True)

    # Definir la función de aptitud
    def fitness_function(model_params):
        try:
            model = InceptionMNISTModel(model_params).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            model.train()
            num_batches = 100  # Número de lotes para entrenar (ajustar según sea necesario)
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
            fitness, accuracy, _ = evaluate_model(model, val_loader, device)
            return fitness
        except Exception as e:
            print(f"Evaluación del modelo fallida: {e}")
            return 0  # Peor aptitud posible

    # Parámetros del CRO
    cro = CoralReefOptimization(
        reef_size=(20, 10),
        rho_0=0.6,
        Fb=0.98,
        Fa=0.05,
        Pa=0.001,
        Fd=0.05,
        Pd=0.01,
        kappa=3,
        mutation_rate=0.2,
        fitness_function=fitness_function,
        max_generations=100,
        max_no_improve=40
    )

    print(" ======================== ")
    cro.run()
    print(" ======================== ")

    # Graficar el mejor fitness y el fitness promedio a lo largo de las generaciones
    generations = range(1, len(cro.fitness_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(generations, cro.fitness_history, label='Mejor Fitness')
    plt.plot(generations, cro.avg_fitness_history, label='Fitness Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evolución del Fitness a lo Largo de las Generaciones')
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica
    plot_filename = 'fitness_convergence.png'
    plt.savefig(plot_filename)
    print(f"Fitness convergence plot guardado como: {plot_filename}")

    # Evaluar el mejor modelo en el conjunto de prueba
    best_model_params = cro.best_coral['solution']
    best_model = InceptionMNISTModel(best_model_params).to(device)
    optimizer = optim.Adam(best_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    # Parámetros para Early Stopping
    patience = 40  # Número de épocas sin mejora para detener el entrenamiento
    best_val_loss = float('inf')
    epochs_no_improve = 0
    num_epochs = 300  # Máximo número de épocas
    val_loss_history = []

    
    for epoch in range(num_epochs):
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
        avg_train_loss = running_loss / len(train_loader)

        # Evaluar en el conjunto de validación
        best_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = best_model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Comprobar si hay mejora en la pérdida de validación
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Opcional: Guardar el modelo si es el mejor hasta ahora
            torch.save(best_model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    # Cargar el mejor modelo guardado (opcional)
    best_model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # Evaluar en el conjunto de prueba
    test_fitness, test_accuracy, _ = evaluate_model(best_model, test_loader, device)
    print(f"Precisión del mejor modelo en el conjunto de prueba: {test_accuracy:.2f}%")


    # Mostrar el número de parametros totales y entrenables del mejor modelo
    num_total_params = sum(p.numel() for p in best_model.parameters())
    num_trainable_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    print(f"Número total de parámetros del mejor modelo: {num_total_params}")
    print(f"Número de parámetros entrenables del mejor modelo: {num_trainable_params}")


    # Guardar resultados en un archivo CSV
    results = {
        'reef_size': cro.N * cro.M,
        'rho_0': cro.rho_0,
        'Fb': cro.Fb,
        'Fa': cro.Fa,
        'Pd': cro.Pd,
        'kappa': cro.kappa,
        'mutation_rate': cro.mutation_rate,
        'max_generations': cro.max_generations,
        'best_fitness': cro.best_coral['fitness'],
        'test_fitness': test_fitness,
        'test_accuracy': test_accuracy,
        'num_total_params': num_total_params,
        'best_model_params': best_model_params
    }

    output_file = 'cro_results.csv'
    save_results_to_csv(results, output_file)
    print(f"Resultados guardados en {output_file}")

def save_results_to_csv(results, filename):
    """
    Guarda los resultados de la ejecución en un archivo CSV.

    Args:
        results (dict): Diccionario con los resultados y parámetros.
        filename (str): Nombre del archivo CSV.
    """
    fieldnames = list(results.keys())
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Escribir encabezados si el archivo es nuevo

        writer.writerow(results)

if __name__ == '__main__':
    main()
