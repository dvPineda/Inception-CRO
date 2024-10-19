import numpy as np
import random
import copy
import os
from models import InceptionMNISTModel
from visualization import visualize_inception_module

class CoralReefOptimization:
    def __init__(
        self,
        reef_size,
        rho_0,
        Fb,
        Fa,
        Pd,
        kappa,
        mutation_rate,
        fitness_function,
        max_generations
    ):
        """
        Inicializa el algoritmo Coral Reef Optimization.

        Args:
            reef_size (tuple): Tamaño del arrecife (N, M).
            rho_0 (float): Porcentaje inicial de ocupación del arrecife.
            Fb (float): Fracción de corales para reproducción sexual.
            Fa (float): Fracción de corales para reproducción asexual.
            Pd (float): Probabilidad de depredación.
            kappa (int): Número de intentos de asentamiento de larvas.
            mutation_rate (float): Tasa de mutación.
            fitness_function (callable): Función de aptitud.
            max_generations (int): Número máximo de generaciones.
        """
        self.N, self.M = reef_size
        self.rho_0 = rho_0
        self.Fb = Fb
        self.Fa = Fa
        self.Pd = Pd
        self.kappa = kappa
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function
        self.max_generations = max_generations

        self.generation = 0  # Contador de generación
        self.visualization_dir = 'visualizations'
        os.makedirs(self.visualization_dir, exist_ok=True)

        self.reef = self.initialize_reef()
        self.best_coral = None

    def initialize_reef(self):
        reef = np.full((self.N, self.M), None)
        num_initial_corals = int(self.rho_0 * self.N * self.M)

        for _ in range(num_initial_corals):
            while True:
                i, j = random.randint(0, self.N - 1), random.randint(0, self.M - 1)
                if reef[i, j] is None:
                    solution = self.random_solution()
                    fitness = self.fitness_function(solution)
                    reef[i, j] = {'solution': solution, 'fitness': fitness}
                    break

        # Visualizar los corales iniciales
        corals = [
            (idx, reef[i, j]['solution'])
            for idx, (i, j) in enumerate(
                [(i, j) for i in range(self.N) for j in range(self.M) if reef[i, j] is not None]
            )
        ]
        for idx, model_params in corals:
            model = InceptionMNISTModel(model_params)
            try:
                visualize_inception_module(model, self.generation, idx, self.visualization_dir)
            except Exception as e:
                print(f"Error al visualizar el modelo en la generación {self.generation}, coral {idx}: {e}")
        return reef

    def random_solution(self):
        # Generar parámetros aleatorios para el módulo Inception dinámico
        num_branches = random.randint(2, 4)
        branches_params = []
        for _ in range(num_branches):
            depth = random.randint(1, 4)
            filter_sizes = [random.choice([1, 3, 5, 7, 9]) for _ in range(depth)]
            filter_channels = [random.randint(4, 64) for _ in range(depth)]
            use_pooling = random.random() < 0.5
            branch_param = {
                'depth': depth,
                'filter_sizes': filter_sizes,
                'filter_channels': filter_channels,
                'use_pooling': use_pooling
            }
            branches_params.append(branch_param)
        params = {
            'branches_params': branches_params
        }
        return params

    def broadcast_spawning(self):
        selected_corals = self.select_corals(self.Fb)
        new_larvae = []
        for i in range(0, len(selected_corals), 2):
            if i + 1 < len(selected_corals):
                parent1 = selected_corals[i]['solution']
                parent2 = selected_corals[i + 1]['solution']
                new_solution = self.crossover(parent1, parent2)
                new_fitness = self.fitness_function(new_solution)
                new_larvae.append({'solution': new_solution, 'fitness': new_fitness})
        return new_larvae

    def brooding(self):
        selected_corals = self.select_corals(self.Fa)
        new_larvae = []
        for coral in selected_corals:
            new_solution = self.mutate(coral['solution'])
            new_fitness = self.fitness_function(new_solution)
            new_larvae.append({'solution': new_solution, 'fitness': new_fitness})
        return new_larvae

    def budding(self):
        selected_corals = self.select_corals(self.Fa)
        new_larvae = []
        for coral in selected_corals:
            new_solution = copy.deepcopy(coral['solution'])
            # La aptitud del nuevo coral es la misma que la del padre
            new_larvae.append({'solution': new_solution, 'fitness': coral['fitness']})
        return new_larvae

    def larvae_settlement(self, larvae):
        for larva in larvae:
            settled = False
            for _ in range(self.kappa):
                i, j = random.randint(0, self.N - 1), random.randint(0, self.M - 1)
                if self.reef[i, j] is None:
                    self.reef[i, j] = larva
                    settled = True
                    print(f"Larva con fitness {larva['fitness']:.2f} se asentó en posición vacía ({i}, {j}).")
                    break
                else:
                    # Reemplazar si la larva es mejor (mayor fitness)
                    if larva['fitness'] > self.reef[i, j]['fitness']:
                        print(f"Larva con fitness {larva['fitness']:.2f} reemplaza al coral con fitness {self.reef[i, j]['fitness']:.2f} en posición ({i}, {j}).")
                        self.reef[i, j] = larva
                        settled = True
                        break
            if not settled:
                # Si no se pudo asentar, buscar un coral menos apto para reemplazar
                worst_position = None
                worst_fitness = larva['fitness']
                for x in range(self.N):
                    for y in range(self.M):
                        if self.reef[x, y] is not None and self.reef[x, y]['fitness'] < worst_fitness:
                            worst_position = (x, y)
                            worst_fitness = self.reef[x, y]['fitness']
                if worst_position:
                    i, j = worst_position
                    print(f"Larva con fitness {larva['fitness']:.2f} reemplaza al coral con menor fitness {self.reef[i, j]['fitness']:.2f} en posición ({i}, {j}) tras no poder asentarse.")
                    self.reef[i, j] = larva

    def predation(self):
        flat_reef = [
            ((i, j), self.reef[i, j])
            for i in range(self.N)
            for j in range(self.M)
            if self.reef[i, j] is not None
        ]
        num_predated = int(self.Pd * len(flat_reef))
        # Ordenar corales de peor a mejor (menor a mayor fitness)
        sorted_corals = sorted(
            flat_reef,
            key=lambda x: x[1]['fitness']
        )
        for (i, j), coral in sorted_corals[:num_predated]:
            print(f"Depredación: Coral con fitness {coral['fitness']:.2f} en posición ({i}, {j}) ha sido eliminado.")
            self.reef[i, j] = None

    def select_corals(self, fraction):
        flat_reef = [
            coral for row in self.reef for coral in row if coral is not None
        ]
        num_selected = max(1, int(fraction * len(flat_reef)))
        # Selección de los mejores corales (mayor fitness)
        sorted_corals = sorted(flat_reef, key=lambda x: x['fitness'], reverse=True)
        return sorted_corals[:num_selected]

    def crossover(self, parent1, parent2):
        # Cruzar los parámetros de las ramas de los padres
        child = {'branches_params': []}
        branches1 = parent1['branches_params']
        branches2 = parent2['branches_params']
        num_branches = random.choice([len(branches1), len(branches2)])
        for i in range(num_branches):
            if i < len(branches1) and i < len(branches2):
                # Combinar ramas de ambos padres
                if random.random() < 0.5:
                    branch = copy.deepcopy(branches1[i])
                else:
                    branch = copy.deepcopy(branches2[i])
            elif i < len(branches1):
                branch = copy.deepcopy(branches1[i])
            else:
                branch = copy.deepcopy(branches2[i])
            child['branches_params'].append(branch)
        return child

    def mutate(self, solution):
        new_solution = copy.deepcopy(solution)
        # Mutar los parámetros de las ramas
        for branch_param in new_solution['branches_params']:
            if random.random() < self.mutation_rate:
                # Mutar profundidad
                branch_param['depth'] = random.randint(1, 4)
                # Mutar tamaños de filtro y canales
                branch_param['filter_sizes'] = [
                    random.choice([1, 3, 5, 7, 9]) for _ in range(branch_param['depth'])
                ]
                branch_param['filter_channels'] = [
                    random.randint(4, 64) for _ in range(branch_param['depth'])
                ]
                branch_param['use_pooling'] = random.random() < 0.5
            else:
                # Posiblemente mutar parámetros individuales
                for idx in range(branch_param['depth']):
                    if random.random() < self.mutation_rate:
                        branch_param['filter_sizes'][idx] = random.choice([1, 3, 5, 7, 9])
                        branch_param['filter_channels'][idx] = random.randint(4, 64)
        # Posiblemente agregar o eliminar una rama
        if random.random() < self.mutation_rate:
            # Agregar una nueva rama
            depth = random.randint(1, 4)
            filter_sizes = [random.choice([1, 3, 5, 7, 9]) for _ in range(depth)]
            filter_channels = [random.randint(4, 64) for _ in range(depth)]
            use_pooling = random.random() < 0.5
            new_branch = {
                'depth': depth,
                'filter_sizes': filter_sizes,
                'filter_channels': filter_channels,
                'use_pooling': use_pooling
            }
            new_solution['branches_params'].append(new_branch)
        elif len(new_solution['branches_params']) > 2 and random.random() < self.mutation_rate:
            # Eliminar una rama
            idx = random.randint(0, len(new_solution['branches_params']) - 1)
            del new_solution['branches_params'][idx]
        return new_solution

    def update_best_coral(self):
        flat_reef = [
            coral for row in self.reef for coral in row if coral is not None
        ]
        if not flat_reef:
            return
        best_coral = max(flat_reef, key=lambda x: x['fitness'])
        if self.best_coral is None or best_coral['fitness'] > self.best_coral['fitness']:
            self.best_coral = best_coral
            print(f"Nuevo mejor coral encontrado con fitness {self.best_coral['fitness']:.2f}")
            print(f"Parámetros del mejor coral: {self.best_coral['solution']}")

    def run(self):
        for generation in range(self.max_generations):
            self.generation = generation + 1  # Actualizar el contador de generación
            print(f"\n=== Generación {self.generation} ===")
            # Reproducción
            new_larvae = self.broadcast_spawning() + self.brooding() + self.budding()
            # Asentamiento de larvas
            self.larvae_settlement(new_larvae)
            # Depredación
            self.predation()
            # Actualizar el mejor coral
            self.update_best_coral()
            # Visualizar los corales actuales
            flat_reef = [
                (idx, coral['solution'])
                for idx, coral in enumerate(
                    [coral for row in self.reef for coral in row if coral is not None]
                )
            ]
            for idx, model_params in flat_reef:
                model = InceptionMNISTModel(model_params)
                try:
                    visualize_inception_module(
                        model,
                        self.generation,
                        idx,
                        self.visualization_dir
                    )
                except Exception as e:
                    print(f"Error al visualizar el modelo en la generación {self.generation}, coral {idx}: {e}")
            print(f"Mejor fitness en generación {self.generation}: {self.best_coral['fitness']:.2f}")
        print("\nOptimización completada.")
