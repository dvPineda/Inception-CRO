import numpy as np
import random
import copy
import os
from src.models import InceptionMNISTModel
from src.visualization import visualize_inception_module

class CoralReefOptimization:
    def __init__(
        self,   
        reef_size,
        rho_0,
        Fb,
        Fa,
        Pa,
        Fd,
        kappa,
        Pd,
        mutation_rate,
        fitness_function,
        max_generations,
        max_no_improve=None, # Nuevo parámetro para early stopping
    ):
        """
        Inicializa los parámetros del algoritmo CRO y configura el arrecife.

        Args:
            reef_size (tuple): Tamaño del arrecife (N, M).
            rho_0 (float): Porcentaje inicial de ocupación del arrecife.
            Fb (float): Fracción de corales para reproducción sexual.
            Fa (float): Fracción de corales para reproducción asexual (budding).
            Pa (float): Probabilidad de reproducción asexual.
            Fd (float): Fracción de corales que serán depredados.
            Pd (float): Probabilidad adicional para la depredación.
            kappa (int): Número máximo de intentos de asentamiento de una larva.
            mutation_rate (float): Tasa de mutación.
            fitness_function (callable): Función para evaluar la aptitud de una solución.
            max_generations (int): Número máximo de generaciones.
            max_no_improve (int, optional): Número máximo de generaciones sin mejora para detener el algoritmo.
        """
        self.N, self.M = reef_size
        self.rho_0 = rho_0
        self.Fb = Fb
        self.Fa = Fa
        self.Pa = Pa
        self.Fd = Fd
        self.Pd = Pd
        self.kappa = kappa
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function
        self.max_generations = max_generations
        self.max_no_improve = max_no_improve  # Parada anticipada
        self.no_improve_counter = 0  # Contador de generaciones sin mejora


        self.generation = 0  # Contador de generación
        self.visualization_dir = 'visualizations'
        os.makedirs(self.visualization_dir, exist_ok=True)

        self.reef = self.initialize_reef()
        self.best_coral = None
        self.fitness_history = []  # Historial del mejor fitness
        self.avg_fitness_history = [] # Historial del fitness promedio

        self.larvae_pool = []  # Piscina de larvas que persiste entre generaciones

    def initialize_reef(self):
        """
        Inicializa el arrecife con corales aleatorios según la ocupación inicial rho_0.

        Returns:
            np.ndarray: Arrecife inicializado con corales.
        """
        reef = np.full((self.N, self.M), None)
        num_initial_corals = int(self.rho_0 * self.N * self.M)
        placed_corals = 0

        while placed_corals < num_initial_corals:
            i, j = random.randint(0, self.N - 1), random.randint(0, self.M - 1)
            if reef[i, j] is None:
                solution = self.random_solution()
                fitness = self.fitness_function(solution)
                reef[i, j] = {'solution': solution, 'fitness': fitness}
                placed_corals += 1

        return reef

    def random_solution(self):
        """
        Genera una solución aleatoria (arquitectura del módulo Inception).

        Returns:
            dict: Parámetros de la solución generada.
        """
        num_branches = random.randint(2, 4)
        branches_params = []
        possible_heights = [1, 3, 5, 7, 9]
        possible_widths = [1, 3, 5, 7, 9]
        for _ in range(num_branches):
            depth = random.randint(1, 4)
            filter_sizes = [
                (
                    random.choice(possible_heights),
                    random.choice(possible_widths)
                )
                for _ in range(depth)
            ]
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

    def broadcast_spawning(self, broadcast_corals):
        """
        Implementa la reproducción sexual (broadcast spawning).

        Args:
            broadcast_corals (list): Lista de corales seleccionados para broadcast spawning.

        Returns:
            list: Lista de nuevas larvas generadas.
        """
        new_larvae = []

        num_parents = len(broadcast_corals)
        if num_parents < 2:
            return new_larvae  # No hay suficientes corales para reproducirse

        # Asegurar que tenemos un número par de padres
        if num_parents % 2 != 0:
            # Eliminar aleatoriamente un coral para tener pares completos
            removed_coral = random.choice(broadcast_corals)
            broadcast_corals.remove(removed_coral)
            num_parents -= 1

        # Mezclar aleatoriamente los corales seleccionados
        random.shuffle(broadcast_corals)

        # Emparejar los corales sin repetición
        for i in range(0, num_parents, 2):
            parent1 = broadcast_corals[i]['solution']
            parent2 = broadcast_corals[i + 1]['solution']
            new_solution = self.crossover(parent1, parent2)
            new_fitness = self.fitness_function(new_solution)
            # Inicializar contador de intentos para la larva
            larva = {'solution': new_solution, 'fitness': new_fitness, 'attempts': self.kappa}
            new_larvae.append(larva)

        return new_larvae

    def brooding(self, brooding_corals):
        """
        Implementa la reproducción asexual por brooding (mutación) sobre los corales no seleccionados en broadcast spawning.

        Args:
            brooding_corals (list): Lista de corales para brooding.

        Returns:
            list: Lista de nuevas larvas generadas.
        """
        new_larvae = []
        for coral in brooding_corals:
            new_solution = self.mutate(coral['solution'])
            new_fitness = self.fitness_function(new_solution)
            # Inicializar contador de intentos para la larva
            larva = {'solution': new_solution, 'fitness': new_fitness, 'attempts': self.kappa}
            new_larvae.append(larva)
        return new_larvae

    def budding(self):
        """
        Implementa la reproducción asexual por gemación (budding) para una fracción Fa de corales.

        Returns:
            list: Lista de nuevas larvas generadas.
        """
        # Seleccionar corales para budding
        corals = [coral for row in self.reef for coral in row if coral is not None]
        num_budding = int(self.Fa * len(corals))
        if num_budding == 0:
            return []

        # Si no se cumple la probabilidad Pa, no se generan nuevas larvas
        if np.random.uniform(0, 1) > self.Pa:
            return []

        # Seleccionar los mejores corales para budding
        sorted_corals = sorted(corals, key=lambda x: x['fitness'], reverse=True)
        selected_corals = sorted_corals[:num_budding]

        new_larvae = []
        for coral in selected_corals:
            new_solution = copy.deepcopy(coral['solution'])
            # La aptitud de la nueva larva es la misma que la del padre
            larva = {'solution': new_solution, 'fitness': coral['fitness'], 'attempts': self.kappa}
            new_larvae.append(larva)
        return new_larvae

    def larvae_settlement(self):
        """
        Gestiona el asentamiento de las larvas en el arrecife desde el larvae_pool.
        """
        new_pool = []
        for larva in self.larvae_pool:
            settled = False
            # Intento de asentamiento una vez por generación
            i, j = random.randint(0, self.N - 1), random.randint(0, self.M - 1)
            if self.reef[i, j] is None or larva['fitness'] > self.reef[i, j]['fitness']:
                self.reef[i, j] = {'solution': larva['solution'], 'fitness': larva['fitness']}
                settled = True
                print(f"Larva con fitness {larva['fitness']:.2f} se asentó en posición ({i}, {j}).")
            else:
                # Reducir contador de intentos
                larva['attempts'] -= 1
                if larva['attempts'] > 0:
                    new_pool.append(larva)
                else:
                    print(f"Larva con fitness {larva['fitness']:.2f} ha agotado sus intentos y será eliminada.")
        # Actualizar larvae_pool con las larvas que aún tienen intentos restantes
        self.larvae_pool = new_pool

    def predation(self):
        """
        Simula la depredación eliminando los corales menos aptos del arrecife con una probabilidad adicional.
        """
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
            if random.random() < self.Fd:
                print(f"Depredación: Coral con fitness {coral['fitness']:.2f} en posición ({i}, {j}) ha sido eliminado.")
                self.reef[i, j] = None

    def crossover(self, parent1, parent2):
        """
        Realiza el cruce entre dos soluciones (padres).

        Args:
            parent1 (dict): Parámetros del primer padre.
            parent2 (dict): Parámetros del segundo padre.

        Returns:
            dict: Parámetros de la nueva solución generada.
        """
        # Obtener el número de ramas de los padres
        num_branches_parent1 = len(parent1['branches_params'])
        num_branches_parent2 = len(parent2['branches_params'])
        min_branches = min(num_branches_parent1, num_branches_parent2)
        max_branches = max(num_branches_parent1, num_branches_parent2)

        # Elegir aleatoriamente el número de ramas del hijo entre min y max
        num_branches_child = random.randint(min_branches, max_branches)

        # Generar las ramas del hijo tomando aleatoriamente de los padres
        child_branches = []
        for _ in range(num_branches_child):
            # Seleccionar aleatoriamente un padre
            if random.random() < 0.5:
                selected_parent = parent1
            else:
                selected_parent = parent2
            # Seleccionar aleatoriamente una rama del padre seleccionado
            selected_branches = selected_parent['branches_params']
            branch = copy.deepcopy(random.choice(selected_branches))
            child_branches.append(branch)

        child = {'branches_params': child_branches}
        return child

    def mutate(self, solution):
        """
        Aplica mutaciones a una solución dada.

        Args:
            solution (dict): Parámetros de la solución a mutar.

        Returns:
            dict: Parámetros de la nueva solución mutada.
        """
        new_solution = copy.deepcopy(solution)
        # Mutar los parámetros de las ramas
        possible_heights = [1, 3, 5, 7, 9]
        possible_widths = [1, 3, 5, 7, 9]
        for branch_param in new_solution['branches_params']:
            if random.random() < self.mutation_rate:
                # Mutar profundidad
                branch_param['depth'] = random.randint(1, 4)
                # Mutar tamaños de filtro y canales
                branch_param['filter_sizes'] = [
                    (
                        random.choice(possible_heights),
                        random.choice(possible_widths)
                    )
                    for _ in range(branch_param['depth'])
                ]
                branch_param['filter_channels'] = [
                    random.randint(4, 64) for _ in range(branch_param['depth'])
                ]
                branch_param['use_pooling'] = random.random() < 0.5
            else:
                # Posiblemente mutar parámetros individuales
                for idx in range(branch_param['depth']):
                    if random.random() < self.mutation_rate:
                        branch_param['filter_sizes'][idx] = (
                            random.choice(possible_heights),
                            random.choice(possible_widths)
                        )
                        branch_param['filter_channels'][idx] = random.randint(4, 64)
        # Posiblemente agregar o eliminar una rama
        if random.random() < self.mutation_rate:
            # Agregar una nueva rama
            depth = random.randint(1, 4)
            filter_sizes = [
                (
                    random.choice(possible_heights),
                    random.choice(possible_widths)
                )
                for _ in range(depth)
            ]
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
        """
        Actualiza el mejor coral encontrado y verifica si hubo mejora.
        También almacena el fitness promedio de la generación.

        Returns:
            bool: True si se encontró un nuevo mejor coral, False en caso contrario.
        """
        flat_reef = [
            coral for row in self.reef for coral in row if coral is not None
        ]
        if not flat_reef:
            return False

        # Calcular el fitness promedio de la generación
        avg_fitness = np.mean([coral['fitness'] for coral in flat_reef])
        self.avg_fitness_history.append(avg_fitness)

        # Obtener el mejor coral de la generación actual
        best_coral_current_generation = max(flat_reef, key=lambda x: x['fitness'])

        if self.best_coral is None or best_coral_current_generation['fitness'] > self.best_coral['fitness']:
            self.best_coral = copy.deepcopy(best_coral_current_generation)
            print(f"Nuevo mejor coral encontrado con fitness {self.best_coral['fitness']:.2f}")
            print(f"Parámetros del mejor coral: {self.best_coral['solution']}")
            self.fitness_history.append(self.best_coral['fitness'])
            return True
        else:
            self.fitness_history.append(self.best_coral['fitness'])
            return False


    def visualize_best_coral(self):
        """
        Visualiza y guarda la arquitectura del mejor coral cuando cambia.
        """
        best_model_params = self.best_coral['solution']
        best_model = InceptionMNISTModel(best_model_params)
        try:
            best_coral_dir = os.path.join(self.visualization_dir, 'best_coral')
            os.makedirs(best_coral_dir, exist_ok=True)
            visualize_inception_module(
                best_model,
                self.generation,
                'best_coral',
                best_coral_dir
            )
        except Exception as e:
            print(f"Error al visualizar el mejor coral en la generación {self.generation}: {e}")

    def visualize_current_corals(self):
        """
        Visualiza y guarda las arquitecturas de los corales actuales (opcional).
        """
        flat_reef = [
            (idx, coral['solution'])
            for idx, coral in enumerate(
                [coral for row in self.reef for coral in row if coral is not None]
            )
        ]
        gen_dir = os.path.join(self.visualization_dir, f'generation_{self.generation}')
        os.makedirs(gen_dir, exist_ok=True)
        for idx, model_params in flat_reef:
            model = InceptionMNISTModel(model_params)
            try:
                visualize_inception_module(
                    model,
                    self.generation,
                    idx,
                    gen_dir
                )
            except Exception as e:
                print(f"Error al visualizar el modelo en la generación {self.generation}, coral {idx}: {e}")

    def run(self):
        """
        Ejecuta el algoritmo CRO iterando sobre las generaciones.
        """
        for generation in range(self.max_generations):
            self.generation = generation + 1  # Actualizar el contador de generación
            print(f"\n=== Generación {self.generation} ===")

            # Obtener lista de corales ocupados en el arrecife
            corals = [coral for row in self.reef for coral in row if coral is not None]

            # Verificar si hay suficientes corales para reproducirse
            if len(corals) == 0:
                print("No hay corales en el arrecife.")
                continue

            # Mezclar los corales
            random.shuffle(corals)

            # Número de corales para reproducción sexual (broadcast spawning)
            num_broadcast = int(self.Fb * len(corals))
            broadcast_corals = corals[:num_broadcast]

            # Corales no seleccionados para broadcast se usan para brooding
            brooding_corals = corals[num_broadcast:]

            # Reproducción
            new_broadcast_larvae = self.broadcast_spawning(broadcast_corals)
            new_brooding_larvae = self.brooding(brooding_corals)
            new_budding_larvae = self.budding()

            # Añadir nuevas larvas al larvae_pool
            self.larvae_pool.extend(new_broadcast_larvae)
            self.larvae_pool.extend(new_brooding_larvae)
            self.larvae_pool.extend(new_budding_larvae)

            # Asentamiento de larvas desde el larvae_pool
            self.larvae_settlement()

            # Depredación
            self.predation()

            # Actualizar el mejor coral y almacenar el fitness
            improved = self.update_best_coral()
            if improved:
                self.no_improve_counter = 0  # Reiniciar el contador si hay mejora
                # Guardar visualización del mejor coral
                self.visualize_best_coral()
            else:
                self.no_improve_counter += 1  # Incrementar el contador si no hay mejora

            # Verificar si se alcanza el máximo de generaciones sin mejora
            if self.max_no_improve is not None and self.no_improve_counter >= self.max_no_improve:
                print(f"\nNo se encontró un nuevo mejor coral en {self.max_no_improve} generaciones consecutivas. Finalizando la optimización.")
                break

            # Visualizar los corales actuales (opcional)
            # self.visualize_current_corals()
            print(f"Mejor fitness en generación {self.generation}: {self.best_coral['fitness']:.2f}")
        print("\nOptimización completada.")
