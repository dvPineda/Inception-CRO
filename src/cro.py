# src/cro.py

import numpy as np
import random
import copy
import os

from src.models import InceptionMNISTModel
from src.visualization import visualize_inception_module

class CoralReefOptimization:
    """
    Implementation of the Coral Reef Optimization (CRO) metaheuristic,
    adapted for evolving neural architectures.

    The reef is a 2D grid (N x M). Each cell can hold a coral (model parameters + fitness) or be empty.
    The main steps are:
      1. Initialization: Random solutions in the reef.
      2. BROADCAST SPAWNING + BROODING:
         - Some corals reproduce sexually (broadcast spawning), generating new larvae.
         - Others reproduce asexually (brooding), generating new larvae.
         - Budding (a special kind of asexual reproduction) is optional for top corals.
      3. LARVAE SETTLEMENT: New larvae try to settle in the reef, replacing weaker corals or occupying empty cells.
      4. PREDATION: Some weaker corals are removed from the reef.
      5. Check if a better solution is found, track best coral.

    In this version, the "solution" is a dictionary describing an Inception module's branches.
    The fitness_function is provided by the user (and typically calls a partial training + evaluate_model).
    """

    def __init__(
        self,
        reef_size,
        rho_0,
        Fb,
        Fa,
        Pa,
        Fd,
        Pd,
        kappa,
        mutation_rate,
        fitness_function,
        max_generations,
        max_no_improve=None,
        visualization_dir=None
    ):
        """
        Initializes CRO parameters and sets up the reef.

        Args:
            reef_size (tuple): (N, M) dimensions of the reef (e.g., (20, 10)).
            rho_0 (float): Initial occupation ratio [0,1].
            Fb (float): Fraction of corals for broadcast spawning.
            Fa (float): Fraction of corals for asexual reproduction.
            Pa (float): Probability that a coral effectively reproduces asexually.
            Fd (float): Fraction for predation.
            Pd (float): Probability threshold for predation.
            kappa (int): Max attempts for a larva to settle.
            mutation_rate (float): Probability of mutation in offspring.
            fitness_function (callable): Function that takes a solution (dict) -> fitness float.
            max_generations (int): Maximum number of CRO iterations (generations).
            max_no_improve (int, optional): Early stopping if no improvement in best coral after these many gens.
            visualization_dir (str, optional): Directory for saving coral visualizations (best coral, etc.).
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
        self.max_no_improve = max_no_improve  # Early stopping threshold
        self.no_improve_counter = 0  # tracks consecutive gens without improvement

        self.generation = 0  # current generation index

        if visualization_dir is None:
            visualization_dir = "visualizations"  # default fallback
        self.visualization_dir = visualization_dir
        os.makedirs(self.visualization_dir, exist_ok=True)

        # Initialize reef (2D, can contain corals or None if empty)
        self.reef = self.initialize_reef()
        self.best_coral = None
        self.fitness_history = []     # best fitness per generation
        self.avg_fitness_history = [] # average fitness per generation

        self.larvae_pool = []  # newly generated larvae waiting to settle

    def initialize_reef(self):
        """
        Randomly initializes the reef with corals according to rho_0 occupation ratio.

        Returns:
            np.ndarray: 2D array of shape (N, M) with coral dictionaries or None.
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
        Generates a random Inception-based solution (branch configurations).

        Returns:
            dict: 'branches_params' describing filter sizes, channels, etc.
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
            use_pooling = (random.random() < 0.5)

            branch_param = {
                'depth': depth,
                'filter_sizes': filter_sizes,
                'filter_channels': filter_channels,
                'use_pooling': use_pooling
            }
            branches_params.append(branch_param)

        return {'branches_params': branches_params}

    def broadcast_spawning(self, broadcast_corals):
        """
        Sexual reproduction among broadcast corals (crossover).
        Pairs of parents generate offspring larva.

        Args:
            broadcast_corals (list): list of coral dicts selected for sexual reproduction.

        Returns:
            list: new larvae generated by crossover.
        """
        new_larvae = []
        num_parents = len(broadcast_corals)
        if num_parents < 2:
            return new_larvae

        # Ensure even number of parents
        if num_parents % 2 != 0:
            removed_coral = random.choice(broadcast_corals)
            broadcast_corals.remove(removed_coral)
            num_parents -= 1

        # Shuffle
        random.shuffle(broadcast_corals)

        # Pair them
        for i in range(0, num_parents, 2):
            parent1 = broadcast_corals[i]['solution']
            parent2 = broadcast_corals[i + 1]['solution']
            new_solution = self.crossover(parent1, parent2)
            new_fitness = self.fitness_function(new_solution)
            larva = {'solution': new_solution, 'fitness': new_fitness, 'attempts': self.kappa}
            new_larvae.append(larva)

        return new_larvae

    def brooding(self, brooding_corals):
        """
        Asexual reproduction via mutation for corals not selected in broadcast spawning.

        Args:
            brooding_corals (list): coral dicts for brooding.

        Returns:
            list: new larvae generated by mutation.
        """
        new_larvae = []
        for coral in brooding_corals:
            new_solution = self.mutate(coral['solution'])
            new_fitness = self.fitness_function(new_solution)
            larva = {'solution': new_solution, 'fitness': new_fitness, 'attempts': self.kappa}
            new_larvae.append(larva)
        return new_larvae

    def budding(self):
        """
        Special asexual reproduction for top corals. 
        A fraction Fa of the best corals can replicate exactly (no changes).

        Returns:
            list: new larvae identical to their parents.
        """
        corals = [c for row in self.reef for c in row if c is not None]
        if len(corals) == 0:
            return []

        num_budding = int(self.Fa * len(corals))
        if num_budding == 0:
            return []

        # Probability Pa for budding to actually happen
        if random.random() > self.Pa:
            return []

        # Select best corals
        sorted_corals = sorted(corals, key=lambda x: x['fitness'], reverse=True)
        selected_corals = sorted_corals[:num_budding]

        new_larvae = []
        for coral in selected_corals:
            new_solution = copy.deepcopy(coral['solution'])
            larva = {'solution': new_solution, 'fitness': coral['fitness'], 'attempts': self.kappa}
            new_larvae.append(larva)
        return new_larvae

    def larvae_settlement(self):
        """
        Attempts to settle larvae from larvae_pool into the reef,
        replacing weaker corals if needed.
        """
        new_pool = []
        for larva in self.larvae_pool:
            i, j = random.randint(0, self.N - 1), random.randint(0, self.M - 1)

            if self.reef[i, j] is None or larva['fitness'] > self.reef[i, j]['fitness']:
                self.reef[i, j] = {'solution': larva['solution'], 'fitness': larva['fitness']}
            else:
                larva['attempts'] -= 1
                if larva['attempts'] > 0:
                    new_pool.append(larva)

        self.larvae_pool = new_pool

    def predation(self):
        """
        Removes weaker corals (lowest fitness) with probability Fd,
        applied to a fraction Pd of corals.
        """
        flat_reef = [
            ((i, j), self.reef[i, j])
            for i in range(self.N)
            for j in range(self.M)
            if self.reef[i, j] is not None
        ]
        if len(flat_reef) == 0:
            return

        num_predated = int(self.Pd * len(flat_reef))
        if num_predated == 0:
            return

        # Sort from worst to best
        sorted_corals = sorted(flat_reef, key=lambda x: x[1]['fitness'])
        for (i, j), coral in sorted_corals[:num_predated]:
            if random.random() < self.Fd:
                self.reef[i, j] = None

    def crossover(self, parent1, parent2):
        """
        Sexual reproduction: merges branches from two solutions.

        Args:
            parent1 (dict): 'branches_params'
            parent2 (dict): 'branches_params'

        Returns:
            dict: child's branches_params after merging.
        """
        num_branches_1 = len(parent1['branches_params'])
        num_branches_2 = len(parent2['branches_params'])
        min_branches = min(num_branches_1, num_branches_2)
        max_branches = max(num_branches_1, num_branches_2)

        child_branches = []
        num_branches_child = random.randint(min_branches, max_branches)
        for _ in range(num_branches_child):
            if random.random() < 0.5:
                selected_branches = parent1['branches_params']
            else:
                selected_branches = parent2['branches_params']
            branch = copy.deepcopy(random.choice(selected_branches))
            child_branches.append(branch)

        return {'branches_params': child_branches}

    def mutate(self, solution):
        """
        Mutation on a single solution: can alter depth, filter sizes, channels,
        pooling usage, or add/remove branches with some probability.

        Args:
            solution (dict): 'branches_params' describing the model.

        Returns:
            dict: mutated solution.
        """
        new_solution = copy.deepcopy(solution)
        possible_heights = [1, 3, 5, 7, 9]
        possible_widths = [1, 3, 5, 7, 9]

        for branch_param in new_solution['branches_params']:
            if random.random() < self.mutation_rate:
                # Mutate entire branch
                branch_param['depth'] = random.randint(1, 4)
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
                branch_param['use_pooling'] = (random.random() < 0.5)
            else:
                # Smaller chance to mutate individual elements
                for idx in range(branch_param['depth']):
                    if random.random() < self.mutation_rate:
                        branch_param['filter_sizes'][idx] = (
                            random.choice(possible_heights),
                            random.choice(possible_widths)
                        )
                        branch_param['filter_channels'][idx] = random.randint(4, 64)

        # Possibly add a branch
        if random.random() < self.mutation_rate:
            depth = random.randint(1, 4)
            new_branch_fsizes = [
                (
                    random.choice(possible_heights),
                    random.choice(possible_widths)
                )
                for _ in range(depth)
            ]
            new_branch_channels = [random.randint(4, 64) for _ in range(depth)]
            use_pooling = (random.random() < 0.5)
            new_branch = {
                'depth': depth,
                'filter_sizes': new_branch_fsizes,
                'filter_channels': new_branch_channels,
                'use_pooling': use_pooling
            }
            new_solution['branches_params'].append(new_branch)
        # Possibly remove a branch
        elif len(new_solution['branches_params']) > 2 and random.random() < self.mutation_rate:
            idx = random.randint(0, len(new_solution['branches_params']) - 1)
            del new_solution['branches_params'][idx]

        return new_solution

    def update_best_coral(self):
        """
        Update self.best_coral if the current generation yields a better one.
        Also track average fitness across reef for logging.

        Returns:
            bool: True if found a new best coral, False otherwise.
        """
        flat_reef = [c for row in self.reef for c in row if c is not None]
        if not flat_reef:
            return False

        avg_fitness = np.mean([c['fitness'] for c in flat_reef])
        self.avg_fitness_history.append(avg_fitness)

        best_in_generation = max(flat_reef, key=lambda x: x['fitness'])
        if self.best_coral is None or best_in_generation['fitness'] > self.best_coral['fitness']:
            self.best_coral = copy.deepcopy(best_in_generation)
            self.fitness_history.append(self.best_coral['fitness'])
            print(f"Nuevo mejor coral con fitness {self.best_coral['fitness']:.2f}")
            print(f"Parámetros del mejor coral: {self.best_coral['solution']}")
            return True
        else:
            self.fitness_history.append(self.best_coral['fitness'])
            return False

    def visualize_best_coral(self):
        """
        Visualize the best coral's Inception architecture if possible,
        saving it under 'best_coral' inside self.visualization_dir.
        """
        if self.best_coral is None:
            return

        best_model_params = self.best_coral['solution']
        best_model = InceptionMNISTModel(best_model_params)

        # Subdirectory: <visualizations_dir>/best_coral
        best_coral_dir = os.path.join(self.visualization_dir, 'best_coral')
        os.makedirs(best_coral_dir, exist_ok=True)

        try:
            visualize_inception_module(
                best_model,
                self.generation,
                'best_coral',
                best_coral_dir
            )
        except Exception as e:
            print(f"Error al visualizar el mejor coral en la generación {self.generation}: {e}")

    def run(self):
        """
        Core CRO loop:
         1. Reproduction (broadcast + brooding + budding).
         2. Settlement of larvae.
         3. Predation.
         4. Update best coral, check improvement or early stop.
        """
        for generation in range(self.max_generations):
            self.generation = generation + 1
            print(f"\n=== Generación {self.generation} ===")

            corals = [c for row in self.reef for c in row if c is not None]
            if len(corals) == 0:
                print("No hay corales en el arrecife.")
                continue

            # Shuffle
            random.shuffle(corals)

            # Broadcast fraction
            num_broadcast = int(self.Fb * len(corals))
            broadcast_corals = corals[:num_broadcast]
            brooding_corals = corals[num_broadcast:]

            # Reproduction
            new_broadcast_larvae = self.broadcast_spawning(broadcast_corals)
            new_brooding_larvae = self.brooding(brooding_corals)
            new_budding_larvae = self.budding()

            # Combine all larvae
            self.larvae_pool.extend(new_broadcast_larvae)
            self.larvae_pool.extend(new_brooding_larvae)
            self.larvae_pool.extend(new_budding_larvae)

            # Settlement
            self.larvae_settlement()

            # Predation
            self.predation()

            # Update best coral and track improvement
            improved = self.update_best_coral()
            if improved:
                self.no_improve_counter = 0
                # Visualize best coral's architecture
                self.visualize_best_coral()
            else:
                self.no_improve_counter += 1

            # Early stopping if no improvement
            if (self.max_no_improve is not None
                and self.no_improve_counter >= self.max_no_improve):
                print(f"\nNo se encontró un nuevo mejor coral en {self.max_no_improve} "
                      f"generaciones consecutivas. Finalizando la optimización.")
                break

            print(f"Mejor fitness hasta generación {self.generation}: {self.best_coral['fitness']:.2f}")

        print("\nOptimización completada.")
