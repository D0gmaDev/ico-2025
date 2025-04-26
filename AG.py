import numpy as np
import random

from ico import *

def crossover(parent1, parent2, nb_clients):
    size = min(len(parent1), len(parent2))

    start, end = sorted(random.sample(range(1, size - 1), 2))
    child = [-1] * size

    child[start:end] = parent1[start:end]

    present_genes = set(child[start:end])

    current_pos = 0
    for i in range(size):
        if child[i] == -1:
            gene = parent2[i]

            while (gene in present_genes and gene != 0):
                current_pos += 1
                if current_pos == size:
                    print("FATAL - boucle infinie évitée")
                    exit()
                gene = parent2[current_pos]

            child[i] = gene
            if gene != 0:
                present_genes.add(gene)

    for i in range(nb_clients):
        zeroes = [j for j in range(len(child[1:-1])) if child[j] == 0]
        if i not in child:
            if zeroes:
                selected_zero = random.randint(0, len(zeroes) - 1)
                child[zeroes[selected_zero]] = i

    return child

def swap_mutation(solution, mutation_rate):
    # solution = solution.copy()
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(1, len(solution) - 1), 2)
        solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
    return solution

def inversion_mutation(solution, mutation_rate):
    # solution = solution.copy()
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(1, len(solution) - 1), 2))
        solution[start:end] = solution[start:end][::-1]
    return solution

def tournament_selection(population, fitness_values, tournament_size=5):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    best_idx = min(tournament_indices, key=lambda i: fitness_values[i])
    return population[best_idx]

def roulette_selection(population, fitness_values):
    inverse_fitness = 1 / (np.array(fitness_values) + 1e-6)
    probabilities = inverse_fitness / np.sum(inverse_fitness)
    return population[np.random.choice(len(population), p=probabilities)]

def genetic_algorithm(state, population, population_size, distance_matrix, generations, mutation_rate, elitism, selection_method='tournament', mutation_method='swap'):
    fitness_values = np.array([fitness(state, sol, distance_matrix) for sol in population])

    best_solution = population[np.argmin(fitness_values)]
    best_fitness = min(fitness_values)

    best_solutions = []
    best_fitnesss = []

    for _ in range(generations):
        sorted_indices = np.argsort(fitness_values)
        population = [population[i] for i in sorted_indices]
        fitness_values = fitness_values[sorted_indices]

        new_population = population[:elitism]
        new_fitness_values = list(fitness_values[:elitism])

        best_solutions.append(new_population[0])
        best_fitnesss.append(new_fitness_values[0])

        existing = set(map(tuple, new_population))

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_values) if selection_method == 'tournament' else roulette_selection(population, fitness_values)
            parent2 = tournament_selection(population, fitness_values) if selection_method == 'tournament' else roulette_selection(population, fitness_values)

            child = crossover(parent1, parent2, len(state["orders"]))
            
            child = swap_mutation(child, mutation_rate) if mutation_method == 'swap' else inversion_mutation(child, mutation_rate)
            child_fitness = fitness(state, child, distance_matrix)

            if tuple(child) not in existing:
                new_population.append(child)
                new_fitness_values.append(child_fitness)
                existing.add(tuple(child))

            if child_fitness < best_fitness:
                best_solution = child
                best_fitness = child_fitness

        population = new_population
        fitness_values = np.array(new_fitness_values)

    return best_solutions, best_fitnesss

def AG(state, start_population, distance_matrix, iterations=200, population_size=120, mutation_rate=0.2, elitism=10):
    best_solution, best_fitness = genetic_algorithm(state, start_population, population_size, distance_matrix, iterations, mutation_rate, elitism, selection_method='tournament', mutation_method='inversion')
    best_solution = best_solution[-1]
    best_fitness = best_fitness[-1]
    return best_solution, best_fitness
