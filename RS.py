import numpy as np
import random

from ico import *

state = {
    "position": [
        (0, 0),  # Depot (0)
        (5, 2), (3, -3), (2, 2), (1, 0),  # Customers (1-4)
        (3, 3), (2, 1), (-5, 0), (7, 1),  # Customers (5-8)
        (-3, -2), (5, 1), (1, 2), (4, 2)  # Customers (9-12)
    ],
    "orders": [0, 5, 10, 7, 8, 6, 9, 4, 3, 1, 2, 3, 7]  # Order demand per customer (0 is depot)
}

# Compute the distance matrix
distance_matrix = compute_distance_matrix(state)

# Initial solution
solution = [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 0]  # Returns to depot

def mutate(solution):
    new_solution = solution.copy()

    if random.random() < 0.7:
        # Swap two random points
        i, j = random.sample(range(0, len(new_solution)), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    elif random.random() < 0.5:
        # Add a 0 somewhere in the middle
        i = random.randint(1, len(new_solution) - 1)
        new_solution.insert(i, 0)
    else:
        zero_indices = [i for i, val in enumerate(new_solution) if val == 0]
        if zero_indices:  # Ensure there's at least one 0
            random_index = random.choice(zero_indices)  # Pick a random 0 index
            del new_solution[random_index]

    return new_solution

def RS(state, initial_solution, distance_matrix, iterations=200):
    # Initialisation
    current_solution = initial_solution.copy()
    current_fitness = fitness(state, current_solution, distance_matrix)
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # Paramètres du recuit simulé
    T = 1000  # Température initiale
    T_min = 0.1  # Température minimale
    alpha = 0.99  # Facteur de refroidissement
    
    # Exécution du recuit simulé
    while T > T_min:
        for _ in range(iterations):
            new_solution = mutate(current_solution)
            new_fitness = fitness(state, new_solution, distance_matrix)
            
            # Si la nouvelle solution est meilleure, l'accepter
            if new_fitness < current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness

                # Mettre à jour la meilleure solution
                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            # Sinon, l'accepter avec une certaine probabilité
            else:
                delta = new_fitness - current_fitness
                probability = np.exp(-delta / T)
                if random.random() < probability:
                    current_solution = new_solution
                    current_fitness = new_fitness
        
        T *= alpha  # Réduire la température

    return best_solution, best_fitness


print("Initial solution:", solution)
print("Initial fitness:", fitness(state, solution, distance_matrix))

print("=" * 20)

best_solution, best_fitness = RS(state, solution, distance_matrix)
print("Meilleure solution trouvée:", best_solution)
print("Meilleure fitness:", best_fitness)

plot_solution(state, best_solution)
