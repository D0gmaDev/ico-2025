import numpy as np
import random as rd

from ico import *
from load_database import load_data

def eps_greedy(state_idx, eps, Q):
    p = rd.random()
    if p <= eps:
        action = rd.randint(0, Q.shape[1] - 1)
    else:
        action = np.argmax(Q[state_idx])
    return action

def mutate_swap(solution):
    new_solution = solution.copy()
    i, j = rd.sample(range(len(new_solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def mutate_add_zero(solution):
    new_solution = solution.copy()
    if len(new_solution) > 1:
        i = rd.randint(1, len(new_solution) - 1)
        new_solution.insert(i, 0)
    return new_solution

def mutate_remove_zero(solution):
    new_solution = solution.copy()
    zero_indices = [i for i, val in enumerate(new_solution) if val == 0]
    if zero_indices:
        random_index = rd.choice(zero_indices)
        del new_solution[random_index]
    return new_solution

Actions = [mutate_swap, mutate_add_zero, mutate_remove_zero]

def generate_neighbors(solution, action):
    return Actions[action](solution)

def AdaptativeLocalSearchQLearning(problem_data, distance_matrix, initial_solution):
    Lr = 0.1
    Dr = 0.9
    eps = 0.999
    Decr = 0.99
    max_iterations_without_improvement = 10

    num_states = 3
    num_actions = 3
    Q = np.zeros((num_states, num_actions))

    improved = True
    no_improvement = 0

    x_star = initial_solution.copy()
    x = initial_solution.copy()

    state_idx = 0

    while improved:
        x = x_star.copy()

        next_action = eps_greedy(state_idx, eps, Q)
        x_new = generate_neighbors(x, next_action)

        fitness_old = fitness(problem_data, x_star, distance_matrix)
        fitness_new = fitness(problem_data, x_new, distance_matrix)

        reward = fitness_old - fitness_new

        if reward > 0:
            x_star = x_new.copy()
            no_improvement = 0
        else:
            no_improvement += 1

        next_state = next_action
        best_next_action = np.argmax(Q[next_state])
        Q[state_idx][next_action] += Lr * (reward + Dr * Q[next_state][best_next_action] - Q[state_idx][next_action])

        state_idx = next_state

        if no_improvement > max_iterations_without_improvement:
            if eps > 0.1:
                eps *= Decr
            no_improvement = 0
            improved = False

    return x_star, Q

def RS_QLearning(state, initial_solution, distance_matrix, Q,
                 iterations=200, T=1000, T_min=0.1, alpha=0.99, epsilon=0.1):
    # Initialisation
    current_solution = initial_solution.copy()
    current_fitness = fitness(state, current_solution, distance_matrix)
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness

    current_state = 0

    # Exécution du recuit simulé
    while T > T_min:
        for _ in range(iterations):
            # 1. Choisir action avec epsilon-greedy sur Q
            action = eps_greedy(current_state, epsilon, Q)

            # 2. Appliquer la mutation correspondant à l'action
            new_solution = generate_neighbors(current_solution, action)
            new_fitness = fitness(state, new_solution, distance_matrix)

            # 3. Accepter ou pas la nouvelle solution
            if new_fitness < current_fitness:
                current_solution = new_solution
                current_fitness = new_fitness
                current_state = action

                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness
            else:
                delta = new_fitness - current_fitness
                probability = np.exp(-delta / T)
                if rd.random() < probability:
                    current_solution = new_solution
                    current_fitness = new_fitness
                    current_state = action

        T *= alpha

    return best_solution, best_fitness

# ======================= EXECUTION ===========================

# Chargement des données du problème
problem_data = load_data()
distance_matrix = compute_distance_matrix(problem_data)

# Construction d'une solution initiale
solutions = construct_initial_solutions(problem_data, 1)
initial_solution = solutions[0]

# Apprentissage de la table Q
best_solution, Q = AdaptativeLocalSearchQLearning(problem_data, distance_matrix, initial_solution)

# Exécution de RS_QLearning
final_solution, final_fitness = RS_QLearning(
    state=problem_data,
    initial_solution=initial_solution,
    distance_matrix=distance_matrix,
    Q=Q,
    iterations=200,
    T=1000,
    T_min=0.1,
    alpha=0.99,
    epsilon=0.1
)

print("\nSolution finale obtenue avec RS guidé par Q-learning :")
print(final_solution)

print("\nFitness de la solution finale :", final_fitness)
