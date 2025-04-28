import numpy as np
import random

from ico import *
from load_database import load_data


def init_q_table(num_states, num_actions):
    """
    Initialise la table Q avec des zéros.
    """
    return np.zeros((num_states, num_actions))

def epsilon_greedy(current_state, epsilon, q_table):
    p = random.random()
    if p <= epsilon:
        # Choix aléatoire parmi toutes les actions possibles
        action = random.randint(0, q_table.shape[1] - 1)
    else:
        # Choix de l'action avec la plus grande valeur Q
        action = np.argmax(q_table[current_state])
    return action
            
def mutate_swap(solution):
    new_solution = solution.copy()
    i, j = random.sample(range(0, len(new_solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def mutate_add_zero(solution):
    new_solution = solution.copy()
    if len(new_solution) > 1:
        i = random.randint(1, len(new_solution) - 1)
        new_solution.insert(i, 0)
    return new_solution

def mutate_remove_zero(solution):
    new_solution = solution.copy()
    zero_indices = [i for i, val in enumerate(new_solution) if val == 0]
    if zero_indices:
        random_index = random.choice(zero_indices)
        del new_solution[random_index]
    return new_solution

def apply_action(solution, action_id):
    if action_id == 0:
        return mutate_swap(solution)
    elif action_id == 1:
        return mutate_add_zero(solution)
    elif action_id == 2:
        return mutate_remove_zero(solution)
    else:
        raise ValueError("Action inconnue :", action_id)
    

def RS_Qlearning(state, initial_QL_state, initial_solution, distance_matrix, Q,  
                 iterations=200, T=1000, T_min=0.1, alpha=0.9, learning_rate=0.1, gamma=0.9, epsilon = 0.1):
    current_solution = initial_solution.copy()
    current_fitness = fitness(state, current_solution, distance_matrix)
    
    best_solution = current_solution.copy()
    best_fitness = current_fitness

    current_state = initial_QL_state

    while T > T_min:
        for _ in range(iterations):
            # --- Choix de l'action avec epsilon-greedy ---
            action = epsilon_greedy(current_state, epsilon, Q)

            # --- Appliquer la mutation correspondant à l'action ---
            new_solution = apply_action(current_solution, action)
            new_fitness = fitness(state, new_solution, distance_matrix)

            # --- Récompense (reward) ---
            reward = current_fitness - new_fitness  # + si on s'améliore

            # --- Mise à jour Q-learning ---
            next_state = action  # ou définis un vrai état si tu veux
            best_next_action = np.argmax(Q[next_state]) # Action avec la meilleure valeur Q pour l'état suivant
            # Mettre à jour la table Q
            Q[current_state][action] += learning_rate * (reward + gamma * Q[next_state][best_next_action] - Q[current_state][action])

            # --- Critère d'acceptation RS ---
            delta = new_fitness - current_fitness
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                current_solution = new_solution
                current_fitness = new_fitness
                current_state = next_state # Mettre à jour l'état seulement si la solution est acceptée

                if current_fitness < best_fitness:
                    best_solution = current_solution.copy()
                    best_fitness = current_fitness

        T *= alpha

    return best_solution, best_fitness, Q


state = load_data()
distance_matrix = compute_distance_matrix(state)
solutions = construct_initial_solutions(state, 1)

# --- Paramètres de l'algorithme Q-learning ---
num_actions = 3
num_states = 3   # pour l’instant on garde un seul état
Q = init_q_table(num_states, num_actions)
initial_QL_state = 0

# --- Test de RS_Qlearning ---
for _ in range(10):
    best_solution, best_fitness, updated_Q = RS_Qlearning(state, initial_QL_state, solutions[0], distance_matrix, Q)                                                 
    print("Meilleure fitness : ", best_fitness)
    print("Table Q mise à jour : ", updated_Q)

