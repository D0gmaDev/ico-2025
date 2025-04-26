import numpy as np
import random

from ico import *
from load_database import *

# ---------------------- MUTATIONS (actions) ----------------------

def mutate_swap(solution):
    new_solution = solution.copy()
    i, j = random.sample(range(len(new_solution)), 2)
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

def best_neighbor(state_id, solution):
    if state_id == 0:
        return mutate_swap(solution)
    elif state_id == 1:
        return mutate_add_zero(solution)
    elif state_id == 2:
        return mutate_remove_zero(solution)
    else:
        return mutate_swap(solution)

# ---------------------- FITNESS WRAPPER ----------------------

def get_fitness_learning(state, solution, distance_matrix):
    return fitness(state, solution, distance_matrix)

# ---------------------- Q-LEARNING LOGIC ----------------------

def choose_an_action(state, type_function, q_table=None, epsilon=0.1):
    if type_function == 1:
        return epsilon_greedy(state, epsilon, q_table)
    elif type_function == 2:
        return random.randint(0, q_table.shape[1] - 1)
    else:
        raise ValueError("type_function must be 1 (epsilon-greedy) or 2 (random)")

def epsilon_greedy(current_state, epsilon, q_table):
    p = random.random()
    if p <= epsilon:
        return random.randint(0, q_table.shape[1] - 1)
    else:
        return np.argmax(q_table[current_state])

def calculate_q_value(q_table, state, next_state, reward, alpha=0.1, gamma=0.9):
    best_next_q = np.max(q_table[next_state])
    td_target = reward + gamma * best_next_q
    td_delta = td_target - q_table[state, next_state]
    q_table[state, next_state] += alpha * td_delta

# ---------------------- RS Q-LEARNING ----------------------

def RS_QLearning(state, initial_solution, distance_matrix,
                 q_table=None, q_size=3, alpha=0.1, gamma=0.9, epsilon=1.0,
                 decay_rate=0.95, max_iterations_without_improvement=100):

    if q_table is None:
        q_table = np.zeros((q_size, q_size))  # 3 actions: 0, 1, 2

    improved = True
    no_improvement = 0
    x_best = initial_solution.copy()
    x = initial_solution.copy()

    while improved:
        reward = 0
        states_visited_count = 0
        next_state = choose_an_action(0, 2, q_table)
        x = best_neighbor(next_state, x)

        if get_fitness_learning(state, x, distance_matrix) < get_fitness_learning(state, x_best, distance_matrix):
            x_best = x.copy()
            reward = get_fitness_learning(state, x, distance_matrix)
        else:
            states_visited_count += 1

        while no_improvement <= max_iterations_without_improvement and states_visited_count < q_size:
            if no_improvement == 0:
                current_state = next_state
                next_state = choose_an_action(current_state, 1, q_table, epsilon)
            else:
                next_state = choose_an_action(0, 2, q_table)

            x = best_neighbor(next_state, x)

            if get_fitness_learning(state, x, distance_matrix) < get_fitness_learning(state, x_best, distance_matrix):
                x_best = x.copy()
                improved = True
                no_improvement = 0
                reward += get_fitness_learning(state, x, distance_matrix)
                calculate_q_value(q_table, current_state, next_state, reward, alpha, gamma)
            else:
                no_improvement += 1
                states_visited_count += 1

            if no_improvement > max_iterations_without_improvement and states_visited_count == q_size:
                improved = False

        epsilon *= decay_rate

    return x_best, get_fitness_learning(state, x_best, distance_matrix), q_table

nbr_state = 10
nbr_action = 3
state = load_data()
distance_matrix = compute_distance_matrix(state)
initial_solution = construct_initial_solutions(state, 1)[0]
q_table = None

best_sol, best_fit, q_table = RS_QLearning(state, initial_solution, distance_matrix, q_table=q_table)

